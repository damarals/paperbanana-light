"""Embedding-based retrieval using Gemini Embedding 2 + ChromaDB.

Replaces LLM-based retriever ($3.37/call, 813k tokens) with
embedding similarity search (~$0.0001/call).
"""

import json
import time
from pathlib import Path

import chromadb
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "PaperBananaBench"
INDEX_DIR = PROJECT_ROOT / "retriever" / "index"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
BATCH_SIZE = 50


def _load_references(task_name: str) -> list[dict]:
    ref_path = DATA_DIR / task_name / "ref.json"
    with open(ref_path) as f:
        return json.load(f)


def _get_collection(task_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    return client.get_or_create_collection(name=f"{task_name}_references")


def _embed_texts(texts: list[str], api_key: str) -> list[list[float]]:
    client = genai.Client(api_key=api_key)
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=batch
        )
        all_embeddings.extend([e.values for e in result.embeddings])
        if i + BATCH_SIZE < len(texts):
            time.sleep(1)
    return all_embeddings


def index_references(task_name: str, api_key: str) -> None:
    """Build ChromaDB index from reference embeddings.

    Skips if the collection already has the expected number of documents.
    """
    refs = _load_references(task_name)
    collection = _get_collection(task_name)

    if collection.count() == len(refs):
        print(f"[retriever] {task_name}: index already exists ({len(refs)} docs), skipping")
        return

    texts = [f"{r['content']} {r['visual_intent']}" for r in refs]

    print(f"[retriever] {task_name}: embedding {len(texts)} references...")
    embeddings = _embed_texts(texts, api_key)

    ids = [r["id"] for r in refs]
    metadatas = [
        {
            "id": r["id"],
            "visual_intent": r["visual_intent"],
            "category": r.get("category", ""),
            "content": r["content"][:500],
        }
        for r in refs
    ]

    for i in range(0, len(ids), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        print(f"[retriever] {task_name}: indexed {batch_end}/{len(ids)}")

    print(f"[retriever] {task_name}: indexing complete")


def search(
    task_name: str,
    query_text: str,
    api_key: str,
    top_k: int = 10,
) -> list[dict]:
    """Find the most similar references to query_text."""
    collection = _get_collection(task_name)
    if collection.count() == 0:
        raise RuntimeError(
            f"Index for '{task_name}' is empty. Call index_references() first."
        )

    query_embedding = _embed_texts([query_text], api_key)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    refs_by_id = {r["id"]: r for r in _load_references(task_name)}

    matches = []
    for i, ref_id in enumerate(results["ids"][0]):
        ref = refs_by_id[ref_id]
        matches.append(
            {
                "id": ref_id,
                "content": ref["content"],
                "visual_intent": ref["visual_intent"],
                "category": ref.get("category", ""),
                "distance": results["distances"][0][i],
            }
        )
    return matches


def ensure_index(task_name: str, api_key: str) -> None:
    """Build the index if it doesn't already exist."""
    index_references(task_name, api_key)
