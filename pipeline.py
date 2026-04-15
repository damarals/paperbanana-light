"""Pipeline orchestration: connects retriever, planner, stylist, visualizer, and critic."""

import asyncio
import base64
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

from agents import planner, visualizer, stylist, critic
from retriever.embedder import ensure_index, search

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data" / "PaperBananaBench"


def _load_image_base64(relative_path: str) -> Optional[str]:
    """Load a reference image from disk and return base64-encoded JPEG."""
    image_path = DATA_DIR / "diagram" / relative_path
    if not image_path.exists():
        logger.warning("Reference image not found: %s", image_path)
        return None
    return base64.b64encode(image_path.read_bytes()).decode()


def _enrich_examples_with_images(
    matches: list[dict], ref_lookup: dict[str, dict],
) -> list[dict]:
    """Add image_base64 to retriever matches using the full ref.json data."""
    enriched = []
    for match in matches:
        ref = ref_lookup.get(match["id"], {})
        image_path = ref.get("path_to_gt_image", "")
        image_b64 = _load_image_base64(image_path) if image_path else None
        enriched.append({
            "content": match["content"],
            "visual_intent": match["visual_intent"],
            "image_base64": image_b64,
        })
    return enriched


def _get_aspect_ratio(data: dict) -> str:
    return data.get("additional_info", {}).get("rounded_ratio", "16:9")


async def _run_retriever(data: dict, api_key: str) -> list[dict]:
    """Run embedding-based retrieval and return enriched examples."""
    import json

    ensure_index(api_key)

    query = f"{data['content']} {data['visual_intent']}"
    matches = search(query, api_key, top_k=10)

    ref_path = DATA_DIR / "diagram" / "ref.json"
    with open(ref_path) as f:
        refs = json.load(f)
    ref_lookup = {r["id"]: r for r in refs}

    return _enrich_examples_with_images(matches, ref_lookup)


async def _run_critic_loop(
    data: dict,
    api_key: str,
    max_rounds: int,
) -> dict:
    """Run critic-visualizer loop up to max_rounds times.

    Args:
        data: pipeline data dict (mutated in place)
        api_key: Gemini API key
        max_rounds: maximum critic iterations
    """
    for round_idx in range(max_rounds):
        # Determine which description and image to critique
        if round_idx == 0:
            desc_key = "stylist_desc"
            img_key = "stylist_image"
        else:
            desc_key = f"critic_desc_{round_idx - 1}"
            img_key = f"critic_image_{round_idx - 1}"

        description = data.get(desc_key, "")
        image_b64 = data.get(img_key)

        feedback = await critic.run(
            description=description,
            content=data["content"],
            visual_intent=data["visual_intent"],
            image_base64=image_b64,
            api_key=api_key,
        )

        data[f"critic_suggestions_{round_idx}"] = feedback.critic_suggestions

        if feedback.critic_suggestions.strip() == "No changes needed.":
            data[f"critic_desc_{round_idx}"] = description
            logger.info("Critic round %d: no changes needed, stopping", round_idx)
            break

        data[f"critic_desc_{round_idx}"] = feedback.revised_description

        # Regenerate image from revised description
        new_image = await visualizer.run(
            description=feedback.revised_description,
            api_key=api_key,
            aspect_ratio=_get_aspect_ratio(data),
        )
        if new_image:
            data[f"critic_image_{round_idx}"] = new_image
            logger.info("Critic round %d: visualization succeeded", round_idx)
        else:
            logger.warning("Critic round %d: visualization failed, stopping", round_idx)
            break

    return data


async def run_pipeline(
    data: dict,
    api_key: str,
    max_critic_rounds: int = 3,
) -> dict:
    """Run the full pipeline on a single data item.

    Args:
        data: dict with content, visual_intent, filename, additional_info
        api_key: Gemini API key
        max_critic_rounds: max critic-visualizer iterations

    Returns:
        The data dict enriched with results from each stage.
    """
    aspect_ratio = _get_aspect_ratio(data)

    # Retriever (unless already retrieved)
    if "retrieved_examples" not in data:
        data["retrieved_examples"] = await _run_retriever(data, api_key)

    # Planner
    description = await planner.run(
        content=data["content"],
        visual_intent=data["visual_intent"],
        examples=data["retrieved_examples"],
        api_key=api_key,
    )
    data["planner_desc"] = description

    # Stylist
    styled_desc = await stylist.run(
        description=description,
        content=data["content"],
        visual_intent=data["visual_intent"],
        api_key=api_key,
    )
    data["stylist_desc"] = styled_desc

    # Visualizer
    image_b64 = await visualizer.run(
        description=styled_desc,
        api_key=api_key,
        aspect_ratio=aspect_ratio,
    )
    data["stylist_image"] = image_b64

    # Critic loop
    data = await _run_critic_loop(data, api_key, max_critic_rounds)

    return data


async def run_batch(
    data_list: list[dict],
    api_key: str,
    max_concurrent: int = 10,
    max_critic_rounds: int = 3,
) -> AsyncGenerator[dict, None]:
    """Run the pipeline on a batch of items with shared retrieval.

    Retriever runs once on the first item, then results are shared.
    Items are processed concurrently up to max_concurrent.

    Yields:
        Completed data dicts as they finish (unordered).
    """
    if not data_list:
        return

    # Run retriever once and share results
    logger.info("Running retriever once for batch of %d items", len(data_list))
    examples = await _run_retriever(data_list[0], api_key)
    for data in data_list:
        data["retrieved_examples"] = examples

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process(data: dict) -> dict:
        async with semaphore:
            return await run_pipeline(data, api_key, max_critic_rounds)

    tasks = [asyncio.create_task(_process(d)) for d in data_list]

    for future in asyncio.as_completed(tasks):
        yield await future
