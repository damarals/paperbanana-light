"""Pipeline orchestration: connects retriever, planner, stylist, visualizer, and critic."""

import asyncio
import base64
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

from agents import planner, visualizer, stylist, critic, vanilla
from retriever.embedder import ensure_index, search

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data" / "PaperBananaBench"

VALID_MODES = ("vanilla", "dev_planner", "dev_planner_critic", "demo_full")


def _load_image_base64(task_name: str, relative_path: str) -> Optional[str]:
    """Load a reference image from disk and return base64-encoded JPEG."""
    image_path = DATA_DIR / task_name / relative_path
    if not image_path.exists():
        logger.warning("Reference image not found: %s", image_path)
        return None
    return base64.b64encode(image_path.read_bytes()).decode()


def _enrich_examples_with_images(
    task_name: str, matches: list[dict], ref_lookup: dict[str, dict],
) -> list[dict]:
    """Add image_base64 to retriever matches using the full ref.json data."""
    enriched = []
    for match in matches:
        ref = ref_lookup.get(match["id"], {})
        image_path = ref.get("path_to_gt_image", "")
        image_b64 = _load_image_base64(task_name, image_path) if image_path else None
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

    task_name = data["task_name"]
    ensure_index(task_name, api_key)

    query = f"{data['content']} {data['visual_intent']}"
    matches = search(task_name, query, api_key, top_k=10)

    ref_path = DATA_DIR / task_name / "ref.json"
    with open(ref_path) as f:
        refs = json.load(f)
    ref_lookup = {r["id"]: r for r in refs}

    return _enrich_examples_with_images(task_name, matches, ref_lookup)


async def _run_critic_loop(
    data: dict,
    api_key: str,
    source: str,
    max_rounds: int,
) -> dict:
    """Run critic-visualizer loop up to max_rounds times.

    Args:
        data: pipeline data dict (mutated in place)
        api_key: Gemini API key
        source: "planner" or "stylist" -- determines initial description/image keys
        max_rounds: maximum critic iterations
    """
    task_name = data["task_name"]

    for round_idx in range(max_rounds):
        # Determine which description and image to critique
        if round_idx == 0:
            if source == "stylist":
                desc_key = f"target_{task_name}_stylist_desc0"
                img_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            else:
                desc_key = f"target_{task_name}_desc0"
                img_key = f"target_{task_name}_desc0_base64_jpg"
        else:
            desc_key = f"target_{task_name}_critic_desc{round_idx - 1}"
            img_key = f"target_{task_name}_critic_desc{round_idx - 1}_base64_jpg"

        description = data.get(desc_key, "")
        image_b64 = data.get(img_key)

        feedback = await critic.run(
            task_name=task_name,
            description=description,
            content=data["content"],
            visual_intent=data["visual_intent"],
            image_base64=image_b64,
            api_key=api_key,
        )

        data[f"target_{task_name}_critic_suggestions{round_idx}"] = feedback.critic_suggestions

        if feedback.critic_suggestions.strip() == "No changes needed.":
            data[f"target_{task_name}_critic_desc{round_idx}"] = description
            logger.info("Critic round %d: no changes needed, stopping", round_idx)
            break

        data[f"target_{task_name}_critic_desc{round_idx}"] = feedback.revised_description

        # Regenerate image from revised description
        new_image = await visualizer.run(
            task_name=task_name,
            description=feedback.revised_description,
            api_key=api_key,
            aspect_ratio=_get_aspect_ratio(data),
        )
        if new_image:
            data[f"target_{task_name}_critic_desc{round_idx}_base64_jpg"] = new_image
            logger.info("Critic round %d: visualization succeeded", round_idx)
        else:
            logger.warning("Critic round %d: visualization failed, stopping", round_idx)
            break

    return data


async def run_pipeline(
    data: dict,
    mode: str,
    api_key: str,
    max_critic_rounds: int = 3,
) -> dict:
    """Run the full agent pipeline on a single data item.

    Args:
        data: dict with task_name, content, visual_intent, filename, additional_info
        mode: one of "vanilla", "dev_planner", "dev_planner_critic", "demo_full"
        api_key: Gemini API key
        max_critic_rounds: max critic-visualizer iterations

    Returns:
        The data dict enriched with results from each stage.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Must be one of {VALID_MODES}")

    task_name = data["task_name"]
    aspect_ratio = _get_aspect_ratio(data)

    if mode == "vanilla":
        image_b64 = await vanilla.run(
            task_name=task_name,
            content=data["content"],
            visual_intent=data["visual_intent"],
            api_key=api_key,
            aspect_ratio=aspect_ratio,
        )
        data[f"vanilla_{task_name}_base64_jpg"] = image_b64
        return data

    # All other modes start with retriever (unless already retrieved)
    if "retrieved_examples" not in data:
        data["retrieved_examples"] = await _run_retriever(data, api_key)

    # Planner
    description = await planner.run(
        task_name=task_name,
        content=data["content"],
        visual_intent=data["visual_intent"],
        examples=data["retrieved_examples"],
        api_key=api_key,
    )
    data[f"target_{task_name}_desc0"] = description

    # Stylist (only in demo_full)
    if mode == "demo_full":
        styled_desc = await stylist.run(
            task_name=task_name,
            description=description,
            content=data["content"],
            visual_intent=data["visual_intent"],
            api_key=api_key,
        )
        data[f"target_{task_name}_stylist_desc0"] = styled_desc
        vis_description = styled_desc
        vis_key = f"target_{task_name}_stylist_desc0_base64_jpg"
    else:
        vis_description = description
        vis_key = f"target_{task_name}_desc0_base64_jpg"

    # Visualizer
    image_b64 = await visualizer.run(
        task_name=task_name,
        description=vis_description,
        api_key=api_key,
        aspect_ratio=aspect_ratio,
    )
    data[vis_key] = image_b64

    # For dev_planner mode without stylist, also store under the planner key
    # (vis_key already equals the planner key in that case)

    # Critic loop (dev_planner_critic and demo_full)
    if mode in ("dev_planner_critic", "demo_full"):
        source = "stylist" if mode == "demo_full" else "planner"
        data = await _run_critic_loop(data, api_key, source, max_critic_rounds)

    return data


async def run_batch(
    data_list: list[dict],
    mode: str,
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
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: {mode!r}. Must be one of {VALID_MODES}")

    if not data_list:
        return

    # Run retriever once and share results (all non-vanilla modes need it)
    if mode != "vanilla":
        logger.info("Running retriever once for batch of %d items", len(data_list))
        examples = await _run_retriever(data_list[0], api_key)
        for data in data_list:
            data["retrieved_examples"] = examples

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process(data: dict) -> dict:
        async with semaphore:
            return await run_pipeline(data, mode, api_key, max_critic_rounds)

    tasks = [asyncio.create_task(_process(d)) for d in data_list]

    for future in asyncio.as_completed(tasks):
        yield await future
