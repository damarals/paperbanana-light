"""Polish agent: two-step style refinement of existing images.

Step 1: Critique image against style guide (text LLM via Agno).
Step 2: Apply suggestions to regenerate image (Gemini image generation).
"""

import asyncio
import base64
from pathlib import Path
from typing import Optional

from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from google import genai
from google.genai import types

from agents.config import load_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIAGRAM_SUGGESTION_SYSTEM_PROMPT = """\
You are a senior art director for NeurIPS 2025. Your task is to critique a diagram against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts, icons).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the diagram is substantially compliant, output "No changes needed"."""

DIAGRAM_POLISH_SYSTEM_PROMPT = """\
## ROLE
You are a professional diagram polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing diagram image and a list of specific improvement suggestions. Your task is to generate a polished version of this diagram by applying these suggestions while preserving the semantic logic and structure of the original diagram.

## OUTPUT
Generate a polished diagram image that maintains the original content while applying the improvement suggestions."""

async def run(
    image_base64: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Polish an existing image using style guidelines.

    Args:
        image_base64: base64-encoded JPEG of the source image
        api_key: Gemini API key
        aspect_ratio: aspect ratio for regenerated image

    Returns:
        Base64-encoded JPEG of the polished image, or None on failure.
    """
    # Step 1: generate suggestions
    suggestions = await _generate_suggestions(image_base64, api_key)
    if not suggestions or "No changes needed" in suggestions:
        return None

    # Step 2: apply suggestions via image generation
    return await _apply_suggestions(
        image_base64, suggestions, api_key, aspect_ratio,
    )


async def _generate_suggestions(
    image_base64: str,
    api_key: Optional[str] = None,
) -> str:
    """Step 1: critique image against the NeurIPS style guide."""
    system_prompt = DIAGRAM_SUGGESTION_SYSTEM_PROMPT

    style_guide_path = (
        PROJECT_ROOT / "assets" / "style_guide.md"
    )
    style_guide = style_guide_path.read_text(encoding="utf-8")

    user_prompt = (
        f"Here is the style guide:\n{style_guide}\n\n"
        "Please analyze the provided image against this style guide and "
        "list up to 10 specific improvement suggestions to make the image "
        "visually more appealing. If the image is already perfect, just say "
        "'No changes needed'."
    )

    images = [Image(content=base64.b64decode(image_base64), format="jpeg")]

    # Use planner model for text analysis (polish config only has image model)
    model: Gemini = load_models()["planner"]
    model.temperature = 1.0
    agent = Agent(
        model=model,
        system_message=system_prompt,
    )
    response = await agent.arun(input=user_prompt, images=images)
    return response.content.strip()


async def _apply_suggestions(
    image_base64: str,
    suggestions: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Step 2: regenerate image with suggestions applied via Gemini."""
    system_prompt = DIAGRAM_POLISH_SYSTEM_PROMPT

    models = load_models()
    model_id = models["polish"].id

    image_bytes = base64.b64decode(image_base64)
    prompt_text = (
        f"Please polish this image based on the following suggestions:\n\n"
        f"{suggestions}\n\nPolished Image:"
    )

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model_id,
        contents=[
            prompt_text,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            candidate_count=1,
            max_output_tokens=50000,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size="1k",
            ),
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            return base64.b64encode(part.inline_data.data).decode()
    return None
