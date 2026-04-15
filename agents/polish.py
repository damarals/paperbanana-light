"""Polish agent: two-step style refinement of existing images.

Step 1: Critique image against style guide (text LLM via Agno).
Step 2: Apply suggestions to regenerate image (Gemini image generation).
"""

import asyncio
import base64
from pathlib import Path
from typing import Optional

from agno.agent import Agent
from google import genai
from google.genai import types

from agents.config import load_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIAGRAM_SUGGESTION_SYSTEM_PROMPT = """\
You are a senior art director for NeurIPS 2025. Your task is to critique a diagram against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts, icons).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the diagram is substantially compliant, output "No changes needed"."""

PLOT_SUGGESTION_SYSTEM_PROMPT = """\
You are a senior data visualization expert for NeurIPS 2025. Your task is to critique a plot against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the plot is substantially compliant, output "No changes needed"."""

DIAGRAM_POLISH_SYSTEM_PROMPT = """\
## ROLE
You are a professional diagram polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing diagram image and a list of specific improvement suggestions. Your task is to generate a polished version of this diagram by applying these suggestions while preserving the semantic logic and structure of the original diagram.

## OUTPUT
Generate a polished diagram image that maintains the original content while applying the improvement suggestions."""

PLOT_POLISH_SYSTEM_PROMPT = """\
## ROLE
You are a professional plot polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing statistical plot image and a list of specific improvement suggestions. Your task is to generate a polished version of this plot by applying these suggestions while preserving all the data and quantitative information.

**Important Instructions:**
1. **Preserve Data:** Do NOT alter any data points, values, or quantitative information in the plot.
2. **Apply Suggestions:** Enhance the visual aesthetics according to the provided suggestions (colors, fonts, layout, etc.).
3. **Maintain Accuracy:** Ensure all numerical values and relationships remain accurate.
4. **Professional Quality:** Ensure the output meets publication standards for top-tier conferences.

## OUTPUT
Generate a polished plot image that maintains the original data while applying the improvement suggestions."""


async def run(
    task_name: str,
    image_base64: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Polish an existing image using style guidelines.

    Args:
        task_name: "diagram" or "plot"
        image_base64: base64-encoded JPEG of the source image
        api_key: Gemini API key
        aspect_ratio: aspect ratio for regenerated image

    Returns:
        Base64-encoded JPEG of the polished image, or None on failure.
    """
    # Step 1: generate suggestions
    suggestions = await _generate_suggestions(task_name, image_base64, api_key)
    if not suggestions or "No changes needed" in suggestions:
        return None

    # Step 2: apply suggestions via image generation
    return await _apply_suggestions(
        task_name, image_base64, suggestions, api_key, aspect_ratio,
    )


async def _generate_suggestions(
    task_name: str,
    image_base64: str,
    api_key: Optional[str] = None,
) -> str:
    """Step 1: critique image against the NeurIPS style guide."""
    system_prompt = (
        DIAGRAM_SUGGESTION_SYSTEM_PROMPT if task_name == "diagram"
        else PLOT_SUGGESTION_SYSTEM_PROMPT
    )

    style_guide_path = (
        PROJECT_ROOT / "style_guides" / f"neurips2025_{task_name}_style_guide.md"
    )
    style_guide = style_guide_path.read_text(encoding="utf-8")

    user_parts: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Here is the style guide:\n{style_guide}\n\n"
                "Please analyze the provided image against this style guide and "
                "list up to 10 specific improvement suggestions to make the image "
                "visually more appealing. If the image is already perfect, just say "
                "'No changes needed'."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
            },
        },
    ]

    models = load_models()
    # Use planner model for text analysis (polish config only has image model)
    agent = Agent(
        model=models["planner"],
        system_prompt=system_prompt,
        temperature=1.0,
    )
    response = await agent.arun(message=user_parts)
    return response.content.strip()


async def _apply_suggestions(
    task_name: str,
    image_base64: str,
    suggestions: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Step 2: regenerate image with suggestions applied via Gemini."""
    system_prompt = (
        DIAGRAM_POLISH_SYSTEM_PROMPT if task_name == "diagram"
        else PLOT_POLISH_SYSTEM_PROMPT
    )

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
