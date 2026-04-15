"""Visualizer agent: renders diagram images from detailed descriptions.

Diagrams use Gemini image generation directly (response_modalities=["IMAGE"]).
"""

import asyncio
import base64
from typing import Optional

from google import genai
from google.genai import types

from agents.config import load_models

DIAGRAM_VISUALIZER_SYSTEM_PROMPT = (
    "You are an expert scientific diagram illustrator. "
    "Generate high-quality scientific diagrams based on user requests."
)


async def run(
    description: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Render a diagram from a detailed description.

    Args:
        description: detailed figure description
        api_key: Gemini API key
        aspect_ratio: aspect ratio for image generation

    Returns:
        Base64-encoded JPEG string, or None on failure.
    """
    return await _generate_diagram(description, api_key, aspect_ratio)


async def _generate_diagram(
    description: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Generate a diagram image via Gemini image generation."""
    models = load_models()
    model_id = models["visualizer"].id

    prompt = (
        f"Render an image based on the following detailed description: {description}\n"
        " Note that do not include figure titles in the image. Diagram: "
    )

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model_id,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=DIAGRAM_VISUALIZER_SYSTEM_PROMPT,
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
