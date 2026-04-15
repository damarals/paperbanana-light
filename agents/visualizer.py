"""Visualizer agent: renders images from detailed descriptions.

Diagrams use Gemini image generation directly (response_modalities=["IMAGE"]).
Plots use LLM code generation + matplotlib subprocess execution.
"""

import asyncio
import base64
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from agents.config import load_models

DIAGRAM_VISUALIZER_SYSTEM_PROMPT = (
    "You are an expert scientific diagram illustrator. "
    "Generate high-quality scientific diagrams based on user requests."
)

PLOT_VISUALIZER_SYSTEM_PROMPT = (
    "You are an expert statistical plot illustrator. "
    "Write code to generate high-quality statistical plots based on user requests."
)


async def run(
    task_name: str,
    description: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Render a figure from a detailed description.

    Args:
        task_name: "diagram" or "plot"
        description: detailed figure description
        api_key: Gemini API key
        aspect_ratio: aspect ratio for image generation (diagrams only)

    Returns:
        Base64-encoded JPEG string, or None on failure.
    """
    if task_name == "diagram":
        return await _generate_diagram(description, api_key, aspect_ratio)
    return await _generate_plot(description, api_key)


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


async def _generate_plot(
    description: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Generate a plot by producing matplotlib code and executing it."""
    from agno.agent import Agent

    models = load_models()
    prompt = (
        f"Use python matplotlib to generate a statistical plot based on the "
        f"following detailed description: {description}\n"
        " Only provide the code without any explanations. Code:"
    )

    agent = Agent(
        model=models["visualizer"],
        system_prompt=PLOT_VISUALIZER_SYSTEM_PROMPT,
        temperature=1.0,
    )
    response = await agent.arun(message=prompt)
    code_text = response.content

    return await asyncio.to_thread(_execute_plot_code, code_text)


def _execute_plot_code(code_text: str) -> Optional[str]:
    """Extract and execute matplotlib code, return base64 JPEG."""
    match = re.search(r"```python(.*?)```", code_text, re.DOTALL)
    code_clean = match.group(1).strip() if match else code_text.strip()

    # Ensure the code saves to a known path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        script = f.name
        out_path = script.replace(".py", ".jpg")
        # Prepend backend setup and append save logic
        wrapper = (
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "plt.rcdefaults()\n"
            f"{code_clean}\n"
            "import matplotlib.pyplot as _plt\n"
            "if _plt.get_fignums():\n"
            f"    _plt.savefig({out_path!r}, format='jpeg', bbox_inches='tight', dpi=300)\n"
            "    _plt.close('all')\n"
        )
        f.write(wrapper)

    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"[visualizer] plot code failed: {result.stderr[:500]}")
            return None

        out = Path(out_path)
        if out.exists():
            return base64.b64encode(out.read_bytes()).decode()
        return None
    except subprocess.TimeoutExpired:
        print("[visualizer] plot code execution timed out")
        return None
    finally:
        Path(script).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
