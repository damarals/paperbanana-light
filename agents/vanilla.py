"""Vanilla agent: direct single-step figure generation without few-shot examples.

Diagrams use Gemini image generation directly.
Plots use LLM code generation + matplotlib execution.
"""

import asyncio
import base64
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from agno.agent import Agent
from google import genai
from google.genai import types

from agents.config import load_models

DIAGRAM_VANILLA_SYSTEM_PROMPT = """\
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with a "Method Section" and a "Diagram Caption". Your task is to generate a high-quality scientific diagram that effectively illustrates the method described in the text, as the caption requires, and adhering strictly to modern academic visualization standards.

**CRITICAL INSTRUCTION ON CAPTION:**
The "Diagram Caption" is provided solely to describe the visual content and logic you need to draw. **DO NOT render, write, or include the caption text itself (e.g., "Figure 1: ...") inside the generated image.**

## INPUT DATA
-   **Method Section**: [Content of method section]
-   **Diagram Caption**: [Diagram caption]
## OUTPUT
Generate a single, high-resolution image that visually explains the method and aligns well with the caption."""

PLOT_VANILLA_SYSTEM_PROMPT = """\
## ROLE
You are an expert statistical plot illustrator for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with "Plot Raw Data" and a "Visual Intent of the Desired Plot". Your task is to write matplotlib code to generate a high-quality statistical plot that effectively visualizes the data according to the visual intent, adhering strictly to modern academic visualization standards.

## INPUT DATA
-   **Plot Raw Data**: [Raw data to be visualized]
-   **Visual Intent of the Desired Plot**: [Description of what the plot should convey]

## OUTPUT
Write Python matplotlib code to generate the plot. Only provide the code without any explanations."""


async def run(
    task_name: str,
    content: str,
    visual_intent: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Generate a figure directly without few-shot examples.

    Args:
        task_name: "diagram" or "plot"
        content: methodology text or raw data
        visual_intent: caption or plot intent
        api_key: Gemini API key
        aspect_ratio: aspect ratio for image generation (diagrams only)

    Returns:
        Base64-encoded JPEG string, or None on failure.
    """
    if isinstance(content, (dict, list)):
        content = json.dumps(content)

    if task_name == "diagram":
        return await _generate_diagram(content, visual_intent, api_key, aspect_ratio)
    return await _generate_plot(content, visual_intent, api_key)


async def _generate_diagram(
    content: str,
    visual_intent: str,
    api_key: Optional[str] = None,
    aspect_ratio: str = "16:9",
) -> Optional[str]:
    """Generate a diagram image directly via Gemini."""
    models = load_models()
    model_id = models["vanilla"].id

    prompt = (
        f"**Method Section**: {content}\n"
        f"**Diagram Caption**: {visual_intent}\n"
        "Note that do not include figure titles in the image."
        "**Generated Diagram**: "
    )

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model_id,
        contents=[prompt],
        config=types.GenerateContentConfig(
            system_instruction=DIAGRAM_VANILLA_SYSTEM_PROMPT,
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
    content: str,
    visual_intent: str,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Generate a plot by producing matplotlib code and executing it."""
    models = load_models()

    prompt = (
        f"**Plot Raw Data**: {content}\n"
        f"**Visual Intent of the Desired Plot**: {visual_intent}\n"
        "\nUse python matplotlib to generate a statistical plot based on the "
        "above information. Only provide the code without any explanations. Code:"
    )

    agent = Agent(
        model=models["vanilla"],
        system_prompt=PLOT_VANILLA_SYSTEM_PROMPT,
        temperature=1.0,
    )
    response = await agent.arun(message=prompt)
    code_text = response.content

    return await asyncio.to_thread(_execute_plot_code, code_text)


def _execute_plot_code(code_text: str) -> Optional[str]:
    """Extract and execute matplotlib code, return base64 JPEG."""
    match = re.search(r"```python(.*?)```", code_text, re.DOTALL)
    code_clean = match.group(1).strip() if match else code_text.strip()

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        script = f.name
        out_path = script.replace(".py", ".jpg")
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
            print(f"[vanilla] plot code failed: {result.stderr[:500]}")
            return None

        out = Path(out_path)
        if out.exists():
            return base64.b64encode(out.read_bytes()).decode()
        return None
    except subprocess.TimeoutExpired:
        print("[vanilla] plot code execution timed out")
        return None
    finally:
        Path(script).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)
