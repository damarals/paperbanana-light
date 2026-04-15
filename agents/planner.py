"""Planner agent: generates detailed figure descriptions from methodology + caption."""

import base64
import json
from typing import Optional

from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini

from agents.config import load_models

DIAGRAM_PLANNER_SYSTEM_PROMPT = """\
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better."""

async def run(
    content: str,
    visual_intent: str,
    examples: Optional[list[dict]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate a detailed figure description using few-shot examples.

    Args:
        content: methodology text
        visual_intent: diagram caption
        examples: list of dicts with keys: content, visual_intent, image_base64 (optional)
        api_key: Gemini API key (passed via model config if needed)

    Returns:
        Detailed description string.
    """
    content_label = "Methodology Section"
    intent_label = "Diagram Caption"

    if isinstance(content, (dict, list)):
        content = json.dumps(content)

    # Build text prompt with few-shot examples; collect images separately
    text_parts: list[str] = []
    images: list[Image] = []
    for idx, ex in enumerate(examples or []):
        ex_content = ex["content"]
        if isinstance(ex_content, (dict, list)):
            ex_content = json.dumps(ex_content)

        text_parts.append(
            f"Example {idx + 1}:\n"
            f"{content_label}: {ex_content}\n"
            f"{intent_label}: {ex['visual_intent']}\n"
            f"Reference Diagram:"
        )

        if ex.get("image_base64"):
            images.append(
                Image(content=base64.b64decode(ex["image_base64"]), format="jpeg")
            )

    # Target query
    query_text = (
        f"Now, based on the following {content_label.lower()} and "
        f"{intent_label.lower()}, provide a detailed description for the "
        f"figure to be generated.\n"
        f"{content_label}: {content}\n"
        f"{intent_label}: {visual_intent}\n"
        f"Detailed description of the target figure to be generated"
        f" (do not include figure titles):"
    )
    text_parts.append(query_text)
    user_prompt = "\n".join(text_parts)

    model: Gemini = load_models()["planner"]
    model.temperature = 1.0
    agent = Agent(
        model=model,
        system_message=DIAGRAM_PLANNER_SYSTEM_PROMPT,
    )
    response = await agent.arun(input=user_prompt, images=images or None)
    return response.content.strip()
