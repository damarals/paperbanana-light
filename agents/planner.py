"""Planner agent: generates detailed figure descriptions from methodology + caption."""

import base64
import json
from typing import Optional

from agno.agent import Agent

from agents.config import load_models

DIAGRAM_PLANNER_SYSTEM_PROMPT = """\
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better."""

PLOT_PLANNER_SYSTEM_PROMPT = """\
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that are both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data.  Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes)."""

_CONTENT_LABELS = {
    "diagram": ("Methodology Section", "Diagram Caption"),
    "plot": ("Plot Raw Data", "Visual Intent of the Desired Plot"),
}


async def run(
    task_name: str,
    content: str,
    visual_intent: str,
    examples: Optional[list[dict]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate a detailed figure description using few-shot examples.

    Args:
        task_name: "diagram" or "plot"
        content: methodology text or raw data
        visual_intent: caption or plot intent
        examples: list of dicts with keys: content, visual_intent, image_base64 (optional)
        api_key: Gemini API key (passed via model config if needed)

    Returns:
        Detailed description string.
    """
    system_prompt = (
        DIAGRAM_PLANNER_SYSTEM_PROMPT if task_name == "diagram"
        else PLOT_PLANNER_SYSTEM_PROMPT
    )
    content_label, intent_label = _CONTENT_LABELS[task_name]

    if isinstance(content, (dict, list)):
        content = json.dumps(content)

    # Build multimodal prompt with few-shot examples
    user_parts: list[dict] = []
    for idx, ex in enumerate(examples or []):
        ex_content = ex["content"]
        if isinstance(ex_content, (dict, list)):
            ex_content = json.dumps(ex_content)

        text = (
            f"Example {idx + 1}:\n"
            f"{content_label}: {ex_content}\n"
            f"{intent_label}: {ex['visual_intent']}\n"
            f"Reference {task_name.capitalize()}: "
        )
        user_parts.append({"type": "text", "text": text})

        if ex.get("image_base64"):
            user_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{ex['image_base64']}",
                },
            })

    # Target query
    query_text = (
        f"Now, based on the following {content_label.lower()} and "
        f"{intent_label.lower()}, provide a detailed description for the "
        f"figure to be generated.\n"
        f"{content_label}: {content}\n"
        f"{intent_label}: {visual_intent}\n"
        f"Detailed description of the target figure to be generated"
    )
    if task_name == "diagram":
        query_text += " (do not include figure titles)"
    query_text += ":"
    user_parts.append({"type": "text", "text": query_text})

    models = load_models()
    agent = Agent(
        model=models["planner"],
        system_prompt=system_prompt,
        temperature=1.0,
    )
    response = await agent.arun(message=user_parts)
    return response.content.strip()
