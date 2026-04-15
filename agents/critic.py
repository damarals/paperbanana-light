"""Critic agent: validates generated images against descriptions."""

import base64
import json
from typing import Optional

from agno.agent import Agent
from agno.media import Image
from agno.models.google import Gemini
from pydantic import BaseModel

from agents.config import load_models

DIAGRAM_CRITIC_SYSTEM_PROMPT = """\
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Your task is to conduct a sanity check and provide a critique of the target diagram based on its content and presentation. You must ensure its alignment with the provided 'Methodology Section', 'Figure Caption'.

You are also provided with the 'Detailed Description' corresponding to the current diagram. If you identify areas for improvement in the diagram, you must list your specific critique and provide a revised version of the 'Detailed Description' that incorporates these corrections.

## CRITIQUE & REVISION RULES

1. Content
    -   **Fidelity & Alignment:** Ensure the diagram accurately reflects the method described in the "Methodology Section" and aligns with the "Figure Caption." Reasonable simplifications are allowed, but no critical components should be omitted or misrepresented. Also, the diagram should not contain any hallucinated content. Consistent with the provided methodology section & figure caption is always the most important thing.
    -   **Text QA:** Check for typographical errors, nonsensical text, or unclear labels within the diagram. Suggest specific corrections.
    -   **Validation of Examples:** Verify the accuracy of illustrative examples. If the diagram includes specific examples to aid understanding (e.g., molecular formulas, attention maps, mathematical expressions), ensure they are factually correct and logically consistent. If an example is incorrect, provide the correct version.
    -   **Caption Exclusion:** Ensure the figure caption text (e.g., "Figure 1: Overview...") is **not** included within the image visual itself. The caption should remain separate.

2. Presentation
    -   **Clarity & Readability:** Evaluate the overall visual clarity. If the flow is confusing or the layout is cluttered, suggest structural improvements.
    -   **Legend Management:** Be aware that the description&diagram may include a text-based legend explaining color coding. Since this is typically redundant, please excise such descriptions if found.

** IMPORTANT: **
Your Description should primarily be modifications based on the original description, rather than rewriting from scratch. If the original description has obvious problems in certain parts that require re-description, your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background, colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.

## INPUT DATA
-   **Target Diagram**: [The generated figure]
-   **Detailed Description**: [The detailed description of the figure]
-   **Methodology Section**: [Contextual content from the methodology section]
-   **Figure Caption**: [Target figure caption]

## OUTPUT
Provide your response strictly in the following JSON format.

```json
{
    "critic_suggestions": "Insert your detailed critique and specific suggestions for improvement here. If the diagram is perfect, write 'No changes needed.'",
    "revised_description": "Insert the fully revised detailed description here, incorporating all your suggestions. If no changes are needed, write 'No changes needed.'",
}
```"""

class CriticFeedback(BaseModel):
    critic_suggestions: str
    revised_description: str


async def run(
    description: str,
    content: str,
    visual_intent: str,
    image_base64: Optional[str] = None,
    api_key: Optional[str] = None,
) -> CriticFeedback:
    """Critique a generated image against its description.

    Args:
        description: detailed figure description
        content: methodology text
        visual_intent: diagram caption
        image_base64: base64-encoded JPEG of the generated image
        api_key: Gemini API key

    Returns:
        CriticFeedback with suggestions and revised description.
    """
    content_label = "Methodology Section"
    intent_label = "Figure Caption"
    critique_header = "Target Diagram for Critique:"

    if isinstance(content, (dict, list)):
        content = json.dumps(content)

    # Build text prompt; pass image via images= parameter
    images: list[Image] = []
    if image_base64 and len(image_base64) > 100:
        images.append(
            Image(content=base64.b64decode(image_base64), format="jpeg")
        )
        user_prompt = (
            f"{critique_header}\n"
            f"Detailed Description: {description}\n"
            f"{content_label}: {content}\n"
            f"{intent_label}: {visual_intent}\n"
            f"Your Output:"
        )
    else:
        user_prompt = (
            f"{critique_header}\n"
            "[SYSTEM NOTICE] The image could not be generated based on the "
            "current description. Please check the description for errors and "
            "provide a revised version.\n"
            f"Detailed Description: {description}\n"
            f"{content_label}: {content}\n"
            f"{intent_label}: {visual_intent}\n"
            f"Your Output:"
        )

    model: Gemini = load_models()["critic"]
    model.temperature = 1.0
    agent = Agent(
        model=model,
        system_message=DIAGRAM_CRITIC_SYSTEM_PROMPT,
        output_schema=CriticFeedback,
    )
    response = await agent.arun(input=user_prompt, images=images or None)
    return response.content
