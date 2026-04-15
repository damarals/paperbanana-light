"""Stylist agent: refines descriptions using NeurIPS style guidelines."""

import json
from pathlib import Path
from typing import Optional

from agno.agent import Agent
from agno.models.google import Gemini

from agents.config import load_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIAGRAM_STYLIST_SYSTEM_PROMPT = """\
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Our goal is to generate high-quality, publication-ready diagrams, given the methodology section and the caption of the desired diagram. The diagram should illustrate the logic of the methodology section, while adhering to the scope defined by the caption. Before you, a planner agent has already generated a preliminary description of the target diagram. However, this description may lack specific aesthetic details, such as element shapes, color palettes, and background styling. Your task is to refine and enrich this description based on the provided [NeurIPS 2025 Style Guidelines] to ensure the final generated image is a high-quality, publication-ready diagram that adheres to the NeurIPS 2025 aesthetic standards where appropriate.

## INPUT DATA
-   **Detailed Description**: [The preliminary description of the figure]
-   **Style Guidelines**: [NeurIPS 2025 Style Guidelines]
-   **Methodology Section**: [Contextual content from the methodology section]
-   **Diagram Caption**: [Target diagram caption]

Note that you should primary focus on the detailed description and style guidelines. The methodology section and diagram caption are provided for context only, there's no need to regenerate a description from scratch, solely based on them, while ignoring the detailed description we already have.

**Crucial Instructions:**
1.  **Preserve Semantic Content:** Do NOT alter the semantic content, logic, or structure of the diagram. Your job is purely aesthetic refinement, not content editing. However, if you find some phrases or descriptions too verbose, you may simplify them appropriately while referencing the original methodology section to ensure semantic accuracy.
2.  **Preserve High-Quality Aesthetics and Intervene Only When Necessary:** First, evaluate the aesthetic quality implied by the input description. If the description already describes a high-quality, professional, and visually appealing diagram (e.g., nice 3D icons, rich textures, good color harmony), **PRESERVE IT**. Only apply strict Style Guide adjustments if the current description lacks detail, looks outdated, or is visually cluttered. Your goal is specific refinement, not blind standardization.
3.  **Respect Diversity:** Different domains have different styles. If the input describes a specific style (e.g., illustrative for agents) that works well, keep it.
4.  **Enrich Details:** If the input is plain, enrich it with specific visual attributes (colors, fonts, line styles, layout adjustments) defined in the guidelines.
5.  **Handle Icons with Care:** Be cautious when modifying icons as they may carry specific semantic meanings. Some icons have conventional technical meanings (e.g., snowflake = frozen/non-trainable, flame = trainable) - when encountering such icons, reference the original methodology section to verify their intent before making changes. However, purely decorative or symbolic icons can be freely enhanced and beautified. For examples, agent papers often use cute 2D robot avatars to represent agents.

## OUTPUT
Output ONLY the final polished Detailed Description. Do not include any conversational text or explanations."""

async def run(
    description: str,
    content: str,
    visual_intent: str,
    api_key: Optional[str] = None,
) -> str:
    """Refine a description using the NeurIPS style guide.

    Args:
        description: preliminary detailed description from planner
        content: methodology text
        visual_intent: diagram caption
        api_key: Gemini API key

    Returns:
        Polished description string.
    """
    system_prompt = DIAGRAM_STYLIST_SYSTEM_PROMPT
    content_label = "Methodology Section"
    intent_label = "Diagram Caption"

    style_guide_path = (
        PROJECT_ROOT / "assets" / "style_guide.md"
    )
    style_guide = style_guide_path.read_text(encoding="utf-8")

    if isinstance(content, (dict, list)):
        content = json.dumps(content)

    user_prompt = (
        f"Detailed Description: {description}\n"
        f"Style Guidelines: {style_guide}\n"
        f"{content_label}: {content}\n"
        f"{intent_label}: {visual_intent}\n"
        f"Your Output:"
    )

    model = load_models()["stylist"]
    model.temperature = 1.0
    agent = Agent(
        model=model,
        system_message=system_prompt,
    )
    response = await agent.arun(input=user_prompt)
    return response.content.strip()
