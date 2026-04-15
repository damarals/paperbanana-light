"""Captioner agent: proposes a figure caption from methodology + reference captions."""

from typing import Optional

from agno.agent import Agent

from agents.config import load_models

SYSTEM_PROMPT = """\
You are an expert at writing concise, informative figure captions for academic \
papers. Given the methodology section of a paper and examples of well-crafted \
captions from similar papers, you propose ONE figure caption for a Figure 1 \
overview diagram that summarizes the proposed approach.

Guidelines:
- Start with "Figure 1:".
- 1-3 sentences, dense with information.
- Name the main components and how they connect.
- Match the style and register of the reference captions.
- Do not hedge ("we propose...", "this paper presents..."). State the diagram.
- Output ONLY the caption text. No preamble, no markdown, no quotes."""


async def run(
    content: str,
    examples: Optional[list[dict]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Propose a figure caption grounded in methodology and reference captions.

    Args:
        content: methodology section text
        examples: list of dicts with key 'visual_intent' (reference captions)
        api_key: unused; kept for call-site symmetry with other agents

    Returns:
        Caption string starting with 'Figure 1:'.
    """
    ref_block = "\n".join(
        f"- {ex['visual_intent']}"
        for ex in (examples or [])[:5]
        if ex.get("visual_intent")
    )

    user_prompt = (
        "Reference captions from similar papers (for style only, not content):\n"
        f"{ref_block}\n\n"
        "Methodology section:\n"
        f"{content}\n\n"
        "Your proposed Figure 1 caption:"
    )

    model = load_models()["captioner"]
    model.temperature = 0.7
    agent = Agent(model=model, system_message=SYSTEM_PROMPT)
    response = await agent.arun(input=user_prompt)
    return response.content.strip()
