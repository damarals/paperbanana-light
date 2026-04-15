"""Shared model configuration loader."""

import yaml
from pathlib import Path

from agno.models.google import Gemini

_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


def load_models() -> dict:
    """Load models.yaml and return a dict of {agent_name: Gemini instance}."""
    with open(_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    return {name: Gemini(id=v["model"]) for name, v in cfg.items() if "model" in v}
