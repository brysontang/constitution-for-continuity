"""Constitution principle loader and sampler."""

import json
import random
from pathlib import Path


def load_principles(path: str = "data/principles.json") -> list[dict]:
    """Load principles from JSON file."""
    with open(path) as f:
        return json.load(f)


def sample_principles(principles: list[dict], n: int = 1) -> list[dict]:
    """Sample n random principles without replacement (or all if n >= len)."""
    n = min(n, len(principles))
    return random.sample(principles, n)


def sample_principle(principles: list[dict]) -> dict:
    """Sample a single random principle."""
    return random.choice(principles)


def get_principles_by_category(
    principles: list[dict], category: str
) -> list[dict]:
    """Filter principles by category."""
    return [p for p in principles if p["category"] == category]


def format_principle_for_critique(principle: dict) -> str:
    """Format a principle into a critique prompt string."""
    return (
        f"Principle ({principle['category']}): {principle['statement']}\n\n"
        f"Critique instruction: {principle['critique_prompt']}"
    )
