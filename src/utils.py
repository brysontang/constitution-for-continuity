"""Shared utilities: config loading, seeding, wandb init."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb(config: dict, run_name_suffix: str = "", dry_run: bool = False):
    """Initialize wandb run. Returns the run object or None if dry_run."""
    if dry_run:
        return None
    import wandb

    name = config["experiment"]["name"]
    if run_name_suffix:
        name = f"{name}-{run_name_suffix}"
    return wandb.init(
        project=config["wandb"]["project"],
        name=name,
        config=config,
        tags=config["wandb"].get("tags", []),
    )


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent
