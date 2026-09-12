"""
Deterministic seeding utilities.

Why this file exists:
Every experiment in this project must be reproducible (same config + seed
must always produce the same metrics). This module is the single place
where random seeds are set across all libraries that introduce randomness
(Python's random module, NumPy, PyTorch). Every script that runs an
experiment (scripts/run_experiment.py, evaluation/experiments.py) must
call set_seed() before doing anything else.

This logic does not belong in individual modules (e.g. detection/train.py)
because seeding must be applied globally, once, before any random
operation happens anywhere in the pipeline.
"""

import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch (if installed).

    Args:
        seed: Integer seed value, should come from experiment config.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
