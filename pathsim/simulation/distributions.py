"""
Probability distribution samplers.

Each function accepts a size parameter and returns a numpy array of
samples clipped to [0, 1] so that all factor values stay in a
normalised range that the scoring model expects.
"""

from __future__ import annotations

import numpy as np


def sample_normal(
    mean: float,
    std: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a normal distribution, clipped to [0, 1]."""
    raw = rng.normal(loc=mean, scale=std, size=size)
    return np.clip(raw, 0.0, 1.0)


def sample_beta(
    alpha: float,
    beta: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a Beta distribution.  Output is already in [0, 1]."""
    return rng.beta(a=alpha, b=beta, size=size)


def sample_uniform(
    low: float,
    high: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a uniform distribution, clipped to [0, 1]."""
    raw = rng.uniform(low=low, high=high, size=size)
    return np.clip(raw, 0.0, 1.0)


def sample_factor(
    distribution: str,
    params: dict[str, float],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to the appropriate sampler based on distribution name."""
    if distribution == "normal":
        return sample_normal(
            mean=params["mean"],
            std=params["std"],
            size=size,
            rng=rng,
        )
    elif distribution == "beta":
        return sample_beta(
            alpha=params["alpha"],
            beta=params["beta"],
            size=size,
            rng=rng,
        )
    elif distribution == "uniform":
        return sample_uniform(
            low=params["low"],
            high=params["high"],
            size=size,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown distribution: '{distribution}'")
