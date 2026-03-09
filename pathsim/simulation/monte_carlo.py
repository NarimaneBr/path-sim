"""
Monte Carlo simulation runner.

Takes a list of Factor definitions, samples them in vectorised form,
applies the scoring weights, and returns raw scores for every run.
"""

from __future__ import annotations

import numpy as np

from pathsim.models import Factor
from pathsim.simulation.distributions import sample_factor


def run_simulation(
    factors: list[Factor],
    runs: int,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Execute `runs` Monte Carlo iterations.

    Parameters
    ----------
    factors:
        Ordered list of Factor objects.  Weights do not need to sum to
        exactly 1.0 — they are normalised internally so that the
        scoring model is always well-defined.
    runs:
        Number of simulation iterations.
    seed:
        Optional integer seed for reproducibility.

    Returns
    -------
    scores:
        1-D array of shape (runs,) with composite scores in [0, 1].
    samples:
        Dict mapping factor name → array of shape (runs,) with the
        raw sampled values, for use in sensitivity analysis.
    """
    rng = np.random.default_rng(seed)

    total_weight = sum(f.weight for f in factors)
    if total_weight == 0:
        raise ValueError("At least one factor must have a non-zero weight.")

    samples: dict[str, np.ndarray] = {}
    scores = np.zeros(runs, dtype=np.float64)

    for factor in factors:
        values = sample_factor(
            distribution=factor.distribution,
            params=factor.params,
            size=runs,
            rng=rng,
        )
        samples[factor.name] = values
        scores += (factor.weight / total_weight) * values

    return scores, samples
