"""
Shared scoring utilities.

The weighted linear combination is intentionally kept simple.
Interpretability matters more than model sophistication for a
heuristic tool.
"""

from __future__ import annotations

import numpy as np

from pathsim.models import Factor


def weighted_score(
    factors: list[Factor],
    samples: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute a weighted linear combination of sampled factor values.

    Weights are normalised so they always sum to 1.0 regardless of
    how individual factors are defined.

    Parameters
    ----------
    factors:
        Factor definitions with weights.
    samples:
        Dict of factor_name → array of sampled values.

    Returns
    -------
    np.ndarray of shape (runs,) with scores in [0, 1].
    """
    total_weight = sum(f.weight for f in factors)
    if total_weight == 0:
        raise ValueError("Total factor weight must be > 0.")

    scores = np.zeros(len(next(iter(samples.values()))), dtype=np.float64)
    for factor in factors:
        scores += (factor.weight / total_weight) * samples[factor.name]

    return scores


def experience_to_float(experience: str | None) -> float:
    """
    Convert a human-readable experience level to a numeric prior.

    Used by scenarios to adjust factor distribution parameters based
    on CLI input.
    """
    mapping = {
        "low": 0.25,
        "mid": 0.50,
        "high": 0.75,
    }
    if experience is None:
        return 0.50
    value = mapping.get(experience.lower())
    if value is None:
        raise ValueError(
            f"Unknown experience level '{experience}'. "
            f"Valid values: {list(mapping.keys())}"
        )
    return value


def capital_to_factor(capital: float | None, reference: float = 500_000.0) -> float:
    """
    Normalise capital to a [0, 1] factor prior.

    `reference` is the amount considered a reasonable starting point
    for the scenario.  Capital above 4× reference saturates at 1.0.
    """
    if capital is None:
        return 0.50
    ratio = float(capital) / reference
    return float(np.clip(ratio / 4.0, 0.0, 1.0))
