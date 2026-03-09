"""
Sensitivity analysis via Spearman rank correlation.

For each factor, compute the correlation between its sampled values
and the final composite score across all Monte Carlo runs.  Higher
absolute correlation means that factor drives outcomes more.

Spearman (rather than Pearson) is used because factor distributions
are non-Gaussian and relationships may be monotonic but non-linear.
"""

from __future__ import annotations

import numpy as np

from pathsim.models import Factor, SensitivityResult


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Spearman rank correlation between two 1-D arrays.

    Implemented from scratch to avoid a scipy dependency.
    """
    n = len(x)
    if n < 2:
        return 0.0

    rank_x = _rank(x)
    rank_y = _rank(y)

    d = rank_x - rank_y
    d_sq_sum = float(np.sum(d**2))

    rho = 1.0 - (6.0 * d_sq_sum) / (n * (n**2 - 1))
    return float(np.clip(rho, -1.0, 1.0))


def _rank(arr: np.ndarray) -> np.ndarray:
    """Return ordinal ranks (1-based) for a 1-D array."""
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    return ranks


def compute_sensitivity(
    factors: list[Factor],
    samples: dict[str, np.ndarray],
    scores: np.ndarray,
    top_n: int | None = None,
) -> list[SensitivityResult]:
    """
    Return sensitivity results sorted by absolute Spearman correlation,
    descending.

    Parameters
    ----------
    factors:
        Factor definitions (used for labels).
    samples:
        Dict of factor_name → sampled values array from the simulation.
    scores:
        Composite scores array from the simulation.
    top_n:
        If given, return only the top N most influential factors.
    """
    results: list[SensitivityResult] = []

    for factor in factors:
        values = samples.get(factor.name)
        if values is None:
            continue

        rho = _spearman_correlation(values, scores)

        results.append(
            SensitivityResult(
                factor_name=factor.name,
                label=factor.label,
                correlation=abs(rho),
            )
        )

    results.sort(key=lambda r: r.correlation, reverse=True)

    if top_n is not None:
        results = results[:top_n]

    return results
