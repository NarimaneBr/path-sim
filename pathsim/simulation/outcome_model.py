"""
Outcome classification model.

The scoring model maps a raw score in [0, 1] to one of three outcome
categories.  Thresholds are defined here and intentionally exposed so
that callers can inspect or override them for testing.
"""

from __future__ import annotations

import numpy as np

from pathsim.models import OutcomeCategory, OutcomeDistribution

# Default classification thresholds
SUCCESS_THRESHOLD: float = 0.65
FAILURE_THRESHOLD: float = 0.35


def classify_scores(
    scores: np.ndarray,
    success_threshold: float = SUCCESS_THRESHOLD,
    failure_threshold: float = FAILURE_THRESHOLD,
) -> np.ndarray:
    """
    Classify an array of raw scores into outcome categories.

    Returns an array of strings ("success" | "moderate" | "failure")
    with the same length as `scores`.
    """
    categories = np.where(
        scores >= success_threshold,
        OutcomeCategory.SUCCESS.value,
        np.where(
            scores >= failure_threshold,
            OutcomeCategory.MODERATE.value,
            OutcomeCategory.FAILURE.value,
        ),
    )
    return categories


def compute_outcome_distribution(
    scores: np.ndarray,
    success_threshold: float = SUCCESS_THRESHOLD,
    failure_threshold: float = FAILURE_THRESHOLD,
) -> OutcomeDistribution:
    """
    Given all simulation scores, return the fraction that fall into each
    outcome category.
    """
    n = len(scores)
    categories = classify_scores(scores, success_threshold, failure_threshold)

    return OutcomeDistribution(
        success=float(np.sum(categories == OutcomeCategory.SUCCESS.value) / n),
        moderate=float(np.sum(categories == OutcomeCategory.MODERATE.value) / n),
        failure=float(np.sum(categories == OutcomeCategory.FAILURE.value) / n),
    )
