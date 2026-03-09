"""
Tests for the outcome classification model.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathsim.simulation.outcome_model import (
    SUCCESS_THRESHOLD,
    FAILURE_THRESHOLD,
    classify_scores,
    compute_outcome_distribution,
)
from pathsim.models import OutcomeCategory


class TestClassifyScores:
    def test_success_region(self):
        scores = np.array([0.70, 0.80, 0.90])
        result = classify_scores(scores)
        assert all(r == OutcomeCategory.SUCCESS.value for r in result)

    def test_moderate_region(self):
        scores = np.array([0.40, 0.50, 0.60])
        result = classify_scores(scores)
        assert all(r == OutcomeCategory.MODERATE.value for r in result)

    def test_failure_region(self):
        scores = np.array([0.10, 0.20, 0.30])
        result = classify_scores(scores)
        assert all(r == OutcomeCategory.FAILURE.value for r in result)

    def test_threshold_boundaries(self):
        scores = np.array([SUCCESS_THRESHOLD, FAILURE_THRESHOLD])
        result = classify_scores(scores)
        assert result[0] == OutcomeCategory.SUCCESS.value
        assert result[1] == OutcomeCategory.MODERATE.value


class TestComputeOutcomeDistribution:
    def test_fractions_sum_to_one(self):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0.0, 1.0, size=10_000)
        dist = compute_outcome_distribution(scores)
        total = dist.success + dist.moderate + dist.failure
        assert abs(total - 1.0) < 1e-9

    def test_all_success(self):
        scores = np.full(100, 0.9)
        dist = compute_outcome_distribution(scores)
        assert dist.success == pytest.approx(1.0)
        assert dist.moderate == pytest.approx(0.0)
        assert dist.failure == pytest.approx(0.0)

    def test_all_failure(self):
        scores = np.full(100, 0.1)
        dist = compute_outcome_distribution(scores)
        assert dist.failure == pytest.approx(1.0)

    def test_custom_thresholds(self):
        scores = np.array([0.50, 0.50, 0.50])
        dist = compute_outcome_distribution(scores, success_threshold=0.40, failure_threshold=0.30)
        assert dist.success == pytest.approx(1.0)
