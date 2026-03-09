"""
Tests for sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathsim.models import Factor
from pathsim.analysis.sensitivity import compute_sensitivity, _spearman_correlation


class TestSpearmanCorrelation:
    def test_perfect_positive_correlation(self):
        x = np.arange(100, dtype=float)
        rho = _spearman_correlation(x, x)
        assert abs(rho - 1.0) < 1e-6

    def test_perfect_negative_correlation(self):
        x = np.arange(100, dtype=float)
        y = x[::-1]
        rho = _spearman_correlation(x, y)
        assert abs(rho - (-1.0)) < 1e-6

    def test_uncorrelated_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(size=10_000)
        y = rng.uniform(size=10_000)
        rho = _spearman_correlation(x, y)
        assert abs(rho) < 0.05

    def test_too_short_returns_zero(self):
        assert _spearman_correlation(np.array([0.5]), np.array([0.5])) == 0.0


class TestComputeSensitivity:
    def _make_factors(self) -> list[Factor]:
        return [
            Factor("a", "Alpha", "uniform", {"low": 0.0, "high": 1.0}, weight=0.6),
            Factor("b", "Beta",  "uniform", {"low": 0.0, "high": 1.0}, weight=0.4),
        ]

    def test_returns_one_result_per_factor(self):
        factors = self._make_factors()
        rng = np.random.default_rng(0)
        samples = {
            "a": rng.uniform(size=1000),
            "b": rng.uniform(size=1000),
        }
        scores = 0.6 * samples["a"] + 0.4 * samples["b"]
        results = compute_sensitivity(factors, samples, scores)
        assert len(results) == 2

    def test_sorted_descending_by_correlation(self):
        factors = self._make_factors()
        rng = np.random.default_rng(7)
        samples = {
            "a": rng.uniform(size=2000),
            "b": rng.uniform(size=2000),
        }
        scores = 0.9 * samples["a"] + 0.1 * samples["b"]
        results = compute_sensitivity(factors, samples, scores)
        # Factor a should rank first
        assert results[0].factor_name == "a"
        correlations = [r.correlation for r in results]
        assert correlations == sorted(correlations, reverse=True)

    def test_top_n_limits_output(self):
        factors = self._make_factors()
        rng = np.random.default_rng(1)
        samples = {f.name: rng.uniform(size=500) for f in factors}
        scores = sum(samples[f.name] for f in factors) / len(factors)
        results = compute_sensitivity(factors, samples, scores, top_n=1)
        assert len(results) == 1
