"""
Tests for the Monte Carlo simulation runner.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathsim.models import Factor
from pathsim.simulation.monte_carlo import run_simulation


def _basic_factors() -> list[Factor]:
    return [
        Factor(
            name="a",
            label="Factor A",
            distribution="normal",
            params={"mean": 0.6, "std": 0.1},
            weight=0.5,
        ),
        Factor(
            name="b",
            label="Factor B",
            distribution="beta",
            params={"alpha": 3.0, "beta": 2.0},
            weight=0.5,
        ),
    ]


class TestRunSimulation:
    def test_output_shapes(self):
        factors = _basic_factors()
        scores, samples = run_simulation(factors, runs=500, seed=1)
        assert scores.shape == (500,)
        assert set(samples.keys()) == {"a", "b"}
        assert samples["a"].shape == (500,)

    def test_scores_in_unit_interval(self):
        factors = _basic_factors()
        scores, _ = run_simulation(factors, runs=5_000, seed=2)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_reproducibility_with_seed(self):
        factors = _basic_factors()
        scores_1, _ = run_simulation(factors, runs=1000, seed=99)
        scores_2, _ = run_simulation(factors, runs=1000, seed=99)
        np.testing.assert_array_equal(scores_1, scores_2)

    def test_different_seed_different_results(self):
        factors = _basic_factors()
        scores_1, _ = run_simulation(factors, runs=1000, seed=1)
        scores_2, _ = run_simulation(factors, runs=1000, seed=2)
        assert not np.array_equal(scores_1, scores_2)

    def test_zero_weight_raises(self):
        factors = [
            Factor(
                name="z",
                label="Zero",
                distribution="uniform",
                params={"low": 0.0, "high": 1.0},
                weight=0.0,
            )
        ]
        with pytest.raises(ValueError, match="non-zero weight"):
            run_simulation(factors, runs=100)

    def test_single_factor(self):
        factors = [
            Factor(
                name="x",
                label="Only",
                distribution="uniform",
                params={"low": 0.3, "high": 0.7},
                weight=1.0,
            )
        ]
        scores, samples = run_simulation(factors, runs=1000, seed=7)
        assert scores.shape == (1000,)
        np.testing.assert_array_equal(scores, samples["x"])
