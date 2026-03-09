"""
Tests for probability distribution samplers.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathsim.simulation.distributions import (
    sample_beta,
    sample_factor,
    sample_normal,
    sample_uniform,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestSampleNormal:
    def test_output_shape(self, rng):
        result = sample_normal(0.5, 0.1, size=1000, rng=rng)
        assert result.shape == (1000,)

    def test_clipped_to_unit_interval(self, rng):
        # Extreme std to force values outside [0,1] before clipping
        result = sample_normal(0.5, 10.0, size=5000, rng=rng)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_mean_approximately_correct(self, rng):
        result = sample_normal(0.5, 0.05, size=50_000, rng=rng)
        assert abs(np.mean(result) - 0.5) < 0.01


class TestSampleBeta:
    def test_output_in_unit_interval(self, rng):
        result = sample_beta(2.0, 5.0, size=1000, rng=rng)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_skewed_left_for_high_beta(self, rng):
        # alpha=1, beta=5 should give mean ~ 1/6
        result = sample_beta(1.0, 5.0, size=50_000, rng=rng)
        assert np.mean(result) < 0.25


class TestSampleUniform:
    def test_output_in_range(self, rng):
        result = sample_uniform(0.2, 0.8, size=1000, rng=rng)
        assert np.all(result >= 0.2)
        assert np.all(result <= 0.8)

    def test_output_shape(self, rng):
        result = sample_uniform(0.0, 1.0, size=500, rng=rng)
        assert result.shape == (500,)


class TestSampleFactor:
    def test_dispatch_normal(self, rng):
        result = sample_factor("normal", {"mean": 0.5, "std": 0.1}, size=100, rng=rng)
        assert result.shape == (100,)

    def test_dispatch_beta(self, rng):
        result = sample_factor("beta", {"alpha": 2.0, "beta": 3.0}, size=100, rng=rng)
        assert result.shape == (100,)

    def test_dispatch_uniform(self, rng):
        result = sample_factor("uniform", {"low": 0.0, "high": 1.0}, size=100, rng=rng)
        assert result.shape == (100,)

    def test_unknown_distribution_raises(self, rng):
        with pytest.raises(ValueError, match="Unknown distribution"):
            sample_factor("poisson", {"lambda": 1.0}, size=100, rng=rng)
