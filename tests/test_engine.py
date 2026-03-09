"""
Tests for the engine orchestration layer.
"""

from __future__ import annotations

import pytest

from pathsim.engine import simulate, list_scenarios
from pathsim.models import SimulationConfig, SimulationResult


class TestSimulate:
    def test_startup_returns_result(self):
        config = SimulationConfig(decision="startup", runs=500, seed=0)
        result = simulate(config)
        assert isinstance(result, SimulationResult)

    def test_career_change_returns_result(self):
        config = SimulationConfig(decision="career-change", runs=500, seed=0)
        result = simulate(config)
        assert isinstance(result, SimulationResult)

    def test_investment_returns_result(self):
        config = SimulationConfig(decision="investment", runs=500, seed=0)
        result = simulate(config)
        assert isinstance(result, SimulationResult)

    def test_natural_language_decision(self):
        config = SimulationConfig(decision="start a startup", runs=200, seed=42)
        result = simulate(config)
        assert "startup" in result.scenario_name.lower()

    def test_outcome_fractions_sum_to_one(self):
        config = SimulationConfig(decision="startup", runs=2000, seed=1)
        result = simulate(config)
        total = result.outcomes.success + result.outcomes.moderate + result.outcomes.failure
        assert abs(total - 1.0) < 1e-9

    def test_sensitivity_sorted_descending(self):
        config = SimulationConfig(decision="startup", runs=2000, seed=5)
        result = simulate(config)
        correlations = [s.correlation for s in result.sensitivity]
        assert correlations == sorted(correlations, reverse=True)

    def test_unknown_decision_raises(self):
        config = SimulationConfig(decision="buy a house", runs=100)
        with pytest.raises(ValueError, match="Cannot map"):
            simulate(config)

    def test_experience_high_increases_success(self):
        low_cfg = SimulationConfig(decision="startup", runs=5000, seed=10, experience="low")
        high_cfg = SimulationConfig(decision="startup", runs=5000, seed=10, experience="high")
        low_result = simulate(low_cfg)
        high_result = simulate(high_cfg)
        assert high_result.outcomes.success > low_result.outcomes.success


class TestListScenarios:
    def test_returns_three_scenarios(self):
        scenarios = list_scenarios()
        assert len(scenarios) == 3

    def test_all_have_required_keys(self):
        for s in list_scenarios():
            assert "name" in s
            assert "display" in s
            assert "description" in s
