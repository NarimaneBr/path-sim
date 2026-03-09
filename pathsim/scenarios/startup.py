"""
Startup scenario.

Models the probability distribution of outcomes when founding a
technology company.  Calibration is based on commonly cited early-stage
startup statistics (high failure rates, rare breakout successes).

The scoring model:

    score = 0.30 * founder_skill
          + 0.25 * market_size
          + 0.20 * market_timing
          + 0.15 * capital
          + 0.10 * competition_inverse

All weights are normalised inside the engine, so individual weight
declarations here represent relative importance, not absolute fractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathsim.models import Factor, SimulationConfig


@dataclass
class StartupScenario:
    name: str = "startup"
    display_name: str = "Start a startup"
    description: str = (
        "Simulate the outcome of founding an early-stage technology company."
    )

    def build_factors(self, config: SimulationConfig) -> list[Factor]:
        """
        Construct factor list, optionally adjusted by CLI parameters.

        - --experience shifts the founder_skill mean
        - --capital shifts the capital factor mean
        - --team shifts the founder_skill std (more founders = less variance)
        """
        from pathsim.scoring import capital_to_factor, experience_to_float

        exp_mean = experience_to_float(config.experience)
        capital_mean = capital_to_factor(config.capital, reference=500_000.0)

        # More team members slightly reduce uncertainty in founder skill
        team = config.team or 2
        skill_std = max(0.10, 0.25 - 0.02 * (team - 1))

        return [
            Factor(
                name="founder_skill",
                label="founder experience",
                distribution="normal",
                params={"mean": exp_mean, "std": skill_std},
                weight=0.30,
            ),
            Factor(
                name="market_size",
                label="market size",
                distribution="beta",
                params={"alpha": 2.0, "beta": 3.0},
                weight=0.25,
            ),
            Factor(
                name="market_timing",
                label="market timing",
                distribution="beta",
                params={"alpha": 2.0, "beta": 4.0},
                weight=0.20,
            ),
            Factor(
                name="capital",
                label="capital",
                distribution="normal",
                params={"mean": capital_mean, "std": 0.15},
                weight=0.15,
            ),
            Factor(
                name="competition_inverse",
                label="competition",
                distribution="beta",
                params={"alpha": 2.0, "beta": 5.0},
                weight=0.10,
            ),
        ]
