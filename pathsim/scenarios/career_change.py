"""
Career change scenario.

Models the probability distribution of outcomes when switching to a
new field or role.

Scoring model:

    score = 0.30 * transferable_skills
          + 0.25 * market_demand
          + 0.20 * financial_runway
          + 0.15 * network_strength
          + 0.10 * personal_risk_tolerance

"Success" represents a materially better position (compensation +
satisfaction) within two years.  "Moderate" represents lateral movement
with limited improvement.  "Failure" represents a worse outcome
(compensation drop, regret, return to prior field).
"""

from __future__ import annotations

from dataclasses import dataclass

from pathsim.models import Factor, SimulationConfig


@dataclass
class CareerChangeScenario:
    name: str = "career-change"
    display_name: str = "Career change"
    description: str = (
        "Simulate the outcome of transitioning to a new role or industry."
    )

    def build_factors(self, config: SimulationConfig) -> list[Factor]:
        from pathsim.scoring import capital_to_factor, experience_to_float

        # Interpret experience as transferable skill level
        skill_mean = experience_to_float(config.experience)

        # Interpret capital as financial runway (savings, severance, etc.)
        runway_mean = capital_to_factor(config.capital, reference=60_000.0)

        return [
            Factor(
                name="transferable_skills",
                label="transferable skills",
                distribution="normal",
                params={"mean": skill_mean, "std": 0.20},
                weight=0.30,
            ),
            Factor(
                name="market_demand",
                label="market demand",
                distribution="beta",
                params={"alpha": 3.0, "beta": 2.5},  # generally positive
                weight=0.25,
            ),
            Factor(
                name="financial_runway",
                label="financial runway",
                distribution="normal",
                params={"mean": runway_mean, "std": 0.15},
                weight=0.20,
            ),
            Factor(
                name="network_strength",
                label="network strength",
                distribution="beta",
                params={"alpha": 2.5, "beta": 3.0},
                weight=0.15,
            ),
            Factor(
                name="risk_tolerance",
                label="personal risk tolerance",
                distribution="uniform",
                params={"low": 0.20, "high": 0.80},
                weight=0.10,
            ),
        ]
