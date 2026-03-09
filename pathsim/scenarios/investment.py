"""
Investment scenario.

Models the probability distribution of outcomes when committing capital
to a speculative asset (equity, real estate, early-stage company, etc.).

Scoring model:

    score = 0.30 * asset_quality
          + 0.25 * market_conditions
          + 0.20 * position_sizing
          + 0.15 * timing
          + 0.10 * liquidity

"Success" represents realising a meaningful positive return.
"Moderate" represents breaking even or a marginal gain.
"Failure" represents significant capital loss.

Calibration favours realistic distributions: most speculative
investments do not produce outsized returns.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathsim.models import Factor, SimulationConfig


@dataclass
class InvestmentScenario:
    name: str = "investment"
    display_name: str = "Make an investment"
    description: str = (
        "Simulate the outcome of committing capital to a speculative asset."
    )

    def build_factors(self, config: SimulationConfig) -> list[Factor]:
        from pathsim.scoring import capital_to_factor, experience_to_float

        # Experience = research depth / due diligence quality
        research_mean = experience_to_float(config.experience)

        # Capital = position sizing appropriateness
        # Higher capital relative to reference = larger bet = more variance
        capital_raw = capital_to_factor(config.capital, reference=100_000.0)
        # Larger positions increase variance, not the mean
        position_size_std = 0.10 + 0.15 * capital_raw

        return [
            Factor(
                name="asset_quality",
                label="asset quality",
                distribution="beta",
                params={"alpha": 2.0, "beta": 3.0},
                weight=0.30,
            ),
            Factor(
                name="market_conditions",
                label="market conditions",
                distribution="beta",
                params={"alpha": 2.5, "beta": 2.5},  # symmetric uncertainty
                weight=0.25,
            ),
            Factor(
                name="position_sizing",
                label="position sizing",
                distribution="normal",
                params={"mean": research_mean, "std": position_size_std},
                weight=0.20,
            ),
            Factor(
                name="timing",
                label="entry timing",
                distribution="beta",
                params={"alpha": 2.0, "beta": 4.0},
                weight=0.15,
            ),
            Factor(
                name="liquidity",
                label="liquidity",
                distribution="beta",
                params={"alpha": 3.0, "beta": 2.0},
                weight=0.10,
            ),
        ]
