"""
Engine — orchestration layer.

Ties together scenario resolution, simulation, outcome classification,
and sensitivity analysis.  The CLI and any programmatic callers go
through this module; they should not call the sub-modules directly.
"""

from __future__ import annotations

from pathsim.analysis.sensitivity import compute_sensitivity
from pathsim.models import SimulationConfig, SimulationResult
from pathsim.simulation.monte_carlo import run_simulation
from pathsim.simulation.outcome_model import compute_outcome_distribution

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------
# Maps both formal names ("startup") and common natural-language fragments
# ("start a startup", "launch a company") to the scenario class.
# Keys are lowercase and whitespace-stripped for fuzzy matching.

_SCENARIO_ALIASES: dict[str, str] = {
    # startup
    "startup": "startup",
    "start a startup": "startup",
    "launch a startup": "startup",
    "start a company": "startup",
    "launch a company": "startup",
    "found a startup": "startup",
    # career-change
    "career-change": "career-change",
    "career change": "career-change",
    "change career": "career-change",
    "switch careers": "career-change",
    "new job": "career-change",
    "change jobs": "career-change",
    # investment
    "investment": "investment",
    "invest": "investment",
    "make an investment": "investment",
    "invest money": "investment",
}


def _resolve_scenario(decision: str):
    """
    Return the appropriate scenario object for a decision string.

    Tries exact match first, then substring match.  Raises ValueError
    if the decision cannot be mapped to any known scenario.
    """
    from pathsim.scenarios.career_change import CareerChangeScenario
    from pathsim.scenarios.investment import InvestmentScenario
    from pathsim.scenarios.startup import StartupScenario

    scenario_classes = {
        "startup": StartupScenario,
        "career-change": CareerChangeScenario,
        "investment": InvestmentScenario,
    }

    key = decision.strip().lower()

    # Exact match
    if key in _SCENARIO_ALIASES:
        return scenario_classes[_SCENARIO_ALIASES[key]]()

    # Substring match — picks first alias that appears in the decision string
    for alias, canonical in _SCENARIO_ALIASES.items():
        if alias in key:
            return scenario_classes[canonical]()

    available = sorted({v for v in _SCENARIO_ALIASES.values()})
    raise ValueError(
        f"Cannot map '{decision}' to a known scenario.\n"
        f"Available scenarios: {', '.join(available)}\n"
        f"Try: pathsim startup | pathsim career-change | pathsim investment"
    )


def simulate(config: SimulationConfig) -> SimulationResult:
    """
    Run a full simulation for the given configuration.

    1. Resolve the decision to a scenario.
    2. Build factors (possibly adjusted by CLI overrides).
    3. Run Monte Carlo simulation.
    4. Classify outcomes.
    5. Compute sensitivity.
    6. Return a SimulationResult.
    """
    scenario = _resolve_scenario(config.decision)
    factors = scenario.build_factors(config)

    scores, samples = run_simulation(
        factors=factors,
        runs=config.runs,
        seed=config.seed,
    )

    outcomes = compute_outcome_distribution(scores)
    sensitivity = compute_sensitivity(factors, samples, scores, top_n=5)

    return SimulationResult(
        config=config,
        scores=scores.tolist(),
        outcomes=outcomes,
        sensitivity=sensitivity,
        scenario_name=scenario.display_name,
    )


def list_scenarios() -> list[dict[str, str]]:
    """Return metadata for all registered scenarios."""
    from pathsim.scenarios.career_change import CareerChangeScenario
    from pathsim.scenarios.investment import InvestmentScenario
    from pathsim.scenarios.startup import StartupScenario

    return [
        {"name": s.name, "display": s.display_name, "description": s.description}
        for s in (StartupScenario(), CareerChangeScenario(), InvestmentScenario())
    ]
