"""
Example: programmatic usage of PathSim.

Run:
    python examples/basic_simulation.py
"""

from pathsim.engine import simulate
from pathsim.models import SimulationConfig

# --- Startup simulation with explicit overrides ---

config = SimulationConfig(
    decision="startup",
    runs=10_000,
    seed=42,
    capital=200_000,
    team=2,
    experience="mid",
)

result = simulate(config)

print(f"\nDecision: {result.scenario_name}")
print(f"Runs:     {result.config.runs:,}\n")

print("Simulated futures")
print(f"  Success:          {result.outcomes.success:.1%}")
print(f"  Moderate outcome: {result.outcomes.moderate:.1%}")
print(f"  Failure:          {result.outcomes.failure:.1%}")

print("\nMost influential factors")
for i, s in enumerate(result.sensitivity, start=1):
    print(f"  {i}. {s.label:<22} sensitivity: {s.correlation:.2f}")

# --- Career change simulation ---

print("\n" + "=" * 50)

career_config = SimulationConfig(
    decision="career-change",
    runs=10_000,
    seed=7,
    capital=50_000,
    experience="high",
)

career_result = simulate(career_config)

print(f"\nDecision: {career_result.scenario_name}")
print(f"Runs:     {career_result.config.runs:,}\n")
print("Simulated futures")
print(f"  Success:          {career_result.outcomes.success:.1%}")
print(f"  Moderate outcome: {career_result.outcomes.moderate:.1%}")
print(f"  Failure:          {career_result.outcomes.failure:.1%}")
