"""
Example: reproducible simulation with seed pinning.

Demonstrates that the same seed always produces the same result,
useful for sharing findings or embedding in documentation.

Run:
    python examples/reproducible_run.py
"""

from pathsim.engine import simulate
from pathsim.models import SimulationConfig


def run(seed: int) -> None:
    config = SimulationConfig(
        decision="investment",
        runs=10_000,
        seed=seed,
        capital=100_000,
        experience="mid",
    )
    result = simulate(config)
    print(
        f"  seed={seed}  "
        f"success={result.outcomes.success:.3f}  "
        f"moderate={result.outcomes.moderate:.3f}  "
        f"failure={result.outcomes.failure:.3f}"
    )


print("Reproducibility check — running with the same seed three times:\n")
for _ in range(3):
    run(seed=2024)

print("\nDifferent seeds produce different results:\n")
for seed in [1, 2, 3]:
    run(seed=seed)
