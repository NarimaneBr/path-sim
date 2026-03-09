"""
Example: chart generation.

Runs a startup simulation and saves a score histogram to disk.

Run:
    python examples/generate_chart.py
"""

from pathlib import Path

from pathsim.engine import simulate
from pathsim.models import SimulationConfig
from pathsim.visualization.charts import save_score_histogram

config = SimulationConfig(
    decision="startup",
    runs=20_000,
    seed=0,
    capital=300_000,
    team=3,
    experience="mid",
)

result = simulate(config)
output = Path("pathsim_startup_example.png")
save_score_histogram(result.scores, result.scenario_name, output)

print(f"Chart saved to: {output.resolve()}")
