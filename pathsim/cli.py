"""
CLI — command-line interface for PathSim.

Entrypoint: pathsim (registered in pyproject.toml)

All output is routed through `rich` for consistent terminal formatting.
Progress indicators and tables use rich.progress and rich.table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box
from rich.text import Text
from rich.rule import Rule

from pathsim import __version__
from pathsim.models import SimulationConfig, SimulationResult

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pathsim",
        description=(
            "PathSim — a local decision simulation engine.\n"
            "Run probabilistic Monte Carlo simulations to explore outcome distributions.\n\n"
            "Results are heuristic estimates, not predictions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  pathsim "start a startup"\n'
            "  pathsim startup --runs 20000 --capital 200000 --team 3 --experience high\n"
            "  pathsim career-change --experience mid --capital 50000\n"
            "  pathsim investment --capital 100000 --experience high --chart\n"
            "  pathsim --list\n"
        ),
    )

    parser.add_argument(
        "decision",
        nargs="?",
        help="Decision to simulate, e.g. 'startup', 'career-change', 'investment'.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10_000,
        metavar="N",
        help="Number of Monte Carlo iterations (default: 10000).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="Available capital in currency units. Interpretation depends on scenario.",
    )
    parser.add_argument(
        "--team",
        type=int,
        default=None,
        metavar="N",
        help="Team size (startup scenario). Default: 2.",
    )
    parser.add_argument(
        "--experience",
        choices=["low", "mid", "high"],
        default=None,
        help="Experience level. Adjusts founder/skill distributions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Save a histogram chart of simulation scores.",
    )
    parser.add_argument(
        "--chart-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for the chart image (default: pathsim_<scenario>.png).",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Request an LLM explanation using a local runtime (requires Ollama).",
    )
    parser.add_argument(
        "--model",
        default="mistral",
        metavar="MODEL",
        help="Local model name for --explain (default: mistral).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available scenarios and exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"pathsim {__version__}",
    )

    return parser


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def _bar(fraction: float, width: int = 28) -> str:
    """Render a simple ASCII progress bar."""
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def render_result(result: SimulationResult) -> None:
    """Print a complete simulation result to the terminal."""
    outcomes = result.outcomes
    cfg = result.config

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]PathSim[/bold cyan] [dim]v{__version__}[/dim] — Decision Simulation Engine",
            border_style="cyan",
        )
    )
    console.print()
    console.print(f"  [bold]Decision:[/bold] {result.scenario_name}")
    console.print(f"  [dim]Runs: {cfg.runs:,}[/dim]")
    if cfg.experience:
        console.print(f"  [dim]Experience: {cfg.experience}[/dim]")
    if cfg.capital is not None:
        console.print(f"  [dim]Capital: {cfg.capital:,.0f}[/dim]")
    if cfg.team is not None:
        console.print(f"  [dim]Team size: {cfg.team}[/dim]")
    console.print()

    # Outcome table
    console.print(Rule("[bold]Simulated futures[/bold]", style="dim"))
    console.print()

    outcome_rows = [
        ("✓", "Success",          outcomes.success,  "green"),
        ("~", "Moderate outcome", outcomes.moderate, "yellow"),
        ("✗", "Failure",          outcomes.failure,  "red"),
    ]

    for icon, label, fraction, color in outcome_rows:
        pct = f"{fraction * 100:.1f}%"
        bar = _bar(fraction)
        line = (
            f"  [{color}]{icon}[/{color}] "
            f"[bold]{label:<18}[/bold] "
            f"[{color}]{pct:>6}[/{color}]  "
            f"[{color}]{bar}[/{color}]"
        )
        console.print(line)

    console.print()

    # Sensitivity table
    if result.sensitivity:
        console.print(Rule("[bold]Most influential factors[/bold]", style="dim"))
        console.print()
        for rank, s in enumerate(result.sensitivity, start=1):
            bar = _bar(s.correlation, width=20)
            console.print(
                f"  [bold cyan]{rank}.[/bold cyan] "
                f"[bold]{s.label:<22}[/bold] "
                f"[dim]sensitivity: {s.correlation:.2f}[/dim]  "
                f"[cyan]{bar}[/cyan]"
            )
        console.print()

    # Chart path
    if result.chart_path:
        console.print(f"  [green]Chart saved to:[/green] {result.chart_path}")
        console.print()

    # Explanation
    if result.explanation:
        console.print(Rule("[bold]LLM Explanation[/bold]", style="dim"))
        console.print()
        console.print(f"  {result.explanation}")
        console.print()

    # Disclaimer
    console.print(
        "  [dim italic]Results are probabilistic estimates based on heuristic "
        "calibration, not predictions.[/dim italic]"
    )
    console.print()


def render_scenario_list() -> None:
    from pathsim.engine import list_scenarios

    scenarios = list_scenarios()
    console.print()
    console.print(Panel.fit("[bold cyan]PathSim[/bold cyan] — Available Scenarios", border_style="cyan"))
    console.print()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Name", style="cyan bold", no_wrap=True)
    table.add_column("Description", style="white")

    for s in scenarios:
        table.add_row(s["name"], s["description"])

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        render_scenario_list()
        return 0

    if not args.decision:
        parser.print_help()
        return 1

    config = SimulationConfig(
        decision=args.decision,
        runs=args.runs,
        seed=args.seed,
        capital=args.capital,
        team=args.team,
        experience=args.experience,
    )

    # Run simulation with a progress spinner
    from pathsim.engine import simulate

    result: SimulationResult | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"  Running {config.runs:,} simulations…", total=None
        )
        try:
            result = simulate(config)
        except ValueError as exc:
            console.print(f"\n[red]Error:[/red] {exc}\n")
            return 1

    # Optional chart
    if args.chart:
        from pathsim.visualization.charts import save_score_histogram

        chart_path = args.chart_path or Path(f"pathsim_{result.scenario_name.replace(' ', '_').lower()}.png")
        save_score_histogram(result.scores, result.scenario_name, chart_path)
        result.chart_path = str(chart_path)

    # Optional LLM explanation
    if args.explain:
        from pathsim.llm.explanation import explain_result

        result.explanation = explain_result(result, model=args.model)

    render_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
