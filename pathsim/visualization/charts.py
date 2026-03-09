"""
Matplotlib chart generation.

Charts are optional.  This module is only imported when --chart is
passed so that matplotlib is never required for the core simulation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_score_histogram(
    scores: list[float],
    scenario_name: str,
    output_path: Path,
    bins: int = 60,
    dpi: int = 150,
) -> None:
    """
    Save a histogram of simulation scores to `output_path`.

    The vertical lines mark the success and failure thresholds.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe on all platforms
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from pathsim.simulation.outcome_model import SUCCESS_THRESHOLD, FAILURE_THRESHOLD

    arr = np.asarray(scores, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    # Colour-code bars by outcome region
    n, bin_edges = np.histogram(arr, bins=bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    colours = []
    for center in bin_centers:
        if center >= SUCCESS_THRESHOLD:
            colours.append("#2ea043")   # green
        elif center >= FAILURE_THRESHOLD:
            colours.append("#e3b341")   # yellow
        else:
            colours.append("#f85149")   # red

    ax.bar(bin_edges[:-1], n, width=np.diff(bin_edges), color=colours,
           align="edge", edgecolor="none", alpha=0.85)

    # Threshold lines
    ax.axvline(SUCCESS_THRESHOLD, color="#2ea043", linestyle="--", linewidth=1.2,
               label=f"Success threshold ({SUCCESS_THRESHOLD})")
    ax.axvline(FAILURE_THRESHOLD, color="#f85149", linestyle="--", linewidth=1.2,
               label=f"Failure threshold ({FAILURE_THRESHOLD})")

    ax.set_title(
        f"PathSim — {scenario_name}\nScore distribution ({len(arr):,} runs)",
        color="white", fontsize=13, pad=14,
    )
    ax.set_xlabel("Composite score", color="#8b949e", fontsize=11)
    ax.set_ylabel("Frequency", color="#8b949e", fontsize=11)
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    success_patch = mpatches.Patch(color="#2ea043", label="Success")
    moderate_patch = mpatches.Patch(color="#e3b341", label="Moderate")
    failure_patch = mpatches.Patch(color="#f85149", label="Failure")
    legend = ax.legend(
        handles=[success_patch, moderate_patch, failure_patch],
        loc="upper right",
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
    )

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
