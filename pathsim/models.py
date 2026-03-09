"""
Data models used throughout PathSim.

All structures are plain dataclasses — no ORM, no validation framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutcomeCategory(str, Enum):
    SUCCESS = "success"
    MODERATE = "moderate"
    FAILURE = "failure"


@dataclass(frozen=True)
class Factor:
    """A single named input variable with its distribution parameters."""

    name: str
    label: str
    distribution: str  # "normal" | "beta" | "uniform"
    params: dict[str, float]
    weight: float  # contribution to the scoring model; weights should sum to 1.0

    def __post_init__(self) -> None:
        allowed = {"normal", "beta", "uniform"}
        if self.distribution not in allowed:
            raise ValueError(f"Unknown distribution '{self.distribution}'. Must be one of {allowed}.")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"Factor weight must be in [0, 1], got {self.weight}.")


@dataclass
class SimulationConfig:
    """Runtime parameters supplied by the user (CLI flags or API callers)."""

    decision: str
    runs: int = 10_000
    seed: int | None = None

    # Optional domain-specific overrides — scenarios interpret these freely
    capital: float | None = None
    team: int | None = None
    experience: str | None = None  # "low" | "mid" | "high"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError("runs must be >= 1.")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")


@dataclass
class OutcomeDistribution:
    """Aggregate outcome probabilities from a completed simulation."""

    success: float
    moderate: float
    failure: float

    def as_dict(self) -> dict[str, float]:
        return {
            OutcomeCategory.SUCCESS: self.success,
            OutcomeCategory.MODERATE: self.moderate,
            OutcomeCategory.FAILURE: self.failure,
        }


@dataclass
class SensitivityResult:
    """Spearman correlation of one factor against the final score."""

    factor_name: str
    label: str
    correlation: float  # absolute value; higher = more influential


@dataclass
class SimulationResult:
    """Complete output of a single simulation run."""

    config: SimulationConfig
    scores: list[float]  # raw scores for each Monte Carlo run
    outcomes: OutcomeDistribution
    sensitivity: list[SensitivityResult]
    scenario_name: str
    chart_path: str | None = None
    explanation: str | None = None
