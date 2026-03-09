"""
factors.py — shared factor definitions.

Scenarios import from here when they need common factors (e.g.
competition, market timing) rather than re-defining them inline.
This avoids duplication and keeps distribution calibration in one place.
"""

from __future__ import annotations

from pathsim.models import Factor

# ---------------------------------------------------------------------------
# Reusable factor templates
# ---------------------------------------------------------------------------

MARKET_TIMING = Factor(
    name="market_timing",
    label="market timing",
    distribution="beta",
    params={"alpha": 2.0, "beta": 3.0},  # skewed toward poor timing
    weight=0.20,
)

COMPETITION_INVERSE = Factor(
    name="competition_inverse",
    label="competition",
    distribution="beta",
    params={"alpha": 2.0, "beta": 4.0},  # competitive markets are common
    weight=0.10,
)

MACRO_ENVIRONMENT = Factor(
    name="macro_environment",
    label="macro environment",
    distribution="beta",
    params={"alpha": 3.0, "beta": 3.0},  # symmetric uncertainty
    weight=0.10,
)
