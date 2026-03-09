# Contributing to PathSim

Thank you for taking the time to contribute.

## Development setup

```bash
git clone https://github.com/NarimaneBr/path-sim
cd path-sim
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
pytest --cov=pathsim   # with coverage
```

## Adding a scenario

1. Create `pathsim/scenarios/my_scenario.py` with a dataclass that implements `build_factors(config)`.
2. Register the scenario name and aliases in `pathsim/engine.py` (`_SCENARIO_ALIASES` and `_SCENARIO_CLASSES`).
3. Add at least one test in `tests/test_engine.py`.

## Code style

Run `ruff check pathsim` before submitting. Line length is capped at 100 characters.

## Reporting issues

Open an issue on [GitHub](https://github.com/NarimaneBr/path-sim/issues).
Include the command you ran, the full error output, and your Python version.
