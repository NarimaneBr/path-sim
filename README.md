# PathSim

A local decision simulation engine. Given a decision — starting a company, switching careers, making an investment — PathSim runs thousands of probabilistic scenarios and reports outcome distributions.

Results are heuristic and exploratory. PathSim does not predict the future.

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/NarimaneBr/path-sim
cd path-sim
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

For optional local AI explanations (requires [Ollama](https://ollama.com)):

```bash
pip install -e ".[llm]"
```

---

## Usage

```bash
# Simulate a startup decision
pathsim "start a startup"

# Use a named scenario
pathsim startup
pathsim career-change
pathsim investment

# Tune parameters
pathsim startup --runs 20000 --capital 500000 --team 4 --experience high

# Save a chart
pathsim startup --chart

# Use a local LLM to explain results (requires Ollama)
pathsim startup --explain --model mistral
```

---

## Example

```
$ pathsim startup --runs 10000 --capital 200000 --team 2 --experience mid

  PathSim — Decision Simulation Engine
  Decision: start a startup

  Simulated futures (10,000 runs)

  ✓ Success           9.1%   ████
  ~ Moderate outcome  27.3%  ████████████
  ✗ Failure           63.6%  ████████████████████████████

  Most influential factors

  1. market_timing       sensitivity: 0.82
  2. founder_experience  sensitivity: 0.71
  3. capital             sensitivity: 0.43
  4. team_size           sensitivity: 0.31
  5. competition         sensitivity: 0.18

  Note: results are probabilistic estimates, not predictions.
```

---

## Design

PathSim uses Monte Carlo simulation: each run samples independent variables from calibrated probability distributions (normal, beta, uniform), feeds them through a weighted linear scoring model, and classifies the outcome. Running 10,000+ iterations produces stable outcome distributions.

### Scoring model (startup scenario)

```
score =
    0.30 × founder_skill
  + 0.25 × market_size
  + 0.20 × market_timing
  + 0.15 × capital
  + 0.10 × competition_inverse
```

Outcome thresholds:

| Score     | Classification     |
|-----------|--------------------|
| ≥ 0.65    | Success            |
| 0.35–0.65 | Moderate outcome   |
| < 0.35    | Failure            |

Sensitivity analysis uses Spearman rank correlation between each input variable and the final score across all runs.

---

## Limitations

- Scores are calibrated by hand, not fitted to real outcome data.
- Distributions are approximations based on general patterns, not domain research.
- The tool cannot account for individual skill, timing detail, or macro conditions.
- Local AI explanations reflect model interpretation, not domain expertise.

Use this tool to build intuition, not to make financial or career decisions.

---

## Project Structure

```
pathsim/
├── pathsim/
│   ├── cli.py              CLI entrypoint
│   ├── engine.py           Orchestration layer
│   ├── models.py           Data models (dataclasses)
│   ├── factors.py          Variable definitions
│   ├── scoring.py          Scoring functions
│   ├── simulation/
│   │   ├── monte_carlo.py  Simulation runner
│   │   ├── distributions.py Distribution samplers
│   │   └── outcome_model.py Outcome classification
│   ├── scenarios/
│   │   ├── startup.py
│   │   ├── career_change.py
│   │   └── investment.py
│   ├── analysis/
│   │   └── sensitivity.py  Sensitivity analysis
│   ├── llm/
│   │   └── explanation.py  Optional local LLM integration
│   └── visualization/
│       └── charts.py       Matplotlib chart generation
├── examples/
├── tests/
└── docs/
```

---

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=pathsim

# Lint
ruff check pathsim
```

To add a new scenario, subclass `BaseScenario` in `pathsim/scenarios/` and register it in `pathsim/engine.py`.

---

## Maintainer

Narimane Berradj — [@NarimaneBr](https://github.com/NarimaneBr)
