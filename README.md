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

## Monte Carlo Simulation

Monte Carlo simulation is the core computational technique behind PathSim. This section explains what it is, why it suits decision modeling, and how PathSim applies it.

### What Monte Carlo simulation is

Monte Carlo simulation estimates the behavior of a system by running it many times with randomly sampled inputs, then aggregating the results into a distribution of outcomes.

Rather than asking "what will happen?", it asks "across all plausible combinations of inputs, what range of outcomes emerges, and how often?" The answer is a probability distribution — not a single number.

The method works particularly well when a system involves several uncertain variables whose combined effect is hard to reason about analytically. By sampling those variables thousands of times, the simulation maps the outcome space without needing closed-form equations.

Three properties make it useful in practice:

- it handles non-linear relationships and arbitrary distributions
- it scales naturally with more variables
- it quantifies uncertainty explicitly, rather than hiding it inside a point estimate

### How PathSim uses it

Each simulation run executes the following steps:

```
for each run in 1..N:
    for each factor (founder_skill, market_size, ...):
        value = sample(factor.distribution, factor.params)

    score = sum(factor.weight * value for each factor)
    score = score / total_weight          # normalise to [0, 1]

    if score >= 0.65:   outcome = "success"
    elif score >= 0.35: outcome = "moderate"
    else:               outcome = "failure"

return distribution of outcomes across all N runs
```

PathSim runs this loop 10,000 times by default. Each iteration independently samples all variables, computes a weighted composite score, and classifies the outcome. After all runs complete, the fraction of runs in each category becomes the reported probability.

Sensitivity analysis runs after the simulation by computing the Spearman rank correlation between each factor's sample array and the final score array. Higher correlation means the factor drove outcomes more.

### Example simulation

Consider a single startup simulation run. Each variable is sampled independently:

```
founder_skill  ~ Normal(mean=0.50, std=0.20)  →  0.61
market_size    ~ Beta(alpha=2, beta=3)         →  0.38
market_timing  ~ Beta(alpha=2, beta=4)         →  0.22   ← unfavorable draw
capital        ~ Normal(mean=0.40, std=0.15)   →  0.44
competition    ~ Beta(alpha=2, beta=5)         →  0.19

score = (0.30 × 0.61) + (0.25 × 0.38) + (0.20 × 0.22)
      + (0.15 × 0.44) + (0.10 × 0.19)
      = 0.183 + 0.095 + 0.044 + 0.066 + 0.019
      = 0.407  →  moderate outcome
```

A different run might draw `market_timing = 0.71` and `founder_skill = 0.82`, pushing the score above 0.65 and producing a success. Over 10,000 runs, the distribution of these draws stabilises into the reported percentages.

### What the results mean

The output probabilities reflect the model, not the world. When PathSim reports "63% failure", it means: given the specified distributions and scoring weights, 63% of simulated runs produced a score below the failure threshold.

That number depends entirely on:

- the probability distributions assigned to each variable
- the weights in the scoring model
- the outcome thresholds

Change any of those, and the percentages change. The model is not fitted to historical outcome data. It is a structured exploration of a manually calibrated parameter space.

Use the results to identify which factors matter most and how sensitive the outcome is to assumptions — not as probability estimates of real-world events.

### Where Monte Carlo simulation is used

Monte Carlo methods appear across quantitative disciplines wherever uncertainty must be modeled explicitly:

- **Finance**: Value-at-Risk models and portfolio simulations use Monte Carlo to estimate loss distributions under varying market conditions. See [NIST/SEMATECH Handbook — Monte Carlo](https://www.itl.nist.gov/div898/handbook/eda/section3/eda366.htm).
- **Engineering reliability**: Structural and mechanical engineers simulate component failure rates under uncertain load conditions.
- **Physics and chemistry**: Particle transport simulations (including early nuclear weapons design) used Monte Carlo to model probabilistic collision chains.
- **Aerospace**: NASA uses Monte Carlo analysis in trajectory planning and uncertainty quantification for mission design. See [NASA Technical Reports Server](https://ntrs.nasa.gov).
- **Quantitative finance research**: Option pricing (e.g. via the Black-Scholes framework extended to path-dependent options) routinely uses Monte Carlo when closed-form solutions do not exist.

The [NIST Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/) documents Monte Carlo as a standard tool in uncertainty analysis. IBM's technical literature also covers it extensively in the context of risk simulation.

### Historical context

Monte Carlo methods were formalised in the 1940s at Los Alamos National Laboratory by **Stanislaw Ulam** and **John von Neumann**, initially to solve neutron diffusion problems in nuclear research. Ulam reportedly conceived the approach while playing solitaire and thinking about the probability of a particular card layout succeeding.

The name "Monte Carlo" was coined in reference to the Monaco casino — an acknowledgment that the method relies on randomness, much like games of chance. Nicholas Metropolis, who coined the name and worked with Ulam and von Neumann, described the method in a 1949 paper that remains a foundational reference in scientific computing.

Since then, Monte Carlo simulation has become a standard tool in computational science, enabled by the availability of fast pseudo-random number generators and cheap compute.

### Strengths of the method

Monte Carlo is well-suited to decision modeling because:

- it handles many interacting uncertain variables without requiring independence assumptions to be analytically tractable
- it requires no closed-form solution — only the ability to sample from distributions and evaluate a scoring function
- it produces a full outcome distribution, not just a mean, making it possible to reason about tail risks
- it identifies which variables matter most (via sensitivity analysis) rather than treating all inputs as equally important
- results converge reliably as run count increases; 10,000 iterations is generally sufficient for stable percentages at two significant figures

### Limitations of Monte Carlo methods

The output is only as good as the model. Specific failure modes to be aware of:

- **Miscalibrated distributions**: if the Beta or Normal parameters don't reflect real-world behavior, the output percentages are systematically biased. PathSim's parameters are hand-tuned, not empirically derived.
- **Independence assumption**: PathSim samples each factor independently. In practice, variables like market timing and capital availability are correlated. Ignoring that correlation underestimates joint risk.
- **Model structure**: a weighted linear combination is interpretable but oversimplified. Real outcomes involve non-linearities, thresholds, and interaction effects that a linear model cannot capture.
- **Threshold sensitivity**: changing the success threshold from 0.65 to 0.60 can move several percentage points of runs from "moderate" to "success". The thresholds are somewhat arbitrary.

These limitations apply to Monte Carlo simulation generally. They are not specific to PathSim, but they are more pronounced here because the model is not calibrated against real outcome data.

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
