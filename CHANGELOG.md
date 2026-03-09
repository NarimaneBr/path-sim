# Changelog

All notable changes to PathSim are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
PathSim uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.1.0] — 2026-03-09

### Added
- Monte Carlo simulation engine with vectorised NumPy implementation
- Three built-in scenarios: startup, career-change, investment
- Distribution support: normal, beta, uniform
- Sensitivity analysis via Spearman rank correlation (no scipy dependency)
- CLI with `rich` formatting, progress spinner, and outcome bars
- `--chart` flag for matplotlib score histograms
- `--explain` flag for optional Ollama LLM explanation
- `--seed` flag for reproducible results
- `--list` flag to enumerate available scenarios
- Natural-language decision matching (e.g. `"start a startup"` → startup scenario)
- Programmatic API (`pathsim.engine.simulate`)
- Full pytest test suite
- GitHub Pages documentation site
- MIT license
