"""
Optional local LLM explanation via Ollama.

This module will silently degrade when Ollama is not running.
It is never called unless the user explicitly passes --explain.

Supported runtimes:
  - Ollama (default, http://localhost:11434)

The prompt is intentionally minimal: it gives the model the outcome
numbers and asks for a plain-text analysis of the key risks.
"""

from __future__ import annotations

from pathsim.models import SimulationResult

_OLLAMA_URL = "http://localhost:11434/api/generate"

_PROMPT_TEMPLATE = """\
You are a decision analysis assistant.

A Monte Carlo simulation was run for the decision: "{decision}"

Results ({runs} runs):
- Success:          {success:.1%}
- Moderate outcome: {moderate:.1%}
- Failure:          {failure:.1%}

Most influential factors (by sensitivity):
{factors}

In 3-4 concise sentences, explain what these results suggest and which \
risks deserve the most attention. Do not add disclaimers or hedges. \
Be direct and specific.
"""


def _format_factors(sensitivity) -> str:
    lines = []
    for i, s in enumerate(sensitivity, start=1):
        lines.append(f"  {i}. {s.label} (sensitivity: {s.correlation:.2f})")
    return "\n".join(lines) if lines else "  (none)"


def _call_ollama(prompt: str, model: str, timeout: int = 60) -> str:
    """
    Send a generate request to the Ollama HTTP API.

    Returns the generated text or raises RuntimeError on failure.
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError(
            "httpx is required for LLM explanations.\n"
            "Install it with: pip install 'pathsim[llm]'"
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 256},
    }

    try:
        response = httpx.post(_OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except httpx.ConnectError:
        raise RuntimeError(
            "Cannot reach Ollama at localhost:11434.\n"
            "Start it with: ollama serve"
        )
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}.")
    except Exception as exc:
        raise RuntimeError(f"LLM call failed: {exc}")


def explain_result(result: SimulationResult, model: str = "mistral") -> str:
    """
    Generate a natural-language explanation of simulation results.

    Returns an explanation string, or an error message (never raises)
    so that the CLI can always display something.
    """
    prompt = _PROMPT_TEMPLATE.format(
        decision=result.config.decision,
        runs=result.config.runs,
        success=result.outcomes.success,
        moderate=result.outcomes.moderate,
        failure=result.outcomes.failure,
        factors=_format_factors(result.sensitivity),
    )

    try:
        return _call_ollama(prompt, model=model)
    except RuntimeError as exc:
        return f"[LLM unavailable] {exc}"
