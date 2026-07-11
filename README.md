# eval-harness (LIT-39)

PoC: a CLI + GitHub Action that gates PR merges when a "prompt file" (a system prompt, a Claude Code skill, an agent instruction file — anything that defines AI behavior) changes. It replays a sampled golden set through the old and new prompt versions, judges the difference semantically (not string match), reports a real dollar cost per PR, and gates on statistical significance against a rolling baseline instead of a bare threshold.

Tracked in the Labs Initiatives Tracker as LIT-39, under epic "Factory Backbone" (LIT-48).

## Why

A prompt change is a behavior change. Without replay against a golden set, every prompt edit is a silent production change you only catch when a user complains. This harness samples golden-set examples (it does not require running on every commit) and reports the delta before merge.

## Status

Early PoC, under active implementation. See `docs/superpowers/plans/2026-07-11-lit-39-eval-harness.md` for the full task-by-task plan.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Running tests

```bash
.venv/bin/pytest
```

Run a single test file or test:

```bash
.venv/bin/pytest tests/test_anthropic_client.py -v
.venv/bin/pytest tests/test_anthropic_client.py::test_complete_returns_completion_result -v
```

## Architecture

A Python package (`eval_harness`) of independently testable modules, each swappable in tests without a live Anthropic API key:

- `anthropic_client.py` — thin wrapper around the Anthropic SDK; every other module talks to the API only through this, so tests substitute a fake.
- `golden_set.py` — golden-set YAML schema, loader, validator.
- `prompt_registry.py` — maps a prompt file path to its golden-set file + config.
- `replay.py` — runs golden-set inputs through a given prompt version.
- `embedding_filter.py` — local `sentence-transformers` cosine-similarity pre-filter (free), used only to cut judge calls, never as the source of truth for the semantic diff.
- `judge.py` — LLM-as-judge scorer (pointwise 0-5 + reason), reference-anchored against the golden set's expected answer.
- `sampler.py` — deterministic sampling of a golden set (eval runs on a sample, not every commit).
- `cost_tracker.py` — token usage → USD cost accounting, computed from real API usage, never estimated.
- `baseline_store.py` — JSON-backed rolling baseline of past run scores (PoC-scale; a real deployment would use a time-series DB).
- `stats_gate.py` — Welch's t-test + effect-size-floor regression decision against the rolling baseline.
- `report.py` — assembles a run's results into a markdown PR report.
- `cli.py` — `eval-harness run` entrypoint orchestrating all of the above.

A GitHub Action (`.github/workflows/eval-gate.yml`) wraps the CLI, posts the report as a PR comment, and fails the job (blocking merge) on a real regression.

Two golden sets exercise the system: one dogfooding our own `jira-ticket-kickoff` skill (a real artifact), and a small purpose-built demo agent with a deliberately regressed prompt version for a clean scripted "regression caught live" demo.

## Design notes

- No exact/string-match comparison in the judging path — the diff is semantic. String match may only short-circuit byte-identical outputs (an obvious non-regression, not a judgment).
- The statistical drift gate (Welch's t-test + effect-size floor vs. a rolling baseline) is inspired by a single non-adopted vendor blog post (FutureAGI), not an industry standard — it is the PoC's genuine novelty, and this README calls that out explicitly rather than overclaiming.
- The judge is calibrated against a ~25-example labeled set (target >90% agreement) before being trusted in CI.
