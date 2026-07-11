# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git commits

**Never run `git commit` (or `git add` + `git commit`) for any task's work without Edgar's explicit confirmation first**, even if a prior task in this same plan was committed without asking. Implement and test the task, then stop and ask before committing — do not assume standing permission carries over from one task to the next.

## Commands

```bash
# Setup (editable install with dev deps)
.venv/bin/pip install -e ".[dev]"

# Run all tests
.venv/bin/pytest

# Run a single test file / test
.venv/bin/pytest tests/test_anthropic_client.py -v
.venv/bin/pytest tests/test_anthropic_client.py::test_complete_returns_completion_result -v
```

Note: an `rtk` shell hook intercepts and rewrites commands; it does not resolve `pytest` on PATH. Always invoke `.venv/bin/pytest` directly (or `.venv/bin/python3 -m pytest`) rather than bare `pytest`.

## Architecture

This is a PoC eval harness (Jira ticket LIT-39) that gates CI merges when a "prompt file" (any AI-behavior-defining file: a system prompt, a Claude Code skill, an agent instruction file) changes. It replays a sampled golden set through the old and new prompt versions, judges the difference semantically, and reports cost + a statistically-gated regression verdict before merge.

The full task-by-task implementation plan lives at `docs/superpowers/plans/2026-07-11-lit-39-eval-harness.md` — read it before adding new modules; it defines the exact interfaces each module must expose so later modules can consume them without redesign.

**Data flow:** `cli.py` orchestrates: `prompt_registry` resolves a changed prompt file to its golden set → `sampler` picks a subset → `replay` runs both prompt versions through `anthropic_client` → `embedding_filter` cheaply pre-filters obviously-identical pairs → `judge` scores the rest via LLM-as-judge → `cost_tracker` totals real token spend → `baseline_store` + `stats_gate` compare the run's scores against a rolling JSON-backed baseline via Welch's t-test → `report` renders the verdict as a markdown PR comment, and the GitHub Action fails the job on a real regression.

**Key invariant:** every module that would otherwise need a live Anthropic API key calls it only through `anthropic_client.AnthropicClient.complete(...)`, which returns a `CompletionResult` dataclass (`text`, `input_tokens`, `output_tokens`, `model`). Tests mock the `Anthropic` SDK client at that one seam — no other module talks to the SDK directly.

**Design constraints carried through every module** (see the plan's "Global Constraints" section for full rationale):
- No exact/string-match comparison in the judging path — string match may only short-circuit byte-identical outputs, never stand in for the semantic diff itself.
- Every cost figure is computed from real token usage returned by the API, never estimated or hardcoded.
- The statistical drift gate (Welch's t-test + effect-size floor vs. a rolling baseline) is explicitly documented in code/README as inspired by one non-adopted vendor blog post, not an industry standard — don't let it drift into being presented as a solved/standard technique.
- Where a component would need production hardening (auth, a persistent time-series DB, horizontal scaling) that isn't needed to prove the pattern, simplify and note it as a documented PoC limitation (e.g. `baseline_store.py` is a single JSON file) rather than building it out.
