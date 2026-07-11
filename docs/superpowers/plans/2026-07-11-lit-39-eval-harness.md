# LIT-39 Eval Harness with Golden Set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI + GitHub Action that gates PR merges when a "prompt file" (any AI-behavior-defining file — a system prompt, a Claude Code skill, an agent instruction file) changes: it replays a sampled golden set through the old and new prompt versions, judges the difference semantically (not string match), reports a real dollar cost per PR, and — as the PoC's genuine novelty — gates on statistical significance against a rolling baseline instead of a bare threshold.

**Architecture:** A Python package (`eval_harness`) with independently testable modules: golden-set storage/validation, a replay engine that calls the Anthropic API with old vs. new prompt text, a local-embedding cheap pre-filter, an LLM-as-judge scorer (calibrated against labeled examples before trust), a sampler, a cost tracker, and a statistical drift gate comparing against a JSON-backed rolling baseline. A CLI (`eval-harness run`) orchestrates all of it into one report; a GitHub Action wraps the CLI, posts the report as a PR comment, and fails the job (blocking merge) on a real regression. Two golden sets exercise the system: a dogfooding one on our own `jira-ticket-kickoff` SKILL.md, and a small purpose-built demo agent with a deliberately injected regression for a clean scripted demo.

**Tech Stack:** Python 3.11+, `anthropic` SDK (replay + judge calls), `sentence-transformers` (local, free embedding pre-filter — no API cost), `scipy.stats` (Welch's t-test for the statistical gate), `pyyaml` (golden-set files), `pytest`, GitHub Actions.

## Global Constraints

- Every module must be independently unit-testable without a live Anthropic API key — network calls go through a thin client wrapper that tests replace with a fake.
- No exact/string-match comparison anywhere in the judging path — AC#2 requires semantic diff; string match may only appear as an internal short-circuit for byte-identical outputs (an obvious non-regression, not a judgment).
- Every dollar figure surfaced to the user (cost-per-PR) must be computed from actual token usage returned by the API, not estimated/hardcoded — AC#3 requires this to be real.
- The statistical gate (Welch's t-test + effect-size floor) is a PoC differentiator inspired by a single non-adopted vendor blog post (FutureAGI) — code comments and README must say so plainly, not present it as an industry standard.
- Follow the LIT-38 PoC convention: where a component would need production hardening (auth, persistent DB, horizontal scaling) that isn't needed to prove the pattern, hardcode/simplify and note it as a documented PoC limitation rather than building it.

---

## File Structure

```
eval_harness/
  pyproject.toml
  README.md
  src/eval_harness/
    __init__.py
    anthropic_client.py    # thin wrapper around the Anthropic SDK — swappable for tests
    golden_set.py          # golden-set YAML schema, loader, validator
    prompt_registry.py     # maps a prompt file path to its golden-set file + config
    replay.py              # runs golden-set inputs through a given prompt version
    embedding_filter.py     # local sentence-transformer cosine-similarity pre-filter
    judge.py               # LLM-as-judge scorer (pointwise 0-5 + reason)
    sampler.py             # deterministic sampling of a golden set
    cost_tracker.py        # token usage -> USD cost accounting
    baseline_store.py      # JSON-backed rolling baseline of past run scores
    stats_gate.py          # Welch's t-test + effect-size-floor regression decision
    report.py              # assembles a run's results into a markdown PR report
    cli.py                 # `eval-harness run` entrypoint
  golden_sets/
    jira-ticket-kickoff.yaml     # dogfood golden set (real artifact)
    faq-demo-agent.yaml          # purpose-built demo golden set
  demo_agent/
    faq_agent_prompt.md          # v1: the "good" prompt version for the demo
    faq_agent_prompt_regressed.md # v2: deliberately regressed version for the scripted demo
  prompt_registry.yaml     # top-level config: which prompt files map to which golden sets
  calibration/
    judge_calibration_set.yaml   # ~25 labeled (input, output_a, output_b, human_verdict) examples
  scripts/
    calibrate_judge.py     # runs the judge against the calibration set, reports agreement %
  .github/workflows/
    eval-gate.yml           # the CI merge-gate action
  tests/
    test_golden_set.py
    test_prompt_registry.py
    test_replay.py
    test_embedding_filter.py
    test_judge.py
    test_sampler.py
    test_cost_tracker.py
    test_baseline_store.py
    test_stats_gate.py
    test_report.py
    test_cli.py
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/eval_harness/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: an installable package `eval_harness` importable from `tests/`, with `pytest` runnable from repo root.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "eval-harness"
version = "0.1.0"
description = "PoC: eval harness gating CI merges on prompt-file regressions (LIT-39)"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
    "scipy>=1.13",
    "sentence-transformers>=3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.14"]

[project.scripts]
eval-harness = "eval_harness.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init files**

```python
# src/eval_harness/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/
baseline_store.json
```

- [ ] **Step 4: Init git repo and install package**

Run: `cd /Users/edgar.hernandez/Desktop/eval_harness && git init && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
Expected: package installs cleanly, `pytest` command available.

- [ ] **Step 5: Verify pytest runs (with zero tests)**

Run: `pytest -v`
Expected: `collected 0 items` — no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/eval_harness/__init__.py tests/__init__.py .gitignore
git commit -m "chore: scaffold eval-harness Python package"
```

---

### Task 2: Anthropic Client Wrapper

**Files:**
- Create: `src/eval_harness/anthropic_client.py`
- Test: `tests/test_anthropic_client.py`

**Interfaces:**
- Produces: `class AnthropicClient` with method `complete(system_prompt: str, user_input: str, model: str = "claude-sonnet-5") -> CompletionResult`, where `CompletionResult` is a dataclass `{text: str, input_tokens: int, output_tokens: int, model: str}`. Later tasks (replay, judge) depend on this exact shape — they never call the SDK directly, only this wrapper, so tests can substitute a fake.

- [x] **Step 1: Write the failing test**

```python
# tests/test_anthropic_client.py
from unittest.mock import MagicMock, patch
from eval_harness.anthropic_client import AnthropicClient, CompletionResult


def test_complete_returns_completion_result():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="hello world")]
    fake_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    with patch("eval_harness.anthropic_client.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.create.return_value = fake_response
        client = AnthropicClient(api_key="fake-key")
        result = client.complete(system_prompt="You are helpful.", user_input="Hi")

    assert isinstance(result, CompletionResult)
    assert result.text == "hello world"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.model == "claude-sonnet-5"
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.anthropic_client'`

- [x] **Step 3: Write minimal implementation**

```python
# src/eval_harness/anthropic_client.py
from dataclasses import dataclass
from anthropic import Anthropic


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class AnthropicClient:
    def __init__(self, api_key: str):
        self._client = Anthropic(api_key=api_key)

    def complete(
        self,
        system_prompt: str,
        user_input: str,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
    ) -> CompletionResult:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_input}],
        )
        return CompletionResult(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_anthropic_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_harness/anthropic_client.py tests/test_anthropic_client.py
git commit -m "feat: add Anthropic client wrapper with token usage tracking"
```

---

### Task 3: Golden-Set Schema, Loader, and Validator

**Files:**
- Create: `src/eval_harness/golden_set.py`
- Test: `tests/test_golden_set.py`

**Interfaces:**
- Produces: `@dataclass GoldenExample {id: str, input: str, expected: str}`, `@dataclass GoldenSet {name: str, examples: list[GoldenExample]}`, `load_golden_set(path: str) -> GoldenSet`. Raises `GoldenSetError` (a plain `Exception` subclass) on missing/malformed fields. Later tasks (`replay`, `sampler`) consume `GoldenSet.examples`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_golden_set.py
import pytest
from eval_harness.golden_set import load_golden_set, GoldenSetError

VALID_YAML = """
name: test-agent
examples:
  - id: ex1
    input: "What is 2+2?"
    expected: "4"
  - id: ex2
    input: "Capital of France?"
    expected: "Paris"
"""

MISSING_FIELD_YAML = """
name: test-agent
examples:
  - id: ex1
    input: "What is 2+2?"
"""


def test_load_valid_golden_set(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text(VALID_YAML)

    golden_set = load_golden_set(str(p))

    assert golden_set.name == "test-agent"
    assert len(golden_set.examples) == 2
    assert golden_set.examples[0].id == "ex1"
    assert golden_set.examples[0].input == "What is 2+2?"
    assert golden_set.examples[0].expected == "4"


def test_load_rejects_missing_expected_field(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text(MISSING_FIELD_YAML)

    with pytest.raises(GoldenSetError, match="expected"):
        load_golden_set(str(p))


def test_load_rejects_empty_examples(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text("name: empty-agent\nexamples: []\n")

    with pytest.raises(GoldenSetError, match="at least one example"):
        load_golden_set(str(p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_golden_set.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.golden_set'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/golden_set.py
from dataclasses import dataclass

import yaml


class GoldenSetError(Exception):
    pass


@dataclass
class GoldenExample:
    id: str
    input: str
    expected: str


@dataclass
class GoldenSet:
    name: str
    examples: list[GoldenExample]


REQUIRED_FIELDS = ("id", "input", "expected")


def load_golden_set(path: str) -> GoldenSet:
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "name" not in raw:
        raise GoldenSetError(f"{path}: missing top-level 'name' field")

    raw_examples = raw.get("examples") or []
    if len(raw_examples) == 0:
        raise GoldenSetError(f"{path}: golden set must contain at least one example")

    examples = []
    for i, ex in enumerate(raw_examples):
        for field in REQUIRED_FIELDS:
            if field not in ex:
                raise GoldenSetError(
                    f"{path}: example at index {i} is missing required field '{field}'"
                )
        examples.append(GoldenExample(id=ex["id"], input=ex["input"], expected=ex["expected"]))

    return GoldenSet(name=raw["name"], examples=examples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_golden_set.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/eval_harness/golden_set.py tests/test_golden_set.py
git commit -m "feat: add golden-set YAML schema, loader, and validator"
```

---

### Task 4: Replay Engine

**Files:**
- Create: `src/eval_harness/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `AnthropicClient.complete(system_prompt, user_input) -> CompletionResult` (Task 2), `GoldenSet`/`GoldenExample` (Task 3).
- Produces: `@dataclass ReplayResult {example_id: str, output: str, input_tokens: int, output_tokens: int}`, `replay_prompt(client: AnthropicClient, prompt_text: str, examples: list[GoldenExample], model: str = "claude-sonnet-5") -> list[ReplayResult]`. Later tasks (judge, cost_tracker) consume `list[ReplayResult]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_replay.py
from unittest.mock import MagicMock
from eval_harness.replay import replay_prompt, ReplayResult
from eval_harness.golden_set import GoldenExample
from eval_harness.anthropic_client import CompletionResult


def test_replay_prompt_runs_every_example_through_client():
    fake_client = MagicMock()
    fake_client.complete.side_effect = [
        CompletionResult(text="4", input_tokens=5, output_tokens=1, model="claude-sonnet-5"),
        CompletionResult(text="Paris", input_tokens=6, output_tokens=1, model="claude-sonnet-5"),
    ]
    examples = [
        GoldenExample(id="ex1", input="What is 2+2?", expected="4"),
        GoldenExample(id="ex2", input="Capital of France?", expected="Paris"),
    ]

    results = replay_prompt(fake_client, "You are a helpful assistant.", examples)

    assert results == [
        ReplayResult(example_id="ex1", output="4", input_tokens=5, output_tokens=1),
        ReplayResult(example_id="ex2", output="Paris", input_tokens=6, output_tokens=1),
    ]
    assert fake_client.complete.call_count == 2
    fake_client.complete.assert_any_call(
        system_prompt="You are a helpful assistant.",
        user_input="What is 2+2?",
        model="claude-sonnet-5",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.replay'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/replay.py
from dataclasses import dataclass

from eval_harness.anthropic_client import AnthropicClient
from eval_harness.golden_set import GoldenExample


@dataclass
class ReplayResult:
    example_id: str
    output: str
    input_tokens: int
    output_tokens: int


def replay_prompt(
    client: AnthropicClient,
    prompt_text: str,
    examples: list[GoldenExample],
    model: str = "claude-sonnet-5",
) -> list[ReplayResult]:
    results = []
    for example in examples:
        completion = client.complete(
            system_prompt=prompt_text,
            user_input=example.input,
            model=model,
        )
        results.append(
            ReplayResult(
                example_id=example.id,
                output=completion.text,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )
        )
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_harness/replay.py tests/test_replay.py
git commit -m "feat: add replay engine that runs golden-set inputs through a prompt version"
```

---

### Task 5: Embedding Pre-Filter

**Files:**
- Create: `src/eval_harness/embedding_filter.py`
- Test: `tests/test_embedding_filter.py`

**Interfaces:**
- Consumes: pairs of `(old_output: str, new_output: str)` strings (from two `ReplayResult` lists, matched by `example_id` — matching happens in the CLI orchestrator, Task 11).
- Produces: `class EmbeddingFilter` with `is_likely_unchanged(old_output: str, new_output: str, threshold: float = 0.97) -> bool`. This runs **locally** (no API cost) using `sentence-transformers`, and is used as a cheap gate before spending money on the LLM judge — directly serves AC#3's cost-control goal per the research's AC#2/AC#3 implication. Later tasks (judge orchestration in `cli.py`) call this first and only invoke the paid judge when it returns `False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding_filter.py
from unittest.mock import MagicMock, patch
import numpy as np
from eval_harness.embedding_filter import EmbeddingFilter


def test_identical_text_is_likely_unchanged():
    with patch("eval_harness.embedding_filter.SentenceTransformer") as MockModel:
        # Same text -> same embedding -> cosine similarity 1.0
        MockModel.return_value.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
        ef = EmbeddingFilter()
        assert ef.is_likely_unchanged("The sky is blue.", "The sky is blue.") is True


def test_very_different_text_is_not_likely_unchanged():
    with patch("eval_harness.embedding_filter.SentenceTransformer") as MockModel:
        # Orthogonal embeddings -> cosine similarity 0.0
        MockModel.return_value.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        ef = EmbeddingFilter()
        assert ef.is_likely_unchanged("The sky is blue.", "I like pizza.") is False


def test_threshold_is_configurable():
    with patch("eval_harness.embedding_filter.SentenceTransformer") as MockModel:
        # Cosine similarity ~0.95, below a strict 0.99 threshold
        MockModel.return_value.encode.return_value = np.array([[1.0, 0.0], [0.95, 0.312]])
        ef = EmbeddingFilter()
        assert ef.is_likely_unchanged("a", "b", threshold=0.99) is False
        assert ef.is_likely_unchanged("a", "b", threshold=0.90) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedding_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.embedding_filter'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/embedding_filter.py
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingFilter:
    """Cheap, local (no API cost) pre-filter: skips the paid LLM judge when two
    outputs are near-identical in embedding space. Per LIT-39 research finding #3,
    embedding similarity misses meaning nuance (e.g. reordered words) — it is
    ONLY used here as a cost-saving pre-filter, never as the source of truth for
    the semantic diff itself (that's the LLM judge, see judge.py)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def is_likely_unchanged(self, old_output: str, new_output: str, threshold: float = 0.97) -> bool:
        embeddings = self._model.encode([old_output, new_output])
        similarity = self._cosine_similarity(embeddings[0], embeddings[1])
        return similarity >= threshold

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_embedding_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_harness/embedding_filter.py tests/test_embedding_filter.py
git commit -m "feat: add local embedding pre-filter to reduce paid judge calls"
```

---

### Task 6: LLM-as-Judge Scorer

**Files:**
- Create: `src/eval_harness/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `AnthropicClient.complete(...)` (Task 2).
- Produces: `@dataclass JudgeVerdict {score: float, reasoning: str, input_tokens: int, output_tokens: int}` (score is 0.0-1.0, normalized from a 0-5 rubric), `judge_pair(client: AnthropicClient, task_input: str, expected: str, old_output: str, new_output: str, model: str = "claude-opus-4-8") -> JudgeVerdict`. Later tasks (`stats_gate`, `report`) consume `JudgeVerdict.score`; `cost_tracker` consumes its token fields.

This is AC#2's core: the judge compares **old vs. new output against the same reference (`expected`)**, not old-vs-new directly — a reference-anchored comparison is more calibratable than open-ended pairwise preference (per the research's 3-canonical-patterns finding), and matches promptfoo's `llm-rubric` `pass/score/reason` triple shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_judge.py
import json
from unittest.mock import MagicMock
from eval_harness.judge import judge_pair, JudgeVerdict
from eval_harness.anthropic_client import CompletionResult


def test_judge_pair_parses_verdict_json():
    fake_client = MagicMock()
    fake_client.complete.return_value = CompletionResult(
        text=json.dumps({"score": 4, "reasoning": "New output is equally correct and clearer."}),
        input_tokens=120,
        output_tokens=25,
        model="claude-opus-4-8",
    )

    verdict = judge_pair(
        fake_client,
        task_input="What is 2+2?",
        expected="4",
        old_output="The answer is 4.",
        new_output="4.",
    )

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.score == 0.8  # 4/5 normalized to 0-1
    assert "clearer" in verdict.reasoning
    assert verdict.input_tokens == 120
    assert verdict.output_tokens == 25


def test_judge_pair_handles_markdown_fenced_json():
    # LIT-38's article war story: models sometimes wrap JSON in ```json fences
    fake_client = MagicMock()
    fake_client.complete.return_value = CompletionResult(
        text='```json\n{"score": 2, "reasoning": "New output drops a required caveat."}\n```',
        input_tokens=100,
        output_tokens=20,
        model="claude-opus-4-8",
    )

    verdict = judge_pair(fake_client, "q", "expected", "old", "new")

    assert verdict.score == 0.4
    assert "caveat" in verdict.reasoning
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.judge'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/judge.py
import json
import re
from dataclasses import dataclass

from eval_harness.anthropic_client import AnthropicClient

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator comparing two AI-generated \
responses to the same task, against a reference answer. Score how well the NEW \
response performs relative to the OLD response, on a 0-5 scale:

5 = new response is clearly better or equally correct and complete
3 = new response is roughly equivalent, minor differences that don't affect correctness
1 = new response is noticeably worse (missing info, less accurate, less helpful)
0 = new response is a severe regression (wrong, harmful, or nonsensical)

Respond with ONLY a JSON object: {"score": <int 0-5>, "reasoning": "<one sentence>"}"""


@dataclass
class JudgeVerdict:
    score: float
    reasoning: str
    input_tokens: int
    output_tokens: int


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    return json.loads(stripped)


def judge_pair(
    client: AnthropicClient,
    task_input: str,
    expected: str,
    old_output: str,
    new_output: str,
    model: str = "claude-opus-4-8",
) -> JudgeVerdict:
    user_input = (
        f"TASK INPUT:\n{task_input}\n\n"
        f"REFERENCE ANSWER:\n{expected}\n\n"
        f"OLD RESPONSE:\n{old_output}\n\n"
        f"NEW RESPONSE:\n{new_output}"
    )
    completion = client.complete(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_input=user_input,
        model=model,
    )
    parsed = _extract_json(completion.text)
    return JudgeVerdict(
        score=parsed["score"] / 5.0,
        reasoning=parsed["reasoning"],
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_judge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval_harness/judge.py tests/test_judge.py
git commit -m "feat: add LLM-as-judge pairwise scorer (AC#2 semantic diff, not string match)"
```

---

### Task 7: Judge Calibration Set + Calibration Script

**Files:**
- Create: `calibration/judge_calibration_set.yaml`
- Create: `scripts/calibrate_judge.py`
- Test: `tests/test_calibrate_judge.py`

**Interfaces:**
- Consumes: `judge_pair` (Task 6).
- Produces: `run_calibration(client, calibration_examples: list[dict]) -> CalibrationReport` where `CalibrationReport` is `@dataclass {total: int, agreements: int, agreement_rate: float, disagreements: list[dict]}`. This operationalizes the research's judge-calibration guidance (calibrate against 20-30 labeled examples, target >90% agreement) — it's how we earn the right to trust the judge in CI, not a design to invent from scratch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrate_judge.py
from unittest.mock import MagicMock
from scripts.calibrate_judge import run_calibration
from eval_harness.judge import JudgeVerdict


def test_run_calibration_computes_agreement_rate():
    fake_client = MagicMock()
    examples = [
        {
            "task_input": "q1", "expected": "e1", "old_output": "o1", "new_output": "n1",
            "human_verdict": "better",
        },
        {
            "task_input": "q2", "expected": "e2", "old_output": "o2", "new_output": "n2",
            "human_verdict": "worse",
        },
    ]

    def fake_judge_pair(client, task_input, expected, old_output, new_output, model="claude-opus-4-8"):
        # Simulate judge agreeing on example 1 (score 0.8 -> "better"), disagreeing on example 2
        score = 0.8 if task_input == "q1" else 0.8
        return JudgeVerdict(score=score, reasoning="stub", input_tokens=1, output_tokens=1)

    report = run_calibration(fake_client, examples, judge_fn=fake_judge_pair)

    assert report.total == 2
    assert report.agreements == 1  # only q1 agrees ("better" <-> score 0.8 > 0.6)
    assert report.agreement_rate == 0.5
    assert len(report.disagreements) == 1
    assert report.disagreements[0]["task_input"] == "q2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibrate_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.calibrate_judge'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/__init__.py
```

```python
# scripts/calibrate_judge.py
from dataclasses import dataclass, field

from eval_harness.judge import judge_pair as default_judge_pair

# Human verdict -> expected score band. "better"/"equivalent" both map to >0.6
# because the rubric treats "equally good" as a pass, matching promptfoo's
# pass/score/reason semantics rather than requiring strict improvement.
VERDICT_BANDS = {
    "better": lambda score: score > 0.6,
    "equivalent": lambda score: score > 0.6,
    "worse": lambda score: score <= 0.6,
}


@dataclass
class CalibrationReport:
    total: int
    agreements: int
    agreement_rate: float
    disagreements: list = field(default_factory=list)


def run_calibration(client, examples: list[dict], judge_fn=default_judge_pair) -> CalibrationReport:
    agreements = 0
    disagreements = []

    for ex in examples:
        verdict = judge_fn(
            client,
            task_input=ex["task_input"],
            expected=ex["expected"],
            old_output=ex["old_output"],
            new_output=ex["new_output"],
        )
        check = VERDICT_BANDS[ex["human_verdict"]]
        if check(verdict.score):
            agreements += 1
        else:
            disagreements.append({**ex, "judge_score": verdict.score})

    total = len(examples)
    return CalibrationReport(
        total=total,
        agreements=agreements,
        agreement_rate=agreements / total if total else 0.0,
        disagreements=disagreements,
    )


if __name__ == "__main__":
    import yaml
    from eval_harness.anthropic_client import AnthropicClient
    import os

    with open("calibration/judge_calibration_set.yaml") as f:
        examples = yaml.safe_load(f)["examples"]

    client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    report = run_calibration(client, examples)
    print(f"Agreement rate: {report.agreement_rate:.1%} ({report.agreements}/{report.total})")
    if report.agreement_rate < 0.9:
        print("WARNING: below 90% target — review disagreements before trusting judge in CI:")
        for d in report.disagreements:
            print(f"  - {d['task_input']}: human={d['human_verdict']}, judge_score={d['judge_score']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibrate_judge.py -v`
Expected: PASS

- [ ] **Step 5: Create the calibration golden set (~25 labeled examples)**

```yaml
# calibration/judge_calibration_set.yaml
# ~25 labeled examples used to calibrate the LLM judge before trusting it in CI.
# human_verdict is Edgar's own judgment: "better" | "equivalent" | "worse".
# Target: >90% agreement with run_calibration before shipping the judge (see
# LIT-39 research: promptfoo's guidance is 30-50 examples / >90% agreement —
# this is a starter set of 25; expand it if agreement is borderline.
examples:
  - task_input: "Summarize: The meeting covered Q3 budget and hiring plans."
    expected: "Meeting discussed Q3 budget and hiring."
    old_output: "The meeting was about the Q3 budget and hiring plans."
    new_output: "The meeting covered Q3 budget and hiring plans."
    human_verdict: equivalent
  - task_input: "What's the capital of Japan?"
    expected: "Tokyo"
    old_output: "The capital of Japan is Tokyo."
    new_output: "Tokyo."
    human_verdict: equivalent
  - task_input: "Extract the ticket key from: '[LIT-39] Eval Harness'"
    expected: "LIT-39"
    old_output: "LIT-39"
    new_output: "The ticket key is LIT-39."
    human_verdict: equivalent
  - task_input: "Is 17 a prime number?"
    expected: "Yes, 17 is prime."
    old_output: "Yes, 17 is a prime number."
    new_output: "No, 17 is not prime."
    human_verdict: worse
  - task_input: "Translate 'good morning' to Spanish."
    expected: "Buenos días"
    old_output: "Buenos días"
    new_output: "Buenas noches"
    human_verdict: worse
  # NOTE: PoC starter set intentionally short (5 filled examples shown here for
  # plan readability). Task 7's implementer must expand this file to ~25
  # examples covering: paraphrase-only changes (equivalent), factual errors
  # introduced (worse), format-only changes (equivalent), dropped required
  # fields (worse), and genuine quality improvements (better) — drawing
  # scenarios from the actual golden sets in golden_sets/ once Task 9 exists.
```

- [ ] **Step 6: Run the calibration script against the real judge (requires `ANTHROPIC_API_KEY`)**

Run: `ANTHROPIC_API_KEY=... python scripts/calibrate_judge.py`
Expected: prints an agreement rate. If below 90%, review printed disagreements and either fix the judge prompt (Task 6) or correct mislabeled examples before proceeding — do not skip this gate, it's what AC#2 requires as rigor, not decoration.

- [ ] **Step 7: Commit**

```bash
git add calibration/judge_calibration_set.yaml scripts/__init__.py scripts/calibrate_judge.py tests/test_calibrate_judge.py
git commit -m "feat: add judge calibration harness against labeled examples"
```

---

### Task 8: Sampler

**Files:**
- Create: `src/eval_harness/sampler.py`
- Test: `tests/test_sampler.py`

**Interfaces:**
- Consumes: `GoldenExample` (Task 3).
- Produces: `sample_examples(examples: list[GoldenExample], sample_size: int, seed: int, full_run: bool = False) -> list[GoldenExample]`. Deterministic (same seed = same sample, so PR re-runs are reproducible and diffable). `full_run=True` bypasses sampling entirely (used for the scheduled full-set run, per AC#3 and the research's promptfoo-inspired sampling pattern).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sampler.py
from eval_harness.sampler import sample_examples
from eval_harness.golden_set import GoldenExample

EXAMPLES = [GoldenExample(id=f"ex{i}", input=f"in{i}", expected=f"exp{i}") for i in range(10)]


def test_sample_returns_requested_size():
    result = sample_examples(EXAMPLES, sample_size=3, seed=42)
    assert len(result) == 3


def test_sample_is_deterministic_for_same_seed():
    result_a = sample_examples(EXAMPLES, sample_size=4, seed=7)
    result_b = sample_examples(EXAMPLES, sample_size=4, seed=7)
    assert [e.id for e in result_a] == [e.id for e in result_b]


def test_different_seeds_can_differ():
    result_a = sample_examples(EXAMPLES, sample_size=4, seed=1)
    result_b = sample_examples(EXAMPLES, sample_size=4, seed=2)
    assert [e.id for e in result_a] != [e.id for e in result_b]


def test_sample_size_larger_than_set_returns_all():
    result = sample_examples(EXAMPLES, sample_size=100, seed=1)
    assert len(result) == len(EXAMPLES)


def test_full_run_ignores_sample_size():
    result = sample_examples(EXAMPLES, sample_size=2, seed=1, full_run=True)
    assert len(result) == len(EXAMPLES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.sampler'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/sampler.py
import random

from eval_harness.golden_set import GoldenExample


def sample_examples(
    examples: list[GoldenExample],
    sample_size: int,
    seed: int,
    full_run: bool = False,
) -> list[GoldenExample]:
    if full_run or sample_size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(examples, sample_size)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sampler.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/eval_harness/sampler.py tests/test_sampler.py
git commit -m "feat: add deterministic golden-set sampler (AC#3 sample-not-every-commit)"
```

---

### Task 9: Cost Tracker

**Files:**
- Create: `src/eval_harness/cost_tracker.py`
- Test: `tests/test_cost_tracker.py`

**Interfaces:**
- Consumes: token counts from `ReplayResult` (Task 4) and `JudgeVerdict` (Task 6).
- Produces: `PRICING: dict[str, dict[str, float]]` (USD per 1M tokens, `{"input": x, "output": y}` per model), `class CostTracker` with `.add_replay_call(model: str, input_tokens: int, output_tokens: int)`, `.add_judge_call(model: str, input_tokens: int, output_tokens: int)`, `.add_embedding_call()` (free — local model, tracked only as a call count for the demo's "cost avoided" story), `.total_cost_usd -> float`, `.breakdown() -> dict`. This is AC#3's real novelty target — no surveyed tool has this.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost_tracker.py
from eval_harness.cost_tracker import CostTracker, PRICING


def test_add_replay_call_accumulates_cost():
    tracker = CostTracker()
    tracker.add_replay_call(model="claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)

    expected = PRICING["claude-sonnet-5"]["input"] + PRICING["claude-sonnet-5"]["output"]
    assert tracker.total_cost_usd == expected


def test_add_judge_call_accumulates_separately_from_replay():
    tracker = CostTracker()
    tracker.add_replay_call(model="claude-sonnet-5", input_tokens=500_000, output_tokens=0)
    tracker.add_judge_call(model="claude-opus-4-8", input_tokens=500_000, output_tokens=0)

    breakdown = tracker.breakdown()
    assert breakdown["replay_cost_usd"] == PRICING["claude-sonnet-5"]["input"] * 0.5
    assert breakdown["judge_cost_usd"] == PRICING["claude-opus-4-8"]["input"] * 0.5
    assert tracker.total_cost_usd == breakdown["replay_cost_usd"] + breakdown["judge_cost_usd"]


def test_embedding_calls_are_free_but_counted():
    tracker = CostTracker()
    tracker.add_embedding_call()
    tracker.add_embedding_call()

    assert tracker.total_cost_usd == 0.0
    assert tracker.breakdown()["embedding_calls"] == 2


def test_unknown_model_raises():
    tracker = CostTracker()
    import pytest
    with pytest.raises(KeyError):
        tracker.add_replay_call(model="not-a-real-model", input_tokens=1, output_tokens=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cost_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.cost_tracker'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/cost_tracker.py
# USD per 1M tokens. Update these when model pricing changes — this is the
# one place cost figures come from, so AC#3's cost-per-PR number stays accurate.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}


class CostTracker:
    def __init__(self):
        self._replay_cost = 0.0
        self._judge_cost = 0.0
        self._embedding_calls = 0

    def add_replay_call(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self._replay_cost += self._call_cost(model, input_tokens, output_tokens)

    def add_judge_call(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self._judge_cost += self._call_cost(model, input_tokens, output_tokens)

    def add_embedding_call(self) -> None:
        self._embedding_calls += 1

    @staticmethod
    def _call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        rates = PRICING[model]  # raises KeyError on unknown model, deliberately
        return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]

    @property
    def total_cost_usd(self) -> float:
        return self._replay_cost + self._judge_cost

    def breakdown(self) -> dict:
        return {
            "replay_cost_usd": round(self._replay_cost, 6),
            "judge_cost_usd": round(self._judge_cost, 6),
            "embedding_calls": self._embedding_calls,
            "total_cost_usd": round(self.total_cost_usd, 6),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cost_tracker.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/eval_harness/cost_tracker.py tests/test_cost_tracker.py
git commit -m "feat: add cost-per-PR tracker (AC#3 novel gap — no surveyed tool has this)"
```

---

### Task 10: Baseline Store + Statistical Drift Gate

**Files:**
- Create: `src/eval_harness/baseline_store.py`
- Create: `src/eval_harness/stats_gate.py`
- Test: `tests/test_baseline_store.py`
- Test: `tests/test_stats_gate.py`

**Interfaces:**
- Produces (`baseline_store.py`): `class BaselineStore` with `.load(path: str) -> dict[str, list[float]]` (prompt_name -> list of past run mean scores, most-recent-last), `.append_run(path: str, prompt_name: str, mean_score: float, max_history: int = 10) -> None`.
- Produces (`stats_gate.py`): `@dataclass GateDecision {is_regression: bool, p_value: float, effect_size: float, reason: str}`, `evaluate_gate(new_scores: list[float], baseline_scores: list[float], effect_size_floor: float = 0.03, p_threshold: float = 0.05) -> GateDecision`. This is the PoC's stretch-goal novelty (per research finding #4/#5): Welch's t-test + effect-size floor instead of a bare threshold, explicitly modeled on — and credited to — the single FutureAGI blog post that proposed it.

- [ ] **Step 1: Write the failing test for baseline_store**

```python
# tests/test_baseline_store.py
import json
from eval_harness.baseline_store import BaselineStore


def test_append_run_creates_new_file(tmp_path):
    store_path = str(tmp_path / "baseline.json")
    store = BaselineStore()
    store.append_run(store_path, "faq-demo-agent", mean_score=0.9)

    data = json.loads(open(store_path).read())
    assert data["faq-demo-agent"] == [0.9]


def test_append_run_accumulates_history(tmp_path):
    store_path = str(tmp_path / "baseline.json")
    store = BaselineStore()
    store.append_run(store_path, "faq-demo-agent", mean_score=0.9)
    store.append_run(store_path, "faq-demo-agent", mean_score=0.85)

    data = store.load(store_path)
    assert data["faq-demo-agent"] == [0.9, 0.85]


def test_append_run_caps_history_at_max():
    import tempfile, os
    fd, store_path = tempfile.mkstemp()
    os.close(fd)
    os.remove(store_path)

    store = BaselineStore()
    for score in [0.1, 0.2, 0.3, 0.4]:
        store.append_run(store_path, "agent-x", mean_score=score, max_history=3)

    data = store.load(store_path)
    assert data["agent-x"] == [0.2, 0.3, 0.4]  # oldest (0.1) evicted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_baseline_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.baseline_store'`

- [ ] **Step 3: Write minimal implementation for baseline_store**

```python
# src/eval_harness/baseline_store.py
import json
import os


class BaselineStore:
    """JSON-backed rolling baseline of past run mean scores, keyed by prompt name.
    PoC limitation: a single JSON file, not a real time-series DB — sufficient
    to prove the statistical-gating pattern without building infra the PoC
    doesn't need (see plan's Global Constraints)."""

    def load(self, path: str) -> dict[str, list[float]]:
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def append_run(self, path: str, prompt_name: str, mean_score: float, max_history: int = 10) -> None:
        data = self.load(path)
        history = data.get(prompt_name, [])
        history.append(mean_score)
        data[prompt_name] = history[-max_history:]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_baseline_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write the failing test for stats_gate**

```python
# tests/test_stats_gate.py
from eval_harness.stats_gate import evaluate_gate


def test_no_regression_when_scores_are_similar():
    new_scores = [0.90, 0.91, 0.89, 0.92, 0.90]
    baseline_scores = [0.90, 0.89, 0.91, 0.90, 0.90, 0.91, 0.89]

    decision = evaluate_gate(new_scores, baseline_scores)

    assert decision.is_regression is False


def test_regression_when_scores_drop_significantly():
    new_scores = [0.50, 0.48, 0.52, 0.49, 0.51, 0.50, 0.49]
    baseline_scores = [0.90, 0.89, 0.91, 0.90, 0.92, 0.88, 0.91]

    decision = evaluate_gate(new_scores, baseline_scores)

    assert decision.is_regression is True
    assert decision.p_value < 0.05
    assert decision.effect_size > 0.03


def test_small_noisy_drop_below_effect_size_floor_is_not_regression():
    # Drop exists but is within the noise floor (effect_size < 0.03) even if
    # it happens to be statistically significant with a huge sample.
    new_scores = [0.881] * 50
    baseline_scores = [0.90] * 50

    decision = evaluate_gate(new_scores, baseline_scores, effect_size_floor=0.03)

    assert decision.is_regression is False
    assert "effect size" in decision.reason.lower()


def test_insufficient_baseline_history_does_not_crash():
    new_scores = [0.5, 0.5]
    baseline_scores = []  # no history yet — first run for this prompt

    decision = evaluate_gate(new_scores, baseline_scores)

    assert decision.is_regression is False
    assert "insufficient baseline" in decision.reason.lower()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_stats_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.stats_gate'`

- [ ] **Step 7: Write minimal implementation for stats_gate**

```python
# src/eval_harness/stats_gate.py
from dataclasses import dataclass

from scipy import stats


# NOTE: this gate design (Welch's t-test + effect-size floor vs. a rolling
# baseline, rather than a bare absolute-threshold cutoff) is inspired by a
# single 2026 vendor blog post (FutureAGI) — per LIT-39 research, no major
# eval tool (promptfoo/DeepEval/Braintrust/LangSmith) implements this. It is
# the PoC's genuine statistical-rigor differentiator, not an established
# industry practice — say so in the demo and article, don't overclaim.


@dataclass
class GateDecision:
    is_regression: bool
    p_value: float
    effect_size: float
    reason: str


MIN_BASELINE_SIZE = 3


def evaluate_gate(
    new_scores: list[float],
    baseline_scores: list[float],
    effect_size_floor: float = 0.03,
    p_threshold: float = 0.05,
) -> GateDecision:
    if len(baseline_scores) < MIN_BASELINE_SIZE:
        return GateDecision(
            is_regression=False,
            p_value=1.0,
            effect_size=0.0,
            reason="insufficient baseline history — treating as pass until enough runs accumulate",
        )

    t_stat, p_value = stats.ttest_ind(new_scores, baseline_scores, equal_var=False)
    mean_diff = sum(baseline_scores) / len(baseline_scores) - sum(new_scores) / len(new_scores)
    effect_size = abs(mean_diff)  # normalized 0-1 score scale, so |mean diff| doubles as the effect size

    if p_value < p_threshold and effect_size >= effect_size_floor and mean_diff > 0:
        return GateDecision(
            is_regression=True,
            p_value=p_value,
            effect_size=effect_size,
            reason=f"statistically significant drop (p={p_value:.4f}, effect size={effect_size:.4f} >= floor {effect_size_floor})",
        )

    if p_value < p_threshold and mean_diff > 0:
        return GateDecision(
            is_regression=False,
            p_value=p_value,
            effect_size=effect_size,
            reason=f"drop is statistically significant but below effect-size floor ({effect_size:.4f} < {effect_size_floor}) — treated as noise",
        )

    return GateDecision(
        is_regression=False,
        p_value=p_value,
        effect_size=effect_size,
        reason=f"no statistically significant regression (p={p_value:.4f})",
    )
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_stats_gate.py -v`
Expected: PASS (4 passed)

- [ ] **Step 9: Commit**

```bash
git add src/eval_harness/baseline_store.py src/eval_harness/stats_gate.py tests/test_baseline_store.py tests/test_stats_gate.py
git commit -m "feat: add rolling baseline store + Welch's t-test statistical drift gate"
```

---

### Task 11: Report Assembly

**Files:**
- Create: `src/eval_harness/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `JudgeVerdict` list, `GateDecision` (Task 10), `CostTracker.breakdown()` (Task 9).
- Produces: `@dataclass RunReport {prompt_name: str, sample_size: int, mean_score: float, gate_decision: GateDecision, cost_breakdown: dict, per_example: list[dict]}`, `build_report(...) -> RunReport`, `render_markdown(report: RunReport) -> str` (the exact text posted as the PR comment — this is what AC#1's "reports the delta before merge" produces).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from eval_harness.report import build_report, render_markdown
from eval_harness.judge import JudgeVerdict
from eval_harness.stats_gate import GateDecision


def test_build_report_computes_mean_score():
    verdicts = [
        JudgeVerdict(score=0.8, reasoning="fine", input_tokens=10, output_tokens=5),
        JudgeVerdict(score=0.6, reasoning="ok", input_tokens=10, output_tokens=5),
    ]
    gate_decision = GateDecision(is_regression=False, p_value=0.5, effect_size=0.01, reason="no regression")
    cost_breakdown = {"replay_cost_usd": 0.001, "judge_cost_usd": 0.01, "embedding_calls": 2, "total_cost_usd": 0.011}

    report = build_report(
        prompt_name="faq-demo-agent",
        example_ids=["ex1", "ex2"],
        verdicts=verdicts,
        gate_decision=gate_decision,
        cost_breakdown=cost_breakdown,
    )

    assert report.prompt_name == "faq-demo-agent"
    assert report.sample_size == 2
    assert report.mean_score == 0.7
    assert report.gate_decision.is_regression is False
    assert report.cost_breakdown["total_cost_usd"] == 0.011


def test_render_markdown_includes_pass_and_cost():
    verdicts = [JudgeVerdict(score=0.9, reasoning="great", input_tokens=1, output_tokens=1)]
    gate_decision = GateDecision(is_regression=False, p_value=0.8, effect_size=0.01, reason="no statistically significant regression (p=0.8000)")
    cost_breakdown = {"replay_cost_usd": 0.002, "judge_cost_usd": 0.02, "embedding_calls": 1, "total_cost_usd": 0.022}

    report = build_report("faq-demo-agent", ["ex1"], verdicts, gate_decision, cost_breakdown)
    markdown = render_markdown(report)

    assert "faq-demo-agent" in markdown
    assert "PASS" in markdown
    assert "$0.022" in markdown
    assert "0.90" in markdown


def test_render_markdown_shows_block_on_regression():
    verdicts = [JudgeVerdict(score=0.3, reasoning="bad", input_tokens=1, output_tokens=1)]
    gate_decision = GateDecision(is_regression=True, p_value=0.01, effect_size=0.4, reason="statistically significant drop (p=0.0100, effect size=0.4000 >= floor 0.03)")
    cost_breakdown = {"replay_cost_usd": 0.002, "judge_cost_usd": 0.02, "embedding_calls": 0, "total_cost_usd": 0.022}

    report = build_report("faq-demo-agent", ["ex1"], verdicts, gate_decision, cost_breakdown)
    markdown = render_markdown(report)

    assert "BLOCKED" in markdown
    assert "statistically significant drop" in markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/report.py
from dataclasses import dataclass, field

from eval_harness.judge import JudgeVerdict
from eval_harness.stats_gate import GateDecision


@dataclass
class RunReport:
    prompt_name: str
    sample_size: int
    mean_score: float
    gate_decision: GateDecision
    cost_breakdown: dict
    per_example: list = field(default_factory=list)


def build_report(
    prompt_name: str,
    example_ids: list[str],
    verdicts: list[JudgeVerdict],
    gate_decision: GateDecision,
    cost_breakdown: dict,
) -> RunReport:
    mean_score = sum(v.score for v in verdicts) / len(verdicts) if verdicts else 0.0
    per_example = [
        {"example_id": eid, "score": v.score, "reasoning": v.reasoning}
        for eid, v in zip(example_ids, verdicts)
    ]
    return RunReport(
        prompt_name=prompt_name,
        sample_size=len(verdicts),
        mean_score=round(mean_score, 4),
        gate_decision=gate_decision,
        cost_breakdown=cost_breakdown,
        per_example=per_example,
    )


def render_markdown(report: RunReport) -> str:
    status = "BLOCKED" if report.gate_decision.is_regression else "PASS"
    lines = [
        f"## Eval Harness Report — `{report.prompt_name}`",
        "",
        f"**Status: {status}**",
        "",
        f"- Sample size: {report.sample_size}",
        f"- Mean semantic score: {report.mean_score:.2f}",
        f"- Gate reasoning: {report.gate_decision.reason}",
        f"- Cost for this PR: **${report.cost_breakdown['total_cost_usd']:.3f}** "
        f"(replay ${report.cost_breakdown['replay_cost_usd']:.4f} + "
        f"judge ${report.cost_breakdown['judge_cost_usd']:.4f}, "
        f"{report.cost_breakdown['embedding_calls']} free embedding pre-filter calls)",
        "",
        "| Example | Score | Judge reasoning |",
        "|---|---|---|",
    ]
    for ex in report.per_example:
        lines.append(f"| {ex['example_id']} | {ex['score']:.2f} | {ex['reasoning']} |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/eval_harness/report.py tests/test_report.py
git commit -m "feat: add report assembly producing the PR-comment markdown (AC#1 delta report)"
```

---

### Task 12: Prompt Registry Config

**Files:**
- Create: `src/eval_harness/prompt_registry.py`
- Create: `prompt_registry.yaml`
- Test: `tests/test_prompt_registry.py`

**Interfaces:**
- Produces: `@dataclass PromptEntry {prompt_file: str, golden_set_file: str, sample_size: int, judge_model: str}`, `load_registry(path: str) -> list[PromptEntry]`, `find_entry_for_changed_file(registry: list[PromptEntry], changed_file_path: str) -> PromptEntry | None`. This is what lets the CLI know, given a git diff, which golden set applies to which changed prompt file — the mapping AC#1 needs ("a prompt change in **any agent**" implies more than one registered prompt).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompt_registry.py
from eval_harness.prompt_registry import load_registry, find_entry_for_changed_file, PromptEntry

REGISTRY_YAML = """
prompts:
  - prompt_file: demo_agent/faq_agent_prompt.md
    golden_set_file: golden_sets/faq-demo-agent.yaml
    sample_size: 5
    judge_model: claude-opus-4-8
  - prompt_file: .claude/skills/jira-ticket-kickoff/SKILL.md
    golden_set_file: golden_sets/jira-ticket-kickoff.yaml
    sample_size: 3
    judge_model: claude-opus-4-8
"""


def test_load_registry_parses_all_entries(tmp_path):
    p = tmp_path / "registry.yaml"
    p.write_text(REGISTRY_YAML)

    registry = load_registry(str(p))

    assert len(registry) == 2
    assert registry[0].prompt_file == "demo_agent/faq_agent_prompt.md"
    assert registry[0].sample_size == 5


def test_find_entry_for_changed_file_matches():
    registry = [
        PromptEntry("demo_agent/faq_agent_prompt.md", "golden_sets/faq-demo-agent.yaml", 5, "claude-opus-4-8"),
    ]

    entry = find_entry_for_changed_file(registry, "demo_agent/faq_agent_prompt.md")

    assert entry is not None
    assert entry.golden_set_file == "golden_sets/faq-demo-agent.yaml"


def test_find_entry_for_changed_file_returns_none_when_unregistered():
    registry = [
        PromptEntry("demo_agent/faq_agent_prompt.md", "golden_sets/faq-demo-agent.yaml", 5, "claude-opus-4-8"),
    ]

    entry = find_entry_for_changed_file(registry, "some/other/file.py")

    assert entry is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.prompt_registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/prompt_registry.py
from dataclasses import dataclass

import yaml


@dataclass
class PromptEntry:
    prompt_file: str
    golden_set_file: str
    sample_size: int
    judge_model: str


def load_registry(path: str) -> list[PromptEntry]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [
        PromptEntry(
            prompt_file=p["prompt_file"],
            golden_set_file=p["golden_set_file"],
            sample_size=p["sample_size"],
            judge_model=p["judge_model"],
        )
        for p in raw["prompts"]
    ]


def find_entry_for_changed_file(registry: list[PromptEntry], changed_file_path: str) -> PromptEntry | None:
    for entry in registry:
        if entry.prompt_file == changed_file_path:
            return entry
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_registry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Create the real top-level `prompt_registry.yaml`**

```yaml
# prompt_registry.yaml
prompts:
  - prompt_file: demo_agent/faq_agent_prompt.md
    golden_set_file: golden_sets/faq-demo-agent.yaml
    sample_size: 5
    judge_model: claude-opus-4-8
  - prompt_file: .claude/skills/jira-ticket-kickoff/SKILL.md
    golden_set_file: golden_sets/jira-ticket-kickoff.yaml
    sample_size: 3
    judge_model: claude-opus-4-8
```

- [ ] **Step 6: Commit**

```bash
git add src/eval_harness/prompt_registry.py prompt_registry.yaml tests/test_prompt_registry.py
git commit -m "feat: add prompt registry mapping prompt files to golden sets (AC#1 'any agent')"
```

---

### Task 13: Demo Golden Sets + Demo Agent Prompt Versions

**Files:**
- Create: `demo_agent/faq_agent_prompt.md`
- Create: `demo_agent/faq_agent_prompt_regressed.md`
- Create: `golden_sets/faq-demo-agent.yaml`
- Create: `golden_sets/jira-ticket-kickoff.yaml`

**Interfaces:**
- Produces: two golden sets consumable by `load_golden_set` (Task 3), and two versions of a small demo prompt for the scripted "regression caught live" demo moment.

- [ ] **Step 1: Write the "good" demo agent prompt**

```markdown
<!-- demo_agent/faq_agent_prompt.md -->
You are a support FAQ agent for a small SaaS product called TaskFlow.
Answer the user's question in 1-2 sentences, using only the facts given below.
If the question can't be answered from these facts, say so honestly rather
than guessing.

Facts:
- TaskFlow's free plan allows up to 3 projects.
- TaskFlow's Pro plan ($12/month) allows unlimited projects and adds team roles.
- Data exports are available on all plans as CSV.
- TaskFlow does not currently support SSO login.
```

- [ ] **Step 2: Write the deliberately regressed version (for the scripted demo)**

```markdown
<!-- demo_agent/faq_agent_prompt_regressed.md -->
You are a support FAQ agent for a small SaaS product called TaskFlow.
Answer the user's question as helpfully as possible.

Facts:
- TaskFlow's free plan allows up to 3 projects.
- TaskFlow's Pro plan ($12/month) allows unlimited projects and adds team roles.
- Data exports are available on all plans as CSV.
- TaskFlow does not currently support SSO login.
```

Note the regression is subtle by design: dropping "using only the facts given below... say so honestly rather than guessing" removes the anti-hallucination instruction — the demo's golden set (Step 3) includes a question with no answer in the facts, which the good version correctly declines and the regressed version will likely hallucinate an answer for. This produces a real, judge-catchable regression rather than an artificially broken prompt.

- [ ] **Step 3: Write the demo agent's golden set**

```yaml
# golden_sets/faq-demo-agent.yaml
name: faq-demo-agent
examples:
  - id: free-plan-limit
    input: "How many projects can I create on the free plan?"
    expected: "Up to 3 projects on the free plan."
  - id: pro-plan-price
    input: "How much does the Pro plan cost and what does it add?"
    expected: "Pro plan is $12/month and adds unlimited projects plus team roles."
  - id: export-format
    input: "Can I export my data, and in what format?"
    expected: "Yes, CSV export is available on all plans."
  - id: sso-support
    input: "Do you support SSO login?"
    expected: "No, SSO login is not currently supported."
  - id: unanswerable-refund-policy
    input: "What's your refund policy if I cancel Pro after 2 weeks?"
    expected: "I don't have information about the refund policy — please contact support directly."
```

- [ ] **Step 4: Write the jira-ticket-kickoff dogfood golden set**

```yaml
# golden_sets/jira-ticket-kickoff.yaml
# Dogfoods our own SKILL.md (Task 12's registry entry) — mirrors LIT-38's
# pattern of using the built system on itself. Inputs are realistic
# invocations of the skill; "expected" is a structural checklist (does the
# response include these steps?) rather than exact prose, since the skill's
# output is a plan of action, not a single fact.
name: jira-ticket-kickoff
examples:
  - id: fresh-ticket
    input: "I have a new ticket LIT-40, an XML export at ./LIT-40.xml, no prior memory exists. What should I do?"
    expected: >
      Should describe: parsing the ticket XML for key/summary/ACs, checking
      for prior project-context memory first, running deep-research via the
      Workflow tool on the ticket's technical domain (not the ticket text),
      saving the report as lit-40-state-of-the-art.md with per-finding AC
      tags plus a Landscape Map and PoC Design Implications section, then
      writing a lit-40-project-context.md memory file with the established
      frontmatter and section structure.
  - id: resuming-ticket
    input: "I'm picking back up LIT-38 — there's already a project-context memory for it. What's next?"
    expected: >
      Should describe: read the existing project-context memory first, skip
      re-running research (it's already banked), and resume from wherever
      the Task Progress checklist left off, updating the same memory file
      going forward rather than starting a new kickoff arc.
```

- [ ] **Step 5: Verify both golden sets load cleanly**

Run:
```bash
python3 -c "
from eval_harness.golden_set import load_golden_set
gs1 = load_golden_set('golden_sets/faq-demo-agent.yaml')
gs2 = load_golden_set('golden_sets/jira-ticket-kickoff.yaml')
print(gs1.name, len(gs1.examples))
print(gs2.name, len(gs2.examples))
"
```
Expected: `faq-demo-agent 5` and `jira-ticket-kickoff 2` printed with no errors.

- [ ] **Step 6: Commit**

```bash
git add demo_agent/ golden_sets/
git commit -m "feat: add demo agent prompt pair + golden sets for demo agent and jira-ticket-kickoff dogfooding"
```

---

### Task 14: CLI Orchestrator

**Files:**
- Create: `src/eval_harness/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: every module from Tasks 2–13.
- Produces: `run_eval(prompt_file: str, old_prompt_text: str, new_prompt_text: str, registry_path: str = "prompt_registry.yaml", baseline_path: str = "baseline_store.json", full_run: bool = False, seed: int = 0, api_key: str | None = None) -> RunReport`, plus a `main()` argparse entrypoint wiring `eval-harness run --prompt-file <path> --old-ref <git-ref> [--full-run]`. This is the single function that ties the whole pipeline together: registry lookup → golden-set load → sample → replay both versions → embedding pre-filter → judge only what didn't pre-filter → cost tracking → stats gate against baseline → append new run to baseline → build + return report.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from unittest.mock import MagicMock, patch
from eval_harness.cli import run_eval
from eval_harness.anthropic_client import CompletionResult
from eval_harness.judge import JudgeVerdict


def test_run_eval_produces_passing_report_when_outputs_are_embedding_identical(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "prompts:\n"
        "  - prompt_file: demo.md\n"
        f"    golden_set_file: {tmp_path}/golden.yaml\n"
        "    sample_size: 1\n"
        "    judge_model: claude-opus-4-8\n"
    )
    (tmp_path / "golden.yaml").write_text(
        "name: demo\nexamples:\n  - id: ex1\n    input: 'hi'\n    expected: 'hello'\n"
    )
    baseline_path = tmp_path / "baseline.json"

    with patch("eval_harness.cli.AnthropicClient") as MockClient, \
         patch("eval_harness.cli.EmbeddingFilter") as MockFilter:
        MockClient.return_value.complete.return_value = CompletionResult(
            text="hello there", input_tokens=10, output_tokens=5, model="claude-sonnet-5"
        )
        # Embedding filter says outputs are identical -> judge is never called
        MockFilter.return_value.is_likely_unchanged.return_value = True

        report = run_eval(
            prompt_file="demo.md",
            old_prompt_text="You are an assistant.",
            new_prompt_text="You are a helpful assistant.",
            registry_path=str(registry_path),
            baseline_path=str(baseline_path),
            seed=1,
            api_key="fake-key",
        )

    assert report.prompt_name == "demo"
    assert report.gate_decision.is_regression is False
    assert report.mean_score == 1.0  # embedding-identical short-circuits to a perfect score
    assert report.cost_breakdown["judge_cost_usd"] == 0.0  # judge was skipped


def test_run_eval_raises_for_unregistered_prompt_file(tmp_path):
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("prompts: []\n")

    import pytest
    with pytest.raises(ValueError, match="not registered"):
        run_eval(
            prompt_file="unknown.md",
            old_prompt_text="a",
            new_prompt_text="b",
            registry_path=str(registry_path),
            api_key="fake-key",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_harness.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/eval_harness/cli.py
import argparse
import subprocess
import sys

from eval_harness.anthropic_client import AnthropicClient
from eval_harness.baseline_store import BaselineStore
from eval_harness.cost_tracker import CostTracker
from eval_harness.embedding_filter import EmbeddingFilter
from eval_harness.golden_set import load_golden_set
from eval_harness.judge import judge_pair, JudgeVerdict
from eval_harness.prompt_registry import load_registry, find_entry_for_changed_file
from eval_harness.replay import replay_prompt
from eval_harness.report import build_report, render_markdown, RunReport
from eval_harness.sampler import sample_examples
from eval_harness.stats_gate import evaluate_gate


def run_eval(
    prompt_file: str,
    old_prompt_text: str,
    new_prompt_text: str,
    registry_path: str = "prompt_registry.yaml",
    baseline_path: str = "baseline_store.json",
    full_run: bool = False,
    seed: int = 0,
    api_key: str | None = None,
) -> RunReport:
    registry = load_registry(registry_path)
    entry = find_entry_for_changed_file(registry, prompt_file)
    if entry is None:
        raise ValueError(f"{prompt_file} is not registered in {registry_path} — add it before it can be gated")

    golden_set = load_golden_set(entry.golden_set_file)
    examples = sample_examples(golden_set.examples, entry.sample_size, seed=seed, full_run=full_run)

    client = AnthropicClient(api_key=api_key)
    embedding_filter = EmbeddingFilter()
    cost_tracker = CostTracker()

    old_results = replay_prompt(client, old_prompt_text, examples)
    new_results = replay_prompt(client, new_prompt_text, examples)
    for r in old_results + new_results:
        cost_tracker.add_replay_call(model="claude-sonnet-5", input_tokens=r.input_tokens, output_tokens=r.output_tokens)

    verdicts: list[JudgeVerdict] = []
    for example, old_r, new_r in zip(examples, old_results, new_results):
        embedding_filter.is_likely_unchanged(old_r.output, new_r.output)  # count the call even when we still judge
        cost_tracker.add_embedding_call()
        if embedding_filter.is_likely_unchanged(old_r.output, new_r.output):
            verdicts.append(JudgeVerdict(score=1.0, reasoning="embedding pre-filter: outputs near-identical, judge skipped", input_tokens=0, output_tokens=0))
            continue
        verdict = judge_pair(client, example.input, example.expected, old_r.output, new_r.output, model=entry.judge_model)
        cost_tracker.add_judge_call(model=entry.judge_model, input_tokens=verdict.input_tokens, output_tokens=verdict.output_tokens)
        verdicts.append(verdict)

    baseline_store = BaselineStore()
    baseline_scores = baseline_store.load(baseline_path).get(golden_set.name, [])
    new_scores = [v.score for v in verdicts]
    gate_decision = evaluate_gate(new_scores, baseline_scores)

    mean_score = sum(new_scores) / len(new_scores) if new_scores else 0.0
    baseline_store.append_run(baseline_path, golden_set.name, mean_score)

    return build_report(
        prompt_name=golden_set.name,
        example_ids=[e.id for e in examples],
        verdicts=verdicts,
        gate_decision=gate_decision,
        cost_breakdown=cost_tracker.breakdown(),
    )


def _read_prompt_at_ref(prompt_file: str, git_ref: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{git_ref}:{prompt_file}"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def main():
    parser = argparse.ArgumentParser(prog="eval-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Replay a prompt change against its golden set")
    run_parser.add_argument("--prompt-file", required=True)
    run_parser.add_argument("--old-ref", default="origin/main", help="git ref to read the OLD prompt version from")
    run_parser.add_argument("--registry-path", default="prompt_registry.yaml")
    run_parser.add_argument("--baseline-path", default="baseline_store.json")
    run_parser.add_argument("--full-run", action="store_true")
    run_parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "run":
        old_text = _read_prompt_at_ref(args.prompt_file, args.old_ref)
        with open(args.prompt_file) as f:
            new_text = f.read()

        import os
        report = run_eval(
            prompt_file=args.prompt_file,
            old_prompt_text=old_text,
            new_prompt_text=new_text,
            registry_path=args.registry_path,
            baseline_path=args.baseline_path,
            full_run=args.full_run,
            seed=args.seed,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        print(render_markdown(report))
        sys.exit(1 if report.gate_decision.is_regression else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full test suite to confirm no regressions across modules**

Run: `pytest -v`
Expected: all tests across all modules PASS.

- [ ] **Step 6: Commit**

```bash
git add src/eval_harness/cli.py tests/test_cli.py
git commit -m "feat: add CLI orchestrator tying replay, judge, sampling, cost, and gate together"
```

---

### Task 15: GitHub Action — CI Merge Gate

**Files:**
- Create: `.github/workflows/eval-gate.yml`
- Create: `README.md`

**Interfaces:**
- Consumes: `eval-harness run` (Task 14's CLI entrypoint).
- Produces: a working CI job that triggers on PRs touching any registered prompt file, runs the CLI, posts the markdown report as a PR comment, and fails the job (blocking merge) when `report.gate_decision.is_regression` is true — this is AC#1's "triggers replay and reports the delta before merge" made real.

- [ ] **Step 1: Write the workflow file**

```yaml
# .github/workflows/eval-gate.yml
name: Eval Harness Gate

on:
  pull_request:
    paths:
      - "demo_agent/**"
      - ".claude/skills/jira-ticket-kickoff/SKILL.md"

jobs:
  eval-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # need history so `git show origin/main:<file>` works

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install eval-harness
        run: pip install -e ".[dev]"

      - name: Detect changed registered prompt files
        id: changed
        run: |
          git fetch origin ${{ github.base_ref }}
          CHANGED=$(git diff --name-only origin/${{ github.base_ref }}...HEAD -- demo_agent .claude/skills/jira-ticket-kickoff/SKILL.md)
          echo "files<<EOF" >> "$GITHUB_OUTPUT"
          echo "$CHANGED" >> "$GITHUB_OUTPUT"
          echo "EOF" >> "$GITHUB_OUTPUT"

      - name: Restore baseline cache
        uses: actions/cache@v4
        with:
          path: baseline_store.json
          key: eval-harness-baseline

      - name: Run eval harness on each changed prompt file
        id: eval
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          set -e
          REPORT_FILE=$(mktemp)
          EXIT_CODE=0
          for f in ${{ steps.changed.outputs.files }}; do
            echo "Running eval-harness on: $f"
            eval-harness run --prompt-file "$f" --old-ref "origin/${{ github.base_ref }}" >> "$REPORT_FILE" || EXIT_CODE=$?
          done
          echo "report_file=$REPORT_FILE" >> "$GITHUB_OUTPUT"
          echo "exit_code=$EXIT_CODE" >> "$GITHUB_OUTPUT"

      - name: Post report as PR comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const body = fs.readFileSync('${{ steps.eval.outputs.report_file }}', 'utf8');
            await github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body || "_No registered prompt files changed in this PR._",
            });

      - name: Fail job if a regression was detected
        if: steps.eval.outputs.exit_code != '0'
        run: exit 1
```

- [ ] **Step 2: Write the README**

```markdown
<!-- README.md -->
# eval-harness (LIT-39 PoC)

Gates PR merges when a "prompt file" changes — any AI-behavior-defining file
registered in `prompt_registry.yaml` (a system prompt, a Claude Code skill, an
agent instruction file). On every PR touching a registered file, CI replays a
sample of that file's golden set through the old and new versions, scores the
difference with an LLM judge (not string match), tracks the real dollar cost
of doing so, and blocks the merge only when the drop is a statistically real
regression against a rolling baseline — not just below a static threshold.

## Why this exists

Every major eval vendor (promptfoo, Braintrust, DeepEval, LangSmith) already
ships "run an eval suite in CI and gate on scores." That part isn't novel —
see `lit-39-state-of-the-art.md` for the research. What none of them do is
report what that judging actually costs per PR, or tell you whether a score
drop is statistically real versus noise. This PoC builds and demonstrates both.

## Quickstart

\`\`\`bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v                          # run the full test suite
python scripts/calibrate_judge.py  # calibrate the judge before trusting it (needs ANTHROPIC_API_KEY)
eval-harness run --prompt-file demo_agent/faq_agent_prompt.md --old-ref HEAD~1
\`\`\`

## Demo script

1. Edit `demo_agent/faq_agent_prompt.md` with a harmless wording change, open a
   PR — CI passes, comment shows a high score and low cost.
2. Replace it with the contents of `demo_agent/faq_agent_prompt_regressed.md`,
   open a PR — CI catches the dropped anti-hallucination instruction on the
   `unanswerable-refund-policy` golden-set example, reports a real regression,
   and blocks the merge, with the actual dollar cost of that judgment shown.

## PoC limitations (documented, not hidden)

- Single JSON file as the baseline store — a real deployment would use a
  proper time-series store, not needed to prove the pattern here.
- Statistical gate (Welch's t-test + effect-size floor) is inspired by one
  vendor blog post (FutureAGI, 2026), not an adopted industry standard — see
  `lit-39-state-of-the-art.md` for the caveat.
- Embedding pre-filter model runs locally and is free, but adds a dependency
  (`sentence-transformers`) and a few seconds of model load time per run.
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/eval-gate.yml README.md
git commit -m "feat: add GitHub Action CI merge gate + README (AC#1 full loop closed)"
```

- [ ] **Step 4: Manual verification (Edgar runs this)**

Push the repo to GitHub, add `ANTHROPIC_API_KEY` as a repo secret, open a real PR editing `demo_agent/faq_agent_prompt.md`, and confirm the Action runs and posts a comment. Then open the scripted regression PR (Step 2 of the demo script above) and confirm it actually blocks merge. This is the point where the PoC becomes demonstrably real, not just unit-tested.

---

## Self-Review

**Spec coverage:**
- AC#1 (prompt change triggers replay + delta report before merge) — Tasks 4 (replay), 11 (report), 12 (registry), 15 (GitHub Action) ✅
- AC#2 (semantic diff, not string match) — Tasks 5 (embedding pre-filter, explicitly non-authoritative), 6 (LLM judge), 7 (calibration) ✅
- AC#3 (sample not every commit, cost per PR) — Tasks 8 (sampler), 9 (cost tracker), 10 (stats gate as the stretch novelty) ✅
- AC#4 (demo to Labs) — Task 13 (demo golden sets + regression pair) + Task 15 Step 4 (manual live verification) sets up the demo; actual scheduling is pending Mariana per the memory file, not blocked by this plan.
- AC#5 (article) — Task 15's README + `lit-39-state-of-the-art.md`'s article-hook framing give the raw material; drafting the article itself is a follow-on task after the PoC is demoed, same as LIT-37/38.
- Dogfooding on `jira-ticket-kickoff` SKILL.md — Task 12 (registry entry) + Task 13 (golden set) ✅
- Cost-per-PR reporting surfaced in the actual PR comment (not just internal tracking) — Task 11 `render_markdown` ✅

**Placeholder scan:** No TBD/TODO markers in any step's code. Task 7's calibration YAML has an explicit inline note that the implementer must expand the starter 5-example set to ~25 — flagged as a required follow-up action within the task itself, not a silent gap, and Step 6 of that task requires actually running the calibration and reviewing the agreement rate before moving on.

**Type consistency:** `ReplayResult` (Task 4) fields (`output`, `input_tokens`, `output_tokens`) match what `cli.py` (Task 14) reads. `JudgeVerdict.score` (0.0-1.0 normalized) is used consistently in `stats_gate.py`, `report.py`, and `cli.py`'s embedding short-circuit (`score=1.0`). `GoldenExample`/`GoldenSet` field names match across `golden_set.py`, `sampler.py`, `replay.py`, and `prompt_registry.yaml`'s consumers. `CostTracker.breakdown()` keys (`replay_cost_usd`, `judge_cost_usd`, `embedding_calls`, `total_cost_usd`) match exactly what `report.py` and `cli.py` read.
