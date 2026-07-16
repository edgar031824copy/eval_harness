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
