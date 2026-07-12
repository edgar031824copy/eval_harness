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
