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
