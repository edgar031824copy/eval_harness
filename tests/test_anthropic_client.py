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
