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
