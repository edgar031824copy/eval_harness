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
