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
