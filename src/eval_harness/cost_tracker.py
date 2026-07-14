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
