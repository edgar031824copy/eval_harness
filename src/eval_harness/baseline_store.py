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
