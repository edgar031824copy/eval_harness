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
