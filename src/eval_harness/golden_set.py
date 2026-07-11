from dataclasses import dataclass

import yaml


class GoldenSetError(Exception):
    pass


@dataclass
class GoldenExample:
    id: str
    input: str
    expected: str


@dataclass
class GoldenSet:
    name: str
    examples: list[GoldenExample]


REQUIRED_FIELDS = ("id", "input", "expected")


def load_golden_set(path: str) -> GoldenSet:
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "name" not in raw:
        raise GoldenSetError(f"{path}: missing top-level 'name' field")

    raw_examples = raw.get("examples") or []
    if len(raw_examples) == 0:
        raise GoldenSetError(f"{path}: golden set must contain at least one example")

    examples = []
    for i, ex in enumerate(raw_examples):
        for field in REQUIRED_FIELDS:
            if field not in ex:
                raise GoldenSetError(
                    f"{path}: example at index {i} is missing required field '{field}'"
                )
        examples.append(GoldenExample(id=ex["id"], input=ex["input"], expected=ex["expected"]))

    return GoldenSet(name=raw["name"], examples=examples)
