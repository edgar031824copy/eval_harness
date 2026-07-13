import random

from eval_harness.golden_set import GoldenExample


def sample_examples(
    examples: list[GoldenExample],
    sample_size: int,
    seed: int,
    full_run: bool = False,
) -> list[GoldenExample]:
    if full_run or sample_size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(examples, sample_size)
