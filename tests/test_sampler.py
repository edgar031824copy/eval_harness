from eval_harness.sampler import sample_examples
from eval_harness.golden_set import GoldenExample

EXAMPLES = [GoldenExample(id=f"ex{i}", input=f"in{i}", expected=f"exp{i}") for i in range(10)]


def test_sample_returns_requested_size():
    result = sample_examples(EXAMPLES, sample_size=3, seed=42)
    assert len(result) == 3


def test_sample_is_deterministic_for_same_seed():
    result_a = sample_examples(EXAMPLES, sample_size=4, seed=7)
    result_b = sample_examples(EXAMPLES, sample_size=4, seed=7)
    assert [e.id for e in result_a] == [e.id for e in result_b]


def test_different_seeds_can_differ():
    result_a = sample_examples(EXAMPLES, sample_size=4, seed=1)
    result_b = sample_examples(EXAMPLES, sample_size=4, seed=2)
    assert [e.id for e in result_a] != [e.id for e in result_b]


def test_sample_size_larger_than_set_returns_all():
    result = sample_examples(EXAMPLES, sample_size=100, seed=1)
    assert len(result) == len(EXAMPLES)


def test_full_run_ignores_sample_size():
    result = sample_examples(EXAMPLES, sample_size=2, seed=1, full_run=True)
    assert len(result) == len(EXAMPLES)
