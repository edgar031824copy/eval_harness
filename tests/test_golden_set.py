import pytest
from eval_harness.golden_set import load_golden_set, GoldenSetError

VALID_YAML = """
name: test-agent
examples:
  - id: ex1
    input: "What is 2+2?"
    expected: "4"
  - id: ex2
    input: "Capital of France?"
    expected: "Paris"
"""

MISSING_FIELD_YAML = """
name: test-agent
examples:
  - id: ex1
    input: "What is 2+2?"
"""


def test_load_valid_golden_set(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text(VALID_YAML)

    golden_set = load_golden_set(str(p))

    assert golden_set.name == "test-agent"
    assert len(golden_set.examples) == 2
    assert golden_set.examples[0].id == "ex1"
    assert golden_set.examples[0].input == "What is 2+2?"
    assert golden_set.examples[0].expected == "4"


def test_load_rejects_missing_expected_field(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text(MISSING_FIELD_YAML)

    with pytest.raises(GoldenSetError, match="expected"):
        load_golden_set(str(p))


def test_load_rejects_empty_examples(tmp_path):
    p = tmp_path / "golden.yaml"
    p.write_text("name: empty-agent\nexamples: []\n")

    with pytest.raises(GoldenSetError, match="at least one example"):
        load_golden_set(str(p))
