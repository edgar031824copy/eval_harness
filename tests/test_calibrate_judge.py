from unittest.mock import MagicMock
from scripts.calibrate_judge import run_calibration
from eval_harness.judge import JudgeVerdict


def test_run_calibration_computes_agreement_rate():
    fake_client = MagicMock()
    examples = [
        {
            "task_input": "q1", "expected": "e1", "old_output": "o1", "new_output": "n1",
            "human_verdict": "better",
        },
        {
            "task_input": "q2", "expected": "e2", "old_output": "o2", "new_output": "n2",
            "human_verdict": "worse",
        },
    ]

    def fake_judge_pair(client, task_input, expected, old_output, new_output, model="claude-opus-4-8"):
        # Simulate judge agreeing on example 1 (score 0.8 -> "better"), disagreeing on example 2
        score = 0.8 if task_input == "q1" else 0.8
        return JudgeVerdict(score=score, reasoning="stub", input_tokens=1, output_tokens=1)

    report = run_calibration(fake_client, examples, judge_fn=fake_judge_pair)

    assert report.total == 2
    assert report.agreements == 1  # only q1 agrees ("better" <-> score 0.8 > 0.6)
    assert report.agreement_rate == 0.5
    assert len(report.disagreements) == 1
    assert report.disagreements[0]["task_input"] == "q2"
