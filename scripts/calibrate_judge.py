from dataclasses import dataclass, field

from eval_harness.judge import judge_pair as default_judge_pair

# Human verdict -> expected score band. "better"/"equivalent" both map to >0.6
# because the rubric treats "equally good" as a pass, matching promptfoo's
# pass/score/reason semantics rather than requiring strict improvement.
VERDICT_BANDS = {
    "better": lambda score: score > 0.6,
    "equivalent": lambda score: score > 0.6,
    "worse": lambda score: score <= 0.6,
}


@dataclass
class CalibrationReport:
    total: int
    agreements: int
    agreement_rate: float
    disagreements: list = field(default_factory=list)


def run_calibration(client, examples: list[dict], judge_fn=default_judge_pair) -> CalibrationReport:
    agreements = 0
    disagreements = []

    for ex in examples:
        verdict = judge_fn(
            client,
            task_input=ex["task_input"],
            expected=ex["expected"],
            old_output=ex["old_output"],
            new_output=ex["new_output"],
        )
        check = VERDICT_BANDS[ex["human_verdict"]]
        if check(verdict.score):
            agreements += 1
        else:
            disagreements.append({**ex, "judge_score": verdict.score})

    total = len(examples)
    return CalibrationReport(
        total=total,
        agreements=agreements,
        agreement_rate=agreements / total if total else 0.0,
        disagreements=disagreements,
    )


if __name__ == "__main__":
    import yaml
    from eval_harness.anthropic_client import AnthropicClient
    import os

    with open("calibration/judge_calibration_set.yaml") as f:
        examples = yaml.safe_load(f)["examples"]

    client = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    report = run_calibration(client, examples)
    print(f"Agreement rate: {report.agreement_rate:.1%} ({report.agreements}/{report.total})")
    if report.agreement_rate < 0.9:
        print("WARNING: below 90% target — review disagreements before trusting judge in CI:")
        for d in report.disagreements:
            print(f"  - {d['task_input']}: human={d['human_verdict']}, judge_score={d['judge_score']}")
