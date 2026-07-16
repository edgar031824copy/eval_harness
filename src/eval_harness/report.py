from dataclasses import dataclass, field

from eval_harness.judge import JudgeVerdict
from eval_harness.stats_gate import GateDecision


@dataclass
class RunReport:
    prompt_name: str
    sample_size: int
    mean_score: float
    gate_decision: GateDecision
    cost_breakdown: dict
    per_example: list = field(default_factory=list)


def build_report(
    prompt_name: str,
    example_ids: list[str],
    verdicts: list[JudgeVerdict],
    gate_decision: GateDecision,
    cost_breakdown: dict,
) -> RunReport:
    mean_score = sum(v.score for v in verdicts) / len(verdicts) if verdicts else 0.0
    per_example = [
        {"example_id": eid, "score": v.score, "reasoning": v.reasoning}
        for eid, v in zip(example_ids, verdicts)
    ]
    return RunReport(
        prompt_name=prompt_name,
        sample_size=len(verdicts),
        mean_score=round(mean_score, 4),
        gate_decision=gate_decision,
        cost_breakdown=cost_breakdown,
        per_example=per_example,
    )


def render_markdown(report: RunReport) -> str:
    status = "BLOCKED" if report.gate_decision.is_regression else "PASS"
    lines = [
        f"## Eval Harness Report — `{report.prompt_name}`",
        "",
        f"**Status: {status}**",
        "",
        f"- Sample size: {report.sample_size}",
        f"- Mean semantic score: {report.mean_score:.2f}",
        f"- Gate reasoning: {report.gate_decision.reason}",
        f"- Cost for this PR: **${report.cost_breakdown['total_cost_usd']:.3f}** "
        f"(replay ${report.cost_breakdown['replay_cost_usd']:.4f} + "
        f"judge ${report.cost_breakdown['judge_cost_usd']:.4f}, "
        f"{report.cost_breakdown['embedding_calls']} free embedding pre-filter calls)",
        "",
        "| Example | Score | Judge reasoning |",
        "|---|---|---|",
    ]
    for ex in report.per_example:
        lines.append(f"| {ex['example_id']} | {ex['score']:.2f} | {ex['reasoning']} |")
    return "\n".join(lines)
