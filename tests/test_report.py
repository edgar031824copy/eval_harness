from eval_harness.report import build_report, render_markdown
from eval_harness.judge import JudgeVerdict
from eval_harness.stats_gate import GateDecision


def test_build_report_computes_mean_score():
    verdicts = [
        JudgeVerdict(score=0.8, reasoning="fine", input_tokens=10, output_tokens=5),
        JudgeVerdict(score=0.6, reasoning="ok", input_tokens=10, output_tokens=5),
    ]
    gate_decision = GateDecision(is_regression=False, p_value=0.5, effect_size=0.01, reason="no regression")
    cost_breakdown = {"replay_cost_usd": 0.001, "judge_cost_usd": 0.01, "embedding_calls": 2, "total_cost_usd": 0.011}

    report = build_report(
        prompt_name="faq-demo-agent",
        example_ids=["ex1", "ex2"],
        verdicts=verdicts,
        gate_decision=gate_decision,
        cost_breakdown=cost_breakdown,
    )

    assert report.prompt_name == "faq-demo-agent"
    assert report.sample_size == 2
    assert report.mean_score == 0.7
    assert report.gate_decision.is_regression is False
    assert report.cost_breakdown["total_cost_usd"] == 0.011


def test_render_markdown_includes_pass_and_cost():
    verdicts = [JudgeVerdict(score=0.9, reasoning="great", input_tokens=1, output_tokens=1)]
    gate_decision = GateDecision(is_regression=False, p_value=0.8, effect_size=0.01, reason="no statistically significant regression (p=0.8000)")
    cost_breakdown = {"replay_cost_usd": 0.002, "judge_cost_usd": 0.02, "embedding_calls": 1, "total_cost_usd": 0.022}

    report = build_report("faq-demo-agent", ["ex1"], verdicts, gate_decision, cost_breakdown)
    markdown = render_markdown(report)

    assert "faq-demo-agent" in markdown
    assert "PASS" in markdown
    assert "$0.022" in markdown
    assert "0.90" in markdown


def test_render_markdown_shows_block_on_regression():
    verdicts = [JudgeVerdict(score=0.3, reasoning="bad", input_tokens=1, output_tokens=1)]
    gate_decision = GateDecision(is_regression=True, p_value=0.01, effect_size=0.4, reason="statistically significant drop (p=0.0100, effect size=0.4000 >= floor 0.03)")
    cost_breakdown = {"replay_cost_usd": 0.002, "judge_cost_usd": 0.02, "embedding_calls": 0, "total_cost_usd": 0.022}

    report = build_report("faq-demo-agent", ["ex1"], verdicts, gate_decision, cost_breakdown)
    markdown = render_markdown(report)

    assert "BLOCKED" in markdown
    assert "statistically significant drop" in markdown
