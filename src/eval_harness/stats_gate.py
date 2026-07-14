from dataclasses import dataclass

from scipy import stats


# NOTE: this gate design (Welch's t-test + effect-size floor vs. a rolling
# baseline, rather than a bare absolute-threshold cutoff) is inspired by a
# single 2026 vendor blog post (FutureAGI) — per LIT-39 research, no major
# eval tool (promptfoo/DeepEval/Braintrust/LangSmith) implements this. It is
# the PoC's genuine statistical-rigor differentiator, not an established
# industry practice — say so in the demo and article, don't overclaim.


@dataclass
class GateDecision:
    is_regression: bool
    p_value: float
    effect_size: float
    reason: str


MIN_BASELINE_SIZE = 3


def evaluate_gate(
    new_scores: list[float],
    baseline_scores: list[float],
    effect_size_floor: float = 0.03,
    p_threshold: float = 0.05,
) -> GateDecision:
    if len(baseline_scores) < MIN_BASELINE_SIZE:
        return GateDecision(
            is_regression=False,
            p_value=1.0,
            effect_size=0.0,
            reason="insufficient baseline history — treating as pass until enough runs accumulate",
        )

    t_stat, p_value = stats.ttest_ind(new_scores, baseline_scores, equal_var=False)
    mean_diff = sum(baseline_scores) / len(baseline_scores) - sum(new_scores) / len(new_scores)
    effect_size = abs(mean_diff)  # normalized 0-1 score scale, so |mean diff| doubles as the effect size

    if p_value < p_threshold and effect_size >= effect_size_floor and mean_diff > 0:
        return GateDecision(
            is_regression=True,
            p_value=p_value,
            effect_size=effect_size,
            reason=f"statistically significant drop (p={p_value:.4f}, effect size={effect_size:.4f} >= floor {effect_size_floor})",
        )

    if p_value < p_threshold and mean_diff > 0:
        return GateDecision(
            is_regression=False,
            p_value=p_value,
            effect_size=effect_size,
            reason=f"drop is statistically significant but below the effect size floor ({effect_size:.4f} < {effect_size_floor}) — treated as noise",
        )

    return GateDecision(
        is_regression=False,
        p_value=p_value,
        effect_size=effect_size,
        reason=f"no statistically significant regression (p={p_value:.4f})",
    )
