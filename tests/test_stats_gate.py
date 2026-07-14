from eval_harness.stats_gate import evaluate_gate


def test_no_regression_when_scores_are_similar():
    new_scores = [0.90, 0.91, 0.89, 0.92, 0.90]
    baseline_scores = [0.90, 0.89, 0.91, 0.90, 0.90, 0.91, 0.89]

    decision = evaluate_gate(new_scores, baseline_scores)

    assert decision.is_regression is False


def test_regression_when_scores_drop_significantly():
    new_scores = [0.50, 0.48, 0.52, 0.49, 0.51, 0.50, 0.49]
    baseline_scores = [0.90, 0.89, 0.91, 0.90, 0.92, 0.88, 0.91]

    decision = evaluate_gate(new_scores, baseline_scores)

    assert decision.is_regression is True
    assert decision.p_value < 0.05
    assert decision.effect_size > 0.03


def test_small_noisy_drop_below_effect_size_floor_is_not_regression():
    # Drop exists but is within the noise floor (effect_size < 0.03) even if
    # it happens to be statistically significant with a huge sample.
    new_scores = [0.881] * 50
    baseline_scores = [0.90] * 50

    decision = evaluate_gate(new_scores, baseline_scores, effect_size_floor=0.03)

    assert decision.is_regression is False
    assert "effect size" in decision.reason.lower()


def test_insufficient_baseline_history_does_not_crash():
    new_scores = [0.5, 0.5]
    baseline_scores = []  # no history yet — first run for this prompt

    decision = evaluate_gate(new_scores, baseline_scores)

    assert decision.is_regression is False
    assert "insufficient baseline" in decision.reason.lower()
