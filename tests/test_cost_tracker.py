from eval_harness.cost_tracker import CostTracker, PRICING


def test_add_replay_call_accumulates_cost():
    tracker = CostTracker()
    tracker.add_replay_call(model="claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)

    expected = PRICING["claude-sonnet-5"]["input"] + PRICING["claude-sonnet-5"]["output"]
    assert tracker.total_cost_usd == expected


def test_add_judge_call_accumulates_separately_from_replay():
    tracker = CostTracker()
    tracker.add_replay_call(model="claude-sonnet-5", input_tokens=500_000, output_tokens=0)
    tracker.add_judge_call(model="claude-opus-4-8", input_tokens=500_000, output_tokens=0)

    breakdown = tracker.breakdown()
    assert breakdown["replay_cost_usd"] == PRICING["claude-sonnet-5"]["input"] * 0.5
    assert breakdown["judge_cost_usd"] == PRICING["claude-opus-4-8"]["input"] * 0.5
    assert tracker.total_cost_usd == breakdown["replay_cost_usd"] + breakdown["judge_cost_usd"]


def test_embedding_calls_are_free_but_counted():
    tracker = CostTracker()
    tracker.add_embedding_call()
    tracker.add_embedding_call()

    assert tracker.total_cost_usd == 0.0
    assert tracker.breakdown()["embedding_calls"] == 2


def test_unknown_model_raises():
    tracker = CostTracker()
    import pytest
    with pytest.raises(KeyError):
        tracker.add_replay_call(model="not-a-real-model", input_tokens=1, output_tokens=1)
