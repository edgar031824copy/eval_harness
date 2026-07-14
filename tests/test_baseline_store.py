import json
from eval_harness.baseline_store import BaselineStore


def test_append_run_creates_new_file(tmp_path):
    store_path = str(tmp_path / "baseline.json")
    store = BaselineStore()
    store.append_run(store_path, "faq-demo-agent", mean_score=0.9)

    data = json.loads(open(store_path).read())
    assert data["faq-demo-agent"] == [0.9]


def test_append_run_accumulates_history(tmp_path):
    store_path = str(tmp_path / "baseline.json")
    store = BaselineStore()
    store.append_run(store_path, "faq-demo-agent", mean_score=0.9)
    store.append_run(store_path, "faq-demo-agent", mean_score=0.85)

    data = store.load(store_path)
    assert data["faq-demo-agent"] == [0.9, 0.85]


def test_append_run_caps_history_at_max():
    import tempfile, os
    fd, store_path = tempfile.mkstemp()
    os.close(fd)
    os.remove(store_path)

    store = BaselineStore()
    for score in [0.1, 0.2, 0.3, 0.4]:
        store.append_run(store_path, "agent-x", mean_score=score, max_history=3)

    data = store.load(store_path)
    assert data["agent-x"] == [0.2, 0.3, 0.4]  # oldest (0.1) evicted
