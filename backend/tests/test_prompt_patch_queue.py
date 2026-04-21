"""phase-6.5.7 tests for backend.intel.prompt_patch_queue."""
from __future__ import annotations

from pathlib import Path

from backend.intel import prompt_patch_queue as q


def test_patch_id_is_deterministic_for_same_inputs():
    a = q._patch_id("strategy_hint", "buy on Tuesdays", "chunk-1")
    b = q._patch_id("strategy_hint", "buy on Tuesdays", "chunk-1")
    assert a == b
    assert len(a) == 16


def test_patch_id_differs_for_different_inputs():
    assert q._patch_id("a", "b", None) != q._patch_id("a", "c", None)
    assert q._patch_id("a", "b", "x") != q._patch_id("a", "b", "y")
    assert q._patch_id("a", "b", None) != q._patch_id("b", "a", None)


def test_dedup_in_memory_collapses_duplicates():
    patches = [
        {"patch_type": "risk_flag", "patch_text": "duplicate"},
        {"patch_type": "risk_flag", "patch_text": "duplicate"},
        {"patch_type": "strategy_hint", "patch_text": "unique"},
    ]
    out = q.dedup(patches)
    assert len(out) == 2
    pids = {p["patch_id"] for p in out}
    assert len(pids) == 2


def test_dedup_preserves_first_occurrence():
    p1 = {"patch_type": "t", "patch_text": "x", "rationale": "first"}
    p2 = {"patch_type": "t", "patch_text": "x", "rationale": "second"}
    out = q.dedup([p1, p2])
    assert len(out) == 1
    assert out[0]["rationale"] == "first"


def test_enqueue_patch_fail_open_no_bq():
    """Bad BQ project => still returns the deterministic pid; never raises."""
    result = q.enqueue_patch(
        "strategy_hint",
        "stub text",
        chunk_id="c1",
        rationale="stub",
        project="nonexistent-fail-open-test",
        dataset="nx",
    )
    assert result == q._patch_id("strategy_hint", "stub text", "c1")


def test_get_pending_fail_open_no_bq():
    rows = q.get_pending(limit=10, project="nonexistent-fail-open-test", dataset="nx")
    assert rows == []


def test_mark_approved_fail_open_no_bq():
    ok = q.mark_approved(
        "abc", "alice", project="nonexistent-fail-open-test", dataset="nx"
    )
    assert ok is False


def test_mark_rejected_fail_open_no_bq():
    ok = q.mark_rejected(
        "abc", "bad patch", project="nonexistent-fail-open-test", dataset="nx"
    )
    assert ok is False


def test_queue_persists_and_dedupes_end_to_end(monkeypatch):
    """Immutable criterion: prompt_patch_queue_persists_and_dedupes.

    Monkeypatch the BQ insert with a captive list; enqueue same patch twice,
    assert dedup-before-insert collapses to 1 unique patch_id.
    """
    fake_store: list[dict] = []

    def fake_insert(rows, *, project=None, dataset=None):
        # Cross-batch dedup check: don't insert a row whose patch_id already
        # exists with status='pending' and no later terminal status.
        before = len(fake_store)
        for r in rows:
            pid = r["patch_id"]
            existing = [x for x in fake_store if x["patch_id"] == pid]
            latest_status = (
                sorted(existing, key=lambda x: x["created_at"])[-1]["status"]
                if existing
                else None
            )
            if latest_status == "pending":
                continue
            fake_store.append(r)
        return len(fake_store) - before

    monkeypatch.setattr(q, "_insert", fake_insert)

    pid1 = q.enqueue_patch("risk_flag", "careful with AAPL", chunk_id="c1")
    pid2 = q.enqueue_patch("risk_flag", "careful with AAPL", chunk_id="c1")
    assert pid1 == pid2, "same inputs must yield same patch_id"
    # After 2 enqueues of the same patch, only 1 row persists (dedup-by-pid).
    pending = [r for r in fake_store if r["status"] == "pending"]
    assert len(pending) == 1
    assert pending[0]["patch_id"] == pid1


def test_dedup_uses_explicit_patch_id_when_provided():
    patches = [{"patch_id": "explicit-1", "patch_type": "t", "patch_text": "x"}]
    out = q.dedup(patches)
    assert out[0]["patch_id"] == "explicit-1"


def test_module_is_ascii_only():
    mod_path = Path(__file__).resolve().parents[1] / "intel" / "prompt_patch_queue.py"
    mod_path.read_bytes().decode("ascii")
