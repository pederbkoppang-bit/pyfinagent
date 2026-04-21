"""phase-9.3 tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.jobs.weekly_fred_refresh import run, JOB_NAME
from backend.slack_bot.job_runtime import IdempotencyStore, IdempotencyKey


def test_run_writes_via_injected_fns():
    store = IdempotencyStore()
    out = run(
        series=["DGS10"],
        fetch_fn=lambda s: {k: [1, 2, 3] for k in s},
        write_fn=lambda r: len(r),
        store=store,
        iso_year_week="2026-W17",
    )
    assert out["written"] == 1
    assert not out["skipped"]


def test_idempotency_by_iso_week():
    store = IdempotencyStore()
    key = IdempotencyKey.weekly(JOB_NAME, iso_year_week="2026-W17")
    run(series=["DGS10"], fetch_fn=lambda s: {}, write_fn=lambda r: 0, store=store, iso_year_week="2026-W17")
    assert store.seen(key)
    out2 = run(series=["DGS10"], fetch_fn=lambda s: {"x": 1}, write_fn=lambda r: 99, store=store, iso_year_week="2026-W17")
    assert out2["skipped"] is True


def test_no_live_fredapi(monkeypatch):
    import sys as _s
    _s.modules.pop("fredapi", None)
    store = IdempotencyStore()
    run(series=["DGS10"], fetch_fn=lambda s: {}, write_fn=lambda r: 0, store=store, iso_year_week="2026-W17")
    assert "fredapi" not in _s.modules
