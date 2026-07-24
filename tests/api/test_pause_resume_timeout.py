"""phase-23.1.20: pause/resume + kill-switch timeout hardening regression."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from backend.services import kill_switch


@pytest.fixture(autouse=True)
def _isolated_kill_switch_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """phase-23.2.22: redirect kill_switch._AUDIT_PATH to tmp so this file's
    `pause_trading` / `resume_trading` calls don't write real pause events to
    production handoff/kill_switch_audit.jsonl. test_pause_unaffected_no_bq_call
    in particular invokes the live pause endpoint which goes through the
    module-level _state singleton."""
    p = tmp_path / "kill_switch_audit.jsonl"
    monkeypatch.setattr(kill_switch, "_AUDIT_PATH", p)
    return p


def _slow_get_paper_portfolio(_pid: str = "default"):
    """Simulate a hung BQ — sleeps a bit past the 5s endpoint timeout. Kept
    short so asyncio.run's loop-shutdown wait (which blocks on the lingering
    threadpool thread until time.sleep returns) doesn't dominate the test
    runtime. In production the response is sent at the 5s timeout regardless
    of when the thread actually finishes — the lingering thread just
    occupies a slot until BQ eventually returns or the worker dies."""
    time.sleep(6)
    return {"total_nav": 1.0, "starting_capital": 1.0}


def test_resume_returns_503_when_bq_hangs():
    """Fix A: resume_trading must return 503 within ~5-6s when BQ hangs,
    not a 30s frontend AbortController hang."""
    from backend.api.paper_trading import resume_trading, KillSwitchActionRequest

    req = KillSwitchActionRequest(confirmation="RESUME")

    # phase-75.15 fix: resume_trading calls get_bq_client() (an @lru_cache
    # singleton factory, backend/db/bigquery_client.py:1116, added
    # phase-75.9), not `BigQueryClient(...)` directly -- patching the class
    # was a no-op post-75.9 and let the real cached client's live BQ data
    # leak through (verify_phase_23_1_22.py started exit=1'ing silently).
    # Patch the actual call site's name instead.
    with patch(
        "backend.api.paper_trading.get_bq_client",
        return_value=SimpleNamespace(get_paper_portfolio=_slow_get_paper_portfolio),
    ):
        t0 = time.monotonic()
        resp = asyncio.run(resume_trading(req))
        elapsed = time.monotonic() - t0

    # 5s timeout + ~1s coroutine overhead floor — must fire well under 30s.
    # Test harness asyncio.run() waits for the threadpool thread to finish
    # at loop shutdown (the 6s time.sleep inside _slow_get_paper_portfolio).
    # In production the response is released at 5s; lingering thread does
    # not block the long-lived uvicorn loop.
    assert elapsed < 9.0, f"resume_trading took {elapsed:.2f}s (expected <9s incl. test-harness shutdown)"
    # FastAPI JSONResponse object — inspect status_code + headers.
    assert getattr(resp, "status_code", None) == 503, \
        f"expected 503, got {getattr(resp, 'status_code', None)}"
    assert resp.headers.get("Retry-After") == "5", \
        f"expected Retry-After: 5; got {resp.headers.get('Retry-After')}"


def test_kill_switch_status_degrades_gracefully_when_bq_hangs():
    """Fix A2: kill-switch GET must still return a useful payload when BQ
    hangs (rather than a 30s hang). Pause flag + thresholds come from
    in-memory state, so the response is informative even with portfolio=None."""
    from backend.api.paper_trading import get_kill_switch_state

    # phase-75.15 fix: see test_resume_returns_503_when_bq_hangs -- same
    # stale patch target (get_bq_client() singleton, not the BigQueryClient
    # class, is what get_kill_switch_state actually calls).
    with patch(
        "backend.api.paper_trading.get_bq_client",
        return_value=SimpleNamespace(get_paper_portfolio=_slow_get_paper_portfolio),
    ):
        t0 = time.monotonic()
        body = asyncio.run(get_kill_switch_state())
        elapsed = time.monotonic() - t0

    assert elapsed < 8.0, f"kill-switch GET took {elapsed:.2f}s (expected <8s)"
    assert "paused" in body and "thresholds" in body, \
        "kill-switch GET must still return paused + thresholds even on BQ timeout"
    assert body["current_nav"] == 0.0, \
        "BQ timeout -> portfolio=None -> nav defaults to 0.0"


def test_pause_unaffected_no_bq_call():
    """Fix B (deferred): pause endpoint has no BQ I/O, doesn't need timeout."""
    from backend.api.paper_trading import pause_trading, KillSwitchActionRequest

    req = KillSwitchActionRequest(confirmation="PAUSE")
    t0 = time.monotonic()
    body = asyncio.run(pause_trading(req))
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"pause should be near-instant; took {elapsed:.2f}s"
    assert body["status"] == "paused"
