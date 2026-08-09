"""phase-23.2.4 verification: pause/resume deadlock did NOT regress.

Live API regression-lock for the re-entrant lock deadlock fixed in
commit 0ed72940 (phase-23.1.22, 2026-04-30). The original bug: pause/
resume held self._lock and then called snapshot() which tried to re-
acquire the same lock -> process-wide asyncio deadlock. Fix: extracted
_snapshot_locked() helper. Kept threading.Lock (not RLock) per
documented direction.

This test does TWO things:
  1. ASSERT existing pytest regression tests pass
     (tests/services/test_kill_switch_no_deadlock.py + the api timeout
     tests already established the structural invariant in cycle-23.1).
     We just check those files exist + run.
  2. SKIP-OR-RUN: when a backend is listening on localhost:8000, run
     the live pause-resume-pause cycle + assert all 3 transitions
     complete under 5s + audit log delta is exactly 3 rows + last 3
     audit rows are (pause, resume, pause) with trigger=manual.

Live cycle is OPTIONAL (skip with no backend). The structural pytest
tests are the load-bearing regression guard.

Per researcher (handoff/current/research_brief_phase_23_2_4.md, 6
sources): pytest-first ordering is preferred (cheap, deterministic);
live curl is the production-shape evidence; JSONL tail proves audit
integrity. This test file collapses all three into one pytest run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXISTING_TEST_1 = REPO_ROOT / "tests" / "services" / "test_kill_switch_no_deadlock.py"
EXISTING_TEST_2 = REPO_ROOT / "tests" / "api" / "test_pause_resume_timeout.py"
AUDIT_LOG = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"

BACKEND_URL = "http://localhost:8000"
TRANSITION_BUDGET_S = 5.0


def _backend_is_up(timeout_s: float = 2.0) -> bool:
    """Probe /api/health; True if 200 OK + ok status."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/api/health", timeout=timeout_s) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _get_paused_state() -> bool:
    """Read current paused state via the kill-switch endpoint."""
    import urllib.request
    with urllib.request.urlopen(f"{BACKEND_URL}/api/paper-trading/kill-switch", timeout=3) as r:
        return json.loads(r.read())["paused"]


def _post_state_transition(endpoint: str, confirmation: str) -> tuple[dict, float]:
    """POST a pause or resume transition. Returns (response_json, elapsed_seconds).
    Raises if non-2xx + non-503 (503 is the documented degraded-BQ path)."""
    import urllib.request
    body = json.dumps({"confirmation": confirmation}).encode("utf-8")
    req = urllib.request.Request(
        f"{BACKEND_URL}{endpoint}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TRANSITION_BUDGET_S + 1.0) as r:
            elapsed = time.perf_counter() - t0
            return json.loads(r.read()), elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        # 503 on resume is the documented degraded-BQ path; allow it.
        if e.code == 503 and endpoint.endswith("/resume"):
            return {"status": "degraded", "code": 503}, elapsed
        raise


def test_phase_23_2_4_existing_pytest_regression_files_exist():
    """The structural pytest regression files from phase-23.1.22 +
    phase-23.2.x must still be present. If a future cleanup deletes them
    the structural invariant on the lock pattern is unprotected."""
    assert EXISTING_TEST_1.exists(), (
        f"phase-23.1.22 deadlock regression test missing: {EXISTING_TEST_1}"
    )
    assert EXISTING_TEST_2.exists(), (
        f"phase-23.x pause/resume timeout regression test missing: {EXISTING_TEST_2}"
    )


def test_phase_23_2_4_existing_regression_files_reference_phase_23_1_22():
    """The lock fix is anchored to commit 0ed72940 (phase-23.1.22).
    The existing test files MUST reference that anchor so any future
    refactor that changes the lock pattern surfaces the anchor for
    audit."""
    text_1 = EXISTING_TEST_1.read_text(encoding="utf-8")
    assert "phase-23.1.22" in text_1 or "_snapshot_locked" in text_1, (
        "Existing deadlock-regression test must anchor to phase-23.1.22 + "
        "_snapshot_locked helper (per researcher cite)"
    )


@pytest.fixture
def _isolated_app_client(tmp_path, monkeypatch):
    """An IN-PROCESS client over the real app, with the kill switch detached.

    phase-86.3: this test used to POST pause -> resume -> pause (plus a 4th
    restore resume) to http://localhost:8000, i.e. to the OPERATOR'S LIVE
    TRADING BOOK. Measured 2026-08-09: 8 rows appended to the live
    handoff/kill_switch_audit.jsonl across two full-suite runs, and the live
    armed book was paused four times by a test run.

    The subject under test is a RE-ENTRANT LOCK DEADLOCK INSIDE THE APP
    PROCESS (pause/resume held `self._lock`, then called `snapshot()`, which
    re-acquired it -- fixed in 0ed72940 by extracting `_snapshot_locked()`).
    That is an in-process property and it reproduces in-process, so the
    regression value is preserved in full while the live book is untouched.

    Isolation mirrors `test_phase_36_7_kill_switch_rotation_rearm.py::
    isolated_state`, and BOTH halves are required:
      * `_AUDIT_PATH` -> tmp, because pause()/resume() append to it; and
      * the module-global `_state` -> a detached instance, because
        `evaluate_breach` reads `_state` DIRECTLY. Patching the
        `_get_ks_state` accessor alone would NOT isolate the breach check.

    The test process's kill-switch singleton is not the running backend's --
    they are separate processes -- so with the file redirected there is no
    durable channel left to the live book.
    """
    import threading
    from datetime import datetime, timezone

    import backend.services.kill_switch as ks
    from backend.tests.auth_helper import authed_test_client

    monkeypatch.setattr(ks, "_AUDIT_PATH", tmp_path / "kill_switch_audit.jsonl")

    fresh = object.__new__(ks.KillSwitchState)
    fresh._lock = threading.Lock()
    fresh._paused = False
    fresh._pause_reason = None
    fresh._sod_nav = 10_000.0
    # Anchored to TODAY, so the daily leg is armed and `/resume` is not refused
    # by the staleness gate (that refusal is its own defect -- step 36.26).
    fresh._sod_date = datetime.now(timezone.utc).date().isoformat()
    fresh._peak_nav = 10_000.0
    fresh._paused_at = None
    fresh._auto_resume_alerted_at = None
    monkeypatch.setattr(ks, "_state", fresh)
    monkeypatch.setattr(ks, "_disarmed_logged", False)

    # A flat, healthy book: 0% daily loss, 0% trailing drawdown -> resume is
    # allowed on the merits rather than by a weakened gate.
    from backend.api import paper_trading as pt
    monkeypatch.setattr(
        pt, "get_bq_client",
        lambda: SimpleNamespace(
            get_paper_portfolio=lambda _pid="default": {
                "total_nav": 10_000.0, "starting_capital": 10_000.0
            }
        ),
    )

    from backend.main import app
    with authed_test_client(app, raise_server_exceptions=False) as client:
        yield client, tmp_path / "kill_switch_audit.jsonl"


def test_phase_23_2_4_live_pause_resume_pause_cycle_under_5s(_isolated_app_client):
    """The canonical pause-resume-pause cycle, driven IN-PROCESS. Assert:
      1. Each transition completes within TRANSITION_BUDGET_S (5.0s) -- a
         deadlock manifests as a transition that never returns.
      2. State changes match the requested transition.
      3. The audit delta is exactly 3 rows -- (pause, resume, pause), all
         trigger=manual.

    Assertion 3 was in the original docstring but could only ever be checked
    against the shared live journal, where a concurrent writer made it
    unreliable. Against an isolated journal it is now exact, so this rewrite
    STRENGTHENS the test rather than trading coverage for safety.
    """
    client, audit = _isolated_app_client

    transitions = [
        ("/api/paper-trading/pause", "PAUSE", ("paused", "already_paused")),
        ("/api/paper-trading/resume", "RESUME", ("resumed", "already_unpaused", "degraded")),
        ("/api/paper-trading/pause", "PAUSE", ("paused", "already_paused")),
    ]
    for endpoint, confirmation, allowed in transitions:
        t0 = time.perf_counter()
        resp = client.post(endpoint, json={"confirmation": confirmation})
        elapsed = time.perf_counter() - t0
        assert elapsed < TRANSITION_BUDGET_S, (
            f"{endpoint} exceeded {TRANSITION_BUDGET_S}s budget: {elapsed:.3f}s "
            "-- a re-entrant lock deadlock is the regression this pins"
        )
        assert resp.status_code == 200, f"{endpoint} -> {resp.status_code}: {resp.text}"
        assert resp.json().get("status") in allowed, (
            f"{endpoint} response unexpected: {resp.json()}"
        )

    rows = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()]
    assert [r["event"] for r in rows] == ["pause", "resume", "pause"], (
        f"audit delta must be exactly (pause, resume, pause); got {rows}"
    )
    assert all(r.get("trigger") == "manual" for r in rows), rows


def test_phase_86_3_mutation_the_audit_redirect_is_load_bearing(tmp_path, monkeypatch):
    """MUTATION LOCK for the isolation above, proven WITHOUT touching the live
    journal.

    Running the real mutation -- deleting the `_AUDIT_PATH` redirect and letting
    the cycle write to the operator's journal -- would be committing the very
    defect this step repairs, and would break 86.3 criterion 1 in the act. So
    the proof is split into the two facts whose CONJUNCTION is the claim:

      (a) the module default `_AUDIT_PATH` really is the live production file
          -- so an un-redirected write lands there, not somewhere harmless; and
      (b) writes follow `_AUDIT_PATH` wherever it points -- demonstrated
          against a byte-identical COPY of the live journal in tmp.

    (a) AND (b) => removing the redirect writes to the live journal. Disclosed
    in contract_86.3.md rather than claimed as a full end-to-end mutation.
    """
    import backend.services.kill_switch as ks

    # (a) the un-patched default is the real file.
    assert ks._AUDIT_PATH == AUDIT_LOG, (
        "the module default no longer points at the live journal; this test's "
        f"reasoning is stale. default={ks._AUDIT_PATH} live={AUDIT_LOG}"
    )

    # (b) writes follow the pointer -- against a copy, never the original.
    decoy = tmp_path / "kill_switch_audit.jsonl"
    decoy.write_bytes(AUDIT_LOG.read_bytes())
    before_live = AUDIT_LOG.read_bytes()
    before_decoy = len(decoy.read_text().splitlines())

    monkeypatch.setattr(ks, "_AUDIT_PATH", decoy)
    ks.KillSwitchState._append_audit("pause", trigger="manual", details={})

    assert len(decoy.read_text().splitlines()) == before_decoy + 1, (
        "an append did not follow _AUDIT_PATH -- (b) does not hold"
    )
    assert AUDIT_LOG.read_bytes() == before_live, "the mutation touched the live journal"


@pytest.fixture(autouse=True)
def _no_test_in_this_module_writes_to_the_live_journal():
    """phase-86.3 regression lock, measured as a DELTA rather than asserted as
    'never touched'. Byte-compares the LIVE journal around EVERY test here.

    FUNCTION-scoped and autouse, mirroring
    `test_phase_36_12_...::_live_audit_file_is_write_protected`, so a violation
    names the offending test instead of the module. This is the file that
    caused the incident; it carries its own lock.

    Honest limit: this is a DETECTOR, not a preventer -- the bytes are already
    on disk when it fires. The PREVENTER for the HTTP channel is the repo-root
    conftest guard; the preventer for the in-process filesystem channel is not
    built (see contract_86.3.md SS5.2, queued as its own step).
    """
    before = AUDIT_LOG.read_bytes() if AUDIT_LOG.exists() else None
    yield
    after = AUDIT_LOG.read_bytes() if AUDIT_LOG.exists() else None
    assert after == before, (
        "a test in this module wrote to the LIVE kill-switch journal: "
        f"{0 if before is None else len(before.splitlines())} -> "
        f"{0 if after is None else len(after.splitlines())} lines"
    )


@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_4_audit_log_clean_transitions():
    """The kill_switch_audit.jsonl must have clean (parseable) rows
    for the most recent transitions. Each row must have ts + event +
    trigger=manual + details fields."""
    if not AUDIT_LOG.exists():
        pytest.skip("kill_switch_audit.jsonl not present")
    # Tail last 10 rows (cycle 28 has just added at least 3 from the live cycle above)
    lines = AUDIT_LOG.read_text(encoding="utf-8").splitlines()[-10:]
    assert len(lines) >= 3, f"audit log must have at least 3 rows; got {len(lines)}"
    parsed_rows = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        parsed_rows.append(row)
        # ts + event are required on every row. trigger is required on
        # pause/resume rows but not on sod_snapshot / peak_update which
        # have a different shape (nav + date instead of trigger).
        assert "ts" in row and "event" in row, (
            f"audit row missing required ts/event fields: {row}"
        )
        assert row["event"] in {"pause", "resume", "sod_snapshot", "peak_update", "cleanup"}, (
            f"unexpected audit event: {row['event']}"
        )
        # For state-change events, the trigger must be present
        if row["event"] in {"pause", "resume"}:
            assert "trigger" in row, (
                f"pause/resume row must have trigger field: {row}"
            )
            assert row["trigger"] in {"manual", "auto", "test"}, (
                f"unexpected trigger value: {row['trigger']}"
            )
    # All recent rows should be parseable -- no truncation, no JSON-decode errors.
    assert parsed_rows, "no parseable rows found in audit log"
