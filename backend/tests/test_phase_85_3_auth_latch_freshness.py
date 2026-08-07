"""phase-85.3: auth-latch freshness criteria tests.

Drives scripts/away_ops/auth_state.py via subprocess over SYNTHETIC state
under tmp_path (criterion 2 -- none of the real handoff/away_ops/ artifacts
are read; the seam takes --ops/--state explicitly and has no repo-path
fallback). `now` is INJECTED via --now so the 27-day regression case is
built now-relative and can never drift (the single most important fixture
decision per the research brief). mtimes are set explicitly with os.utime
because mtime is BOTH the newest-file selector and the age input.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import sys

SEAM = str(pathlib.Path(__file__).resolve().parents[2] / "scripts/away_ops/auth_state.py")
NOW = datetime.datetime(2026, 8, 7, 12, 0, 0, tzinfo=datetime.timezone.utc)

# Verbatim-shaped 401 session payload (both JSON spellings covered across
# fixtures; "subtype":"success" included so a future key-on-subtype
# regression is caught -- research brief section 1.4).
SESSION_401_SPACED = '{"subtype":"success", "api_error_status": 401, "note": "x"}'
SESSION_401_TIGHT = '{"subtype":"success","api_error_status":401}'
SESSION_CLEAN = '{"subtype":"success", "result": "pong"}'


def _mk_session(ops: pathlib.Path, name: str, body: str, age: datetime.timedelta):
    f = ops / name
    f.write_text(body)
    t = (NOW - age).timestamp()
    os.utime(f, (t, t))
    return f


def _run(ops, state, status_ok="true", apply=False, window_s=None, now=NOW):
    cmd = [sys.executable, SEAM, "--ops", str(ops), "--state", str(state),
           "--status-ok", status_ok, "--now", now.isoformat()]
    if apply:
        cmd.append("--apply")
    if window_s is not None:
        cmd += ["--window-s", str(window_s)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    out = p.stdout.strip().split(None, 1)
    return out[0], (out[1] if len(out) > 1 else "")


def _ops(tmp_path) -> tuple[pathlib.Path, pathlib.Path]:
    ops = tmp_path / "ops"
    ops.mkdir()
    return ops, ops / "auth_page_state.json"


# ── C3: the four fixture cases ───────────────────────────────────────


def test_c3_stale_401_older_than_window_yields_true(tmp_path):
    ops, state = _ops(tmp_path)
    # 3 session files with DISTINCT mtimes so the newest-selection is real
    _mk_session(ops, "session_old1.json", SESSION_CLEAN, datetime.timedelta(days=40))
    _mk_session(ops, "session_old2.json", SESSION_CLEAN, datetime.timedelta(days=35))
    _mk_session(ops, "session_am_stale401.json", SESSION_401_SPACED, datetime.timedelta(days=27))
    ok, detail = _run(ops, state)
    assert ok == "true", f"stale 401 held the latch: {detail}"
    assert detail.startswith("stale_401_ignored_")


def test_c3_fresh_401_within_window_yields_false(tmp_path):
    ops, state = _ops(tmp_path)
    _mk_session(ops, "session_old.json", SESSION_CLEAN, datetime.timedelta(days=3))
    _mk_session(ops, "session_fresh401.json", SESSION_401_TIGHT, datetime.timedelta(hours=2))
    ok, detail = _run(ops, state)
    assert ok == "false"
    assert detail == "401_in_session_fresh401.json"


def test_c3_fresh_clean_yields_true(tmp_path):
    ops, state = _ops(tmp_path)
    _mk_session(ops, "session_clean.json", SESSION_CLEAN, datetime.timedelta(hours=1))
    ok, detail = _run(ops, state)
    assert (ok, detail) == ("true", "ok")


def test_c3_probe_error_yields_unknown_fail_open(tmp_path):
    # the caller's `|| echo "unknown probe_error"` wrapper is the outer
    # fail-open; the seam's own crash guard is exercised by feeding an
    # unparseable --now (argparse/fromisoformat error path).
    ops, state = _ops(tmp_path)
    p = subprocess.run(
        [sys.executable, SEAM, "--ops", str(ops), "--state", str(state),
         "--status-ok", "true", "--now", "not-a-timestamp"],
        capture_output=True, text=True,
    )
    assert p.returncode == 0, "the seam must fail OPEN (exit 0), never crash"
    assert p.stdout.strip() == "unknown probe_error"


# ── C4: today's exact stuck state, now-relative ──────────────────────


def test_c4_regression_todays_exact_state_now_yields_true(tmp_path):
    ops, state = _ops(tmp_path)
    # newest session = a 401 dated 27 days before the injected now
    _mk_session(ops, "session_old_a.json", SESSION_CLEAN, datetime.timedelta(days=40))
    _mk_session(ops, "session_old_b.json", SESSION_CLEAN, datetime.timedelta(days=30))
    _mk_session(ops, "session_am_20260710T053005Z.json", SESSION_401_SPACED,
                datetime.timedelta(days=27))
    # state file: incident_open true, NO cleared_at key at all
    state.write_text(json.dumps({
        "incident_open": True,
        "opened_at": "2026-07-10T05:30:11+00:00",
        "detail": "401_in_session_am_20260710T053005Z.json",
        "paged": True, "paged_by": "healthcheck",
    }))
    ok, detail = _run(ops, state)
    assert ok == "true", (
        f"the exact state stuck since 2026-07-10 still latches: {detail}"
    )


# ── C5: the close path needs no away session ─────────────────────────


def test_c5_apply_clears_open_incident_on_healthy_probe(tmp_path):
    ops, state = _ops(tmp_path)
    _mk_session(ops, "session_stale401.json", SESSION_401_SPACED,
                datetime.timedelta(days=27))
    state.write_text(json.dumps({"incident_open": True,
                                 "opened_at": "2026-07-10T05:30:11+00:00",
                                 "paged": True}))
    ok, _ = _run(ops, state, apply=True)
    assert ok == "true"
    st = json.loads(state.read_text())
    assert st["incident_open"] is False
    assert st["cleared_at"], "cleared_at not written"
    assert st["cleared_by"] == "healthcheck_healthy"


def test_c5_dd5_unknown_probe_must_not_clear_a_real_incident(tmp_path):
    """DD-5 fix: the pre-85.3 elif arm cleared on auth_ok=='unknown' too --
    a transient probe error could close a REAL incident. The seam's clear
    is strict-true only."""
    ops, state = _ops(tmp_path)
    state.write_text(json.dumps({"incident_open": True,
                                 "opened_at": "2026-08-07T00:00:00+00:00",
                                 "paged": True}))
    p = subprocess.run(
        [sys.executable, SEAM, "--ops", str(ops), "--state", str(state),
         "--status-ok", "true", "--now", "not-a-timestamp", "--apply"],
        capture_output=True, text=True,
    )
    assert p.returncode == 0 and p.stdout.strip() == "unknown probe_error"
    st = json.loads(state.read_text())
    assert st["incident_open"] is True, "a probe ERROR cleared a real incident"


def test_c5_fresh_401_does_not_clear(tmp_path):
    """Latch-exit guard: the clear must not fire while a FRESH 401 stands."""
    ops, state = _ops(tmp_path)
    _mk_session(ops, "session_fresh401.json", SESSION_401_SPACED,
                datetime.timedelta(hours=1))
    state.write_text(json.dumps({"incident_open": True,
                                 "opened_at": "2026-08-07T09:00:00+00:00",
                                 "paged": True}))
    ok, _ = _run(ops, state, apply=True)
    assert ok == "false"
    assert json.loads(state.read_text())["incident_open"] is True


# ── C6: fail-open on state-file damage ───────────────────────────────


def test_c6_missing_malformed_unreadable_state_fail_open(tmp_path):
    ops, state = _ops(tmp_path)
    _mk_session(ops, "session_clean.json", SESSION_CLEAN, datetime.timedelta(hours=1))
    # missing state file
    ok, _ = _run(ops, state)
    assert ok == "true"
    # malformed state file
    state.write_text("{truncated")
    ok, _ = _run(ops, state)
    assert ok == "true"
    # unreadable state file
    state.write_text("{}")
    os.chmod(state, 0o000)
    try:
        ok, _ = _run(ops, state)
    finally:
        os.chmod(state, 0o644)
    assert ok == "true", "an unreadable state file manufactured a failure"


# ── cycle-2 guards (Q/A wf_9bdd4eb6-03d) ─────────────────────────────


def _load_seam_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("auth_state_85_3", SEAM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_c5_dd5_apply_transition_guard_driven_directly(tmp_path):
    """Cycle-1 Q/A: the subprocess DD-5 test crashes before apply_transition
    runs, so the strict-true guard itself was uncovered (mutant MQ-A
    survived). This drives the function DIRECTLY: 'unknown' and 'false'
    derivations must leave an open incident open; only 'true' clears."""
    mod = _load_seam_module()
    state = tmp_path / "auth_page_state.json"
    for verdict in ("unknown", "false"):
        state.write_text(json.dumps({"incident_open": True, "paged": True}))
        mod.apply_transition(str(state), verdict, NOW)
        assert json.loads(state.read_text())["incident_open"] is True, (
            f"apply_transition cleared on {verdict!r} -- the strict-true "
            "guard is gone (DD-5 regression)"
        )
    state.write_text(json.dumps({"incident_open": True, "paged": True}))
    mod.apply_transition(str(state), "true", NOW)
    st = json.loads(state.read_text())
    assert st["incident_open"] is False and st["cleared_by"] == "healthcheck_healthy"


def test_after_clear_suppresses_repage_for_pre_clear_401(tmp_path):
    """MQ-C coverage: the pre-existing re-page-suppression rule -- a 401
    session OLDER than the latch's cleared_at never re-opens the incident,
    even when inside the freshness window. Carried from healthcheck.sh:92-93
    into the seam; previously zero coverage."""
    ops, state = _ops(tmp_path)
    _mk_session(ops, "session_401.json", SESSION_401_SPACED,
                datetime.timedelta(hours=6))  # fresh by AGE
    cleared = (NOW - datetime.timedelta(hours=3)).isoformat()  # AFTER the 401
    state.write_text(json.dumps({"incident_open": False, "cleared_at": cleared,
                                 "cleared_by": "healthcheck_healthy"}))
    ok, detail = _run(ops, state)
    assert ok == "true", (
        f"a pre-clear 401 re-opened the incident: {detail} -- the "
        "re-page-suppression rule is gone"
    )


# ── L2: the active re-check leg is load-bearing ──────────────────────


def test_l2_dead_auth_status_fails_regardless_of_stale_evidence(tmp_path):
    ops, state = _ops(tmp_path)
    _mk_session(ops, "session_stale401.json", SESSION_401_SPACED,
                datetime.timedelta(days=27))
    ok, detail = _run(ops, state, status_ok="false")
    assert ok == "false"
    assert detail == "auth_status_rc_nonzero"


# ── seam hygiene: no repo-path fallback ──────────────────────────────


def test_seam_reads_only_the_given_paths(tmp_path):
    """C2's isolation clause: the seam must resolve ONLY --ops/--state.
    Source-level: argparse marks both required with no default pointing at
    handoff/away_ops; behavioural: an empty ops dir yields 'ok' even though
    the REAL ops dir currently contains a 401 session file."""
    ops, state = _ops(tmp_path)
    ok, detail = _run(ops, state)
    assert (ok, detail) == ("true", "ok")
    src = pathlib.Path(SEAM).read_text()
    assert "handoff/away_ops" not in src, "seam carries a repo-path fallback"
