"""phase-4000.2 -- the CC-rail E2E smoke script, tested as a real child process.

Every scenario drives scripts/qa/smoke_cc_rail_e2e.py via subprocess.run and
asserts on .returncode (criterion 7) against a REAL-socket stdlib stub backend
(pytest-httpserver is absent from the venv; contract_4000.2.md non-scope forbids
adding deps without operator sign-off; prior art test_phase_76_9_2_max_bridge.py).

The claude CLI is stubbed by an executable literally named `claude` (contract
R1: CLAUDE_CODE_BINARY is NOT a safe hook -- shutil.which wins first, and the
_DEFAULT_SEARCH_PATHS literal is frozen at import). Tests either pass
--claude-binary <stub> explicitly or run --no-probe; NO test relies on ambient
resolution without a controlled PATH, so the real CLI is never invocable.

MUTATION COVERAGE (criterion 8), named per the masterplan; test ids are exact
(Q/A 4000.2 cycle-1 finding 3a corrected a stale id here):
  (m1)  breaking is_rail_row so it matches zero rows -> test_live_all_pass
        turns red (E1 requires N-of-N matched with N>0).
  (m2a) neutering E2's metered-row leg (`len(metered) == 0` -> `True`) ->
        test_live_single_check_fail_exits_nonzero[E2_metered_row_present]
        turns red (its fixture plants a metered gpt-model row, requires exit 1).
  (m2b) neutering E2's spend-delta leg (`spend_delta == 0.0` -> `True`) ->
        test_live_single_check_fail_exits_nonzero[E2_spend_delta] turns red
        (its fixture brackets 1.0 -> 2.5 and requires exit 1).
  (mP)  neutering E3a's cost-sum conjunct (`if total is None or ...` ->
        `if False:`) -> test_live_single_check_fail_exits_nonzero[E3a_cost_sum_mismatch]
        turns red (all-domestic envelope, sum != total, foreign leg green).
  (mS)  neutering E4's status/error leg (`if body.get("status") != ...` ->
        `if False:`) -> test_live_single_check_fail_exits_nonzero[E4_status_failed_report_present]
        turns red (failed status with a GOOD report, so no other leg masks it).
The executor runs the matrix once against this suite, records the verbatim
output in the handoff, and restores the script.

E4 SCOPE DISCLOSURE (Q/A 4000.2 cycle-2 finding 1): the frozen baseline's E4
has FOUR legs; legs 1-3 (terminal-completed, report present, no synthetic
0.0/HOLD) are implemented and fixtured here; leg 4 (persisted analysis row)
is NOT PROVABLE from the sync poll (in-memory task dict, analysis.py:392) --
it is disclosed in the emitted E4 verdict, queued as its own masterplan step,
and owed as 4000.3 live_check evidence against the real persisted surface.

WATCHER DISCLOSURE (Q/A cycle-1 m15): the during-window watcher thread is a
best-effort EARLY-abort optimization and is deliberately not what the cap
guarantee rests on -- the authoritative gate is the post-window count
(test_postwindow_cap_authoritative), which is timing-independent. A mutant
that never starts the watcher is behaviorally masked BY DESIGN.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "qa" / "smoke_cc_rail_e2e.py"

GOOD_ENVELOPE = {
    "total_cost_usd": 0.15372910,
    "modelUsage": {
        "claude-haiku-4-5": {"canonicalModel": "claude-haiku-4-5", "costUSD": 0.091379},
        "claude-haiku-4-5-20251001": {"canonicalModel": "claude-haiku-4-5",
                                      "costUSD": 0.0623501},
    },
    "result": "OK",
}
# Criterion 5: configured tier FIRST, foreign model SECOND. total_cost_usd
# deliberately equals the FIRST entry's costUSD alone (Q/A cycle-1 finding 3b):
# a first-key-only reader sees cost 0.05 == total AND a haiku canonicalModel,
# so it genuinely PASSES -- only iterating ALL keys finds the foreign model
# (and the true sum mismatch). This fixture is the criterion-5 discriminator.
FOREIGN_ENVELOPE = {
    "total_cost_usd": 0.05,
    "modelUsage": {
        "claude-haiku-4-5": {"canonicalModel": "claude-haiku-4-5", "costUSD": 0.05},
        "gpt-9-preview": {"canonicalModel": "gpt-9-preview", "costUSD": 0.05},
    },
    "result": "OK",
}

# Kills QA-cycle-2 mutant mP: all-DOMESTIC two-key envelope whose costUSD
# values do not sum to total_cost_usd -- the foreign-model leg passes, so the
# cost-sum conjunct is the ONLY thing that can fail it.
BAD_SUM_ENVELOPE = {
    "total_cost_usd": 0.20,
    "modelUsage": {
        "claude-haiku-4-5": {"canonicalModel": "claude-haiku-4-5", "costUSD": 0.05},
        "claude-haiku-4-5-20251001": {"canonicalModel": "claude-haiku-4-5",
                                      "costUSD": 0.05},
    },
    "result": "OK",
}

RAIL_ROW = {"provider": "anthropic", "model": "claude-sonnet-4-6", "agent": "cc_rail",
            "ok": True, "ticker": "TEST", "input_tok": 100, "output_tok": 50,
            "cache_creation_tok": 1000, "cache_read_tok": 0, "ts": "2026-08-06T12:00:00+00:00"}
LITE_ROW = {**RAIL_ROW, "provider": "claude-code", "agent": "lite_trader"}
METERED_ROW = {**RAIL_ROW, "provider": "anthropic", "agent": None}
# Fails ONLY E2 (not claude-role, so E1 ignores it) -- keeps m2 uniquely killable.
METERED_GPT_ROW = {**METERED_ROW, "model": "gpt-9-preview"}
FAILED_RAIL_ROW = {**RAIL_ROW, "ok": False}

COMPLETED_BODY = {"status": "completed", "error": None,
                  "report": {"recommendation": "Hold", "final_score": 0.5}}


class _Handler(BaseHTTPRequestHandler):
    state: dict = {}

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        s = self.state
        if self.path.startswith("/api/health"):
            self._send({"status": "ok"})
        elif self.path.startswith("/api/settings"):
            self._send(s["settings"])
        elif self.path.startswith("/api/analysis/"):
            self._send(s["analysis_body"])
        elif self.path.startswith("/llmlog"):
            self._send(s["llmlog_rows"])
        elif self.path.startswith("/spend"):
            vals = s["spend_values"]
            self._send({"daily_usd": vals.pop(0) if len(vals) > 1 else vals[0]})
        else:
            self._send({"detail": "not found"}, 404)

    def do_POST(self):
        s = self.state
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path.startswith("/api/analysis"):
            if s.get("analysis_post_500"):
                self._send({"detail": "boom"}, 500)
            else:
                self._send({"analysis_id": "A1", "ticker": "TEST", "status": "pending"})
        else:
            self._send({"detail": "not found"}, 404)

    def do_PUT(self):
        s = self.state
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path.startswith("/api/settings"):
            s["puts"].append(body)
            s["settings"].update(body)
            self._send(s["settings"])
        else:
            self._send({"detail": "not found"}, 404)


@pytest.fixture()
def stub_backend():
    state = {
        "settings": {"paper_use_claude_code_route": False,
                     "gemini_model": "claude-sonnet-4-6",
                     "deep_think_model": "claude-opus-4-8"},
        "puts": [],
        "analysis_body": COMPLETED_BODY,
        "llmlog_rows": [RAIL_ROW, LITE_ROW],
        "analysis_post_500": False,
        "spend_values": [0.0],
    }
    handler = type("H", (_Handler,), {"state": state})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", state
    finally:
        srv.shutdown()


@pytest.fixture()
def stub_claude(tmp_path):
    def make(envelope: dict) -> str:
        d = tmp_path / "bin"
        d.mkdir(exist_ok=True)
        payload = json.dumps(envelope)
        marker = tmp_path / "invocations.log"
        path = d / "claude"
        path.write_text("#!/bin/sh\n"
                        f"echo invoked >> {marker}\n"
                        f"cat << 'ENVELOPE_EOF'\n{payload}\nENVELOPE_EOF\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return str(path)

    return make


def run_smoke(*argv, env_extra=None, timeout=120):
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, str(SCRIPT), *argv],
                          capture_output=True, text=True, timeout=timeout, env=env)


def live_args(base, extra=()):
    return ["--live", "--ticker", "TEST", "--backend-url", base,
            "--llm-log-source", f"{base}/llmlog", "--spend-source", f"{base}/spend",
            "--expected-calls-per-analysis", "10",
            "--flush-wait-s", "0", "--watch-interval", "1",
            "--analysis-deadline", "60", *extra]


# ── criterion 1: dry mode mutates nothing ────────────────────────────────────
def test_dry_makes_zero_puts(stub_backend):
    base, state = stub_backend
    r = run_smoke("--dry", "--no-probe", "--backend-url", base,
                  "--llm-log-source", f"{base}/llmlog")
    assert r.returncode == 0, r.stdout + r.stderr
    assert state["puts"] == []          # ZERO settings mutations of any kind
    assert '"rail_state": "OFF"' in r.stdout  # honest state for flag=False


# ── criterion 3: binary resolution via the imported production logic ─────────
def test_binary_resolution_through_stub_search_path(stub_backend, stub_claude, tmp_path):
    base, _ = stub_backend
    stub = stub_claude(GOOD_ENVELOPE)
    # PATH holds ONLY the stub dir; the script's resolve_binary(None) imports
    # _resolve_claude_binary, whose shutil.which() must find the stub first.
    r = run_smoke("--dry", "--no-probe", "--backend-url", base,
                  "--llm-log-source", f"{base}/llmlog",
                  env_extra={"PATH": str(Path(stub).parent)})
    assert r.returncode == 0, r.stdout + r.stderr
    preflight = next(json.loads(l) for l in r.stdout.splitlines()
                     if '"event": "preflight"' in l)
    assert preflight["binary"] == stub


# ── usage caps ───────────────────────────────────────────────────────────────
def test_three_tickers_rejected(stub_backend):
    base, _ = stub_backend
    r = run_smoke(*live_args(base), "--ticker", "T2", "--ticker", "T3", "--no-probe")
    assert r.returncode == 4
    assert "2-ticker cap" in r.stderr


def test_cap_cannot_be_raised(stub_backend):
    base, _ = stub_backend
    r = run_smoke(*live_args(base), "--max-rail-calls", "31", "--no-probe")
    assert r.returncode == 4
    assert "never raise" in r.stderr


# ── criterion 2: the <=30-rail-call budget aborts loudly ─────────────────────
def test_expected_31_calls_trips_cap(stub_backend):
    base, state = stub_backend
    r = run_smoke("--live", "--ticker", "TEST", "--backend-url", base,
                  "--llm-log-source", f"{base}/llmlog",
                  "--expected-calls-per-analysis", "31",
                  "--flush-wait-s", "0", "--no-probe")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "cap" in r.stdout            # message names the cap
    assert "<=30" in r.stdout
    # the refusal happened BEFORE any analysis started, and the flag was restored
    assert state["puts"][0] == {"paper_use_claude_code_route": True}
    assert state["puts"][-1] == {"paper_use_claude_code_route": False}


# ── criterion 4 direction 1 + criterion 5 fixtures ───────────────────────────
def test_live_all_pass(stub_backend, stub_claude):
    base, state = stub_backend
    stub = stub_claude(GOOD_ENVELOPE)
    r = run_smoke(*live_args(base, ("--claude-binary", stub)))
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [json.loads(l) for l in r.stdout.splitlines()]
    summary = next(l for l in lines if l.get("summary"))
    assert summary["all_ok"] is True
    checks = dict(tuple(c) for c in summary["checks"])
    assert set(checks) == {"E1", "E2", "E3", "E4", "E5", "E6"} and all(checks.values())
    # restore PUT fired (no --keep-on): flip then restore to the pre value
    assert state["puts"] == [{"paper_use_claude_code_route": True},
                             {"paper_use_claude_code_route": False}]


def test_live_keep_on_cancels_restore(stub_backend, stub_claude):
    base, state = stub_backend
    stub = stub_claude(GOOD_ENVELOPE)
    r = run_smoke(*live_args(base, ("--claude-binary", stub, "--keep-on")))
    assert r.returncode == 0, r.stdout + r.stderr
    assert state["puts"] == [{"paper_use_claude_code_route": True}]  # no restore


def _fail_fixture(state, stub_claude, name):
    """Mutate the stub state so exactly the named check fails; return the
    --claude-binary to use."""
    if name == "E1_nonrail_claude_row":
        state["llmlog_rows"] = [RAIL_ROW, METERED_ROW]  # metered claude row breaks N-of-N
        return None
    if name == "E2_metered_row_present":
        state["llmlog_rows"] = [RAIL_ROW, METERED_GPT_ROW]  # E2-only failure
        return None
    if name == "E3_foreign_model_second_key":
        return FOREIGN_ENVELOPE
    if name == "E4_failed_analysis":
        state["analysis_body"] = {"status": "failed", "error": "[Boom] step died",
                                 "report": None}
        return None
    if name == "E5_failed_rail_row":
        state["llmlog_rows"] = [RAIL_ROW, FAILED_RAIL_ROW]
        return None
    if name == "E2_spend_delta":
        state["spend_values"] = [1.0, 2.5]  # metered spend moved during window
        return None
    if name == "E4_report_absent":
        # kills QA-survivor m11: completed WITHOUT error but report null
        state["analysis_body"] = {"status": "completed", "error": None, "report": None}
        return None
    if name == "E3b_stray_model":
        # kills QA-survivor m17: rail row whose model is outside the configured
        # expectation set (rail + claude-role, so E1 stays green)
        state["llmlog_rows"] = [RAIL_ROW,
                                {**RAIL_ROW, "model": "claude-9-frontier"}]
        return None
    if name == "E3a_cost_sum_mismatch":
        return BAD_SUM_ENVELOPE  # kills QA-cycle-2 mutant mP
    if name == "E4_synthetic_zero_hold":
        # frozen-baseline E4 leg 3: completed, report present, but the
        # synthetic 0.0/HOLD shape (61.2 defect class)
        state["analysis_body"] = {"status": "completed", "error": None,
                                 "report": {"recommendation": {"action": "Hold"},
                                            "final_weighted_score": 0.0}}
        return None
    if name == "E4_status_failed_report_present":
        # isolates E4's status/error leg (QA-cycle-2 mutant mS): report is GOOD,
        # so neither the report-absent nor the synthetic leg can mask the kill
        state["analysis_body"] = {"status": "failed", "error": "[Boom] step died",
                                 "report": {"recommendation": {"action": "Buy"},
                                            "final_weighted_score": 0.7}}
        return None
    raise AssertionError(name)


# criterion 4 direction 2: each single-check-fail fixture exits non-zero.
@pytest.mark.parametrize("name", ["E1_nonrail_claude_row", "E2_metered_row_present",
                                  "E2_spend_delta", "E3_foreign_model_second_key",
                                  "E3a_cost_sum_mismatch", "E3b_stray_model",
                                  "E4_failed_analysis", "E4_report_absent",
                                  "E4_synthetic_zero_hold",
                                  "E4_status_failed_report_present",
                                  "E5_failed_rail_row"])
def test_live_single_check_fail_exits_nonzero(stub_backend, stub_claude, name):
    base, state = stub_backend
    envelope = _fail_fixture(state, stub_claude, name)
    stub = stub_claude(envelope or GOOD_ENVELOPE)
    r = run_smoke(*live_args(base, ("--claude-binary", stub)))
    assert r.returncode == 1, f"{name}: {r.stdout}{r.stderr}"
    lines = [json.loads(l) for l in r.stdout.splitlines()]
    summary = next(l for l in lines if l.get("summary"))
    assert summary["all_ok"] is False
    # the flag restore STILL fired on the failing path
    assert state["puts"][-1] == {"paper_use_claude_code_route": False}


# ── QA-survivor kills: positive controls, cap authority, env-var resolution ──
def test_empty_window_fails_positive_controls(stub_backend, stub_claude):
    """Kills QA-survivors m10+m16: an EMPTY window (0-of-0 rows) must FAIL --
    E1's n>0 and E2's rail>0 positive controls are load-bearing (contract R6)."""
    base, state = stub_backend
    state["llmlog_rows"] = []
    stub = stub_claude(GOOD_ENVELOPE)
    r = run_smoke(*live_args(base, ("--claude-binary", stub)))
    assert r.returncode == 1, r.stdout + r.stderr
    lines = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    checks = dict(tuple(c) for c in next(l for l in lines if l.get("summary"))["checks"])
    assert checks["E1"] is False and checks["E2"] is False


def test_postwindow_cap_authoritative(stub_backend, stub_claude):
    """The cap guarantee is timing-independent: 31 rail rows in the window trip
    the POST-WINDOW gate (exit 3) even if the watcher never fires; the restore
    PUT still lands (ExitStack)."""
    base, state = stub_backend
    state["llmlog_rows"] = [dict(RAIL_ROW, ticker=f"T{i}") for i in range(31)]
    stub = stub_claude(GOOD_ENVELOPE)
    r = run_smoke(*live_args(base, ("--claude-binary", stub)))
    assert r.returncode == 3, r.stdout + r.stderr
    assert "cap" in r.stdout
    assert state["puts"][-1] == {"paper_use_claude_code_route": False}


def test_binary_resolution_env_var_fallback(stub_backend, stub_claude):
    """Kills QA-survivor m6: with claude ABSENT from PATH, production
    resolution falls through to the CLAUDE_CODE_BINARY candidate in
    _DEFAULT_SEARCH_PATHS (claude_code_client.py:49-54, captured at the child
    process's import). A which()-only reimplementation cannot resolve this and
    would report the bare string 'claude'."""
    base, _ = stub_backend
    stub = stub_claude(GOOD_ENVELOPE)
    r = run_smoke("--dry", "--no-probe", "--backend-url", base,
                  "--llm-log-source", f"{base}/llmlog",
                  env_extra={"PATH": "/usr/bin:/bin", "CLAUDE_CODE_BINARY": stub})
    assert r.returncode == 0, r.stdout + r.stderr
    preflight = next(json.loads(l) for l in r.stdout.splitlines()
                     if '"event": "preflight"' in l)
    assert preflight["binary"] == stub


# ── criterion 6: crash mid-window still restores via PUT ─────────────────────
def test_crash_mid_window_still_restores(stub_backend, stub_claude):
    base, state = stub_backend
    state["analysis_post_500"] = True   # POST /api/analysis/ dies -> HTTPError
    stub = stub_claude(GOOD_ENVELOPE)
    r = run_smoke(*live_args(base, ("--claude-binary", stub)))
    assert r.returncode == 2, r.stdout + r.stderr
    # ExitStack unwound: flip PUT then restore PUT, despite the crash
    assert state["puts"] == [{"paper_use_claude_code_route": True},
                             {"paper_use_claude_code_route": False}]
