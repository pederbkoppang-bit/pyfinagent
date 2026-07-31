"""phase-71.3 -- verdict_gate.py fail-open decision logic.

The status-flip gate reads the machine-readable Q/A verdict
(handoff/current/evaluator_critique.json) instead of prose. The gate is
FAIL-OPEN: it only HOLDS on an explicit, step-matched, non-PASS verdict.
"""
import importlib.util
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE = os.path.join(REPO, ".claude", "hooks", "lib", "verdict_gate.py")

_spec = importlib.util.spec_from_file_location("verdict_gate", GATE)
verdict_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verdict_gate)


def _write(tmp_path, obj):
    p = tmp_path / "evaluator_critique.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_pass_verdict_matching_step_returns_passed(tmp_path):
    p = _write(tmp_path, {"step_id": "71.3", "ok": True, "verdict": "PASS"})
    assert verdict_gate.gate_decision(p, "71.3") == "passed"


def test_conditional_verdict_matching_step_holds(tmp_path):
    p = _write(tmp_path, {"step_id": "71.3", "ok": False, "verdict": "CONDITIONAL"})
    assert verdict_gate.gate_decision(p, "71.3") == "hold"


def test_fail_verdict_holds(tmp_path):
    p = _write(tmp_path, {"step_id": "71.3", "ok": False, "verdict": "FAIL"})
    assert verdict_gate.gate_decision(p, "71.3") == "hold"


def test_pass_but_ok_false_holds(tmp_path):
    # ok=false is not a clean PASS -> hold (defensive).
    p = _write(tmp_path, {"step_id": "71.3", "ok": False, "verdict": "PASS"})
    assert verdict_gate.gate_decision(p, "71.3") == "hold"


def test_missing_json_returns_no_input(tmp_path):
    """phase-81.0 -- DELIBERATE CONTRACT CHANGE, not a regression.

    Until 81.0 this asserted 'proceed', which made "no gate input existed"
    indistinguishable from "the gate ran and passed" in auto-push.log. That
    ambiguity is exactly why the gate went dark for 13 consecutive step
    closes after 2026-07-20 with nobody noticing. The file-absent case now
    returns its own token so the caller can log it.

    Still FAIL-OPEN: the token is not 'hold'. The hook continues; only the
    log line changes. See the module docstring in verdict_gate.py.
    """
    decision = verdict_gate.gate_decision(str(tmp_path / "nope.json"), "71.3")
    assert decision == "no_input"
    assert decision != "hold", "the absent-input case must never block a push"


def test_no_input_is_distinguishable_from_every_other_token(tmp_path):
    """The whole point: 'nothing checked' must not collide with any other state."""
    absent = verdict_gate.gate_decision(str(tmp_path / "nope.json"), "71.3")
    unreadable_path = tmp_path / "bad.json"
    unreadable_path.write_text("{not valid json", encoding="utf-8")
    unreadable = verdict_gate.gate_decision(str(unreadable_path), "71.3")
    passed = verdict_gate.gate_decision(
        _write(tmp_path, {"step_id": "71.3", "ok": True, "verdict": "PASS"}), "71.3"
    )
    assert absent == "no_input"
    # A file that exists but cannot gate is a DIFFERENT situation from no file
    # at all, and must keep its original token.
    assert unreadable == "proceed"
    assert len({absent, unreadable, passed}) == 3


def test_mismatched_step_fails_open_proceed(tmp_path):
    # A stale JSON for a different step must NOT block this step.
    p = _write(tmp_path, {"step_id": "70.5", "ok": True, "verdict": "PASS"})
    assert verdict_gate.gate_decision(p, "71.3") == "proceed"


def test_no_verdict_field_fails_open_proceed(tmp_path):
    p = _write(tmp_path, {"step_id": "71.3", "ok": True})
    assert verdict_gate.gate_decision(p, "71.3") == "proceed"


def test_unreadable_json_fails_open_proceed(tmp_path):
    p = tmp_path / "evaluator_critique.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert verdict_gate.gate_decision(str(p), "71.3") == "proceed"


def test_no_step_id_in_json_still_gates_on_verdict(tmp_path):
    # No step_id present -> fail-open on the step match, but a present verdict
    # still applies (a PASS -> passed).
    p = _write(tmp_path, {"ok": True, "verdict": "PASS"})
    assert verdict_gate.gate_decision(p, "71.3") == "passed"
