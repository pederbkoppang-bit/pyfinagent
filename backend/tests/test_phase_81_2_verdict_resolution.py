"""phase-81.2 -- verdict_gate order-independent input resolution.

WHY THIS EXISTS. The verdict gate resolved exactly ONE literal path, and a miss
fails open. So any file move disarmed it silently. That happened: STEP_ID_RE in
scripts/housekeeping/*.py matches 0 of the 127 .md files in handoff/current/, so
backfill_handoff_archive.py classified every step artifact as unclassifiable and
moved it to handoff/archive/misc/. Commit fa9aaf8e (2026-07-24) swept 315 files
that way, took handoff/current/evaluator_critique.json with them, and the gate
ran dark for 13 consecutive step closes with nobody noticing.

The fixtures below RE-CREATE that sweep rather than describing it: they put a
CONDITIONAL verdict only in the archive and assert the gate still holds.
"""
import importlib.util
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE = os.path.join(REPO, ".claude", "hooks", "lib", "verdict_gate.py")

_spec = importlib.util.spec_from_file_location("verdict_gate_81_2", GATE)
verdict_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verdict_gate)


def _handoff(tmp_path):
    (tmp_path / "current").mkdir()
    (tmp_path / "archive" / "misc").mkdir(parents=True)
    return tmp_path


def _put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _snapshot(root):
    return sorted(
        os.path.relpath(os.path.join(d, f), root)
        for d, _, fs in os.walk(root)
        for f in fs
    )


# --- criterion 1 + 4: the disarm-by-move failure mode is closed ---------------

def test_conditional_found_only_in_archive_still_holds(tmp_path):
    """Reproduces the 2026-07-24 sweep: current/ is empty, verdict is archived.

    Before 81.2 this returned no_input/proceed -- a CONDITIONAL verdict silently
    stopped holding the push the moment housekeeping ran.
    """
    h = _handoff(tmp_path)
    _put(h / "archive" / "phase-80.5" / "evaluator_critique_80.5.json",
         {"step_id": "80.5", "ok": False, "verdict": "CONDITIONAL"})
    decision, source = verdict_gate.gate_decision_with_source("80.5", str(h))
    assert decision == "hold", "an archived CONDITIONAL must still hold the push"
    assert source == "archive:step"


def test_archived_pass_gates_and_names_the_archive_as_its_source(tmp_path):
    h = _handoff(tmp_path)
    _put(h / "archive" / "misc" / "evaluator_critique_71.3.json",
         {"step_id": "71.3", "ok": True, "verdict": "PASS"})
    decision, source = verdict_gate.gate_decision_with_source("71.3", str(h))
    assert decision == "passed"
    # criterion 3: a verdict answered from the archive must be DISTINGUISHABLE
    # from one answered from handoff/current/, or the operator cannot tell a
    # fresh gate from a stale one.
    assert source == "archive:misc"
    assert source != "current:per-step"


# --- criterion 2: resolution order, first hit wins ---------------------------

def test_per_step_current_beats_rolling_current_beats_archive(tmp_path):
    """All three populated with DIFFERENT verdicts; only the winner may be read."""
    h = _handoff(tmp_path)
    _put(h / "current" / "evaluator_critique_81.2.json",
         {"step_id": "81.2", "ok": True, "verdict": "PASS"})
    _put(h / "current" / "evaluator_critique.json",
         {"step_id": "81.2", "ok": False, "verdict": "FAIL"})
    _put(h / "archive" / "phase-81.2" / "evaluator_critique_81.2.json",
         {"step_id": "81.2", "ok": False, "verdict": "CONDITIONAL"})
    decision, source = verdict_gate.gate_decision_with_source("81.2", str(h))
    assert source == "current:per-step"
    assert decision == "passed", "a later source must not be consulted once one hits"

    # Remove the winner -> the next source in the chain takes over.
    (h / "current" / "evaluator_critique_81.2.json").unlink()
    decision, source = verdict_gate.gate_decision_with_source("81.2", str(h))
    assert source == "current:rolling"
    assert decision == "hold"

    # Remove that too -> the archive answers.
    (h / "current" / "evaluator_critique.json").unlink()
    decision, source = verdict_gate.gate_decision_with_source("81.2", str(h))
    assert source == "archive:step"
    assert decision == "hold"


# --- criterion 5: nothing anywhere still fails open --------------------------

def test_no_input_anywhere_fails_open(tmp_path):
    h = _handoff(tmp_path)
    decision, source = verdict_gate.gate_decision_with_source("99.9", str(h))
    assert decision == "no_input"
    assert source == "none"
    assert decision != "hold", "absent input must never block a push"


def test_another_steps_verdict_does_not_gate_this_step(tmp_path):
    """A neighbouring step's archived FAIL must not hold THIS step's push."""
    h = _handoff(tmp_path)
    _put(h / "archive" / "phase-70.1" / "evaluator_critique_70.1.json",
         {"step_id": "70.1", "ok": False, "verdict": "FAIL"})
    decision, source = verdict_gate.gate_decision_with_source("81.2", str(h))
    assert decision == "no_input"
    assert source == "none"


# --- criterion 6: strictly additive -----------------------------------------

def test_legacy_two_arg_signature_unchanged(tmp_path):
    """The phase-71.3 contract must survive verbatim."""
    p = _put(tmp_path / "evaluator_critique.json",
             {"step_id": "71.3", "ok": True, "verdict": "PASS"})
    assert verdict_gate.gate_decision(str(p), "71.3") == "passed"
    assert verdict_gate.gate_decision(str(tmp_path / "nope.json"), "71.3") == "no_input"


# --- criterion 7: the gate must not mutate the tree it inspects --------------

def test_resolution_performs_no_writes_or_moves(tmp_path):
    h = _handoff(tmp_path)
    _put(h / "archive" / "phase-80.5" / "evaluator_critique_80.5.json",
         {"step_id": "80.5", "ok": True, "verdict": "PASS"})
    before = _snapshot(str(h))
    for sid in ("80.5", "99.9", "81.2"):
        verdict_gate.gate_decision_with_source(sid, str(h))
    assert _snapshot(str(h)) == before, "the gate moved or created a file"


def test_resolver_returns_none_for_missing_root(tmp_path):
    """A nonexistent handoff root must fail open, not raise."""
    path, label = verdict_gate.resolve_verdict_source("81.2", str(tmp_path / "nope"))
    assert path is None and label == "none"
