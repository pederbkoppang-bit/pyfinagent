#!/usr/bin/env python3
"""Mutation matrix for phase-90.1 -- attempt-row accounting.

Subjects: `scripts/harness/attempt_gate.py` and `scripts/harness/attempt_outcomes.py`.

House discipline (per mutation_matrix_86_71.py): CONTROL observed GREEN before
any cell; a cell is KILLED only when a NAMED check fails for a stated reason; a
mutant that does not run scores ERROR and is NEVER a kill; the real tree is
never written (md5 before == after, mutants run from a temp copy via subprocess
so the CALL SITE is what is tested).

Two things this matrix is careful about, because the project has been bitten by
both:

- DISCRIMINATION. Every check must be able to come back green on the real file
  and red on a mutant. The null-mutant cell (`N0`) is a comment-only edit: if it
  scores KILLED, the harness itself is broken and every other kill in the run is
  meaningless. A matrix whose checks all fail for an environmental reason scores
  a full house and proves nothing.
- CONTAINMENT. Every drive redirects the ledger, the verdict ledger, the
  escalation dir, the masterplan AND the run-record root. During development an
  earlier revision of the gate's self-test wrote a real
  `escalation_unknown_step_id_9.9.md` into production `handoff/current/`
  because only the ledger had been redirected.

    python3 scripts/qa/mutation_matrix_90_1.py            # report
    python3 scripts/qa/mutation_matrix_90_1.py --verify   # exit 1 on a survivor
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "harness" / "attempt_gate.py"
OUTCOMES = REPO / "scripts" / "harness" / "attempt_budget.py"
RESOLVER = REPO / "scripts" / "harness" / "attempt_outcomes.py"

#: A step id that IS in the synthetic plan of record each drive writes.
REAL_SID = "9.1"
#: Well-formed, dotted-numeric, and in no plan. The `.1` suffix is the exact
#: bypass this step closes.
FAKE_SID = "9.1.7"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _hook_stdin(step_id: str) -> str:
    return json.dumps({
        "tool_name": "Workflow",
        "tool_input": {"scriptPath": ".claude/workflows/qa-verdict.js",
                       "args": {"step_id": step_id}},
        "tool_use_id": "toolu_mm90_1", "session_id": "mm",
    })


def _env(tmp: Path, gate_dir: Path) -> dict:
    """Every output channel redirected into tmp. See the docstring."""
    plan = tmp / "masterplan.json"
    if not plan.is_file():
        # NOTE the nested member: the plan of record is not uniformly
        # phases[].steps[], and the cycle-1 BLOCK was a walk that assumed it
        # was. `9.9` here lives under subphases[], exactly like the real
        # 46.0-46.8, so a shallow walk denies it and cell M13 dies.
        plan.write_text(json.dumps({"phases": [
            {"id": "phase-9",
             "steps": [{"id": f"9.{n}"} for n in range(1, 6)],
             "subphases": [{"id": "phase-9.9",
                            "steps": [{"id": "9.9.1", "status": "pending",
                                       "harness_required": True}]}]},
        ]}), encoding="utf-8")
    esc = tmp / "escalations"
    esc.mkdir(exist_ok=True)
    records = tmp / "records"
    records.mkdir(exist_ok=True)
    return dict(
        os.environ,
        ATTEMPT_GATE_LEDGER=str(tmp / "attempts.jsonl"),
        ATTEMPT_GATE_VERDICT_LEDGER=str(tmp / "absent_verdicts.jsonl"),
        ATTEMPT_GATE_ESCALATION_DIR=str(esc),
        ATTEMPT_GATE_MASTERPLAN=str(plan),
        ATTEMPT_GATE_RUN_RECORDS=str(records),
        PYTHONPATH=os.pathsep.join([
            str(gate_dir), str(REPO / "scripts" / "harness"),
            str(REPO / "scripts" / "qa"), os.environ.get("PYTHONPATH", "")]),
    )


def _seed(tmp: Path, rows: list[dict]) -> None:
    (tmp / "attempts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def drive_hook(gate: Path, tmp: Path, step_id: str,
               rows: list[dict] | None = None) -> dict:
    _seed(tmp, rows or [])
    env = _env(tmp, gate.parent)
    r = subprocess.run([sys.executable, str(gate)], input=_hook_stdin(step_id),
                       capture_output=True, text=True, env=env, timeout=90)
    esc = tmp / "escalations"
    return {"rc": r.returncode, "stderr": r.stderr,
            "files": sorted(p.name for p in esc.iterdir()),
            "rows_after": sum(1 for _ in (tmp / "attempts.jsonl").open())}


def drive_status(gate: Path, tmp: Path, rows: list[dict]) -> dict:
    _seed(tmp, rows)
    env = _env(tmp, gate.parent)
    r = subprocess.run([sys.executable, str(gate), "--status", REAL_SID],
                       capture_output=True, text=True, env=env, timeout=90)
    try:
        return {"rc": r.returncode, "json": json.loads(r.stdout or "{}")}
    except json.JSONDecodeError:
        return {"rc": r.returncode, "json": {}}


def drive_preserve(gate: Path, tmp: Path) -> dict:
    """criterion 2: a NON-exhaustion denial must not touch an exhaustion record.

    A REAL exhaustion escalation is planted at the exact path the pre-90.1
    fixed-path/forged-body code would have overwritten. sha256 before and after.
    """
    esc = tmp / "escalations"
    esc.mkdir(exist_ok=True)
    victim = esc / f"escalation_attempt_budget_{REAL_SID}.md"
    victim.write_text(
        "# BUDGET EXHAUSTED -- step 9.1 -- OPERATOR DECISION REQUIRED\n\n"
        "- attempts used : 5 / 5\n\nHAND-AUTHORED CONTENT THAT MUST SURVIVE.\n",
        encoding="utf-8")
    before = sha256(victim)
    out = drive_hook(gate, tmp, FAKE_SID)      # a NON-exhaustion denial
    return {"rc": out["rc"], "files": out["files"], "stderr": out["stderr"],
            "sha_before": before,
            "sha_after": sha256(victim) if victim.is_file() else None}


def drive_forge(gate: Path, tmp: Path) -> dict:
    """Call write_escalation with NO body and a non-exhaustion reason.

    The first revision of this matrix let M4 SURVIVE, and the reason was a real
    coverage hole rather than a bad mutant: every call site passes an explicit
    body, so the `# BUDGET EXHAUSTED` fallback was unreachable from any drive
    and its guard was never executed. An unexercised guard is indistinguishable
    from an absent one -- so this drive reaches it directly.
    """
    env = _env(tmp, gate.parent)
    code = (
        "import importlib.util, json, sys" + chr(10) +
        f"spec = importlib.util.spec_from_file_location('g', {str(gate)!r})" + chr(10) +
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)" + chr(10) +
        "st = m.BudgetState(step_id='9.1')" + chr(10) +   # fresh: NOT exhausted
        "out = {'raised': False, 'wrote': None, 'body': None}" + chr(10) +
        "try:" + chr(10) +
        "    p = m.write_escalation(st, reason='some_other_reason')" + chr(10) +
        "    out['wrote'] = p.name" + chr(10) +
        "    out['body'] = p.read_text(encoding='utf-8')[:80]" + chr(10) +
        "except ValueError:" + chr(10) +
        "    out['raised'] = True" + chr(10) +
        "print(json.dumps(out))" + chr(10)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=90)
    try:
        return json.loads(r.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {}


def _plant_run_record(tmp: Path, step_id: str, ts_iso: str,
                      offset_ms: int, verdict: str, tokens: int,
                      name: str = "wf_planted") -> None:
    """A synthetic Workflow run record offset from the attempt row's ts.

    The offset is what exercises the JOIN TOLERANCE. Without a record on disk
    every row resolves UNKNOWN no matter what the tolerance is, which is why the
    first revision of this matrix let M10 survive.
    """
    import datetime
    t = datetime.datetime.strptime(ts_iso, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc)
    d = tmp / "records" / "sess" / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    (d / (name + ".json")).write_text(json.dumps({
        "runId": name, "workflowName": "qa-verdict",
        "startTime": int(t.timestamp() * 1000) + offset_ms,
        "timestamp": "2026-08-19T23:59:59.000Z",     # deliberately far away:
                                                     # a join on `timestamp`
                                                     # must NOT find this
        "status": "completed", "totalTokens": tokens,
        "args": {"step_id": step_id},
        "result": {"verdict": verdict, "violated_criteria": []},
    }), encoding="utf-8")


def drive_join(gate: Path, tmp: Path) -> dict:
    """Resolve a row against a record 900ms away -- inside 30s, outside 0s."""
    env = _env(tmp, gate.parent)
    led = tmp / "attempts.jsonl"
    ts = "2026-08-19T10:00:00Z"
    led.write_text(json.dumps({"ts": ts, "type": "attempt",
                               "step_id": REAL_SID,
                               "workflow": "qa-verdict.js"}) + "\n",
                   encoding="utf-8")
    _plant_run_record(tmp, REAL_SID, ts, 900, "FAIL", 4242)
    # A DECOY for the same step, placed exactly at the MEASURED ambiguity
    # threshold -- 386s, re-derived from the live ledger, NOT read off a
    # docstring.
    #
    # phase-90.1 cycle-3. The cycle-2 Q/A swept this and found the decoy sat at
    # 7,200,000 ms, so the cell defended the DECOY's boundary rather than the
    # real one: every tolerance from 1 to 7199 SURVIVED, including 3600 -- which
    # on the real ledger collapses summed tokens and turns most rows ambiguous
    # (measured 2026-08-21 over 106 attempt rows: 23,170,826 -> 4,015,422 and 83
    # ambiguous at 3600s; the cycle-3 figures were 20,365,361 -> 4,015,375 and 71
    # ambiguous over 89 rows -- the ledger grows, so the numbers move and the
    # corpus must be stated with them). A guard calibrated to its own fixture
    # rather than to the property is the illusory-guard shape.
    #
    # CORRECTED phase-90.1 cycle 5. This comment previously said "the DOCUMENTED
    # ambiguity threshold (the module docstring: ambiguity first appears at
    # 900s)" and "Moved to 950s", while the line below has planted 386_000 since
    # cycle 4. Both halves were stale: the docstring's 900s was itself the
    # un-corrected claim (see attempt_outcomes.py), and 950s never matched the
    # code. Measured 2026-08-21: 385s -> 0 ambiguous, 386s -> 1. The decoy sits
    # AT 386s -- unambiguous at the shipped 30s, ambiguous at or past the
    # measured threshold.
    _plant_run_record(tmp, REAL_SID, ts, 386_000, "PASS", 999999,
                      name="wf_decoy")
    resolver = gate.parent / "attempt_outcomes.py"
    r = subprocess.run([sys.executable, str(resolver), "--backfill"],
                       capture_output=True, text=True, env=env, timeout=90)
    try:
        row = json.loads(led.read_text(encoding="utf-8").splitlines()[0])
    except Exception:  # noqa: BLE001
        row = {}
    return {"rc": r.returncode, "outcome": row.get("outcome"),
            "reason": row.get("outcome_reason"),
            "tokens": row.get("total_tokens"), "run_id": row.get("run_id"),
            "stderr": r.stderr}


def drive_nested(gate: Path, tmp: Path) -> dict:
    """A step id nested under subphases[] must be ADMITTED.

    phase-90.1 cycle-2. The cycle-1 walk read only phases[].steps[] and so
    DENIED 10 real pending, harness-required steps (38.13, 46.0-46.8) while
    telling the operator they were "not a step in .claude/masterplan.json".
    Nothing in the matrix noticed, because every drive used ids the shallow
    walk happened to reach -- a recall gap, not a precision gap.
    """
    return drive_hook(gate, tmp, "9.9.1")


def drive_recall(gate: Path, tmp: Path) -> dict:
    """Every dotted id the plan contains must be admitted -- derived from the file.

    This is the check the cycle-1 blast-radius measurement could not be: it
    re-reads the plan with an INDEPENDENT walk instead of reusing the function
    under test, so the two cannot share a traversal bug.
    """
    env = _env(tmp, gate.parent)
    code = (
        "import importlib.util, json, sys" + chr(10) +
        f"spec = importlib.util.spec_from_file_location('r', {str(gate.parent / 'attempt_outcomes.py')!r})" + chr(10) +
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)" + chr(10) +
        "print(json.dumps(m.assert_membership_recall()))" + chr(10)
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=90)
    try:
        return json.loads(r.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {}


def drive_launch_row_backfill(gate: Path, tmp: Path) -> dict:
    """The backfill must survive the row shape the GATE ACTUALLY WRITES.

    phase-90.1 cycle-2, criterion 1 BLOCK. Cycle 1's fixtures all seeded the
    pre-90.1 row shape -- no resolution keys at all -- so they could never see
    that the gate now writes those keys present-and-null and that the projection
    then read null->UNKNOWN as an illegal mutation. The first real launch after
    the commit broke `--backfill`, and every fixture stayed green. This drive
    seeds the shape the gate writes, so the two halves are tested together.
    """
    env = _env(tmp, gate.parent)
    led = tmp / "attempts.jsonl"
    launch_row = {"ts": "2026-08-19T10:00:00Z", "type": "attempt",
                  "step_id": REAL_SID, "workflow": "qa-verdict.js",
                  "attempt_number_inclusive": 1,
                  "outcome": None, "outcome_reason": "unresolved_at_launch",
                  "total_tokens": None, "run_id": None,
                  "note": "recorded at launch (PreToolUse)"}
    led.write_text(json.dumps(launch_row) + "\n", encoding="utf-8")
    _plant_run_record(tmp, REAL_SID, "2026-08-19T10:00:00Z", 500, "CONDITIONAL", 777)
    resolver = gate.parent / "attempt_outcomes.py"
    r = subprocess.run([sys.executable, str(resolver), "--backfill"],
                       capture_output=True, text=True, env=env, timeout=90)
    try:
        after = json.loads(led.read_text(encoding="utf-8").splitlines()[0])
    except Exception:  # noqa: BLE001
        after = {}
    return {"rc": r.returncode, "outcome": after.get("outcome"),
            "tokens": after.get("total_tokens"),
            "note_kept": after.get("note") == launch_row["note"],
            "stderr": r.stderr}


def drive_backfill(gate: Path, tmp: Path) -> dict:
    """The resolver's additive-only invariant, driven through its own CLI."""
    env = _env(tmp, gate.parent)
    led = tmp / "attempts.jsonl"
    original = {"ts": "2026-08-19T10:00:00Z", "type": "attempt",
                "step_id": REAL_SID, "workflow": "qa-verdict.js",
                "note": "MUST SURVIVE VERBATIM"}
    led.write_text(json.dumps(original) + "\n", encoding="utf-8")
    resolver = gate.parent / "attempt_outcomes.py"
    r = subprocess.run([sys.executable, str(resolver), "--backfill"],
                       capture_output=True, text=True, env=env, timeout=90)
    try:
        after = json.loads(led.read_text(encoding="utf-8").splitlines()[0])
    except Exception:  # noqa: BLE001
        after = {}
    kept = all(after.get(k) == v for k, v in original.items())
    return {"rc": r.returncode, "kept_original_fields": kept,
            "gained_outcome": "outcome" in after,
            "gained_tokens": "total_tokens" in after,
            "outcome": after.get("outcome"), "stderr": r.stderr}


#: Exception types that mean the mutated code could not RESOLVE what it needs
#: -- i.e. it never ran -- as distinct from a domain exception the code raises
#: on purpose (AssertionError, ValueError), which is a mutant RUNNING and
#: misbehaving and must stay a KILL.
#
# phase-90.12, AFTER the research gate. Three things the literature changed here:
#
# 1. UnboundLocalError WAS MISSING, and it is a live blind spot rather than a
#    tidy-up. It subclasses NameError, but the printed name is "UnboundLocalError:",
#    which does NOT contain the substring "NameError" -- and this scan matches TYPE
#    NAMES AS STRINGS, so subclass relationships do not carry. A mutant that moves a
#    binding after its use raises it and was being scored KILLED.
# 2. TypeError is DELIBERATELY ABSENT. cosmic-ray issue #310 is this exact defect in
#    the wild, in reverse: a TypeError -- a legitimate domain error -- was classed
#    non-viable and the mutant mis-scored. Adding it would trade a false negative for
#    a false positive, and a false positive here silently DELETES a cell.
# 3. Excluding non-viable mutants from the score is settled prior art, not a local
#    invention: Stryker excludes them from the denominator (score = detected/valid),
#    PIT has a NON_VIABLE status, cosmic-ray calls them "incompetent". The published
#    risk is the ORACLE'S PRECISION, not the doctrine -- so the false-exclusion rate
#    is MEASURED on a labelled sample in verify_error_discriminator_90_12.py rather
#    than assumed to be zero.
#
# TYPE IS RUNG 4 OF A LADDER, not the whole instrument: parse -> import -> run ->
# type. Each rung catches what the one before it cannot, and the type test is the
# last resort precisely because string-matched type names are the weakest of the four.
UNRESOLVABLE_ERRORS = ("ModuleNotFoundError", "ImportError", "NameError",
                       "UnboundLocalError", "AttributeError")


def _drive_unresolvable(obs: dict) -> str | None:
    """The first unhandled traceback any drive produced, or None.

    Scored BEFORE the checks, because a mutant that crashed on every drive
    fails every check for a reason that says nothing about the guard.
    """
    # Every drive, both subject modules. Cycle-4 first scanned only the hook
    # drives and QX3 -- the same authoring slip in the RESOLVER -- still scored
    # KILLED, because its crash surfaces on the --backfill drives instead.
    #
    # But "raised an exception" is NOT "could not run", and conflating them is
    # its own defect: the first version of this scan flagged M14 as ERROR, and
    # M14 is a perfectly good mutant whose whole POINT is to reintroduce a bug
    # that raises AssertionError. Scoring a designed failure as ERROR silently
    # deletes a cell from the matrix -- the over-eager probe is as bad as the
    # blind one, which is why NULLCTL and a real-kill control are both drilled.
    #
    # The discriminator is the exception TYPE. A mutant that cannot run fails
    # to RESOLVE A NAME: the function, module or attribute it needs is not
    # there. A mutant that runs and misbehaves raises a DOMAIN exception the
    # code itself chose to raise.
    # phase-90.12 -- THE FOURTH RELOCATION OF THIS SEAM, and the one that FAILED
    # step 90.1 at cycle 5: parse -> import -> run -> RUNS BUT IS SWALLOWED.
    #
    # This scan used to require the literal string "Traceback (most recent call
    # last)". `attempt_gate.handle_hook` ends in a blanket `except Exception` that
    # prints ONE LINE -- "[attempt-gate] INTERNAL ERROR -- NameError: name 'x' is
    # not defined -- failing OPEN ..." -- and returns 0. That handler is CORRECT and
    # must stay: a broken gate must not break the harness. But it means NO failure
    # raised inside the hook's try block ever produces a traceback, so this scan
    # returned None for the entire class and the mutants scored KILLED.
    #
    # Measured: renaming ONE call site (attempt_gate.py:393,
    # extract_step_id_claim -> ..._v2) yields exit 0, that one-line stderr, and
    # traceback count ZERO. The Q/A authored three such cells and all three scored
    # KILLED where criterion 5 clause 3 requires ERROR -- while QA1b defeats no
    # guard and fails 7 of 25 checks, so a build that never runs green-washed
    # criteria 2, 3 and 4 at once.
    #
    # THE FIX IS TO READ THE TYPE, NOT THE SHAPE. A traceback is one way an
    # exception type reaches stderr; a fail-open handler that formats it into a
    # message is another. Neither is the property. The property is: did the code
    # fail to RESOLVE A NAME? That is still typed, so a DOMAIN exception -- the
    # AssertionError M14 exists to reintroduce -- stays a KILL either way, whether
    # it arrives as a traceback or through the same handler.
    for key in ("below", "at", "unknown", "over", "under", "nested",
                "backfill", "launch", "join"):
        err = (obs.get(key) or {}).get("stderr") or ""
        if not err.strip():
            continue
        # (a) an UNHANDLED traceback: the type is the last line's prefix.
        if "Traceback (most recent call last)" in err:
            last = [ln for ln in err.strip().splitlines() if ln.strip()]
            tail = last[-1] if last else ""
            if any(tail.lstrip().startswith(x) for x in UNRESOLVABLE_ERRORS):
                return f"{key} drive [traceback]: {tail[:130]}"
        # (b) a SWALLOWED exception the fail-open handler formatted into a line.
        # Anchored on "<Type>:" so it cannot fire on prose merely containing the
        # word, and typed so a domain exception through the same handler is not
        # mistaken for a mutant that could not run.
        for t in UNRESOLVABLE_ERRORS:
            m = re.search(rf"\b{t}: [^\n]+", err)
            if m:
                return f"{key} drive [swallowed by the fail-open handler]: {m.group(0)[:130]}"
    return None


def observations(gate: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        below = drive_hook(gate, Path(td), REAL_SID)
    with tempfile.TemporaryDirectory() as td:
        at = drive_hook(gate, Path(td), REAL_SID,
                        [{"ts": "2026-08-19T10:00:00Z", "type": "attempt",
                          "step_id": REAL_SID} for _ in range(5)])
    with tempfile.TemporaryDirectory() as td:
        unknown = drive_hook(gate, Path(td), FAKE_SID)
    with tempfile.TemporaryDirectory() as td:
        preserve = drive_preserve(gate, Path(td))
    with tempfile.TemporaryDirectory() as td:
        over = drive_hook(gate, Path(td), REAL_SID,
                          [{"ts": "2026-08-19T10:00:00Z", "type": "attempt",
                            "step_id": REAL_SID, "outcome": "CONDITIONAL",
                            "total_tokens": 1_200_001}])
    with tempfile.TemporaryDirectory() as td:
        under = drive_hook(gate, Path(td), REAL_SID,
                           [{"ts": "2026-08-19T10:00:00Z", "type": "attempt",
                             "step_id": REAL_SID, "outcome": "CONDITIONAL",
                             "total_tokens": 1_199_999}])
    with tempfile.TemporaryDirectory() as td:
        drop = drive_status(gate, Path(td),
                            [{"ts": "2026-08-19T10:00:00Z", "type": "attempt",
                              "step_id": REAL_SID, "outcome": "NO_VERDICT",
                              "total_tokens": 100}])
    with tempfile.TemporaryDirectory() as td:
        graded = drive_status(gate, Path(td),
                              [{"ts": "2026-08-19T10:00:00Z", "type": "attempt",
                                "step_id": REAL_SID, "outcome": "FAIL",
                                "total_tokens": 100}])
    with tempfile.TemporaryDirectory() as td:
        backfill = drive_backfill(gate, Path(td))
    with tempfile.TemporaryDirectory() as td:
        forge = drive_forge(gate, Path(td))
    with tempfile.TemporaryDirectory() as td:
        join = drive_join(gate, Path(td))
    with tempfile.TemporaryDirectory() as td:
        nested = drive_nested(gate, Path(td))
    with tempfile.TemporaryDirectory() as td:
        recall = drive_recall(gate, Path(td))
    with tempfile.TemporaryDirectory() as td:
        launch = drive_launch_row_backfill(gate, Path(td))
    return {"below": below, "at": at, "unknown": unknown, "preserve": preserve,
            "over": over, "under": under, "drop": drop, "graded": graded,
            "backfill": backfill, "forge": forge, "join": join,
            "nested": nested, "recall": recall, "launch": launch}


CHECKS = [
    # --- pre-existing behaviour that must survive this step ------------------
    ("a below-ceiling launch for a REAL step is ALLOWED and COUNTED",
     lambda o: o["below"]["rc"] == 0 and o["below"]["rows_after"] == 1),
    ("an at-ceiling launch is DENIED (exit 2) and writes the attempt_budget "
     "escalation under its unchanged name",
     lambda o: o["at"]["rc"] == 2
     and any(n.startswith("escalation_attempt_budget_") for n in o["at"]["files"])),
    ("a denied launch is NOT counted as an attempt",
     lambda o: o["at"]["rows_after"] == 5),

    # --- criterion 4: the id must resolve ------------------------------------
    ("a launch claiming an id ABSENT from the plan of record is DENIED (c4)",
     lambda o: o["unknown"]["rc"] == 2),
    ("the unknown-id denial names the rejected claim and says the launch cost "
     "nothing",
     lambda o: FAKE_SID in o["unknown"]["stderr"]
     and "not a step in" in o["unknown"]["stderr"]),
    ("the unknown-id denial writes its OWN reason-named artifact (c2/c4)",
     lambda o: any(n == f"escalation_unknown_step_id_{FAKE_SID}.md"
                   for n in o["unknown"]["files"])),
    ("an unresolvable id is NOT recorded as an attempt (it must not consume "
     "budget it was denied for claiming)",
     lambda o: o["unknown"]["rows_after"] == 0),

    # --- criterion 2: no collateral damage, proven by sha256 -----------------
    ("a NON-exhaustion denial leaves a pre-existing exhaustion escalation "
     "BYTE-IDENTICAL (c2, sha256 before == after)",
     lambda o: o["preserve"]["sha_after"] is not None
     and o["preserve"]["sha_before"] == o["preserve"]["sha_after"]),
    ("and it still wrote its own record rather than staying silent",
     lambda o: any(n.startswith("escalation_unknown_step_id_")
                   for n in o["preserve"]["files"])),

    # --- criterion 3: the token ceiling actually fires -----------------------
    ("ONE attempt over DEFAULT_MAX_TOKENS is DENIED on the token ceiling (c3)",
     lambda o: o["over"]["rc"] == 2),
    ("one token UNDER the ceiling is ALLOWED -- the check discriminates rather "
     "than always denying",
     lambda o: o["under"]["rc"] == 0),

    # --- criterion 1/5: a drop is a drop -------------------------------------
    ("a NO_VERDICT row reports dropped=1 / verdicts_seen=0 (c1, c5)",
     lambda o: o["drop"]["json"].get("dropped") == 1
     and o["drop"]["json"].get("verdicts_seen") == 0),
    ("a graded row reports dropped=0 / verdicts_seen=1 -- so the probe "
     "DISCRIMINATES and is not just always-zero",
     lambda o: o["graded"]["json"].get("dropped") == 0
     and o["graded"]["json"].get("verdicts_seen") == 1),
    ("tokens are summed from the rows rather than the constant 0 the pre-90.1 "
     "gate always reported",
     lambda o: o["drop"]["json"].get("tokens_used") == 100),

    # --- criterion 1: the backfill enriches without rewriting ---------------
    ("--backfill ADDS outcome and total_tokens to an existing row (c1)",
     lambda o: o["backfill"]["gained_outcome"] and o["backfill"]["gained_tokens"]),
    ("--backfill leaves every ORIGINAL field byte-identical (append-only "
     "enrichment, never a rewrite)",
     lambda o: o["backfill"]["kept_original_fields"]),
    ("a row with no matching run record resolves UNKNOWN, never a guess",
     lambda o: o["backfill"]["outcome"] == "UNKNOWN"),

    # --- criterion 2: the forged-exhaustion guard is EXECUTED, not assumed ---
    ("write_escalation RAISES rather than forging '# BUDGET EXHAUSTED' for a "
     "step that is not exhausted (c2)",
     lambda o: o["forge"].get("raised") is True),
    ("and it writes NO file when it refuses -- a refusal must not leave a "
     "half-record behind",
     lambda o: o["forge"].get("wrote") is None),

    # --- criterion 1: the join tolerance is EXERCISED against a real record --
    ("a row 900ms from its run record RESOLVES to the returned verdict -- the "
     "join tolerance is exercised, not assumed (c1)",
     lambda o: o["join"].get("outcome") == "FAIL"
     and o["join"].get("reason") == "graded"),
    ("and it carries that record's tokens and run_id onto the row, giving the "
     "attempt stream a shared key with the verdict ledger",
     lambda o: o["join"].get("tokens") == 4242
     and o["join"].get("run_id") == "wf_planted"),

    # --- criterion 4 RECALL: the plan's own members must be admitted --------
    ("a step id nested under subphases[] is ADMITTED -- the plan of record is "
     "not uniformly phases[].steps[] (cycle-1 BLOCK)",
     lambda o: o["nested"]["rc"] == 0),
    ("every dotted id the plan contains is admitted, checked by an INDEPENDENT "
     "walk of the file rather than by the function under test",
     lambda o: o["recall"].get("ok") is True
     and o["recall"].get("members", 0) > 0),

    # --- criterion 1: the backfill survives the row the GATE writes ---------
    ("--backfill is re-runnable against the row shape the gate ACTUALLY writes "
     "(resolution keys present-and-null), exiting 0 and filling them",
     lambda o: o["launch"]["rc"] == 0
     and o["launch"]["outcome"] == "CONDITIONAL"
     and o["launch"]["tokens"] == 777),
    ("and filling them leaves the row's other fields untouched",
     lambda o: o["launch"]["note_kept"] is True),
]

#: (id, target, description, find, replace) -- find must appear exactly once.
CELLS = [
    ("N0", GATE, "NULL MUTANT (comment only) -- must SURVIVE. If this scores "
                 "KILLED the harness is broken and every other kill this run "
                 "is meaningless.",
     "def _outcome_mix(state: BudgetState) -> dict:",
     "def _outcome_mix(state: BudgetState) -> dict:  # null mutant"),

    ("M1", GATE, "criterion 5, NAMED: a NO_VERDICT attempt is recorded as a "
                 "GRADED outcome -- the drop is laundered into a verdict",
     '        recorded = str(r.get("outcome") or "")\n'
     '        if recorded in Outcome.__members__:\n'
     '            outcome = Outcome(recorded)',
     '        recorded = str(r.get("outcome") or "")\n'
     '        if recorded == "NO_VERDICT":\n'
     '            outcome = Outcome.CONDITIONAL\n'
     '        elif recorded in Outcome.__members__:\n'
     '            outcome = Outcome(recorded)'),

    ("M2", GATE, "criterion 5, NAMED: the unresolvable-step-id DENY becomes "
                 "exit 0 -- an unrecognised key silently mints an allowance again",
     "                file=sys.stderr)\n            return 2\n        rows = read_ledger()",
     "                file=sys.stderr)\n            return 0\n        rows = read_ledger()"),

    ("M3", GATE, "the membership check is dropped -- shape validation alone, "
                 "i.e. the pre-90.1 behaviour",
     "    return sid if sid in known else None",
     "    return sid"),

    ("M4", GATE, "criterion 2: the forged '# BUDGET EXHAUSTED' fallback is "
                 "restored, so a non-exhaustion denial writes a false "
                 "exhaustion record",
     '        body = state.escalation_summary()\n'
     '        if not body:\n'
     '            raise ValueError(',
     '        body = state.escalation_summary() or (\n'
     '            f"# BUDGET EXHAUSTED -- step {state.step_id}\\n")\n'
     '        if False:\n'
     '            raise ValueError('),

    ("M5", GATE, "criterion 2: the escalation path goes back to being FIXED, "
                 "so every denial reason writes to the exhaustion file",
     '    p = ESCALATION_DIR / f"escalation_{reason}_{state.step_id}.md"',
     '    p = ESCALATION_DIR / f"escalation_attempt_budget_{state.step_id}.md"'),

    ("M6", GATE, "criterion 3: tokens stop being passed to record(), which is "
                 "exactly the pre-90.1 defect that made the ceiling inert",
     "        state.record(outcome, tokens=int(tokens) if isinstance(tokens, int) else 0,\n"
     "                     run_id=str(r.get(\"run_id\") or \"\"))",
     "        state.record(outcome)"),

    ("M7", OUTCOMES, "criterion 3: the token half of `exhausted` is removed, so "
                     "only the attempt ceiling can ever bind",
     "        return (self.attempts_used >= self.max_attempts\n"
     "                or self.tokens_used >= self.max_tokens)",
     "        return self.attempts_used >= self.max_attempts"),

    # phase-90.1 cycle-2: re-anchored. The BLOCK-2 fix rewrote this block, so
    # the cycle-1 anchor matched 0 times and the cell scored ERROR -- which is
    # the ERROR path behaving correctly (a cell that cannot be applied is not a
    # kill), but an ERROR proves nothing about the guard, so it is re-pointed at
    # the code that now carries it.
    ("M8", RESOLVER, "criterion 1: the additive-only guard is disabled, so the "
                     "backfill may silently rewrite a NON-resolution field",
     "        if projection != original:",
     "        merged[\"note\"] = \"REWRITTEN BY BACKFILL\"\n"
     "        if False:"),

    ("M9", RESOLVER, "criterion 1: an ambiguous or absent match is GUESSED "
                     "instead of resolving UNKNOWN",
     '    if not hits:\n        return unknown',
     '    if not hits:\n        return {"outcome": "PASS", "outcome_reason": "guessed",\n'
     '                "total_tokens": 0, "run_id": None}'),

    # phase-90.1 cycle-2, WARN: M10's description said "widening the tolerance"
    # while the code NARROWED 30 -> 0. Relabelled to what it does, and the two
    # things it was conflating are now separate cells that both run.
    ("M10", RESOLVER, "the join tolerance is NARROWED to 0, so a row whose run "
                      "record is milliseconds away no longer resolves",
     "DEFAULT_TOLERANCE_S = 30",
     "DEFAULT_TOLERANCE_S = 0"),

    ("M11", RESOLVER, "the join tolerance is WIDENED to the RE-MEASURED "
                      "ambiguity threshold (386s, where the first ambiguous "
                      "match actually appears on the live ledger) -- cycle 3 "
                      "calibrated this to a stale docstring figure of 900s and "
                      "the cell survived the entire [386,899] band",
     "DEFAULT_TOLERANCE_S = 30",
     "DEFAULT_TOLERANCE_S = 386"),

    ("M11b", RESOLVER, "the join tolerance is WIDENED past the measured "
                      "ambiguity threshold -- the upper bound had no cell at "
                      "all, and widening collapses token accounting (measured: "
                      "at 86400s the summed tokens fall to ~9%), which re-opens "
                      "the ceiling inertness criterion 3 exists to close",
     "DEFAULT_TOLERANCE_S = 30",
     "DEFAULT_TOLERANCE_S = 86400"),

    ("M12", RESOLVER, "the join reads `timestamp` (written at COMPLETION) "
                      "instead of `startTime` (the launch moment)",
     '        start = d.get("startTime")',
     '        start = d.get("timestamp")'),

    ("M13", RESOLVER, "the masterplan walk stops at phases[].steps[] -- the "
                      "cycle-1 BLOCK, which denied 10 pending harness-required "
                      "steps because their ids live under subphases[]",
     "    walk(mp)\n    return ids",
     "    for ph in (mp.get('phases') or []):\n"
     "        for s in ((ph or {}).get('steps') or []):\n"
     "            sid = str((s or {}).get('id') or '').strip()\n"
     "            if sid:\n"
     "                ids.add(sid)\n"
     "                if sid.startswith('phase-'):\n"
     "                    ids.add(sid[len('phase-'):])\n"
     "    return ids"),

    ("M14", RESOLVER, "the backfill re-freezes on any present key, which is the "
                      "cycle-1 BLOCK that made --backfill exit 1 the moment the "
                      "gate wrote a null-placeholder launch row",
     "        projection = {k: merged[k] for k in parsed\n"
     "                      if k in merged and k not in RESOLUTION_KEYS}\n"
     "        original = {k: v for k, v in parsed.items() if k not in RESOLUTION_KEYS}",
     "        projection = {k: merged[k] for k in parsed if k in merged}\n"
     "        original = dict(parsed)"),
]


def run_cell(cell) -> dict:
    cid, target, desc, find, repl = cell
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "harness"
        shutil.copytree(GATE.parent, work)
        mutant_file = work / target.name
        src = mutant_file.read_text(encoding="utf-8")
        n = src.count(find)
        if n != 1:
            return {"id": cid, "desc": desc, "score": "ERROR",
                    "why": f"anchor appears {n} times in {target.name}, expected 1"}
        mutated = src.replace(find, repl, 1)
        mutant_file.write_text(mutated, encoding="utf-8")
        # phase-90.1 cycle-2, WARN: `subprocess.run` does not raise on a
        # non-zero exit, so a mutant that could not even be IMPORTED came back
        # with every check failing and was credited as a KILL. A build failure
        # is not evidence about the guard. Parse it first; unparseable is ERROR.
        try:
            ast.parse(mutated)
        except SyntaxError as exc:
            return {"id": cid, "desc": desc, "score": "ERROR",
                    "why": f"mutant does not parse ({target.name}:{exc.lineno}): "
                           f"{exc.msg} -- a build failure is not a kill"}
        # phase-90.1 cycle-3: PARSING IS NOT RUNNING. The cycle-2 fix added the
        # ast.parse above and closed only the SyntaxError subset; the cycle-2
        # Q/A then executed three mutants that parse cleanly and still cannot be
        # imported -- a module-scope RuntimeError, a NameError, and an
        # ImportError -- and every one scored KILLED, because subprocess.run
        # does not raise on a non-zero exit and the mutant simply failed every
        # check. So the mutant is now IMPORTED before any check runs, and a
        # non-zero import is ERROR. This is what criterion 5 clause 3 actually
        # asks for: "a mutant that fails to RUN scores ERROR".
        probe = subprocess.run(
            [sys.executable, "-c",
             # The module MUST be registered in sys.modules before exec_module:
             # `@dataclass` resolves its annotations through the module object,
             # and without the registration a perfectly importable file raises
             # AttributeError: 'NoneType' object has no attribute '__dict__'.
             # Measured on attempt_budget.py, which scored a spurious ERROR
             # until this was added -- a probe that reports a false failure is
             # as bad as one that misses a real one.
             "import importlib.util,sys;"
             f"spec=importlib.util.spec_from_file_location('m',{str(mutant_file)!r});"
             "mod=importlib.util.module_from_spec(spec);"
             "sys.modules['m']=mod;spec.loader.exec_module(mod)"],
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join([
                str(work), str(REPO / "scripts" / "harness"),
                str(REPO / "scripts" / "qa"), os.environ.get("PYTHONPATH", "")])))
        if probe.returncode != 0:
            last = [ln for ln in (probe.stderr or "").strip().splitlines() if ln.strip()]
            return {"id": cid, "desc": desc, "score": "ERROR",
                    "why": f"mutant parses but does NOT import ({target.name}): "
                           f"{last[-1][:140] if last else 'non-zero exit'} "
                           "-- a mutant that cannot run is not a kill"}
        gate = work / GATE.name
        try:
            obs = observations(gate)
        except Exception as exc:  # noqa: BLE001
            return {"id": cid, "desc": desc, "score": "ERROR",
                    "why": f"mutant did not run: {type(exc).__name__}: {exc}"}
        # phase-90.1 cycle-4. THIRD relocation of the same seam: parse -> import
        # -> RUN. The cycle-3 smoke-import closes the IMPORT seam only, and the
        # cycle-3 Q/A executed three mutants that parse AND import cleanly and
        # still cannot run -- a deferred missing import on the hook branch, and
        # `handle_hook` -> `handle_hook_v2` (the realistic authoring slip) in
        # each subject module. All three scored KILLED, and failure-COUNT gave
        # no signal: QX3 failed 5 of 25 checks, exactly like the genuine kill M3.
        #
        # The discriminator the Q/A measured, and which I re-measured: a mutant
        # that cannot run dies with an UNHANDLED TRACEBACK on a drive's stderr,
        # while all 16 shipped cells produce none. Note what does NOT work: a
        # benign-path smoke (`--status 9.1`) returns rc=0 for those mutants,
        # because the mutation sits on the hook branch only.
        broken = _drive_unresolvable(obs)
        if broken:
            return {"id": cid, "desc": desc, "score": "ERROR",
                    "why": f"mutant imports but does NOT run: {broken} "
                           "-- a mutant that cannot run is not a kill"}
        failed = []
        for name, fn in CHECKS:
            try:
                if not fn(obs):
                    failed.append(name)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{name} [raised {type(exc).__name__}]")
        if not failed:
            return {"id": cid, "desc": desc, "score": "SURVIVED", "why": ""}
        return {"id": cid, "desc": desc, "score": "KILLED",
                "why": "; ".join(failed[:3])}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verify = "--verify" in argv
    fingerprints = {p: md5(p) for p in (GATE, OUTCOMES, RESOLVER)}

    print("== CONTROL (real tree, unmutated) ==")
    try:
        base = observations(GATE)
    except Exception as exc:  # noqa: BLE001
        print(f"  CONTROL DID NOT RUN: {type(exc).__name__}: {exc}")
        print("  Scoring nothing. A matrix whose control cannot run proves "
              "nothing about its mutants.")
        return 1
    control_failed = [n for n, fn in CHECKS if not fn(base)]
    for name, fn in CHECKS:
        print(f"  {'ok  ' if fn(base) else 'FAIL'}  {name}")
    if control_failed:
        print("\nCONTROL IS NOT GREEN -- refusing to score cells. "
              "A red control means a kill cannot be attributed to the mutation.")
        return 1
    print("  CONTROL GREEN\n")

    print("== CELLS ==")
    results = [run_cell(c) for c in CELLS]
    for r in results:
        print(f"  {r['score']:<9} {r['id']:<4} {r['desc']}")
        if r["why"]:
            print(f"                 -> {r['why']}")

    after = {p: md5(p) for p in (GATE, OUTCOMES, RESOLVER)}
    untouched = fingerprints == after
    print(f"\nreal tree untouched (md5 before == after): {untouched}")
    for p in (GATE, OUTCOMES, RESOLVER):
        print(f"  {p.relative_to(REPO)}: {after[p]}")

    survivors = [r for r in results if r["score"] == "SURVIVED" and r["id"] != "N0"]
    errors = [r for r in results if r["score"] == "ERROR"]
    null = next((r for r in results if r["id"] == "N0"), None)
    null_ok = null is not None and null["score"] == "SURVIVED"
    killed = sum(1 for r in results if r["score"] == "KILLED")
    print(f"\nKILLED {killed} | SURVIVED {len(survivors)} (excl. N0) | "
          f"ERROR {len(errors)} | null mutant survived: {null_ok}")
    if not null_ok:
        print("NULL MUTANT WAS KILLED -- the harness, not the code, is what "
              "these checks are detecting. Every kill above is void.")
    if verify:
        bad = survivors or errors or not null_ok or not untouched
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
