#!/usr/bin/env python3
"""Mutation matrix for `scripts/harness/attempt_gate.py` (phase-86.71).

House discipline (per mutate_rail_turn_cap.py / mutation_matrix_86_85.py):
CONTROL observed GREEN before any cell; a cell is KILLED only when a named
check fails for a stated reason; a mutant that does not run is ERROR, never a
kill; the real tree is never written (md5 before == after, mutants run from a
temp copy via subprocess so the CALL SITE is what is tested -- the project's
guards-stop-one-seam-short lesson).

    python3 scripts/qa/mutation_matrix_86_71.py            # report
    python3 scripts/qa/mutation_matrix_86_71.py --verify   # exit 1 on a survivor
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "scripts" / "harness" / "attempt_gate.py"

HOOK_STDIN = json.dumps({
    "tool_name": "Workflow",
    "tool_input": {"scriptPath": ".claude/workflows/qa-verdict.js",
                   "args": {"step_id": "77.7"}},
    "tool_use_id": "toolu_mm86_71", "session_id": "mm",
})


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def drive(gate: Path, tmp: Path, seed_attempts: int,
          seed_corrupt: int = 0, verdict_ledger_isdir: bool = False) -> dict:
    """Run the gate as a SUBPROCESS with a temp ledger seeded to N attempts.

    phase-86.71 cycle-2 (Q/A finding, Circular_Reasoning): the first version
    set no PYTHONPATH, so a mutant relocated to the temp dir raised
    ModuleNotFoundError on `from attempt_budget import ...` (attempt_gate
    computes REPO from ITS OWN __file__) and died before any gate logic --
    every check failed, every cell scored KILLED, including a null mutant.
    The env now carries the import path, and the discrimination controls
    below make that class of harness breakage loud forever.

    `seed_corrupt` prepends N non-JSON lines: a corrupt row must COUNT as an
    attempt (over-count escalates early, the safe direction), which is the
    behaviour cell G4 mutates -- previously unexercised by these checks.
    """
    led = tmp / "attempts.jsonl"
    with led.open("w", encoding="utf-8") as fh:
        for _ in range(seed_corrupt):
            fh.write("not json -- corrupt attempt row\n")
        for _ in range(seed_attempts):
            fh.write(json.dumps({"ts": "2026-08-17T00:00:00Z", "type": "attempt",
                                 "step_id": "77.7"}) + "\n")
    # cycle-5 (cycle-4 Q/A, V1/V2): the ABSENT path returns [] quietly and
    # never reaches verdict_outcomes' except branch, which is why the loud-
    # swallow fix had zero automated coverage. A DIRECTORY raises
    # IsADirectoryError and reaches it.
    if verdict_ledger_isdir:
        vpath = tmp / "verdict_isadir"
        vpath.mkdir(exist_ok=True)
    else:
        vpath = tmp / "absent_verdicts.jsonl"
    env = dict(os.environ,
               ATTEMPT_GATE_LEDGER=str(led),
               ATTEMPT_GATE_VERDICT_LEDGER=str(vpath),
               ATTEMPT_GATE_ESCALATION_DIR=str(tmp),
               PYTHONPATH=os.pathsep.join([
                   str(REPO / "scripts" / "harness"),
                   str(REPO / "scripts" / "qa"),
                   os.environ.get("PYTHONPATH", ""),
               ]))
    r = subprocess.run([sys.executable, str(gate)], input=HOOK_STDIN,
                       capture_output=True, text=True, env=env, timeout=60)
    rows_after = (led.read_text(encoding="utf-8").count("\n")
                  if led.is_file() else 0)
    return {"rc": r.returncode, "stderr": r.stderr, "rows_after": rows_after,
            "escalation_written": any(p.name.startswith("escalation_attempt_budget_")
                                      for p in tmp.iterdir())}


def _corrupt_probe(gate: Path) -> dict:
    """Does the gate's reader TAG a corrupt row as an attempt?

    cycle-2: G4's subject is read_ledger's corrupt-row handling; the hook
    drives never exercised it (well-formed seeds only), so its filed kill was
    actually made by --self-test -- a different artifact. This probe runs the
    MUTANT's own read_ledger in a subprocess, so the matrix's checks own the
    kill.
    """
    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "l.jsonl"
        led.write_text("not json\n", encoding="utf-8")
        code = (
            "import importlib.util, json, sys\n"
            f"spec = importlib.util.spec_from_file_location('g', {str(gate)!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            f"rows = m.read_ledger(m.Path({str(led)!r}))\n"
            "print(json.dumps([r.get('step_id') for r in rows]))\n"
        )
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([
            str(REPO / "scripts" / "harness"), str(REPO / "scripts" / "qa"),
            os.environ.get("PYTHONPATH", "")]))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, timeout=60)
        try:
            tags = json.loads(r.stdout.strip() or "[]")
        except json.JSONDecodeError:
            tags = []
        return {"corrupt_tagged": tags == ["__corrupt__"], "rc": r.returncode}


def observations(gate: Path) -> dict:
    """The behavioural fingerprint every cell is scored against."""
    with tempfile.TemporaryDirectory() as td:
        below = drive(gate, Path(td), seed_attempts=0)
    with tempfile.TemporaryDirectory() as td:
        at = drive(gate, Path(td), seed_attempts=5)
    with tempfile.TemporaryDirectory() as td:
        at_vlerr = drive(gate, Path(td), seed_attempts=5,
                         verdict_ledger_isdir=True)
    probe = _corrupt_probe(gate)
    ext = _extend_probe(gate)
    return {"below": below, "at": at, "at_vlerr": at_vlerr,
            "corrupt": probe, "extend": ext}


CHECKS = [
    ("below-ceiling launch is ALLOWED", lambda o: o["below"]["rc"] == 0),
    ("below-ceiling launch is COUNTED (row appended at the call site)",
     lambda o: o["below"]["rows_after"] == 1),
    ("at-ceiling launch is DENIED with exit 2",
     lambda o: o["at"]["rc"] == 2),
    ("the denial names the escalation and the operator command",
     lambda o: "operator-extend" in o["at"]["stderr"]),
    ("the denial writes the escalation file",
     lambda o: o["at"]["escalation_written"]),
    ("a denied launch is NOT counted as an attempt",
     lambda o: o["at"]["rows_after"] == 5),
    ("a corrupt ledger row is TAGGED as an attempt by the reader (over-count "
     "escalates early -- the safe direction)",
     lambda o: o["corrupt"]["corrupt_tagged"] and o["corrupt"]["rc"] == 0),
    ("an operator extension WITHOUT --reason is REFUSED and appends no row",
     lambda o: o["extend"]["refused_rc"] == 2
     and o["extend"]["rows_after_refusal"] == 0),
    ("an operator extension WITH a reason appends its labelled row",
     lambda o: o["extend"]["accepted_rc"] == 0
     and o["extend"]["rows_after_accept"] == 1),
    # cycle-5 (cycle-4 Q/A, V1/V2): both properties of the loud fail-closed
    # swallow, driven through the branch a directory-ledger makes reachable.
    ("a verdict-ledger read error is LOUD on stderr (the swallow is observable)",
     lambda o: "verdict-ledger read failed" in o["at_vlerr"]["stderr"]),
    ("a verdict-ledger read error grants NO PASS exception -- at ceiling stays "
     "DENIED (fail-closed direction)",
     lambda o: o["at_vlerr"]["rc"] == 2),
]

#: (id, description, find, replace) -- find must appear exactly once.
CELLS = [
    ("G1", "deny branch removed -- exhaustion silently allows",
     "        if decision == \"deny\":",
     "        if False:"),
    ("G2", "attempt row write dropped at the CALL SITE -- launches stop being counted",
     "        append_row({\n            \"ts\": _now(), \"type\": \"attempt\", \"step_id\": sid,",
     "        _ = ({\n            \"ts\": _now(), \"type\": \"attempt\", \"step_id\": sid,"),
    ("G3", "step-id extraction neutered -- every launch reads as unattributable",
     "    args = tool_input.get(\"args\")",
     "    args = None"),
    ("G4", "corrupt ledger row silently skipped -- the count can only shrink",
     "            rows.append({\"step_id\": \"__corrupt__\", \"type\": \"attempt\"})",
     "            continue"),
    ("G5", "deny demoted to allow -- exit 2 becomes exit 0 with the message kept",
     "            return 2\n        append_row({",
     "            return 0\n        append_row({"),
    ("G6", "ceiling comparison bypassed -- disposition read but ignored",
     "    if d is Disposition.ESCALATE:\n        return \"deny\", state",
     "    if d is Disposition.ESCALATE:\n        return \"allow\", state"),
    # cycle-3 (cycle-2 Q/A, H4): the accountability guard on the ONLY path
    # that raises the ceiling.
    ("G7", "--reason requirement removed from operator-extend -- a silent, "
           "unexplained ceiling raise becomes possible",
     "    if not reason.strip():",
     "    if False:"),
    # cycle-5 (cycle-4 Q/A): the evaluator's own two surviving mutants of the
    # loud fail-closed swallow, kept permanent. Both anchor on the full
    # print+return block of verdict_outcomes' except branch.
    ("G8", "loud swallow reverted to SILENT (V1) -- the ledger-read failure "
           "vanishes from stderr while the fail-closed direction is kept",
     "        print(f\"[attempt-gate] verdict-ledger read failed for step {step_id}: \"\n"
     "              f\"{type(exc).__name__}: {exc} -- proceeding WITHOUT the PASS \"\n"
     "              \"exception (fail-closed: this can only deny more, never allow \"\n"
     "              \"more)\", file=sys.stderr)\n"
     "        return []",
     "        return []"),
    ("G9", "swallow made fail-OPEN (V2) -- a ledger read error grants the "
           "permanent PASS exception and bypasses the budget entirely",
     "        print(f\"[attempt-gate] verdict-ledger read failed for step {step_id}: \"\n"
     "              f\"{type(exc).__name__}: {exc} -- proceeding WITHOUT the PASS \"\n"
     "              \"exception (fail-closed: this can only deny more, never allow \"\n"
     "              \"more)\", file=sys.stderr)\n"
     "        return []",
     "        return [Outcome.PASS]"),
]


def _extend_probe(gate: Path) -> dict:
    """Drive cmd_extend as a SUBPROCESS: refused without --reason, row with."""
    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "l.jsonl"
        env = dict(os.environ,
                   ATTEMPT_GATE_LEDGER=str(led),
                   ATTEMPT_GATE_VERDICT_LEDGER=str(Path(td) / "v.jsonl"),
                   ATTEMPT_GATE_ESCALATION_DIR=str(td),
                   PYTHONPATH=os.pathsep.join([
                       str(REPO / "scripts" / "harness"),
                       str(REPO / "scripts" / "qa"),
                       os.environ.get("PYTHONPATH", "")]))
        r_no = subprocess.run([sys.executable, str(gate), "--operator-extend",
                               "77.8", "--by", "1"],
                              capture_output=True, text=True, env=env, timeout=60)
        rows_no = led.read_text(encoding="utf-8").count("\n") if led.is_file() else 0
        r_yes = subprocess.run([sys.executable, str(gate), "--operator-extend",
                                "77.8", "--by", "1", "--reason", "matrix drive"],
                               capture_output=True, text=True, env=env, timeout=60)
        rows_yes = led.read_text(encoding="utf-8").count("\n") if led.is_file() else 0
        return {"refused_rc": r_no.returncode, "rows_after_refusal": rows_no,
                "accepted_rc": r_yes.returncode, "rows_after_accept": rows_yes}


def run_matrix(verify: bool) -> int:
    before = md5(TARGET)
    print("MUTATION MATRIX -- scripts/harness/attempt_gate.py (phase-86.71)")
    print("=" * 78)

    src = TARGET.read_text(encoding="utf-8")
    ctrl = observations(TARGET)
    ctrl_fail = [name for name, fn in CHECKS if not fn(ctrl)]
    if ctrl_fail:
        print("CONTROL IS RED -- the matrix is meaningless. Failing checks:")
        for n in ctrl_fail:
            print(f"  - {n}")
        return 1
    print(f"CONTROL green: all {len(CHECKS)} behavioural checks hold "
          f"(below rc={ctrl['below']['rc']} rows={ctrl['below']['rows_after']}; "
          f"at-ceiling rc={ctrl['at']['rc']})")

    # phase-86.71 cycle-2 DISCRIMINATION CONTROLS -- permanent, abort on
    # failure. The Q/A proved the first harness scored an UNMUTATED copy and
    # a NULL MUTANT as KILLED (ModuleNotFoundError at the temp path), i.e.
    # the matrix could not tell a mutant from the subject. These two controls
    # make that breakage loud forever.
    with tempfile.TemporaryDirectory() as td:
        clean = Path(td) / "attempt_gate_clean.py"
        clean.write_text(src, encoding="utf-8")
        relocated = observations(clean)
        rel_fail = [name for name, fn in CHECKS if not fn(relocated)]
    if rel_fail:
        print("DISCRIMINATION CONTROL RED: an UNMUTATED copy at the temp path "
              "fails checks -- the harness cannot run relocated code and every "
              "kill below would be a mirage. Failing:")
        for n in rel_fail:
            print(f"  - {n}")
        return 1
    print("relocated-unmutated control: SURVIVES (all checks hold)")
    with tempfile.TemporaryDirectory() as td:
        nullm = Path(td) / "attempt_gate_null.py"
        nullm.write_text(src.replace("def _now() -> str:",
                                     "def _now() -> str:  # null-mutant marker",
                                     1), encoding="utf-8")
        nullobs = observations(nullm)
        null_fail = [name for name, fn in CHECKS if not fn(nullobs)]
    if null_fail:
        print("NULL-MUTANT CONTROL RED: a comment-only change fails checks -- "
              "the harness kills everything and discriminates nothing. Failing:")
        for n in null_fail:
            print(f"  - {n}")
        return 1
    print("null-mutant control: SURVIVES (a comment-only change kills nothing)\n")

    survivors, errors = [], []
    for cid, desc, find, repl in CELLS:
        n = src.count(find)
        if n != 1:
            errors.append((cid, f"anchor matched {n}x, expected 1"))
            print(f"  {cid}  ERROR      anchor matched {n}x -- {desc}")
            continue
        mutated = src.replace(find, repl, 1)
        with tempfile.TemporaryDirectory() as td:
            mgate = Path(td) / "attempt_gate_mut.py"
            mgate.write_text(mutated, encoding="utf-8")
            try:
                obs = observations(mgate)
                # cycle-3 (cycle-2 Q/A, latent finding): a mutant that cannot
                # even IMPORT has been tested by nothing -- score it ERROR,
                # never let its universal check-failure read as a kill. The
                # smaller form of the cycle-1 class, closed at the cell level.
                # cycle-4 (cycle-3 Q/A, probe Z3): the marker list was
                # three names wide and a NameError-at-import mutant scored
                # KILLED. The class test is now ANY Python traceback OR an
                # exit code outside the gate's two legitimate outcomes
                # (0 allow / 2 deny) -- a crashed mutant has been tested by
                # nothing, whatever the exception was called.
                # cycle-5 (cycle-4 Q/A named fix, stronger OR-branch taken):
                # the crash test covers ALL THREE hook drives, not only
                # "below" -- a mutant that crashes only at-ceiling or only
                # under the isdir verdict-ledger must score ERROR, never a
                # kill. NOTE at_vlerr's stderr legitimately carries the loud
                # IsADirectoryError DISCLOSURE (no traceback); the marker is
                # the traceback header, so the disclosure does not trip this.
                # Auxiliary probes (_corrupt_probe/_extend_probe) stay outside
                # this guard's scope, stated in the artifacts.
                broken = any(
                    "Traceback (most recent call last)" in o.get("stderr", "")
                    or o.get("rc") not in (0, 2)
                    for o in (obs["below"], obs["at"], obs["at_vlerr"]))
                if broken:
                    errors.append((cid, "mutant crashed in a hook drive -- not tested"))
                    print(f"  {cid}  ERROR      mutant crashed in a hook drive -- {desc}")
                    continue
                failed = [name for name, fn in CHECKS if not fn(obs)]
            except Exception as exc:  # noqa: BLE001
                errors.append((cid, f"{type(exc).__name__}: {exc}"))
                print(f"  {cid}  ERROR      mutant did not run -- {desc}")
                continue
        if failed:
            print(f"  {cid}  KILLED     {desc}")
            print(f"            by: {failed[0]}")
        else:
            survivors.append((cid, desc))
            print(f"  {cid}  SURVIVED   {desc}   <-- REAL SURVIVOR")

    after = md5(TARGET)
    print(f"\nBYTE-IDENTICAL RESTORE: {'ok' if before == after else 'CHANGED'} "
          f"(md5 {after}; mutants ran from temp copies, the real tree was never written)")
    print(f"cells={len(CELLS)}  killed={len(CELLS) - len(survivors) - len(errors)}  "
          f"real survivors={len(survivors)}  errors={len(errors)}")

    if verify:
        if before != after or survivors or errors:
            print("\nVERIFY: FAIL")
            return 1
        print("\nVERIFY: PASS -- control green, 0 survivors, 0 errors, tree unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(run_matrix("--verify" in sys.argv[1:]))
