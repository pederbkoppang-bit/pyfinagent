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


def drive(gate: Path, tmp: Path, seed_attempts: int) -> dict:
    """Run the gate as a SUBPROCESS with a temp ledger seeded to N attempts."""
    led = tmp / "attempts.jsonl"
    with led.open("w", encoding="utf-8") as fh:
        for _ in range(seed_attempts):
            fh.write(json.dumps({"ts": "2026-08-17T00:00:00Z", "type": "attempt",
                                 "step_id": "77.7"}) + "\n")
    env = dict(os.environ,
               ATTEMPT_GATE_LEDGER=str(led),
               ATTEMPT_GATE_VERDICT_LEDGER=str(tmp / "absent_verdicts.jsonl"),
               ATTEMPT_GATE_ESCALATION_DIR=str(tmp))
    r = subprocess.run([sys.executable, str(gate)], input=HOOK_STDIN,
                       capture_output=True, text=True, env=env, timeout=60)
    rows_after = (led.read_text(encoding="utf-8").count("\n")
                  if led.is_file() else 0)
    return {"rc": r.returncode, "stderr": r.stderr, "rows_after": rows_after,
            "escalation_written": any(p.name.startswith("escalation_attempt_budget_")
                                      for p in tmp.iterdir())}


def observations(gate: Path) -> dict:
    """The behavioural fingerprint every cell is scored against."""
    with tempfile.TemporaryDirectory() as td:
        below = drive(gate, Path(td), seed_attempts=0)
    with tempfile.TemporaryDirectory() as td:
        at = drive(gate, Path(td), seed_attempts=5)
    return {"below": below, "at": at}


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
]


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
          f"at-ceiling rc={ctrl['at']['rc']})\n")

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
