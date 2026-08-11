#!/usr/bin/env python3
"""phase-86.33 criterion 4 + 6 -- mutate the guard, prove the checks discriminate.

The subjects are the two provers, driven as whole programs:

    scripts/qa/prove_qa_write_separation_86_31.py     15 cases, 8 BLOCK / 7 ALLOW
    scripts/qa/prove_researcher_rail_unbroken_86_33.py 34 spellings + 1 control

CELL M3 IS THE ONE THAT MATTERS, and it is criterion 6.
`.claude/hooks/qa-write-guard.sh` embeds its python inside a bash SINGLE-QUOTED
block. A single apostrophe anywhere in that body terminates the block early, the
hook stops being valid bash, and it degrades to **allow-everything, silently**. That
is not hypothetical: the log carries traces from 2026-08-10 where exactly this was
demonstrated by a deliberate probe. M3 injects one apostrophe and requires the
checks to notice.

DISCIPLINE
----------
Control runs FIRST and the run aborts if it is not green -- a mutation score against
a red control measures nothing. An anchor that does not match, or matches more than
once, is reported rather than silently mutating nothing, because a no-op replacement
leaves everything green and is indistinguishable from a surviving mutant. The guard
is restored from a byte copy in a `finally:` block and the restore is verified by
hash.

Run:  python scripts/qa/mutation_matrix_86_33.py
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
GUARD = REPO / ".claude" / "hooks" / "qa-write-guard.sh"
SEPARATION = REPO / "scripts" / "qa" / "prove_qa_write_separation_86_31.py"
RESEARCHER = REPO / "scripts" / "qa" / "prove_researcher_rail_unbroken_86_33.py"

CELLS = [
    {
        "id": "M1-exact-qa-match",
        "what": "revert the 86.31 widening: match ONLY the literal agent_type 'qa'",
        "old": 'if is_qa_role(agent_type) and tool_name in ("Write", "Edit"):',
        "new": 'if agent_type == "qa" and tool_name in ("Write", "Edit"):',
        "expect_red": ["separation"],
        "why": "the 27 named qa-* spawns walk straight past an exact match",
    },
    {
        "id": "M2-drop-payload-keys",
        "what": "stop recording the payload key set (criterion 2's measurement)",
        "old": '"file_path": file_path,\n                       "payload_keys": payload_keys})',
        "new": '"file_path": file_path})',
        "expect_red": ["keyset"],
        "why": "criterion 2 cannot be answered if the key set is never recorded",
    },
    {
        "id": "M3-apostrophe-trap",
        "what": "CRITERION 6: inject ONE apostrophe into the single-quoted python body",
        "old": "MEMORY_DIR = \".claude/agent-memory/qa/\"",
        "new": "MEMORY_DIR = \".claude/agent-memory/qa/\"  # the guards own dir isn't safe",
        "expect_red": ["liveness"],
        "why": "one apostrophe ends the bash single-quoted block and the guard "
               "silently degrades to allow-everything",
    },
]


def run(script: pathlib.Path) -> bool:
    p = subprocess.run([sys.executable, str(script)], cwd=REPO,
                       capture_output=True, text=True)
    return p.returncode == 0


def guard_is_alive() -> bool:
    """bash -n: does the hook still PARSE? The apostrophe trap breaks exactly this."""
    p = subprocess.run(["bash", "-n", str(GUARD)], capture_output=True, text=True)
    return p.returncode == 0


def keyset_recorded() -> bool:
    """Does driving the real hook still record a payload key set?

    Reads the LOG FILE, not our captured stderr: the hook routes its own stderr
    into handoff/logs/qa_write_guard.log (`2>>"$GUARD_LOG"` at :132), so a
    subprocess capture sees nothing. The control-green-first rule caught this --
    an earlier revision of this check read stderr and was RED against a healthy
    guard, which would have made every later cell meaningless.
    """
    import json
    log = REPO / "handoff" / "logs" / "qa_write_guard.log"
    before = log.stat().st_size if log.exists() else 0
    payload = {"agent_type": "qa", "agent_id": "aPROBE", "tool_name": "Write",
               "tool_input": {"file_path": ".claude/agent-memory/qa/verdicts/v.md"},
               "hook_event_name": "PreToolUse"}
    subprocess.run(["bash", str(GUARD)], input=json.dumps(payload),
                   capture_output=True, text=True, cwd=REPO)
    if not log.exists():
        return False
    with log.open("r", errors="replace") as fh:
        fh.seek(before)
        for line in fh:
            i = line.find("{")
            if i < 0:
                continue
            try:
                if json.loads(line[i:]).get("payload_keys"):
                    return True
            except Exception:  # noqa: BLE001
                continue
    return False


CHECKS = {
    "liveness":   ("guard parses (bash -n)", guard_is_alive),
    "separation": ("qa write-separation prover", lambda: run(SEPARATION)),
    "researcher": ("researcher rail prover", lambda: run(RESEARCHER)),
    "keyset":     ("payload key set recorded", keyset_recorded),
}


def evaluate() -> dict[str, bool]:
    return {k: fn() for k, (_lbl, fn) in CHECKS.items()}


def main() -> int:
    src = GUARD.read_text()
    md5_before = hashlib.md5(src.encode()).hexdigest()
    backup = pathlib.Path(tempfile.gettempdir()) / "qa_write_guard_86_33_backup.sh"
    backup.write_text(src)

    print("=" * 80)
    print("phase-86.33 mutation matrix -- qa-write-guard")
    print("=" * 80)
    print(f"\n[target] {GUARD.relative_to(REPO)}  md5={md5_before}")

    control = evaluate()
    print("[control]")
    for k, ok in control.items():
        print(f"    {CHECKS[k][0]:<32} {'GREEN' if ok else 'RED'}")
    if not all(control.values()):
        print("\n  ABORT -- control is not fully green; no later red would mean anything.")
        return 1

    results = []
    try:
        for cell in CELLS:
            n = src.count(cell["old"])
            if n != 1:
                results.append((cell["id"], f"ANCHOR-{'MISS' if n == 0 else 'AMBIGUOUS'}", []))
                print(f"\n[{cell['id']}] anchor matches {n}x -- vacuous, NOT a pass")
                continue
            GUARD.write_text(src.replace(cell["old"], cell["new"], 1))
            res = evaluate()
            red = [k for k, ok in res.items() if not ok]
            named_red = [k for k in cell["expect_red"] if k in red]
            status = "KILLED" if named_red else ("MIS-ATTRIB" if red else "SURVIVED")
            results.append((cell["id"], status, red))
            print(f"\n[{cell['id']}] {status}  ({cell['what']})")
            print(f"    why: {cell['why']}")
            print(f"    expected red: {cell['expect_red']}")
            print(f"    actual  red: {red or '(none)'}")
    finally:
        GUARD.write_text(backup.read_text())
        restored = hashlib.md5(GUARD.read_text().encode()).hexdigest() == md5_before
        print(f"\n[restore] byte-identical: {restored}")

    print("\n" + "=" * 80)
    for cid, status, red in results:
        print(f"  {status:<16} {cid}")
    print("=" * 80)
    if not restored:
        print(f"RESULT: GUARD NOT RESTORED -- recover from {backup}")
        return 2
    bad = [r for r in results if r[1] != "KILLED"]
    if bad:
        print(f"RESULT: {len(bad)} of {len(results)} cells NOT killed.")
        return 1
    after = evaluate()
    print(f"post-restore all green: {all(after.values())}")
    print(f"RESULT: all {len(results)} cells KILLED, control green, guard restored.")
    return 0 if all(after.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
