#!/usr/bin/env python3
"""Mutation matrix for phase-86.85's writer -- `scripts/qa/verdict_ledger_write.py`.

A guard that cannot fail does not count. Each cell REVERTS one guard and asserts
the writer's own `--self-test` goes RED. The control is observed GREEN first; a
cell whose control was not green is scored UNSCORABLE rather than KILLED, because
a red-on-red proves nothing.

ZERO REPO WRITES. Every mutation is applied to a COPY under the OS tmpdir and the
mutated copy is executed there. The real `verdict_ledger_write.py` is never opened
for writing -- its sha256 is printed before and after so that claim is checkable
rather than asserted. This avoids the restore step entirely, which is the only way
to be sure a restore was not gotten wrong.

Run:  python scripts/qa/mutation_matrix_86_85.py
Exit: 0 if every cell KILLED, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "verdict_ledger_write.py"

#: (cell, description, find, replace). `find` must appear EXACTLY once -- an
#: operation that cannot fail loudly is its own defect, so a no-match or a
#: multi-match is an error, not a silent skip.
CELLS: list[tuple[str, str, str, str]] = [
    (
        "M1",
        "remove the dedup refusal -- a duplicate (step_id, run_id) must append",
        "    if key in existing_keys(path):",
        "    if False:",
    ),
    (
        "M2",
        "remove the verdict vocabulary guard -- an unknown verdict must be written",
        # Anchored WITH its preceding line: the bare `if verdict not in
        # VALID_VERDICTS:` is a SUBSTRING of emit_sequence's more-indented copy
        # (M7's anchor), so on its own it matches twice and the cell scores
        # UNSCORABLE rather than KILLED. `str.count` does substring matching, not
        # line matching -- an anchor must be unique as TEXT, not merely as a line.
        '    verdict = (verdict or "").strip().upper()\n    if verdict not in VALID_VERDICTS:',
        '    verdict = (verdict or "").strip().upper()\n    if False:',
    ),
    (
        "M3",
        "allow an unkeyed row -- a row with neither run_id nor cycle must append",
        '    raise LedgerError(\n        "no dedup key available',
        '    return (step, "")\n    raise LedgerError(\n        "no dedup key available',
    ),
    (
        "M4",
        "swallow a corrupt ledger line -- a corrupt file must read as 'no verdicts'",
        "            raise LedgerError(\n                f\"{path}:{lineno} is not valid JSON",
        "            continue\n            raise LedgerError(\n                f\"{path}:{lineno} is not valid JSON",
    ),
    (
        "M5",
        "collapse event time into write time -- a backfill masquerades as history",
        '        "date": event_date or stamp.date().isoformat(),',
        '        "date": stamp.isoformat(),',
    ),
    # ---- cells added after the cycle-1 Q/A found M1-M5 blind to ordering ----
    (
        "M6",
        "REVERSE emit_sequence -- this is the cycle-1 Q/A's QA-M1, which SURVIVED "
        "against the old palindromic fixture. Reversing [PASS,C,C] to [C,C,PASS] "
        "takes enforceEscalation from n=2/auto_fail=true to n=0/auto_fail=false, "
        "silently DISARMING the escalation",
        "\n    return out\n",
        "\n    return out[::-1]\n",
    ),
    (
        "M7",
        "remove the out-of-vocabulary loudness in emit_sequence -- an unrecognised "
        "token must not be laundered into a clean, shorter, confident sequence",
        "        if verdict not in VALID_VERDICTS:",
        "        if False:",
    ),
    # ---- cells added after the cycle-2 Q/A found M1-M7 left NEW guards uncovered.
    # Enumerated from source (every `raise LedgerError` + every _dedup_key branch),
    # not from the two findings that were reported -- cover the CLASS.
    (
        "M8",
        "revert the fail-loud I/O guard to a silent success -- the writer would "
        "PRINT the row, exit 0 and write NOTHING, manufacturing the exact "
        "absent-row state the reader is built to refuse (cycle-2 QA-M6)",
        '        raise LedgerError(f"failed to append to {path}: {exc}", EXIT_IO) from exc',
        "        return row",
    ),
    (
        "M9",
        "drop step_id from the dedup key -- the same run_id then collides ACROSS "
        "steps and a legitimate second row is refused and LOST, under-counting a "
        "consecutive run in the fail-OPEN direction (cycle-2 QA-M4)",
        '        return (step, f"run:{run}")',
        '        return ("", f"run:{run}")',
    ),
    (
        "M10",
        "remove the empty-step_id guard -- an unkeyed, unattributable row appends",
        '        raise LedgerError("--step is required and must be non-empty.", EXIT_INVALID)',
        "        pass",
    ),
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_self_test(script: Path) -> tuple[int, str]:
    r = subprocess.run(
        [sys.executable, str(script), "--self-test"],
        capture_output=True, text=True,
    )
    return r.returncode, (r.stdout + r.stderr)


def main() -> int:
    if not TARGET.exists():
        print(f"FATAL: {TARGET} not found")
        return 1

    before = sha256(TARGET)
    print(f"target      : {TARGET}")
    print(f"sha256 before: {before}\n")

    # ---- CONTROL, observed FIRST. A red control makes every cell unscorable. ----
    rc, out = run_self_test(TARGET)
    control_green = rc == 0
    print(f"CONTROL      : rc={rc} -> {'GREEN' if control_green else 'RED'}")
    if not control_green:
        print("control is RED; every cell is UNSCORABLE. Fix the writer first.")
        print(out[-1500:])
        return 1
    print()

    src = TARGET.read_text()
    results: list[tuple[str, str]] = []

    with tempfile.TemporaryDirectory() as td:
        for cell, desc, find, repl in CELLS:
            n = src.count(find)
            if n != 1:
                results.append((cell, f"UNSCORABLE (anchor matched {n}x, expected 1)"))
                print(f"{cell}  UNSCORABLE  anchor matched {n}x -- {desc}")
                continue

            mutant_path = Path(td) / f"mutant_{cell}.py"
            mutant_path.write_text(src.replace(find, repl, 1))

            m_rc, m_out = run_self_test(mutant_path)
            killed = m_rc != 0
            # A KILL must be a real assertion failure, not an import/syntax error --
            # otherwise a typo'd mutation would score as a kill it did not earn.
            genuine = "SELF-TEST FAILED" in m_out
            if killed and genuine:
                status = f"KILLED (rc={m_rc})"
            elif killed:
                status = f"UNSCORABLE (rc={m_rc}, but no SELF-TEST FAILED -- likely a broken mutant)"
            else:
                status = "SURVIVED"
            results.append((cell, status))
            print(f"{cell}  {status:<18}  {desc}")
            if killed and genuine:
                for line in m_out.splitlines():
                    if line.strip().startswith("FAIL"):
                        print(f"        {line.strip()}")

    after = sha256(TARGET)
    print(f"\nsha256 after : {after}")
    print(f"UNCHANGED    : {before == after}  (mutations ran on temp copies; "
          f"the real file was never written)")

    survived = [c for c, s in results if s == "SURVIVED"]
    unscorable = [c for c, s in results if s.startswith("UNSCORABLE")]
    print(f"\n{len(results)} cells: "
          f"{len(results) - len(survived) - len(unscorable)} killed, "
          f"{len(survived)} survived, {len(unscorable)} unscorable")
    if survived:
        print(f"SURVIVORS (each is a guard that cannot fail): {', '.join(survived)}")
    if before != after:
        print("FATAL: target file changed -- investigate before trusting any cell.")
        return 1
    return 0 if not survived and not unscorable else 1


if __name__ == "__main__":
    raise SystemExit(main())
