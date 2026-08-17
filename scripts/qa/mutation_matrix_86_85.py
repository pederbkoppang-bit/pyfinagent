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
        # (cycle-7 retarget: the old replacement wrote a full TIMESTAMP, which
        # the new ISO-date guard now refuses loudly -- the mutant died at the
        # WRONG guard and scored UNSCORABLE. This form keeps the value
        # date-shaped so it reaches the ordering fixtures, which kill it:
        # a backfilled row stamped with TODAY's date sorts last.)
        '        "date": event_date or stamp.date().isoformat(),',
        '        "date": stamp.date().isoformat(),',
    ),
    # ---- cells added after the cycle-1 Q/A found M1-M5 blind to ordering ----
    (
        "M6",
        "REVERSE emit_sequence -- this is the cycle-1 Q/A's QA-M1, which SURVIVED "
        "against the old palindromic fixture. Reversing [PASS,C,C] to [C,C,PASS] "
        "takes enforceEscalation from n=2/auto_fail=true to n=0/auto_fail=false, "
        "silently DISARMING the escalation. (Anchor retargeted phase-86.85 "
        "cycle 6: the tail is now the event-date sort.)",
        "\n    return [v for _, _, v in sorted(keyed, key=lambda t: (t[0], t[1]))]",
        "\n    return [v for _, _, v in sorted(keyed, key=lambda t: (t[0], t[1]), reverse=True)]",
    ),
    # ── cycle-6 cells: the QA-MUT-B class (cycle-5 Q/A). emit_sequence returned
    # FILE order while claiming oldest->newest, and the shipped --date backfill
    # made the divergence reachable: an older PASS appended after two
    # CONDITIONALs cleared a live escalation. The fix sorts by EVENT date; these
    # cells pin the fix and its new guard.
    (
        "M15",
        "collapse EVENT order back to FILE order (the surviving QA-MUT-B mutant "
        "itself, as a permanent cell) -- the backfill fixture must catch it",
        "        keyed.append((event_date, pos, verdict))",
        '        keyed.append(("", pos, verdict))',
    ),
    (
        "M16",
        "remove the undated-row loudness in emit_sequence -- a row that cannot "
        "be ordered must refuse, not float to an arbitrary position",
        "        if not event_date:",
        "        if False:",
    ),
    # ── cycle-7 cells (QA-M-POS-const + QA-C6-1). The cycle-6 fix shipped a
    # sort whose tuple fell through to the VERDICT STRING on same-date rows --
    # the common case -- and an unvalidated ISO-date precondition with 11/52
    # real rows already violating it. These pin both.
    (
        "M17",
        "verdict string participates in same-date ordering -- the 99.9/4.7 "
        "fixtures (file order C,P,F vs alphabetical C,F,P) must catch it. "
        "(cycle-7 note: the originally-named mutant -- plain sorted(keyed) -- "
        "became EQUIVALENT once the key excluded the verdict, because pos still "
        "discriminates; likewise pos-to-constant alone degrades to stable file "
        "order. The defect now needs the key itself to consult the verdict, "
        "which is this cell.)",
        "    return [v for _, _, v in sorted(keyed, key=lambda t: (t[0], t[1]))]",
        "    return [v for _, _, v in sorted(keyed, key=lambda t: (t[0], t[2]))]",
    ),
    (
        "M18",
        "same-date position reversed -- file order flips within a date",
        "    return [v for _, _, v in sorted(keyed, key=lambda t: (t[0], t[1]))]",
        "    return [v for _, _, v in sorted(keyed, key=lambda t: (t[0], -t[1]))]",
    ),
    (
        "M19",
        "emit-side non-ISO date refusal removed -- a malformed stored date is "
        "silently ordered again",
        "        if event_date and not valid_event_date(event_date):",
        "        if False:",
    ),
    (
        "M21",
        "calendar half of the date validation removed -- 2026-18-10 passes the "
        "shape regex, sorts after every real August date, and clears an "
        "escalation (cycle-8 QA-C7-1)",
        "    try:\n        _date.fromisoformat(s)\n    except ValueError:\n        return False\n    return True",
        "    return True",
        # NOTE: replacement keeps valid_event_date returning True after the
        # regex, i.e. shape-only validation -- exactly the cycle-7 state.
    ),
    (
        "M22",
        "shape half of the date validation removed -- fromisoformat alone "
        "ACCEPTS compact 20260810, which sorts lexicographically LAST: the "
        "escalation-clearing direction (cycle-9 QA-C9-1)",
        "    if not ISO_DATE_RE.match(s):\n        return False",
        "    if False:\n        return False",
    ),
    (
        "M20",
        "write-side non-ISO date refusal removed -- --date 2026-8-10 writes",
        "    if event_date is not None and not valid_event_date(str(event_date).strip()):",
        "    if False:",
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
    # ---- cells added after the cycle-3 Q/A found the CYCLE FALLBACK branch
    # uncovered. Two prior cycles claimed "every distinguishing branch of
    # _dedup_key" was covered; _dedup_key has THREE outcomes (run:, cycle:, raise)
    # and only two were. The missed one is LIVE -- 5 real rows use it, all on 86.74.
    (
        "M11",
        "make the cycle fallback key CONSTANT -- run_id-less rows stop being "
        "distinguishable, appends are refused under the benign EXIT_DUPLICATE, and "
        "verdicts vanish from the sequence enforceEscalation consumes (cycle-3 QA-M2)",
        '        return (step, f"cycle:{cycle}")',
        '        return (step, "cycle:X")',
    ),
    (
        "M12",
        "DELETE the cycle fallback entirely -- every run_id-less row becomes "
        "unkeyable and is refused, so a rail drop recorded without a run_id is "
        "silently dropped from history (cycle-3 QA-M1)",
        '    cycle = str(row.get("cycle") or "").strip()\n    if cycle:\n        return (step, f"cycle:{cycle}")',
        '    cycle = str(row.get("cycle") or "").strip()\n    if False:\n        return (step, f"cycle:{cycle}")',
    ),
    # ── cycle-4 (C8). NOT hand-spotted. `verify_matrix_coverage_86_85.py`
    # DERIVED these two gaps from the writer's own AST: `main`'s CLI argument
    # validation had no cell aiming at it, while the matrix reported 12/12
    # KILLED. That is the whole 86.85 failure class in miniature -- a matrix
    # complete over its own list, not over the subject's guards -- and it is why
    # cells are no longer added by reading the file and thinking hard.
    (
        "M13",
        "remove the --emit-sequence arg guard -- `--emit-sequence` with no --step "
        "must be refused; without it the reader is invoked with an empty step id "
        "and returns a sequence for the wrong (or no) step, which the escalation "
        "rule then consumes as if it were history",
        '            if not args.step:\n                raise LedgerError("--emit-sequence requires --step.", EXIT_INVALID)',
        '            if False:\n                raise LedgerError("--emit-sequence requires --step.", EXIT_INVALID)',
    ),
    (
        "M14",
        "remove the append arg guard -- an append with a missing --step or "
        "--verdict must be refused at the CLI boundary rather than reaching "
        "build_row, so the refusal is attributable to the caller's invocation",
        "        if not args.step or not args.verdict:",
        "        if False:",
    ),
    # ── cycle-12 (QA-C11-A). The per-step filter in emit_sequence was exact-
    # match in SOURCE, but both fixture pairs ("99.4"/"99.2" in the self-test,
    # "4.1"/"4.2" in pytest) were prefix-UNRELATED, so broadening the filter
    # swept nothing in any fixture: two independently-constructed mutants
    # survived the whole suite while emit_sequence("86.9") gained step 86.90's
    # PASS -- the escalation-CLEARING direction (869 strict-prefix pairs among
    # the 1413 masterplan ids). The fixtures are now prefix-related in both
    # directions, and the evaluator's own two mutants are kept permanent here.
    (
        "M23",
        "per-step filter broadened to startswith -- a foreign extension step's "
        "PASS is swept into the query step's sequence (QA-C11-A MUT-A)",
        "        if (row.get(\"step_id\") or \"\").strip() != step_id:",
        "        if not (row.get(\"step_id\") or \"\").strip().startswith(step_id):",
    ),
    (
        "M24",
        "per-step filter broadened to containment -- same sweep, independent "
        "construction (QA-C11-A MUT-B)",
        "        if (row.get(\"step_id\") or \"\").strip() != step_id:",
        "        if step_id not in (row.get(\"step_id\") or \"\").strip():",
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

    # ── COMPLETENESS IS DERIVED, NOT CLAIMED (cycle-4, C8) ────────────────
    # "14 cells killed" answers "can these guards fail?" It does NOT answer
    # "is there a guard with no cell?" -- and THAT is the question this step
    # failed three times in a row, each time because the guard list was
    # written by hand and its omissions were invisible from the inside.
    #
    # So the matrix is RED when a guard has no cell, even if every cell it
    # does have was killed. On the run that introduced this call the matrix
    # reported 12/12 KILLED while `main`'s CLI argument validation had NO
    # cell aiming at it: a perfect score over an incomplete list.
    print()
    print("=== DERIVED COVERAGE (does any guard have NO cell?) ===")
    import verify_matrix_coverage_86_85 as cov  # local import: same directory

    cov_rc = cov.main()

    # phase-86.89 cycle 2 WIRING -- REVERTED at cycle 3, and the reason is a
    # HAZARD I introduced rather than a preference.
    #
    # Wiring `verify_cell_vacuity_86_89.main()` in here made this file WRITE TO
    # ITSELF: the vacuity check drops each cell by rewriting the matrix on disk.
    # MEASURED during one run -- 14 distinct truncated states, 11,734..12,228
    # bytes against a pristine 12,407 -- in a tree whose auto-commit hook does
    # `git add -A`. An interrupt mid-run leaves a truncated matrix that the next
    # hook invocation would stage and commit.
    #
    # This file's own docstring promises ZERO REPO WRITES precisely so no restore
    # can be gotten wrong, and the wiring broke that promise. "Standing" has to be
    # earned by a mechanism that does not mutate the repo -- an in-memory driver,
    # or a copy under a temp root -- not by calling a disk-mutating checker from
    # the file it mutates.
    #
    # The vacuity check remains runnable on its own; the "standing" claim is
    # WITHDRAWN from the artifacts rather than propped up by an unsafe wire.
    if cov_rc != 0:
        print("\nFATAL: the mutation matrix is INCOMPLETE over the writer's "
              "guards (see above). Every cell may still have been killed -- "
              "that is precisely the failure this check exists to catch.")
        return 1

    return 0 if not survived and not unscorable else 1


if __name__ == "__main__":
    raise SystemExit(main())
