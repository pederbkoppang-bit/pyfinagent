#!/usr/bin/env python3
"""phase-85.5.1 criterion 4 -- prove the guard bites and the fixture is load-bearing.

Criterion 4: "a mutation proves the guard bites -- reverting the fix makes the
test fail again, demonstrated in the evidence."

M1 is that literal requirement: put the old 2-key mock back and
`test_valid_nav_still_breaches` must go RED again.

M2-M4 go further, because a fixture that merely restores green is not the same as
a guard that still protects. They mutate the PRODUCTION breach logic and require
the book-safety suite to catch each one -- otherwise "14 passed" would mean the
test file cannot tell a working kill switch from a broken one.

Same discipline as the 85.4/85.6 matrices: assert the target exists and is unique
before replacing, restore from an in-memory backup (never a file-level
`git checkout`), verify restoration byte-for-byte, and invoke pytest with a LIST
argv so an unquoted shell variable cannot silently run nothing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KS = ROOT / "backend/services/kill_switch.py"
T = "backend/tests/test_book_safety_69.py"
TEST = ROOT / T
RED = f"{T}::test_valid_nav_still_breaches"

MUTATIONS: list[tuple[str, Path, str, str, list[str]]] = [
    (
        "M1 CRITERION 4: restore the 2-key mock -- the RED test must go RED again",
        TEST,
        # Anchored on the RED test's own `today =` line, which is unique to it.
        # The bare state-construction lines are NOT unique any more: the disarm
        # test added by this step uses the same two, and the harness's uniqueness
        # guard correctly refused to mutate an ambiguous target rather than
        # silently picking the wrong site.
        "    today = datetime.now(timezone.utc).date().isoformat()\n"
        "    st.update_sod_nav(100.0, date=today)",
        "    monkeypatch.setattr(st, \"snapshot\",\n"
        "                        lambda: {\"sod_nav\": 100.0, \"peak_nav\": 100.0})  # MUTANT\n"
        "    today = datetime.now(timezone.utc).date().isoformat()",
        [RED],
    ),
    (
        "M1b the anchor date is hardcoded instead of computed (rots at midnight)",
        TEST,
        "    st.update_sod_nav(100.0, date=today)",
        '    st.update_sod_nav(100.0, date="2020-01-01")  # MUTANT',
        [RED],
    ),
    (
        "M2 the daily-loss leg stops firing at all",
        KS,
        "    daily_loss_breached = daily_loss_pct >= daily_loss_limit_pct",
        "    daily_loss_breached = False  # MUTANT",
        [RED],
    ),
    (
        "M3 the trailing-DD leg stops firing at all",
        KS,
        "    trailing_dd_breached = trailing_dd_pct >= trailing_dd_limit_pct",
        "    trailing_dd_breached = False  # MUTANT",
        [RED],
    ),
    (
        "M4 the staleness guard stops disarming (phase-36.9 F1 regression)",
        KS,
        "def _sod_date_is_stale(sod_date: Any, sod_nav: Optional[float]) -> bool:",
        "def _sod_date_is_stale(sod_date: Any, sod_nav: Optional[float]) -> bool:\n    return False  # MUTANT",
        [T],  # the whole file: some OTHER test must catch a guard that never disarms
    ),
]

ALL_FILES = {KS, TEST}


def run_tests(node_ids: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "--timeout=120", "-p", "no:randomly",
           *node_ids]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def _live_journal_lines() -> int:
    j = ROOT / "handoff" / "kill_switch_audit.jsonl"
    return len(j.read_text(encoding="utf-8").splitlines()) if j.exists() else -1


def main() -> int:
    backups = {f: f.read_text(encoding="utf-8") for f in ALL_FILES}
    journal_before = _live_journal_lines()

    rc, out = run_tests([T])
    if rc != 0:
        print("PRECONDITION FAILED: test_book_safety_69.py is not green at HEAD.")
        print(out)
        return 2
    print(f"precondition OK -- baseline green: {out.strip().splitlines()[-1]}\n")

    failures: list[str] = []
    for name, path, find, repl, node_ids in MUTATIONS:
        original = backups[path]
        n = original.count(find)
        if n != 1:
            failures.append(f"{name}: target occurs {n}x in {path.name} "
                            f"(need exactly 1) -- the harness is stale, not the code")
            print(f"[SKIP  ] {name}\n         target count = {n}")
            continue
        try:
            path.write_text(original.replace(find, repl, 1), encoding="utf-8")
            rc, out = run_tests(node_ids)
            if rc == 0:
                failures.append(f"{name}: tests still PASSED with the change reverted")
                print(f"[LIVE  ] {name}\n         *** tests passed anyway ***")
            else:
                tail = [l for l in out.splitlines() if " passed" in l or " failed" in l]
                print(f"[KILLED] {name}\n         {tail[-1].strip() if tail else 'red'}")
        finally:
            path.write_text(original, encoding="utf-8")

    for f, content in backups.items():
        if f.read_text(encoding="utf-8") != content:
            failures.append(f"RESTORE FAILED: {f} differs from its backup")

    rc, out = run_tests([T])
    if rc != 0:
        failures.append("POST-CONDITION FAILED: not green after restoration")
        print(out)

    journal_after = _live_journal_lines()
    if journal_after != journal_before:
        failures.append(
            f"LIVE JOURNAL MUTATED by this run: {journal_before} -> {journal_after} "
            f"lines. The suite must never write to handoff/kill_switch_audit.jsonl."
        )
    else:
        print(f"\nlive kill-switch journal untouched ({journal_before} lines "
              f"before and after)")

    print()
    if failures:
        print(f"MUTATION MATRIX FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"MUTATION MATRIX PASSED -- {len(MUTATIONS)}/{len(MUTATIONS)} killed, "
          f"tree restored byte-for-byte, suite green, live journal untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
