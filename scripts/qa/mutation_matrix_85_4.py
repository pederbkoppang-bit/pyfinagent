#!/usr/bin/env python3
"""phase-85.4 mutation matrix -- prove every new guard can actually FAIL.

A test that passes both with and without the fix proves nothing. This harness
reverts each phase-85.4 change one at a time and asserts the tests that claim
to cover it go RED.

Discipline enforced here (learned the hard way, see auto-memory):
  * every mutation asserts its target string EXISTS before replacing, and that
    the file CHANGED after -- a no-match str.replace looks exactly like success
  * the STUB is mutated too (M9), so a fixture that never drives the real gate
    is caught
  * files are restored from an explicit in-memory backup, never via a
    file-level `git checkout` (which silently reverts unrelated live edits)
  * restoration is verified byte-for-byte at the end

Usage:  source .venv/bin/activate && python scripts/qa/mutation_matrix_85_4.py
Exit 0 iff every mutation killed the tests it was supposed to kill AND the
tree is byte-identical afterwards.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LOOP = ROOT / "backend/services/autonomous_loop.py"
HEALTH = ROOT / "backend/services/cycle_health.py"
SCHED = ROOT / "backend/slack_bot/scheduler.py"
T_LOUD = "backend/tests/test_phase_85_4_cycle_loudness.py"
T_AGE = "backend/tests/test_phase_85_4_completed_age_alarm.py"
T_DISP = "backend/tests/test_phase_85_4_dispatch_barrier.py"

# (name, file, find, replace, tests-that-must-go-red)
MUTATIONS: list[tuple[str, Path, str, str, list[str]]] = [
    (
        "M1 kill-switch halt leaks the ':362' placeholder status again",
        LOOP,
        '                summary["status"] = "halted_kill_switch"\n',
        "",
        [
            f"{T_LOUD}::test_c3_killswitch_halt_records_a_real_terminal_status_not_running",
            f"{T_LOUD}::test_c3_halted_status_is_not_counted_as_a_completion",
        ],
    ),
    (
        "M2 alert title stops naming the phase",
        LOOP,
        "title=f\"Autonomous trading cycle {_final_status} in phase '{_died_in}'\",",
        'title=f"Autonomous trading cycle {_final_status}",',
        [
            f"{T_LOUD}::test_c3_timeout_midanalysis_writes_terminal_row_and_names_the_phase",
            f"{T_LOUD}::test_c3_crash_midanalysis_writes_terminal_row_and_names_the_phase",
        ],
    ),
    (
        "M3 alert details drop died_in_phase",
        LOOP,
        '                        "died_in_phase": _died_in,\n',
        "",
        [
            f"{T_LOUD}::test_c3_timeout_midanalysis_writes_terminal_row_and_names_the_phase",
            f"{T_LOUD}::test_c3_crash_midanalysis_writes_terminal_row_and_names_the_phase",
            f"{T_LOUD}::test_c3_killswitch_halt_records_a_real_terminal_status_not_running",
        ],
    ),
    (
        "M4 a timeout is treated as a completion",
        HEALTH,
        '_COMPLETED_STATUSES: frozenset[str] = frozenset({"completed"})',
        '_COMPLETED_STATUSES: frozenset[str] = frozenset({"completed", "timeout"})',
        [
            f"{T_AGE}::test_daily_timeouts_keep_the_old_clock_green_and_the_new_clock_red",
            f"{T_AGE}::test_a_ledger_with_no_completion_at_all_pages",
        ],
    ),
    (
        "M5 the completed-age clock never goes stale",
        HEALTH,
        "            success_stale = success_age_sec > completed_threshold_sec",
        "            success_stale = False",
        [
            f"{T_AGE}::test_daily_timeouts_keep_the_old_clock_green_and_the_new_clock_red",
            f"{T_AGE}::test_one_extra_failed_weekday_past_the_weekend_does_page",
            f"{T_AGE}::test_halted_kill_switch_rows_are_not_completions",
        ],
    ),
    (
        "M6 a ledger with zero completions falls back to the benign sentinel",
        HEALTH,
        "            success_stale = last_row is not None",
        "            success_stale = False",
        [
            f"{T_AGE}::test_a_ledger_with_no_completion_at_all_pages",
            f"{T_LOUD}::test_c3_halted_status_is_not_counted_as_a_completion",
        ],
    ),
    (
        "M7 merged dispatch silently falls back to the two-gather barrier",
        LOOP,
        "    n_new = len(new_tickers)\n    combined = await asyncio.gather(",
        "    n_new = len(new_tickers)\n    if True:\n        return await dispatch_analyses("
        "runner, new_tickers, reeval_tickers, merged=False)\n    combined = await asyncio.gather(",
        [
            f"{T_DISP}::test_a_merged_path_starts_the_reeval_as_soon_as_a_slot_frees",
            f"{T_DISP}::test_a_merged_path_strictly_reduces_makespan_at_the_same_concurrency",
        ],
    ),
    (
        "M8 the watchdog stops paging on the completed-age verdict",
        SCHED,
        "        if success_stale_now and prior_success_stale is not True:\n"
        "            fire_cycle_completed_stale_alarm(verdict)",
        "        if False and prior_success_stale is not True:\n"
        "            fire_cycle_completed_stale_alarm(verdict)",
        [
            f"{T_AGE}::test_watchdog_fires_the_completed_stale_p1_exactly_once_per_transition",
            f"{T_AGE}::test_watchdog_p1_payload_names_the_last_completion_and_the_last_status",
        ],
    ),
    (
        "M9 STUB MUTATION: the injected kill-switch state stops being paused",
        ROOT / T_LOUD,
        "    monkeypatch.setattr(ks, \"get_state\", lambda: _StubKillSwitchState(True))",
        '    monkeypatch.setattr(ks, "get_state", lambda: _StubKillSwitchState(False))',
        [
            f"{T_LOUD}::test_c3_killswitch_halt_records_a_real_terminal_status_not_running",
            f"{T_LOUD}::test_c3_halted_status_is_not_counted_as_a_completion",
        ],
    ),
]

ALL_FILES = {LOOP, HEALTH, SCHED, ROOT / T_LOUD}


def run_tests(node_ids: list[str]) -> tuple[int, str]:
    """Run pytest on explicit node ids. Uses a LIST argv -- never a string --
    because an unquoted shell variable would collapse to one argument and the
    run would silently execute nothing while printing '0 failures'."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--timeout=120", "-p", "no:randomly", *node_ids]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2500:]


def main() -> int:
    backups = {f: f.read_text(encoding="utf-8") for f in ALL_FILES}

    # Precondition: the suite must be GREEN before any mutation, or a "red"
    # result below would mean nothing.
    rc, out = run_tests([T_LOUD, T_AGE, T_DISP])
    if rc != 0:
        print("PRECONDITION FAILED: the phase-85.4 suite is not green at HEAD.")
        print(out)
        return 2
    print(f"precondition OK -- baseline suite green\n{out.strip().splitlines()[-1]}\n")

    failures: list[str] = []
    for name, path, find, repl, node_ids in MUTATIONS:
        original = backups[path]
        if find not in original:
            failures.append(f"{name}: MUTATION TARGET NOT FOUND in {path.name} -- "
                            f"the harness is stale, not the code")
            print(f"[SKIP] {name}\n       target string absent from {path.name}")
            continue
        mutated = original.replace(find, repl, 1)
        if mutated == original:
            failures.append(f"{name}: replace was a no-op")
            continue
        try:
            path.write_text(mutated, encoding="utf-8")
            rc, out = run_tests(node_ids)
            if rc == 0:
                failures.append(f"{name}: tests still PASSED with the fix reverted "
                                f"-- these guards do not actually guard")
                print(f"[LIVE ] {name}\n        *** tests passed anyway ***")
            else:
                tail = [l for l in out.splitlines() if " passed" in l or " failed" in l]
                print(f"[KILLED] {name}\n         {tail[-1].strip() if tail else 'red'}")
        finally:
            path.write_text(original, encoding="utf-8")

    # Verify restoration byte-for-byte.
    for f, content in backups.items():
        if f.read_text(encoding="utf-8") != content:
            failures.append(f"RESTORE FAILED: {f} differs from its backup")

    rc, out = run_tests([T_LOUD, T_AGE, T_DISP])
    if rc != 0:
        failures.append("POST-CONDITION FAILED: suite is not green after restoration")
        print(out)

    print()
    if failures:
        print(f"MUTATION MATRIX FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"MUTATION MATRIX PASSED -- {len(MUTATIONS)}/{len(MUTATIONS)} mutations killed, "
          f"tree restored byte-for-byte, suite green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
