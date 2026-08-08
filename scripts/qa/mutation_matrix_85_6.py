#!/usr/bin/env python3
"""phase-85.6 mutation matrix -- prove the deadlock guards can actually FAIL.

Same discipline as scripts/qa/mutation_matrix_85_4.py: every mutation asserts its
target string EXISTS before replacing (a no-match str.replace looks exactly like
success), files are restored from an explicit in-memory backup rather than via a
file-level `git checkout`, restoration is verified byte-for-byte, and pytest is
invoked with a LIST argv so an unquoted shell variable cannot collapse the run to
nothing while printing "0 failures".

M1 is the one that matters: it re-creates the production deadlock by deleting the
Step-0 roll, and criterion 1's cycle test must go red.

Usage:  source .venv/bin/activate && python scripts/qa/mutation_matrix_85_6.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LOOP = ROOT / "backend/services/autonomous_loop.py"
TRADER = ROOT / "backend/services/paper_trader.py"
API = ROOT / "backend/api/paper_trading.py"
T = "backend/tests/test_phase_85_6_anchor_deadlock.py"

MUTATIONS: list[tuple[str, Path, str, str, list[str]]] = [
    (
        "M1 THE DEADLOCK RESTORED: Step 0 roll deleted from the cycle",
        LOOP,
        '            summary["steps"].append("sod_anchor_roll")\n'
        "            _anchor = await asyncio.to_thread(trader.roll_daily_anchor)\n"
        '            summary["sod_anchor_roll"] = _anchor\n',
        "",
        [
            f"{T}::test_c1_a_cycle_that_dies_mid_analysis_still_leaves_a_fresh_anchor",
            f"{T}::test_c1_the_roll_runs_before_screening_not_after_analysis",
            f"{T}::test_c1_the_anchor_survives_a_crash_in_analysis_too",
        ],
    ),
    (
        "M2 the roll drifts back BEHIND screening (ordering guard)",
        LOOP,
        '            summary["steps"].append("sod_anchor_roll")\n'
        "            _anchor = await asyncio.to_thread(trader.roll_daily_anchor)\n"
        '            summary["sod_anchor_roll"] = _anchor\n\n'
        "            # ── Step 1: Screen universe (free) ───────────────────────\n"
        '            logger.info("Paper trading: Step 1 -- Screening universe")\n'
        '            summary["steps"].append("screening")',
        "            # ── Step 1: Screen universe (free) ───────────────────────\n"
        '            logger.info("Paper trading: Step 1 -- Screening universe")\n'
        '            summary["steps"].append("screening")\n'
        '            summary["steps"].append("sod_anchor_roll")\n'
        "            _anchor = await asyncio.to_thread(trader.roll_daily_anchor)\n"
        '            summary["sod_anchor_roll"] = _anchor',
        [
            f"{T}::test_c1_the_roll_runs_before_screening_not_after_analysis",
            f"{T}::test_c2_the_mechanism_the_message_names_actually_exists",
        ],
    ),
    (
        "M3 the roller ignores the date guard and re-anchors mid-day",
        TRADER,
        "            if not sod_anchor_needs_reroll(snap, today):",
        "            if False:",
        [f"{T}::test_roll_is_a_same_day_noop"],
    ),
    (
        "M4 the roller latches a non-positive anchor (phase-36.9 F3 regression)",
        TRADER,
        "            rolled = post.get(\"sod_date\") == today and (post.get(\"sod_nav\") or 0) > 0",
        '            rolled = True',
        [
            f"{T}::test_roll_refuses_a_non_positive_nav_and_says_so",
            f"{T}::test_c1_the_cycle_records_the_roll_outcome_for_the_operator",
        ],
    ),
    (
        "M5 the roller anchors on the CURRENT mark instead of the stored NAV "
        "(loosening on a falling book)",
        TRADER,
        '                portfolio.get("total_nav") or portfolio.get("starting_capital") or 0.0',
        '                portfolio.get("starting_capital") or 0.0',
        [f"{T}::test_c5_on_a_falling_book_the_early_anchor_is_never_more_forgiving"],
    ),
    (
        "M6 the roller un-pauses / mutates the peak (scope creep into risk state)",
        TRADER,
        "            self._sod_anchor_provisional = True\n            post = state.snapshot()",
        "            self._sod_anchor_provisional = True\n"
        "            state.update_peak(0.01)\n"
        "            post = state.snapshot()",
        [
            f"{T}::test_c5_the_roller_changes_no_threshold_and_disarms_nothing",
            f"{T}::test_c1_the_roll_runs_before_screening_not_after_analysis",
        ],
    ),
    (
        "M7 the false 409 promise comes back",
        API,
        '            "UNBLOCK CONDITION: a paper-trading cycle must START and run its "',
        '            "NO operator action is required: this refusal clears itself. "\n'
        '            "UNBLOCK CONDITION: a paper-trading cycle must START and run its "',
        [f"{T}::test_c2_the_409_no_longer_makes_the_two_false_promises"],
    ),
    (
        "M8 the 409 stops naming the real roller",
        API,
        '            "start-of-day roll (Step 0, backend/services/autonomous_loop.py, "\n'
        '            "PaperTrader.roll_daily_anchor -> kill_switch.update_sod_nav). That "',
        '            "start-of-day roll. That "',
        [f"{T}::test_c2_the_409_names_the_actual_unblock_mechanism"],
    ),
    (
        "M9 STUB MUTATION: the fake state stops mirroring the F3 refusal",
        ROOT / T,
        "        if nav is None or float(nav) <= 0:\n            return",
        "        if False:\n            return",
        [f"{T}::test_roll_refuses_a_non_positive_nav_and_says_so"],
    ),
    (
        "M10 the provisional anchor is never upgraded (Q/A pass-1 hazard restored)",
        TRADER,
        "        elif self._sod_anchor_provisional and snap.get(\"sod_date\") == today:",
        "        elif False:",
        [f"{T}::test_c5_a_multi_session_stale_anchor_does_not_become_a_same_day_loss"],
    ),
    (
        "M11 Step 0 stops flagging its anchor provisional",
        TRADER,
        "            self._sod_anchor_provisional = True\n"
        "            post = state.snapshot()",
        "            post = state.snapshot()",
        [
            f"{T}::test_c5_the_provisional_flag_is_set_by_step0_and_cleared_by_the_upgrade",
            f"{T}::test_c5_a_multi_session_stale_anchor_does_not_become_a_same_day_loss",
        ],
    ),
    (
        "M12 a same-day no-op roll wrongly flags provisional (mid-day re-anchor)",
        TRADER,
        '                    "reason": "anchor_already_current",',
        '                    "reason": (setattr(self, "_sod_anchor_provisional", True)'
        ' or "anchor_already_current"),',
        [f"{T}::test_c5_a_same_day_noop_roll_does_not_flag_provisional"],
    ),
]

ALL_FILES = {LOOP, TRADER, API, ROOT / T}


def run_tests(node_ids: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "--timeout=60",
           "--timeout-method=thread", "-p", "no:randomly", *node_ids]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2500:]


def main() -> int:
    backups = {f: f.read_text(encoding="utf-8") for f in ALL_FILES}

    rc, out = run_tests([T])
    if rc != 0:
        print("PRECONDITION FAILED: the phase-85.6 suite is not green at HEAD.")
        print(out)
        return 2
    print(f"precondition OK -- baseline green: {out.strip().splitlines()[-1]}\n")

    failures: list[str] = []
    for name, path, find, repl, node_ids in MUTATIONS:
        original = backups[path]
        if find not in original:
            failures.append(f"{name}: MUTATION TARGET NOT FOUND in {path.name} -- "
                            f"the harness is stale, not the code")
            print(f"[SKIP  ] {name}\n         target absent from {path.name}")
            continue
        if original.count(find) != 1:
            failures.append(f"{name}: target occurs {original.count(find)}x -- ambiguous")
            print(f"[SKIP  ] {name}\n         target is not unique")
            continue
        mutated = original.replace(find, repl, 1)
        if mutated == original:
            failures.append(f"{name}: replace was a no-op")
            continue
        try:
            path.write_text(mutated, encoding="utf-8")
            rc, out = run_tests(node_ids)
            if rc == 0:
                failures.append(f"{name}: tests still PASSED with the fix reverted")
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
        failures.append("POST-CONDITION FAILED: suite not green after restoration")
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
