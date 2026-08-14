#!/usr/bin/env python3
"""phase-86.74 mutation matrix -- prove the guards can actually FAIL.

    python3 scripts/qa/mutation_matrix_86_74.py

Criterion 9: "mutation-test with the control observed GREEN first and a
byte-identical restore: (M1) restore `if pct:` in the helper -- a 0% verdict must
go red; (M2) restore `or 10.0` at the sizing seam -- the same; (M3) delete the
persistence write -- criterion 4's check must go red; each cell scored
UNSCORABLE if its control was not green".

WHY EACH RULE IS HERE, all of them paid for previously:

* **Control green FIRST.** A cell whose control was already red proves nothing --
  the mutant "failing" is then indistinguishable from the suite being broken.
  Such a cell is scored UNSCORABLE, never KILLED.
* **Byte-identical restore, verified by sha256.** Not "looks restored".
* **The probe must DISCRIMINATE.** A cell is only KILLED when the control passed
  AND the mutant failed. If a mutation cannot even be APPLIED (its target text is
  absent) that is scored NOT_APPLIED -- a no-match `str.replace` looks exactly
  like success and would silently score a mutation that never happened.
* **Mutate the SUBJECT, not a copy.** Each mutation edits the real module the
  tests import.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PM = ROOT / "backend" / "services" / "portfolio_manager.py"
SUITE = "backend/tests/test_phase_66_2_risk_judge_shape.py"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(*args: str) -> tuple[int, str]:
    r = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest", *args, "-q"],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    return r.returncode, (r.stdout + r.stderr)[-400:]


def selected(target: str) -> int:
    """How many tests does `-k <target>` actually SELECT?

    THE VACUITY HOLE THIS CLOSES (found by the 86.74 Q/A, not by me):
    mutant cells are scored `killed = rc != 0`, and **pytest exits 5 when `-k`
    selects NOTHING**. So a renamed, typo'd or since-deleted target would select
    zero tests, exit 5, and be scored **KILLED** -- a cell that ran no assertions
    at all reported as proof the guard works. Every cell must therefore prove its
    selector is live BEFORE its verdict is believed.
    """
    r = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), "-m", "pytest", SUITE,
         "-k", target, "--collect-only", "-q"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    if r.returncode == 5:  # "no tests ran" -- the selector matches nothing
        return 0
    for line in reversed((r.stdout or "").strip().splitlines()):
        if "test" in line and "collected" in line:
            for tok in line.split():
                if tok.isdigit():
                    return int(tok)
    return sum(1 for ln in (r.stdout or "").splitlines() if "::" in ln)


#: (id, description, old, new, the test that must go RED)
MUTATIONS = [
    (
        "M1",
        "restore the falsy-zero `if pct:` in the helper",
        "    for source in (\n"
        "        risk_assessment.get(\"recommended_position_pct\"),\n"
        "        analysis.get(\"risk_judge_position_pct\"),\n"
        "    ):\n"
        "        verdict = _coerce_pct(source)\n"
        "        if verdict is not None:\n"
        "            return verdict\n"
        "    return PositionVerdict(ABSENT)",
        "    for source in (\n"
        "        risk_assessment.get(\"recommended_position_pct\"),\n"
        "        analysis.get(\"risk_judge_position_pct\"),\n"
        "    ):\n"
        "        if source:  # MUTANT: falsy -- 0.0 falls through\n"
        "            verdict = _coerce_pct(source)\n"
        "            if verdict is not None:\n"
        "                return verdict\n"
        "    return PositionVerdict(ABSENT)",
        "TestHelperDistinguishesZeroFromAbsent",
    ),
    (
        "M2",
        "restore `or 10.0` at the main sizing seam",
        "        position_pct = _sizing_pct(cand)\n"
        "        target_amount = nav * (position_pct / 100.0)",
        "        position_pct = cand[\"position_pct\"] or 10.0  # MUTANT\n"
        "        target_amount = nav * (position_pct / 100.0)",
        "TestRejectBindsInBothFlagStates",
    ),
    (
        "M3",
        "delete the persistence write (criterion 4 must go red)",
        "            risk_judge_decision=str(_rj.get(\"decision\") or \"\"),\n"
        "            risk_level=str(_rj.get(\"risk_level\") or \"\"),\n"
        "            recommended_position_pct=_rj_pct,",
        "            # MUTANT: persistence write deleted",
        "TestVerdictIsPersistedPerTicker",
    ),
    (
        "M4",
        "make the sizing default reachable from an UNPARSEABLE verdict",
        "    if state == UNPARSEABLE:\n        return 0.0",
        "    if state == UNPARSEABLE:\n        return DEFAULT_POSITION_PCT  # MUTANT",
        "TestSizingDefaultReachableOnlyFromAbsent",
    ),
    (
        "M5",
        "drop the RiskJudge row again when pct is 0 (criterion 6)",
        "        if decision or reasoning or pos_pct is not None:",
        "        if decision or reasoning:  # MUTANT",
        "TestRiskJudgeAppearsInFactorsJson",
    ),
    (
        "M6",
        "remove the ticker from the completion log line (criterion 5)",
        "        f\"Risk debate complete: ticker={ticker}, \"",
        "        f\"Risk debate complete: \"  # MUTANT",
        "test_risk_debate_completion_log_carries_the_ticker",
    ),
    (
        # phase-86.74 cycle-4. THE CELL THAT CAUGHT A PROXY ASSERTION.
        # The fix claims "emitting neither leg, so the SELL cannot orphan". This
        # deletes the early floor and re-applies it AFTER the SELL append, so the
        # SELL is emitted alone -- the exact net -1 harm. Against the original
        # BUY-subset assertion this SURVIVED (rc 0, 3 passed); against the
        # whole-order-list assertion it is KILLED. TWO edits, because a guard for
        # the BUY is not a guard for the harm and the difference is two lines apart.
        "M7",
        "let the SELL orphan: floor re-applied AFTER the SELL append",
        [
            ('        if buy_amount < 50:\n'
             '            logger.info(\n'
             '                "Swap skip %s -> %s: sized BUY $%.2f below the $50 floor "\n'
             '                "(position_pct=%s) -- emitting neither leg, so the SELL cannot "\n'
             '                "orphan.",\n'
             '                weakest["ticker"], cand["ticker"], buy_amount, position_pct,\n'
             '            )\n'
             '            continue\n',
             ''),
            ('        swap_orders.append(TradeOrder(\n'
             '            ticker=cand["ticker"],\n'
             '            action="BUY",',
             '        if buy_amount < 50:\n'
             '            continue  # MUTANT: SELL already appended -> ORPHANED\n'
             '        swap_orders.append(TradeOrder(\n'
             '            ticker=cand["ticker"],\n'
             '            action="BUY",'),
        ],
        None,  # `old` carries the (old,new) pairs for multi-edit cells
        "TestSwapPathSizingIsBehaviourallyGuarded",
    ),
]

#: Cells M3/M5/M6 mutate files OTHER than portfolio_manager.py, so the harness
#: below must snapshot and restore each subject independently -- restoring only
#: the primary file would leave a mutation live in the tree.
SUBJECTS = {
    "M1": PM, "M2": PM, "M4": PM, "M7": PM,
    "M3": ROOT / "backend" / "services" / "autonomous_loop.py",
    "M5": ROOT / "backend" / "services" / "signal_attribution.py",
    "M6": ROOT / "backend" / "agents" / "risk_debate.py",
}


def main() -> int:
    # Snapshot EVERY subject up front, so a crash mid-run cannot leave a mutation
    # live in a file this run never got around to restoring.
    baseline = {p: (p.read_text(), sha(p)) for p in set(SUBJECTS.values())}
    for p, (_, s) in sorted(baseline.items(), key=lambda kv: str(kv[0])):
        print(f"subject : {p.relative_to(ROOT)}  sha={s[:12]}")
    print()

    print("=== CONTROL (must be GREEN before any cell is scorable) ===")
    rc, tail = run(SUITE)
    control_green = rc == 0
    print(f"  control: {'GREEN' if control_green else 'RED'}\n")

    results = []
    for mid, desc, old, new, target in MUTATIONS:
        # A cell may carry ONE (old,new) or a LIST of them -- M7 needs two edits
        # because emitting the SELL and suppressing the BUY are two lines apart.
        edits = old if isinstance(old, list) else [(old, new)]
        subject = SUBJECTS[mid]
        original, original_sha = baseline[subject]

        if not control_green:
            results.append((mid, "UNSCORABLE", "control was not green"))
            continue

        if any(o not in original for o, _ in edits):
            # A no-match replace looks exactly like success. Never score it.
            results.append((mid, "NOT_APPLIED", f"target absent in {subject.name}"))
            print(f"  {mid}: NOT_APPLIED -- target absent, cell not scored")
            continue

        # The selector must select something, or `killed` is vacuous (pytest
        # exits 5 on an empty -k selection, and 5 != 0 scores as KILLED).
        n_sel = selected(target)
        if n_sel == 0:
            results.append((mid, "UNSCORABLE", f"-k {target!r} selected 0 tests"))
            print(f"  {mid}: UNSCORABLE -- selector matches nothing, cell not scored")
            continue

        try:
            mutated = original
            for o, n in edits:
                mutated = mutated.replace(o, n, 1)
            subject.write_text(mutated)
            assert sha(subject) != original_sha, "mutation did not change the file"
            rc_m, tail_m = run(SUITE, "-k", target)
            # rc 5 would mean "no tests ran"; the selector was proven live above,
            # so treat only a genuine test failure (1) as a kill.
            killed = rc_m == 1
            verdict = "KILLED" if killed else ("UNSCORABLE" if rc_m == 5 else "SURVIVED")
            results.append((
                mid, verdict, f"{subject.name} [{n_sel} sel]: {desc}",
            ))
            print(f"  {mid}: {verdict:<10} ({n_sel} tests selected)  {desc}")
        finally:
            subject.write_text(original)
            back = sha(subject)
            if back != original_sha:
                print(f"  !!! RESTORE FAILED for {mid}: {back[:16]} != {original_sha[:16]}")
                return 3

    all_restored = all(sha(p) == s for p, (_, s) in baseline.items())
    print(f"\nrestore verified byte-identical for all {len(baseline)} subjects: {all_restored}")
    if not all_restored:
        return 3
    print("\n=== MATRIX ===")
    for mid, verdict, note in results:
        print(f"  {mid:<4} {verdict:<12} {note}")

    bad = [m for m, v, _ in results if v != "KILLED"]
    if bad:
        print(f"\nRESULT: cells not KILLED: {bad}")
        return 1
    print("\nRESULT: every cell KILLED -- the guards can fail, so green means something.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
