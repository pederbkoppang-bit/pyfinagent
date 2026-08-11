#!/usr/bin/env python3
"""phase-86.32 mutation matrix -- is the attempt budget's suite load-bearing?

Eight cells, each breaking exactly one property the step claims. Six target the
budget's logic; M7 and M8 reproduce the two halves of the fixture defect that
FAILED this step at cycle 1 -- an inverted outcome pair, and a research-gate run
posing as a Q/A attempt -- so neither can recur silently. Every cell names
the test that must go red; a cell whose reds are all in OTHER tests is reported
MIS-ATTRIB, not KILLED, because a red somewhere is not evidence about the property
under test.

The control is run FIRST and the run aborts if it is not green -- a mutation score
computed against a red control measures nothing. An anchor that fails to match, or
matches more than once, is reported rather than silently mutating nothing: a
no-match replacement leaves the suite green and is indistinguishable from a
surviving mutant.

Cells M3 and M4 are the safety-critical ones. They break the two properties whose
failure direction is DANGEROUS -- exhaustion manufacturing a pass, and the
product/evidence split becoming a door a FAIL can walk through.

Run:  python scripts/qa/mutation_matrix_86_32.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "scripts" / "harness" / "attempt_budget.py"
SUITE = REPO / "backend" / "tests" / "test_phase_86_32_attempt_budget.py"

CELLS = [
    {
        "id": "M1-reset-on-conditional",
        "what": "reinstate the production defect: CONDITIONAL resets the budget",
        "old": """    def record(self, outcome: Outcome, tokens: int = 0, run_id: str = "",
               note: str = "") -> Attempt:
        a = Attempt(n=self.attempts_used + 1, outcome=outcome, tokens=tokens,
                    run_id=run_id, note=note)
        self.attempts.append(a)
        return a""",
        "new": """    def record(self, outcome: Outcome, tokens: int = 0, run_id: str = "",
               note: str = "") -> Attempt:
        if outcome is Outcome.CONDITIONAL:
            self.attempts.clear()
        a = Attempt(n=self.attempts_used + 1, outcome=outcome, tokens=tokens,
                    run_id=run_id, note=note)
        self.attempts.append(a)
        return a""",
        "expect_named": [
            "test_no_outcome_ever_decrements_the_attempt_count",
            "test_conditional_does_not_reset_the_budget",
        ],
    },
    {
        "id": "M2-drops-are-free",
        "what": "make a dropped spawn cost nothing against the budget",
        "old": """    @property
    def attempts_used(self) -> int:
        \"\"\"EVERY attempt, including those that returned no verdict.\"\"\"
        return len(self.attempts)""",
        "new": """    @property
    def attempts_used(self) -> int:
        \"\"\"EVERY attempt, including those that returned no verdict.\"\"\"
        return sum(1 for a in self.attempts if a.outcome is not Outcome.NO_VERDICT)""",
        "expect_named": [
            "test_dropped_spawns_count_against_the_budget",
            "test_the_gap_between_attempts_and_verdicts_is_visible",
        ],
    },
    {
        "id": "M3-exhaustion-auto-passes",
        "what": "SAFETY: let an exhausted budget close green",
        "old": """        if self.exhausted:
            return Disposition.ESCALATE
        return Disposition.CONTINUE""",
        "new": """        if self.exhausted:
            return Disposition.CLOSED_PASS
        return Disposition.CONTINUE""",
        "expect_named": ["test_exhaustion_cannot_auto_pass"],
    },
    {
        "id": "M4-residuals-door-opens-for-a-fail",
        "what": "SAFETY: let the product/evidence split close a step with no PASS",
        "old": """        d = self.disposition()
        if d is not Disposition.CLOSED_PASS:
            return d.value""",
        "new": """        d = self.disposition()
        if False:
            return d.value""",
        "expect_named": ["test_a_fail_stays_a_fail_under_every_flag_combination"],
    },
    {
        "id": "M5-exhaustion-checked-before-pass",
        "what": "throw away a PASS earned on the final permitted attempt",
        "old": """        if any(a.outcome is Outcome.PASS for a in self.attempts):
            return Disposition.CLOSED_PASS
        if self.exhausted:
            return Disposition.ESCALATE""",
        "new": """        if self.exhausted:
            return Disposition.ESCALATE
        if any(a.outcome is Outcome.PASS for a in self.attempts):
            return Disposition.CLOSED_PASS""",
        "expect_named": ["test_pass_on_the_final_permitted_attempt_still_closes_green"],
    },
    {
        "id": "M6-summary-unconditional",
        "what": "emit the escalation summary even when not exhausted",
        # The trailing newline before the closing quotes is deliberate: the anchor
        # ends in `return ""`, which would otherwise abut this file's own `"""`.
        "old": """        if self.disposition() is not Disposition.ESCALATE:
            return ""
""",
        "new": """        if False:
            return ""
""",
        "expect_named": ["test_no_summary_when_not_exhausted"],
    },
    {
        "id": "M7-fixture-outcomes-inverted",
        "what": "reproduce the SHIPPED cycle-1 defect: invert the drop/FAIL pair",
        "old": """    (3, "wf_01c83c86-09d", Outcome.NO_VERDICT,  "evaluator_critique_86.28.md::ledger"),
    (3, "wf_e262facc-cdc", Outcome.FAIL,        "evaluator_critique_86.28.md::ledger"),""",
        "new": """    (3, "wf_01c83c86-09d", Outcome.FAIL,        "evaluator_critique_86.28.md::ledger"),
    (3, "wf_e262facc-cdc", Outcome.NO_VERDICT,  "evaluator_critique_86.28.md::ledger"),""",
        "expect_named": ["test_fixture_matches_the_recorded_ledger"],
    },
    {
        "id": "M8-non-attempt-leaks-in",
        "what": "reproduce the other half: a research-gate run posing as an attempt",
        "old": """    (1, "wf_10c6cbd2-cad", Outcome.CONDITIONAL, "evaluator_critique_86.28.md::ledger"),""",
        "new": """    (1, "wf_23d9ed4b-22c", Outcome.NO_VERDICT, "evaluator_critique_86.28.md::ledger"),""",
        "expect_named": [
            "test_fixture_matches_the_recorded_ledger",
            "test_no_non_attempt_run_id_leaked_into_the_fixture",
        ],
    },
]


def run_suite() -> tuple[bool, list[str]]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(SUITE), "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    failed = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("FAILED ") or s.startswith("ERROR "):
            frag = s.split(" ", 1)[1].split(" - ")[0]
            failed.append(frag.rsplit("::", 1)[-1])
    return (proc.returncode == 0 and not failed), failed


def main() -> int:
    src = TARGET.read_text()
    md5_before = hashlib.md5(src.encode()).hexdigest()
    backup = Path(tempfile.gettempdir()) / "attempt_budget_86_32_backup.py"
    backup.write_text(src)

    print("=" * 76)
    print("phase-86.32 mutation matrix -- cumulative attempt budget")
    print("=" * 76)
    print(f"\n[target] {TARGET.relative_to(REPO)}  md5={md5_before}")

    green, failed = run_suite()
    print(f"[control] unmutated suite green: {green}")
    if not green:
        print(f"  ABORT -- control is RED ({failed}); no later red would mean anything.")
        return 1

    results = []
    try:
        for cell in CELLS:
            n = src.count(cell["old"])
            if n == 0:
                results.append((cell["id"], "ANCHOR-MISS", cell["what"], []))
                print(f"\n[{cell['id']}] ANCHOR NOT FOUND -- vacuous, NOT a pass")
                continue
            if n > 1:
                results.append((cell["id"], "ANCHOR-AMBIGUOUS", cell["what"], []))
                print(f"\n[{cell['id']}] anchor matches {n}x -- ambiguous")
                continue

            TARGET.write_text(src.replace(cell["old"], cell["new"], 1))
            _, failed = run_suite()
            named_red = [t for t in cell["expect_named"] if t in failed]
            status = "KILLED" if named_red else ("MIS-ATTRIB" if failed else "SURVIVED")
            results.append((cell["id"], status, cell["what"], failed))
            print(f"\n[{cell['id']}] {status}  ({cell['what']})")
            print(f"    expected red: {cell['expect_named']}")
            print(f"    actual  red: {failed or '(none)'}")
    finally:
        TARGET.write_text(backup.read_text())
        md5_after = hashlib.md5(TARGET.read_text().encode()).hexdigest()
        restored = md5_after == md5_before
        print(f"\n[restore] md5 {md5_after} -- byte-identical: {restored}")

    print("\n" + "=" * 76)
    for cid, status, what, failed in results:
        print(f"  {status:<18} {cid:<32} {what}")
    print("=" * 76)

    if not restored:
        print(f"RESULT: TARGET NOT RESTORED -- recover from {backup}")
        return 2
    bad = [r for r in results if r[1] != "KILLED"]
    if bad:
        print(f"RESULT: {len(bad)} of {len(results)} cells NOT killed -- the suite has holes.")
        return 1
    green_after, failed_after = run_suite()
    print(f"post-restore suite green: {green_after} {failed_after or ''}")
    print(f"RESULT: all {len(results)} cells KILLED, control green, target restored.")
    return 0 if green_after else 1


if __name__ == "__main__":
    raise SystemExit(main())
