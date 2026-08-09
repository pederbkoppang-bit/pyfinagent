#!/usr/bin/env python3
"""phase-86.21 -- a 3rd-CONDITIONAL counter that can SEE an in-flight step.

THE DEFECT
----------
The escalation rule ("3 consecutive CONDITIONALs must auto-FAIL") tells the Q/A
to count prior verdicts by grepping `handoff/harness_log.md`. That file is
written at step CLOSE. So a step still in its remediation loop -- which is
exactly when the rule is meant to bite -- has ZERO rows, and the counter reads
zero however many cycles have run. It fails OPEN, and silently.

Measured across phase-86.20's three Q/A cycles on 2026-08-09/10: the prescribed
grep returned 0 every time, and each Q/A was hand-fed its verdict history by
Main -- the party the rule constrains.

THREE THINGS THIS COUNTER DOES DIFFERENTLY, EACH FOR A MEASURED REASON
----------------------------------------------------------------------
1. **It reads a per-CYCLE ledger**, not the close-time log. LOG-is-last is
   deliberate and is preserved: nothing here writes `harness_log.md`.

2. **A silent zero is impossible.** `harness_log.md` is ~48% unparseable as a
   counting source -- measured: 574 of 1189 `## Cycle` headers carry no `phase=`
   at all -- and a parser keyed on one heading depth returns 0 for a file with
   five real verdicts in it (measured on 36.17, whose critique uses depth 1
   while 86.17/86.20 use depth 2). So this returns a STATUS, and "I could not
   parse" is a different status from "there are no verdicts". RFC 9413 5.1.

3. **The reset is not `== "PASS"`.** The corpus carries 12+ distinct `result=`
   tokens including `PASS_WITH_FINDINGS` and `PASS_AFTER_RETRY`. A reset written
   as an equality test never fires on those and silently extends a "consecutive"
   run straight across a real pass. Anything that is not a CONDITIONAL resets.

ONE PREDICATE, STATED
---------------------
The rule currently exists in two forms: `CLAUDE.md` says CONSECUTIVE-with-reset;
`.claude/agents/qa.md` says a CUMULATIVE grep while calling it consecutive.
They disagree. This implements **CLAUDE.md's**: consecutive CONDITIONALs since
the last non-CONDITIONAL verdict.

INDEPENDENCE -- STATED PLAINLY, NOT OVERSOLD
--------------------------------------------
Main writes the ledger. The Q/A has no `Write` tool and the Workflow runtime has
no filesystem access, so Main or a hook is the only possible writer. **A count
derived from a file the audited party writes is therefore ADVISORY, not
authoritative.** What makes it auditable is that the ledger is append-only and
git-committed: a retro-edit shows up as a diff in history instead of vanishing
into prose. That is a weaker claim than independence and is deliberately not
dressed up as a stronger one.

    python scripts/qa/verdict_history_86_21.py --step 36.17
    python scripts/qa/verdict_history_86_21.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "handoff" / "verdict_ledger.jsonl"
HARNESS_LOG = REPO_ROOT / "handoff" / "harness_log.md"

#: Verdict tokens that COUNT toward the consecutive run.
CONDITIONAL = "CONDITIONAL"

#: Statuses. These are the point of the whole module: "no verdicts recorded" and
#: "the source is there but I could not read it" must never collapse into 0.
OK = "ok"
LEDGER_MISSING = "ledger_missing"
NO_ROWS_FOR_STEP = "no_rows_for_step"
UNPARSEABLE = "unparseable"


class VerdictHistory:
    def __init__(self, status: str, verdicts: list[str], detail: str = "",
                 bad_lines: int = 0):
        self.status = status
        self.verdicts = verdicts
        self.detail = detail
        self.bad_lines = bad_lines

    @property
    def consecutive_conditionals(self) -> int | None:
        """Consecutive CONDITIONALs at the END of the history.

        Returns None when the count is NOT KNOWABLE. A caller that treats None
        as 0 has reintroduced the defect; the CLI below refuses to print a bare
        number in that case.
        """
        if self.status == UNPARSEABLE:
            return None
        n = 0
        for v in reversed(self.verdicts):
            if v == CONDITIONAL:
                n += 1
            else:
                break                     # ANY non-CONDITIONAL resets
        return n

    @property
    def would_auto_fail(self) -> bool | None:
        c = self.consecutive_conditionals
        if c is None:
            return None
        return c >= 2                     # a THIRD would be the auto-FAIL


def read_ledger(step_id: str, path: Path = LEDGER) -> VerdictHistory:
    if not path.exists():
        return VerdictHistory(
            LEDGER_MISSING, [],
            f"{path} does not exist -- no cycle has been recorded yet by any step")

    verdicts: list[str] = []
    bad = 0
    seen_step = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad += 1                      # counted and reported, never ignored
            continue
        if not isinstance(row, dict):
            bad += 1
            continue
        if str(row.get("step_id", "")) != step_id:
            continue
        seen_step = True
        v = row.get("verdict")
        if not isinstance(v, str) or not v.strip():
            bad += 1
            continue
        verdicts.append(v.strip().upper())

    if bad:
        # LOUD. A corrupt ledger must not be reported as an absence of verdicts:
        # that is precisely the fail-open shape this step exists to remove.
        return VerdictHistory(
            UNPARSEABLE, verdicts,
            f"{bad} unparseable row(s) in {path.name} -- the count is NOT KNOWABLE",
            bad_lines=bad)
    if not seen_step:
        return VerdictHistory(
            NO_ROWS_FOR_STEP, [],
            f"no rows for step {step_id} -- it has genuinely not been graded yet")
    return VerdictHistory(OK, verdicts, f"{len(verdicts)} verdict(s) from the ledger")


def prescribed_grep_count(step_id: str, path: Path = HARNESS_LOG) -> int:
    """What the CURRENT rule tells the Q/A to do. Kept to show the contrast."""
    if not path.exists():
        return 0
    pat = re.compile(r"^## Cycle .*phase=" + re.escape(step_id) + r" result=CONDITIONAL",
                     re.MULTILINE)
    return len(pat.findall(path.read_text(encoding="utf-8")))


def _report(step_id: str, h: VerdictHistory) -> int:
    print(f"step            : {step_id}")
    print(f"source          : {LEDGER.relative_to(REPO_ROOT)}")
    print(f"status          : {h.status}")
    print(f"detail          : {h.detail}")
    print(f"verdicts        : {' -> '.join(h.verdicts) if h.verdicts else '(none)'}")
    if h.status == LEDGER_MISSING:
        print("\nNOTE: the ledger does not exist at all. That is the BOOTSTRAP state,")
        print("not evidence that this step has no prior verdicts. Until a ledger")
        print("exists, this counter cannot improve on the grep -- treat the rule as")
        print("ARMED and supply the history explicitly. Exit code is non-zero.")
    c = h.consecutive_conditionals
    if c is None:
        print("consecutive     : NOT KNOWABLE (refusing to print 0 -- see status)")
        print("auto-FAIL armed : UNKNOWN -- treat as ARMED until the source is fixed")
    else:
        print(f"consecutive     : {c}")
        nth = {1: "1st", 2: "2nd", 3: "3rd"}.get(c + 1, f"{c + 1}th")
        print(f"auto-FAIL armed : {h.would_auto_fail}  "
              f"(a further CONDITIONAL would be the {nth})")
    g = prescribed_grep_count(step_id)
    print(f"\nfor contrast, the grep the rule currently prescribes: {g} row(s)")
    if c is not None and g != c:
        print(f"  DISAGREEMENT: ledger says {c}, harness_log grep says {g}.")
        # Be precise about WHICH divergence this is. There are two, and saying
        # the wrong one is the claim-accuracy defect this project keeps hitting.
        if g == 0 and c > 0:
            print("  CAUSE: harness_log is written at step CLOSE, so it reads 0 while")
            print("  the step is in flight. That is the blindness this counter removes.")
        else:
            print("  CAUSE: the two use DIFFERENT PREDICATES, not different data.")
            print(f"  The grep counts CUMULATIVE CONDITIONAL rows ({g}); this counter")
            print(f"  counts CONSECUTIVE ones since the last non-CONDITIONAL ({c}).")
            print("  CLAUDE.md specifies consecutive-with-reset; qa.md specifies the")
            print("  cumulative grep while calling it consecutive. Until those two are")
            print("  reconciled the escalation is ambiguous regardless of the source.")
    return 0 if h.status in (OK, NO_ROWS_FOR_STEP) else 1


def self_test() -> int:
    """Prove the counter can FAIL, and that it never returns a silent zero."""
    import tempfile

    print("SELF-TEST -- the counter must distinguish absence from corruption\n")
    ok = True

    def rows(*items):
        return "\n".join(json.dumps({"step_id": s, "verdict": v})
                         for s, v in items) + "\n"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # (i) 36.17's REAL five-verdict history, incl. the reset-on-FAIL path.
        p = tmp / "l1.jsonl"
        p.write_text(rows(("36.17", "CONDITIONAL"), ("36.17", "FAIL"),
                          ("36.17", "FAIL"), ("36.17", "CONDITIONAL"),
                          ("36.17", "CONDITIONAL")))
        h = read_ledger("36.17", p)
        got = h.consecutive_conditionals
        print(f"   (i)   36.17 real history -> consecutive={got} (expect 2), "
              f"armed={h.would_auto_fail}")
        ok &= (h.status == OK and got == 2 and h.would_auto_fail is True)

        # (ii) the reset must fire on a NON-'PASS' pass token.
        p2 = tmp / "l2.jsonl"
        p2.write_text(rows(("X", "CONDITIONAL"), ("X", "CONDITIONAL"),
                           ("X", "PASS_WITH_FINDINGS"), ("X", "CONDITIONAL")))
        h2 = read_ledger("X", p2)
        print(f"   (ii)  reset on PASS_WITH_FINDINGS -> consecutive="
              f"{h2.consecutive_conditionals} (expect 1)")
        ok &= (h2.consecutive_conditionals == 1)

        # (iii) a CORRUPT ledger must NOT report zero.
        p3 = tmp / "l3.jsonl"
        p3.write_text('{"step_id": "Y", "verdict": "CONDITIONAL"}\nnot json at all\n')
        h3 = read_ledger("Y", p3)
        print(f"   (iii) corrupt ledger -> status={h3.status}, "
              f"consecutive={h3.consecutive_conditionals} (expect None, NOT 0)")
        ok &= (h3.status == UNPARSEABLE and h3.consecutive_conditionals is None)

        # (iv) a MISSING ledger is distinct from an unparseable one.
        h4 = read_ledger("Z", tmp / "does_not_exist.jsonl")
        print(f"   (iv)  missing ledger -> status={h4.status} (expect ledger_missing)")
        ok &= (h4.status == LEDGER_MISSING)

        # (v) a step with no rows is genuinely zero, and says so distinctly.
        h5 = read_ledger("NOPE", p)
        print(f"   (v)   unknown step -> status={h5.status}, "
              f"consecutive={h5.consecutive_conditionals} (expect no_rows_for_step, 0)")
        ok &= (h5.status == NO_ROWS_FOR_STEP and h5.consecutive_conditionals == 0)

    print("\nSELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.step:
        ap.error("--step is required (or use --self-test)")
    return _report(args.step, read_ledger(args.step))


if __name__ == "__main__":
    raise SystemExit(main())
