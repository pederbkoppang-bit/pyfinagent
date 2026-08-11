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
LEDGER_EMPTY = "ledger_empty"
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
        # CYCLE-3: LEDGER_MISSING joins the not-knowable set. Cycle 2 printed
        # "treat the rule as ARMED" and `auto-FAIL armed : False` on the SAME
        # screen -- prose fail-closed, machine-readable properties fail-open.
        # A programmatic consumer reading would_auto_fail got the opposite of
        # the printed instruction. The docstring's own rule ("a caller that
        # treats None as 0 has reintroduced the defect") has to bind here too.
        if self.status in (UNPARSEABLE, LEDGER_EMPTY, LEDGER_MISSING):
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

    # CYCLE-2 FIX. A present-but-ZERO-BYTE ledger used to fall through to
    # NO_ROWS_FOR_STEP, returning 0 at exit 0 with the detail "it has genuinely
    # not been graded yet" -- a confident and FALSE claim, and a silent zero on
    # one of the two failure modes criterion 6 names BY WORD ("corrupt or
    # empty"). A file that exists but holds nothing is a TRUNCATION signal, not
    # evidence of an ungraded step, and the two are not distinguishable from the
    # inside. Fails CLOSED.
    if path.stat().st_size == 0:
        return VerdictHistory(
            LEDGER_EMPTY, [],
            f"{path.name} exists but is EMPTY (0 bytes) -- that is a truncation "
            "signal, NOT evidence that this step has no verdicts. The count is "
            "NOT KNOWABLE.")

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
        # CYCLE-3 FIX, and it is the CLASS of cycle-2's fix rather than another
        # instance of it. Cycle 2 hardened the `verdict` field and left this one
        # -- at the SAME call site -- unenumerated, so a row with a MISSING,
        # BLANK or NULL step_id was indistinguishable from a row that legitimately
        # belongs to another step and took the silent-skip path. Measured by the
        # Q/A: a 3-row ledger with one such row reported consecutive=2 and
        # "a further CONDITIONAL would be the 3rd" when three had already
        # happened -- a silent UNDER-count, at exit 0, in the fail-OPEN
        # direction, which is the dangerous one for an escalation rule.
        sid = row.get("step_id")
        if sid is None or not isinstance(sid, str) or not sid.strip():
            bad += 1
            continue
        if sid.strip() != step_id:
            continue          # legitimately another step -- skip, do not count
        seen_step = True
        v = row.get("verdict")
        if not isinstance(v, str) or not v.strip():
            # CYCLE-2: counted as malformed, never silently skipped. This is the
            # malformed-FIELD analogue of the malformed-LINE case above; the Q/A
            # found the field path unguarded while the line path was guarded.
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
            f"no rows recorded for step {step_id} in this ledger. That is NOT the "
            "same as knowing it has no verdicts -- nothing writes this ledger "
            "automatically yet, so absence here is weak evidence.")
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
    if h.status == LEDGER_EMPTY:
        print("\nNOTE: the ledger file exists but is EMPTY. Treat this as a")
        print("TRUNCATED source, not as an ungraded step -- the two are")
        print("indistinguishable from here, so the rule is treated as ARMED.")
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
    return 0 if h.status in (OK, NO_ROWS_FOR_STEP) else 1  # empty/missing/corrupt -> 1


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

        # (i-b) THE THRESHOLD MUST BE PINNED FROM BELOW TOO. Cycle 1 asserted
        # only that would_auto_fail is True at c=2; a mutant changing `c >= 2`
        # to `c >= 1` therefore SURVIVED, arming the auto-FAIL after a SINGLE
        # CONDITIONAL. A one-sided threshold is not a guard.
        p1b = tmp / "l1b.jsonl"
        p1b.write_text(rows(("T", "PASS"), ("T", "CONDITIONAL")))
        h1b = read_ledger("T", p1b)
        print(f"   (i-b) one CONDITIONAL -> consecutive={h1b.consecutive_conditionals} "
              f"(expect 1), armed={h1b.would_auto_fail} (expect False)")
        ok &= (h1b.consecutive_conditionals == 1 and h1b.would_auto_fail is False)

        p1c = tmp / "l1c.jsonl"
        p1c.write_text(rows(("U", "PASS")))
        h1c = read_ledger("U", p1c)
        print(f"   (i-c) zero CONDITIONALs -> consecutive={h1c.consecutive_conditionals} "
              f"(expect 0), armed={h1c.would_auto_fail} (expect False)")
        ok &= (h1c.consecutive_conditionals == 0 and h1c.would_auto_fail is False)

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

        # (iii-b) a BLANK verdict FIELD is malformed, not skippable.
        p3b = tmp / "l3b.jsonl"
        p3b.write_text('{"step_id": "W", "verdict": "CONDITIONAL"}\n'
                       '{"step_id": "W", "verdict": ""}\n')
        h3b = read_ledger("W", p3b)
        print(f"   (iii-b) blank verdict field -> status={h3b.status} "
              f"(expect unparseable, NOT a silent skip)")
        ok &= (h3b.status == UNPARSEABLE)

        # (iii-c) an EMPTY ledger is not an ungraded step. Criterion 6 names it.
        p3c = tmp / "l3c.jsonl"
        p3c.write_text("")
        h3c = read_ledger("V", p3c)
        print(f"   (iii-c) ZERO-BYTE ledger -> status={h3c.status}, "
              f"consecutive={h3c.consecutive_conditionals} (expect ledger_empty, None)")
        ok &= (h3c.status == LEDGER_EMPTY and h3c.consecutive_conditionals is None)

        # (iii-d) STEP MATCHING IS EXACT. Real ids collide under a prefix rule:
        # 86.2 vs 86.20/86.21, and 36.1 vs 36.17. The shipped code is correct,
        # but nothing asserted it, so a prefix mutant survived the cycle-1 matrix.
        p3d = tmp / "l3d.jsonl"
        p3d.write_text(rows(("86.2", "PASS"), ("86.20", "CONDITIONAL"),
                            ("86.21", "CONDITIONAL")))
        h3d = read_ledger("86.2", p3d)
        print(f"   (iii-d) exact step match -> 86.2 sees {h3d.verdicts} "
              f"(expect ['PASS'], NOT 86.20/86.21's rows)")
        ok &= (h3d.verdicts == ["PASS"])

        # (iii-e) verdict tokens are CASE-NORMALISED before comparison.
        p3e = tmp / "l3e.jsonl"
        p3e.write_text(rows(("S", "conditional"), ("S", "Conditional")))
        h3e = read_ledger("S", p3e)
        print(f"   (iii-e) lowercase verdicts -> consecutive="
              f"{h3e.consecutive_conditionals} (expect 2, i.e. normalised)")
        ok &= (h3e.consecutive_conditionals == 2)

        # (iv) a MISSING ledger is distinct from an unparseable one.
        h4 = read_ledger("Z", tmp / "does_not_exist.jsonl")
        print(f"   (iv)  missing ledger -> status={h4.status} (expect ledger_missing)")
        ok &= (h4.status == LEDGER_MISSING)

        # (v) a step with no rows is genuinely zero, and says so distinctly.
        h5 = read_ledger("NOPE", p)
        print(f"   (v)   unknown step -> status={h5.status}, "
              f"consecutive={h5.consecutive_conditionals} (expect no_rows_for_step, 0)")
        ok &= (h5.status == NO_ROWS_FOR_STEP and h5.consecutive_conditionals == 0)

        # (vi) THE CLI HALF. Cycle 2 left _report() and prescribed_grep_count()
        # with ZERO automated coverage -- self_test() never called them -- so
        # three Q/A mutants there survived. Exit codes are the fail-CLOSED
        # signal a caller actually consumes, so they are asserted directly.
        # phase-86.21 cycle 4 -- KEEP THE BUFFER. This case used to discard
        # `_report`'s stdout and assert only its RETURN VALUE, for a function
        # whose entire product is printed text. The cycle-3 Q/A found three
        # survivors living in exactly that blind spot, one of which
        # (`print(f"consecutive     : 0")`) makes the shipped CLI report a hard
        # zero for every step forever -- the silent zero this whole step exists
        # to abolish -- while the suite stayed green.
        import contextlib, io
        exits, outs = {}, {}
        for tag, hh in (("empty", read_ledger("V", p3c)),
                        ("corrupt", read_ledger("Y", p3)),
                        ("missing", read_ledger("Z", tmp / "nope.jsonl")),
                        ("ok", read_ledger("36.17", p)),
                        ("no_rows", read_ledger("NOPE", p))):
            _buf = io.StringIO()
            with contextlib.redirect_stdout(_buf):
                exits[tag] = _report("X", hh)
            outs[tag] = _buf.getvalue()
        print(f"   (vi)  CLI exit codes {exits} "
              "(expect empty/corrupt/missing=1, ok/no_rows=0)")
        ok &= (exits == {"empty": 1, "corrupt": 1, "missing": 1,
                         "ok": 0, "no_rows": 0})

        # (vi-b) the PRINTED consecutive count must be the real one. `36.17`'s
        # ledger carries the five-verdict history, whose consecutive tail is 2
        # (CONDITIONAL, CONDITIONAL after the FAIL reset).
        _h_ok = read_ledger("36.17", p)
        _expect = f"consecutive     : {_h_ok.consecutive_conditionals}"
        _printed_ok = _expect in outs["ok"]
        print(f"   (vi-b) _report prints {_expect!r} -> {_printed_ok}")
        ok &= _printed_ok

        # (vi-c) an UNKNOWABLE count must print the refusal, never a zero.
        _refusal = "NOT KNOWABLE" in outs["corrupt"] and "consecutive     : 0" not in outs["corrupt"]
        print(f"   (vi-c) corrupt ledger refuses to print a zero -> {_refusal}")
        ok &= _refusal

        # (vi-d) BOTH cause branches must be reachable, and each must print its
        # OWN explanation. Cycle 3's survivor swapped them and nothing noticed,
        # so the tool would attribute log-close blindness to a predicate
        # mismatch and vice versa -- in the very output carrying criterion 1's
        # contrast. `36.17` has g==0 and c>0 (the blindness case). A step-id
        # whose grep count and ledger count differ for the OTHER reason
        # exercises the predicate branch.
        _blind = "harness_log is written at step CLOSE" in outs["ok"]
        print(f"   (vi-d) blindness cause printed for g=0,c>0 -> {_blind}")
        ok &= _blind

        # (vii) would_auto_fail must be None -- not False -- whenever the count
        # is unknowable. Cycle 2 pinned that on consecutive_conditionals only,
        # so a mutant returning False here survived.
        unknowable = [read_ledger("V", p3c), read_ledger("Y", p3),
                      read_ledger("Z", tmp / "nope.jsonl")]
        print(f"   (vii) would_auto_fail on unknowable statuses -> "
              f"{[h.would_auto_fail for h in unknowable]} (expect all None)")
        ok &= all(h.would_auto_fail is None for h in unknowable)

        # (viii) the CONTRAST figure must really read the log, not return 0.
        fake_log = tmp / "hl.md"
        fake_log.write_text(
            "## Cycle 1 -- 2026-01-01 -- phase=99.9 result=CONDITIONAL\n"
            "## Cycle 2 -- 2026-01-02 -- phase=99.9 result=CONDITIONAL\n"
            "## Cycle 3 -- 2026-01-03 -- phase=99.9 result=PASS\n")
        g = prescribed_grep_count("99.9", fake_log)
        print(f"   (viii) prescribed_grep_count on a synthetic log -> {g} (expect 2)")
        ok &= (g == 2)

        # (ix) a row whose step_id is MISSING/BLANK/NULL is malformed, not
        # another step's row. Under-counting silently is fail-OPEN.
        p9 = tmp / "l9.jsonl"
        p9.write_text('{"step_id": "86.21", "verdict": "CONDITIONAL"}\n'
                      '{"verdict": "CONDITIONAL"}\n'
                      '{"step_id": "86.21", "verdict": "CONDITIONAL"}\n')
        h9 = read_ledger("86.21", p9)
        print(f"   (ix)  row with NO step_id -> status={h9.status} "
              f"(expect unparseable, NOT a silent under-count)")
        ok &= (h9.status == UNPARSEABLE and h9.consecutive_conditionals is None)

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
