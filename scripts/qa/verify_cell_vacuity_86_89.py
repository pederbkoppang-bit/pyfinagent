#!/usr/bin/env python3
"""phase-86.89 -- CELL VACUITY: does each mutation cell actually demand a guard?

WHY THIS EXISTS, AND WHY IT IS NOT MORE AST RULES.
`verify_matrix_coverage_86_85.py` derives guards from source and asks "is every
guard touched by a cell?". It answers that honestly and reports RESULT: OK. What
nobody asked is the DUAL question -- "does every cell demand a guard?" -- and the
answer today is no for 5 of 14 cells.

Kupferman (CONCUR 2006) names the distinction exactly: *"in vacuity, mutations
are in the SPECIFICATIONS, whereas in coverage, mutations are in the SYSTEM."*
Both shipped artefacts mutate the system. This one mutates the SPECIFICATION --
the matrix itself -- which is the missing half. ~20% of specifications pass
vacuously in the published corpora, and a vacuous pass always points at a real
problem.

WHY NOT JUST ENRICH THE AST RULE. The guards it misses -- ordering of a returned
sequence, composition of a dedup key, a fallback that returns rather than raises
-- are StaAgent Type-2 (INADEQUATE), not Type-1 (imprecise). Broadening a pattern
fixes imprecision, not inadequacy: SpecFuzzer's filtering moved precision
67.83% -> 74.17% with recall UNCHANGED at 54.57%. So a richer `raise`/`ast.If`
rule would buy precision and no recall, which is the wrong trade for this defect.

WHAT THIS DOES *NOT* DO -- read this before quoting any number below.
It finds cells that demand nothing. It CANNOT discover a guard nobody wrote a
cell for: a matrix with one good cell and no others scores 1/1 here. The
"1 of 4" style figure is `Recall_SD` (recall against a sample), and ISSTA'24
Thm 3.5 says that is NOT convertible to population recall without `Recall_DG`.

    python scripts/qa/verify_cell_vacuity_86_89.py
    -> exit 0 all cells demand something, exit 1 with the vacuous cells named

NOT READ-ONLY, AND THE EARLIER CLAIM THAT IT WAS IS FALSE. This checker DROPS
each cell by REWRITING `mutation_matrix_86_85.py` ON DISK and restoring it in a
`finally`. Measured over one run: **14 distinct truncated states**, 11,734..12,228
bytes against a pristine 12,407. The original's sha256 IS asserted unchanged at
exit ([3]), but that is a post-hoc check, not in-memory isolation.

This matters because the repo's auto-commit hook runs `git add -A`: an interrupt
mid-run leaves a truncated matrix that the next hook invocation would stage. Run
this DELIBERATELY, never from another script -- a cycle-2 attempt to wire it into
`mutation_matrix_86_85.py` made that file rewrite itself and was reverted.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "scripts" / "qa" / "mutation_matrix_86_85.py"
GATE = REPO / "scripts" / "qa" / "verify_matrix_coverage_86_85.py"

#: ACKNOWLEDGED DEBT, measured 2026-08-16 -- not an excuse list.
#: These five cells demand NO enumerated guard. The phase-86.85 artifacts called
#: them "coverage-redundant", which would mean another cell covers their target.
#: Measured, that is FALSE in the informative direction: each covers ZERO
#: enumerated guards and NO guard anywhere is covered by more than one cell, so
#: there is no redundancy in the map at all. Their targets are INVISIBLE to the
#: enumeration rule -- ordering of a returned sequence (M6), composition of a
#: dedup key (M9), a cycle fallback that returns rather than refuses (M11, M12),
#: and event-time vs write-time separation (M5).
#:
#: "Redundant" says the gate is complete; "invisible" says it is blind, and the
#: 86.85 artifacts recorded the reassuring one. Fixing them needs a mechanism
#: that can SEE behavioural guards, which is not this check's job -- this check's
#: job is to stop the count from being mistaken for coverage.
#: CYCLE 2 (Q/A C3). This was a bare id set, and nothing bound an id to its
#: CONTENT: the Q/A repurposed M6 -- the ORDERING cell, the defect that opened
#: this whole series -- to a benign no-op, and the run stayed byte-identically
#: GREEN. An id-keyed baseline silently re-covers whatever later takes that id.
#: CYCLE-3 CORRECTION -- THIS BINDING IS CIRCULAR AND DOES NOT WORK.
#: `payloads[cid]` is the WHOLE cell tuple, which INCLUDES the description line
#: these fingerprints were copied out of. So [6] asserts that the description
#: still contains words taken from the description. MEASURED by the cycle-2 Q/A:
#: a cell keeping its description while its behavioural payload is swapped for a
#: no-op -- or for a DUPLICATE of another cell's payload -- passes [6] at 8/8.
#: After the duplicate variant, NO cell anywhere mutates `emit_sequence` ordering
#: (the defect that opened this series) with the whole gate green.
#:
#: My cycle-2 claim that "the Q/A's repurpose mutant now KILLS" was a
#: MIS-ATTRIBUTED CREDIT: that mutant dies in the MATRIX, by a different
#: mechanism, and survives this checker.
#:
#: The real fix is to fingerprint the MUTATION PAYLOAD ONLY -- the find/replace
#: strings, excluding the description -- which is cycle-3 work, not claimed here.
#: Fingerprints are taken VERBATIM from each cell's own description line in the
#: matrix -- not from this step's prose. My first attempt wrote them from the
#: filed description and four of five did not match, which is the same
#: assert-instead-of-measure habit this series keeps surfacing.
KNOWN_VACUOUS_FINGERPRINTS = {
    "M5":  "collapse event time into write time",
    "M6":  "REVERSE emit_sequence",
    "M9":  "drop step_id from the dedup key",
    "M11": "make the cycle fallback key CONSTANT",
    "M12": "DELETE the cycle fallback entirely",
}
KNOWN_VACUOUS = set(KNOWN_VACUOUS_FINGERPRINTS)

PASSED: list[str] = []
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}{' -- ' + detail if detail else ''}")
        print(f"  FAIL {name}{' -- ' + detail if detail else ''}")


def cell_ids(src: str) -> list[str]:
    """Every cell id, derived from the matrix rather than hand-listed."""
    return [m.group(1) for m in re.finditer(r'^\s*"(M\d+)",\s*$', src, re.M)]


def cell_span(lines: list[str], cid: str) -> tuple[int, int] | None:
    """Line span of one cell tuple.

    LINE-BASED ON PURPOSE. Paren-depth counting is WRONG here -- the mutation
    strings contain `LedgerError(` with no closing paren in the same string, so a
    depth counter never rebalances and silently reports "cell not found". That
    exact failure produced a "0 of 5" reading during this step's own
    reproduction; a probe that finds nothing looks identical to a gate that
    demands nothing.
    """
    try:
        idx = next(i for i, l in enumerate(lines) if l.strip() == f'"{cid}",')
    except StopIteration:
        return None
    start = next((i for i in range(idx, -1, -1) if lines[i].rstrip("\n") == "    ("), None)
    end = next((i for i in range(idx, len(lines)) if lines[i].rstrip("\n") == "    ),"), None)
    return (start, end + 1) if start is not None and end is not None else None


def run_gate() -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(GATE)], capture_output=True, text=True,
                       cwd=REPO, timeout=400)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), "(no RESULT line)")
    return r.returncode, line


def main() -> int:
    src = MATRIX.read_text(encoding="utf-8")
    before = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    lines = src.splitlines(keepends=True)
    ids = cell_ids(src)

    print("phase-86.89 -- cell vacuity (mutating the SPECIFICATION, not the system)\n")
    print(f"  cells derived from the matrix: {len(ids)} -> {', '.join(ids)}\n")
    check("[0] the matrix yields a non-empty cell list", bool(ids),
          "no cells parsed -- every result below would be vacuous")
    if not ids:
        return 1

    rc, line = run_gate()
    check("[0] CONTROL: the gate is GREEN before any cell is dropped", rc == 0, f"exit {rc}: {line}")
    if rc != 0:
        print("\n  control is not green -- every cell result below would be unscorable")
        return 1

    payloads: dict[str, str] = {}
    demanding: list[str] = []
    vacuous: list[str] = []
    unscorable: list[str] = []
    try:
        for cid in ids:
            span = cell_span(lines, cid)
            if span is None:
                unscorable.append(cid)
                continue
            payloads[cid] = "".join(lines[span[0]:span[1]])
            MATRIX.write_text("".join(lines[:span[0]] + lines[span[1]:]), encoding="utf-8")
            rc2, _ = run_gate()
            (demanding if rc2 != 0 else vacuous).append(cid)
    finally:
        MATRIX.write_text(src, encoding="utf-8")

    after = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    print(f"  demanding : {len(demanding):2d}  {demanding}")
    print(f"  VACUOUS   : {len(vacuous):2d}  {vacuous}")
    if unscorable:
        print(f"  unscorable: {len(unscorable):2d}  {unscorable}")
    print()

    # A probe that can only ever report "demanding" proves nothing. At least one
    # cell must have been observed in EACH state, or this check cannot
    # discriminate -- the paired safe/unsafe shape NIST prescribes for
    # over-crediting, applied to the probe itself.
    check("[1] the probe DISCRIMINATES (at least one demanding cell observed)",
          bool(demanding), "no cell demanded anything -- the probe cannot tell the states apart")
    check("[2] every cell is scorable", not unscorable, f"could not locate {unscorable}")
    check("[3] the matrix is restored byte-identically", before == after,
          f"{before[:12]} -> {after[:12]}")
    # THE BASELINE, and why it is not the failure this series is about.
    # Shipping this check RED would make it a dead gate within a day -- the exact
    # 86.92 failure (a checker red for a known reason stops signalling). So the
    # five ALREADY-vacuous cells are recorded as acknowledged DEBT, and the gate
    # fails on anything else.
    #
    # This is a declaration, so the research's rule applies: a declaration may ADD
    # an obligation but must never be the DENOMINATOR. It is not one here --
    # recall is still measured against the author-independent known set (the
    # three 86.85 FAIL ledger rows), never against this list.
    #
    # It cannot rot in either direction: a NEW vacuous cell fails [4], and a
    # baselined cell that starts demanding something fails [5], which forces the
    # debt list to shrink rather than sit.
    new_vacuous = [c for c in vacuous if c not in KNOWN_VACUOUS]
    check("[4] no NEW vacuous cell", not new_vacuous,
          f"{len(new_vacuous)} cell(s) demand nothing and are not in the baseline: "
          f"{new_vacuous} -- a cell that cannot fail is not evidence")
    # CYCLE 2 (Q/A C4 note): the old message said "now demand a guard", which is
    # FALSE for the commonest cause -- the cell being ABSENT. Red for the right
    # reason, diagnosed wrongly. The two causes are now distinguished.
    fixed = [c for c in KNOWN_VACUOUS if c not in vacuous]
    absent = [c for c in fixed if c not in demanding]
    now_demanding = [c for c in fixed if c in demanding]
    detail = []
    if now_demanding:
        detail.append(f"{now_demanding} now DEMAND a guard -- delete them from "
                      "KNOWN_VACUOUS so the debt list shrinks instead of hiding progress")
    if absent:
        detail.append(f"{absent} are baselined but NO LONGER PRESENT in the matrix -- "
                      "a baselined cell was deleted; remove it from KNOWN_VACUOUS too")
    check("[5] every baselined cell is still present AND still vacuous", not fixed,
          " ; ".join(detail))

    drifted = [cid for cid, want in KNOWN_VACUOUS_FINGERPRINTS.items()
               if cid in payloads and want.lower() not in payloads[cid].lower()]
    # KNOWN-WEAK, and labelled so rather than removed: this compares the
    # fingerprint against the WHOLE cell text including the description it was
    # copied from, so it catches a description rewrite and NOT a payload swap.
    # Measured defeat recorded above. Left in place because it does catch the
    # description-rewrite case, but its name no longer over-claims.
    check("[6] each baselined cell still carries its recorded DESCRIPTION "
          "(NOTE: does NOT bind the payload -- see the cycle-3 correction)",
          not drifted,
          f"{drifted} no longer match their recorded description")

    print()
    print("LICENCE -- what this run does and does NOT license:")
    # CYCLE 2 (Q/A C6). This read "every cell in this matrix demands at least one
    # enumerated guard" -- printed in the SAME output as "VACUOUS: 5". A
    # completeness claim its own run contradicts, in exactly the object criterion
    # 6 governs. The baseline carve-out is now part of the sentence.
    print(f"  DOES: 'every cell OUTSIDE the acknowledged baseline demands at least one")
    print(f"        enumerated guard'. {len(vacuous)} baselined cell(s) demand NOTHING: {sorted(vacuous)}.")
    print("  NOT : that the guard set is complete. This CANNOT discover a guard nobody")
    print("        wrote a cell for -- a matrix of one good cell scores 1/1 here.")
    print("  NOT : population recall. Any figure derived from a chosen known-member set")
    print("        is Recall_SD; ISSTA'24 Thm 3.5 forbids converting it without Recall_DG.")
    # CARDINALITY FLOOR. phase-86.89 cycle 1, from mutation-testing this very
    # file: deleting assertion [5] or [1] left it reporting ALL GREEN, because a
    # checker cannot notice its own missing check. Two cells SURVIVED on that
    # basis. The floor is the standard remedy already used by
    # verify_escalation_86_78.mjs -- it makes a DELETED assertion fail rather
    # than silently shrink the evidence.
    ASSERTION_FLOOR = 8
    emitted = len(PASSED) + len(FAILURES)
    if emitted < ASSERTION_FLOOR:
        FAILURES.append(
            f"[floor] only {emitted} assertions ran, floor is {ASSERTION_FLOOR} -- a check "
            "was deleted or skipped. A shrinking checker reports the same ALL GREEN as a "
            "passing one, so the count is asserted."
        )
        print(f"  FAIL [floor] {emitted} assertions < floor {ASSERTION_FLOOR}")
    else:
        print(f"  ok   [floor] {emitted} assertions ran (floor {ASSERTION_FLOOR})")

    print()
    if FAILURES:
        print(f"FAILED: {len(PASSED)} passed, {len(FAILURES)} failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"ALL GREEN: {len(PASSED)} passed, 0 failed")
    return 0


def self_test() -> int:
    """Anti-NEUTERING check. phase-86.89 cycle 2, from the Q/A.

    The cardinality floor catches a DELETED assertion (count drops below the
    floor) but NOT a NEUTERED one: replacing a guard's condition with `True`
    leaves the count at 8/8 and prints ALL GREEN over a genuinely red state.
    Measured by the Q/A on two separate guards.

    A count cannot detect that, because the assertion still runs -- it just
    stops biting. So this drives the checker against a state it MUST reject and
    requires a non-zero exit. If [4] or [5] is neutered, this returns GREEN on a
    known-bad input and fails.

        python scripts/qa/verify_cell_vacuity_86_89.py --self-test
    """
    import tempfile

    print("SELF-TEST: the checker must REJECT a known-bad state\n")
    src = MATRIX.read_text(encoding="utf-8")
    before = hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    failures = 0

    # Case A: a phantom id in the baseline -- [5] must fire.
    globals()["KNOWN_VACUOUS"] = KNOWN_VACUOUS | {"M999"}
    globals()["KNOWN_VACUOUS_FINGERPRINTS"] = {**KNOWN_VACUOUS_FINGERPRINTS, "M999": "phantom"}
    PASSED.clear(); FAILURES.clear()
    rc = main()
    globals()["KNOWN_VACUOUS"] = set(KNOWN_VACUOUS_FINGERPRINTS) - {"M999"}
    globals()["KNOWN_VACUOUS_FINGERPRINTS"] = {k: v for k, v in KNOWN_VACUOUS_FINGERPRINTS.items() if k != "M999"}
    print(f"\n  case A (phantom baselined id): exit {rc} -> "
          f"{'REJECTED (guard bites)' if rc != 0 else 'ACCEPTED -- [5] IS NEUTERED'}")
    failures += rc == 0

    # Case B: a baselined cell removed from the baseline -- [4] must fire.
    globals()["KNOWN_VACUOUS"] = set(KNOWN_VACUOUS_FINGERPRINTS) - {"M6"}
    PASSED.clear(); FAILURES.clear()
    rc = main()
    globals()["KNOWN_VACUOUS"] = set(KNOWN_VACUOUS_FINGERPRINTS)
    print(f"  case B (M6 un-baselined):      exit {rc} -> "
          f"{'REJECTED (guard bites)' if rc != 0 else 'ACCEPTED -- [4] IS NEUTERED'}")
    failures += rc == 0

    ok_restore = before == hashlib.sha256(MATRIX.read_bytes()).hexdigest()
    print(f"  matrix restored byte-identical: {ok_restore}")
    _ = tempfile
    if failures or not ok_restore:
        print(f"\nSELF-TEST FAILED: {failures} guard(s) accepted a known-bad state")
        return 1
    print("\nSELF-TEST OK: every guard bit on a state it must reject")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(self_test())
    raise SystemExit(main())
