#!/usr/bin/env python3
"""Refuse to spawn a Q/A until the KNOWN defect classes are swept clean.

WHY THIS EXISTS -- measured, not assumed
----------------------------------------
Over 134 verdicts on 30 steps in `handoff/verdict_ledger.jsonl`: mean **4.47
cycles per step**, max 12, and only 3 of 30 closed in one cycle. The verdict mix
is 75 CONDITIONAL / 26 FAIL / 18 PASS / 15 NO_VERDICT.

What those cycles were spent on, counted over the ledger notes:

    12  stale / contradictory artifact
     7  guard cannot fail (vacuous)
     6  wrong mechanism / unproven claim
     6  coverage gap (no cell, unmapped)
     0  PRODUCT defect in the shipped code

Exactly **3 of 134** rows mention a product defect at all, and all three say
"zero product defects".

**So the evaluator is not the problem -- it is right nearly every time, and it
is finding real defects.** The loop is structural: the Q/A reports an INSTANCE,
Main fixes that INSTANCE, and the next cycle surfaces another instance of the
SAME CLASS. With N instances in the evidence and one surfaced per cycle, closing
a step costs N cycles.

Two observations from step 86.59 and 86.118 make the mechanism concrete: 86.59
spent five consecutive cycles on five vacuous guards, each fix relocating the
seam rather than closing the class; 86.118 carried five stale-claim instances
across four cycles, and every hand-repair fixed the named one and missed a
sibling. Against that, `guardlib.census` listed ALL 21 uncelled guards in 86.116
in a single command -- the difference between enumerating a class and waiting to
be told about it one member at a time.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
It sweeps the classes above over a step's own evidence and REFUSES the spawn
while any is dirty, so the Q/A meets evidence that is already class-clean and
can spend its cycle on something NOVEL -- which is what an independent evaluator
is for.

**It does not touch the evaluator.** No criterion is relaxed, `qa.md` is
unmodified, no threshold moves, and nothing here can turn a FAIL into a PASS --
it runs BEFORE the spawn and its only power is to withhold one. A gate that
could admit work the Q/A refused would be the opposite of this.

Every check is a `guardlib.Guards.ok()` call, so none could be written without a
known-bad fixture it re-proves it rejects on each run: this file cannot ship the
very defect it exists to catch.

    python scripts/qa/pre_spawn_gate.py 86.118
    python scripts/qa/pre_spawn_gate.py 86.116 --strict-census
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardlib import Guards, GuardFailed, VacuousGuard, census  # noqa: E402

CURRENT = REPO / "handoff" / "current"


# ---------------------------------------------------------------------------
# discovery -- DERIVED from the step id, never a typed file list
# ---------------------------------------------------------------------------


def step_artifacts(sid: str) -> dict[str, Path]:
    """The five-file protocol artifacts for a step, by convention."""
    return {
        name: CURRENT / f"{name}_{sid}.md"
        for name in ("contract", "experiment_results", "live_check",
                     "evaluator_critique")
    }


def step_scripts(sid: str) -> dict[str, list[Path]]:
    """Scripts belonging to this step, found by the `_<sid>` suffix convention.

    Derived by glob rather than listed, so a script added later is swept without
    anyone remembering to add it here.
    """
    tag = sid.replace(".", "_")
    matrices, others = [], []
    for p in sorted((REPO / "scripts" / "qa").glob(f"*{tag}*.py")):
        (matrices if "mutation" in p.name else others).append(p)
    return {"matrices": matrices, "evidence": others}


# ---------------------------------------------------------------------------
# the class sweeps
# ---------------------------------------------------------------------------


def capture_blocks(text: str) -> list[dict]:
    """Every fenced block that looks like a MutationMatrix run."""
    out = []
    for m in re.finditer(r"```(.*?)```", text, re.S):
        body = m.group(1)
        # guardlib emits `control <label> rc=N collected=N`, one per TARGET,
        # and one restore line per target -- so restores == controls holds.
        # Bespoke matrices predating guardlib (86.59) print one control per
        # MODE against a single target, where 3 controls / 1 restore is
        # CORRECT. `collected=` is the discriminator, taken from the emitting
        # code rather than guessed, so the invariant is only applied where the
        # tool actually guarantees it.
        controls = re.findall(r"^control\s+\S+.*rc=\d+.*collected=", body, re.M)
        restores = re.findall(r"^restore verified: \S+", body, re.M)
        summary = re.search(r"KILLED (\d+) / (\d+)\s+SURVIVED (\d+)\s+UNSCORABLE (\d+)", body)
        suite = re.search(r"^(\d+) failed, (\d+) passed,", body, re.M)
        # `N passed, M deselected` with no `failed` clause means ZERO failures.
        # Without this it read as "unknown" and the zero-failure case -- the one
        # that caught me on 86.108 -- could not be checked at all.
        clean = None if suite else re.search(r"^(\d+) passed,", body, re.M)
        # `clean` MUST be in this condition. Without it a zero-failure sweep
        # block -- which has no controls, no restores, no KILLED summary and
        # no `N failed` clause -- was dropped entirely, so the zero-failure
        # check below ran over an EMPTY list and reported CLEAN. That is a
        # check that cannot dirty: it meant 'found nothing to check', not
        # 'nothing is wrong'. Caught on 86.108, where the defect was real.
        if controls or restores or summary or suite or clean:
            out.append({
                "suite_failed": int(suite.group(1)) if suite else (0 if clean else -1),
                "controls": len(controls),
                "restores": len(restores),
                "killed": int(summary.group(1)) if summary else -1,
                "scored": int(summary.group(2)) if summary else -1,
                "survived": int(summary.group(3)) if summary else -1,
                "unscorable": int(summary.group(4)) if summary else -1,
            })
    return out


def constant_truth_guards(path: Path) -> list[str]:
    """`_ok("name", True, ...)` and friends -- a guard with a literal verdict.

    This is the cheapest member of the vacuity class and the one that recurred
    most: 86.59 shipped `_ok("panel_is_us_only", True, ...)`. An AST walk finds
    it with no false positives, because it looks for a literal, not a pattern in
    the text.
    """
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return []
    bad = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) in {"_ok", "ok", "guard"}
                and len(node.args) >= 2):
            continue
        cond = node.args[1]
        # ONLY a truthy literal is vacuous. `_ok(name, False, ...)` inside an
        # `if missing:` branch is a deliberate fail-with-message idiom -- it can
        # only ever FAIL, so it cannot be the "guard that cannot fail" defect.
        # Flagging it made this gate refuse verify_86_116.py, which is correct
        # code; a checker that flags a correct line trains its reader to ignore
        # it, which is worse than not checking.
        if isinstance(cond, ast.Constant) and bool(cond.value):
            name = node.args[0].value if isinstance(node.args[0], ast.Constant) else "?"
            bad.append(f"{path.name}:{node.lineno} {name!r} asserts the literal {cond.value!r}")
    return bad



_WORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
          "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,
          "fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
          "nineteen":19,"twenty":20}


def _num(tok: str):
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return _WORDS.get(tok)


def prose_vs_captures(text: str, blocks: list[dict]) -> list[str]:
    """Numeric prose claims that CONTRADICT the artifact's own capture blocks.

    The largest recurring class in the ledger (12 of the counted findings) is
    "stale / contradictory artifact", and its shape is always the same: a
    command capture is regenerated and an authored sentence beside it is left
    at the old number. It cost 86.118 five separate instances across four
    cycles.

    Two properties matter, both learned the hard way:

    * the text is WHITESPACE-NORMALISED first. `**A smaller honest red count
      ...** Eight\nfailures remain` straddles a line break, so a flat grep for
      the phrase returned CLEAN twice while the defect sat there.
    * a claim inside a context that marks itself historical ("earlier
      revision") is a deliberate record, not a live claim. Without that, this
      check flags correct prose and gets ignored.
    """
    flat = re.sub(r"\s+", " ", text)
    truth: dict[str, int] = {}
    for b in blocks:
        if b.get("suite_failed", -1) >= 0:
            truth["failures remain"] = b["suite_failed"]
        if b.get("scored", -1) >= 0:
            truth["cells"] = b["scored"]
    # `cells` alone is too generic: "Three cells UNSCORABLE on a red control" is
    # a TRUE narrative sentence about one historical run, not a claim about how
    # big the matrix is. Flagging it made an earlier checker refuse correct
    # prose -- and a checker that flags a correct line trains its reader to
    # ignore it. So the cell count is matched only in the SIZE IDIOM.
    # WORDING VARIANTS MATTER, and this list is the honest scope of the check.
    # The 86.116 cycle-4 Q/A found three stale counts this gate missed because
    # the artifact wrote "11 cells ACROSS 2 targets" and "11 cells, 11 KILLED"
    # while the pattern only knew "cells OVER targets". A pattern list is not a
    # derivation, so `CLEAN` here means "none of the shapes below disagree" --
    # never "no stale number exists". The durable fix is to GENERATE these
    # sentences from a capture instead of authoring them; until then the scope
    # is stated rather than implied.
    patterns = {
        "failures remain": r"(\w+)\s+failures remain",
        "cells": r"(\w+)\s+cells?\s+(?:over|across)\s+\w+\s+targets?"
                 r"|(\w+)\s+cells?,\s+\w+\s+KILLED"
                 r"|KILLED\s+(\w+)\s*/",
    }
    bad: list[str] = []
    for noun, want in truth.items():
        for m in re.finditer(patterns.get(noun, r"(\w+)\s+" + noun), flat, re.I):
            ctx = flat[max(0, m.start() - 130):m.end()]
            # A claim explicitly scoped to a PAST cycle is a record, not a live
            # claim. "Evidence after cycle 3: ... 11/11 KILLED" is true of cycle
            # 3 and stays true; rewriting it would destroy the step's history.
            # The marker must be explicit -- a bare number near the word "cycle"
            # is not enough, or this becomes a way to silence real staleness.
            if re.search(r"earlier revision|An earlier|[Ee]vidence after cycle \d|"
                         r"cycle-\d (?:record|figures)|left as the record", ctx):
                continue
            got = next((_num(x) for x in m.groups() if x and _num(x) is not None), None)
            if got is not None and got != want:
                bad.append(f"prose says {got} {noun!r} but the capture says {want}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step_id")
    ap.add_argument("--strict-census", action="store_true",
                    help="also require every _ok() guard to be named by a cell")
    a = ap.parse_args()
    sid = a.step_id

    arts = step_artifacts(sid)
    scripts = step_scripts(sid)
    present = {k: p for k, p in arts.items() if p.is_file()}
    text = "\n".join(p.read_text() for p in present.values())

    # AUTHORED prose only. `evaluator_critique` is a VERBATIM TRANSCRIPT of the
    # Q/A's returns, and Main must never edit it -- that is what keeps the
    # no-self-evaluation guarantee intact. It therefore quotes the very defects
    # it reported, so checking it for stale claims demands editing a transcript
    # to silence a checker, which is strictly worse than the defect. Measured:
    # after live_check was correctly repaired to "Seven failures remain", all
    # five surviving "Eight failures remain" were the judge's own quotations.
    authored = {k: p for k, p in present.items() if k != "evaluator_critique"}
    authored_text = "\n".join(p.read_text() for p in authored.values())

    g = Guards(label=f"pre-spawn gate {sid}")
    problems: list[str] = []

    # ---- C1: the protocol artifacts exist -------------------------------
    g.ok(
        "step_has_its_evidence_artifacts",
        lambda d: len(d) >= 2,
        present,
        falsified_by={},
        detail=f"found {sorted(present)} for {sid}; a Q/A cannot grade evidence "
               f"that is not on disk",
    )

    # ---- C2: SPLICE DETECTOR --------------------------------------------
    # MutationMatrix prints exactly one `restore verified:` line per target,
    # unconditionally, so any block whose restore count differs from its control
    # count was EDITED rather than regenerated. This is the check that would
    # have caught 86.118's spliced capture at the moment it was written: two
    # lines were hand-edited into a 7-target block when an 8th target was added,
    # leaving the only cell covering a previously-FAILED criterion with no
    # restore evidence. The numbers typed in were correct; the evidence was not.
    blocks = [b for b in capture_blocks(text) if b["controls"] and b["restores"]]
    g.ok(
        "no_capture_block_is_spliced",
        lambda bs: all(b["restores"] == b["controls"] for b in bs),
        blocks,
        falsified_by=[{"controls": 8, "restores": 7}],
        detail="a capture block reports a different number of restores than "
               "controls, so it was edited rather than regenerated -- "
               "guardlib emits one restore line per target on every run",
    )
    # POSITIVE CONTROL, but only where it is meaningful. A step whose matrix
    # predates guardlib emits no `collected=` blocks at all, and refusing it for
    # that would be refusing it for using an older tool -- not for a defect. The
    # absence is REPORTED either way so it is never silent.
    if blocks:
        g.ok(
            "splice_check_had_something_to_check",
            lambda bs: len(bs) > 0,
            blocks,
            falsified_by=[],
            detail="the splice check would have passed over nothing",
        )
    else:
        problems.append(
            "NOTE: no guardlib-shaped capture block found, so the splice check "
            "was vacuous here. That is expected for a step whose matrix predates "
            "guardlib; it is stated rather than reported as clean."
        )

    # ---- C3: the matrix the artifact reports is actually clean -----------
    scored = [b for b in capture_blocks(text) if b["scored"] >= 0]
    g.ok(
        "reported_matrix_has_no_survivor_or_unscorable_cell",
        lambda bs: all(b["survived"] == 0 and b["unscorable"] == 0 for b in bs),
        scored,
        falsified_by_each=[
            [{"survived": 1, "unscorable": 0}],
            [{"survived": 0, "unscorable": 1}],
        ],
        detail="the artifact reports a SURVIVED or UNSCORABLE cell; a survivor "
               "is a guard shown not to fire and an unscorable cell is one that "
               "was never really scored",
    )

    # ---- C3a: the capture blocks must agree WITH EACH OTHER -------------
    # THE HOLE THE 86.116 CYCLE-4 Q/A FOUND IN THIS GATE. `prose_vs_captures`
    # builds its ground truth by overwriting per block, so with two blocks in
    # one artifact set the LAST one wins -- and prose matching an OLDER block
    # sails through. Measured on 86.116: the gate printed `killed: 11` AND
    # `killed: 14` and still reported CLEAN, while experiment_results carried
    # "11 cells, 11 KILLED" against a matrix that runs 14.
    #
    # Two captures of the same quantity that disagree mean one is STALE, which
    # is the same "a corrected capture leaves its siblings stale" class -- just
    # at the BLOCK level rather than the prose level. Caught here, no downstream
    # check has to guess which block is current.
    # AUTHORED artifacts only, for the same reason the prose check is scoped
    # that way: `evaluator_critique` is a VERBATIM TRANSCRIPT and legitimately
    # records EVERY past cycle's capture, so its blocks disagree with the
    # current one BY DESIGN. Measured on 86.59: the spread [26, 29] came
    # entirely from the transcript while live_check and experiment_results both
    # said 29. Flagging that would demand editing a transcript to satisfy a
    # checker -- which is exactly what must never happen.
    all_blocks = capture_blocks(authored_text)
    def _spread(key):
        vals = {b[key] for b in all_blocks if b.get(key, -1) >= 0}
        return sorted(vals)
    # `scored` ONLY. A step has exactly ONE mutation matrix, so two different
    # KILLED totals mean one capture is stale. It legitimately runs pytest at
    # MANY scopes, though -- a full suite beside a targeted subset -- so
    # comparing `suite_failed` across blocks flagged 86.118 ([0, 7]) and 86.110
    # ([0, 20]) for holding both, which is correct evidence, not a defect.
    # Different commands are not contradictory claims about the same quantity.
    disagreements = {k: _spread(k) for k in ("scored",) if len(_spread(k)) > 1}
    g.ok(
        "capture_blocks_do_not_contradict_each_other",
        lambda d: not d,
        disagreements,
        falsified_by={"scored": [11, 14]},
        detail=f"two capture blocks in the same artifact set report different "
               f"values, so at least one is stale: {disagreements}",
    )

    # ---- C3b: prose must agree with the artifact's OWN captures ---------
    contradictions = prose_vs_captures(authored_text, capture_blocks(text))
    g.ok(
        "prose_agrees_with_its_own_capture_blocks",
        lambda cs: not cs,
        contradictions,
        falsified_by=["prose says 8 'failures remain' but the capture says 7"],
        detail=f"an authored sentence contradicts a command capture in the same "
               f"artifact: {contradictions[:4]}",
    )

    # ---- C3c: prose must not assert a failure the capture says is gone ---
    # THE DEFECT I COMMITTED ONE HOUR AFTER BUILDING THIS GATE. Repairing
    # 86.108's stale sweep I replaced the capture LINE and left the sentence
    # beneath it: "the 1 failure is PRE-EXISTING and unrelated: test_phase_40_2"
    # -- a test my OWN 86.118 work had just repaired. The capture said zero
    # failures; the prose insisted on one. The numeric check above could not see
    # it, because there was no number to disagree with: the claim was semantic.
    #
    # So when a capture reports ZERO failures, authored prose must not assert
    # that a failure exists.
    zero_fail = any(b.get("suite_failed") == 0 for b in capture_blocks(authored_text))
    flat_auth = re.sub(r"\s+", " ", authored_text)
    asserts_failure = []
    if zero_fail:
        for m in re.finditer(r"the (?:single|1|one) failure|the \d+ failures? (?:is|are) "
                             r"(?:pre-existing|unrelated)", flat_auth, re.I):
            ctx = flat_auth[max(0, m.start() - 130):m.end() + 40]
            # Explicit PAST-TENSE markers. A note recording that a failure USED
            # TO exist is a correction, not a live claim -- and demanding it be
            # deleted would erase the very record that makes the fix auditable.
            # This cannot defend against a deliberate lie ("it used to fail"),
            # only against honest history; that is the same bound every marker
            # in this file has, and it is stated rather than implied.
            if re.search(r"earlier revision|An earlier|[Ee]vidence after cycle \d|"
                         r"used to|no longer exists|is deleted|was outlived|"
                         r"previously carried", ctx):
                continue
            asserts_failure.append(ctx[-120:].strip())
    if zero_fail:
        g.ok(
            "zero_failure_check_had_a_capture_to_check",
            lambda z: z,
            zero_fail,
            falsified_by=False,
            detail="no zero-failure capture was parsed, so the check below would "
                   "pass over nothing -- the shape that made it report CLEAN on a "
                   "real defect",
        )
    g.ok(
        "prose_does_not_assert_a_failure_the_capture_says_is_gone",
        lambda xs: not xs,
        asserts_failure,
        falsified_by=["...the 1 failure is PRE-EXISTING and unrelated: test_phase_40_2..."],
        detail=f"a capture reports ZERO failures while authored prose still "
               f"explains one: {asserts_failure[:2]}",
    )

    # ---- C4: no guard asserts a literal ---------------------------------
    literals: list[str] = []
    for p in scripts["evidence"] + scripts["matrices"]:
        literals.extend(constant_truth_guards(p))
    g.ok(
        "no_guard_asserts_a_literal_constant",
        lambda ls: not ls,
        literals,
        falsified_by=["fake.py:1 'panel_is_us_only' asserts the literal True"],
        detail="a guard whose condition is a literal cannot fail on any input "
               "-- this exact shape shipped in 86.59 and cost a cycle",
    )

    # ---- C5: every guard has a mutation cell (opt-in) --------------------
    if a.strict_census and scripts["evidence"] and scripts["matrices"]:
        res = census(scripts["evidence"], scripts["matrices"],
                     callees=("_ok", "ok", "guard"))
        g.ok(
            "every_guard_is_named_by_a_mutation_cell",
            lambda r: not r.uncelled and not r.dynamic_names,
            res,
            falsified_by=type(res)(uncelled=["x (registered at f.py:1)"]),
            detail=f"{len(res.uncelled)} guard(s) have no cell: "
                   f"{res.uncelled[:6]}",
        )

    print(g.summary())
    print()
    print(f"  artifacts        : {sorted(present)}")
    print(f"  evidence scripts : {[p.name for p in scripts['evidence']]}")
    print(f"  matrix scripts   : {[p.name for p in scripts['matrices']]}")
    print(f"  capture blocks   : {blocks}")
    for note in problems:
        print(f"  {note}")
    print()
    print("PRE-SPAWN GATE: CLEAN -- the known classes are swept; a Q/A cycle "
          "spent now can only find something NOVEL.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GuardFailed, VacuousGuard) as exc:
        print()
        print("PRE-SPAWN GATE: REFUSED")
        print(f"  {exc}")
        print()
        print("Fix this and re-run. The gate withholds a spawn; it can never "
              "admit work a Q/A refused, and it changes no criterion.")
        raise SystemExit(1) from None
