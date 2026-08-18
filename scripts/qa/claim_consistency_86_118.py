#!/usr/bin/env python3
"""phase-86.118 -- check the artifacts' PROSE against DERIVED ground truth.

Five stale-claim defects were found across four Q/A cycles on this one step, and
every repair I made by hand missed at least one sibling. The cycle-4 verdict
named why my last attempt did not count:

    "Its scope is a HAND-ASSEMBLED PHRASE LIST, the exact 'scopes must be
     DERIVED, not typed' defect; CLEAN means 'none of the phrases I chose',
     not 'no live stale claim'."

So this derives its ground truth instead of listing phrases:

* `cells` and `targets` come from an **AST parse of the matrix source**;
* `KILLED`, the control lines and the restore lines come from the artifact's own
  **verbatim capture block**;
* the suite counts come from the artifact's own **suite capture block**;
* the prose is then checked against those, and the residual TABLE is counted.

**The check that would have caught the worst defect** is
`capture_block_has_one_restore_line_per_control`. `guardlib.MutationMatrix`
prints exactly one `restore verified:` line per target, unconditionally, so a
capture whose restore count differs from its control count was EDITED rather
than regenerated. That is precisely what happened: two lines were spliced into a
7-target capture when an 8th target was added, leaving the only cell covering
the criterion-5 fix with no restore evidence. The numbers in the edited lines
were correct and the evidence was still false.

Every check is a `guardlib.Guards.ok()` call, so none of them can be registered
without a known-bad fixture that it is re-proved to reject on every run.

    python scripts/qa/claim_consistency_86_118.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardlib import Guards  # noqa: E402

MATRIX = REPO / "scripts/qa/mutation_86_118.py"
LIVE_CHECK = REPO / "handoff/current/live_check_86.118.md"
RESULTS = REPO / "handoff/current/experiment_results_86.118.md"


def derive_from_source() -> dict:
    """cells and targets, parsed rather than counted by eye or by grep."""
    tree = ast.parse(MATRIX.read_text())
    cells = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Cell"
    )
    targets = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(
            getattr(t, "id", "") == "TARGETS" for t in n.targets
        ):
            targets = len(n.value.elts)  # type: ignore[attr-defined]
    return {"cells": cells, "targets": targets}


def derive_from_capture(text: str) -> dict:
    """Everything the artifact claims a COMMAND printed."""
    controls = re.findall(r"^control\s+\S+\s+rc=\d+", text, re.M)
    restores = re.findall(r"^restore verified: \S+", text, re.M)
    killed = re.search(r"KILLED (\d+) / (\d+)\s+SURVIVED (\d+)\s+UNSCORABLE (\d+)", text)
    suite = re.search(r"^(\d+) failed, (\d+) passed,", text, re.M)
    return {
        "controls": len(controls),
        "restores": len(restores),
        "killed": int(killed.group(1)) if killed else -1,
        "cells_scored": int(killed.group(2)) if killed else -1,
        "survived": int(killed.group(3)) if killed else -1,
        "unscorable": int(killed.group(4)) if killed else -1,
        "suite_failed": int(suite.group(1)) if suite else -1,
        "suite_passed": int(suite.group(2)) if suite else -1,
    }


def matrix_size_claims(text: str) -> list[tuple[int, int]]:
    """Every `<N> cells over <M> targets` claim, as pairs.

    Deliberately narrow. An earlier revision matched any `<N> cells`, which also
    caught the narrative sentence "Three cells UNSCORABLE on a red control" --
    a true statement about one historical run, not a claim about how big the
    matrix IS. A checker that flags a correct sentence trains its reader to
    ignore it, so the pattern matches the SIZE IDIOM only.
    """
    flat = re.sub(r"\s+", " ", text)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12, "thirteen": 13, "fourteen": 14}

    def num(tok: str) -> int | None:
        tok = tok.lower()
        return int(tok) if tok.isdigit() else words.get(tok)

    out: list[tuple[int, int]] = []
    for m in re.finditer(r"(\w+)\s+cells?\s+over\s+(\w+)\s+targets?", flat, re.I):
        ctx = flat[max(0, m.start() - 130):m.end()]
        if "earlier revision" in ctx or "An earlier" in ctx:
            continue
        a, b = num(m.group(1)), num(m.group(2))
        if a is not None and b is not None:
            out.append((a, b))
    return out


def prose_claims(text: str, noun: str) -> list[int]:
    """Every `<N> <noun>` in the prose, whitespace-insensitively.

    Flattened first because a claim can STRADDLE A LINE BREAK -- an earlier
    hand-written sweep reported CLEAN because `Eight` and `failures remain` sat
    on either side of a newline.
    """
    flat = re.sub(r"\s+", " ", text)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12, "thirteen": 13, "fourteen": 14}
    out: list[int] = []
    for m in re.finditer(r"(\w+)\s+" + noun, flat, re.I):
        tok = m.group(1).lower()
        # Skip a claim explicitly marked as a historical record.
        ctx = flat[max(0, m.start() - 130):m.end()]
        if "earlier revision" in ctx or "An earlier" in ctx:
            continue
        if tok.isdigit():
            out.append(int(tok))
        elif tok in words:
            out.append(words[tok])
    return out


def main() -> int:
    src = derive_from_source()
    lc_text = LIVE_CHECK.read_text()
    res_text = RESULTS.read_text()
    cap = derive_from_capture(lc_text)

    g = Guards(label="86.118 claim-consistency")

    # THE SPLICE DETECTOR. guardlib prints one restore line per target,
    # unconditionally, so these two counts are equal in any real run.
    g.ok(
        "capture_block_has_one_restore_line_per_control",
        lambda d: d["restores"] == d["controls"],
        cap,
        falsified_by={"restores": 7, "controls": 8},
        detail="the capture was EDITED, not regenerated -- guardlib emits one "
               "restore line per target on every run",
    )
    g.ok(
        "capture_control_count_equals_declared_targets",
        lambda d: d["cap"]["controls"] == d["src"]["targets"],
        {"cap": cap, "src": src},
        falsified_by={"cap": {"controls": 7}, "src": {"targets": 8}},
        detail="the capture is from a run with a different target list than the "
               "matrix source declares",
    )
    g.ok(
        "every_declared_cell_was_scored",
        lambda d: d["cap"]["cells_scored"] == d["src"]["cells"],
        {"cap": cap, "src": src},
        falsified_by={"cap": {"cells_scored": 13}, "src": {"cells": 14}},
        detail="the capture scored a different number of cells than the source "
               "declares -- a cell was added or removed after the run",
    )
    g.ok(
        "matrix_is_fully_green",
        lambda d: d["killed"] == d["cells_scored"] and d["survived"] == 0 and d["unscorable"] == 0,
        cap,
        falsified_by_each=[
            {"killed": 13, "cells_scored": 14, "survived": 0, "unscorable": 0},
            {"killed": 14, "cells_scored": 14, "survived": 1, "unscorable": 0},
            {"killed": 14, "cells_scored": 14, "survived": 0, "unscorable": 1},
        ],
        detail="a survivor or an unscorable cell is not a passing matrix",
    )

    # PROSE vs the capture it sits beside, in BOTH artifacts.
    for name, text in (("live_check", lc_text), ("experiment_results", res_text)):
        claims = matrix_size_claims(text)
        g.ok(
            f"{name}_prose_matrix_size_matches_source",
            lambda cs, want=(src["cells"], src["targets"]): all(c == want for c in cs),
            claims,
            falsified_by_each=[[(13, 8)], [(14, 7)]],
            detail=f"a prose 'N cells over M targets' disagrees with the "
                   f"({src['cells']}, {src['targets']}) derived from the matrix "
                   f"source; claims found: {claims}",
        )
        g.ok(
            f"{name}_states_the_matrix_size_at_least_once",
            lambda cs: len(cs) > 0,
            claims,
            falsified_by=[],
            detail="neither artifact states the matrix size, so this check would "
                   "pass over nothing -- an empty claim list is not agreement",
        )

    g.ok(
        "live_check_prose_residual_count_matches_the_suite_capture",
        lambda claims, want=cap["suite_failed"]: all(c == want for c in claims),
        prose_claims(lc_text, "failures remain"),
        falsified_by=[8],
        detail=f"a prose 'N failures remain' disagrees with the "
               f"{cap['suite_failed']} in the suite capture -- this is the "
               f"newline-straddling defect that a flat grep missed twice",
    )

    print(g.summary())
    print()
    print(f"  DERIVED from source : cells={src['cells']} targets={src['targets']}")
    print(f"  FROM the capture    : controls={cap['controls']} restores={cap['restores']} "
          f"KILLED {cap['killed']}/{cap['cells_scored']} "
          f"SURVIVED {cap['survived']} UNSCORABLE {cap['unscorable']}")
    print(f"  suite               : {cap['suite_failed']} failed, {cap['suite_passed']} passed")
    print("\nCLAIM CONSISTENCY: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
