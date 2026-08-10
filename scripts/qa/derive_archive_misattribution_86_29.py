#!/usr/bin/env python3
"""phase-86.29 -- which archived step directories hold the WRONG step's files?

MECHANISM (measured, not assumed). `.claude/hooks/archive-handoff.sh` has two
branches. The step-specific one at `:160` iterates
`"$CURRENT_DIR/${sid}-"*.md` and `"$CURRENT_DIR/phase-${sid}-"*.md`. The project
actually names its per-step files in the SUFFIX form (`contract_86.6.md`), which
matches NEITHER glob -- verified for four independent step ids, 0 matches each
against 5 matches for the suffix form. So that branch never fires, and only the
rolling branch at `:148` runs, copying whatever was last written to the
UNSUFFIXED names (`handoff/current/contract.md`, ...). Every step archived since
that rolling file was last touched is archived AS THAT FILE'S step.

THE CLASSIFIER IS THE HARD PART, NOT THE COUNT. A census is only as good as its
recall, so this script:

  1. VALIDATES RECALL FIRST against the two known positives -- `phase-86.6` and
     `phase-86.26`, both of which are known to contain phase-82.54's contract.
     **If either is reported clean the script REFUSES to print a census**, per
     the step's criterion 1: a method that misses a known member is rejected,
     not adjusted.
  2. Reports the UNCLASSIFIED remainder as its own bucket. The prior pass left
     610 dirs unparsed and those are NOT evidence of cleanliness -- folding
     them into "agree" would be the flattering error.

Read-only: it opens files and writes nothing.

    $ python scripts/qa/derive_archive_misattribution_86_29.py
    $ python scripts/qa/derive_archive_misattribution_86_29.py --list-wrong
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "handoff" / "archive"

#: The two dirs the step names as known positives. Recall is validated against
#: these BEFORE any census is believed.
KNOWN_POSITIVES = ("phase-86.6", "phase-86.26")

#: Patterns that declare which step a contract belongs to. Ordered; first hit
#: wins. Deliberately several, because the header format has drifted over 800
#: steps and a single pattern is how the prior pass left 610 unclassified.
#: A step id segment may be ALPHANUMERIC: real ids include `25.A`, `25.A10`,
#: `5.5.1`, `76.9.2`. The first version of this pattern used `[0-9]+` only, so
#: `phase-25.A` truncated to `25`, the dir name `25.A` did not equal `25`, and
#: 46 CORRECT directories were reported as mismatches. Recall had been validated
#: (2/2) and precision had NOT -- so the census read 211 when it was not.
_SID = r"[0-9]+(?:\.[0-9A-Za-z]+)*"

#: The harness runner writes its OWN per-cycle contract with no step id. Those
#: are not per-step artifacts and must not be counted as unexplained.
_HARNESS_CYCLE_RE = re.compile(r"^#\s*Sprint Contract\s*--\s*Cycle\s*\d+\s*$", re.I)

_DECLARE = [
    re.compile(rf"^#\s*Contract\s*--\s*step\s*`?({_SID})`?", re.M | re.I),
    re.compile(rf"^#\s*(?:Sprint\s+)?Contract\s*--\s*(?:.*?)phase-({_SID})", re.M | re.I),
    re.compile(rf"^\*\*Step ID\*\*:\s*`?(?:phase-)?({_SID})`?", re.M | re.I),
    re.compile(rf"^\*\*Step\*\*:\s*`?(?:phase-)?({_SID})`?", re.M),
    re.compile(rf"^step:\s*(?:phase-)?({_SID})\s*$", re.M | re.I),
    re.compile(rf"^\*\*Step id:\*\*\s*`?(?:phase-)?({_SID})`?", re.M | re.I),
    re.compile(rf"^#.*?\bphase-({_SID})\b", re.M),
]



def declared_step(text: str) -> str | None:
    head = text[:4000]
    for rx in _DECLARE:
        m = rx.search(head)
        if m:
            return m.group(1)
    return None


def classify(d: pathlib.Path):
    """(verdict, declared) for one archive dir."""
    dir_sid = d.name[len("phase-"):] if d.name.startswith("phase-") else d.name
    contract = d / "contract.md"
    if not contract.exists():
        alt = sorted(d.glob("contract_*.md"))
        if not alt:
            return "no_contract", None
        contract = alt[0]
    try:
        text = contract.read_text(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        return "unreadable", None
    got = declared_step(text)
    if got is None:
        return "unclassified", None
    return ("agree" if got == dir_sid else "mismatch"), got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-wrong", action="store_true")
    ns = ap.parse_args()

    if not ARCHIVE.is_dir():
        print(f"no archive at {ARCHIVE}")
        return 1
    dirs = sorted(p for p in ARCHIVE.glob("phase-*") if p.is_dir())

    # ---- 1. RECALL GATE, before any census is printed --------------------
    print("=" * 74)
    print("RECALL VALIDATION -- the census is not printed unless this passes")
    print("=" * 74)
    recall_ok = True
    for name in KNOWN_POSITIVES:
        d = ARCHIVE / name
        if not d.is_dir():
            print(f"  {name:16s} MISSING from the archive -- cannot validate")
            recall_ok = False
            continue
        verdict, got = classify(d)
        ok = verdict == "mismatch"
        print(f"  {name:16s} -> {verdict:13s} declares={got!r}   {'FLAGGED (correct)' if ok else 'NOT FLAGGED -- METHOD REJECTED'}")
        recall_ok &= ok
    if not recall_ok:
        print("\nRECALL FAILED. Per criterion 1 the method is REJECTED, not adjusted,")
        print("and no census is reported from it.")
        return 1
    print("  recall 2/2 -- proceeding\n")

    # ---- 2. census -------------------------------------------------------
    buckets = collections.Counter()
    wrong = []
    for d in dirs:
        verdict, got = classify(d)
        buckets[verdict] += 1
        if verdict == "mismatch":
            wrong.append((d.name, got))

    print("=" * 74)
    print(f"CENSUS over {len(dirs)} `handoff/archive/phase-*` directories")
    print("=" * 74)
    for k in ("mismatch", "agree", "unclassified", "no_contract", "unreadable"):
        if buckets[k]:
            print(f"  {k:14s} {buckets[k]:4d}")
    print()
    if buckets["unclassified"]:
        print(f"  {buckets['unclassified']} dirs matched none of the {len(_DECLARE)} declaration patterns.")
        print("  They are NOT evidence of cleanliness. Broken down rather than left opaque:")
        shapes = collections.Counter()
        for d in dirs:
            if classify(d)[0] != "unclassified":
                continue
            c = d / "contract.md"
            if not c.exists():
                alt = sorted(d.glob("contract_*.md"))
                c = alt[0] if alt else None
            if c is None:
                shapes["(no readable contract)"] += 1
                continue
            first = next((l for l in c.read_text(errors="replace")[:400].splitlines() if l.strip()), "")
            if _HARNESS_CYCLE_RE.match(first.strip()):
                shapes["harness per-cycle contract (declares NO step, by design)"] += 1
            else:
                shapes["genuinely opaque -- needs a human read"] += 1
        for k, v in shapes.most_common():
            print(f"      {v:4d}  {k}")
        print("  Only the 'genuinely opaque' row is an open question; the harness")
        print("  per-cycle contracts are not per-step artifacts at all.")
    print()

    by_declared = collections.Counter(g for _n, g in wrong)
    print("  what the mismatched dirs actually declare (top 8):")
    for step, n in by_declared.most_common(8):
        print(f"      declares phase-{step:10s} {n:4d} dir(s)")

    if ns.list_wrong:
        print("\n  every mismatched directory:")
        for name, got in wrong:
            print(f"      {name:22s} contains {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
