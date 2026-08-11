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

#: phase-86.29 cycle 2 -- THE SEPARATOR IS NOT ALWAYS ASCII. The grammar below
#: originally hard-coded `--`, so a header written `# Contract — Step 76.9.2`
#: (EM-DASH) matched nothing and the directory fell into "unclassified". That is
#: not a cosmetic gap: MEASURED, 38 of the 255 unclassified dirs carry an
#: en/em-dash heading and **7 of them are genuine mismatches the census was not
#: counting** (phase-75.5.12 / 76.9.3 / 78.0 / 78.16 / 78.2 / 79.2 all hold
#: 76.9.2's contract; phase-75.1 holds 75.2's). The reported total was therefore
#: a FLOOR, not a count -- a one-character grammar gap hiding real members of the
#: population. Found by the cycle-1 Q/A run that dropped before returning a
#: verdict; re-measured by Main before the fix.
_DASH = r"(?:--|—|–)"

#: The harness runner writes its OWN per-cycle contract with no step id. Those
#: are not per-step artifacts and must not be counted as unexplained.
_HARNESS_CYCLE_RE = re.compile(r"^#\s*Sprint Contract\s*--\s*Cycle\s*\d+\s*$", re.I)

_DECLARE = [
    re.compile(rf"^#\s*Contract\s*{_DASH}\s*step\s*`?({_SID})`?", re.M | re.I),
    re.compile(rf"^#\s*(?:Sprint\s+)?Contract\s*{_DASH}\s*(?:.*?)phase-({_SID})", re.M | re.I),
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


#: phase-86.29 -- SYNTHETIC CONTROLS. Recall alone is not enough: a classifier
#: that answered "mismatch" for everything would score 2/2 on the known
#: positives and be worthless. These fixtures force the method to produce BOTH
#: answers, and the third exercises the exact shape that made the 86.19 census
#: report 211 when it was not: an ALPHANUMERIC segment (`25.A`) that an earlier
#: `[0-9]+` pattern truncated to `25`, turning 46 correct dirs into mismatches.
_CONTROLS = [
    ("positive", "99.7", "# Contract -- phase-82.54\n", "mismatch"),
    ("negative", "99.8", "# Contract -- step `99.8`\n", "agree"),
    ("alnum_sid", "25.A", "# Contract -- phase-25.A\n", "agree"),
    ("alnum_sid_wrong", "25.A", "# Contract -- phase-25\n", "mismatch"),
]


def run_controls() -> bool:
    """Prove the classifier can return BOTH answers, on fixtures in a temp dir."""
    import shutil
    import tempfile

    ok = True
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="census86_29_controls_"))
    try:
        for label, dir_sid, body, expected in _CONTROLS:
            d = tmp / f"phase-{dir_sid}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "contract.md").write_text(body)
            got, declared = classify(d)
            good = got == expected
            ok &= good
            print(f"  {label:16s} dir=phase-{dir_sid:8s} -> {got:12s} "
                  f"(declares {declared!r}, expected {expected})"
                  f"   {'ok' if good else 'CONTROL FAILED'}")
            shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ok


def _contract_head(d: pathlib.Path) -> str:
    contract = d / "contract.md"
    if not contract.exists():
        alt = sorted(d.glob("contract_*.md"))
        if not alt:
            return ""
        contract = alt[0]
    try:
        return contract.read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception:                                              # noqa: BLE001
        return ""


def _mentions_sid_anywhere(d: pathlib.Path, dir_sid: str) -> bool:
    """The BROAD property: does the dir's sid appear at all in the head?

    Deliberately distinct from `confirm_mismatch`, which tests the NARROW
    property (does it appear in a DECLARATION). Keeping both callable is what
    stops the report claiming the broad one while measuring the narrow one.
    """
    return bool(re.search(
        r"(?<![0-9A-Za-z.])" + re.escape(dir_sid) + r"(?![0-9A-Za-z.])",
        _contract_head(d),
    ))


def confirm_mismatch(d: pathlib.Path, dir_sid: str) -> bool:
    """INDEPENDENT second opinion on a mismatch, to measure precision.

    The census picks the FIRST declaration pattern that hits. If a contract
    happens to declare its own step somewhere the ordered patterns did not look
    first, the census would call it a mismatch wrongly -- which is exactly the
    86.19 failure (recall validated, precision not). So: scan the whole head for
    EVERY declaration the grammar can find, and confirm the mismatch only when
    the directory's own sid appears among NONE of them.
    """
    contract = d / "contract.md"
    if not contract.exists():
        alt = sorted(d.glob("contract_*.md"))
        if not alt:
            return False
        contract = alt[0]
    try:
        head = contract.read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception:                                              # noqa: BLE001
        return False
    found = set()
    for rx in _DECLARE:
        found.update(rx.findall(head))
    return dir_sid not in found


def run_precision_controls() -> bool:
    """Prove `confirm_mismatch` can return BOTH answers before its 1.0 is believed.

    A precision oracle that always confirms would report precision 1.0000 on any
    census whatsoever, which is indistinguishable from a real perfect score. So
    it is driven against a fixture engineered to be SUSPECT (declares another
    step first, its own step later in the head) and one engineered to be
    CONFIRMED. If the suspect fixture confirms, the oracle is vacuous and the
    precision figure is withheld.
    """
    import shutil
    import tempfile

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="census86_29_precision_"))
    try:
        d = tmp / "phase-77.7"
        d.mkdir()
        (d / "contract.md").write_text(
            "# Contract -- phase-82.54\n\nprose\n\n**Step**: `77.7`\n")
        suspect_ok = confirm_mismatch(d, "77.7") is False

        d2 = tmp / "phase-77.8"
        d2.mkdir()
        (d2 / "contract.md").write_text("# Contract -- phase-82.54\n\nno self-reference\n")
        confirmed_ok = confirm_mismatch(d2, "77.8") is True

        print(f"  can report SUSPECT   (self-declaring dir) : {suspect_ok}")
        print(f"  can report CONFIRMED (clean mismatch)     : {confirmed_ok}")
        return suspect_ok and confirmed_ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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

    # ---- 1b. SYNTHETIC CONTROLS (phase-86.29) ----------------------------
    # Recall 2/2 is satisfied by a classifier that says "mismatch" always.
    # These force both answers out of it before the census is believed.
    print("=" * 74)
    print("CONTROLS -- the method must be able to answer BOTH ways")
    print("=" * 74)
    if not run_controls():
        print("\nA CONTROL FAILED. The census is not reported from a classifier")
        print("that cannot produce the right answer on a fixture with a known truth.")
        return 1
    print("  controls 4/4 -- proceeding\n")

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

    # ---- 2b. PRECISION (phase-86.29) -------------------------------------
    # The research gate's critique of this script was "precision is unmeasured
    # and its 2 known positives are one instance". Answered here rather than
    # smoothed over: every mismatch gets an independent second opinion.
    confirmed, suspect = [], []
    for name, got in wrong:
        dir_sid = name[len("phase-"):]
        (confirmed if confirm_mismatch(ARCHIVE / name, dir_sid) else suspect).append((name, got))
    print("=" * 74)
    print("PRECISION -- every mismatch re-checked by an independent second pass")
    print("=" * 74)
    if not run_precision_controls():
        print("\n  PRECISION ORACLE IS VACUOUS -- it cannot produce both answers.")
        print("  The precision figure is WITHHELD rather than reported as 1.0.")
        return 1
    total_wrong = len(wrong)
    print(f"  mismatches reported          {total_wrong:4d}")
    print(f"  CONFIRMED (dir sid appears in no declaration in the head)  {len(confirmed):4d}")
    print(f"  SUSPECT   (dir sid DOES appear -- possible parser error)   {len(suspect):4d}")
    if total_wrong:
        print(f"  precision                    {len(confirmed) / total_wrong:.4f}")
    for name, got in suspect[:10]:
        print(f"      SUSPECT {name:22s} census said it declares {got!r}")
    if not suspect:
        # phase-86.29 cycle 2 -- THIS SENTENCE USED TO OVERSTATE ITS OWN RESULT.
        # It read "no mismatched dir mentions its own step id anywhere in its
        # contract head". MEASURED: 47 of 153 DO mention it -- e.g.
        # handoff/archive/phase-10.5.0/contract.md heads with
        # "step: phase-10.5-batch (covers 10.5.0, 10.5.1, ...)". The property the
        # oracle actually tests is the NARROWER one, and the narrow one is what
        # supports the conclusion; the broad claim was simply false. Stated at
        # its true size, with the broad number printed alongside so the
        # distinction cannot be glossed again.
        mentions = sum(
            1 for name, _g in wrong
            if _mentions_sid_anywhere(ARCHIVE / name, name[len("phase-"):])
        )
        print("  no suspects: no mismatched dir's own step id appears in any")
        print("  DECLARATION the grammar finds in its contract head, so none of")
        print("  them is the 86.19 truncation shape.")
        print(f"  STATED AT TRUE SIZE: {mentions} of {total_wrong} DO mention their own")
        print("  sid somewhere in the head (batch contracts such as")
        print("  'phase-10.5-batch (covers 10.5.0, ...)'). Mentioning is not")
        print("  declaring; only the narrow property is claimed. Note a batch")
        print("  contract also means the census can over-flag: 153 has")
        print("  contestable positives as well as the known false negatives above.")
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
