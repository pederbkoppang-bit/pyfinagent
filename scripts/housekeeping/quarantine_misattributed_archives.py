"""phase-75.11.4 (criterion 12) -- mark archive dirs that hold another step's work.

WHY A MARKER AND NOT A RE-SHUFFLE. The archive is the record an operator or a
future session consults to reconstruct what a step actually did. Moving files
between archive directories to "correct" them would rewrite that record on the
strength of a heuristic whose own precision is 0.9936, not 1.0 -- and 8
masterplan `verification` references point INTO `handoff/archive/`. OCFL keeps
version directories immutable for exactly this reason, and PREMIS/PROV model
provenance as an ADDITIVE assertion about an object rather than an edit to it.
So this tool only ever ADDS a file. Nothing is moved, renamed, or deleted.

WHY IT DOES NOT SHIP ITS OWN CLASSIFIER. `scripts/qa/
derive_archive_misattribution_86_29.py` already carries the validated one: a
recall gate that REFUSES to emit a census if it misses either known positive,
four synthetic controls that force BOTH answers, and an independent
second-opinion oracle (`confirm_mismatch`) with its own controls. Writing a
second classifier here would mean a second thing to keep in sync -- and the
copy-pasted declaration grammar between the hook and that census has ALREADY
drifted once (en/em-dash separators). Import, do not re-implement.

WHAT THE MARKER DELIBERATELY DOES NOT CLAIM. It says the directory's
`contract.md` declares a DIFFERENT step. It does NOT claim the other files are
wrong, and it does not claim the step never ran -- 43 of the flagged
directories mention their own sid somewhere in the head (batch contracts such
as "phase-10.5-batch (covers 10.5.0, ...)"), so a share of these are
contestable positives. The marker says so on its face, because a quarantine
notice that overstates its own certainty is just a different kind of bad
record.

Usage:
    python scripts/housekeeping/quarantine_misattributed_archives.py            # plan
    python scripts/housekeeping/quarantine_misattributed_archives.py --execute  # write
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "handoff" / "archive"
MARKER = "MISATTRIBUTION_NOTICE.md"

sys.path.insert(0, str(REPO / "scripts" / "qa"))
try:
    from derive_archive_misattribution_86_29 import (  # noqa: E402
        _mentions_sid_anywhere,
        classify,
        confirm_mismatch,
        run_controls,
        run_precision_controls,
    )
except ImportError as exc:  # pragma: no cover - import wiring
    raise SystemExit(
        f"cannot import the phase-86.29 census: {exc}. This tool has no "
        "classifier of its own by design; fix the import rather than adding one."
    ) from exc


def _notice(dir_sid: str, declares: str, contestable: bool) -> str:
    hedge = (
        "\n**This directory MENTIONS its own step id elsewhere in the contract "
        "head.** Batch contracts (\"phase-10.5-batch (covers 10.5.0, ...)\") do "
        "this legitimately, so this particular flag is CONTESTABLE -- mentioning "
        "is not declaring, and the census over-flags in exactly this shape.\n"
        if contestable else ""
    )
    return f"""# Misattribution notice -- `phase-{dir_sid}`

Written by `scripts/housekeeping/quarantine_misattributed_archives.py`
(phase-75.11.4, criterion 12). **Additive only: nothing in this directory was
moved, renamed, or deleted.**

## What was measured

This directory is named for step **{dir_sid}**, but the `contract.md` it
contains declares step **{declares}**.

Cause (phase-86.29, fixed at source): `.claude/hooks/archive-handoff.sh` used
to copy the four BARE rolling filenames (`contract.md`, `experiment_results.md`,
`evaluator_critique.md`, `research_brief.md`) into the closing step's directory
regardless of which step last wrote them, while that step's own suffix-named
artifacts (`contract_{dir_sid}.md`, ...) stayed behind in `handoff/current/`.
The hook now derives names from the step id and guards the rolling fallback
with a content declaration check, so directories created after that fix are not
affected.

## What this notice does NOT claim
{hedge}
- It does not claim the OTHER files here belong to another step; only
  `contract.md` was classified.
- It does not claim step {dir_sid} did not run. Its real artifacts may still be
  in `handoff/current/` under `*_{dir_sid}.md` names.
- The classifier's measured precision is **0.9936**, not 1.0.

## How to re-derive this

    python scripts/qa/derive_archive_misattribution_86_29.py --list-wrong

That script validates recall against two known positives and REFUSES to print a
census if it misses either, so a run that prints is itself evidence the method
has recall.
"""


def main(execute: bool) -> int:
    if not ARCHIVE.is_dir():
        print("no handoff/archive/ -- nothing to do")
        return 0

    # The census's own gates run FIRST. If the classifier cannot prove recall
    # and precision on its controls, this tool must not write anything at all --
    # a marker is a durable claim and an unvalidated classifier has no standing
    # to make one.
    if not run_controls():
        print("REFUSING: the phase-86.29 recall controls did not pass")
        return 1
    if not run_precision_controls():
        print("REFUSING: the phase-86.29 precision controls did not pass")
        return 1

    written = 0
    already = 0
    contestable_n = 0
    scanned = 0

    for d in sorted(ARCHIVE.iterdir()):
        if not d.is_dir() or not d.name.startswith("phase-"):
            continue
        scanned += 1
        dir_sid = d.name[len("phase-"):]
        verdict, declares = classify(d)
        if verdict != "mismatch" or not declares:
            continue
        # TWO DISTINCT PROPERTIES, kept apart on purpose -- the census module's
        # own docstring warns that keeping both callable "is what stops the
        # report claiming the broad one while measuring the narrow one", and an
        # earlier revision of this file made exactly that error (it labelled
        # markers "CONTESTABLE" from the NARROW oracle while the notice text
        # described the BROAD batch-contract case, reporting 1 where the broad
        # property holds for 43).
        #   broad  -- the dir's sid appears ANYWHERE in the contract head
        #   narrow -- it appears inside a DECLARATION (confirm_mismatch False)
        mentions = _mentions_sid_anywhere(d, dir_sid)
        declared_elsewhere = not confirm_mismatch(d, dir_sid)
        contestable = mentions or declared_elsewhere
        if contestable:
            contestable_n += 1
        target = d / MARKER
        if target.exists():
            already += 1
            continue
        if execute:
            target.write_text(_notice(dir_sid, declares, contestable), encoding="utf-8")
        written += 1
        print(f"[{'wrote' if execute else 'would-write'}] {d.name} "
              f"declares={declares}{' CONTESTABLE' if contestable else ''}")

    print()
    print(f"Summary: dirs-scanned={scanned} mismatched-needing-marker={written} "
          f"already-marked={already} contestable={contestable_n}")
    if not execute:
        print("DRY RUN -- nothing written. Re-run with --execute to apply.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Mark misattributed archive dirs. Dry-run by default."
    )
    ap.add_argument("--execute", action="store_true",
                    help="write the notices (default: print the plan)")
    args = ap.parse_args()
    raise SystemExit(main(execute=args.execute))
