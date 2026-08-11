#!/usr/bin/env python3
"""phase-86.36 criterion 1 -- REPRODUCE the destruction before fixing it.

Drives TWO spawns for one step id against the CURRENT path contract and shows,
with byte counts, that the second destroys the first's record.

THIS RUNS ENTIRELY IN A SCRATCH SINK. The real
`.claude/agent-memory/qa/verdicts/` is being written by live Q/A runs in this
session and in a concurrent peer session; driving a reproduction against it would
destroy their evidence, which is the very defect under study.

WHY THE DESTRUCTION IS NOT A BUG IN THE WRITER
----------------------------------------------
phase-86.31 requires the record to be written on the FIRST tool call, before any
analysis, so that a drop leaves something behind. `resolve_wip_path()` returns a
name derived only from the step id. Those two facts compose into: the retry's
opening act erases the prior attempt's testimony. The safety property and the
hazard are the same write, which is why the remedy is a NAME change and not a
timing change -- moving the write later would break 86.31.

Exit 0 = destruction reproduced (expected on the pre-fix contract).
Exit 1 = NOT reproduced, meaning the premise no longer holds and the contract
         needs re-deriving rather than the fix proceeding.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qa_wip  # noqa: E402

STEP = "86.36"


def _spawn_writes(sink_repo: pathlib.Path, step: str, written_at: str,
                  body: str, complete: bool) -> pathlib.Path:
    """Simulate one Q/A spawn's write-first behaviour, faithfully.

    Faithful to qa.md: born INCOMPLETE on the first write, flipped to COMPLETE
    as the final act only if the run reaches its end.
    """
    p = qa_wip.resolve_wip_path(step, repo=sink_repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    marker = qa_wip.COMPLETE_MARKER if complete else qa_wip.INCOMPLETE_MARKER
    p.write_text(
        f"{marker}\n"
        f"STEP: {step}\n"
        f"WRITTEN: {written_at}\n"
        + (f"COMPLETED: {written_at}\n" if complete else "")
        + "\n" + body + "\n",
        encoding="utf-8",
    )
    return p


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pyfin_86_36_repro_"))
    try:
        print(f"scratch sink: {tmp}")
        print("(the REAL verdicts/ dir is untouched -- live Q/As are writing to it)\n")

        # ---- spawn 1: a substantial analysis that then DROPS ----------------
        analysis = "\n".join(
            f"- [finding {i}] a real blocker the evaluator established before it died"
            for i in range(1, 61)
        )
        p1 = _spawn_writes(tmp, STEP, "2026-08-11T06:00:00Z", analysis, complete=True)
        before = p1.stat().st_size
        rep1 = qa_wip.report(STEP, repo=tmp, spawned_at="2026-08-11T06:00:00Z")
        print(f"SPAWN 1 wrote  : {p1.name}")
        print(f"  bytes        : {before}")
        print(f"  status       : {rep1['status']}  recoverable={rep1['recoverable']}")

        # ---- spawn 2: the retry's FIRST write, before any analysis ----------
        p2 = _spawn_writes(tmp, STEP, "2026-08-11T06:40:00Z",
                           "(nothing yet -- this is the first tool call)", complete=False)
        after = p2.stat().st_size
        rep2 = qa_wip.report(STEP, repo=tmp, spawned_at="2026-08-11T06:40:00Z")
        print(f"\nSPAWN 2 wrote  : {p2.name}")
        print(f"  bytes        : {after}")
        print(f"  status       : {rep2['status']}")

        same_path = p1 == p2
        lost = before - after
        print("\n--- RESULT ---")
        print(f"  same path for both spawns : {same_path}   ({p1.name})")
        print(f"  bytes before / after      : {before} -> {after}   (LOST {lost})")
        print(f"  spawn 1's analysis still recoverable : "
              f"{'findings' in p1.read_text()}")

        # The destruction must be REAL, not merely a smaller file.
        destroyed = same_path and after < before and "finding 60" not in p1.read_text()
        if not destroyed:
            print("\nFAIL -- destruction NOT reproduced. The premise of 86.36 does not "
                  "hold on this code; re-derive the contract instead of fixing.")
            return 1
        print("\nOK -- destruction reproduced: one path, second write erases the first's "
              "analysis. This is the pre-fix contract behaving exactly as 86.31 "
              "specified it, which is why the fix is a NAME change, not a timing one.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
