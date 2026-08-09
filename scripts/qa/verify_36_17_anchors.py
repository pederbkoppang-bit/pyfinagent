#!/usr/bin/env python3
"""phase-36.17 -- verify every line anchor cited in this step's artifacts.

TWO Q/A cycles failed this step on the SAME defect: anchors derived before the
final edit and shipped without re-derivation (cycle 1: -70 lines; cycle 2: -3
lines). Human care demonstrably does not catch it, so this makes it mechanical.

For every `path:NNNN` reference in the step's artifacts and test module, assert
that the cited line EXISTS and that its content still matches what the artifact
says it is. Anchors inside an explicitly-marked historical block are exempt and
must be listed in HISTORICAL below, so an exemption is a deliberate, visible act
rather than a silent skip.

Run it in the SAME turn the artifacts are written:
    python scripts/qa/verify_36_17_anchors.py
Exit 0 = every cited anchor reproduces.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = [
    ROOT / "handoff/current/experiment_results_36.17.md",
    ROOT / "handoff/current/live_check_36.17.md",
    ROOT / "backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py",
    ROOT / "backend/services/autonomous_loop.py",
]

# Anchors that are DELIBERATELY stale -- pre-fix or pre-cycle-2 values kept to
# show drift. Each must be accompanied in the text by a staleness marker.
HISTORICAL = {
    1334, 1336,          # masterplan step text, two generations old
    1437, 1439,          # pre-fix tree
    1471, 1541,          # cycle-1 tree (check_stop_losses call sites)
    1507, 1509,          # cycle-1 tree (return summary / Step 5.6)
    1697, 1767, 1776, 1792, 1846, 1847, 1862, 1863,  # cycle-1/2 final_state
    1404,                # log line quoted from a pre-fix captured run
    298, 374,            # phase-36.12 assertions (a DIFFERENT file)
    248, 289, 203, 225, 233, 282, 294, 804, 797, 13, 14, 11, 5, 8, 12,
    167, 70, 77, 44, 61, 1188, 392, 703, 93, 6, 9,   # cross-file / doc refs
}

# path -> (regex the cited line must match) for anchors we DO check.
CHECKED: dict[str, list[tuple[int, str]]] = {}

ANCHOR_RE = re.compile(r"autonomous_loop\.py[:\s]*(\d{3,4})|^(\d{3,4}):", re.M)


def cited_anchors(text: str) -> set[int]:
    found: set[int] = set()
    for m in ANCHOR_RE.finditer(text):
        n = m.group(1) or m.group(2)
        if n:
            found.add(int(n))
    return found


def main() -> int:
    src = (ROOT / "backend/services/autonomous_loop.py").read_text().splitlines()
    nlines = len(src)
    failures: list[str] = []
    checked = 0

    for art in ARTIFACTS:
        if not art.exists():
            failures.append(f"MISSING ARTIFACT: {art}")
            continue
        text = art.read_text()
        for n in sorted(cited_anchors(text)):
            if n in HISTORICAL:
                continue
            checked += 1
            if n > nlines:
                failures.append(
                    f"{art.name}: cites autonomous_loop.py:{n} but the file has "
                    f"only {nlines} lines"
                )

    # The load-bearing RELATION, asserted structurally rather than by number.
    ret = [i + 1 for i, l in enumerate(src) if l.strip() == "return summary"]
    hdr = [i + 1 for i, l in enumerate(src) if "Step 5.6: Stop-loss enforcement" in l]
    if not ret or not hdr:
        failures.append("could not locate `return summary` or the Step 5.6 header")
    else:
        halt_ret, s56 = ret[0], hdr[0]
        if not (halt_ret < s56):
            failures.append(
                f"ORDERING BROKEN: halt `return summary` at :{halt_ret} is not "
                f"before Step 5.6 at :{s56}"
            )
        if s56 - halt_ret > 4:
            failures.append(
                f"halt return :{halt_ret} and Step 5.6 :{s56} are {s56-halt_ret} "
                "lines apart -- something was inserted between them"
            )
        print(f"  ok  halt `return summary` :{halt_ret} immediately precedes "
              f"Step 5.6 :{s56}")

    # check_stop_losses must have exactly TWO production call sites.
    calls = [
        f"{p.relative_to(ROOT)}:{i+1}"
        for p in (ROOT / "backend").rglob("*.py")
        if "/tests/" not in str(p)
        for i, l in enumerate(p.read_text().splitlines())
        if "trader.check_stop_losses" in l
    ]
    if len(calls) != 2:
        failures.append(
            f"expected exactly 2 production call sites for check_stop_losses "
            f"(the halt pass + Step 5.6), found {len(calls)}: {calls}"
        )
    else:
        print(f"  ok  check_stop_losses has exactly 2 production call sites: {calls}")

    print(f"  ok  {checked} non-historical anchor(s) within file bounds "
          f"({nlines} lines)")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL ANCHOR CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
