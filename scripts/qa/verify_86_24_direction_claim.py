#!/usr/bin/env python3
"""phase-86.34 criterion 1 -- a NON-VACUOUS check that the inverted N1 direction
claim is not ASSERTED anywhere in handoff/current/live_check_86.24.md.

WHY THIS FILE EXISTS
--------------------
The cycle-1 remediation offered `grep -cF "one day behind" <file>` = 0 as proof.
That literal has NEVER been in that file -- its wording is "calendar day one\\n
behind UTC" (no "day" between "one" and "behind", and line-wrapped) -- so the
oracle returned 0 at every commit whether or not the claim was present. The Q/A
(wf_839de1e6-c3c) returned FAIL on exactly that, and it was right.

A substring grep also cannot tell an ASSERTION from a QUOTATION, and the
correction deliberately quotes the retired sentence so a reader learns how it got
there. So the property checked here is scoped:

    the claim may appear INSIDE the phase-86.34 correction block;
    it must appear NOWHERE ELSE in the file.

EVERY CHECK BELOW HAS A POSITIVE CONTROL. An assertion nobody has watched fail is
not a guard, which is the lesson this whole step is about.

Exit 0 = clean; 1 = the claim is asserted outside the correction block, or a
positive control did not fire (the checker is broken and must not report clean).
"""
from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
TARGET = REPO / "handoff/current/live_check_86.24.md"

# The load-bearing clause. Chosen because it is what makes the sentence WRONG:
# it equates a local-BEHIND fixture with the real local-AHEAD window.
CLAIM = "which is exactly the 00:00-02:00 CEST window"

BLOCK_OPEN = "[phase-86.34 CORRECTION"
# EXPLICIT END SENTINEL, phase-86.34 cycle 3.
#
# This used to close the block at the next top-level heading ("\n## "), which
# made the "inside the block" region the WHOLE of section A -- lines 12-78 --
# rather than the ~46-line correction. The cycle-2 Q/A (wf_6c44bae0-a83) proved
# the gap by executing it: its cell V3 re-asserted the claim in that ~20-line
# tail, after the correction but before "## B.", and the checker still printed
# "OK". That is vacuity shape #2 (defeated by MOVING the scanned text), in the
# one section where the claim actually lived.
#
# A sentinel makes the region exactly what the author marked, so widening it is
# a visible edit to this file rather than a side effect of where the next
# heading happens to fall.
BLOCK_CLOSE = "[END phase-86.34 CORRECTION]"


def occurrences_outside_block(text: str) -> list[int]:
    """1-indexed line numbers where CLAIM appears outside the correction block."""
    start = text.find(BLOCK_OPEN)
    if start == -1:
        # No correction block at all -> every occurrence counts as asserted.
        lo = hi = -1
    else:
        end = text.find(BLOCK_CLOSE, start)
        if end == -1:
            # Sentinel missing: FAIL CLOSED. Falling back to "rest of file"
            # would silently restore the exact over-wide scope this sentinel
            # was added to remove.
            lo = hi = -1
        else:
            lo, hi = start, end

    out = []
    pos = 0
    while True:
        i = text.find(CLAIM, pos)
        if i == -1:
            break
        if not (lo <= i < hi):
            out.append(text.count("\n", 0, i) + 1)
        pos = i + 1
    return out


def main() -> int:
    if not TARGET.exists():
        print(f"FAIL -- {TARGET} does not exist")
        return 1
    text = TARGET.read_text()

    total = text.count(CLAIM)
    outside = occurrences_outside_block(text)

    print(f"target : {TARGET.relative_to(REPO)}")
    print(f"claim  : {CLAIM!r}")
    print(f"total occurrences        : {total}")
    print(f"occurrences OUTSIDE block: {len(outside)} {outside if outside else ''}")

    # ---- positive controls: prove each check can fail -------------------
    ctl_fail = []

    # C1. If the correction block is removed, the surviving quotations must be
    #     reported as asserted. Otherwise the scoping is an unconditional pass.
    stripped = text.replace(BLOCK_OPEN, "XX-NO-BLOCK-XX")
    if total > 0 and not occurrences_outside_block(stripped):
        ctl_fail.append("C1: deleting the correction block did NOT surface the "
                        "quoted occurrences -- the scope rule passes vacuously")

    # C2. An injected assertion in a later section must be caught.
    injected = text + f"\n\n## Z. injected\n\nThe fixture is {CLAIM} it reproduces.\n"
    if not occurrences_outside_block(injected):
        ctl_fail.append("C2: an injected assertion after the block was NOT caught")

    # C3. The claim must actually be present somewhere, or this checker is
    #     guarding a string that no longer exists in any form and is dead.
    if total == 0:
        ctl_fail.append("C3: the claim string is absent entirely -- the checker "
                        "no longer guards anything; re-derive CLAIM from the file")

    for c in ctl_fail:
        print(f"  CONTROL FAILED -- {c}")
    if ctl_fail:
        print("\nFAIL -- the checker itself is not trustworthy; fix it before "
              "trusting any clean result it prints")
        return 1
    print("  positive controls: C1 ok, C2 ok, C3 ok (each verified able to fail)")

    if outside:
        print(f"\nFAIL -- the inverted direction claim is ASSERTED at line(s) {outside}")
        return 1
    print("\nOK -- the claim appears only inside the phase-86.34 correction block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
