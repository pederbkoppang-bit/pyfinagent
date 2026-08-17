"""phase-75.11.4 -- the ONE step-id recogniser both handoff readers share.

WHY THIS MODULE EXISTS. `backfill_handoff_archive.py` and
`verify_handoff_layout.py` each carried a byte-identical, PREFIX-ONLY
`STEP_ID_RE`:

    ^(?:phase-)?([0-9]+(?:\\.[0-9]+)*)[-.].*\\.md$

Measured 2026-08-17 against the live tree: that pattern matches **0** of the
727 files in `handoff/current/`, while the SUFFIX form every writer actually
uses (`contract_86.29.md`, `live_check_75.11.4.md`) matches **579** `.md`
files (608 counting the 29 `_<sid>.json`). Both numbers are correct under
their own rule; the rule has to be stated with the ratio.

Two consequences followed from that dead regex, and they are why this is P1
rather than housekeeping:

  1. In `verify_handoff_layout.py` the done-step arm became unreachable -- it
     is guarded by a match that never happens -- so the layout invariant
     could not see a done-step artifact at all.
  2. In `backfill_handoff_archive.py` an unmatched name falls to the `misc/`
     sweep, so a bare run relocates ~664 of 668 `.md` files, including
     artifacts of steps that are still IN FLIGHT. That has fired for real:
     commit `fa9aaf8e` swept 315 files and left the verdict gate dark for 13
     consecutive step closes.

THE MIGRATION SHAPE IS Fowler's ParallelChange EXPAND PHASE
(https://martinfowler.com/bliki/ParallelChange.html): recognise BOTH
conventions, rename NOTHING. The 507 historical prefix-style files stay
valid; the live suffix convention is added beside it. Contracting -- deleting
the dead PREFIX glob at `.claude/hooks/archive-handoff.sh:276` -- is a
LATER step and is deliberately not done here (that file is outside this
step's boundary).

WHAT THIS MODULE DELIBERATELY DOES NOT ABSORB. `AUDIT_KEEP_GLOBS` and
`HANDOFF_ROOT_KEEP` stay duplicated literals in both scripts.
`backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py:543`
walks each script's AST and `ast.literal_eval`s that assignment, then asserts
the two agree. Importing it from here would make it un-literal-eval-able and
break the drift test that protects the kill switch's only persistence file.
The duplication is load-bearing; do not "clean it up".

THE NAME IS NEVER THE TRUTH. This resolver returns a CANDIDATE step id from a
filename; the caller must still consult the masterplan for status and must
still honour the referenced-path protection. The archival literature is
unanimous on the point -- BagIt (RFC 8493): "filenames have no given
meaning"; OCFL addresses content by digest, not by name; Git and SWHID the
same. A filename is a hint, and this module's contract is to say so honestly
(including returning None) rather than to guess.
"""
from __future__ import annotations

import re
from typing import NamedTuple

# A step id: dotted decimal, e.g. `4`, `86.29`, `75.11.4`, `4000.1`.
_SID = r"[0-9]+(?:\.[0-9]+)*"

# LEGACY, still live for the 507 already-archived files: `phase-4.5.9-contract.md`
# / `4.5.9-contract.md`. Kept `.md`-only, byte-compatible with the regex it
# replaces -- the expand phase ADDS a form, it does not quietly widen the old one.
PREFIX_RE = re.compile(rf"^(?:phase-)?({_SID})[-.].*\.md$")

# CURRENT convention: `contract_86.29.md`, `live_check_75.11.4.md`,
# `evaluator_critique_36.13.json`, `census_78.json`.
SUFFIX_RE = re.compile(rf"^.+?_({_SID})\.(?:md|json)$")

# CURRENT convention, variant tail: `research_brief_86.59_rerun.md`,
# `research_brief_86.59_v1_backup.md`. `.claude/hooks/archive-handoff.sh`
# archives these under their own names (`${base}_${sid}_*.md`), so they ARE
# step artifacts and must not fall through to the misc sweep.
# The tail excludes `.` so it cannot swallow the extension.
VARIANT_RE = re.compile(rf"^.+?_({_SID})_[^.]*\.(?:md|json)$")


class StepRef(NamedTuple):
    """A step id recovered from a filename, plus which convention produced it.

    `convention` is reported so callers and tests can tell an expand-phase
    SUFFIX hit from a legacy PREFIX hit without re-deriving it.
    """

    sid: str
    convention: str  # "prefix" | "suffix" | "variant"


def resolve_step_id(name: str) -> StepRef | None:
    """Return the step id a handoff filename claims, or None if it claims none.

    None is a first-class answer, not a failure: a file with no step id in its
    name (a rolling file, a day report, an incident note) must be LEFT ALONE by
    the caller, never swept on the strength of an unmatched pattern. That
    inversion -- "unrecognised therefore junk" -- is the defect this module
    exists to remove.

    Order matters. SUFFIX is tried before VARIANT because
    `research_brief_86.59.md` must resolve to `86.59` and not be re-read as a
    variant, and PREFIX is tried last because it is the retiring form.
    """
    for rx, convention in ((SUFFIX_RE, "suffix"), (VARIANT_RE, "variant"), (PREFIX_RE, "prefix")):
        m = rx.match(name)
        if m:
            return StepRef(m.group(1), convention)
    return None


# Statuses whose artifacts may be relocated out of handoff/current/.
# `pending` / `in-progress` / `blocked` are live work. An UNKNOWN id (a phase
# id such as `78`, a typo, a step deleted from the masterplan) is NOT
# archivable either -- it is reported and left in place. The 2026-07-24 and
# 2026-07-25 sweeps both moved files whose step was still open.
ARCHIVABLE_STATUSES = frozenset({"done", "superseded", "dropped"})


def is_archivable(status: str | None) -> bool:
    """True only for a KNOWN, closed status. None/unknown -> False (stay put)."""
    return status in ARCHIVABLE_STATUSES
