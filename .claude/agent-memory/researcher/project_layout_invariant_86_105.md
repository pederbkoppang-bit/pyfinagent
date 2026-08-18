---
name: layout-invariant-86-105
description: handoff layout gate 86.105 -- the step's own audit_basis (6 findings) does not reproduce (667); STEP_ID_RE matches 0/664 and is owned by pending 75.11.4; all 3 move destinations already exist and are stale; the "safe to re-run" backfill would sweep 666/666
metadata:
  type: project
---

Step 86.105 ("handoff layout invariant is red -- six violations on 2026-08-17").
Research gate run 2026-08-17. Every number below was MEASURED, not quoted.

**The step's own `audit_basis` does not reproduce.** It asserts
"exit 1 with exactly the six findings named in notes". Re-running
`python3 scripts/housekeeping/verify_handoff_layout.py` the same day gives
**667**: 664 `has no step-id prefix` + 2 `is a log` + 1 `is audit output`. The
three ROOT-level findings reproduce exactly; the "three stray current/ files" are
3 members of a 664-member class all carrying the *identical* message. A sample was
filed as a census.

**Root cause is a dead regex that is already filed elsewhere.**
`verify_handoff_layout.py:51` `STEP_ID_RE = r"^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$"`
is anchored to a LEADING sid (`82.4-name.md`) while every artifact since ~phase-75
uses the inverse `name_<sid>.md` -- the form `.claude/rules/research-gate.md` and
`research-gate.js`'s `brief_path` both MANDATE. Matches **0 of 664**. So the
`status == "done"` arm at `:120-125` has zero reachable cells and the checker's
primary documented invariant is dead. Commit `c3286524` (2026-07-31, the last
commit to touch the file) already root-caused this -- "matches 0 of 127 .md
files" -- and says verbatim "It does NOT fix the regex -- that belongs to pending
step **75.11.4**, deliberately." Denominator grew 127 -> 664 in 17 days.

**Why to distrust "idempotent; safe to re-run".**
`.claude/rules/research-gate.md:326-327` says that of
`backfill_handoff_archive.py`. FALSE: `:64` carries a byte-identical copy of the
dead regex and `:154-157` routes every non-match to `_move(p, MISC)`. Measured
read-only: **666 of 666** `.md` files in `handoff/current/` would be swept --
including the live `contract_86.97.md` and the in-flight research brief itself.
The same script already did this on 2026-07-24 (commit fa9aaf8e), took
`evaluator_critique.json`, and ran the verdict gate dark for **13 consecutive
step closes**.

**The trap that is easy to miss: all three move destinations ALREADY EXIST.**
`handoff/logs/autoresearch.log` (4,780 B, Apr 19), `handoff/logs/autoresearch.launchd.log`
(0 B, May 7), `handoff/audit/prompt_leak_redteam_audit.jsonl` (**2,035 B, Jun 11**)
-- against live sources of 265,305 B / 0 B / 48,840 B. A bare `mv`/`git mv`
clobbers each. The `.jsonl` case deletes 2,035 bytes of append-only history the
source does not contain, i.e. it satisfies "byte-identical across the move" for
the SOURCE while silently destroying the DESTINATION. This is a split-brain
append-only stream: MERGE, don't overwrite. Precedent already in-repo at
`verify_handoff_layout.py:69-85` (phase-36.8 kill-switch archive merge).

**Ownership of the three writers.** `run_nightly.sh:12` (repo, one-line fix);
`~/Library/LaunchAgents/com.pyfinagent.autoresearch.plist` `StandardOutPath` +
`StandardErrorPath` (**OPERATOR-owned**, needs bootout+bootstrap -> numbered ask);
`scripts/audit/prompt_leak_redteam.py:39` (repo, live). Only the `.jsonl` is
git-tracked -- `.gitignore:77` `handoff/*.log` makes `git mv` inapplicable to the
two logs. `scripts/ops/run_ablation.sh:14` already writes to `handoff/logs/` and
is the proven in-repo template for the repoint.

**Why `current/` has 664 files at all:** `.claude/hooks/archive-handoff.sh:2-3`
COPIES into `handoff/archive/phase-<id>/` (835 archive dirs vs 843 done steps, so
archiving works) and never removes from `current/`; `:42-52` seeds
`.claude/.archive-baseline.json` so it never retro-archives.

**Why:** filed 2026-08-17 from the external harness audit (finding D3). The
criteria are immutable once frozen, and criterion 1 as worded is not satisfiable
-- it demands `exit 0` (needs the 664 class cleared, i.e. 75.11.4's regex) AND
that a six-finding before-run be quoted (no such run exists at this tree state).

**How to apply:** if a future step touches `handoff/` layout -- (1) run the
checker and count the classes before believing any filed violation count;
(2) never run `backfill_handoff_archive.py` until 75.11.4 lands; (3) stat the
DESTINATION before any move; (4) treat `handoff/current/` mass moves as colliding
with the open exposure in [[write-first-collision-86-43]]. See also
[[research-gate-discipline]].
