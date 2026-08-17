# Contract -- step 75.11.4

**Step:** 75.11.4 -- `backfill_handoff_archive.py` archives by filename
pattern, blind to step status.
**Boundary (from the masterplan step name):** `scripts/housekeeping/**` +
its tests only.

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** (`wf_d4ad1550-ecf`; 18 sources read in full, 72 distinct
URLs, audit-class loop-until-dry: 8 rounds, 2 consecutive dry, `dry=true`;
brief `research_brief_75.11.4.md`, 53,030 chars; all 18 claimed URLs
independently confirmed present in the brief by the script's artifact
cross-check).

Five findings that change what gets built. Each is a correction to the
step text, and the step text explicitly instructs re-measurement.

**1. The step's central framing is wrong, and the correct one makes the fix
smaller.** The step says "hook = PREFIX-dash vs backfill = SUFFIX-underscore".
Measured: `archive-handoff.sh` carries **BOTH** -- a live SUFFIX branch
(`:226-242`, phase-86.29) and a legacy PREFIX glob (`:276`) whose own
in-file comment records it matching **zero** files. Meanwhile
`backfill_handoff_archive.py:64` and `verify_handoff_layout.py:51` carry the
**same byte-identical PREFIX-only regex**. So the real shape is **one live
writer convention (SUFFIX) against two PREFIX-only READERS**, and both
readers can be fixed by one shared definition. The hook needs no change,
which is what keeps this inside the boundary.

**2. Three of the four step-text figures do not reproduce.** Re-measured
live (and the tree grew from 725 to 727 files *during* the measurement, so
every count below is stamped, not eternal):

| Quantity | Step text | Measured 2026-08-17 |
|---|---|---|
| `handoff/current/` PREFIX-convention files | 0 | **0** (reproduces) |
| `handoff/current/` SUFFIX-convention files | 20 | **579** `.md`, 608 incl. `.json` |
| prefix-style archived into `phase-<sid>/` | 13,198 | **507** |
| suffix-style stranded in `archive/misc/` | 428 | **488** |
| archive dirs holding another step's doc | 129 of 747 | **156 of 842**, precision 0.9936, 43 contestable |

The 579-vs-608 gap is **not** a disagreement: 579 counts `.md` only, 608
adds the 29 `_<sid>.json` files. Stating the rule with the ratio is the
point -- the same tree yields both numbers under different rules.

**3. The decisive fact the step text never states: the script is
dry-run OPT-IN.** `:218-221` -- `--dry-run` is a flag, so a **bare
invocation executes** `shutil.move` on 664 of 668 `.md` files, and the
docstring advertises the bare form as normal usage. This is not
hypothetical: it already fired (commit `fa9aaf8e` swept 315 files and left
the verdict gate dark for 13 consecutive step closes). 86.105's brief
independently quarantined the script for the same reason.

**4. "Idempotent" in this script means non-destructive, NOT convergent.**
`_safe_target` (`:114-122`) mints `-v2`, `-v3` on re-run rather than
converging -- that is exactly how `kill_switch_audit.jsonl` reached `-v3`
and `-v4`. Criterion 4 must therefore be read as *"a second run moves
nothing"*, which is a stronger property than the current code has.

**5. Do NOT write a new misattribution classifier.** `scripts/qa/
derive_archive_misattribution_86_29.py` already ships a recall gate
(refuses to print a census if it misses a known positive), synthetic
controls, and a control-tested precision oracle. Reuse it. External
consensus is unanimous that the filename must never be the truth (BagIt:
"filenames have no given meaning"; OCFL digest-not-filename; Git;
SWHID), and Fowler's ParallelChange gives the migration shape:
**expand** (recognise both conventions, rename nothing), migrate, contract.

## Two live constraints found while planning, which bound the design

Both would have broken existing green tests, so they are recorded here
rather than discovered during GENERATE:

- **`AUDIT_KEEP_GLOBS` must stay a LITERAL assignment in each script.**
  `backend/tests/test_phase_36_8_...py:543` walks each file's AST and calls
  `ast.literal_eval` on that assignment, then asserts the two sets are
  equal. Moving it into a shared module would make it un-literal-eval-able
  and break the drift test. **Therefore: share the step-id classifier,
  keep the two safety allowlists duplicated-and-drift-tested as they are.**
- **`main(dry_run: bool)`'s signature and module-global reads must not
  change.** `backend/tests/test_phase_36_7_...py:968-996` loads the script
  by `importlib`, monkeypatches `mod.REPO/HANDOFF/CURRENT/ARCHIVE/AUDIT/
  LOGS/MISC/MASTERPLAN`, then calls `mod.main(dry_run=False)`. The
  dry-run-by-default inversion therefore happens **only in the `__main__`
  argument parsing**, never in the function contract.

## Hypothesis

The defect is one shared PREFIX-only regex in two readers, plus a
migration script that executes by default, has no notion of step status,
and no notion of which files other tooling reads by literal path. Fixing
the regex in one shared place, adding a status gate and a
referenced-path refusal, and inverting the CLI default converts a
destructive sweep into a reviewable plan -- without renaming a single
artifact and without touching `.claude/hooks/**`.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. backfill_handoff_archive.py resolves the owning step id for step-suffixed handoff/current artifacts and refuses to move files whose step status is not done/superseded/dropped; unknown ids are left in place with a WARN line
2. A fixture run proves a pending-step research_brief/live_check is NOT moved while a done-step sibling IS (both asserted in one run)
3. MUTATION: remove the status check -> the pending-step fixture file moves and the test goes red; mutate the fixture (mark the pending step done) -> the not-moved assertion must flip, proving the fixture is load-bearing
4. Idempotency preserved: a second run moves nothing and exits 0
5. The script never moves a file referenced by any masterplan step's verification.command or verification.live_check -- proven by a test that plants such a reference and asserts the file stays put
6. Default invocation is a DRY-RUN printing the plan; executing requires an explicit flag
7. MUTATION: point a step's verification.live_check at a file the classifier would otherwise sweep -> the protection test goes red when the guard is removed
8. The archive hook actually archives a modern suffix-named artifact on a status flip -- proven by flipping a scratch step and observing the files land in handoff/archive/phase-<sid>/, not by reading the glob
9. verify_handoff_layout.py and archive-handoff.sh agree on ONE convention; a test asserts the same filename is classified identically by both
10. No archive directory contains a document belonging to a different step -- proven by a checker over the whole handoff/archive tree, not a spot check
11. The closing step's OWN artifacts (including suffix-named ones like live_check_<sid>.md) land in its archive directory
12. Existing falsely-populated archive directories are identified and remediated (or explicitly quarantined with a marker), with the count re-measured rather than inherited from this step's text
13. MUTATION: close a scratch step whose rolling files were last written by a different step -> the checker catches the mismatch

**Immutable verification command:**
`.venv/bin/python -m pytest backend/tests/test_phase_75_11_4_backfill_status_aware.py -q`

**Immutable live_check:** `handoff/current/live_check_75.11.4.md`: verbatim
fixture-run output (both directions), the mutation results, and one real-repo
idempotent re-run.

## Plan

**P1 -- one shared classifier (`scripts/housekeeping/handoff_naming.py`,
new, in-boundary).** Exposes the step-id resolver that accepts BOTH the
legacy `(?:phase-)?<sid>[-.]...` PREFIX form and the live
`<base>_<sid>.(md|json)` SUFFIX form, returning `(sid, convention)` or
`None`. ParallelChange **expand**: nothing is renamed, the prefix form
keeps working for the 507 historical files. Both readers import it;
`AUDIT_KEEP_GLOBS`/`HANDOFF_ROOT_KEEP` stay literal per the constraint above.

**P2 -- status gate (criteria 1-3).** Resolve sid -> masterplan status.
Archivable iff status in {done, superseded, dropped}. `pending`/`in-progress`/
`blocked` stay put. **Unknown id stays put with a WARN** -- today the unknown
branch MOVES to `misc/` while printing "left in current/ for manual review",
a summary line that contradicts the action it just took; that is fixed as
part of criterion 1.

**P3 -- referenced-path refusal (criteria 5, 7).** Build the protected set
from every `verification.command` + `verification.live_check` in the
masterplan (re-derived in cycle 2 with the function's own regex: **577**
references, **386** distinct paths, **415** into `handoff/current/`; the
557/373/395 written here in cycle 1 was carried from the brief and does not
reproduce). Refuse to move any of them, print the refusal and the step id that
claims it.

**P4 -- safe-by-default (criterion 6).** `__main__` defaults to a plan;
`--execute` performs. `--dry-run` retained as an accepted no-op alias so no
documented invocation becomes an error. `main(dry_run=...)` untouched.
Prefer `git mv` with `shutil.move` fallback, mirroring `archive-handoff.sh:279`.

**P5 -- convention agreement (criterion 9).** A test asserting that for the
same filename, `verify_handoff_layout.py`'s classifier and the names
`archive-handoff.sh` DERIVES (`${base}_${short_sid}.md` for base in
contract/experiment_results/evaluator_critique/research_brief/live_check)
agree on the step id. Both now read through P1's single definition, so
agreement is structural rather than coincidental.

**P6 -- hook behaviour (criteria 8, 11, 13) driven HERMETICALLY.**
`archive-handoff.sh:29` resolves everything from `CLAUDE_PROJECT_DIR`, so
the hook runs against a scratch tree. **This deliberately replaces
criterion 8's "flipping a scratch step"**: flipping a real step fires
`auto-commit-and-push.sh` -> `git add -A`, which would sweep a peer
session's uncommitted work into a commit named after a probe step. The
hermetic drive is strictly stronger evidence (deterministic, repeatable,
assertable in pytest) and is recorded as a deviation with its reason.
Feasibility already proven in PLAN -- two probes: derived-branch archiving
(`derived=2`), and the declaration guard reached with no derived file
(`copied=1 skipped_rolling=1`, foreign file refused **and** legitimate file
admitted, so the probe discriminates rather than merely refusing).

**P7 -- census + remediation (criteria 10, 12).** Reuse
`derive_archive_misattribution_86_29.py`. Re-measured: **156 of 842**,
precision 0.9936. Remediation is **ADDITIVE** -- write a corrective marker
into each mis-filed directory rather than re-shuffling files (OCFL keeps
version dirs immutable for exactly this reason, and 8 masterplan references
point into `handoff/archive/`). Report the count as measured, with its
contestable-positive bound (43 dirs mention their own sid in a batch
contract; mentioning is not declaring).

**P8 -- mutations (criteria 3, 7, 13)** with the control observed GREEN
first and a byte-identical restore, scored per cell, UNSCORABLE if its
control was not green.

## Scope honesty -- what this step does NOT do

- **It does not close 86.105.** 86.105 runs the fixed backfill to archive
  the 424 done-step files and merges the 3 root files. This step ships and
  proves the tool; it does not perform the archive storm.
- **It does not modify `.claude/hooks/**`.** The hook's SUFFIX branch
  already shipped under 86.29 (still `status: pending`), so criteria 8/11/13
  are *verification of existing behaviour*, not new construction. Measured
  in PLAN: 24 archive dirs carry the 86.29 `PROVENANCE.md` marker and
  **none** of them is in the mismatch list, while all 165 mismatched dirs
  lack it -- so prevention already works and criteria 10/12 are remediation
  of a CLOSED class. Stated with its bound: 24 is the entire post-guard
  population, so this is a small-sample result, not a universal proof.
- **It does not run the backfill against the live tree in execute mode.**
- **No flag is promoted, no `.env` is written, no gate is loosened.**
- `handoff/kill_switch_audit.jsonl` stays excluded under every new
  classifier; the AST drift test must still bind afterwards.

## References

`research_brief_75.11.4.md` (the 5 findings, the 18 sources, the
ParallelChange/OCFL/BagIt/SWHID basis); `contract_86.105.md` +
`research_brief_86.105.md` (the downstream consumer and the 667-finding
census); `scripts/qa/derive_archive_misattribution_86_29.py`;
`.claude/hooks/archive-handoff.sh` (86.29 revision, read not modified);
`backend/tests/test_phase_36_7_*.py` + `test_phase_36_8_*.py` (the two
tests that bound the design); Anthropic harness-design (file-based handoff
as durable state).
