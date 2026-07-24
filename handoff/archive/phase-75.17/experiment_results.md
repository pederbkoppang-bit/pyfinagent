# Experiment results -- Step 75.17 (absent-path verification family: triage + repair)

Date: 2026-07-24. **Execution model: Sonnet executor GENERATE (8th
delegated); Main review + independent re-measurement. Executor draft (3
disclosed deviations) at `experiment_results_75.17_draft.md`; the
executor also authored `live_check_75.17.md` (Main-reviewed and adopted;
its fixed-SHA baseline choice -- byte-identity pinned to 7739922d, the
pre-75.17 commit, rather than a moving HEAD -- is an improvement over the
contract's wording).**

## What shipped

- **The committed classifier** `scripts/qa/sweep_absent_verification_paths.py`
  (refactored from the research gate's re-runnable reference impl): pure
  importable `classify()` core + CLI; handles all four verification
  shapes (HEAD census: 720 dict / 126 str / 13 list / 24 None); negative-
  assertion detection (shell + python, full-path matched); frontend-
  relative + glob-prefix re-resolution; URL/truncated/transient skips;
  git-classification never-existed vs retired. Importability for step
  75.19's preflight recalibration documented in the module docstring --
  no duplicate nightly wiring.
- **10 `superseded_record` annotations** (sibling keys ONLY -- every
  touched step's command + success_criteria BYTE-IDENTICAL to the
  pre-step commit): 9 class-(i) go_live_drills (4.17.2-8, 4.17.11,
  4.17.12 -- plan names never matched disk; each on_disk_equivalent
  smoke_test_4_17_N.py existence-verified) + 4.14.26 class-(ii)
  (retired_by_commit f7e24d0a / phase-26.4). The three previously-
  annotated steps untouched. Masterplan diff purity: +139 lines, and the
  only 10 minus-lines are trailing-comma artifacts on `completed_at`
  lines (values unchanged -- unavoidable JSON sibling insertion).
- **45-test guard file** `test_phase_75_17_verification_paths.py`:
  byte-identity x10 vs the FIXED baseline SHA; exactly-one-
  superseded_record repo-wide (14 holders after +10; the guard encodes
  the VERIFIED 4 prior holders incl. 68.5, correcting a narrower list in
  one contract prose spot -- executor deviation 2, endorsed); the
  classifier returns EMPTY on the live tree and EXACTLY the 10 on the
  baseline tree; all-4-shapes fixture (list-shaped asserted BY SHAPE);
  resolver non-flag unit proofs. The 75.16 collection canary legitimately
  moved 1518/1534 -> 1563/1579 (+45 unmarked; 16-deselected preserved).

## Verification (Main-independent)

- Immutable command: **exit 0**. Guard file: **45 passed** (Main re-run).
- **Classifier reproduced BOTH WAYS by Main via the importable API**:
  live masterplan -> genuine set EMPTY; pre-step (git-show) masterplan ->
  EXACTLY {4.14.26, 4.17.2-8, 4.17.11, 4.17.12}. (Main's first two
  attempts crashed on Main's OWN call-site errors -- str-vs-Path and the
  return shape -- not module defects; the documented API is correct.)
- CI-equivalent tail: **1555 passed / 0 failed / 16 deselected**
  (= 75.16's 1510 + 45). Raw suite (executor shell): 8 failed -- the
  environment-dependent baseline subset, zero new.
- Ruff clean over the derived scope + new files.
- Mutations: executor **M1-M10 all KILLED** (incl. M8 byte-identity-break
  and M9 double-annotation on tmp copies only -- real criteria never
  mutated even transiently). Main independently reproduced **M7** (the
  mandatory fixture mutation: list-shaped fixture converted to dict ->
  the all-4-shapes test FAILS, proving the fixture load-bearing);
  post-restore 45/45.

## Executor deviations (3, disclosed; Main-endorsed)

1. The collection-canary bump (anticipated by that canary's own comment).
2. PRIOR_HOLDERS includes 68.5 (the verified 4-holder reality over the
   narrower 3-item list in one contract prose spot).
3. Two resolver unit tests assert exclusion (`is not None`) rather than a
   specific FP-class label because the well-formedness gate fires first
   for leading-slash tokens -- measured, documented, behaviorally
   equivalent for the guard's purpose.

## Governance meaning

Every status=done step in the masterplan now has either a RUNNABLE
verification command or an explicit, criteria-immutable supersession
record naming the on-disk equivalent or retiring commit -- the
"unreproducible PASS" family is closed, and the committed classifier
makes regression detectable (75.19 charters continuous enforcement).

## Not verified live
Nothing live-relevant: repo + masterplan annotations only; no service,
UI, or deploy surface touched.
