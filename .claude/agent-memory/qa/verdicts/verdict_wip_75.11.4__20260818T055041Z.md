STATUS: INCOMPLETE -- not a verdict
STEP: 75.11.4
WRITTEN: 2026-08-18T05:50:41Z

# Q/A write-first record -- step 75.11.4 (re-grade after operator budget extension)

## A. Attempt / sequence EVIDENCE (gathered, not applied)

`python3 scripts/qa/qa_wip.py 75.11.4 --spawned-at 2026-08-18T05:50:41Z`:
source_present=True, attempt_number=5, attempt_number_status="ok",
attempt_number_is_lower_bound=true, prior_attempts=4, records_retained=5 (GAUGE,
includes my own record), records_pruned_known=null, prior_records = 4 files
(__20260817T194109Z, __20260817T193444Z, __20260817T191121Z, __20260817T185113Z).

`python3 scripts/qa/verdict_history_86_21.py --step 75.11.4 --evidence-only`:
status="ok", "4 verdict(s) from the ledger",
sequence = FAIL -> FAIL -> NO_VERDICT -> CONDITIONAL (NO_VERDICT carried
through as-is, not dropped).

CROSS-CHECK: prior_attempts (4) is NOT greater than the ledger's verdict count
(4) -> the ledger is NOT stale for this step-id.

DISCREPANCY I MEASURED MYSELF (Main disclosed one; mine differs in mechanism).
handoff/audit/attempt_budget_audit.jsonl holds 5 pre-extension `attempt` rows
for 75.11.4: 18:21:03Z **research-gate.js**, then 18:51:08Z / 19:11:16Z /
19:34:39Z / 19:41:05Z **qa-verdict.js**. So there were FOUR Q/A spawns, not
five, and all four produced a WIP record and a ledger row. The escalation file's
"attempt 5: NO_VERDICT" is an off-by-one: the attempt gate counted the
research-gate launch in slot 1. Consequence for grading: the CONDITIONAL
(19:41:05Z launch) is the MOST RECENT Q/A verdict, and the D1-D6 fixes landed
AFTER it -- so this spawn is a re-grade on CHANGED evidence, not a re-run on the
same evidence. Operator extension row present at 2026-08-18T05:49:39Z; my own
launch row at 05:50:37Z.

## B. Harness compliance (5/5)

1. research-gate-before-contract: research_brief_75.11.4.md mtime 2026-08-17
   20:35:04 local; envelope brief_status=COMPLETE, gate_passed=true,
   external_sources_read_in_full=18 (floor 5), urls_collected=72 (floor 10),
   snippet_only=54, recency_scan_performed=true, coverage.audit_class=true,
   dry_rounds=2, K_required=2. PASS.
2. contract-before-generate: brief 20:35:04 < contract 21:05:54 <
   backfill/verifier 22:01:5x < experiment_results 22:02:54 < live_check
   22:03:18 (all local = CEST). Contract mtime post-dates handoff_naming.py
   (20:39) because cycles 2/3 edited numbers inside it -- same limitation the
   cycle-4 evaluator stated; two prior independent measurements recorded the
   original ordering. PASS with the limitation restated, not papered over.
3. experiment_results_75.11.4.md (27,196 B) + live_check_75.11.4.md (22,678 B)
   present. PASS.
4. log-last: `grep -F "phase=75.11.4" handoff/harness_log.md` -> 0 lines;
   masterplan status for 75.11.4 = "pending". Nothing flipped. PASS.
5. no-verdict-shopping: production + evidence changed AFTER the cycle-4
   critique (critique mtime 21:58:57 local; backfill 22:01:54, verifier
   22:01:57, test 22:01:12, experiment_results 22:02:54, live_check 22:03:18).
   Evidence CHANGED. PASS.

## C. Deterministic checks

- IMMUTABLE COMMAND: `.venv/bin/python -m pytest
  backend/tests/test_phase_75_11_4_backfill_status_aware.py -q`
  -> **31 passed in 1.12s, RAW_EXIT=0** (exit captured bare, not through a pipe).
- Test count re-derived: `grep -c 'def test_'` = **31**.
- LINT GATE (derived scope, xargs, non-empty asserted): union of
  `git diff --name-only 2e9597bd^ 2e9597bd -- '*.py'` + `git diff --name-only
  HEAD -- '*.py'` + `git ls-files --others --exclude-standard -- '*.py'`
  = **7 files**; `uvx ruff check --select F821,F401,F811` -> "All checks
  passed!", **exit 0**.
- Gates 1b (frontend) / 1c (live UI) N/A: this step's diff touches no
  `frontend/**` and makes no UI claim. Gate 1d: the only `backend/**` file is
  the test module, which pytest executes.
- SCOPE: the step shipped in commit `2e9597bd` (2026-08-17 22:06 local).
  Working tree today shows NO modification to scripts/housekeeping/**,
  .claude/hooks/**, .claude/masterplan.json or scripts/qa/**. The out-of-scope
  peer edits the artifact discloses (sovereign_api.py, autonomous_loop.py,
  8 frontend files) are still uncommitted and were correctly NOT swept into
  2e9597bd.
- LIVE IDEMPOTENCY (criterion 4) on the REAL tree: two consecutive bare runs,
  archive dirs **848 -> 848**, handoff/current entries **803 -> 803**, exit 0.
  The D1 dry-run-mkdir fix therefore holds in production, not only in tmp.

## D. Claim re-derivation (qa.md 4b)

| Claim in the artifacts | My independent re-derivation | Verdict |
|---|---|---|
| 31 passed | 31 passed, exit 0 | REPRODUCES |
| 156 markers = 156 mismatches | `find handoff/archive -name MISATTRIBUTION_NOTICE.md` = **156**; census "mismatches reported **156**" | REPRODUCES EXACTLY |
| census precision 0.9936, controls True | re-ran the checker: recall+precision controls both True, precision 0.9936, 1 SUSPECT (phase-69) | REPRODUCES |
| "[protected] KEEP on the LIVE tree = 20" | my FIRST probe said 24 and was WRONG -- it omitted the script's own `_is_rolling_keep` pre-filter. Applying the script's real order: **20** (the 4 excluded are contract.md / evaluator_critique.md / experiment_results.md / research_brief.md) | REPRODUCES (my probe was at fault) |
| "protected basenames in the masterplan: 381" | 381 | REPRODUCES |
| 3 empty dirs removed (phase-80.5/81.1/82.23) | all three ABSENT | REPRODUCES |
| census total=842 agree=440 ... no_contract=24 | today: agree=**441**, unclassified=222, no_contract=24, mismatch=156 -> total **843** | DRIFTED BY 1 (live tree grew one `agree` dir in ~10 h). Arithmetic of the stated tuple is internally sound (440+222+156+24=842; 842-222-24=596; 156/596=26.2%). |
| "**27** tests ... incl **8** mutation cells" (What-was-built table) | file now has **31** tests | STALE (see findings) |

## E. My OWN mutation matrices (26 cells, two runs, isolated copy)

Harness: full repo-shaped copy under the session scratchpad; scripts/qa,
handoff/ and .claude/masterplan.json symlinked READ-ONLY; the four housekeeping
scripts, the hook and the test module are real copies I mutate on disk there.
Every cell asserts a green CONTROL first and a byte-identical sha256 restore
after. The real repo tree was never written.

CONTROL: exit=0, 31 passed. sha256: backfill=1b4f88f0df3495f7,
hook=2278ca9910b0bd15, naming=2f426db901fe5746, quar=34ccb01ee6b26ff9,
verifier=f07a33170cfe717a, suite=f1a7a683a118a758. (backfill/hook/verifier
match the artifact's stated baselines EXACTLY.)

MATRIX 1
```
X1  naming   pending added to ARCHIVABLE_STATUSES        exit=1 KILLED (6 failed)
X2  naming   ARCHIVABLE_STATUSES reduced to {"done"}     exit=1 KILLED (1 failed)
X3  backfill ROLLING_KEEP emptied (the OLD half)         exit=0 SURVIVED
X4  backfill protection keyed on full path not basename  exit=1 KILLED (4 failed)
X5  backfill reference regex narrowed to handoff/current exit=0 SURVIVED
X6  naming   PREFIX tried first                          exit=0 SURVIVED
X7  naming   VARIANT_RE dropped from the resolver        exit=1 KILLED (2 failed)
X8  backfill c5 guard relocated one seam upstream        exit=1 KILLED (4 failed)
X9  backfill unknown-id WARN line deleted                exit=1 KILLED (2 failed)
X10 hook     ${base}_${short_sid}.md -> --                exit=1 KILLED (2 failed)
X11 suite    FIXTURE: c6 pending step marked done         exit=1 KILLED (1 failed)
X12 suite    FIXTURE: open-step file removed from verifier tree  exit=0 SURVIVED
X13 backfill unknown status defaults to "done"           exit=1 KILLED (3 failed)
X14 verifier is_archivable SHADOWED to return False      exit=1 KILLED (3 failed)
X15 backfill dead STEP_ID_RE constant deleted            exit=0 SURVIVED
```
MATRIX 2
```
Y1  naming   VARIANT tried before SUFFIX                 exit=1 KILLED (1 failed)
Y2  verifier open-step artifact becomes a violation      exit=1 KILLED (2 failed)
Y3  verifier dead STEP_ID_RE deleted                     exit=0 SURVIVED
Y4  suite    FIXTURE: c8 hook step pending not done      exit=1 KILLED (1 failed)
Y5  suite    FIXTURE: c13 hook step pending not done     exit=1 KILLED (1 failed)
Y6  naming   deferred/blocked/merged made archivable     exit=1 KILLED (1 failed)
Y7  naming   is_archivable -> `status is not None`       exit=1 KILLED (6 failed)
Y8  quar     quarantine tool neutered to a no-op         exit=1 KILLED (1 failed)
Y9  backfill mkdir moved INSIDE the dry-run branch       exit=1 KILLED (1 failed)
Y10 backfill dotfile guard removed                       exit=0 SURVIVED
Y11 backfill contradictory --dry-run/--execute guard gone exit=0 SURVIVED
```
restore verified: True (sha256 identical after every cell).
SURVIVORS (these two matrices): X3, X5, X6, X12, X15, Y3, Y10, Y11.

BEHAVIOURAL DIFFERENTIAL FOR EVERY SURVIVOR (a survivor is not a finding until
it is shown non-equivalent):
- X3 EQUIVALENT. Every ROLLING_KEEP member resolves to `None` under
  resolve_step_id (verified for all 8), so with the misc sweep deleted they are
  kept by the no-step-id branch anyway. ROLLING_KEEP is now redundant
  defence-in-depth, not a live guard.
- X6 EQUIVALENT. PREFIX_RE requires a leading sid; no suffix-named file can
  match it, so PREFIX-vs-SUFFIX order is unobservable. The ORDER claim the
  docstring actually makes (SUFFIX before VARIANT) IS guarded -- Y1 KILLED.
- X15 / Y3 EQUIVALENT. `STEP_ID_RE` is dead in BOTH scripts (grep: definition +
  comments only, no use). verify_handoff_layout.py:60-66 documents it as
  "RETIRED IN PLACE"; backfill:84 carries no such comment. Dead-but-deliberate.
- Y10 EQUIVALENT on any realistic input. The only dotfile in handoff/current/ is
  `.DS_Store`, which resolves to None and is kept regardless.
- X5 NON-EQUIVALENT IN PRINCIPLE, INERT TODAY. Narrowing the reference regex
  loses 80 of 381 protected basenames; **0** of those 80 are currently present
  in handoff/current/. No test plants a masterplan reference written without
  `handoff/current/`. Coverage residual, zero live exposure.
- Y11 NON-EQUIVALENT. Without the guard, `--dry-run --execute` together EXECUTES.
  The shipped code has the guard; no test covers it. Coverage residual beyond
  criterion 6's wording.
- X12 NON-EQUIVALENT (test-side). In
  test_c9_the_verifier_is_actually_EXECUTED..., the negative assertion
  `assert "research_brief_99.1.md" not in out.split("FAIL")[-1]` passes
  vacuously if the fixture file is absent -- it is not anchored to the file's
  presence. The PROPERTY it guards is nonetheless genuinely covered: Y2
  (open-step artifact treated as a violation) is KILLED with the fixture
  present. So: self-anchoring weakness in one assertion, not an unguarded
  property.

FINDING: none of the eight survivors indicates a defect in the SHIPPED
behaviour. Six are equivalent or inert; two are coverage residuals; one is a
non-self-anchoring assertion whose property is guarded elsewhere.

## F. Cycle-4 fix list -- closure check (I enumerated it from the critique myself)

1. mkdir below the dry-run return + remove the 3 dirs + re-measure census
   -> CLOSED. Verified on the LIVE tree (848 -> 848 across two bare runs); dirs
   absent; Y9 (mkdir put back inside the dry-run branch) KILLED.
2. 19 -> 20 -> CLOSED. Re-derives to exactly 20 under the script's own order.
3. Scope every "SURVIVORS: none" -> see section G.
4. Cells for ROLLING_KEEP_PREFIXES=() and _safe_target -> CLOSED; both tests
   present and both carry a CONTROL-NOT-GREEN guard plus a discrimination
   assertion.
5. Drive quarantine_misattributed_archives.py in a test -> CLOSED; my Y8
   (tool neutered to a no-op that still prints DRY RUN) is KILLED.
6. misc_moved tautology -> see section G. STEP_ID_RE comment -> NOT done
   (the "consider" half).
