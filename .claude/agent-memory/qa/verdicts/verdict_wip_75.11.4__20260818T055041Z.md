STATUS: COMPLETE -- write-first record, still NOT a verdict
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
4. log-last: `grep -cF 'phase=75.11.4' handoff/harness_log.md` -> **1** line, and
   it is `## Cycle 1244 -- 2026-08-17 -- phase=75.11.4 result=PARKED (budget
   exhausted)`. That is a disposition written by the attempt gate at last
   night's denial, NOT a Q/A verdict, and the escalation file itself says "THIS
   IS NOT A PASS AND NOT A FAIL". No PASS/CONDITIONAL/FAIL row exists for this
   step and masterplan status is still "pending". PASS -- recorded precisely
   rather than as the "0 lines" I first expected.
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


## G. Findings (all EVIDENCE-class; none PRODUCT-class)

W1. live_check_75.11.4.md:18 -- the file maintains an explicit supersession
    chain (19 passed -> superseded; 22 passed -> superseded) and states in bold
    "**The live reading is section 16's: 27 passed, with all ten mutation cells
    killed (section 14).**" Cycle 5 added SS18c/18e reporting **31 passed** and did
    NOT extend the chain, so the file's own navigational sentence points an
    auditor at a superseded count.
W2. experiment_results_75.11.4.md:16 -- the "What was built" table still reads
    "**27** tests ... incl **8** mutation cells (cycle 2 added M-INV/M3/M4;
    cycle 3 added N5a/N5b/N14/N15 ...)". Actual: 31 tests; cycle 5 added Q3/Q5/
    DRYMK. Present-tense delivery summary that does not reproduce.
    (The  figures INSIDE fenced blocks are correctly labelled
    "cycle-3" captures -- those are legitimate dated records, not findings.)
N1. "total=842 agree=440" re-derives today as total=843 agree=441 (one new
     dir in ~10 h). Load-bearing figure (mismatch=156) is EXACT.
    Present-tense census without a capture stamp.
N2. "[protected] KEEP = 20" is a live-tree gauge; it re-derives to exactly 20
    today but will drift as handoff/current/ grows.
N3. Coverage residuals my matrix found: no test covers a masterplan reference
    written WITHOUT  (X5), and none covers the
     contradiction guard (Y11). Shipped code is correct in
    both cases; these are hypothetical regressions, not live defects.
N4. test_c9_the_verifier_is_actually_EXECUTED's negative assertion
     is not
    self-anchoring: it passes if the fixture file is absent (X12 SURVIVED). The
    PROPERTY is guarded -- Y2 KILLED.
N5. STEP_ID_RE is dead in BOTH housekeeping scripts. verify_handoff_layout.py:
    60-66 documents it "RETIRED IN PLACE"; backfill:84 has no such comment --
    the cycle-4 "consider" item, not done.
N6. ROLLING_KEEP (the .md half) is now behaviourally inert; all 8 members
    resolve to None and are kept by the no-step-id branch regardless.

OPERATIONAL WARNING (not a criterion issue, but Main must not miss it):
.claude/hooks/auto-commit-and-push.sh:360 stages with "git add -A" (its own
comment at :351 says it "will also stage a PEER session's" work). If Main flips
75.11.4 to done right now, **14 out-of-scope entries** -- backend/api/
sovereign_api.py, backend/services/autonomous_loop.py, 10 frontend files,
.archive-baseline.json -- ride into a commit subject-named for this step.
live_check SS18f says "the commit uses an explicit pathspec and never git add -A";
that was true of the hand-made 2e9597bd, and is NOT true of the hook.

## H. Criterion-by-criterion (each with a guard I made fail MYSELF)

 1 MET  X1 X2 X7 X9 X13 Y6 Y7 all KILLED; live run prints 4 unknown-id WARNs
 2 MET  test_c2 both directions in one run; fixture cell X11 KILLED
 3 MET  M1 + test_c3_fixture_is_load_bearing; my X1 KILLED
 4 MET  test_c4 + LIVE double bare run 848->848, exit 0
 5 MET  test_c5 plants the reference; X4 + X8 KILLED
 6 MET  real subprocess drives __main__; M-INV + my Y9 KILLED; live tree unchanged
 7 MET  test_c7_m5/m6/m2; X4/X8 KILLED
 8 MET  real hook driven via CLAUDE_PROJECT_DIR (documented deviation from
        "flip a scratch step", reason: a live flip fires git add -A); X10 hook
        mutation KILLED and Y4 fixture cell KILLED, so the hook truly executes
 9 MET  X10 Y1 Y2 X14 all KILLED (verifier EXECUTED, not source-scanned)
10 MET  whole-tree checker re-run by me; recall+precision controls True
11 MET  content-equality asserts on contract.md + live_check.md; Y4 KILLED
12 MET  156 markers == 156 mismatches, re-derived from classify() today; Y8 KILLED
13 MET  test_c13 discriminates (foreign refused, declaring admitted); Y5 KILLED

## I. Disposition of my policy uncertainty

W1/W2 are the same CLASS the step's cycle 5 was remediating, reproduced by the
act of remediating -- which argued for capping. Resolved against the operator's
dated standing instruction (auto-memory product-fix-vs-evidence-churn,
2026-08-17): classify PRODUCT vs EVIDENCE; "artifact prose staleness" is named
there as EVIDENCE-class and does not buy a re-evaluation cycle; fix in place and
queue as a residual. That doctrine is attempt-INDEPENDENT, so this reasoning
would return the same verdict at attempt 1.  is
explicitly preserved by that memo and was spent here: 26 cells, a behavioural
differential for every survivor, and one probe of my own that I caught and
corrected (the 24-vs-20 protected count).

Process note for the operator, stated rather than buried: the same memo says
"never seek extensions for evidence-only disputes", and this extension WAS
sought after an evidence-only CONDITIONAL.

## J. Safety of my own work
HEAD moved 16b57f81 -> 6a65b5d6 mid-evaluation (86.116 cycle 3);  touches NOTHING in this step's scope. Post-run
sha256 (unchanged, and matching the artifact's stated baselines):
backfill=1b4f88f0df3495f7 verifier=f07a33170cfe717a naming=2f426db901fe5746
quar=34ccb01ee6b26ff9 hook=2278ca9910b0bd15 suite=f1a7a683a118a758.
grep MUTANT = 0 in all six. All mutation ran on a scratchpad copy; the real
tree was never written. handoff/harness_log.md carries one prior row for this
step --  --
which is a disposition from the attempt gate, not a Q/A verdict; masterplan
status is still , so log-last holds.

VERDICT RETURNED: PASS (ok=true). Findings W1/W2/N1-N6 recorded as
EVIDENCE-class residuals, not blockers.

COMPLETED: 2026-08-18T06:31:44Z
