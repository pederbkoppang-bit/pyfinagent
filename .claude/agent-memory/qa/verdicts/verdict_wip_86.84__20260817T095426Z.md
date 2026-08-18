STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-17T09:54:26Z

# Q/A write-first record -- step 86.84, cycle 5

## 0. Sequence evidence (reported, not applied)
- qa_wip.py --spawned-at 2026-08-17T09:54:26Z: attempt_number=5, prior_attempts=4,
  attempt_number_status=ok, is_lower_bound=true, source_present true, identity_checked true.
- verdict_history_86_21.py --step 86.84 --evidence-only: status=ok,
  "CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> CONDITIONAL" (4 rows).
- CROSS-CHECK: attempt_number 5 = 4 prior + this one; ledger count 4 == prior_attempts 4.
  LEDGER IS NOT STALE this cycle (it was on cycles 1-4; Main backfilled it this cycle with
  event/write-time separation disclosed in each row's note).
- Corroborated independently by 4 verbatim transcriptions in evaluator_critique_86.84.md
  (S0 c1, S5 c2, S6 c3, S9 c4) -- all CONDITIONAL.

## 1. Harness compliance (5 items) -- CLEAN
- research_brief_86.84.md mtime 2026-08-14T19:15:02 < contract_86.84.md 19:17:22. Gate PASSED
  per contract S"Research gate" (brief_status COMPLETE, 11 sources, 19 URLs, gate_passed true).
- experiment_results_86.84.md present (281 lines), Cycle-5 GENERATE section present.
- LOG-LAST ok: harness_log carries only result=IN-PROGRESS (Cycle 218) and
  result=EVIDENCE-ADDED (Cycle 198) for phase=86.84; masterplan status=pending. Not yet closed.
- NO VERDICT SHOPPING: evidence CHANGED. rail_turn_cap.py +71 lines / mutate_rail_turn_cap.py
  +184 lines uncommitted; artifact mtimes 11:49-11:53 local vs my spawn 11:54 local.

## 2. Deterministic
- `python3 scripts/qa/rail_turn_cap.py --verify` -> EXIT 0. "VERIFY: PASS -- controls green,
  turn-exhaustion claim holds." 585 records / 1283 spawns / 0 missing.
- `python3 scripts/qa/mutate_rail_turn_cap.py --verify` -> EXIT 0. cells=29 real survivors=0
  known/equivalent survivors (BY OUTCOME)=2 errors=0. CONTROL observed GREEN FIRST.
  md5 before==after on rail_turn_cap.py / qa.md / researcher.md (I md5'd independently
  OUTSIDE the harness: d371d2b6... / 4c9faa6d... / a9592ee0... unchanged).
- node --check qa-verdict.js = 0; research-gate.js = 0.
- node scripts/qa/verify_rail_retry.mjs -> exit 0, 38 passed 0 failed.
- node scripts/qa/verify_research_gate_workflow.mjs -> exit 0, 124 passed 0 failed.
- ruff F821,F401,F811 over a GIT-DERIVED scope (6 files incl. untracked; non-empty asserted,
  xargs -0 so no zsh word-split) -> "All checks passed!" exit 0.
- No frontend/** and no backend/** in THIS step's diff -> 1b/1d N/A for the step scope
  (peer-session files are dirty in the tree; see S6).

## 3. INDEPENDENT re-derivation from RAW transcripts (not their collect())
Walked ~/.claude/projects/*/*/workflows/*.json + subagents/workflows/<runId>/agent-*.jsonl,
counting distinct requestId over assistant lines. 585 records / 1283 spawns (matches).
Cross-slug check: 0 qa/researcher spawns exist under any OTHER project slug, so the
single-slug scope loses nothing.
- qa n=340: turn-value SET on spawns in FAILED runs = {30} EXACTLY (39 spawns).
- researcher n=107: set = {40} EXACTLY (9 spawns).
- BOUNDARY-FREE post-removal derivation (turns > the old pin is impossible under it):
  qa >30 = 34 spawns, 0 failed, 0 non-emitters, min 31 / p50 41 / max 60.
  researcher >40 = 0.
  -> matches the script's past_old_cap = 34 and 0 EXACTLY, derived without their
     session-birth boundary logic. Criterion-3's load-bearing number REPRODUCES.
- Uncapped-type spawns 836, 0 failed, 50 >30 turns.
  Reconciles with the script's 0/888 at-risk 0/88: 836 + 38 post-removal qa +
  14 post-removal researcher = 888; 50 + 34 + 4 = 88. Arithmetic checks out.
- at-cap: qa exactly-30 = 47 (39 failed, 38 non-emitters); researcher exactly-40 = 12
  (9 failed, 11 non-emitters). Script's 57 at-cap / 49 non-emitters = these minus the
  post-removal spawns that coincidentally sat at 30. Reconciles.

## 4. MY OWN MUTATION MATRIX over the cycle-5 floors (control green first, md5 identical)
Technique: source-text mutation of a temp copy loaded via importlib; REPO repointed at a
temp mirror of .claude/agents. Repo md5 d371d2b6... identical before and after.
CONTROL: verify_ok=True; prt = qa(n38,ne0,p50 40,p90 55,max 60,past 34),
researcher(n14,ne0,19,35,38,0).

  QA1  non-emitter floor neutered + injected non-emitter   -> GREEN  (floor IS the killer)
  QA1b unmutated + injected non-emitter (POSITIVE CONTROL) -> RED "POST-REMOVAL NON-EMITTER: 1"
  QA2  p50 floor neutered + _q zeroed                      -> GREEN  (p50 floor IS the killer of S3)
  QA3  cardinality floor neutered + role filter broken     -> GREEN  (card floor IS the killer of S4)
  QA4  _q percentiles REVERSED                             -> RED, monotonicity floor fires
  QA6  p50 reported as MAX                                 -> RED, monotonicity floor fires
  QA8  session_is_post_removal forced False                -> RED (C2, pre-existing guard)
  => kill attribution in the author's matrix is CORRECT, not mis-credited.
  => the monotonicity floor is NON-vacuous (2 killing mutations found) though NO shipped
     cell targets it.

  SURVIVORS OF `--verify` THAT THE SHIPPED MATRIX DOES NOT CONTAIN:
  QA5  past_old_cap comparison inverted (> -> <)  -> verify GREEN, publishes past=2 not 34.
       The author's S5 scores this KILLED via kill_mode ORACLE (mutant report != control
       report). That is a NON-EQUIVALENCE detection by the harness, NOT a guard: nothing in
       verify() asserts anything about past_old_cap, which is the number carrying
       criterion 3. Vacuity shape #11 (mis-attributed kill mechanism), disclosed in the
       cell's docstring but not in the "0 real survivors" headline.
  QA7  non_emitters narrowed to dropped-only + injection -> verify GREEN, non_em=0 with a
       planted non-emitter present. Equivalent-on-corpus (0 post-removal drops today), so
       only an injected truth distinguishes it -- exactly what S7 does for the
       hardcoded-0 shape, but this 3rd shape has no cell.
  QA10 role filter broken for qa ONLY  -> verify GREEN, qa row n=0 (researcher n=14 carries
       the total past MIN_POST_REMOVAL_SPAWNS=10).
  QA11 qa sample truncated to 1 spawn  -> verify GREEN, qa n=1 p50=47 past=1.
  QA12 post_removal set emptied for qa -> verify GREEN, qa n=0.
       => THE CARDINALITY FLOOR IS ON THE SUM ACROSS ROLES, and `if row["n"] <= 0: continue`
          actively skips an emptied role. The qa re-measurement -- the load-bearing sample
          for criterion 3 -- can go to n=0 or n=1 with the immutable command still green.

## 5. Criterion 5 (model attribution) -- re-derived at all three sources
- rail_drop_rate.py: header carries "THE MODEL SPLIT ABOVE IS CONFOUNDED -- DO NOT CITE IT
  AS A MODEL EFFECT"; RUNTIME output (I ran it) prints the CONFOUNDED caveat directly under
  the by-model table with the holding-model-fixed figures. Replacement, not a bare note.
- qa-verdict.js:609-619: "line used to read: `P(0 drops in 73 | true rate 11.2%) = 2e-4 --
  the model split SURVIVES.` The arithmetic was right and the conclusion was wrong" -- the
  cycle-2 finding is fixed; before/after both present, correction governs.
- research-gate.js:876-880: "THAT MODEL SPLIT IS CONFOUNDED AND THE MECHANISM IS NOW PROVEN",
  old UNPROVEN sentence quoted as history.
  => criterion 5 MET.

## 6. Scope
Step's own diff: scripts/qa/rail_turn_cap.py, scripts/qa/mutate_rail_turn_cap.py, the three
86.84 handoff artifacts, handoff/verdict_ledger.jsonl. No agent file, no gate, no threshold,
no verdict semantics. Peer-session files dirty in the tree (backend/api/sovereign_api.py,
frontend/src/{app/page.tsx,components/{HomeQuickActionsPanel,LatestTransactionsBox,
RecentReportsTable,RedLineMonitor}.tsx}, backend/services/experiments/perf_results.tsv) --
Main discloses these and commits to explicit pathspecs.

## 7. FINDINGS (running)
- V5-A (Contradiction, CAPPING-candidate): live_check_86.84.md -- the artifact the step's own
  `verification.live_check` names as required to carry "the mutation matrix" -- still carries
  the SUPERSEDED matrix, un-annotated:
    :236  "Matrix is now **22 cells, 0 real survivors**, 3 known/equivalent (M14, M6, M6b)"
    :280  "**22 cells** (15 at cycle-3; +6 pin-shape cells, +M21 ...)"
    :298  "M14  CAP_REMOVED_AT moved far future (2027)   SURVIVED (equivalent)"
    :313  "**M14** is behaviourally equivalent (the whole corpus already precedes any later
           boundary)"
  Measured: the matrix is 29 cells with 2 known survivors, and M14 KILLS ("C2 FAILED: 34
  capped spawns exceed their cap"). THE M14 EQUIVALENCE CLAIM IS THE EXACT CLAIM cycle-4
  violation 3 named as false; cycle 5 corrected it in mutate_rail_turn_cap.py and left the
  identical false statement standing in the graded artifact. No S1-S7 cell appears anywhere
  in live_check. live_check has no cycle-5 section (file ends at S10, cycle-4).
  This is the same class as cycle-4 violation 4 ("the correction REPLACES the stale claim
  rather than sitting beside it"), one instance further on.
- V5-B (Circular_Reasoning, WARN): the cardinality floor is on the SUM across roles and
  `if row["n"] <= 0: continue` silences an emptied role -- QA10/QA11/QA12 all green.
- V5-C (Overgeneralization, WARN): S5's KILLED is an ORACLE non-equivalence detection, not a
  guard; QA5 proves verify() does not catch past_old_cap inversion. The headline
  "0 real survivors" does not distinguish kill modes.
- V5-D (NOTE): monotonicity floor + "POST-REMOVAL DROP" floor + the `if not prt` branch have
  NO cell. I showed monotonicity is killable (QA4/QA6); the drop floor is redundant with
  three pre-existing guards; `if not prt` is effectively unreachable (prt is built from
  HISTORICAL_CAPS, always 2 rows).
- V5-E (NOTE): experiment_results:15 still describes mutate_rail_turn_cap.py as a
  "15-cell mutation matrix"; :30-35's correction note points at "Current, reproducing output:
  SCycle 4 above", but SCycle 4's block (qa n=36 p90=54 past=32) no longer reproduces
  (n=38 p90=55 past=34) -- a growing-corpus artifact, direction favourable, but the sentence
  makes a reproducibility claim that is false hours later.

- V5-F (Invalid_Precondition, NEW DEFECT IN THE CYCLE-5 GUARD -- executed proof):
  the non-emitter floor counts `killed` spawns as losses. `non_emitters = sum(1 for s in g
  if not s["structured_output"])` has NO status filter, while the SAME FILE carries
  `killed` as its own status (cycle-1 finding F4, rail_turn_cap.py collect(): "`not dropped`
  is not the same thing as `completed` ... never infer one from another") and models killed
  runs as the C3 NEGATIVE CONTROL. Injected a synthetic post-removal qa spawn,
  status=killed, 12 turns (nowhere near any cap), no StructuredOutput:
      verify_ok=FALSE -- "POST-REMOVAL NON-EMITTER: 1 uncapped qa spawn(s) never emitted
      StructuredOutput. The cap was the proven mechanism and it is gone, so this is a NEW
      loss mechanism ... Revisit the diagnosis; do NOT re-pin a cap."
  Same injection WITH structured_output=True -> green, so the discriminator is the SO flag
  alone. An operator abort of any future qa/researcher Workflow run therefore reddens THIS
  STEP'S IMMUTABLE VERIFICATION COMMAND with a message asserting a new loss mechanism.
  Fail-CLOSED (safe direction, cannot manufacture a PASS) but a real false positive, in an
  untested guard, re-committing the exact status-conflation class this step already fixed
  once in the same file. Named fix: exclude `killed` from the non-emitter population, or
  count and name it separately.

## 8. Criterion map (draft)
1 MET   -- re-derived independently from raw transcripts incl. a boundary-free derivation;
           population rule printed beside the ratios; the at-risk requalification is in the
           output itself.
2 MET   -- contract:31-56 answers all three, all NO, each cited (agent() opts; #20625 closed
           not planned; general-purpose re-expands the tool surface per qa-verdict.js:264-273;
           plus "absent maxTurns = No limit" from the agent-loop table).
3 MET   -- removal instead of a number, PLUS an uncensored sample now on disk and
           re-runnable: 34/38 post-removal qa spawns past the old cap of 30, max 60,
           which I reproduced by an independent rule.
4 MET   -- verify_rail_retry.mjs [F] 38/38 and verify_research_gate_workflow.mjs 124/124,
           both exit 0, run by me this cycle.
5 MET   -- all three sources re-derived above; runtime caveat verified by running it.
6 MET   -- removal has no number to outgrow; and the cycle-5 floors convert the
           re-measurement into a standing check with a named revisit rule.
7 MET   -- no gate touched; gate_passed still recomputed (124/124 incl. the drop path).
8 NOT MET -- two of its four clauses fail, both by execution:
           (a) "mutation-test EVERY new guard": monotonicity floor (killable, no cell),
               the non-emitter POPULATION (V5-F defect, no cell), per-role cardinality
               (n=0/n=1 green, no cell), past_old_cap (unguarded, only an ORACLE cell).
           (d) "report survivors rather than dropping them": live_check S4/S5 -- the
               artifact the step's verification.live_check names -- reports 22 cells,
               3 known survivors and "M14 SURVIVED (equivalent)", all false as of this
               cycle's own change; the code output's "0 real survivors" headline also
               does not separate the ORACLE non-equivalence kill from guard kills.
           (b) control green first and (c) byte-identical restore ARE satisfied; I
               observed both independently.

## 9. VERDICT RETURNED: FAIL (criterion 8)
Criteria 1-7 MET with independently reproduced evidence -- the diagnosis, the remedy
rationale and the uncensored re-measurement are correct and I re-derived the load-bearing
numbers from raw transcripts by a boundary-free rule. The step fails on criterion 8: an
untested new guard contains an executed false positive (V5-F) that reddens this step's own
immutable command on any future operator abort, and the survivor report in the artifact the
criterion's live_check names still states verbatim the M14 equivalence that cycle 4
adjudicated false and that cycle 5 corrected in code only.
Tree unchanged during evaluation: rail_turn_cap.py d371d2b6..., mutate a0cc4c02...,
artifact mtimes 11:53 local, HEAD 8000de69 at both ends.

COMPLETED: 2026-08-17T10:09:41Z
