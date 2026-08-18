STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-17T10:30:55Z

# Q/A write-first record -- step 86.84, cycle 6 respawn

Spawn context supplied by Main: attempt_number 7; prior spawn wf_80376bff-7ae died on
API 529 Overloaded mid-evaluation (NO_VERDICT); evidence claimed UNCHANGED since that spawn.
Operator authorization for respawn-after-drop cited.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command `python3 scripts/qa/rail_turn_cap.py --verify` + exit code;
   git status/diff scope; ruff lint gate; scoped tests
C. Independent re-derivation of the numeric claims + mutation of the new guards
D. Criterion-by-criterion MET / NOT MET

## Prior-attempt evidence (gathered, not a trigger)
qa_wip.py --spawned-at 2026-08-17T10:30:55Z: source_present=true, attempt_number=7
(status "ok", is_lower_bound true), prior_attempts=6, records_retained=7 (gauge).
verdict_history_86_21.py --step 86.84 --evidence-only: status "ok", 6 verdicts:
CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> FAIL -> NO_VERDICT.
CROSS-CHECK: attempt_number 7 = 6 prior + this spawn; ledger has 6 rows -> ledger is
CONSISTENT, not stale. NO_VERDICT row carried through as-is.

## FINDING 1 (BLOCKING, deterministic) -- the immutable command EXITS 1
Command: `python3 scripts/qa/rail_turn_cap.py --verify`   EXIT=1
Tail verbatim:
  VERIFY: FAIL
    - POST-REMOVAL NON-EMITTER: 2 uncapped qa spawn(s) never emitted StructuredOutput.
      The cap was the proven mechanism and it is gone, so this is a NEW loss mechanism
      (or the 86.81 retry absorbing losses again). Revisit the diagnosis; do NOT re-pin a cap.
Post-removal distribution rows read: qa n=42 dropped=0 non-emitters=2 p50=40 p90=54 max=60
>old-cap(30)=37 ; researcher n=14 dropped=0 non-emitters=0 p50=19 p90=35 max=38.
Corpus now: 589 run records / 1287 agent spawns (evidence claimed 572/1325 at audit time).
NEXT: determine whether this is reproducible, whether it is HEAD state, and when/why it flipped.

## FINDING 1 detail -- the two post-removal non-emitters IDENTIFIED (re-derived)
python3 -c "import rail_turn_cap as R; ..." over R.collect():
  post-removal spawns = 71; non-emitters = 2, BOTH agentType=qa, workflow=qa-verdict,
  session_started_at 2026-08-17T08:50:12Z (uncapped session), model claude-opus-5[1m]:
    wf_2fafe515-6a2  status=completed  turns=38  toolCalls=56  tokens=204,917  durationMs=773,316
                     record BIRTH 2026-08-17T12:29:57 local (=10:29:57Z)
    wf_80376bff-7ae  status=completed  turns=10  toolCalls=16  tokens=150,657  durationMs=492,140
                     record BIRTH 2026-08-17T12:29:44 local (=10:29:44Z)
  wf_80376bff-7ae is the spawn Main's prompt names as the 529-Overloaded death.
  wf_2fafe515-6a2 is a SECOND loss Main's prompt does not mention, and it ran 38 turns --
  PAST the old cap of 30, i.e. it could not have been a cap exhaustion.
Reproducibility: exit 1 on BOTH interpreters (system python3 3.14.4 and .venv python 3.14.4),
re-run twice, identical message. NOT transient.
Meaning: the step's own remediation guard is firing. Its shipped text instructs
"Revisit the diagnosis; do NOT re-pin a cap." The immutable command is the step's gate and
it is RED at evaluation time.

## FINDING 2 (BLOCKING) -- ROOT CAUSE of the red: an Invalid_Precondition in the shipped guard
Both non-emitters carry, IN THE SAME workflowProgress entry the collector already reads:
  error = "API Error: 529 Overloaded. This is a server-side issue, usually temporary ..."
and run-level `result: None`, `status: "completed"`, ONE workflow_agent entry each (no retry).
So neither is a turn exhaustion (10 and 38 turns) and neither is a "new loss mechanism" in the
sense the guard's message asserts -- they are a transient server-side outage during THIS step's
own evaluation.
collect() at scripts/qa/rail_turn_cap.py:386-420 reads entry.get("agentId"/"agentType"/"model"/
"toolCalls"/"tokens") and NEVER entry.get("error") (grep confirms: no "error" key read anywhere
in the file's entry handling).
This is the SAME SHAPE as the cycle-5 Q/A's Invalid_Precondition that cycle 6 closed for
`killed`: the code comment at :624-631 says non_emitters must count "only spawns that ran to
completion WITHOUT emitting -- the one shape that genuinely signals a new loss mechanism". A
spawn whose agent entry carries an explicit server-side API error did NOT have the chance to
emit, exactly like a killed run. The instance was fixed; the CLASS was not.
CORPUS-WIDE CENSUS of the agent-entry error field (all 589 records):
  4 (completed, Explore, no_structured_output)
  2 (completed, qa, 529)                  <-- the two that redden the gate
  1 (completed, general-purpose, "You've hit your weekly limit ...")
  1 (completed, general-purpose, "You've hit your session limit ...")
  1 (completed, general-purpose, "You're out of usage credits ...")
=> 5 of the 9 errored spawns are ENVIRONMENTAL (529 / quota), i.e. named non-mechanism causes the
non-emitter counter cannot distinguish. The class is pre-existing in the corpus, not hypothetical.
CONSEQUENCE: the corpus is append-only, so these two records are PERMANENT. `--verify` will stay
RED on every future run until the guard excludes API-errored spawns. The step's immutable command
is not transiently red -- it is red from here on.

## FINDING 3 (BLOCKING) -- criterion 8's control is RED, so the matrix scored ZERO cells
$ python3 scripts/qa/mutate_rail_turn_cap.py --verify      MATRIX_EXIT=1
  CONTROL (unmutated) must be GREEN before any mutant is scored.
    control verify_ok=False  live_caps={'qa': None, 'researcher': None}
    CONTROL IS RED -- the matrix is meaningless. Fix the subject first.
md5 before == md5 after (0f4fc394882602ca4dcb4530a7cb9d99 / 9eec183d33d1f4fac8cd30cf5bfa0dac):
tree unchanged, the harness aborted before mutating. Criterion 8 requires "the control observed
GREEN first"; it is observed RED, and the claimed cells=33 / 0 survivors DOES NOT REPRODUCE.

## FINDING 4 -- two "verbatim" claims do not reproduce
experiment_results_86.84.md:141-148 (cycle-5 block):
  "qa  n= 36  dropped=0  non-emitters=0 ... Zero drops, zero non-emitters since removal."
  MEASURED NOW: qa n=42 dropped=0 NON-EMITTERS=2, p90=54, max=60.
experiment_results_86.84.md:307 + live_check_86.84.md:225-226 and :551:
  "`--verify` exit 0 both interpreters".  MEASURED NOW: exit 1 on both.
Both were plausibly true at capture (artifacts mtime 10:21:03Z; the two 529 records were
born 10:29:44Z / 10:29:57Z). The corpus-grows-daily caveat the cycle-6 summary cites was
applied to the MATRIX CELL COUNT and the reproducing-output pointer, not to the verify exit
code or to the unqualified sentence "Zero drops, zero non-emitters since removal".

## HARNESS-COMPLIANCE AUDIT (5 items) -- CLEAN
1. research-gate-before-contract: research_brief_86.84.md brief_status COMPLETE,
   external_sources_read_in_full 11, urls_collected 19, recency_scan_performed true,
   gate_passed true. Git first-appearance ordering: brief 2026-08-14 19:07:46 ->
   contract 19:18:58 -> cap removal 85127353 19:37:50 -> experiment_results 20:00:52. OK.
2. contract-before-generate: OK per the ordering above (the diagnosis script c1797888 at
   19:05:03 predates the brief by 3 min, but it is the DIAGNOSIS, and the remedy postdates
   the contract).
3. experiment_results present: yes (309 lines, cycles 4-6).
4. log-last: masterplan 86.84 status=pending; harness_log has only IN-PROGRESS /
   EVIDENCE-ADDED rows for 86.84, no result= row. OK.
5. no-verdict-shopping: prior spawn was a 529 NO_VERDICT (respawn after a drop is the
   documented recovery), and vs the cycle-5 FAIL the code AND all three artifacts changed
   (rail_turn_cap.py 10:17:52Z, artifacts 10:20-10:21Z). OK.

## OTHER DETERMINISTIC CHECKS
ruff --select F821,F401,F811 over the step's committed .py scope (derived from
`git show --name-only d69da099 | grep '\.py$'` -> mutate_rail_turn_cap.py, rail_turn_cap.py):
  "All checks passed!"  RUFF_EXIT=0.  AST OK on both.
node --check qa-verdict.js OK; node --check research-gate.js OK.
node scripts/qa/verify_rail_retry.mjs           EXIT=0   ALL GREEN: 38 passed, 0 failed
node scripts/qa/verify_research_gate_workflow.mjs EXIT=0 ALL GREEN: 124 passed, 0 failed
[F] is a REAL drive, not a re-implementation: loadRetry() slices the shipped
agentRetryingDrops body out of qa-verdict.js and executes it with an injected faulting
agent (F1 out==='UNSET' && threw!==null; F2 rethrows the original). F3/F4 are source scans,
but enforceGate is separately driven behaviourally in the 124-check verifier.
No UI claims (1c N/A). No backend/** change from this step (1d N/A; sovereign_api.py is the
peer session's dirty file).

## CRITERION-BY-CRITERION
1 MET   -- diagnosis re-derived by me: 589 records / 1287 spawns; qa 39/344 dropped, observed
           turn set on drops = [30]; researcher 9/107, set = [40]; uncapped 0/892 with max 93;
           C1/C2/C3 + detector control all green in the run output. POPULATION RULE and the
           percentile rule are printed beside the ratios. Disagreements with audit_basis are
           reported in-file (49 vs 50 at-cap non-emitters; 347/347; 0/50 AT-RISK requalified).
2 MET   -- contract:31-49 answers all three, including the NOs, each cited: no per-call turn
           budget in agent() opts; forcing the schema call closed as not planned (#20625);
           and these roles must NOT move to the uncapped default subagent (general-purpose
           re-expands Edit/Write/Bash + the deferred MCP surface phase-75.20 pinned away).
3 MET   -- remedy is REMOVAL, not a new number; right-censoring argued head-on
           (contract:51-56, live_check:141-149) AND an uncensored sample was produced and is
           re-runnable: qa n=42 p50=40 p90=54 max=60, 37 past the old cap of 30.
4 MET   -- verify_rail_retry.mjs [F] F1-F4 re-run green by me (exhaustion yields NO value,
           rethrows the original, gate_passed still recomputed, retry assigns no verdict).
5 MET   -- all three sources corrected by REPLACEMENT with the before quoted:
           rail_drop_rate.py:20 "THE CAUSE IS NOW KNOWN, AND THE MODEL SPLIT ABOVE IS
           CONFOUNDED -- DO NOT CITE"; qa-verdict.js "THE 'MODEL SPLIT' IS AN ARTEFACT --
           RETRACTED ... This block used to read: ..." and "This block used to say it was
           UNPROVEN; that is SUPERSEDED"; research-gate.js "THAT MODEL SPLIT IS CONFOUNDED
           AND THE MECHANISM IS NOW PROVEN ... This block used to end ...".
6 WARN  -- addressed in prose (removal leaves no number for the workload to outgrow; the
           re-measurement is re-runnable on every invocation). BUT the named trigger is the
           POST-REMOVAL NON-EMITTER assertion, and FINDING 2 shows it cannot distinguish
           "the workload outgrew the mechanism" from "the API was overloaded" -- the exact
           discrimination this criterion asks the trigger to make. Its supporting sentence
           ("Zero drops, zero non-emitters since removal") no longer reproduces.
7 MET   -- no gate loosened; enforceGate behaviourally recomputed 124/124; [F] F1-F4;
           workflow diffs comment-only; both files node --check green.
8 NOT MET -- the control is observed RED and the matrix scored ZERO cells. The criterion's
           literal precondition ("with the control observed GREEN first") is unmet, and the
           claimed cells=33 / real survivors=0 does not reproduce.

## NOTES (non-blocking)
- HEAD MOVED MID-EVALUATION: d69da099 (this step's cycle 4-6 work), 192ef652, 65e38c3b,
  cadab378 landed while I graded. md5s of rail_turn_cap.py / mutate_rail_turn_cap.py are
  byte-identical before and after, so the graded evidence did NOT change -- but committing a
  step's tree during its own EVALUATE is a freeze-the-tree breach.
- contract:56 says the uncapped types "reach 63 and 56 turns"; live_check:148 says "63 and
  93 turns". Measured: Explore 56, general-purpose 63, None 93. Two artifacts give different
  pairs for the same "only uncensored evidence" sentence.
- qa.md:632 still tells the agent "your real bound is maxTurns" -- a bound this step removed.
  The audit_basis itself flags that bullet as pushing the agent INTO the cap.

## VERDICT REASONING
Deterministic first: the step's immutable verification command exits 1 at evaluation time,
reproducibly, on both interpreters, at the current HEAD; and the mutation harness that is
criterion 8's entire evidence refuses to score a single cell because its own control is red.
The root cause is a one-key Invalid_Precondition in the shipped guard (entry.get("error") is
never read), of the same class the previous cycle closed for `killed`. Because the run-record
corpus is append-only, the red is permanent, not transient. FAIL.

COMPLETED: 2026-08-17T10:41:53Z

## Findings (appended as established)

