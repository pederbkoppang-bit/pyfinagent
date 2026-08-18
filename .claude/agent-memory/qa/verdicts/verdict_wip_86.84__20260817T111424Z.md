STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-17T11:14:24Z

# Q/A write-first record -- step 86.84, cycle 9 (attempt_number=9 per qa_wip)

Launch: Workflow rail (qa-verdict.js). Read qa.md from disk in full at t0.

## Attempt / sequence evidence (gathered, NOT applied)
- `qa_wip.py 86.84 --spawned-at 2026-08-17T11:14:24Z`: source_present=true,
  attempt_number=9, prior_attempts=8, attempt_number_status="ok",
  attempt_number_is_lower_bound=true, records_pruned_known=null,
  records_retained=9 (GAUGE, not used as a counter).
- `verdict_history_86_21.py --step 86.84 --evidence-only`: status=ok,
  "8 verdict(s)", sequence = CONDITIONAL -> CONDITIONAL -> CONDITIONAL ->
  CONDITIONAL -> FAIL -> NO_VERDICT -> FAIL -> FAIL. NO_VERDICT carried as-is.
- Cross-check: prior_attempts (8) == ledger rows (8). Ledger NOT stale here.

## A. Harness compliance (5 items) -- CLEAN
- Research gate: brief_status COMPLETE, 11 sources read in full, 19 URLs,
  recency_scan true, gate_passed true.
- Order (mtime): brief 2026-08-14T19:15 < contract 08-17T12:20 <
  rail_turn_cap.py 13:11 < experiment_results/live_check 13:13 (local).
- experiment_results present (389 lines), live_check present (669).
- Log-last: only `result=IN-PROGRESS` and `result=EVIDENCE-ADDED` rows for
  phase=86.84; masterplan status still "pending". No premature close.
- No verdict-shopping: evidence CHANGED vs cycle 8 -- +373 lines across the
  three artifacts, +105 in rail_turn_cap.py, +41 in the matrix harness,
  3 new cells (S12/S13/S14).

## B. Deterministic
- IMMUTABLE `python3 scripts/qa/rail_turn_cap.py --verify` -> EXIT 0.
  "VERIFY: PASS -- controls green, turn-exhaustion claim holds."
  593 records / 1291 spawns / 0 missing transcripts.
  qa n=46 dropped=0 non-emitters=0 killed=0 errored=0 erased=2(non-emit 2)
     p50=41 p90=55 max=62 >old-cap(30)=42
  researcher n=14 ... erased=0(non-emit 0) p50=19 p90=35 max=38 >old-cap(40)=0
- ruff F821/F401/F811 over a DERIVED 8-file scope (git diff --name-only HEAD
  -- '*.py', non-empty asserted, xargs not an unquoted var): "All checks
  passed!", exit 0.
- npx tsc --noEmit exit 0; npx eslint src -> 0 errors, 55 warnings (gate is
  errors-only). Run because the PHASE diff touches .claude/agents/qa.md
  (committed 85127353); repo-wide `eslint .` is known-red from build-output
  dirs and the tree carries peer-session UI files, so `src` was the scope.
- No backend/** in 86.84's own diff, so gate 1d does not apply to this step.
- md5 before AND after all my work, identical (I mutated nothing in the tree):
  rail_turn_cap.py 8a01f10a5b23d9e9d957da2c715a3a04
  qa.md 4c9faa6d7eb14aba70eea2fc7f804727
  researcher.md a9592ee0950e55d24fc3e1bb65d5c26f
  mutate_rail_turn_cap.py a805ee2c8a536f6f48546cf6ffa3bf65
  qa-verdict.js 8ce02bfd12351dff0c3297d03c6b529b
  research-gate.js a9c1d46c5dc7cda877c6fe5a9b97ef69
  rail_drop_rate.py 838a780bfcbb5e57288b743b88b5decb

## CRITERION 1 -- re-derived BY ME, independently
My own census (not the author's script), all four runs:
  wf_2fafe515-6a2 birth 10:29:57Z mtime 10:48:05Z entries=1 transcripts=2
                  ORPHAN agent-a8a91688009ccc502.jsonl
  wf_80376bff-7ae birth 10:29:44Z mtime 10:44:12Z entries=1 transcripts=2
                  ORPHAN agent-a5fb7a57499f4fb8a.jsonl
  wf_078f4125-57a birth==mtime 11:34:45Z entries=2 transcripts=2 ORPHANS=[]
  wf_a6ea31e7-9b9 birth==mtime 2026-08-13T21:10:02Z entries=2 tr=2 ORPHANS=[]
(My `stat -f %SB` prints LOCAL; converted CEST->UTC. The artifact's snippet
 uses fromtimestamp(...,timezone.utc) and matches byte-for-byte.)
=> RE-DISPATCH REPLACES / RETRY APPENDS: REPRODUCED, incl. the second control
   run the artifact's snippet omits. Both orphans independently corroborated
   as 529-killed: "API Error: 529 Overloaded" present 9x and 3x respectively.
FALSE PREMISE REPLACED, not annotated, at all three sites:
  - rail_turn_cap.py:395-406 -- the only surviving occurrence of
    "permanently red (the corpus is append-only)" is a QUOTE inside the
    retraction. `git show HEAD:scripts/qa/rail_turn_cap.py | grep` -> 0 hits,
    so no stale copy is committed anywhere.
  - experiment_results_86.84.md:323-329 -- "(Cycle-9 REPLACEMENT of this
    paragraph's causal frame ...)".
  - live_check_86.84.md sec.12 -- "(cycle-9 REPLACEMENT of three sentences
    that stood here and measured FALSE ...)", old text quoted, claim gone.

## CRITERIA 2,3,5,7 -- verified
- C2: sec.4 items 1-5 answer all three questions, including the NOs, with
  citations (#20625 closed-not-planned for forcing the schema call; #41143 for
  maxTurns not enforced on the Agent-tool path; no per-call budget in
  Workflow agent() opts; agentType stays 'qa' with the permission-surface
  reason for not moving to general-purpose).
- C3: remedy is REMOVAL, not raising; the UNCENSORED sample exists and I
  reproduced it (qa n=46 p50/p90/max 41/55/62, 42 of 46 above the retired 30).
- C5: all three sources corrected AND the stale text quoted-then-replaced --
  rail_drop_rate.py docstring ("THE CAUSE IS NOW KNOWN, AND THE MODEL SPLIT
  ABOVE IS CONFOUNDED") plus a RUNTIME caveat printed under the by-model table
  (:236-247, so a reader of the OUTPUT cannot miss it); qa-verdict.js:608-627
  ("This line used to read ...") and :628-640 ("This block used to say it was
  UNPROVEN; that is SUPERSEDED"); research-gate.js:879-889 ("This block used
  to end ..."). rail_turn_cap.py also prints "MODEL x agentType ... showing
  the confound explicitly".
- C7: DEMONSTRATED by execution, not asserted --
  `node scripts/qa/verify_research_gate_workflow.mjs` 124/124 exit 0,
  including section [3] which CALLS enforceGate on null/undefined/{}/'oops'/[]
  and asserts gate_passed===false on each, plus brief-on-disk recomputation
  (missing brief, empty brief, source ABSENT from brief, over-claim) and the
  self-report override in BOTH directions.

## CRITERION 4 -- executed, behavioural
`node scripts/qa/verify_rail_retry.mjs` 38/38 exit 0. F1/F2 are EXECUTED, not
scanned: an exhaustion throw yields out==='UNSET' (no value at all, never an
ok/PASS-shaped object) and rethrows the ORIGINAL
"without calling StructuredOutput" error. F3/F4 are regexes but they are
supplementary -- the behavioural coverage of "never a gate_passed" is the
enforceGate execution suite above, so this is not sole-coverage source-scan.

## CRITERION 8 -- matrix re-executed BY ME
`python3 scripts/qa/mutate_rail_turn_cap.py --verify`:
  kills by mode {'VERIFY':28,'ORACLE':2,'INJECTED_TRUTH':2,'MUST_STAY_GREEN':2}
  cells=36  real survivors=0  known/equivalent (BY OUTCOME)=2  errors=0
  Arithmetic reconciles: 34 kills + 2 known survivors (M6, M6b -- agent file
  deleted; disclosed as KNOWN GAP with reasoning, reported not dropped) = 36.
  Control asserted GREEN first; byte-identical restore self-reported and
  INDEPENDENTLY md5-confirmed by me before and after.
  S14 (orphan sweep neutered) KILLED[ORACLE] against the real present signal.
  Diff review of the harness shows no weakening: S2/S7 anchors were rebuilt to
  match the new source (run_source_cell raises on a missing anchor), and the
  three new cells are additions.

## FINDING (WARN) -- the new instrument's own silent-zero is undisclosed
The orphan sweep classifies role from LITERALS in the transcript's first user
message: "IMMUTABLE SUCCESS CRITERIA" -> qa, "OBJECTIVE:" -> researcher, else
role=None; role=None orphans are collected and then dropped by
`erased_for_role = [... if e.get("role") == role ...]`, so they appear in NO
row. Those literals are emitted by `.claude/workflows/qa-verdict.js:365` and
`.claude/workflows/research-gate.js:439`, and NOTHING pins that coupling.
MY EXECUTED MUTATIONS (temp copies via importlib; repo never written):
  CONTROL_unmutated     erased qa=(2,2)  verify_ok=True  problems=[]
  Q1_role_marker_drift  erased qa=(0,0)  verify_ok=True  problems=[]
  Q2_qa_branch_dead     erased qa=(0,0)  verify_ok=True  problems=[]
Corpus census: 44 orphans -- qa/post-removal 2, qa/pre-removal 1, role=None 41
(all pre-removal; 40 from one fan-out run wf_03d6e7c4-fda + 1 from
wf_b9bbd4fd-978). The channel is currently EMPTY of post-removal members, so
the published number is correct today.
WHY IT IS A FINDING RATHER THAN A GRIPE: criterion 6's own disjunction is
"state what makes the mechanism resistant OR say plainly that it is not".
S14's cell text DOES disclose the sibling fragility (the discriminating signal
lives in a rotating corpus and ages out in ~30d, applying the M14 lesson
prospectively) -- so the disclosure discipline was available and was applied
to the test's fragility but not to the production classifier's. grep of both
artifacts and the code comment for the marker literal: 0 hits.
NAMED FIX (cheap, one of): assert in verify() that the two marker literals are
still present in the two workflow files; or add a matrix cell that mutates the
marker; or one sentence of disclosure beside the counter.

## NOTE (not verdict-bearing)
S12 injects an ERRORED spawn but still prints the MUST_STAY_GREEN message
"no false positive on an operator abort" (the killed-cell wording). Flagged by
the cycle-8 Q/A, still present in my run. Cosmetic label/message mismatch;
the cell's outcome and mode are correct.

## Criterion map
1 MET  2 MET  3 MET  4 MET  5 MET  6 MET-IN-SUBSTANCE with the WARN above
7 MET  8 MET.  Harness compliance clean. No unintended production change
attributable to 86.84 (its own diff = rail_turn_cap.py, mutate_rail_turn_cap.py
and the three artifacts; the rest of the dirty tree belongs to 86.71 / 86.85 /
the peer UI session, as disclosed in the spawn args).

## Disagreement with the prior verdict
The cycle-8 FAIL's two NOT-MET findings are both DISCHARGED on this evidence
and I re-derived each myself rather than adopting it: the re-dispatch mechanism
reproduces on four runs, and the erased-attempt channel is now visible, named
and correctly excluded from the distribution. I do not carry cycle 8's verdict
forward; the evidence changed and the numbers reproduce.

COMPLETED: 2026-08-17T11:23:23Z
