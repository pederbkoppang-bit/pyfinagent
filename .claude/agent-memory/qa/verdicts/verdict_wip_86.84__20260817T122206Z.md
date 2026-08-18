STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-17T12:22:06Z

# Cycle 11 Q/A of step 86.84 (turn-cap removal)

## Prior-attempt / prior-verdict EVIDENCE (gathered, not a trigger)
- `qa_wip.py 86.84 --spawned-at 2026-08-17T12:22:06Z`: source_present=true,
  attempt_number=11 (status "ok", is_lower_bound=true), prior_attempts=10,
  records_retained=11 (GAUGE, not a counter), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.84 --evidence-only`: status=ok,
  "10 verdict(s) from the ledger",
  sequence = CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> FAIL ->
  NO_VERDICT -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL (NO_VERDICT carried
  through as-is).
- CROSS-CHECK: prior_attempts (10) == ledger rows (10); attempt_number (11)
  exceeds it by exactly this in-flight spawn => ledger NOT stale for this step.

## A. Harness compliance -- CLEAN
- research_brief_86.84.md envelope read directly: brief_status COMPLETE,
  external_sources_read_in_full 11, snippet_only 8, urls_collected 19,
  recency_scan_performed true, tier complex, gate_passed TRUE.
- mtime order (stat -f%Sm is LOCAL = UTC+2): brief 2026-08-14 19:15 <
  contract 2026-08-17 12:20 < experiment_results/live_check 14:17.
  My WRITTEN stamp 12:22:06Z = 14:22:06 local, so all predate this spawn.
- Cycle-11 GENERATE commit c90910ef @ 2026-08-17T14:18:10+02:00 (12:18:10Z),
  4 min before spawn: rail_turn_cap.py (+51/-4), mutate_rail_turn_cap.py (+70),
  the three artifacts, verdict_ledger.jsonl, attempt_budget_audit.jsonl. IN SCOPE.
- LOG-LAST intact: masterplan 86.84 status="pending"; harness_log carries only
  `result=IN-PROGRESS` (Cycle 218) and `result=EVIDENCE-ADDED` (Cycle 198).
- NOT verdict-shopping: evidence changed vs cycle 10 (the commit above).
- IMMUTABLE CRITERIA NEVER AMENDED: md5 of 86.84 success_criteria is
  c3a939b68f1579f5e1f3ef8e983a789c (8 items) IDENTICAL across all 26
  masterplan commits since the step was filed; earlier 14 = ABSENT.

## B. Deterministic (all exits taken unpiped)
- IMMUTABLE `python3 scripts/qa/rail_turn_cap.py --verify` => VERIFY_EXIT=0.
  "VERIFY: PASS -- controls green, turn-exhaustion claim holds."
  New default-report line reproduced verbatim:
  "unclassified orphans: 41 (post-removal 0) -- role=None erased attempts,
   visible in the DEFAULT report, not only --json"
  My re-derivation: 601 records / 1299 spawns / 0 missing;
  qa cap=30 n=356 drop=39 @cap=39 >cap=0; researcher cap=40 n=107 drop=9 @cap=9;
  drop turn-sets {30} and {40}; C1 1299 turns>0 & 0 zero-with-assistant-lines;
  C2 0 exceed; detector 1231/1241 vs 1/48; C3 killed [1,1,2,2,2,3,4,5,6,16] 0@cap;
  uncapped 0/904 requalified to 0/104 AT RISK vs 12.2% on capped.
  Uncensored sample: qa n=54 p50=41 p90=54 max=62 >old-cap(30)=50, dropped=0
  non-emitters=0; researcher n=14 p50=19 p90=35 max=38 >old-cap(40)=0.
- MATRIX `mutate_rail_turn_cap.py --verify` => exit 0. REPRODUCED EXACTLY:
  cells=41 real survivors=0 known/equivalent(BY OUTCOME)=2 errors=0,
  kills {VERIFY 32, ORACLE 2, INJECTED_TRUTH 2, MUST_STAY_GREEN 3};
  control GREEN observed FIRST; byte-identical restore md5 rail_turn_cap.py
  642401bb1b325f80e9f8f5e0594bbc36 matching my own pre-baseline;
  qa.md 4c9faa6d..., researcher.md a9592ee0... unchanged.
  New cells observed: S16 KILLED[VERIFY], S17 KILLED (stored 0 != recount 42),
  S18 KILLED (stored 0 != recount 1), S18b KILLED[MUST_STAY_GREEN].
- Non-verify matrix run reproduces live_check S15's `tail -3` EXACTLY
  (blank + kills line + cells line) => capture regenerated, not spliced.
- criterion 4/7 executed: `node scripts/qa/verify_rail_retry.mjs` 38/0 EXIT 0
  ([F] F1 exhausted retry yields NO value, F2 rethrows original, F3 gate_passed
  recomputed, F4 no verdict field); `verify_research_gate_workflow.mjs` 124/0
  EXIT 0, and it drives the REAL exported enforceGate (temp re-export + import),
  incl. "a DROPPED stage 1 (null envelope) fails the gate even with a COMPLETE
  brief on disk" and "the drop path does NOT assign gate_passed".
- criterion 5 re-derived at HEAD today: rail_drop_rate.py:19-21 quote-then-
  retract + RUNTIME caveat printed under the by-model table (I ran it, exit 0,
  "^ CONFOUNDED -- NOT a model effect ... 47/379 capped vs 0/417 uncapped");
  qa-verdict.js "This line used to read: `... 2e-4 -- the model split SURVIVES.`"
  + "This block used to say it was UNPROVEN; that is SUPERSEDED";
  research-gate.js:879-890 "This block used to end ...". All three REPLACE.
- ruff F821/F401/F811 over a DERIVED 6-file scope (phase commits + working tree
  + untracked, non-empty asserted, xargs so no zsh word-split): All checks
  passed, exit 0. Non-vacuity proven: `--select E501 --line-length 60` on the
  same file reports 523 errors, so ruff IS reading them.
- frontend gate (qa.md is in the phase diff): `npx tsc --noEmit` EXIT 0;
  `npx eslint src` EXIT 0 (0 errors / 55 warnings). Repo-wide `npx eslint .`
  exit 1 with 26 errors, ALL attributed to .next-audit-36-12 (13) and
  .next-functional (13) build-output dirs; 0 in src/ -- pre-existing queued
  defect, not this step.
- SCOPE: uncommitted tree changes (backend/api/sovereign_api.py, frontend/src/**,
  backend/services/experiments/perf_results.tsv, audit jsonl) were present in the
  session-start snapshot and are NOT in c90910ef -- peer session, out of scope.
  No UI claims => gate 1c N/A. No backend/** in the step diff => gate 1d N/A.

## C. MY OWN MUTATIONS (temp mirror, mod.REPO=mirror, tree NEVER written;
##    md5 642401bb1b325f80e9f8f5e0594bbc36 before AND after)
- CONTROL: verify_ok=True problems=0 unclassified=41 post=0 n_erased=44.
- MUT-D **SURVIVOR**: marker renamed on its single non-comment line
  (qa-verdict.js:365) and retained ONLY inside a `/* ... */` block whose
  continuation line has no leading `*` => pin GREEN, problems=0. Same class as
  the DISCLOSED trailing-comment residual, one shape further. Mitigating: all 4
  block comments in qa-verdict.js are JSDoc with ` * ` continuations (checked),
  which the pin covers, so this is deliberate evasion -- and the code does say
  "not airtight against deliberate evasion". Residual: the artifacts enumerate
  ONE comment shape as if exhaustive.
- MUT-F **SURVIVOR (the disclosed one)**: literal only in a TRAILING same-line
  comment after code => pin GREEN. Confirms the code's own stated residual.
- MUT-E **SURVIVOR**: classifier mutated to never yield role=None (default-role
  defect) => stored 0 AND recount 0 => verify GREEN, 0 problems, while the true
  count is 41. The F2 cross-check is a SAME-SOURCE recount: it kills a hardcode
  at the aggregation site (S17/S18) but cannot see an upstream classification
  defect. Mitigating: the printed line moves 41 -> 0, i.e. ORACLE-detectable --
  the matrix already has ORACLE mode (S5/S14); it simply has no cell for this
  shape. NOT disclosed. Named fix: add an ORACLE cell mutating role assignment.

## D. CRITERION MAP (all independently re-derived this cycle)
1 MET  population rule + at-risk-ratio caveat printed by the command; corpus
       drift vs audit_basis disclosed as "captured at write time".
2 MET  three answers with citations incl. the NOs (no per-call turn budget in
       agent() opts -- decompiled 2.1.232; #20625 closed as not planned;
       general-purpose is uncapped but rejected for the tool surface; #41143).
3 MET  remedy is REMOVAL not a raise, and the UNCENSORED sample is committed
       and printed with its percentile rule.
4 MET  by execution (38/0 exit 0, [F] behavioural F1/F2).
5 MET  all three sources, quote-then-replace, split by model, runtime caveat.
6 MET  removal has no number to outgrow; re-measurement now automatic (the
       realised distribution prints on every run and carries verify() floors);
       cycle-10's undisclosed-residual gap is closed in code + artifacts.
7 MET  by execution (124/0 exit 0 driving the real enforceGate); both JS diffs
       comment-only; agent files lost only the pin line.
8 MET  control GREEN first, 41 cells, byte-identical restore reproduced by me,
       and the S18 first-run SURVIVOR was REPORTED rather than dropped.

## E. Residuals for queueing (EVIDENCE-quality, not product)
- MUT-D: pin residual enumeration incomplete (block-comment shape).
- MUT-E: F2 cross-check reach -- same-source recount cannot see a classifier
  defect; no ORACLE cell for it.
Neither touches the rail fix, any gate, any verdict semantics, or a money path.

COMPLETED: 2026-08-17T12:52:41Z
