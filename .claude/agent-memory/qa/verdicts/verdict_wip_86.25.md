STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.25
WRITTEN: 2026-08-10T11:28:11Z

CYCLE: 2 (cycle-1 returned CONDITIONAL with W1/W2/W3)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command + exit code; git status scope
C. Verify the three cycle-1 fixes (W1 attribution, W2 rename, W3 research numbers)
D. Judge M5/M8 surviving mutants (Main's "dead by construction" position)
E. Check for corrupted historical filename references (self-inflicted sed error)
F. Criterion-by-criterion MET/NOT MET

## Log

### B. DETERMINISTIC
- IMMUTABLE CMD `pytest backend/tests/ -q -k "outcome_tracker or autonomous_loop or learn_loop"`
  => **108 passed, 3303 deselected, EXIT=0** (re-run by me 11:29Z). Reproduces
  experiment_results cycle-2 block exactly.
- `--collect-only | grep -c "test_phase_86_25_outcome_tracker_vocabulary_boundary.py::"` => **16**.
  W2 REMEDY VERIFIED: the rename brings all 16 new tests inside the immutable filter. No
  criterion amended (masterplan verification.command byte-identical).
- git status --short: only audit jsonl / heartbeat / archive-baseline / untracked archive dir +
  my own WIP file. NO unintended production change.

### W3 VERIFIED FIXED
- brief envelope: external_sources_read_in_full 7, snippet_only 22, urls_collected 29,
  recency_scan_performed true, gate_passed true. Tally line: "7 read in full + 22
  snippet-only = 29 unique URLs collected".
- contract_86.25.md sec 1 now: 7 / 22 / 29, with the correction disclosed in a blockquote.
  REPRODUCES. MET.

### *** W1 RESIDUAL -- FINDING (cycle-2 fix INCOMPLETE) ***
- Cycle-1 critique named FOUR artifacts carrying the mis-attribution, and its
  TO-CLEAR-TO-PASS line said verbatim: "Correct the (A)-branch rationale in
  autonomous_loop.py, **recommendation_vocab.py** and experiment_results".
- MEASURED: `git diff 8baecb49 HEAD -- backend/services/recommendation_vocab.py` is EMPTY.
  The file was NOT touched in cycle 2. `git log --oneline -- recommendation_vocab.py`
  newest entry is 8baecb49 (cycle 1).
- recommendation_vocab.py:162-169 STILL reads: "MEASURED 2026-08-10, and it is why this
  resolver exists rather than a lookup: the analyst recommendation is reachable for 0 of 32
  SELL rows. `analysis_id` is empty on 32/32 SELLs (BUYs carry it 33/33), and
  `round_trip_id` is ONE-SIDED ... so a SELL cannot reach its BUY leg either."
  => the exact anchor-based mechanism W1 ruled NOT the operative cause.
- experiment_results cycle-2 W1 claims: "Corrected in `autonomous_loop.py`,
  `nightly_outcome_rebuild.py` and here." -- SUBSTITUTES nightly_outcome_rebuild.py for
  recommendation_vocab.py. Remediation-set substitution (cf.
  feedback_recheck_prior_remediation_list).
- SECOND residual, worse: `autonomous_loop.py:3417-3423` STILL carries the refuted causal
  sentence VERBATIM, immediately ABOVE the cycle-2 correction block: "Nothing is what is
  available -- MEASURED 2026-08-10: the anchor is reachable for 0 of 32 SELL rows ... So
  this resolves to UNKNOWN today". The correction 7 lines below calls that text "An earlier
  version of this comment" -- but it is not earlier, it is still present. Append-only fix.
- THIRD (NOTE): the SAME block was pasted into nightly_outcome_rebuild.py:88-91 asserting
  "An earlier version of this comment blamed the unreachable ANCHOR". MEASURED FALSE for
  that file: `git show 8baecb49:...nightly_outcome_rebuild.py` shows its cycle-1 comment
  never mentioned the anchor.

### W1's CORRECTED mechanism -- independently re-derived, TRUE
- `_production_fns.py:229-231` LEDGER_FETCH_SQL selects exactly 10 named cols:
  trade_id, ticker, action, price, quantity, created_at, analysis_id, risk_judge_decision,
  holding_days, pnl. `analyst_recommendation` ABSENT.
- repo-wide `grep -rn --include='*.py' analyst_recommendation backend scripts`: the ONLY
  emitter of that key is the TEST fixture (test file :156). No production producer.
  => dead BY CONSTRUCTION confirmed.

### LINT
6 git-derived .py files (non-empty), `xargs uvx ruff check --select F821,F401,F811`
=> "All checks passed!" exit=0.

### MUTATION (mine, in-process sys.modules injection, ZERO tree writes)
- CONTROL rc=0, 108 passed.
- S2-revert -> rc=1, **3 failed / 105 passed** KILLED (matches Main's claim EXACTLY)
- V1-sentinel -> rc=1, **10 failed / 98 passed** KILLED (matches EXACTLY)
  => W2 remedy VERIFIED: both die INSIDE the immutable command.
- N1 S1 reads trade ACTION through resolver (FAIL-UNSAFE, would persist a fabricated SELL)
  -> rc=1, 1 failed / 107 passed **KILLED**
- N2 (= predecessor M8) S1 reads risk_judge_decision through resolver -> rc=0 **SURVIVED**
- N3 S2 reads trade ACTION through resolver -> rc=1, 3 failed / 105 passed **KILLED**
  => ADJUDICATION: M5/M8 are EQUIVALENT-ON-THE-SAFE-SIDE. The argument regressions that
  can FABRICATE A DIRECTION (N1, N3) both DIE. M8 survives only because the approval
  vocabulary cannot canonicalise -- which is the boundary property the step built.
  Main's position is CORRECT and now has executed differential evidence. The S1 guard is
  NOT too weak. Residual nuance (NOTE): M8-equivalence is contingent on risk_judge_decision
  never overlapping the recommendation scale, not on construction.

### RUNTIME SMOKE (1d)
All 3 changed backend modules import clean. Live resolver over the FULL measured value set:
resolve(None)='UNKNOWN', resolve('')='UNKNOWN', resolve('APPROVE_REDUCED')='UNKNOWN',
resolve('REJECT')='UNKNOWN', resolve('APPROVE_HEDGED')='UNKNOWN',
resolve('Strong Buy')='STRONG_BUY', is_directional('UNKNOWN')=False.
=> criterion 4 verified LIVE, not just by test.

### A. HARNESS COMPLIANCE -- CLEAN (5/5)
- research gate: research_brief_86.25.md on disk, envelope gate_passed true, 7>=5 sources,
  29>=10 URLs, recency scan performed. mtime 10:32 < contract.
- contract-before-generate: cycle-1 contract 13:01:11 < .py 13:08 < experiment_results
  13:09 (cycle-1 Q/A measured); commit order 64d20023 -> 8baecb49 agrees.
- experiment_results + live_check present.
- log-last: `grep -cF "86.25" handoff/harness_log.md` = 0; masterplan status = "pending".
- no verdict-shopping: evidence CHANGED (f71030b8: 2 comment blocks, 1 rename, 1 contract
  table, artifact appends). Legitimate cycle-2 respawn.
- 3rd-CONDITIONAL counter: 0 logged CONDITIONALs for 86.25; this would be #2. Rule does
  NOT fire.
- 1c live UI gate does NOT bind (no frontend file touched, no UI claim).

### CRITERIA
1 MET  TestReproduceTheScoringDefect drives the REAL evaluate_recommendation (only the
       price source stubbed), asserts precondition return_pct<0 then
       directionally_correct is False. Reproduce-first satisfied.
2 MET  distribution re-derived by measure_86_25_join_hitrate.py; cycle-1 Q/A re-derived
       against live BQ (46/15/3/1, n=65). Unchanged in cycle 2.
3 MET  producer = S2 nightly_outcome_rebuild; cycle-1 corroborated with live rows
       (evaluated_at 04:00:02 cron, price_at_recommendation/beat_benchmark NULL).
4 MET  verified live above + parametrised test over the measured set + M4 killed.
5 MET  premise correction justified and evidenced; asserted at the write chokepoint.
6 MET  both call sites reverted independently and KILLED; V1 killed; my N1/N3 add two
       more kills; survivors adjudicated by executed differential.

### VERDICT: CONDITIONAL (ok=false)
All 6 immutable criteria MET; harness compliance clean; no unintended production change;
W2 and W3 fully fixed and reproduced EXACTLY. Capped on ONE finding: the W1 remediation is
INCOMPLETE and the artifact claims it is complete.

COMPLETED: 2026-08-10T11:47:05Z

