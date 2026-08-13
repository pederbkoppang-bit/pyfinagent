STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.58
WRITTEN: 2026-08-13T20:43:01Z

# Q/A attempt 3 (cycle 3) -- step 86.58

Ledger: attempt 1 = FAIL (wf_b127735e-55b); attempt 2 = CONDITIONAL (wf_1e709e75-776);
attempt 3 = THIS RUN. `qa_wip.py 86.58` -> records_retained 3, prior_records 2.
Per qa.md 3rd-CONDITIONAL rule: PASS or FAIL only.
Cross-check `grep -cF "phase=86.58" handoff/harness_log.md` = 0 (the log is written in
LOG, after EVALUATE -- exactly the systematic under-read phase-86.75 documented).
Ledger governs.

## A. Harness compliance (5 items)
1. RESEARCH-GATE-BEFORE-CONTRACT -- PASS. research_brief_86.58.md 31,129 bytes,
   envelope brief_status COMPLETE, external_sources_read_in_full 6 (floor 5),
   urls_collected 38 (floor 10), recency_scan_performed true. I counted 39 distinct
   URLs in the brief >= 38 claimed, so no over-claim. Contract cites the gate and the
   gate demonstrably changed the design (recommendation_vocab.py already exists).
2. CONTRACT-BEFORE-GENERATE -- BREACHED, DISCLOSED. Criteria 1-2 ran before the
   contract. mtimes: brief 19:22:39Z < contract 19:24:51Z < results 20:25:55Z, so the
   blind mtime check PASSES; Main disclosed it anyway in contract_86.58.md lines 88-97
   and named the blind check. Verified independently: the disclosure is accurate.
   Recorded as a flag, not a criterion miss (none of the 6 criteria concerns ordering).
3. EXPERIMENT_RESULTS present -- PASS (16,558 bytes).
4. LOG-LAST -- PASS. 0 rows matching `phase=86.58` in harness_log.md; masterplan
   status = pending. Correct order (EVALUATE before LOG before flip).
5. NO-VERDICT-SHOPPING -- PASS. e00773dc changed live_check_86.58.md (+139/-34) after
   the cycle-2 verdict. Evidence CHANGED -> documented fresh-respawn, not a re-grade of
   unchanged evidence.

## B. Deterministic
- IMMUTABLE COMMAND: `parses`, **exit=0**.
- UNINTENDED PRODUCTION CHANGE: NONE. Per-commit file lists for cba26085 / cc0d2bff /
  e00773dc; `git diff <each>^ <each> -- backend/ frontend/ .claude/masterplan.json` is
  EMPTY for all three. Working tree clean for backend/ frontend/ scripts/ masterplan.
  (Note backend/agents/cost_tracker.py + api/settings_api.py DID change in the same
  window -- they belong to peer commits 73c8c2ce/e21468d5, not to 86.58.)
- RUFF F821,F401,F811 over a DERIVED scope (git diff cba26085^..HEAD -- '*.py', 5
  files, non-empty asserted; HEAD-diff is empty because the tree is committed --
  the known new-file trap): **All checks passed! exit=0**.
- BACKEND PID: `launchctl list` -> com.pyfinagent.backend pid **93024**;
  `ps -o pid=,lstart= -p 93024` -> `tor. 13 aug. 22.30.59 2026` local = CEST = UTC+2
  = **2026-08-13T20:30:59Z**. Main's corrected header VERIFIED exactly.
- DRIVE SCRIPT re-run by me: output **byte-identical** to the published verbatim
  block, exit=0.
- backend.log (44.5 MB, 212,338 lines) counts re-derived by me: UNRECOGNISED
  recommendation **4**, "healthy position" **0**, "signal_downgrade" **0**. All four
  timestamps match the artifact verbatim (08-10 21:15:12,974 / 08-11 21:21:09,983 /
  08-12 20:23:05,549 / 08-13 21:31:15,781).
- BQ re-derived by me: paper_positions holds 2 rows, BOTH currently held, BOTH
  recommendation='new_buy_signal'; off-vocabulary 2 of 2 = 100.0%, in-closed-set 0 of 2.
  NTAP entry 2026-07-31T18:47:37Z, DELL entry **2026-08-13T19:31:19.212436Z** (the row
  instrument 3 rests on). paper_round_trips: 32 total -- stop_loss_trigger 16,
  swap_for_higher_conviction 13, sell_signal 3, **signal_downgrade 0**.

## C. Flag state -- I closed the gap Main left open
- settings.py:343 `paper_risk_judge_reject_binding` code default = **False**, yet
  `Settings()` returns **True** => the read path provably reaches a non-default source.
  `.env` presence counts: RISK_JUDGE_REJECT_BINDING **1**; POSITION_RECOMMENDATION_FIX
  **0**; RECOMMENDATION_VOCAB_FIX **0**; SYNTHESIS_INTEGRITY **0**.
  => Main's instrument 1 positive control is GENUINE, not a defaults read.
- NEW INSTRUMENT (mine): `com.pyfinagent.backend.plist` EnvironmentVariables =
  {PATH, DEV_LOCALHOST_BYPASS, PYTHONUNBUFFERED} only; 0 hits for any of the 4 flags.
  The backend is launchd-managed, so this closes the launch-time-override residual gap
  Main disclosed as open. Main UNDERSTATED its own position.
- paper_synthesis_integrity_enabled = False, so instrument 2's disjunction
  ("posfix OFF or synthesis_integrity ON") collapses to posfix OFF.

## D. Mutation matrix (mine; in-process module-global patching, ZERO tree writes)
CONTROL observed GREEN first: B fires, A/E dead.
- M1 `_BUY_RECS += 'new_buy_signal'` (lowercase) -> A survived. **BROKEN PROBE, not a
  finding.** Source read: flags-OFF `_resolve_rec` = legacy `(raw or d).upper()`, so
  the resolved token is 'NEW_BUY_SIGNAL'; a lowercase literal can never match.
  Recorded because suspecting the probe first is the rule.
- M1b `_BUY_RECS += 'NEW_BUY_SIGNAL'` (resolved form), flags OFF -> A **KILLED**.
  Cell A is SENSITIVE to the membership test. Not vacuous.
- M2b `_BUY_RECS += '__UNRECOGNISED__'`, flags ON -> E **KILLED**. Confirms Main's
  stated mechanism directly: flags-ON `_resolve_rec('new_buy_signal')` =
  '__UNRECOGNISED__', in NONE of _BUY_RECS/_SELL_RECS/_DOWNGRADE_RECS.
- M5 `_BUY_RECS += 'SWAP_BUY'`, flags OFF -> C **KILLED**.
- M3 `_DOWNGRADE_RECS -= 'HOLD'` -> control B **KILLED** (the fresh-rec half is live).
- M4 ON-settings replaced by a non-overridden copy -> discrimination guard evaluates
  FALSE, i.e. the script's own guard WOULD go red. Not vacuous.
- RESTORE byte-identical: _BUY_RECS == original True, _DOWNGRADE_RECS == original True,
  post-restore re-run reproduces baseline (B fired True, A fired False).
- Shape-7 (RE-IMPLEMENTED test) EXCLUDED: patching the PRODUCTION module's globals
  changes the script's cells, which proves the script executes the real decide_trades.

## E. Criteria
1. MET -- real decide_trades driven; A/C dead flags-OFF; controls B (fires) and D
   (sell_signal pre-empts) present; my mutations prove non-vacuity.
2. MET -- 2 of 2 held rows off-vocabulary, 0 in closed set; query stated; I reproduced
   the members exactly (symmetric difference empty).
3. MET -- flags-ON measured via in-process model_copy (non-live, no .env write, no book
   contact); blast radius 0 of 2; discrimination control proves flag state is read;
   HOLD-only reachable domain derived correctly (`_DOWNGRADE_RECS - _SELL_RECS`),
   confirmed against source (sell_signal `continue`s first).
4. MET -- no flag promoted; both read False now; .env untouched; no backend/ change.
5. MET (not-applicable, correctly reported) -- no guard added, verified by empty
   backend/ diff across all three commits. Conditional antecedent false. I ran a
   mutation matrix against the proof harness anyway.
6. MET -- exactly 1 occurrence of "UNRECOGNISED recommendation" in
   portfolio_manager.py; file clean and untouched by this step (last commit is 86.20
   work); line fired 4x in production incl. today, and twice in my own run.

## F. NOTE-level (non-blocking, recorded)
- experiment_results §Criterion 2 leaves the stale "TOTAL 1 rows" block directly under
  the SQL with the CORRECTED block below it. Correction is adjacent and explicit.
- The displayed SQL has no `quantity > 0` filter while the criterion says "CURRENTLY
  held". Immaterial here: all rows in paper_positions are open positions; both
  operationalizations return {NTAP, DELL}.
- live_check §1 says "backend.log and its rotated archives"; no archives exist. All
  four firings reproduce from the live log alone, so nothing is inflated.
- §2e "three independent instruments" is strongest for posfix; instruments 2 and 3 do
  not speak to the vocab flag, and instrument 2's collapse needs instrument 1. My plist
  instrument strengthens the overall conclusion.

## G. Separation of duties
This Main session edited qa.md (phase-86.75) today, including the counter I ran. The
edit is STRICTER against the author, not laxer: under the retired harness_log grep this
step reads 0 priors and CONDITIONAL would still be available; under the new qa_wip
source it reads 2 priors and CONDITIONAL is FORBIDDEN. My verdict rests on numbers I
re-derived myself, not on any prior verdict.

VERDICT RETURNED: PASS (see the structured return; this file is not a verdict).

COMPLETED: 2026-08-13T20:52:40Z
