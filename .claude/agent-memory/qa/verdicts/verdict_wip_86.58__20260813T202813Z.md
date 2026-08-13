STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.58
WRITTEN: 2026-08-13T20:28:13Z
COMPLETED: 2026-08-13T20:38:13Z

## FINAL CRITERION TABLE (see evidence in the findings log below)
1 MET  -- driven proof re-run by me (exit 0) AND re-derived with my OWN harness on the
          REAL BQ rows; positive control (rec='BUY') fires on both real rows.
2 MET  -- query stated next to counts; I re-ran it: 2 of 2 off-vocab, 0 in closed set.
          Exact reproduction. Cycle-1 stale count fixed, correction disclosed.
3 MET  -- flag-ON entered via in-process model_copy (no .env write, no live book);
          blast radius 0 of 2, independently confirmed, plus my extra
          only-posfix / only-vocab decomposition (all dead). Discrimination control
          genuine; "_pos_rec written only by execute_buy" recall-tested and holds.
4 MET  -- no flag promoted; recommendation withdrawn and replaced on the measured basis.
5 MET as NOT-APPLICABLE -- antecedent unmet (no guard added); re-tested, not inherited.
6 MET  -- 1 source occurrence unchanged, fired 2x in my run, 4x in live backend.log.

## VERDICT REACHED: CONDITIONAL
Worst-of-N lenses (P1 money path): correctness=PASS; does-it-reproduce=CONDITIONAL
(the pid/not-restarted header claim does not reproduce); scope-honesty=CONDITIONAL
("NOT OBTAINABLE" over-claims a dead end where 3 instruments existed, one of them
positive-controlled). min = CONDITIONAL.
Two fixable, non-measurement blockers -- FINDING 1 and FINDING 2. No criterion miss,
so not FAIL. Attempt 2 of the 3rd-CONDITIONAL counter, so CONDITIONAL is permitted.

# Q/A WIP record -- step 86.58, ATTEMPT 2 (per spawn prompt)

Prior verdict on disk: FAIL (wf_b127735e-55b), transcribed in
handoff/current/evaluator_critique_86.58.md.
Evidence CHANGED per prompt: experiment_results_86.58.md (criteria 2/3/4 corrected),
live_check_86.58.md (NEW), scripts/qa/drive_86_58_dead_downgrade.py (REWRITTEN),
commit cc0d2bff.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git diff scope, ruff lint, runtime smoke
C. Re-run the driven proof MYSELF; mutation/vacuity test the probe
D. Criterion-by-criterion MET/NOT MET

## Findings log (appended as established)

### Attempt counter
qa_wip.py 86.58 -> records_retained=2, prior_records=[verdict_wip_86.58__20260813T201406Z.md].
=> THIS IS ATTEMPT 2. Prior sequence: [FAIL (wf_b127735e-55b)]. Not the 3rd; CONDITIONAL
still permissible. harness_log cross-check: 1 grep hit for "86.58" and it is a FILING
reference, not a `result=` cycle row -> ledger governs, and they do not conflict.

### A. Harness compliance (5 items)
1. research-gate-before-contract: PASS. research_brief_86.58.md envelope
   brief_status=COMPLETE, gate_passed=true, external_sources_read_in_full=6 (floor 5),
   urls_collected=38 (floor 10), recency_scan_performed=true, 2 "Recency scan" sections.
   mtime 19:22:39Z < contract 19:24:51Z.
2. contract-before-generate: BREACHED AND SELF-DISCLOSED. contract_86.58.md:88-97 states
   criteria 1+2 ran BEFORE the contract, and explicitly says a file-mtime check would
   PASS. I confirm the mtimes DO pass (research 19:22:39Z < contract 19:24:51Z <
   experiment_results 20:25:55Z) -- so the blind check is blind, exactly as disclosed.
   Disclosure is the mitigation; credited, not excused.
3. experiment_results present: YES (16,558 B).
4. log-last: PASS. masterplan 86.58 status=pending; harness_log has NO `result=` row.
5. no-verdict-shopping: PASS. Evidence CHANGED: cc0d2bff rewrote
   drive_86_58_dead_downgrade.py (192 lines changed), experiment_results (+201/-161),
   live_check_86.58.md NEW (144 lines). Not unchanged evidence.

### B. Deterministic
- IMMUTABLE COMMAND: `parses`, exit=0.
- COMMAND-AS-PRINTED: extracted programmatically from experiment_results (not retyped)
  and executed -> `parses`, exit 0. The prior Q/A's "dropped closing quote / unrunnable"
  blocker is FIXED.
- ruff F821,F401,F811 on commit-derived scope [scripts/qa/drive_86_58_dead_downgrade.py]:
  "All checks passed!", exit 0. Non-empty scope asserted first.
- Unintended production change: NONE from this step. cc0d2bff and cba26085 touch ZERO
  files under backend/. NOTE: 4 backend files DID change in the range cba26085^..HEAD,
  but they belong to interleaved commit 56abdbde (Claude 5 model picker, different work).
  Main's wording scopes to "this step's commits" and is accurate.

### C. Independent re-derivations (I did not trust the artifact)
- CRITERION 2 REPRODUCES EXACTLY. My own BQ query: paper_positions has 2 rows,
  NTAP 'new_buy_signal' (2026-07-31T18:47:37Z), DELL 'new_buy_signal'
  (2026-08-13T19:31:19Z). in_closed_set=False for both. 2 of 2 off-vocab, 0 of 2 in the
  closed set. Matches published figures.
- paper_round_trips REPRODUCES EXACTLY: 32 total, stop_loss_trigger 16 (50.0%),
  swap_for_higher_conviction 13 (40.6%), sell_signal 3 (9.4%), signal_downgrade ABSENT.
  Positive-controlled by the adjacent sell_signal=3.
- MAIN'S SCRIPT re-run by me: exit 0, output reproduces byte-equivalently.
- MY OWN INDEPENDENT HARNESS (not Main's script), driving decide_trades with the REAL
  BQ row values (author did not choose them), real stop_loss/current_price verified
  non-pre-empting:
    real rows, fresh=HOLD, flags OFF        -> []   signal_downgrade=False
    real rows, fresh=HOLD, flags BOTH ON    -> []   signal_downgrade=False
    real rows, fresh=HOLD, ONLY posfix ON   -> []   signal_downgrade=False   (my addition)
    real rows, fresh=HOLD, ONLY vocab ON    -> []   signal_downgrade=False   (my addition)
    POSITIVE CONTROL rec='BUY' on SAME rows, OFF -> BOTH NTAP+DELL fire signal_downgrade
    POSITIVE CONTROL rec='BUY' on SAME rows, ON  -> BOTH NTAP+DELL fire signal_downgrade
  => criterion 1 and the 0-of-2 blast radius are CONFIRMED independently, with a green
  control proving my harness CAN observe the rule firing on these exact rows. My flag
  DECOMPOSITION (only-posfix / only-vocab) is stronger than Main's both-on-only test.
- ANTI-VACUITY OF THE FLAG OVERRIDE: both flags demonstrably take effect in-process.
  The vocab flag changes the 'Strong Buy' outcome (dead OFF / fires ON) AND prints
  `vocab_fix_enabled=True|False` in its own log line; the position flag independently
  drives the :212 interaction warning, which fired in ON cells and not OFF cells.
  So the override is NOT a no-op -- Main's discrimination control is genuine, and the
  :212 warning is a SECOND, independent discriminator Main did not credit.
- COMPLETENESS CLAIM "_pos_rec written only by execute_buy" -- RECALL-TESTED, HOLDS.
  `_pos_rec` occurs at paper_trader.py:452,457,488,512 only (all inside execute_buy).
  Every production save_paper_position caller enumerated: :498,:519 (execute_buy),
  :682 (partial sell, writes position.get("recommendation","") unchanged), :974
  (backfill_missing_stops, `{**pos, "stop_loss_price":...}` = preserves), :1060
  (phase-32.4 company_name, `{**pos, ...}` = preserves), :1643/:1648 (_safe_save_*
  wrappers). No production writer sets recommendation from a fresh analysis outside
  execute_buy. bigquery_client.save_paper_position MERGEs with a None-drop, so an
  omitted key leaves the column untouched. => "flipping a flag does not rewrite rows
  already on disk" is TRUE.
- CRITERION 5 re-tested, NOT inherited: paper_trader.py:452 `_pos_rec = reason` has no
  parse/validate step and the flag gate at :454 is the only conditional; reached only
  from execute_buy. The conditional criterion ("any guard added") has an unmet
  antecedent because no guard was added. NOT-APPLICABLE reading is correct.
- CRITERION 6: 1 occurrence of "UNRECOGNISED recommendation" in portfolio_manager.py
  (unchanged); fired 2x in my re-run of Main's script; 4 occurrences in the LIVE
  backend.log. Preserved, not quieted.

### FINDING 1 (WARN) -- live_check_86.58.md:4 pid claim does not reproduce
Artifact states: "Backend: pid 99231, started tir. 11 aug. 22.26.48 2026 -- not
restarted this session."
MEASURED NOW: pid 99231 DOES NOT EXIST. The process serving :8000 is pid 93024,
started tor. 13 aug. 22.30.59 2026 (= 2026-08-13T20:30:59Z), i.e. ~3m45s AFTER
live_check was written (20:27:14Z). handoff/logs/backend-watchdog.log shows
"20:31:00Z health FAIL (1/3)" then "20:32:00Z health OK" -- consistent with a restart
at 20:30:59Z, and NOT watchdog-initiated (it escalates only at 3/3).
=> The claim was probably TRUE when written and went stale minutes later; I cannot
attribute the restart to Main (two concurrent sessions run in this repo). But §2's
whole "running process" disclosure is now scoped to a DEAD pid. Substance survives:
I re-probed the NEW pid and got 45 keys / 15 paper_* / 0 flag hits -- identical.

### FINDING 2 (WARN) -- "NOT OBTAINABLE" is honest but over-claimed
Reproduced Main's dead-end exactly and independently: GET /api/settings/ -> http 200,
45 keys, 15 paper_* keys, 0 hits for either flag (only 'max_synthesis_iterations'
matches 'synthesis'). Route list DERIVED from settings_api.py: only GET "/", PUT "/",
GET "/models", PUT "/models", GET "/models/available" -- no read route exposes them.
So "not readable through the endpoint" is TRUE and Main did NOT fake it.
BUT two production observables constrain the running flag state and were not used:
  (a) The :212 interaction warning is flag-gated on paper_position_recommendation_fix_
      enabled. In live backend.log it appears 0 times, WHILE the positive control
      (UNRECOGNISED recommendation, 4 hits) proves decide_trades actually ran and the
      log channel works. => posfix OFF OR synthesis_integrity ON. A real constraint.
  (b) DELL was written by the RUNNING process TODAY at 19:31:19Z with
      recommendation='new_buy_signal'. Per paper_trader.py:452-457 a posfix-ON process
      with a non-empty analysis_recommendation would have stored the verdict instead.
      => posfix OFF OR analysis_recommendation empty. A second real constraint.
  (c) Settings model_config carries env_file=backend/.env (settings.py:652), so a fresh
      Settings() reflects the CURRENT .env, not merely hardcoded defaults. Main called
      it "the defaults path", which UNDERSTATES its own evidence -- erring safe.
Neither (a) nor (b) is a direct read, so UNVERIFIED remains the right label; but the
artifact presents a dead end where partial measurement was available.

### Scoped test run (derived scope, not hand-typed)
`pytest $(grep -rln "decide_trades|portfolio_manager|recommendation_vocab" backend/tests/)`
-> 5 failed, 480 passed, 1 xfailed in 44.42s.
CLASSIFIED, NOT ASSUMED: root cause is `s_off.paper_risk_judge_reject_binding is True`
where the test asserts it "ships default-OFF" -- i.e. an OPERATOR .env PROMOTION of an
UNRELATED flag (qa.md vacuity shape #9). Plus one environment-dependent backend.log
assertion (test_phase_23_2_6_..._has_skipping_buy_evidence) that reads a log the backend
just rotated by restarting. 86.58 changed ZERO backend files, so none of these are
attributable to it. NOT a regression from this step.

### POSITIVE CONTROL that upgrades FINDING 2(c)
Those failures prove the `.env` read path in `Settings()` is LIVE: a sibling flag
(`paper_risk_judge_reject_binding`) reads **True** from the same fresh `Settings()`
because it is promoted in backend/.env. Therefore a fresh `Settings()` returning
False for BOTH 86.58 flags is not "merely the defaults path" -- it is a
POSITIVE-CONTROLLED read of the operator's actual .env, showing neither 86.58 flag is
promoted on disk. Combined with the running process (pid 93024) having started TODAY at
20:30:59Z and reading that same .env at startup, the running-process flag values are
derivable with high confidence. Residual gap: a launch-time env var could override .env
and I could not enumerate the full process env (`ps eww 93024` exposed only 14
env-like tokens, 0 flag hits), so this is strong evidence, not proof. "UNVERIFIED"
remains a defensible label; "NOT OBTAINABLE" does not.

### FINDING 3 (NOTE) -- the system was not frozen during EVALUATE
A backend restart (20:30:59Z) and an unrelated backend commit (56abdbde) both landed
inside this step's evaluate window. Neither changes any measured result -- I re-ran
everything against current state -- but it is why FINDING 1 exists.

