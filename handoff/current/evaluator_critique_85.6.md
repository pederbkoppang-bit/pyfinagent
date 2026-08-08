# Evaluator critique — phase-85.6 (cycle 184, EVALUATE pass 1)

**Verdict: CONDITIONAL** · `ok: False` · `harness_compliance_ok: True`

Launch: Workflow `qa-verdict` rail, run `wf_555a4380-3e8`. Evaluated commit
`b5f63525`. Transcribed VERBATIM from the Q/A's returned structured output —
Main records the verdict, never authors it. Machine copies:
`evaluator_critique_85.6_pass1.json` (this pass) and
`evaluator_critique_85.6.json` (the step's current verdict, which the auto-push
verdict gate reads).

---

## reason

Criteria 1, 2, 3, 4, 6 are MET with evidence I reproduced independently (immutable verification command exit=0 twice: {'paused': False, 'sod_date': '2026-08-08'} {'armed': True, 'daily_baseline_stale': False}; ruff F821/F401/F811 exit=0 over a 5-file git-DERIVED non-empty scope; 16 passed/16 dots in 13.91s; mutation matrix 9/9 killed with the tree restored byte-for-byte; sod_snapshot row provenance verified as code-written, append-only). Criterion 5 is NOT fully met: its two NAMED sub-clauses hold (no threshold literal changed -- daily 4.0%/trailing 10.0% confirmed live and absent from the diff; a leg-disabling mutation M5 is killed), but its headline clause "no loosening of the kill switch's protective behaviour" has a reproducible counterexample that I measured with the PRODUCTION evaluate_breach, not by reasoning. Re-sourcing the anchor from today's fresh mark (old :1298) to the last stored mark (new Step 0) is directional, not neutral: on a book that ROSE since the last mark and gives it back on a second same-UTC-day evaluation, NEW anchor=100 -> daily_loss=-4.0%, breached=False while OLD anchor=110 -> daily_loss=5.4545%, breached=True, and the trailing leg does NOT cover the gap (trailing_dd_breached=False for both). The mirror image is worse and is undisclosed anywhere in the handoff: stamping TODAY's date on a value that is N sessions old converts phase-36.9's designed disarm (armed=False, daily_baseline_stale=True, no breach -- I measured it) into armed=True measuring a MULTI-SESSION move as a same-day loss (NEW anchor=23830.46 vs a 22600 mark -> 5.16% -> any_breached=True -> flatten_all+pause, where OLD measured 0.00%). That is the exact hazard kill_switch.py's own phase-36.9 F1 comment records as measured on this book on 2026-07-26 ("a TWO-DAY move reported as a same-day loss ... biases toward a spurious flatten"). It was LIVE-reachable today: cycle c67b3b15 completed at 21:04:09Z, reached Step 5 mark-to-market, and Step 5.5 evaluated the daily leg against an anchor whose value was the 2026-08-05 mark -- it did not flatten only because the 3-session move happened to be -0.0146%. FAIL was considered and rejected: 5 of 6 criteria are met, the criterion's own operationalization holds, and in the dominant path the change is a large protective GAIN (under the old code the roll set sod_nav := the same nav the breach then used, so the daily leg computed exactly 0.00% and could not fire on the cycle that rolled -- verified). The gap is fixable by a bounded change plus honest disclosure, which is what CONDITIONAL is for.

## violated_criteria

- C5: no loosening of the kill switch's protective behaviour (headline clause) -- measured counterexample in the risen-book / second-same-day-evaluation state, uncovered by the trailing leg and untested [WARN]
- C5/(b) claim: the phase-36.12 invariant is preserved in LETTER but not in PURPOSE -- a multi-session-stale anchor value now reads as a same-day loss, re-opening the spurious-flatten path phase-36.9 F1 closed; undisclosed in contract_85.6.md and experiment_results_85.6.md [WARN]
- illusory-guard: BANNED[2] in test_phase_85_6_anchor_deadlock.py cannot fail -- the phrase was never contiguous in the pre-fix source [WARN]

## violation_details

### 1. Overgeneralization

**action**

```
PaperTrader.roll_daily_anchor() at Step 0 anchors get_or_create_portfolio()['total_nav'] (the LAST stored mark) instead of the post-mark_to_market NAV the old :1298 roll used; the only directional guard is test_c5_on_a_falling_book_the_early_anchor_is_never_more_forgiving, which covers ONE direction and computes the loss with test-side arithmetic rather than production evaluate_breach
```

**state**

Measured with production backend.services.kill_switch.evaluate_breach (control verified biting: anchor 200 vs 100 at nav 100 -> 50.0% breach vs 0.0% no-breach). last_mark=100.0, todays_mark=110.0, second same-day evaluation at nav=104.0 -> NEW anchor=100.0 daily_loss=-4.0% daily_breached=False any_breached=False; OLD anchor=110.0 daily_loss=5.4545% daily_breached=True any_breached=True; trailing_dd_breached=False on BOTH (5.45% vs the 10.0% trailing limit), so the trailing leg does not cover the band. Effective daily trigger moves from 105.6 to 96.0 on a 10% rise. Reachable via POST /api/paper-trading/run-now (paper_trading.py:1137), the same endpoint used for this step's own live proof; check_and_enforce_kill_switch has exactly one production caller (autonomous_loop.py:1400), so the flatten decision fires once per cycle and the divergence needs a second same-day cycle.

**constraint**

SEVERITY WARN. masterplan 85.6 success_criteria[5]: 'no loosening of the kill switch's protective behaviour'. experiment_results_85.6.md sec 6 states the falling-book result as 'i.e. strictly more protective'; contract_85.6.md sec 4 discloses the risen direction but argues it away as 'the standard definition ... not a relaxation' with no test and no bound.

### 2. Unjustified_Inference

**action**

```
Step 0 calls state.update_sod_nav(stored_nav, date=today), stamping TODAY's date on a NAV value that may predate today by several sessions; the :1298 roll then no-ops via sod_anchor_needs_reroll, so nothing ever re-anchors to today's fresh mark
```

**state**

Measured: stored=23830.46 (the 2026-08-05 mark) vs a next mark of 22600.0 -> NEW anchor=23830.46 date=2026-08-08 daily_loss=5.1634% daily_breached=True any_breached=True (flatten_all+pause), OLD anchor=22600.0 daily_loss=0.0000% any_breached=False. Same stale value with an HONEST date (phase-36.9 F1 behaviour) -> armed=False, daily_baseline_stale=True, any_breached=False. kill_switch.py evaluate_breach's own comment records this arithmetic as measured on this book 2026-07-26: 'a TWO-DAY move reported as a same-day loss ... biases toward a spurious flatten (a nuisance trip and a diagnostic failure together)'. LIVE today: the journal holds exactly one sod_snapshot for 2026-08-08 (nav=23830.46, written 20:58:29.379594Z by Step 0), backend.log shows Step 5 mark-to-market at 21:03:29Z, and the GET /kill-switch read shows sod_nav=23830.46 / current_nav=23833.94 / daily_loss_pct=-0.0146 / baseline_provenance=null -- i.e. the daily leg WAS evaluated against a 3-session-old anchor value and passed by margin, not by design.

**constraint**

SEVERITY WARN. paper_trader.py:1290-1308 'DO NOT blanket-reorder this' -- the 36.12 invariant exists so the breach is not computed as '(yesterday_sod - today_nav)/yesterday_sod, i.e. a multi-day move read as a same-day loss'. Spawn claim (b) asserts the invariant is preserved because the anchor's DATE is today's by Step 5.5; the invariant's stated PURPOSE is that the anchor's VALUE be today's open. Neither handoff artifact discloses this residual.

### 3. Circular_Reasoning

**action**

```
test_c2_the_409_no_longer_makes_the_two_false_promises asserts three BANNED phrases are absent from inspect.getsource(resume_trading)
```

**state**

Verified against the pre-fix blob (git show 81f81750^:backend/api/paper_trading.py): 'NO operator action is required' -> True, 'this refusal clears itself' -> True, 'at the top of the next paper-trading cycle' -> False. The third phrase was split across two adjacent string literals in the pre-fix source, so that assertion was already satisfied by the code it was written to forbid and cannot fail for any mutation of the message.

**constraint**

SEVERITY WARN (not BLOCK). qa.md sec 4c: a guard that cannot fail when its subject is broken does not count. WARN because two genuinely live banned-phrase guards coexist (M7 killed) plus the verbatim live HTTP body in live_check_85.6.md sec 2 from restarted pid 23676.

## checks_run

- harness_compliance_audit_5_item
- research_gate_envelope_verified (tier=complex, external_sources_read_in_full=7, urls_collected=22, recency_scan_performed=true, gate_passed=true; contract cites the brief 2x)
- contract_before_generate_mtime_chain (brief 22:40:13 < contract 22:43:29 < code 22:54:40-22:55:46 < experiment_results 23:00:16 < live_check 23:01:22; commit order 81f81750 -> 5932ac27 -> b5f63525 corroborates)
- log_last_and_status (harness_log result= rows for 85.6 = 0; masterplan status=pending, retry_count=0/3)
- no_verdict_shopping (first EVALUATE for 85.6)
- git_derived_scope (81f81750^..HEAD UNION worktree UNION untracked, non-empty guard asserted: 5 .py files)
- ruff_F821_F401_F811 (exit=0, 'All checks passed!', 5-file derived scope, array-quoted args)
- immutable_verification_command (exit=0, run twice, output reproduced verbatim)
- pytest_backend_tests_test_phase_85_6_anchor_deadlock (16 passed / 16 progress dots / 13.91s -- internally consistent)
- mutation_matrix_85_6 (9/9 killed, re-run by me; tree restoration verified via git status --porcelain backend/ scripts/ = clean)
- production_diff_review (paper_trader.py +95, autonomous_loop.py +25, api/paper_trading.py +36/-5; no unintended production file)
- backend_runtime_smoke (live :8000 on restarted pid 23676; GET /api/paper-trading/kill-switch 200; POST-fix 409 body confirmed live)
- independent_mutation_probe_production_breach_math (control + Scenario A risen-book + Scenario B multi-session gap + Scenario C phase-36.9 disarm; seam _state identified after a contaminated first run was caught and discarded)
- audit_row_provenance (git diff --numstat 2 added / 0 deleted = append-only; row shape matches _append_audit('sod_snapshot', nav=, date=) at kill_switch.py:550; ts 20:58:29.379594Z matches backend.log paper_trader line 22:58:29,380 CEST; lands 2s after the cycle_history 'started' row)
- live_idempotence_of_the_1298_roll (exactly ONE sod_snapshot for 2026-08-08 despite Step 5.5 running at 21:03:xx)
- banned_phrase_vacuity_check_against_pre_fix_blob
- code_review_heuristics (secrets / eval / exec / os.system / shell=True sweep clean; kill-switch reachability; sod-nav-anchor WARN; broad-except assessed as logged fail-safe, not except-pass)
- 3rd_conditional_counter (0 prior CONDITIONALs for 85.6 -- CONDITIONAL permitted)

## notes

SPAWN QUESTIONS ANSWERED. (a) YES -- a book state exists where the new anchor makes the daily leg fire LATER; measured above, uncovered by the trailing leg. I did NOT simply adopt the author's framing that this alone must fail the step: the immutable criterion's own operationalization (thresholds, leg-disabling mutation) holds, and the change is a net protective GAIN in the dominant path, so this is CONDITIONAL, not FAIL. (b) HALF-VERIFIED. The syntactic half is TRUE and I confirmed it in LIVE, not just in tests: exactly one sod_snapshot exists for 2026-08-08 even though Step 5.5 ran at 21:03:xx, so :1298 no-opped via sod_anchor_needs_reroll. The semantic half is FALSE: 36.12's purpose is that the post-roll anchor be TODAY'S value, and Step 0 now guarantees only today's DATE. (c) SATISFIED. The criterion asks for "a test [that] drives a cycle that dies mid-analysis"; test_c1_a_cycle_that_dies_mid_analysis_still_leaves_a_fresh_anchor drives the real run_daily_cycle with _run_single_analysis hanging, asserts its OWN precondition (entered_analysis non-empty) so it cannot pass vacuously, asserts status=="timeout" and "analyzing" in steps, then asserts sod_date==today; M1 (roll deleted) kills 3 tests on my own run. A pytest with an injected hang is the right seam -- waiting for a real 2.4h production timeout would add no signal. (d) VERIFIED NOT HAND-WRITTEN, by four independent signals: the journal diff is append-only (2 added, 0 deleted -- a forgery of history would show removals), the row's key set matches _append_audit('sod_snapshot', nav=, date=) at kill_switch.py:550 exactly, the ts corroborates the paper_trader log line at 22:58:29,380 CEST from the restarted pid, and it lands 2s after the cycle_history "started" row for c67b3b15. (e) ADEQUATE. Both disclosures are against interest and the remedies are real code I read, not promises: the spend tripwire at test lines 286-294 monkeypatches claude_code_client.claude_code_invoke to raise AssertionError, and the fixture redirects _LOCK_PATH/_HISTORY_PATH/_HEARTBEAT_PATH into tmp_path (lines 249-251). I did NOT independently verify the "llm_call_log 0 rows in 30 min" claim -- it rests on Main's measurement. (f) BOTH FIXES ARE REAL, not cosmetic. M2's fix replaces a call-timing probe with the cycle's own recorded summary["steps"] order AND anchors the source assert on the full call expression "await asyncio.to_thread(trader.roll_daily_anchor)", so the explanatory comment can no longer satisfy .index(); M6's fix pins the roller's CALL SET (set(ks.calls) <= {"snapshot","update_sod_nav"}) instead of observing peak values, defeating the monotonic-stub escape. Both re-killed on my run (M2: 2 failed; M6: 1 failed, 1 passed). RESIDUAL I did not execute: summary["steps"].append("sod_anchor_roll") is appended BEFORE the to_thread call, so steps[0]=="sod_anchor_roll" is a LABEL assertion (vacuity shape 3). It is not sole coverage -- a label/call split is killed by the C1 freshness assertion and by the source-position assert in test_c2_the_mechanism_the_message_names_actually_exists -- so NOTE, not a finding; flagged as reasoned, not executed. NAMED FIXES for cycle 2 (either 1a or 1b closes the blocking pair): (1a) at Step 5.5, re-anchor to the fresh post-mark_to_market NAV when the Step-0 anchor's underlying mark predates today -- keeps the deadlock broken AND restores 36.12's purpose; (1b) record baseline_provenance on the Step-0 roll (it is currently null) with the marked_at of the NAV used, and refuse-or-flag when that mark is older than the previous session; (2) add the mirror directional test for the risen book that BOUNDS the loosening, computed with production evaluate_breach rather than test-side arithmetic; (3) drop or repair BANNED[2] (normalize adjacent-literal concatenation before matching) so it is not counted as evidence. FOR THE RECORD, non-blocking: experiment_results sec 4 point 1 asserts "Monday's cycle still projects to 8529s against 7200s and will still die in analyzing", but cycle c67b3b15 COMPLETED in 342449ms (5.7 min) with breaker_tripped=true, meta_scorer_degraded=true and only 5 new tickers -- not a clean refutation, but the projection is now stale; C3's load-bearing evidence is point 2 (the 2026-08-05 completing cycle that traded nothing because paused), which stands and is strengthened by today's completing cycle. POSITIVE FINDING neither artifact claims: under the old code the roll set sod_nav := the same nav the breach then evaluated, so daily_loss_pct was identically 0.00% and the daily-loss leg could NOT fire on the cycle that rolled it -- on a once-daily cadence the leg was effectively inert. This step revives it. That is the strongest safety argument available for the design and it is absent from the handoff. SAFETY: I triggered no cycle, restarted no service, and never paused/resumed the switch; all probes used injected fake state in my own process and the live journal grew by 0 rows during this evaluation.

---

## Follow-up — Main's cycle-2 remediation (written by Main, NOT by the Q/A)

Everything above is the Q/A's verbatim return. This records what changed, so the
fresh Q/A grades CHANGED evidence rather than re-reading the same tree.

| Finding | Severity | Action |
|---|---|---|
| C5 headline: re-sourcing the anchor is directional, and a multi-session-old value stamped with today's date turns a designed disarm into a spurious flatten (measured with production `evaluate_breach`) | WARN, blocking | **Fixed.** The Step-0 anchor is now PROVISIONAL; `check_and_enforce_kill_switch` upgrades it to today's freshly-marked NAV before any breach decision, so the value judged against is byte-equivalent to pre-85.6 while the anchor still exists from t+2s for `/resume`. New test drives the REAL production method. |
| C5/(b): the 36.12 invariant preserved in LETTER but not PURPOSE; undisclosed in both artifacts | WARN | **Fixed by the same change** (the anchor's VALUE is now today's open by the time the breach runs) **and disclosed** in `experiment_results_85.6.md` §12, including the residual (a cycle dying before Step 5.5 leaves a provisional anchor for that UTC day). |
| Illusory guard: `BANNED[2]` was never contiguous in the pre-fix source, so its absence-assertion could not fail | WARN | **Fixed.** The check normalises string-literal concatenation and whitespace, and now **self-validates** against the real pre-fix blob — every banned phrase must be PRESENT there or the test fails as vacuous. |

Re-verified after the changes:

```
pytest backend/tests/test_phase_85_6_anchor_deadlock.py    19 passed (was 16)
scripts/qa/mutation_matrix_85_6.py                        12/12 killed (was 9/9)
uvx ruff check --select F821,F401,F811 (3-file derived scope)   All checks passed! (exit 0)
```

**M10 was LIVE on its first run** — my hazard test performed the upgrade itself
instead of calling `check_and_enforce_kill_switch`. Corrected to drive the real
seam; it now dies under M10.
