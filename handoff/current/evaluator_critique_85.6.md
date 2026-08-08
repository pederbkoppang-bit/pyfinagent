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

---

# EVALUATE pass 2 — verdict **CONDITIONAL**

`ok: False` · `harness_compliance_ok: True`
Run `wf_4b1efa12-c58`, commit `e369b1de`. Verbatim below. Machine copy:
`evaluator_critique_85.6_pass2.json`.

## reason

Criteria 1, 2, 3, 4 and 6 are MET and I reproduced them independently (immutable verification command exit=0: {'paused': False, 'sod_date': '2026-08-08'} {'armed': True, 'daily_baseline_stale': False}; ruff F821/F401/F811 exit=0 over a git-DERIVED non-empty 5-file scope; 19 passed in 13.95s; mutation matrix 12/12 killed with the tree verified byte-clean afterwards; imports of all three changed backend modules OK). The C5 fix is REAL on the path it was built for: driving the PRODUCTION check_and_enforce_kill_switch, the provisional 23830.46 anchor is upgraded to today's 22600 mark before the breach and triggered=False (pass-1 case (i) closed), and the risen-book case (ii) now behaves exactly as pre-85.6 (cycle-1 upgrade sets 110, the second same-day evaluation at nav=104 reports 5.4545% and FIRES, matching the pre-85.6 number) -- so case (ii) is closed too, not merely mitigated. Criterion 5 is still NOT MET because the mechanism is an in-memory flag on the PaperTrader instance and autonomous_loop.py:407 constructs a NEW PaperTrader inside every run_daily_cycle, so the flag does not survive the CYCLE, not merely "the process". Measured through the production path, not reasoned: cycle 1 rolls the provisional anchor at Step 0 and dies in `analyzing`; the next same-day cycle's Step 0 returns "anchor_already_current" and does not re-flag (provisional=False on the new trader), so the elif upgrade branch is unreachable and Step 5.5 evaluates the breach against the 3-session-stale 23830.46 -> daily_loss_pct 5.1634 -> any_breached True -> flatten_all + pause, i.e. the exact spurious-flatten hazard pass 1 blocked on, still live on the dying-cycle path this step exists for. The mirror direction is also live: with anchor 100 left provisional and today's open 110, a real 9.0909% same-day drawdown reports daily_loss 0.00% and does NOT fire (trailing 9.0909% < 10% does not cover it), where pre-85.6 fires -- a literal loosening of the daily-loss leg. Both disclosures of the residual are therefore materially false: paper_trader.py:1431-1437 says "The next cycle re-rolls at Step 0 and upgrades here, so the window is one cycle" (it does not re-roll and cannot upgrade) and experiment_results_85.6.md sec 12 repeats "nothing is judged against it in the meantime" (the next completing cycle DOES judge against it, and in my measurement it flattens the book). FAIL was considered and rejected: the tree is strictly safer than pass 1's (the hazard moved from every completing cycle to only a same-day cycle following a died cycle -- 10 of 76 days ever had >1 distinct cycle_id, only 2026-07-28 in the last three weeks), the fix's mechanism is sound and mutation-covered, and one bounded change closes it. harness_log holds ZERO phase=85.6 rows so this is verdict 2 of a possible 3 and the 3rd-CONDITIONAL auto-FAIL rule does not bind; a third CONDITIONAL must be FAIL.

## violated_criteria

- C5: no loosening of the kill switch's protective behaviour -- the provisional-anchor upgrade cannot fire after a cycle dies before Step 5.5, because the flag lives on a PaperTrader that autonomous_loop.py:407 rebuilds every cycle; both the spurious-flatten and the loosening direction remain live on that path [BLOCKING for PASS]
- Contradiction: the residual's stated bound is measurably false in BOTH artifacts -- paper_trader.py:1431-1437 'the next cycle re-rolls at Step 0 and upgrades here, so the window is one cycle' and experiment_results_85.6.md sec 12 'nothing is judged against it in the meantime' [WARN]
- illusory-guard (residual): test_c2's new self-validation is behind `if pre.returncode == 0 and pre.stdout:` -- a silent fail-open oracle that degrades to the un-validated form in any checkout where `git show 81f81750^` fails [WARN, not sole coverage]

## violation_details

### 1. Invalid_Precondition

**action**

```
PaperTrader.roll_daily_anchor() sets self._sod_anchor_provisional = True (paper_trader.py:1296); check_and_enforce_kill_switch upgrades only `elif self._sod_anchor_provisional and snap.get("sod_date") == today` (paper_trader.py:1415). autonomous_loop.py:407 constructs `trader = PaperTrader(settings, bq)` INSIDE run_daily_cycle, so the flag is per-cycle, not per-process.
```

**state**

Measured with the production check_and_enforce_kill_switch + evaluate_breach (control verified biting: anchor 23830.46 vs nav 22600 -> 5.1634% breached True; anchor 22600 vs nav 22600 -> 0.0% breached False). PROBE 1 (normal path, cycle completes): Step 0 anchor 23830.46/2026-08-08 provisional=True -> after Step 5.5 anchor=22600.0, triggered=False -- FIX CONFIRMED. PROBE 2 (residual path): cycle 1 Step 0 anchors 23830.46 provisional=True, cycle dies; a NEW PaperTrader (as at :407) runs Step 0 -> reason='anchor_already_current', provisional=False; Step 5.5 -> anchor still 23830.46, TRIGGERED=True, paused_with='limit_breach', daily_loss_pct=5.1634 (trailing 8.378% < 10% did not cause it). Pre-85.6 in the same state re-anchors to 22600 and reports 0.00%, so this is a NEW spurious flatten_all+pause path. PROBE 4b (mirror direction, coherent peak=110): cycle 1 anchors 100 and dies; cycleA at nav 110 no breach; cycleB at nav 100 -> anchor still 100, daily 0.0%, trailing 9.0909% -> TRIGGERED=False, while pre-85.6 (anchor 110) reports daily 9.0909% breached=True -- a real same-day drawdown goes uncaught.

**constraint**

masterplan 85.6 success_criteria[5]: 'no loosening of the kill switch's protective behaviour: arming a stale leg must not weaken any breach threshold'. Also kill_switch.py phase-36.9 F1 ('a TWO-DAY move reported as a same-day loss ... biases toward a spurious flatten') and paper_trader.py:1446-1454's phase-36.12 invariant, whose stated PURPOSE is that the value the breach is judged against be today's open.

### 2. Contradiction

**action**

```
experiment_results_85.6.md sec 12 'Residual, disclosed rather than hidden' and the identical prose at paper_trader.py:1431-1437 bound the exposure to one cycle and assert nothing is judged against the provisional anchor.
```

**state**

Re-derived: the next cycle's Step 0 hits `if not sod_anchor_needs_reroll(snap, today)` (sod_nav=23830.46>0, sod_date==today) and returns early with reason='anchor_already_current' WITHOUT setting the flag, so the upgrade branch at :1415 is unreachable for the remainder of that UTC day; PROBE 2 shows the next completing cycle judging the breach against the provisional anchor and flattening. The exposure window is the rest of the UTC day, and the claim 'the flag does not survive the process' understates it -- it does not survive the CYCLE (fresh PaperTrader at autonomous_loop.py:407).

**constraint**

qa.md sec 4b: every scope/quantified claim in the handoff must reproduce when re-derived; a claim whose reproducing derivation contradicts it is a Contradiction finding. Scope honesty is a graded dimension of the LLM-judgment leg.

### 3. Missing_Assumption

**action**

```
test_c2_the_409_no_longer_makes_the_two_false_promises wraps its new vacuity self-validation in `if pre.returncode == 0 and pre.stdout:` before asserting each BANNED phrase is present in `git show 81f81750^:backend/api/paper_trading.py`.
```

**state**

In MY environment the self-validation executed and passed (19 passed, and M7/M8 killed by the matrix), so BANNED[2] is demonstrably non-vacuous here -- the pass-1 illusory-guard finding is genuinely repaired. But any checkout where that git object is unreachable (shallow clone, worktree-isolation CI path, or after a history rewrite) silently skips the self-validation and reverts the guard to the un-validated form the Q/A flagged.

**constraint**

qa.md sec 4c vacuity shape 8/9 (an oracle with a silent fallback survives an absent subject; executor-environment non-reproducibility). Fix is one character class: `assert pre.returncode == 0 and pre.stdout, 'pre-fix blob unavailable -- the vacuity self-check did not run'`. WARN, not BLOCK: two live banned-phrase guards plus the verbatim live 409 body coexist.

## checks_run

- harness_compliance_audit_5_item
- research_gate_envelope_verified (tier=complex, external_sources_read_in_full=7, urls_collected=22, recency_scan_performed=true, gate_passed=true; contract_85.6.md cites research_brief_85.6 twice)
- contract_before_generate_mtime_chain (brief 22:40:13 < contract 22:43:29 < code commits 5932ac27 22:56:34 / e369b1de 23:27:01; commit order 81f81750 -> 5932ac27 -> b5f63525 -> e369b1de corroborates)
- log_last_and_status (grep -cE 'phase=85\.6' handoff/harness_log.md = 0; masterplan status=pending, retry_count=0/3)
- no_verdict_shopping (evidence CHANGED: pass 1 graded b5f63525, this pass grades e369b1de -- paper_trader.py +51, tests +150, matrix +30)
- git_derived_scope (81f81750^..HEAD UNION worktree UNION untracked, non-empty guard asserted BEFORE reading exit code: 5 .py files)
- ruff_F821_F401_F811 (exit=0, 'All checks passed!', xargs-fed derived scope -- no unquoted-variable word-split)
- ast_parse_all_5_changed_py_files (exit=0)
- immutable_verification_command (exit=0, output reproduced verbatim)
- pytest_backend_tests_test_phase_85_6_anchor_deadlock (19 passed / 19 progress dots / 13.95s -- internally consistent)
- mutation_matrix_85_6 (12/12 killed on my own run incl. M10/M11/M12; tree restoration verified: git status --porcelain backend/ scripts/ empty)
- independent_production_path_probe_probe1_normal_path (C5 fix CONFIRMED: 23830.46 -> 22600 upgrade, triggered=False)
- independent_production_path_probe_probe2_residual (spurious flatten REPRODUCED after a died cycle: 5.1634% -> triggered=True, paused_with=limit_breach)
- independent_production_path_probe_probe3_risen_book (pass-1 case (ii) CLOSED on the completing path: 2nd same-day eval fires at 5.4545%, matching pre-85.6)
- independent_production_path_probe_probe4b_loosening (residual admits an uncaught 9.0909% same-day drawdown; pre-85.6 fires)
- midday_reanchor_regression_check (spawn question c: upgrade cannot fire twice -- flag set only inside the rolled branch, cleared on use, fresh trader per cycle; probe 3 cycle 2 kept anchor 110 and fired; M12 killed)
- reachability_measurement (distinct cycle_ids per UTC day from handoff/cycle_history.jsonl: 1/day is the norm; 10 of 76 days had >1, last was 2026-07-28 -- my first row-count of '2 per day' was WRONG and is corrected here)
- backend_runtime_smoke (imports of paper_trader / autonomous_loop / api.paper_trading OK; live :8000 GET /api/paper-trading/kill-switch 200)
- live_process_vs_commit_check (uvicorn pid 23676 started 22:57:25, BEFORE the C5 fix commit at 23:27:01 -- the fix is NOT in force in the running backend)
- production_diff_review (no unintended production change: only the 5 scoped .py files + handoff docs; worktree dirt is hook-written audit JSONL and researcher agent-memory)
- code_review_heuristics (kill-switch reachability, sod-nav-anchor WARN, no secrets/eval/exec/shell=True, no threshold literal moved, no order/sizing change)
- 3rd_conditional_counter (0 logged CONDITIONALs for 85.6 -- this is verdict 2 of 3; CONDITIONAL permitted, a third must be FAIL)

## notes

SPAWN QUESTIONS ANSWERED. (a)(i) YES, the fix is real on the completing path -- I re-ran the multi-session-stale counterexample through the PRODUCTION check_and_enforce_kill_switch (not Main's arrangement) and the 23830.46 anchor was upgraded to 22600 before the breach, triggered=False. (a)(ii) The risen-book / second-same-day-evaluation case is ALSO closed on that path: after the cycle-1 upgrade the anchor is 110 and the second same-day evaluation at nav=104 reports 5.4545% and FIRES -- byte-identical to pre-85.6. I looked for a survivor there and did not find one. (b) THE RESIDUAL IS MIS-BOUNDED, and this is the blocker. The argument "such a cycle never reaches the breach decision either" is true of the DYING cycle and irrelevant to the one that matters: the NEXT same-day cycle, which does reach it. Two corrections to the disclosure text: (1) the flag does not survive the CYCLE, not the process -- autonomous_loop.py:407 builds a fresh PaperTrader inside every run_daily_cycle; (2) "the next cycle re-rolls at Step 0 and upgrades here" is false -- Step 0 short-circuits on `anchor_already_current` and never re-flags, so the upgrade branch is dead for the rest of that UTC day. Measured consequence in PROBE 2: flatten_all + pause at 5.1634% against a 3-session-old anchor, where pre-85.6 measured 0.00%. (c) CLEAN -- the upgrade cannot fire twice. The flag is set only inside roll_daily_anchor's post-needs_reroll branch, cleared immediately after the upgrade, and each cycle gets a fresh trader; probe 3's second same-day cycle kept the 110 anchor and fired the breach, so phase-36.9's mid-day re-anchor defect is NOT re-introduced (M12 independently killed on my run). (d) STATED EXPLICITLY: `grep -cE 'phase=85\.6' handoff/harness_log.md` returns 0 -- there is no logged verdict for this step, pass 1's CONDITIONAL is the only prior, so this is verdict 2 of a possible 3 and the 3rd-CONDITIONAL auto-FAIL rule does not bind. A third CONDITIONAL on 85.6 MUST be FAIL. NAMED FIXES FOR CYCLE 3 -- any ONE of (1) or (2) closes C5. (1) Make the provisional marker DURABLE instead of instance-local: record it in the kill-switch state / sod_snapshot row (e.g. `provisional: true` plus the `updated_at` of the NAV used) and have Step 5.5 upgrade on that stored fact, so it survives a died cycle and a restart. (2) Make the upgrade STATE-DERIVED rather than flag-derived: at Step 5.5, upgrade whenever the stored anchor's underlying mark predates today's UTC date, regardless of any in-memory flag -- same shape, no durability requirement, and it also self-heals a mark that went stale for any other reason. (3) Regardless of which: correct the two false sentences (paper_trader.py:1431-1437 and experiment_results_85.6.md sec 12) to say the exposure lasts the REST OF THE UTC DAY and that a subsequent completing cycle DOES judge the breach against the provisional anchor -- and add the test that drives died-cycle -> NEW PaperTrader -> completing cycle, which is the shape neither the suite nor the matrix currently covers (M10 deletes the branch but every test that would notice runs on ONE trader instance). (4) Tighten test_c2's self-validation from `if pre.returncode == 0` to an assert. OPERATIONAL FINDING, outside the criteria but material: uvicorn pid 23676 started 2026-08-08 22:57:25, BEFORE the C5 fix commit at 23:27:01 -- the running backend is executing the CYCLE-1 code, in which the hazard is live on EVERY completing cycle, and the book is now UNPAUSED. Restart the backend before the next cycle, or record that the next cycle runs pre-fix code. Reachability, measured and self-corrected: my first count read cycle_history rows and reported "2 cycles/day"; the rows are `started` + terminal for the SAME cycle_id, so the true figure is 1 distinct cycle per UTC day, with >1 on only 10 of 76 days (last: 2026-07-28). The residual therefore needs an operator-triggered or extra same-day cycle -- uncommon, but it is exactly what the new 409 body tells the operator to do ("trigger a cycle") and what Main did tonight, and it also needs a >4% multi-session move (tonight's was -0.0146%). POSITIVE, unclaimed by the handoff: the cycle-2 upgrade restores the pre-85.6 breach VALUE while keeping the anchor present from t+2s, so C1's unblock and C5's arithmetic are no longer in tension on the completing path -- that is the right design, it just needs to survive a cycle boundary. SAFETY: I triggered no cycle, restarted no service, never paused or resumed the switch, and never called the real flatten_all (stubbed in every probe); all probes ran in my own process against injected fake state, so handoff/kill_switch_audit.jsonl grew by 0 rows; the mutation matrix restored the tree byte-for-byte (verified clean).

---

## Follow-up — Main's cycle-3 remediation (written by Main, NOT by the Q/A)

| Finding | Severity | Action |
|---|---|---|
| C5 BLOCKING: the provisional flag lived on a `PaperTrader` that `autonomous_loop.py:407` rebuilds every cycle, so after a died cycle the upgrade branch was unreachable for the rest of the UTC day — spurious flatten AND the mirror loosening both live | BLOCKING | **Fixed by making the marker DURABLE.** `sod_provisional` is now state on `KillSwitchState`, written into the `sod_snapshot` audit row and replayed on boot, so it survives a dead cycle, a new trader, and a restart. The upgrade condition reads `snap["sod_provisional"]`, not an instance flag. |
| Contradiction: the residual's stated bound was measurably FALSE in both the code comment and §12 | WARN | **Fixed.** Both texts replaced with the measured truth, including the Q/A's own numbers (5.1634% on a 4% limit). |
| Illusory guard (residual): the vacuity self-check was behind `if pre.returncode == 0`, a silent fail-open | WARN | **Fixed.** It is now an `assert`, so an unreachable pre-fix blob fails the test instead of skipping the check. |

Re-verified:

```
pytest test_phase_85_6_anchor_deadlock.py                      25 passed (was 19)
all kill-switch + book-safety suites + 85.6                   202 passed, 1 failed
    (the 1 failure is test_book_safety_69::test_valid_nav_still_breaches --
     PRE-EXISTING, one of the known 26, and the subject of step 85.5.1)
scripts/qa/mutation_matrix_85_6.py                            14/14 killed (was 12/12)
uvx ruff check --select F821,F401,F811 (5-file derived scope)  exit 0
```

**M13 and M14 were LIVE on their first run**, for the reason auto-memory warns
about: my tests used `FakeKillSwitchState`, which mirrors the marker itself, so
mutating the REAL persistence/replay changed nothing they could see. Four new
tests now drive the real `KillSwitchState` with its audit journal redirected to
tmp — persistence, replay-after-restart, clearing, and legacy-row compatibility.

**One pre-existing test was deliberately changed, and it is disclosed here
rather than buried:** `test_phase_36_9_kill_switch_armed_liveness.py::
test_phase_36_9_the_resume_409_names_staleness_not_absence` asserted the 409
message *contains* `"NO operator action is required"` — the exact phrase 85.6
criterion 2 requires REMOVED, because it was measured false and the book sat
unresumable for six days behind it. The assertion is inverted with the
supersession explained inline, and the original INTENT (the message must
describe the daily roll that actually clears the refusal, not the lost-history
block) is preserved and still enforced. No other pre-existing test was touched.

---

# EVALUATE pass 3 — verdict **PASS**

`ok: True` · `harness_compliance_ok: True` ·
violated_criteria: **none**

Run `wf_c1ca165d-04e`, commit `a126126b`. Spawned under the 3rd-CONDITIONAL rule
with an explicit instruction to return PASS or FAIL, never a third CONDITIONAL.
Verbatim below. Machine copy: `evaluator_critique_85.6_pass3.json`.

## reason

All 6 immutable criteria MET, verified by independent execution rather than inspection. C1: the roll moved to Step 0 (backend/services/autonomous_loop.py:536, 3 non-comment lines) and a real triggered cycle rolled the anchor at t+2s (live_check §3, log line 20:58:29Z); mutations M1 (Step-0 roll deleted) and M2 (roll drifts back behind screening) both KILLED in my own run of scripts/qa/mutation_matrix_85_6.py. C2: I captured the live post-fix 409 and the phrase "NO operator action is required" is absent, "UNBLOCK CONDITION" present and it names the real mechanism (roll_daily_anchor -> update_sod_nav) INCLUDING the honest weekend/no-cycle-scheduled case; M7/M8 KILLED. C3: argued from three evidence items, the load-bearing one being the measured 2026-08-05 completing cycle that traded nothing because the switch was paused - so "85.4 alone clears it" is refuted, not assumed. C4: I re-ran the immutable verification command myself: exit=0, {'paused': False, 'sod_date': '2026-08-08'} {'armed': True, 'daily_baseline_stale': False}. C5 (the criterion that blocked passes 1 and 2): CLOSED, and I proved it with my own probes against the REAL KillSwitchState + REAL PaperTrader, journal redirected to tmp. PROBE 2 re-run: cycle 1 rolls provisional (sod_provisional=true, 23830.46 stamped today) and dies; a BRAND-NEW trader's Step 0 returns "anchor_already_current" and Step 5.5 now upgrades -> sod_nav 22600.0, provisional false, triggered false, daily_pct 0.0, no flatten. Same result across a simulated PROCESS RESTART (state rebuilt by journal replay). CONTROL (marker force-cleared to simulate the pass-2 defect) reproduces the spurious flatten exactly - triggered true, flatten_all called, daily_loss_pct 5.1634 - so my probe is capable of failing and the author's cited number reproduces under my independent construction. PROBE 4b mirror direction: a real same-day 6% drawdown against a confirmed anchor still fires (daily_loss_breached true, flatten+pause, paused true). A legacy sod_snapshot row with no `provisional` key replays as False and is never upgraded mid-day. C6: the whole-step diff contains zero matches for execute_buy/execute_sell/position_size/max_positions/stop_loss/daily_loss_limit_pct/trailing_dd_limit_pct. Deterministic tier: ruff F821/F401/F811 over a git-DERIVED 7-file scope (git diff 81f81750..HEAD union untracked union working tree) exit=0, non-empty set asserted; 25/25 on backend/tests/test_phase_85_6_anchor_deadlock.py; regression 1 failed / 206 passed / 1 skipped over the explicit kill-switch+book-safety file list and 1 failed / 194 passed under an independent -k selection - the single failure is test_book_safety_69::test_valid_nav_still_breaches and I verified the pre-existing attribution independently (evaluate_breach appears 0 times in the entire 85.6 kill_switch.py diff, and masterplan 85.5.1 records the same failure reproduced at cf4d22d8 in a detached worktree); mutation matrix 14/14 KILLED with git status --short over backend/ scripts/ frontend/ empty afterwards (tree restored). Harness compliance clean: research gate passed (7 read in full, 22 URLs, recency scan present, contract cites it 3x), contract precedes GENERATE by commit order (81f81750 "research gate + contract (GENERATE not started)" -> 5932ac27), experiment_results present, log-last respected (0 "phase=85.6" entries in harness_log), and the evidence CHANGED since the tree pass 2 graded (0f9e92c0 + a126126b after e369b1de) so this is the documented fresh-respawn, not verdict-shopping. On the specific scrutiny asked: the inverted pre-existing 36.9 assertion is a LEGITIMATE criterion-mandated supersession, not a test edited to go green - criterion 2 requires the exact phrase removed, the replacement is strictly stronger (negative assertion + mechanism assertion), it was disclosed prominently, and M7 "the false 409 promise comes back" is KILLED. The class-level default is the right call: it mirrors the documented _baseline_provenance precedent in the same file, production always runs __init__, and the default is conservative (False = pre-85.6 inert). No new hazard from persisting the marker: only Step 0 sets provisional=True and only on a stale date, the upgrade clears it, and a second/third same-day cycle does not re-anchor (measured: anchor held at 10000.0 while the book fell to 9400, daily_pct 6.0 computed against the held anchor).

## checks_run

- qa_md_read_from_disk
- harness_compliance_audit_5_item
- research_gate_envelope
- contract_before_generate_commit_order
- log_last_not_yet_logged
- no_verdict_shopping_evidence_changed
- python_lint_gate_1a_derived_scope
- syntax
- verification_command
- scoped_pytest_85_6_suite
- regression_pytest_killswitch_booksafety_two_independent_scopes
- pre_existing_failure_attribution_reverified
- author_mutation_matrix_14_of_14
- tree_restored_after_mutation_matrix
- independent_probe_cross_cycle_real_state
- independent_probe_cross_restart_replay
- independent_probe_CONTROL_defect_reproduces
- independent_probe_real_drawdown_still_fires
- independent_probe_legacy_row_backward_compat
- independent_probe_no_mid_day_reanchor
- backend_runtime_smoke_live_endpoint
- live_check_artifact_review
- claim_audit_4b
- guard_vacuity_4c
- adversarial_three_lenses
- code_review_heuristics
- evaluator_critique

## notes

Four NOTE-level findings, none blocking, all recorded rather than waived. (1) UNDISCLOSED LIVE-STATE RESIDUAL, found by me: the live sod_snapshot row written by the verification cycle (handoff/kill_switch_audit.jsonl, ts 2026-08-08T20:58:29.379594Z) has NO `provisional` key, because the live proof ran on the cycle-1 backend (pid 23676) that predates the marker. So the CURRENT live anchor (23830.46 stamped 2026-08-08) is provisional in fact but replays as confirmed, and a cycle running before 00:00Z on 08-08 would judge the breach against it without upgrading. Exposure is near-nil (weekday-only cron, Saturday, the stored mark IS 23830.46 so the measured move is ~0%, and the failure direction is over-protective not permissive) and it self-clears at the UTC rollover, but it belongs in experiment_results §13 rather than only in this critique. (2) CLAIM HYGIENE: "All kill-switch + book-safety suites: 202 passed, 1 failed" (experiment_results_85.6.md:384) does not reproduce - I measured 1 failed/206 passed/1 skipped over the explicit 10-file list and 1 failed/194 passed under `-k "kill_switch or book_safety"`. No reproducing command was recorded, so the figure is unreproducible by construction rather than contradicted; the load-bearing half (exactly one failure, pre-existing, book_safety_69) reproduced under BOTH independent scopes. Record the exact pytest invocation next to the number. (3) A stale PRE-EXISTING comment survives directly above the corrected 409 in backend/api/paper_trading.py (~:596-612): "the daily start-of-day roll sets today's anchor at the top of the next cycle, with no operator action at all" and "this refusal self-clears within one cycle and cannot wedge". 85.6 did not author it and criterion 2 governs the MESSAGE, which is correct - but the comment now contradicts the message's own weekend disclosure and is the exact prose class this step exists to kill. (4) Minor internal inconsistencies in the handoff: research_brief_85.6.md:6 says "6 sources read in full" while its own envelope says 7 (both clear the >=5 floor), and experiment_results §7 is still headed "Mutation matrix - 9/9" with §12/§13 superseding it to 14/14. Weak-but-not-sole guard: `assert "roll" in detail` in the superseded 36.9 test is satisfiable by substrings such as "controlled"; it is backed by the "UNBLOCK CONDITION" assertion and by mutation M8, so it is not sole coverage. DISCLOSURE ABOUT MY OWN RUN: my CONTROL probe drove the production breach path on an INJECTED state and therefore emitted a real Slack alert at ~2026-08-08T21:5xZ titled "Kill-switch AUTO-PAUSED trading (trigger=limit_breach)" - that alert is a Q/A artifact, not a live event; no live book state changed. My probes wrote only to a tmp journal (verified: the only 2 uncommitted lines in handoff/kill_switch_audit.jsonl are timestamped 20:58Z and predate my run), I started/killed no server, triggered no cycle, and touched no production file; the mutation matrix restored the tree byte-for-byte (git status --short over backend/ scripts/ frontend/ empty). All UI gates N/A (no frontend/** in the diff, no UI claims).
