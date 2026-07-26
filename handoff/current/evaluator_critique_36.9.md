# Evaluator critique -- masterplan step 36.9

Written every cycle from the start, per the lesson 36.8 paid for: six of its verdicts existed
only inside my own spawn prompts until a Q/A caught it, which defeats the gate's purpose.

## Cycle 1 -- FAIL (verbatim, transcribed, not authored, not edited)

**verdict:** `FAIL`   **ok:** `False`   **harness_compliance_ok:** `True`

### reason

> Immutable command reproduces (162 passed, 1 skipped, exit 0) and criteria 1-5 as literally worded are each MET, but the change ships a NEW, undisclosed money-path regression on a fourth consumer that no guard in the 162-test suite can see: F1's staleness now flows into paper_trader.check_and_enforce_kill_switch:1121/1126, which pre-measures `armed` BEFORE the daily roll at :1167, so the ORDINARY first cycle of every UTC day (sod present+positive, peak present, benign NAV) yields pre_armed=False and trips 36.12's block at :1220 -- blocked=True, block_reason=kill_switch_disarmed_lost_history, a P1 page, a fabricated lost_history_anchor row into the LIVE audit trail, and cycle_halt_reason() halts the loop. Executed end-to-end with a discriminating CONTROL (fresh anchor -> blocked=False) and a pre-36.9 baseline (same book, staleness neutralised -> blocked=False), so the defect is NEW, not pre-existing. experiment_results claim "The autonomous cycle is unaffected because it re-anchors first" is measured FALSE.

### violated_criteria

1. consumer-contract-break [BLOCK] -- evaluate_breach `armed` semantics changed without verifying the order-placing consumer
2. scope-honesty -- 'the autonomous cycle is unaffected because it re-anchors first' does not reproduce
3. guard population has zero coverage of the order-placing path for the new predicate
4. claim audit -- '12 _sod_date assignments across 3 files' does not re-derive

### violation_details

**1. Contradiction**

- *action:* trader.check_and_enforce_kill_switch() on a healthy funded book: sod_nav=23838.19 dated YESTERDAY, peak_nav=24666.57, nav=23800.00, starting_capital=20000.00 (venv, _AUDIT_PATH redirected to tmp, pager captured)
- *state:* POST-36.9 stale(yesterday) anchor -> pre_armed=False blocked=True reason=kill_switch_disarmed_lost_history alerts=[('P1','disarmed_lost_history_block')] | CONTROL fresh(today) anchor -> pre_armed=True blocked=False alerts=[] | PRE-36.9 baseline (_sod_date_is_stale neutralised), same yesterday anchor -> pre_armed=True blocked=False. cycle_halt_reason(blocked cycle) = 'kill_switch_disarmed_lost_history'. Live audit md5 ce8fb93348bb9a3bbe26f2d91b1bc05e UNCHANGED.
- *constraint:* experiment_results_36.9.md:16 'The autonomous cycle is unaffected because it re-anchors first' and contract_36.9.md:40 which cites 36.12's per-cycle order block as the COMPENSATING measure. paper_trader.py:1121 computes `pre` BEFORE the roll at :1167 by 36.12's explicit design ('MEASURE THE ARMED STATE BEFORE MUTATING THE BASELINES'), so a pre-roll anchor is stale by construction on the first cycle of every UTC day and :1220 `if not pre_armed and not first_ever_boot` fires on a book that never lost history.

**2. Invalid_Precondition**

- *action:* grep -rn 'evaluate_breach|pre_armed' backend/ for consumers of the changed `armed` key
- *state:* Live consumers of `armed`: paper_trader.py:1126 (pre_armed -> :1220 order block, :1187 flatten precedence), the 36.7 resume-409 gate, check_auto_resume. Only the read paths (GET /kill-switch, POST /resume, MCP tool) are analysed in the contract and results; :1126 is never mentioned in contract_36.9.md, experiment_results_36.9.md or live_check_36.9.md, and the 36.9 test module never calls check_and_enforce_kill_switch (only unit-tests sod_anchor_needs_reroll at :247/:265/:273).
- *constraint:* code-review heuristic #16 consumer-contract-break [WARN->BLOCK when a live unverified consumer is found]: a change to a returned dict value's SEMANTICS must grep-verify every consumer in the same diff. Escalated to BLOCK: the unverified consumer is the order-placing money path.

**3. Missing_Assumption**

- *action:* Re-run of the guard population against the new failure mode: 162-test immutable suite + the 24-test 36.9 module + the author's 13-mutant matrix
- *state:* All green while the regression is live. The matrix is real and well-built (13/13, and two first-run survivors were honestly fixed rather than explained away), but every mutant is scoped inside the 24-test module, which contains no test that executes the order-placing cycle. 36.12's own tests stay green because their scenario is sod_date=None (lost history), not a stale-but-present anchor.
- *constraint:* qa.md 4c -- name the concrete mutation that makes the guard fail. For 'a stale anchor must not block an ordinary overnight cycle' NO mutation of the current suite can fail, because no guard observes that path. A matrix licenses only 'these 13 mutations were killed', never a safety claim over an uncovered consumer.

**4. Overgeneralization**

- *action:* git grep -n "_sod_date = ['\"]20" HEAD -- backend/tests tests (re-derivation of the fixture-rot census)
- *state:* 11 literal assignments across 2 files (4x 2026-05-22, 1x 2026-07-24, 6x 2026-07-26). The claimed 12th, '2026-04-20', is not an assignment: it is prose inside a docstring at tests/services/test_sod_daily_roll.py:142 ('boot replay sees the legacy 04-20 row -> _sod_date=...'), and it is the only reason the third file appears in the count.
- *constraint:* experiment_results_36.9.md:83 'grep over every _sod_date assignment in tests: 12 across 3 files'. Non-blocking on its own (the 10 remediations themselves re-derive exactly: 4 in 23_2_5 + 6 in 36_7), but a set-membership claim in a 'mechanically enumerated, not hand-picked' section must reproduce.

### checks_run

- `harness_compliance_audit_5_items`
- `immutable_verification_command`
- `python_lint_gate_git_derived_scope`
- `backend_runtime_smoke_live_execution`
- `code_review_heuristics`
- `claim_audit_re_derivation`
- `guard_vacuity_and_independent_mutation_probe`
- `amended_guard_review`
- `threshold_immutability_diff`
- `do_no_harm_md5`

### notes

> HARNESS COMPLIANCE: clean 5/5. research_brief_36.9.md gate_passed:true (11 sources read in full, 42 URLs, recency scan performed) mtime 19:50:53 < contract_36.9.md 19:53:53 < kill_switch.py 19:58:40 < test module 20:03:16 < experiment_results 20:07:07, so research preceded contract and contract preceded generate. `grep -c "phase=36.9" handoff/harness_log.md` = 0 and masterplan 36.9 status=pending, so log-last is intact. Cycle 1, no prior verdict, no verdict-shopping surface; 3rd-CONDITIONAL rule N/A (0 prior CONDITIONALs) -- this FAIL is on merits.
> 
> DETERMINISTIC, VERBATIM: immutable `python -m pytest backend/tests/ -q -k kill_switch` -> "162 passed, 1 skipped, 2126 deselected, 1 warning in 22.01s" (no failures; matches the claimed 162). Lint scope DERIVED (`git diff --name-only HEAD -- '*.py'` + `git ls-files --others --exclude-standard -- '*.py'`), asserted non-empty, count=5, passed as an array (zsh word-split trap avoided): `uvx ruff check --select F821,F401,F811` -> "All checks passed!" RUFF_EXIT=0. No frontend file in the derived scope, so gates 1b/1c are not triggered (the cockpit-badge language in the masterplan `name` is motivation prose, not an immutable criterion; no UI capture taken and none required). Runtime smoke exceeded the import bar: I executed the real `check_and_enforce_kill_switch` in the venv. DO-NO-HARM honoured: `_AUDIT_PATH` redirected to a tmp dir BEFORE any write path, `raise_cron_alert_sync` stubbed so no false P1 reached the operator's Slack, :8000 never touched, :3000 never driven, handoff/kill_switch_audit.jsonl md5 ce8fb93348bb9a3bbe26f2d91b1bc05e verified unchanged after the probe, zero repo writes (the qa-write-guard hook correctly blocked my one attempt to place a probe file in the tree; I re-ran it through stdin instead).
> 
> THE FOUR THINGS YOU ASKED ME TO ATTACK:
> (1) THE AMENDED 36.7 GUARD -- CORRECTLY CORRECTED, NOT WEAKENED. `test_..._baseline_restored_from_rotated_v4_file` restores a row dated 2026-07-24 and used to assert armed True; that assertion encoded the defect. The amendment keeps 36.7's own subject (restoration assertions untouched), keeps the date-independent trailing leg firing (`trailing_dd_breached is True`, `any_breached is True`), adds `daily_loss_pct == 0.0`, and adds a fresh-date sub-case that re-asserts 36.7's original three-way guarantee verbatim. Nothing 36.7 guaranteed was removed. The 10 literal-date -> TODAY_UTC substitutions are genuine fixture-rot repairs (they would have gone red tomorrow), and the audit-row payload literals were correctly left alone.
> (2) IS PROTECTION REDUCED ON THE READ PATHS -- NO, YOUR ARGUMENT HOLDS AND I AM NOT FAILING ON IT. A two-day-old anchor cannot measure a same-day loss; computing a percentage from it is a diagnostic failure biased toward a spurious flatten. The trailing leg is a high-water mark and is untouched, and /resume becoming stricter is the fail-safe direction. Withdrawing a false claim is not losing coverage. My FAIL is NOT this argument -- it is the fourth consumer you did not analyse.
> (3) UNPROVABLE DATE => STALE -- BOUNDED, NO HEALTHY PRODUCTION PATH CAN BE DISARMED BY IT. `_append_audit` (kill_switch.py:412-417) unconditionally stamps `ts` on every row, and `_load_from_audit` falls back to parsing `ts` when `date` is absent, so a production-written sod_snapshot always yields a resolvable date. sod_nav-set + sod_date-None is reachable only from a hand-edited or ts-less row. Your conservative reading is sound.
> (4) VACUITY, THIRD INSTANCE OF THE CLASS -- FOUND, AND IT IS AT THE SUITE LEVEL RATHER THAN INSIDE A SINGLE TEST. Your two in-test instances were real and honestly fixed. The third is that the entire guard population for the new predicate lives inside a module that never executes the order-placing path, so the highest-consequence consumer of `armed` has no guard at all; 13/13 killed is true and licenses nothing about that path. That is why I built an independent probe instead of re-running your matrix, and the probe found the regression the matrix structurally could not.
> 
> VERIFIED-TRUE CLAIMS (so they are not re-done): the healthy-path test really does compare the WHOLE dict (`assert r == {...}`, 11 keys, fixed numbers) and my CONTROL run independently confirms a fresh anchor is unaffected -- criterion 4 as worded is MET. NO threshold moved: the production diff contains no changed numeric limit (the only `4.0` and `10.0` tokens are inside comments); daily/trailing limits, DSR, PBO, sector caps and stops are byte-untouched. The pre-fix reproductions for F1/F2/F3 are the right shape (each fix reverted individually, only that finding's tests run). The zsh word-split gate defect you caught in your own tooling is real and your remediation is the correct one -- I hit the same trap in my own lint run and used an array for the same reason.
> 
> ACTIONABLE FIX DIRECTION (this is a regression OF this change, so it belongs in this step, not a queued follow-up): 36.12's block must discriminate "baselines LOST" from "anchor STALE but present" -- the provenance defect it was built for is absence, not staleness. Gate :1220 on the pre-roll ABSENCE flags (`pre["daily_baseline_missing"] or pre["trailing_baseline_missing"]`), or equivalently on `pre_armed or pre["daily_baseline_stale"]`, rather than on `armed` alone; and add a regression test that drives the real `check_and_enforce_kill_switch` with a yesterday-dated present anchor and asserts blocked is False, plus a control asserting a genuinely lost-history book still blocks. Re-check the same pre-roll question for the 36.7 resume-409 gate and `check_auto_resume`: a book paused overnight now reads disarmed until the next cycle rolls the anchor, which is fail-safe but should be a stated, tested decision rather than an emergent one.

## Main's response to cycle 1

The FAIL was correct and I reproduced it independently before changing anything (stale anchor ->
`blocked=True`, `kill_switch_disarmed_lost_history`, P1 raised; fresh anchor -> `blocked=False`).
Fixed by splitting `baselines_present` (presence -- what the 36.12 order gate needs) from `armed`
(can-fire-now -- what the read surfaces need), guarded by 4 tests that execute the real cycle path
and 2 new mutants. The false claim "the autonomous cycle is unaffected because it re-anchors first"
is withdrawn in place in `experiment_results_36.9.md`, not deleted. The census claim was corrected:
11 real assignments across 2 files; the 12th grep hit is docstring prose.


---

## Cycle 2 -- CONDITIONAL (verbatim, transcribed, not authored, not edited)

**verdict:** `CONDITIONAL`   **ok:** `False`   **harness_compliance_ok:** `True`

### reason

> All 5 immutable criteria MET with executed evidence, and the C1 regression is genuinely closed — I independently mutated the fix from both sides and both mutants died on order-path tests (M2 "order gate reads `armed` again" KILLED by test_phase_36_9_an_overnight_anchor_does_not_halt_the_morning_cycle; M1 "refold staleness into baselines_present" KILLED by 2 tests). 6/6 of my own mutants killed, control 28 passed, audit md5 unchanged. Immutable command reproduces exactly (166 passed, 1 skipped, 2126 deselected). Held at CONDITIONAL, not PASS, on two operator-facing residuals the record does not disclose, both WARN-severity: (F-A) the criterion-mandated strict `armed` makes the cockpit badge read DISARMED with the Resume button disabled on a HEALTHY book every day from 00:00 UTC until that day's first cycle rolls the anchor — KillSwitchPanel.tsx:137 and OpsStatusBar.tsx:318 both key off `breach.armed === false`, and NO frontend reads the new `daily_baseline_stale`, so the UI still conflates "baselines lost" with "anchor is yesterday's" (the exact conflation the fix removed on the money path); (F-B) backend/api/paper_trading.py:598-604 still tells the operator "the loss baselines could not be restored (daily_baseline_missing=False, trailing_baseline_missing=False)" — for the new staleness cause it names a cause that is not the cause and contradicts its own printed diagnostics, the same class Main correctly fixed in the disarm LOG this cycle. Neither is a criterion miss and neither is a wedge (verified: a PAUSED book still reaches the daily roll, so staleness self-clears in one cycle). Not the 3rd CONDITIONAL: grep -cF "phase=36.9" handoff/harness_log.md = 0.

### violated_criteria

1. Missing_Assumption: undisclosed daily DISARMED badge + disabled Resume on a healthy book (scope honesty)
2. Contradiction: POST /resume 409 text misnames the staleness cause and contradicts its own diagnostics

### violation_details

**1. Missing_Assumption**

- *action:* evaluate_breach folds daily_baseline_stale into strict `armed` (kill_switch.py:783) and both frontend consumers derive disarmed = breach.armed === false
- *state:* WARN severity. KillSwitchPanel.tsx:137 and OpsStatusBar.tsx:318 are the ONLY frontend reads of the breach dict (grep-verified); daily_baseline_stale now ships in both return shapes but no frontend reads it. On a healthy funded book the badge therefore renders DISARMED with alarm styling, and the Resume button is disabled (KillSwitchPanel.tsx:219, OpsStatusBar.tsx:368), every day from 00:00 UTC until that day's first autonomous cycle rolls the anchor at paper_trader.py:1177. experiment_results_36.9.md behaviour-item 1 states the flag change but never states this daily-cadence operator-visible consequence; no live UI capture was taken this cycle.
- *constraint:* Scope-honesty lens (qa.md 4a) + operator memory feedback_gate_scope_and_disclosure_completeness: naming every changed file is not describing every changed behaviour. Remediation: disclose the overnight DISARMED window in experiment_results, and queue a research-gated step to have the badge read daily_baseline_stale so it renders a distinct, self-repairing state instead of the durable-fault DISARMED (per feedback_queue_discovered_defects_in_masterplan, a queued step, not prose).

**2. Contradiction**

- *action:* POST /api/paper-trading/resume refuses on the new staleness cause via the unchanged phase-36.7 gate at backend/api/paper_trading.py:598
- *state:* WARN severity. The 409 body still reads 'kill switch is DISARMED -- the loss baselines could not be restored, so neither limit can be verified healthy (daily_baseline_missing=..., trailing_baseline_missing=...)'. For a merely-stale anchor the baselines WERE restored and both interpolated fields print False, so the message asserts a cause its own diagnostics refute, and the remediation sentence points at the lost-history block rather than the daily roll that actually clears staleness. This is the identical defect class Main fixed in the disarm log this cycle (behaviour item 6: a line that 'reported a number it had just called missing'), left unfixed on the operator-facing surface. NOT a wedge: verified check_and_enforce_kill_switch has no is_paused() short-circuit before the roll (first is_paused at paper_trader.py:1197, roll at :1177) and autonomous_loop.py:1314 calls it BEFORE cycle_halt_reason at :1316, so a paused book still rolls and staleness self-clears within one cycle.
- *constraint:* Criterion 1 requires /resume to refuse on a stale anchor (satisfied) but the 36.7 remediation text was not updated for the new cause; the step's own stated doctrine is that a disarm message must name the ACTUAL cause. Remediation: branch the 409 on daily_baseline_stale with accurate text, or queue it as its own step.

### checks_run

- `harness_compliance_audit_5of5`
- `immutable_verification_command`
- `scoped_pytest_module_28`
- `scoped_pytest_paper_trader_autonomous_sod_57`
- `python_lint_gate_ruff_F821_F401_F811_derived_scope`
- `lint_untracked_file_gap_closed`
- `syntax_ast_parse`
- `backend_runtime_import_smoke`
- `independent_mutation_matrix_6_mutants`
- `guard_vacuity_check_fixture_and_pager_capture`
- `consumer_census_armed_baselines_present_daily_baseline_stale`
- `lifecycle_read_point_analysis_paused_book_wedge`
- `claim_audit_reproduce_every_number`
- `code_review_heuristics`
- `evaluator_critique_cycle1_diff`
- `do_no_harm_audit_md5`

### notes

> HARNESS COMPLIANCE 5/5. research_brief_36.9.md gate_passed:true (11 sources read in full, 42 URLs, recency scan true) mtime 19:50:53 < contract_36.9.md 19:53:53 < kill_switch.py 20:17:44 / paper_trader.py 20:19:18 / test module 20:18:41 < experiment_results 20:23:55. grep -cF "phase=36.9" handoff/harness_log.md = 0 and masterplan 36.9 status=pending, so log-last intact. NOT verdict-shopping: production code, tests (24->28) and matrix (13->15) all changed between C1 and C2 — the documented cycle-2 flow. 3rd-CONDITIONAL rule N/A (0 prior CONDITIONALs; C1 was FAIL). FAIL->CONDITIONAL on changed evidence is the honest gradient, not sycophancy.
> 
> DETERMINISTIC (all reproduced by me, verbatim): immutable `pytest backend/tests/ -q -k kill_switch` = "166 passed, 1 skipped, 2126 deselected" exit 0; module = "28 passed" (28 progress dots, internally consistent); "paper_trader or autonomous or sod" = "57 passed". ruff F821,F401,F811 on the git-DERIVED scope (4 tracked .py, non-empty asserted) = "All checks passed!" exit 0. No frontend file changed, so 1b not triggered. No threshold/limit literal appears anywhere in the diff (grep empty). Production changes confined to kill_switch.py + paper_trader.py. Import smoke OK; both new helpers present.
> 
> INDEPENDENT MUTATION (mine, not Main's — in-memory sys.modules injection, _AUDIT_PATH rewritten to tmp BEFORE the singleton was built, zero repo writes, no restore needed). CONTROL 28 passed. M1 refold staleness into baselines_present -> KILLED (overnight-cycle + two-questions tests). M2 order gate reads `armed` again, i.e. the exact C1 regression -> KILLED (overnight-cycle test). M3 staleness detector off -> KILLED (10). M4 reroll predicate `is None` only -> KILLED (1). M5 nav_invalid `armed` back to pre-36.9 -> KILLED (4). M6 update_sod_nav accepts 0.0 -> KILLED (5). handoff/kill_switch_audit.jsonl md5 ce8fb93348bb9a3bbe26f2d91b1bc05e BEFORE and AFTER — unchanged. :8000 never touched, :3000 never driven, no POST, no restart.
> 
> THE SPLIT IS REAL, NOT COSMETIC. baselines_present = not (daily_baseline_missing or trailing_baseline_missing) is byte-identical to the pre-36.9 `armed` expression, and it is emitted in BOTH return shapes — including the nav_invalid early return, where pre-36.9 the gate also received presence-only, so paper_trader:1136 behaves identically there too. M2 proves the gate is genuinely re-pointed.
> 
> GUARD VACUITY (qa.md 4c) — the cycle-2 guards are NOT vacuous. The cycle_probe fixture builds a real PaperTrader (object.__new__ + real settings + real kill_switch state) and calls the REAL check_and_enforce_kill_switch; raise_cron_alert_sync is CAPTURED on backend.services.observability.alerting, which is exactly where paper_trader.py:1246 resolves it at call time via a function-local import, so the monkeypatch genuinely binds. The capture is self-validating: test_phase_36_9_genuine_lost_history_still_blocks asserts len(alerts) == 1, proving the list records when a P1 fires — so the `alerts == []` assertion in the regression test cannot be silently inert (shape #5 closed). C1's sharpest point ("no mutation of my suite could have caught this — every mutant lived in a module that never executed the order path") is genuinely closed: M2 mutates paper_trader and dies on an order-path test.
> 
> CLAIM AUDIT — every load-bearing number re-derived, not read. 166/28/57 reproduce. Matrix file shows "15 killed, 0 survived, of 15 mutants" and every KILLED row reads "1 failed, 27 passed" = baseline 28, internally consistent. The corrected census REPRODUCES EXACTLY: git diff shows 7 date literals REMOVED / 1 added in 36_7 and 4 removed / 1 added in 23_2_5 (the added one is the deliberate STALE_SOD_DATE = "2026-05-22" in each file, correctly literal because staleness is its subject) — so 7 + 4 = 11 across 2 files is true, and the naive net-occurrence deltas (6 and 3) are fully explained. The claimed 12th hit at tests/services/test_sod_daily_roll.py:142 IS docstring prose (read it: inside the triple-quoted docstring of test_legacy_row_then_new_day_rolls_correctly). The cycle-2 withdrawal of "the autonomous cycle is unaffected because it re-anchors first" is correct and is withdrawn IN PLACE, not deleted.
> 
> F-C [NOTE, non-blocking]: the parenthetical "the file now greps 8" (What-shipped row, 36_7) does not reproduce under either instrument I tried — the file now carries 11 ISO date literals and 9 TODAY_UTC/YESTERDAY_UTC/TWO_DAYS_AGO_UTC occurrences — and no reproducing command is recorded. Every load-bearing number around it reproduces; fix by recording the command or dropping the parenthetical.
> 
> F-D [NOTE, gate scope, applies to qa.md itself]: the qa.md-prescribed lint derivation `git diff --name-only HEAD -- '*.py'` returns 4 files and MISSES the new UNTRACKED test module. Main's 5-file scope was the correct superset. I closed the gap myself (git ls-files -o --exclude-standard; the untracked file lints clean, exit 0). Recommend folding untracked-file discovery into the qa.md 1a derivation.
> 
> CRITERION MAP: (1) MET — _sod_date_is_stale gates the daily math and armed on the shared evaluate_breach, which is what GET /kill-switch, POST /resume (paper_trading.py:586) and the MCP risk_server (risk_server.py:80) all call; M3 kills 10 tests, pre-fix reproduction captured. (2) MET — nav_invalid now returns armed:False + nav_invalid_disarmed; M5 kills 4. (3) MET — update_sod_nav refuses non-positive/non-finite (M6 kills 5) AND the extracted sod_anchor_needs_reroll treats <=0 as absent (M4 kills 1), so the 409's "the next cycle re-anchors" promise is true at the root. (4) MET — whole-dict healthy-path test against fixed fixtures passes at control; the two NEW keys are additive (all pre-existing key VALUES unchanged) and the change was disclosed, not slipped in — the whole-dict guard fired and was honestly updated. (5) MET — and independently re-executed above.
> 
> NO LIVE UI CAPTURE was taken by me this cycle: the diff touches no frontend file, so 1c is not triggered by the diff, and the badge consequence in F-A is a code-path inference from the grep-verified frontend read sites plus the payload change, not a capture. If the F-A remediation changes any frontend file, that step needs a live Playwright capture.

## Main's response to cycle 2

Both residuals were real and both are now closed, by different routes:

- **F-B (the /resume 409 text) FIXED HERE.** It is in-scope: criterion 3 requires the 409 remediation
  to be TRUE, and the step's own doctrine is that a disarm message must name the ACTUAL cause -- the
  same defect I had already fixed in the disarm log this cycle. A staleness-specific branch now names
  the offending date, states that the baselines are intact, and describes the daily roll that
  actually clears it. Mutation-proved: removing the branch fails the new test.
- **F-A (the cockpit badge) QUEUED, NOT FIXED.** It is a frontend behaviour change whose evidence
  must come from the isolated :3100 rig, and it is out of this step's authorized scope. Filed as
  masterplan step **36.20**, written for an executor with no memory of the discovery -- including the
  trap that a third state must NOT be encoded as `armed: undefined`, because both backend gates use
  `.get("armed", True)` and would fail OPEN. Disclosed in `experiment_results_36.9.md` as a known
  operator-visible consequence, per the Q/A's scope-honesty finding.

**A third defect surfaced while verifying, also queued: 36.21.** Running the wider selector
`pytest backend/tests/ -q -k "paper_trading or resume"` appends four REAL pause/resume rows to the
git-tracked live audit file. Reproduced twice, deterministically; restored from `git show HEAD:` both
times and md5 re-verified. It is NOT caused by this step (no file I touched writes; twelve candidate
files run individually leave the digest unchanged) but I found it, so it gets its own step rather
than a paragraph.

**One existing guard caught me mid-fix, correctly.** My first wording of the 409 contained the phrase
`next cycle re-anchors`, which `test_phase_36_12_no_operator_string_still_promises_an_automatic_re_anchor`
bans -- because for LOST HISTORY that automatic anchor WAS the defect. My case is the opposite (a
legitimate daily roll), but the phrasing is ambiguous to a reader, so I reworded rather than narrow a
P0 guard to fit my sentence.
