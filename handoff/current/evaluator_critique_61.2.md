# Evaluator critique -- 61.2 Decision-input integrity (DARK BUILD)

Q/A agent (merged qa-evaluator + harness-verifier), first spawn for 61.2.
Date: 2026-07-08. Cycle 74. Build commit: 6186784c (2026-07-08 11:28:45 +0200).

## VERDICT: CONDITIONAL

Every deterministic check and every test-level criterion leg passes, and the
build's factual claims were independently reproduced (including a live BQ
re-run of the criterion-4 root-cause archaeology). The verdict is capped at
CONDITIONAL solely because criterion 3's live leg and the immutable
`verification.live_check` field require BQ evidence from a post-fix DEPLOYED
autonomous cycle, and the backend was deliberately not restarted today (66.2
evidence-window protection). This is the designed intermediate state the
contract predicted (Cycle-68 precedent). Prior CONDITIONAL count for 61.2: 0
(3rd-CONDITIONAL auto-FAIL not in play).

## 1. Harness-compliance audit (5 items, run first)

1. **Researcher gate: PASS.** `research_brief_61.2.md` envelope:
   `gate_passed: true`, tier complex, 8 external sources read in full (>= 5
   floor), 28 URLs, recency-scan section present, 3-variant query discipline
   listed, 19 internal files. Brief mtime 11:04:41 predates contract mtime
   11:05:50.
2. **Contract before GENERATE: PASS (mtime evidence).** All five artifacts
   plus code land in the single commit 6186784c, so git timestamps cannot
   discriminate intra-commit order; the PreToolUse audit stream logs only
   `ts/tool/verdict/reason` (no file paths), so it cannot either. Filesystem
   mtimes give a clean monotonic chain: brief 11:04:41 -> contract 11:05:50 ->
   settings.py 11:06:34 -> meta_scorer.py 11:12:51 -> autonomous_loop.py
   11:16:31 -> test file 11:24:23 -> experiment_results 11:26:49 ->
   live_check 11:27:08 -> commit 11:28:45.
3. **experiment_results_61.2.md: PASS.** Exists with verbatim verification
   output; both pytest counts reproduced exactly (below).
4. **Log-last: PASS.** `grep 'phase=61.2' handoff/harness_log.md` -> no
   result= entry (only forward-looking "Next:" mentions from Cycles 72-73).
   Masterplan 61.2: `status: pending`, `retry_count: 0` -- no flip, as
   designed.
5. **No verdict-shopping: PASS.** No prior 61.2 verdict exists anywhere in
   harness_log.md; this is the first Q/A spawn.

## 2. Deterministic checks (verbatim)

- **Immutable command** (`pytest -k 'synthesis or persist or downgrade or
  meta_scorer or 61_2' -q && test -f handoff/current/live_check_61.2.md`):
  `46 passed, 935 deselected, 1 warning in 4.66s` + live_check file exists,
  exit 0. Matches the experiment_results claim (46 passed) exactly.
- **Regression set** (`-k 'rail_guard or 66_1 or 66_3 or 60_4 or 62_4'`):
  `45 passed, 936 deselected, 1 warning in 17.67s`. Matches exactly.
- **Flag defaults** (Settings.model_fields):
  `paper_synthesis_integrity_enabled=False`,
  `paper_position_recommendation_fix_enabled=False`,
  `claude_code_timeout_s=150`, `claude_code_empty_retry_max=2`. As claimed.
- **Backend untouched**: PID 24910 lstart = Jul 8 10:40:58 2026 +0200
  (08:40:58Z; briefing said ~08:44 UTC -- same process, immaterial delta),
  which PREDATES the build commit 11:28:45 +0200. Tonight's 66.2 evidence
  host is genuinely untouched.
- **66.2 criterion-2 diff spot check** (`git diff c1e6050b..HEAD --
  portfolio_manager.py paper_trader.py`, + lines surviving the filter):
  every surviving line is a comment or logic gated behind
  `paper_position_recommendation_fix_enabled` (the unsafe-combination
  `logger.warning` and the flag-gated verdict store). NO threshold, cap,
  limit, or sizing value changed. No blocker.
- **Frontend gate (qa.md 1b -- diff touches frontend/)**:
  `npx eslint .` -> `55 problems (0 errors, 55 warnings)`, ESLINT_EXIT=0
  (warnings do not fail the gate; NONE of the 55 are in the three
  61.2-touched files: types.ts, reports-columns.tsx, ReportCompareDrawer.tsx).
  `npx tsc --noEmit` -> TSC_EXIT=0.

## 3. LLM judgment

### Contract alignment
All six success criteria in contract_61.2.md match
`.claude/masterplan.json` 61.2 word-for-word (line-wrapping aside). The
verification command and live_check are restated faithfully.

### Mutation-resistance (criterion-1 regression test probed)
- Reverting the FIRST fabrication site (synthesis-error routing in
  `_run_single_analysis`) breaks `test_flag_on_error_synthesis_routes_to_lite`
  (asserts `_path == "lite"`, real BUY not HOLD, `_fallback_reason` prefix
  `SynthesisDegradedError`) and `test_flag_on_both_fail_returns_degraded_marker`.
  CATCHES REVERT.
- Reverting the SECOND fabrication site (`_persist_analysis` :2462-2463
  re-coercion bypass) breaks `test_degraded_marker_persists_nulls_never_hold`,
  which calls the REAL `_persist_analysis` with a mocked BQ client and asserts
  `final_score is None` / `recommendation is None` / `summary DEGRADED:` /
  `$._degraded` reach `save_report`. CATCHES REVERT of the second site.
- Flag-OFF byte-identity is positively asserted (legacy HOLD/0.0 fabrication,
  legacy 10/10/10 clamp, legacy reason-string pos_row).
- Retry tests exercise the real `_generate_with_retry` with a stub model:
  errored-empty retried then success (2 calls), `rail_guard_skipped` never
  retried (1 call), flag OFF no retry, budget bound at 1+2 with legacy empty
  (never None) returned.
- **Weakness (register, non-blocking):** three tests are source-grep
  assertions rather than behavioral -- `test_degraded_marker_never_enters_
  analyses`, `test_streak_warn_wired_at_two`, `test_ctx_gate_references_both_
  flags`. A mutant could alter behavior while preserving the grepped strings.
  Mitigation: I read the RJ gate directly (autonomous_loop.py:822-827, OR of
  both flags, byte-identical both-OFF) and the closures are not directly
  invocable without a cycle harness; disclosed in-test. Register as a
  test-debt item, not a blocker.

### Anti-rubber-stamp probes (4 claims checked against code)
1. **Retry-on-empty breaker interaction** (orchestrator.py:788-884):
   VERIFIED. Flag-gated at :794; predicate at :871-874 retries only
   `text == ""` AND `thoughts.startswith("errored:")`, so
   `rail_guard_skipped:` empties are structurally excluded; full-jitter
   backoff `random.uniform(0, min(15, 2*2**attempt))` at :877; each retry
   re-enters `model.generate_content`, so RailGuard re-check + per-attempt
   failure recording are inherent (no RailGuard change needed) -- the claim
   is accurate, not aspirational.
2. **Meta-scorer tail normalization full-set basis** (meta_scorer.py):
   VERIFIED. `_tail_convs = _rank_normalized_convictions(head + tail)[len(head):]`
   at :274 (full-set basis, head/tail share one scale); flag-gated dispatcher
   at :170-177 returns the legacy per-candidate clamp when OFF
   (byte-identical); `_fallback_all(head + tail)` at :240/:252.
3. **Criterion-2 configurability**: VERIFIED end-to-end --
   llm_client.py:1976 `timeout_s=int(getattr(settings, "claude_code_timeout_s",
   150))` inside make_client; claude_code_client.py:479 `timeout_s: int = 150`
   default, :486 `recommended_step_timeout = timeout_s + 30`, consumed at
   :557. (The unit tests construct the client directly and do not cover the
   make_client threading -- covered here by code read; minor test-debt note.)
4. **settings_api exposure**: VERIFIED -- all four fields in `_FIELD_TO_ENV`
   at settings_api.py:260-263, as the results claim.

### Scope honesty (criterion-4 root cause independently re-run)
Bounded BQ query (llm_call_log, provider='anthropic', agent IS NULL,
2026-05-01..07-07): `fixture_rows(input_tok=1000 AND output_tok=50) = 106`,
`genuine_ok_rows = 100`, `last_genuine_ok = 2026-05-17`. The headline claims
-- last genuine direct-API success 2026-05-17, everything after is
test-fixture pollution, the 06-03..06-10 "window" is a sample of one
continuous credit-death span -- REPRODUCE against the live table. Disclosures
are unusually complete: deploy-pending status, deviation from the brief
(streak counter in its own state file), the 60.2 interaction left out of
scope, and a NEW P0 finding (RiskJudge verdict nesting) explicitly NOT fixed
for scope discipline and filed for operator decision. No overclaim found.

### Research-gate compliance
PASS -- brief cited throughout the contract; 8 full sources spanning official
docs (Azure, Google SRE, Anthropic, Tenacity), canonical blogs (Fowler,
Brooker), peer-reviewed (arXiv 2605.08563) and practitioner tiers; design
adopted from the brief verbatim.

### Sequencing ruling (as tasked)
(a) Early start during 66.2's evidence window: operator-authorized per the
2026-07-08 /goal text (disclosed in contract + results). The two facts that
make it safe are both deterministically verified above: backend PID predates
the commit, and the trading-file diff contains zero gate/threshold/sizing
changes with both behavior flags default-OFF and byte-identical-OFF tested.
No 66.2 contamination. (b) The pending live legs are honestly labeled PENDING
in live_check_61.2.md §A with a concrete deploy plan; ruling on the evidence:
CONDITIONAL, per the criterion's literal live-verb requirement -- not FAIL,
because the pending state is structural (no deployed cycle can exist yet),
and not PASS, because the immutable text explicitly demands post-fix cycle
BQ rows.

## 4. Violations (all of one class: live evidence pending deploy)

1. Criterion 3, live leg -- "live_check shows BQ rows from a post-fix
   autonomous full-path cycle with non-null company_name": no post-fix
   deployed cycle exists (backend deliberately not restarted).
2. Immutable `verification.live_check` -- post-fix cycle BQ rows (non-null
   company_name, zero final_score=0.0 AND final_synthesis.error rows,
   non-constant convictions): same dependency.
3. qa.md §1c live-UI-capture gate -- the diff makes UI claims
   (reports-columns score-cell dash + degraded badge + sparkline null filter,
   ReportCompareDrawer guard) and no live Playwright capture exists. NULL
   rows cannot exist while the flag is OFF, so there is nothing degraded to
   capture today; a post-deploy capture of the reports page (normal render,
   plus degraded-row render once the flag is promoted and a degraded row
   exists) must accompany live_check §A.

## 5. Blockers to clear for PASS (fresh Q/A after evidence lands)

1. Post-cycle backend restart per the live_check deploy plan; then >= 1
   scheduled full-path cycle.
2. Append §A to live_check_61.2.md: BQ rows showing non-null company_name on
   full-path rows (ungated fix -- observable immediately post-restart), zero
   new final_score=0.0 AND $.final_synthesis.error rows, conviction spread
   (the latter two legs become observable on flag promotion -- if the
   operator keeps flags OFF, document that the zero-fabrication leg is
   vacuously satisfied only after promotion and hold the flip until then).
3. Live Playwright capture of the reports page appended to the live_check
   (structure snapshot; screenshot if the degraded badge is visible).

## 6. Register notes (non-blocking)

- Source-grep test debt: 3 tests assert source text, not behavior (see
  mutation-resistance section).
- make_client -> timeout threading has no unit test (verified by code read).
- test_observability.py:230 writes fixture rows to the PROD llm_call_log
  (the pollution that masked the credit death) -- already registered by the
  build; reinforcing: this deserves its own fix step.
- Backend PID observed start 08:40:58Z vs the briefed "08:44 UTC" --
  immaterial (same PID 24910, predates commit), noted for record accuracy.

## JSON envelope

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All deterministic checks pass (immutable cmd 46 passed + live_check file, regression 45 passed, flags OFF/150/2, backend PID predates commit, 66.2 diff clean, eslint 0 errors, tsc clean); criteria 1/2/4/5/6 fully evidenced at test level with strong mutation resistance; criterion-4 root cause independently reproduced via live BQ (last genuine success 2026-05-17). CONDITIONAL solely on the structurally-pending live legs: criterion-3 live_check BQ evidence and the immutable live_check field require a post-fix DEPLOYED cycle (deploy scheduled post-cycle by design), plus the §1c live UI capture for the reports-page changes.",
  "violated_criteria": [
    "criterion_3_live_leg: post-fix cycle BQ rows with non-null company_name",
    "immutable_live_check: post-fix cycle BQ evidence (fabrication-zero, conviction spread)",
    "Missing_Assumption: live UI capture (qa.md 1c)"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "criterion-3 live leg evaluation",
      "state": "live_check_61.2.md §A explicitly PENDING; backend PID 24910 (08:40:58Z) deliberately not restarted; no post-fix deployed cycle exists",
      "constraint": "live_check shows BQ rows from a post-fix autonomous full-path cycle with non-null company_name (immutable criterion 3)"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "immutable verification.live_check evaluation",
      "state": "no post-fix deployed cycle; test-level evidence only",
      "constraint": "BQ rows from >=1 post-fix autonomous cycle: non-null company_name on full-path rows, zero new final_score=0.0 AND final_synthesis.error rows, non-constant convictions in paper_trades.signals"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "UI-claim verification for reports-columns.tsx / ReportCompareDrawer.tsx changes",
      "state": "no live Playwright capture; NULL/degraded rows cannot exist while flags are OFF",
      "constraint": "qa.md §1c: UI claims require live browser_navigate + browser_snapshot/screenshot evidence; capture due with post-deploy live_check §A"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_5item",
    "verification_command",
    "regression_suite",
    "flag_defaults",
    "backend_pid_untouched",
    "66_2_criterion2_diff_spotcheck",
    "frontend_eslint",
    "frontend_tsc",
    "contract_criteria_verbatim_diff",
    "mutation_resistance_review",
    "code_probe_retry_breaker",
    "code_probe_meta_scorer_fullset",
    "code_probe_timeout_threading",
    "code_probe_settings_api",
    "bq_root_cause_reproduction",
    "research_gate_envelope",
    "harness_log_conditional_count"
  ]
}
```
