# Experiment results -- 61.2 Decision-input integrity (DARK BUILD, Cycle 74, 2026-07-08)

Contract: contract_61.2.md. Research: research_brief_61.2.md (gate_passed true,
8 sources in full). Sequencing: operator-authorized early start via the
2026-07-08 /goal; NO backend restart today (PID 24910 = tonight's 66.2 cycle
host); NO masterplan flip.

## What was built (design adopted verbatim from the brief)

1. **settings.py** -- 4 new fields: `paper_synthesis_integrity_enabled=False`
   (umbrella: synthesis-error routing, both-fail degraded row, NULL
   passthrough, retry-on-empty, RJ advisory ctx, meta-scorer
   rank-normalization + streak WARN), `paper_position_recommendation_fix_
   enabled=False` (signal_downgrade revival, separate blast radius),
   `claude_code_timeout_s=150` (ungated scalar), `claude_code_empty_retry_
   max=2`. All four added to settings_api `_FIELD_TO_ENV` (operator-visible).
2. **Synthesis-error routing** (autonomous_loop.py): `SynthesisDegradedError`
   raised inside the full-path try when `final_synthesis.error` or missing
   `scoring_matrix` (flag ON) -> EXISTING lite fallback + 60.1 fallback-rate
   alarm machinery reused; `_fallback_reason` stamped "SynthesisDegradedError:
   synthesis_error: ...".
3. **Both-fail honest absence**: degraded marker dict (`_degraded`,
   `_degraded_reason`, `_path="degraded"`, NULL score/rec) -> persisted via
   `_persist_analysis` (NULL passthrough for marker rows; `$._degraded` +
   `$._degraded_reason` in full_report_json) -> `_run_and_persist_one`
   converts the marker to None so it NEVER enters candidate/holding analyses.
   The SECOND fabrication site (:2462-2463 re-coercions) is bypassed only for
   marker rows.
4. **Retry-on-empty** (orchestrator.py `_generate_with_retry`): result-based
   classification -- `thoughts` "errored:" = retryable (full-jitter backoff
   cap 15s, budget `claude_code_empty_retry_max`), "rail_guard_skipped:" =
   NEVER retried (open breaker). Single retry layer; each attempt re-enters
   the RailGuard accounting (counts toward breaker + llm_call_log).
5. **Timeout** (criterion 2): default 150s via settings; threaded through
   `make_client` -> `ClaudeCodeClient(timeout_s=...)`; instance
   `recommended_step_timeout = timeout_s + 30` keeps the step budget above
   the subprocess timeout for ANY configured value.
6. **company_name** (criterion 3, ungated pure fix): `_persist_analysis`
   falls back `market_data.name -> quant.company_name -> NULL`.
7. **Meta-scorer** (criterion 4): `_rank_normalized_convictions` (percentile
   -> 1-10, midpoint ties) behind the flag for the no-key path, `_fallback_
   all`, AND the below-cap tail (full-set basis so head/tail share a scale);
   flag OFF = legacy saturated clamp (test-asserted). Cross-cycle streak
   counter in `handoff/.conviction_fallback_streak.json` (file-backed;
   module state dies on kickstart) -> P2 (project WARN tier) alert
   `conviction_fallback_streak` at streak >= 2; streak resets on a healthy
   cycle (both legs flag-gated).
8. **signal_downgrade** (criterion 5): `TradeOrder.analysis_recommendation`
   populated from the candidate analysis on BOTH the fresh-BUY and swap-BUY
   paths; `execute_buy(analysis_recommendation=...)` stores the VERDICT in
   paper_positions.recommendation when flag 2 is ON (legacy trade reason
   otherwise); decide_trades logs the unsafe-combination WARNING (flag2 ON +
   flag1 OFF). Old rows keep trade reasons and never match -- no backfill.
9. **RJ advisory ctx** (criterion 6): `_rj_portfolio_ctx` built when EITHER
   the binding flag OR the integrity flag is ON (binding stays OFF).
10. **Read-side NULL guards** (ungated; safe while flags OFF since NULL rows
    cannot exist): `ReportSummary.final_score/recommendation -> Optional` +
    `degraded` field (models.py), types.ts `number|null` mirror,
    reports-columns.tsx score-cell dash + sparkline null filter,
    ReportCompareDrawer null guard, formatters.py `or 0`/`or "DEGRADED"`
    guards (present-None does not trigger `.get` defaults).

## Files changed

backend/config/settings.py, backend/api/settings_api.py,
backend/agents/llm_client.py, backend/agents/claude_code_client.py,
backend/agents/orchestrator.py, backend/services/autonomous_loop.py,
backend/services/meta_scorer.py, backend/services/portfolio_manager.py,
backend/services/paper_trader.py, backend/api/models.py,
backend/slack_bot/formatters.py, frontend/src/lib/types.ts,
frontend/src/components/reports-columns.tsx,
frontend/src/components/ReportCompareDrawer.tsx,
backend/tests/test_phase_61_2_decision_integrity.py (NEW, 33 tests).

## Verification (verbatim)

Immutable command (pytest leg):
```
python -m pytest backend/tests -k 'synthesis or persist or downgrade or meta_scorer or 61_2' -q
46 passed, 935 deselected, 1 warning in 4.43s
```
Regression set: `-k 'rail_guard or 66_1 or 66_3 or 60_4 or 62_4'` ->
`45 passed, 936 deselected, 1 warning in 17.41s`.
IMPORT_OK across all touched modules. `npm run build` clean.
`test -f handoff/current/live_check_61.2.md` -> created (deploy-pending plan).

## Criterion-4 root-cause documentation (immutable requirement)

The "06-03..06-10 LLM unavailability" is one continuous credit-death span,
NOT a discrete window. BQ evidence (queries run 2026-07-08 ~09:30 UTC, ADC):

- `llm_call_log` has ZERO rows for any meta/non-rail anthropic agent label --
  meta_scorer never logs its calls (observability gap; failures ALSO never
  logged: ClaudeClient's log write is unreached on exception).
- All 30 provider='anthropic' agent-NULL "ok" rows 2026-06-01..07-07 are TEST
  FIXTURES (identical input_tok=1000/output_tok=50/latency_ms=123.4; writer
  backend/tests/test_observability.py:230 hits the REAL table -- register).
- Excluding the fixture signature, the LAST GENUINE direct-API success is
  2026-05-17 (51-52 real calls 05-16/17, real token counts). Nothing since.
- Live confirmation: 2026-07-07 18:01:20Z meta_scorer 400 "credit balance is
  too low" (backend.log:32670) -- the alert that finally fired was one of the
  four alerting.py imports DEAD until 66.1 (07-07); no June alert exists in
  Slack (searched "Conviction overlay" 06-01..06-15: zero hits).

Root cause: direct-API Anthropic credits exhausted since ~2026-05-18; the
06-03..06-10 audit observation was a sample of a 7-week outage made invisible
by (a) the dead alert import, (b) no failure logging, (c) fixture pollution
masquerading as successes. Operator decision pending on credit top-up
(metered). The VALUE defect (constant saturated 10.00) is fixed by
rank-normalization; the streak WARN closes the persistent-outage blindness.

## Disclosures

- DARK: both behavior flags default OFF; flag-OFF legacy behavior is
  regression-asserted (fabricated HOLD/0.0 test, saturated-clamp test,
  reason-string pos_row test). Deploy (backend restart) deliberately NOT
  performed today -- criterion 3's live_check leg and the immutable
  live_check BQ evidence require a post-fix DEPLOYED cycle -> expected Q/A
  verdict CONDITIONAL.
- Retry cost: worst case 2 extra rail calls per synthesis-class step, each
  honestly metered as its own llm_call_log row + breaker count; cost tracker
  also records per-attempt (mirrors llm_call_log semantics).
- Deviation from brief detail (decide-and-note): streak counter lives in its
  own state file rather than a field inside .cycle_heartbeat.json (same
  durability, zero coupling to heartbeat semantics). `degraded` API field is
  populated as None for now (frontend infers degraded from null score; a
  reader-side `$._degraded` extraction is a follow-on).
- 60.2 interaction (from the brief): with churn-fix OFF, an absent re-eval
  re-exposes sentinel displacement -- out of 61.2 scope, disclosed.
- NEW P0 finding from the parallel money-engine audit (wf_e26ca01b-6c6,
  adversarially verified, NOT fixed in this step -- scope discipline): the
  full-path RiskJudge verdict nests under risk_assessment['judge'] but
  portfolio_manager reads top-level only -> full-path BUYs size at the 10%
  default, REJECT does not block even with the binding flag ON, and
  risk_judge_decision persists '' (breaks 66.2 criterion-1(a)'s
  "risk_judge_decision recorded" on any BUY tonight). Filed for operator
  decision: hotfix as its own mini-step vs fold into 61.x. See
  live_check_66.2.md section 6 (pending append) + the audit dossier.
