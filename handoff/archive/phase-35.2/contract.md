# phase-35.2 -- RiskJudge telemetry-wrapper restoration (GeminiClient llm_call_log retrofit)

**Step id:** `phase-35.2`
**Date:** 2026-05-22
**Mode:** EXECUTION (backend bug fix; mirror existing ClaudeClient pattern into GeminiClient).
**Cycle:** Cycle 17 (after Cycle 16 phase-44.1 frontend foundation).

---

## North-star delta

**Terms:** B (compute attribution unlocks Burn optimization) + R (audit/compliance + future llm_call_log analytics).

**B (primary):** llm_call_log accumulates per-cycle per-agent token spend; with the retrofit, every Gemini-routed Risk-Judge / Synthesis / Moderator / Critic call writes a row enabling `SELECT agent, SUM(input_tok * pricing + output_tok * pricing) GROUP BY agent` queries. This unblocks Burn-attribution analytics (which agent burns the most per cycle, where to optimize prompt-truncation aggressively). Conservative estimate: 5-15% Burn reduction over 60-day window via prompt-budget tuning informed by these analytics.

**R (secondary):** audit trail for every LLM call (regulator-grade observability per OWASP LLM v2 + SR-11-7 model risk). Also enables phase-43.0 DoD-7 verification (Risk-Judge structured-output >= 95% success rate -- requires telemetry to count).

**P:** N/A (no trading logic; pure observability).

**Caltech arxiv:2502.15800 discount:** N/A.

**How measured:** Pre-step llm_call_log row count for cycle_id `c7801712` = 0 (per closure_roadmap §3 BQ-probe B-3). Post-step expectation: next cron writes >= 14 rows (1 per ticker per agent invocation, ~14 tickers * 5 agents = ~70 rows). Live verification deferred to Monday's 14:00 ET cron + BQ probe.

---

## Research-gate decision

**Researcher SKIPPED.** The fix path is documented in closure_roadmap §3 + §9 audit_basis upgrade (cycle 12 BQ-probe diagnosis was precise: "autonomous_loop's Risk-Judge path bypasses backend/agents/llm_client.py::make_client instrumentation wrapper"). cycle-12 research_brief covered OWASP LLM v2 + SR-11-7 observability patterns. No new external pattern.

---

## Hypothesis

> The closure_roadmap §3 finding said the Risk-Judge call BYPASSES the
> wrapper. Investigation revealed a MORE SPECIFIC cause: `ClaudeClient.
> generate_content` (line 1645+) has the `log_llm_call` retrofit, but
> `GeminiClient.generate_content` (line 849+) does NOT. Since phase-34.1
> flipped both tiers to gemini-2.5-pro, ALL Risk-Judge invocations now route
> through GeminiClient -- which silently dropped the telemetry write. If we
> mirror the ClaudeClient log_llm_call block (with `provider="gemini"` +
> per-call latency timing via `_t0 = _time.perf_counter()`) into
> `GeminiClient.generate_content` right before its `return LLMResponse(...)`,
> THEN every Gemini-routed agent call will write an llm_call_log row.

---

## Immutable success criteria (verbatim from masterplan 35.2.verification)

1. `risk_judge_output_in_llm_call_log_quotes_portfolio_sector_exposure_field` -- **DEFERRED to live observation** (Monday's cron will produce the row; this cycle ships the writer + the structural test)
2. `synthesis_output_contains_portfolio_concentration_warning_text` -- **DEFERRED to live observation** (same)
3. `live_check_quotes_both_verbatim` -- **DEFERRED to live observation** (operator runbook in live_check_35.2.md)

Plus /goal integration gates 1-10.

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Pre-step health check (live loop healthy, kill-switch unpaused) | DONE |
| 2 | Investigation: traced llm_call_log writer to ClaudeClient.generate_content line 1645; found absent in GeminiClient.generate_content line 849-1037 | DONE |
| 3 | Researcher SKIP (closure_roadmap §3 diagnosis + cycle-12 brief sufficient) | DONE |
| 4 | Write this contract | IN FLIGHT |
| 5 | Modify backend/agents/llm_client.py: add `_t0 = _time.perf_counter()` at GeminiClient.generate_content start + log_llm_call write before return | DONE |
| 6 | pytest backend/tests/test_phase_35_2_gemini_telemetry.py (5 tests) | DONE (5/5 pass; total 318 -> 323) |
| 7 | live_check_35.2.md + Q/A + harness_log Cycle 17 + flip 35.2 status | IN FLIGHT |

---

## Files changed

- `backend/agents/llm_client.py` -- 2 inserts in `GeminiClient.generate_content`:
  1. `_t0 = _time.perf_counter()` near method top (post fail-open guard)
  2. `_latency_ms` compute + try/except log_llm_call write before `return LLMResponse(...)`
- `backend/tests/test_phase_35_2_gemini_telemetry.py` (NEW, 95 lines, 5 tests) -- source-grep tests that verify the retrofit is structurally correct (provider="gemini", side-channel pluck, fail-open, timer ordering, signature compat with ClaudeClient pattern)

**NOT changed:** ClaudeClient.generate_content (existing pattern); paper_trader; outcome_tracker; any frontend file.

---

## References

- closure_roadmap.md §3 BQ-probe B-3 (the diagnosis: c7801712 had 0 llm_call_log rows)
- closure_roadmap.md §9 (audit_basis upgrade verbatim)
- backend/agents/llm_client.py:1645-1669 (the reference pattern in ClaudeClient)
- backend/services/observability.py (log_llm_call helper -- single source of truth)
- /goal directive (10 integration gates)
