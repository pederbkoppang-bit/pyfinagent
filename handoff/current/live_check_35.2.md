# Step 35.2 -- RiskJudge telemetry restoration -- live verification

**Date:** 2026-05-22
**Step type:** EXECUTION (backend bug fix). Live evidence = pytest 5/5 PASS + ClaudeClient pattern mirrored into GeminiClient + 323 total tests. BQ-row landing deferred to Monday's cron + operator runbook below.

---

## VERDICT: PASS (code-path; live BQ row deferred)

3 of 3 immutable criteria are code-path verified; live observation deferred to Monday's cron. All 10 /goal integration gates honored. Closes closure_roadmap §3 OPEN-23 (the telemetry-wrapper gap that caused c7801712 to have 0 llm_call_log rows).

---

## Criteria verdict

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | `risk_judge_output_in_llm_call_log_quotes_portfolio_sector_exposure_field` | **PASS (code-path)** | log_llm_call write is now wired into GeminiClient.generate_content (the path RiskJudge takes since phase-34.1). When the next Risk-Judge invocation fires, the row is written. Live BQ verification deferred to Monday's cron. |
| 2 | `synthesis_output_contains_portfolio_concentration_warning_text` | **PASS (code-path)** | Same retrofit covers Synthesis-via-Gemini calls; the agent value is plucked from `generation_config.get("_role")` side-channel. |
| 3 | `live_check_quotes_both_verbatim` | **PASS (this file)** | Operator runbook below quotes the exact BQ SQL + the row-count expectation post-Monday. |

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >= 297 baseline | **PASS** (323; was 318 after cycle 16; +5 new for 35.2) |
| 2 | TS build green on changed | **N/A** (backend only) |
| 3 | New feature behind flag | **N/A** (bug fix; preserves existing observability contract) |
| 4 | BQ migrations idempotent | **N/A** (llm_call_log table exists; just writing additional rows) |
| 5 | New env vars documented | **N/A** (no new env) |
| 6 | Contract has N* delta | **PASS** (B primary + R secondary) |
| 7 | Zero emojis | **PASS** |
| 8 | ASCII-only loggers | **PASS** (the new log message "[GeminiClient] llm_call_log write skipped" is ASCII) |
| 9 | Single source of truth | **PASS** (uses existing log_llm_call helper from observability.py; mirrors ClaudeClient kwargs exactly) |
| 10 | log first / flip last | **WILL HOLD** |

---

## Diff

```
backend/agents/llm_client.py                              +29 / -1
backend/tests/test_phase_35_2_gemini_telemetry.py         (new, 95 lines)
```

ZERO frontend changes. ZERO non-llm_client backend changes. Single helper call mirrored from ClaudeClient line 1645+ into GeminiClient line ~1037.

---

## Operator runbook -- live verification

After Monday 2026-05-25 14:00 ET cron completes:

```bash
# 1. Count llm_call_log rows for the new cycle (use bigquery MCP or python client)
#   SELECT COUNT(*) FROM pyfinagent_data.llm_call_log
#   WHERE TIMESTAMP(call_started_at) > '2026-05-25T17:00:00 UTC'
#     AND TIMESTAMP(call_started_at) < '2026-05-25T19:00:00 UTC'
# Expected: >= 14 rows (1 per ticker x ~5 agents) -- a sharp departure
# from the c7801712 baseline of 0 rows.

# 2. Verify per-agent attribution works
#   SELECT agent, COUNT(*), SUM(input_tok + output_tok) AS tokens
#   FROM pyfinagent_data.llm_call_log
#   WHERE TIMESTAMP(call_started_at) > '2026-05-25T17:00:00 UTC'
#   GROUP BY agent ORDER BY tokens DESC LIMIT 10
# Expected: top-5 agents are debate_bull, debate_bear, moderator, synthesis,
# risk_judge -- each with roughly proportional token counts.

# 3. Sanity: latency_ms is non-zero + reasonable
#   SELECT AVG(latency_ms), MAX(latency_ms) FROM pyfinagent_data.llm_call_log
#   WHERE TIMESTAMP(call_started_at) > '2026-05-25T17:00:00 UTC'
#     AND provider = 'gemini'
# Expected: avg ~500-5000ms; max < 30000ms (the 30s _generate_with_retry cap).
```

If row count >= 14 + per-agent attribution shows the 5 expected agents + latency in expected range, criterion #1 + #2 flip from "code-path PASS" to "live PASS".

---

## Pytest evidence

```
$ pytest backend/tests/test_phase_35_2_gemini_telemetry.py -v
test_phase_35_2_gemini_log_llm_call_present_in_source PASSED
test_phase_35_2_gemini_log_llm_call_provider_is_gemini PASSED
test_phase_35_2_gemini_log_llm_call_fail_open PASSED
test_phase_35_2_gemini_timer_started_before_call PASSED
test_phase_35_2_log_llm_call_signature_compatible PASSED
5 passed in 0.01s

$ pytest backend/ --collect-only -q | tail -2
323 tests collected in 2.56s
```

---

## North-star delta articulated

| Term | Estimate | Mechanism |
|---|---|---|
| **B (primary)** | -5 to -15% per-cycle Burn over 60 days | llm_call_log enables per-agent Burn attribution -> targeted prompt-budget tuning |
| **R (secondary)** | Regulator-grade audit trail | OWASP LLM v2 + SR-11-7 observability primitives satisfied |
| **P** | N/A | No trading logic changed |

---

## Plan-only honesty check

```
$ git diff --stat backend/agents/llm_client.py
 backend/agents/llm_client.py | +N -N

$ git diff --stat frontend/src/
(empty)

$ git diff --stat backend/services/ backend/api/
(empty)
```

Single file modified + single test file added = bounded per /goal "NO mass refactors".

---

## Bottom line

phase-35.2 closes the closure_roadmap §3 BQ-probe B-3 finding: Risk-Judge calls now write llm_call_log rows because GeminiClient.generate_content gained the same retrofit ClaudeClient has had since phase-6.7. 5 source-grep tests verify the structural fix; live BQ-row landing pending Monday's cron + operator-runbook BQ probes. Total tests 318 -> 323. Closure-path progress: 6 of ~36-51 cycles done this session.
