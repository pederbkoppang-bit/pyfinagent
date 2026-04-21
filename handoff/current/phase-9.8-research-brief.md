---
step: phase-9.8
tier: simple
date: 2026-04-20
---

## Research: Cost-budget watcher and circuit breaker for LLM + BQ spend (phase-9.8)

### Queries run (three-variant discipline)
1. **Current-year frontier**: "LLM API cost budget circuit breaker pattern 2026"
2. **Last-2-year window**: "LLM API cost control token budget daily cap production 2025"
3. **Year-less canonical**: "circuit breaker reset cadence software pattern"
4. Supplementary canonical: "BigQuery maximum_bytes_billed cost control query budget 2025"
5. Supplementary frontier: "per-job per-provider LLM spend monitoring multi-scope budget alert"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://markaicode.com/circuit-breaker-resilient-ai-systems/ | 2026-04-20 | blog (AI/LLM focus) | WebFetch | "5 consecutive failures trip the breaker... 60-second reset gives the provider a full minute to recover" |
| https://docs.cloud.google.com/bigquery/docs/best-practices-costs | 2026-04-20 | official doc (Google) | WebFetch | "`maximum_bytes_billed` estimates bytes before execution and fails query without charging if it exceeds limit" |
| https://martinfowler.com/bliki/CircuitBreaker.html | 2026-04-20 | authoritative blog (Fowler) | WebFetch | "self-resetting behavior by trying the protected call again after a suitable interval" — original canonical reference |
| https://skywork.ai/blog/ai-api-cost-throughput-pricing-token-math-budgets-2025/ | 2026-04-20 | industry blog | WebFetch | "Teams combining 50/80/100% budget alerts with 3x rate-of-change detectors routinely catch misconfigured loops within hours" |
| https://mlflow.org/blog/gateway-budget-alerts-limits | 2026-04-20 | official doc (MLflow) | WebFetch | "alert fires once per window"; resets at midnight UTC daily, Sunday UTC weekly, 1st monthly; Redis-backed for multi-replica atomicity |
| https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026 | 2026-04-20 | industry blog (2026) | WebFetch | "Z-score kill switch: auto-pause at >4 standard deviations from 7-day baseline"; per-agent attribution via OTel |
| https://docs.cloud.google.com/bigquery/docs/custom-quotas | 2026-04-20 | official doc (Google) | WebFetch | Quotas reset midnight Pacific; "approximate" safeguards, not strict; project-level `QueryUsagePerDay` + per-user |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.sitepoint.com/claude-api-circuit-breaker-pattern/ | blog | 403 forbidden |
| https://tianpan.co/blog/2026-04-15-backpressure-llm-pipelines | blog | search snippet; covered by markaicode |
| https://cloud.google.com/blog/topics/developers-practitioners/controlling-your-bigquery-costs | official blog | covered by best-practices-costs doc |
| https://docs.litellm.ai/docs/proxy/cost_tracking | official doc | snippet; liteLLM proxy pattern (not in pyfinagent stack) |
| https://oneuptime.com/blog/post/2026-02-17-how-to-control-bigquery-costs-with-custom-daily-query-quotas/view | blog | covered by custom-quotas doc |
| https://dev.to/kuldeep_paul/the-best-tools-for-monitoring-llm-costs-and-usage-in-2025-5f3a | community | snippet; tooling survey only |
| https://igotasite4that.com/blog/llm-api-rate-limiting-cost-control/ | blog | snippet; LiteLLM proxy path |
| https://www.systemsarchitect.io/services/google-bigquery/cost-optimization-best-practices/pt/google-bigquery-cost-optimization-best-practices-set-up-query-cost-controls-with-maximum-bytes-bill | blog | snippet; covered by official doc |
| https://medium.com/google-cloud/how-to-set-hard-limits-on-bigquery-costs-with-custom-quota-f8c26df0b2b8 | community | snippet; covered by custom-quotas |
| https://www.groundcover.com/learn/performance/circuit-breaker-pattern | blog | snippet; covered by Fowler canonical |

---

### Recency scan (2024-2026)

Searched for 2026 and 2025 literature on LLM cost circuit breakers, BQ cost controls, and multi-scope budget alerting.

Found 3 new 2026 findings that complement canonical sources:

1. **GetOnStack incident** (cited in markaicode, 2026): undetected agent-to-agent infinite loop ran 11 days, costs from $127/week to $47,000 -- canonical case for why cost-runaway is a distinct circuit-breaker trigger separate from HTTP error rates.
2. **MLflow AI Gateway budget policies** (mlflow.org, 2026): production-grade per-workspace budget enforcement with Redis-backed atomic state, window-aligned resets, once-per-window alert firing.
3. **Per-agent OTel attribution** (digitalapplied.com, 2026): four token-layer tagging (prompt, tool, memory, response) + z-score kill switch at >4 SD from 7-day baseline -- multi-scope pattern beyond daily/monthly aggregates.

BQ finding: as of September 2025, new BQ projects default to a 200 TiB/day on-demand query limit (Google docs); `maximum_bytes_billed` remains the per-query hard cap. Custom project quotas reset midnight Pacific, noted as "approximate" (not strict).

---

### Key findings

1. **Circuit breaker reset cadence -- self-resetting, not manual**: Fowler's canonical pattern uses a timeout-based Half-Open probe, not manual reset. Typical range: 30-60s for latency-sensitive APIs. For budget-based breakers the natural reset is the budget window boundary (next day, next month) -- not a fixed 60s, because overspend today is not resolved by waiting 60 seconds. (Sources: Fowler 2014, markaicode 2026)

2. **Two distinct circuit-breaker trigger classes**: (a) error-rate/latency triggers for reliability; (b) cost-runaway triggers for spend. Cost-based breakers should watch USD spend accumulation, not HTTP status codes -- the GetOnStack incident had 200 OKs throughout while costs multiplied 370x. (Source: markaicode 2026)

3. **Daily $5 / monthly $50 caps**: At 2026 prices (Claude Sonnet ~$3/MTok input, ~$15/MTok output; Gemini 1.5 Flash ~$0.075/MTok), a research harness running ~10 cycles/day with ~4K token prompts spends roughly $0.12-$0.60/day on LLM alone; $5/day is a ~8-40x headroom cap -- generous for a dev harness, tight for production burst. $50/month allows ~17 daily-cap days of full spend. Literature consensus: tier alerting at 50%/80%/100% rather than single trip, especially for monthly. (Sources: skywork.ai 2025, digitalapplied.com 2026, costgoat.com pricing 2026)

4. **BQ cost control -- two layers**: `maximum_bytes_billed` per-query (query fails without charge if estimate exceeds limit) + project-level custom quota (`QueryUsagePerDay` in TiB, resets midnight Pacific, approximate). On-demand pricing only; reservations use slot caps instead. (Sources: Google docs best-practices-costs, custom-quotas)

5. **Idempotency semantics when tripped**: if the circuit is tripped at 09:00 and the watcher runs again at 18:00 on the same day, the idempotency key (`cost_budget_watcher:2026-04-20`) will mark the run as skipped -- meaning a second-trip alert is suppressed even if spend grows further. This is correct for notification idempotency but means mid-day spend growth is not re-reported. (Source: internal code analysis, job_runtime.py:92-98)

6. **Multi-scope fuse gap**: current implementation checks daily aggregate and monthly aggregate only, with no per-provider (Anthropic vs Gemini vs OpenAI) or per-job breakdown. Literature recommends per-provider alerting to detect model-drift spend anomalies (e.g., unexpected routing to GPT-4o instead of Flash). (Sources: digitalapplied.com 2026, skywork.ai 2025)

7. **Once-per-window alert discipline**: MLflow pattern fires alert exactly once per budget window, preventing notification spam. Current implementation achieves this via idempotency key (daily key = once per day). Correct for daily scope; monthly scope is not idempotent -- multiple daily runs on different days can each fire a monthly-over alert. (Source: mlflow.org 2026, internal code cost_budget_watcher.py:29)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/jobs/cost_budget_watcher.py` | 64 | Daily/monthly spend watcher, trips BudgetEnforcer, fail-open alert | Active, phase-9.8 |
| `backend/autoresearch/budget.py` | 105 | BudgetEnforcer: wallclock + USD enforcer, alert_fn injection, idempotent breach | Active, phase-8.5.2 |
| `backend/slack_bot/job_runtime.py` | 117 | Heartbeat context manager + IdempotencyStore + IdempotencyKey helpers | Active, phase-9.1 |
| `tests/slack_bot/test_cost_budget_watcher.py` | 76 | 4 tests: under-budget, daily-over, monthly-over, alert_fn=None path | Active, phase-9.8 |

---

### Consensus vs debate (external)

**Consensus:**
- Budget-aware circuit breakers are a distinct class from error-rate breakers; both are needed.
- Daily + monthly scopes are the minimum; per-provider and per-job add value but are secondary.
- Self-resetting at window boundary (next day/month) is more appropriate than a fixed timeout for budget breakers.
- Tiered alerting (50%/80%/100%) is preferable to single-trip for monthly caps.
- Alert idempotency (once-per-window) is a production requirement to prevent spam.

**Debate:**
- Cap values are context-dependent. $5/day is discussed as both "generous for dev harness" and "too tight for production burst". Literature does not provide a universal number.
- Hard-stop vs soft-alert: some systems (MLflow) return HTTP 429 to block further requests; others (like the current implementation) only alert and rely on downstream code to check the trip state. The current implementation does not block -- `BudgetEnforcer.tick()` returns a state dict but does not raise; callers must check `state["terminated"]`.

---

### Pitfalls (from literature)

1. **Watching the wrong metric**: error-rate breakers miss cost-runaway on successful (200 OK) agentic loops. Need a dedicated USD-accumulation trigger. (markaicode 2026)
2. **Monthly alert duplication**: if daily idempotency key suppresses re-runs, a monthly-over condition can fire on each new day until month end. Needs a monthly idempotency key, not just daily. (internal analysis + mlflow.org pattern)
3. **BQ quota is approximate**: `QueryUsagePerDay` custom quota is not strictly enforced -- BigQuery may occasionally exceed it. `maximum_bytes_billed` per-query is the only hard block. (Google custom-quotas doc)
4. **Spend data latency**: GCP billing data has up to 24h lag on detailed line items; real-time spend checks must use in-process accumulators or provider usage APIs rather than billing exports. (Google best-practices-costs)
5. **No per-provider visibility**: monthly aggregate hides which provider (Anthropic vs Gemini) is the dominant cost driver. Model drift (unplanned routing to expensive models) is not detectable without per-provider scope. (digitalapplied.com 2026)

---

### Application to pyfinagent (mapping external findings to file:line anchors)

| Finding | File:Line | Gap / Recommendation |
|---------|-----------|----------------------|
| Reset cadence: budget breakers reset at window boundary, not fixed timeout | `cost_budget_watcher.py:29` -- daily key only | Monthly over-trip suppression: add a monthly idempotency key so the monthly-over alert fires once per calendar month, not once per day. Currently the daily key at line 29 prevents re-run entirely for the day, but if the job runs on day 2 of a month-over condition, it will fire again. |
| Tiered alerting 50%/80%/100% | `cost_budget_watcher.py:43-55` -- binary trip | Current: binary (tripped / not). Literature recommends 50%/80%/100% threshold stages. Could add a `warn_fn` for 80% threshold in addition to the hard trip at 100%. Not a blocking gap for the current 4-test criterion. |
| Once-per-window alert discipline | `cost_budget_watcher.py:53-60` + `job_runtime.py:92-98` | Daily idempotency is correct. Monthly idempotency is absent: the key is daily (`{job_name}:{iso_date}`), so a monthly-over condition fires every day the job runs after the monthly cap is exceeded. Recommend a second `monthly_key = IdempotencyKey.daily(JOB_NAME + "_monthly", day=YYYY-MM-01)` to gate monthly alerts. |
| BudgetEnforcer wallclock disabled | `cost_budget_watcher.py:48` -- `wallclock_seconds=10**9` | Intentional: watcher uses USD path only. Correct for this use case. |
| No per-provider scope | `cost_budget_watcher.py:19-28` -- single spend input | Inputs are aggregated totals (`daily_spend_usd`, `monthly_spend_usd`). No per-provider breakdown accepted. Future: accept a `provider_spend: dict[str, float]` kwarg and check per-provider caps. Not in current criterion scope. |
| BQ `maximum_bytes_billed` | not in watcher; BQ calls are in other services | watcher monitors spend after the fact, not per-query. BQ per-query cap should be set at the query site in `data_server.py` / `backtest_engine.py`, not here. |
| Alert_fn fail-open | `cost_budget_watcher.py:59-60` | Correct pattern; matches literature recommendation (never let alerting failure block the run). |
| BudgetEnforcer breach reason | `budget.py:87-88` -- reason is "usd" | `cost_budget_watcher.py:55` maps reason to scope ("daily"/"monthly") via closure capture. Correct. |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total (17 collected: 7 full + 10 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (4 files read)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/phase-9.8-research-brief.md",
  "gate_passed": true
}
```
