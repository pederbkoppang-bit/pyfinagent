# Contract — phase-75.5.1: the cost-budget guard does not measure what its name says

- **Step id:** 75.5.1 (phase-75 follow-up queue, **P1, MONEY-ADJACENT — the $25/day circuit breaker**; executor: opus-tagged → Main-on-Fable GENERATE; gates opus/max via Workflow)
- **Date:** 2026-07-24
- **Boundary:** decide arm (a) vs (b) and implement; if (a), MUST ship flag-gated DARK with an ON-vs-OFF comparison before any default flip. No live BQ billing query without owner approval (live_check constraint).

## Research-gate summary (gate PASSED — wf_9cece795-b16)

Envelope: `tier=moderate, external_sources_read_in_full=6, snippet_only_sources=20, urls_collected=42, recency_scan_performed=true, internal_files_inspected=12, gate_passed=true`. Brief: `handoff/current/research_brief_75.5.1.md`.

Load-bearing findings:

1. **All prices feeding the guard are current and correct** (externally validated: opus-4-8 $5/$25, sonnet-4-6 $3/$15, haiku-4-5 $1/$5, gemini-2.5-flash $0.30/$2.50, 2.5-pro $1.25/$10, cache read 0.1×, cache write 2.0× (1h TTL), batch 0.5×, BQ on-demand $6.25/TiB). The defect is metric-vs-name, not numbers.
2. The breaker is `_check_cost_budget()` (llm_client.py:395-458, fires at :896/:1180/:1428/:2214), consuming `fetch_spend()` = **BigQuery** bytes×$6.25/TiB, while `cost_budget_daily_usd` (settings.py:384) is documented as the **LLM** cap.
3. **llm_call_log has NO per-call cost column**; the only dollar field is `session_cost_usd`, a per-cycle cumulative GAUGE that must never be summed (phase-66.3 doctrine). LLM spend MUST be derived from RAW tokens × MODEL_PRICING at query time — and raw tokens are **fix-invariant** across the 75.5 cache-double-subtract boundary (the fix changed computed cost, not stored tokens).
4. **CRUX LANDMINE:** flat-fee CC-rail rows (`provider='claude-code'` at autonomous_loop.py:2298, OR `provider='anthropic'` + `agent LIKE 'cc_rail:%'` at claude_code_client.py:502-504) record tokens at ~$0 real cost. Pricing them at API rates = phantom spend that trips the $25 breaker on FREE tokens → **false trading halt** (same phantom class as the $42/day session_cost_usd staircase). A metered-only filter is mandatory.
5. In-repo prior art `sovereign_api.py:236-286` already prices llm_call_log × MODEL_PRICING but **ignores the cache columns** (under-counts → breaker trips late). The accurate cache-aware formula is `cost_tracker.py:198-206` (read 0.1×, write 2.0×, Anthropic-only-in-practice per the pinned test) — port THAT.
6. No local ledger: llm_call_log is the existing durable cross-process (backend+harness+slack) daily record; in-process counters reset on restart. BQ cost bounded: column-pruned, date-filtered, 60s-cached like today's guard, timeout=30.
7. Fail-open doctrine (Fowler/Azure circuit-breaker): serve (0,0) on failure but make degradation OBSERVABLE — the arch-04 `_record_degradation` counter + one-shot P2 alert (spend.py:67-101, pinned by test_phase_75_llm_rail.py:577-592) is the seam; the new path must reuse it.
8. DARK-flag convention (measured in settings.py): `<name>_enabled: bool = Field(False, description="phase-X.Y: ...")` — exemplars sign_safe_overlays, kill_switch_peak_reset_enabled, paper_atomic_swap_enabled.
9. **Discovered defects to QUEUE, not fold in** (feedback_queue_discovered_defects_in_masterplan): (i) the three consumers disagree on the CAPS themselves — hard-block $25/$300 (settings.py:384-385) vs tile+watcher hardcoded $5/$50 (cost_budget_api.py:53-54, cost_budget_watcher.py:33-34); (ii) whether metered Gemini/Vertex-only spend warrants a sub-cap. Queued as 75.5.11 below.
10. `_check_session_budget` (autonomous_loop.py:103-116) is per-cycle/in-process — distinct guard, not this step's target.

## Decision: arm (a)

Add a real LLM-spend source and make the budget gate read it behind a default-OFF flag; keep `fetch_spend` (BQ) unchanged as a separate metric. Arm (b) rejected: renaming to a "BigQuery cap" abandons the operator's LLM-cost-ceiling intent and leaves metered-spend runaway ungoverned.

## Hypothesis

A `fetch_llm_spend()` that prices llm_call_log RAW tokens against the live MODEL_PRICING table with the cache-aware cost_tracker formula, restricted to METERED rows only (CC-rail excluded), wired into `_check_cost_budget` behind `cost_budget_use_llm_spend_enabled=False`, gives the $25/day breaker the metric its name promises — with flag-OFF behavior byte-identical to today (trip point unchanged), fail-open preserved through the same degradation seam, and the phantom-free-token false-halt class excluded by construction.

## Plan

1. `backend/config/settings.py`: `cost_budget_use_llm_spend_enabled: bool = Field(False, description="phase-75.5.1: When True the $25/day breaker reads fetch_llm_spend() (metered LLM tokens x MODEL_PRICING) instead of BigQuery bytes-billed spend. OFF -> byte-identical to pre-75.5.1 behavior.")`.
2. `backend/services/observability/spend.py`: add `fetch_llm_spend() -> tuple[float, float]`:
   - BQ query on `<project>.pyfinagent_data.llm_call_log`: SUM of input/output/cache_creation/cache_read tokens GROUP BY model, daily (`DATE(ts)=CURRENT_DATE()`) + month-to-date windows, `WHERE ok AND provider != 'claude-code' AND (agent IS NULL OR agent NOT LIKE 'cc_rail:%')`, column-pruned, `timeout=30` per the BQ rule;
   - price per model in Python with the cache-aware formula ported from cost_tracker.py:198-206 (read 0.1×, write 2.0×) against the imported `MODEL_PRICING` + `_DEFAULT_PRICING` (single source of truth, no copied table);
   - fail-open to (0.0, 0.0) on ANY exception THROUGH `_record_degradation` (same counter/alert seam);
   - module docstring updates the 75.5 SCOPE WARNING to name the new function as the LLM-spend source.
3. `backend/agents/llm_client.py` `_check_cost_budget`: select `fetch_llm_spend` vs `fetch_spend` by the flag; nothing else in the hot path changes (cache TTL, caps, trip logic, fail-open untouched).
4. Tests `backend/tests/test_phase_75_5_1_spend_metric.py` (offline, BQ client mocked):
   - price a fixture against the REAL imported MODEL_PRICING (no hardcoded prices) asserting the cache-aware daily total;
   - metered-scope fixture representing BOTH categories (CC-rail rows via `provider='claude-code'` AND via `provider='anthropic'`+`agent='cc_rail:x'` MUST contribute $0; metered Gemini + Anthropic-SDK rows MUST count) — a fixture that cannot represent both does not count;
   - flag-OFF byte-identity: `_check_cost_budget` routes to `fetch_spend` when OFF, `fetch_llm_spend` when ON, on identical inputs (trip point unchanged — the ON-vs-OFF $0 diff);
   - fail-open regression: an exception in `fetch_llm_spend` returns (0,0) AND fires the arch-04 seam (`degraded_count==1`, `alerted==True`);
   - `ok=False` rows excluded; monthly window included.
5. Mutation matrix (experiment_results per the immutable criterion + qa.md §4c): (i) swap the metric source back under flag-ON; (ii) remove the flag gate; (iii) DROP the CC-rail exclusion (the crux mutation); (iv) break the cache-aware formula (drop the 0.1× read discount); (v) fixture/stub mutation. Each must fail ≥1 test, executed with verbatim capture.
6. Queue step **75.5.11** (caps-disagreement: $25/$300 vs hardcoded $5/$50 across the three consumers; plus the Gemini-sub-cap question as a design note) — research-gated, written for a fresh executor.
7. live_check_75.5.1.md: verification output (exit 0), git diff --stat, the ON-vs-OFF comparison from the offline suite ($0 diff with flag OFF), the CC-rail-exclusion evidence, mutation matrix. NO live BQ billing query.
8. Q/A via qa-verdict Workflow; log; flip; auto-push (manual fallback per the hook-stall pattern).

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json step 75.5.1)

> command: `.venv/bin/python -m pytest backend/tests/test_phase_75_5_1_spend_metric.py -q`

1. "New backend/tests/test_phase_75_5_1_spend_metric.py passes offline and asserts, against the REAL pricing table, that the metric feeding the budget gate reflects LLM spend (or, if arm (b) is chosen, that setting name/docs/consumers consistently say BigQuery and no consumer treats it as LLM spend)"
2. "If arm (a): the new LLM-spend path is flag-gated and the test proves flag-OFF is byte-identical to today's behaviour on the same inputs (no silent change to when the $25/day breaker trips)"
3. "Test asserts the guard still FAILS OPEN on a spend-fetch exception and that the phase-75.5 degradation counter/alert seam still fires (regression guard on arch-04)"
4. "Mutation matrix in experiment_results.md: swapping the metric source back, and removing the flag gate, each fail at least one test"

live_check spec (verbatim): "handoff/current/live_check_75.5.1.md: verbatim verification command output (exit 0) + git diff --stat + an ON-vs-OFF $0 diff showing the breaker's trip point is unchanged with the flag OFF. No live BQ billing query without owner approval."

## References

- `handoff/current/research_brief_75.5.1.md` (6 read-in-full: Anthropic + Gemini + BQ pricing pages, Fowler circuit-breaker, Azure circuit-breaker pattern, LLM budget-manager prior art)
- spend.py (arch-04 seam), cost_tracker.py:198-206 (cache-aware formula), sovereign_api.py:236-286 (cache-blind prior art — anti-pattern), api_call_log.py:221-295 (row schema, `ts` column, gauge doctrine), llm_client.py:395-458 (breaker), autonomous_loop.py:2298 + claude_code_client.py:502-504 (CC-rail row shapes)
- project_return_day_state_2026_07 (session_cost_usd gauge doctrine); test_phase_75_llm_rail.py:577-592 (seam pin)
