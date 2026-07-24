# Research Brief — Step 75.5.1: LLM-spend circuit breaker vs BigQuery-spend metric

**Tier:** moderate | **audit_class:** false | **P1 MONEY-ADJACENT** ($25/day circuit breaker)
**Researcher session started:** 2026-07-24
**Status:** IN PROGRESS (write-first; appended incrementally)

## Question (from step 75.5.1)

`backend/services/observability/spend.py::fetch_spend` prices
`INFORMATION_SCHEMA.JOBS_BY_PROJECT.total_bytes_billed` at $6.25/TiB = **BigQuery** spend,
but `settings.cost_budget_daily_usd` is documented as the **Daily LLM-spend cap** and all
three consumers (llm_client hard-block, /api/cost-budget, Slack watcher) treat it as LLM
spend. Decide:
- **Arm (a):** add a real LLM-spend source (price `pyfinagent_data.llm_call_log` rows against
  `cost_tracker.py::MODEL_PRICING`), flag-gated DARK with ON-vs-OFF comparison; OR
- **Arm (b):** rename setting+docs to match BQ reality.

Step text says (a) is likely correct (operator intent = LLM cost ceiling).

---

## Internal code inventory (file:line anchors)

### The guard chain — what the $25/day breaker actually measures

| Component | File:line | Role | What it measures |
|---|---|---|---|
| `fetch_spend()` | `backend/services/observability/spend.py:103-141` | Cloud-spend fetch (promoted from Slack job by 75.5/arch-04) | **BigQuery** bytes-billed × $6.25/TiB. NOT LLM spend (module docstring warns this at :10-18) |
| `_check_cost_budget()` | `backend/agents/llm_client.py:395-458` | **The $25/day hard-block.** Calls `fetch_spend()`, compares `daily_usd >= cost_budget_daily_usd`. Raises `BudgetBreachError`. Cached 60s (`_BUDGET_CACHE_TTL_S`). Fails open at :434-439 | Consumes BQ spend, named/documented as LLM cap |
| Hot-path call sites | `llm_client.py:896, 1180, 1428, 2214` | `_check_cost_budget()` fires before every `generate_content` (Gemini/Claude/OpenAI paths + advisor) | — |
| `cost_budget_daily_usd` | `backend/config/settings.py:384` | Field(25.0, "Daily **LLM-spend** cap across all cycles") | Documented as LLM; fed BQ bytes |
| `cost_budget_monthly_usd` | `backend/config/settings.py:385` | Field(300.0, "Monthly **LLM-spend** cap") | Same mismatch |
| `/api/cost-budget/today` | `backend/api/cost_budget_api.py:108-167` | Harness-tab tile. Calls `_default_fetch_spend` (=`fetch_spend`, BQ). **Hardcodes `_DAILY_CAP_USD=5.0/_MONTHLY_CAP_USD=50.0`** (:53-54) | BQ spend vs $5/$50 |
| Slack watcher | `backend/slack_bot/jobs/cost_budget_watcher.py:28-78` | APScheduler job; `fetch_fn or _default_fetch_spend` (BQ). Defaults `daily_cap_usd=5.0/monthly=50.0` (:33-34) | BQ spend vs $5/$50 |

**Three consumers, THREE different caps.** Hard-block uses $25/$300 (settings); tile + Slack watcher use $5/$50 (hardcoded). This cap-disagreement is a *related* defect — queue as its own step per `feedback_queue_discovered_defects_in_masterplan`; do NOT silently fix inside 75.5.1.

### The OTHER budget mechanism (LLM-cost-aware, but per-cycle only)
- `_check_session_budget()` `backend/services/autonomous_loop.py:103-116` — raises `BudgetBreachError` when in-process cumulative `_session_cost` (ACTUAL LLM cost from `cost_tracker`) crosses `_SESSION_BUDGET_USD`. This is LLM-cost-aware but **per-cycle / per-process**, resets each cycle, does NOT span cycles or restarts → cannot serve as the daily-across-cycles cap.

### llm_call_log — the candidate LLM-spend source (arm (a))
Schema (`scripts/migrations/add_llm_call_log.py` + `add_session_budget_to_llm_call_log.py` + writer `backend/services/observability/api_call_log.py:201-213, 279-295`):
`ts, provider, model, agent, latency_ms, ttft_ms, input_tok, output_tok, cache_creation_tok, cache_read_tok, request_id, ok, ticker, cycle_id, session_cost_usd`

- **NO per-call `cost_usd` column.** The only dollar field is `session_cost_usd`, a per-cycle cumulative **GAUGE** (`api_call_log.py:254-264`, phase-66.3) — **NEVER SUM it** (summing = staircase phantom spend; the 2026-06 outage read it as ~$42/day phantom). Confirmed by auto-memory `project_return_day_state_2026_07`.
- ⇒ LLM daily spend MUST be derived from **raw tokens × MODEL_PRICING at query time**, never from a stored cost.
- **Raw tokens are FIX-INVARIANT.** The 75.5 cache-token double-subtract fix (`cost_tracker.py:175-206`) changed COMPUTED cost, not the raw `input_tok/output_tok/cache_*_tok` stored in llm_call_log. So pricing raw tokens with the CURRENT corrected formula gives correct spend across the fix boundary — **Q2's "day-window spanning the fix" concern does NOT apply to a token-priced query.** (It WOULD taint any design that summed `session_cost_usd` — another reason to price tokens, not the gauge.)

### Prior art already in-repo (reuse candidate)
- `backend/api/sovereign_api.py:236-286` `_fetch_llm_cost_by_provider(window_days)` — ALREADY prices `llm_call_log` tokens × `MODEL_PRICING` (Python-side join, fail-open to zeros). **BUT it ignores cache columns**: `in_tok*rate_in + out_tok*rate_out` only. For a money guard this UNDER-counts cached-call cost → breaker trips LATE (overspend before trip). The accurate cache-aware formula (`cost_tracker.py:198-206`) adds `cache_read_tok*rate_in*0.1 + cache_creation_tok*rate_in*2.0`. Arm (a) should port the accurate formula, not the sovereign shortcut.
- `backend/api/cost_budget_api.py:71-96` `_fetch_llm_tokens_today()` — already runs `SELECT SUM(input_tok)+SUM(output_tok) ... WHERE DATE(ts)=CURRENT_DATE()` (the exact daily window we need), fail-open (None,None), 30s timeout. A pricing join layered on this = daily LLM spend.
- `backend/api/performance_api.py:49-98` — reads llm_call_log for p95 latency (query shape reference).

### cost_tracker pricing shape (arm (a) pricing source of truth)
- `backend/agents/cost_tracker.py:20-83` `MODEL_PRICING: dict[str,(in,out)]` per 1M tok; `_DEFAULT_PRICING=(0.10,0.40)`. `claude-opus-4-8=(5.00,25.00)` (:29); `gemini-2.5-flash=(0.30,2.50)` (:26); `gemini-2.5-pro=(1.25,10.00)` (:27).
- Cache-aware branch `cost_tracker.py:198-206`: cache_read=0.1× base-in, cache_creation=2.0× base-in (1h TTL). `is_batch` halves (:211-212).

### Fail-open + degradation seam (regression target, arch-04)
- `fetch_spend` fails open to (0.0,0.0), `_record_degradation()` increments `_DEGRADED_COUNT` + alerts once (`spend.py:67-101`); `spend_guard_status()`/`reset_spend_guard_status()` are test seams.
- Pinned by `backend/tests/test_phase_75_llm_rail.py:577-592` (`test_spend_guard_degradation_is_counted_and_alerted_not_just_logged`) + :595 negative control. Arm (a)'s `fetch_llm_spend()` sibling MUST replicate this seam or the regression criterion fails.

### DARK-flag convention (grep of settings.py)
`<name>_enabled: bool = Field(False, description="phase-X.Y: When True ... OFF -> byte-identical ...")`, read via `getattr(settings, "flag", default)`. Live exemplars: `sign_safe_overlays` (:37), `kill_switch_peak_reset_enabled` (:39), `paper_session_budget_reconcile_enabled` (:461), `paper_atomic_swap_enabled` (:458), `skill_modification_review_enabled` (:462). All default False; all say "OFF -> byte-identical".

---

## External research

### Read in full (>=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Fetched | Key finding |
|---|-----|----------|------|---------|-------------|
| 1 | https://ai.google.dev/gemini-api/docs/pricing | 2026-07-24 | Official doc (Google) | WebFetch full | Gemini 2.5 Flash paid = **$0.30 in / $2.50 out** per 1M; 2.5 Pro = **$1.25 in / $10.00 out** (≤200k), $2.50/$15 (>200k). CONFIRMS `MODEL_PRICING` :26-27 |
| 2 | https://www.metacto.com/blogs/anthropic-api-pricing-a-full-breakdown-of-costs-and-integration | 2026-07-24 | Industry (cites Anthropic) | WebFetch full | Opus 4.8 = **$5 / $25**; Sonnet 4.6 = $3/$15; Haiku 4.5 = $1/$5. Cache read **0.1×** in; cache write 1h **2.0×** in; Batch **50% off**. CONFIRMS `MODEL_PRICING` :29,34,36 + `cost_tracker.py:198-212` cache/batch math |
| 3 | https://www.stackscored.com/pricing/data-warehouse/bigquery/ | 2026-07-24 | Industry (live-tracked) | WebFetch full | BQ on-demand = **$6.25/TiB scanned**, first 1 TiB/mo free, cached queries free, verified 2026-04-21. CONFIRMS `spend.py:_BQ_USD_PER_TIB=6.25` |
| 4 | https://docs.litellm.ai/docs/proxy/cost_tracking | 2026-07-24 | Official doc (LiteLLM) | WebFetch full | Prior art: **cost = tokens × model-price map**; per-call `LiteLLM_SpendLogs` rows carry a `spend` column; daily aggregation via `/global/spend/report` grouped by day (UTC); pricing source = a JSON price map (their `MODEL_PRICING` analog) |
| 5 | https://martinfowler.com/bliki/CircuitBreaker.html | 2026-07-24 | Authoritative blog (Fowler) | WebFetch full | On-failure = **application decision** (fail the op OR serve stale/default); **"Any change in breaker state should be logged and breakers should reveal details of their state for deeper monitoring"** — directly grounds the arch-04 degradation counter/alert seam |
| 6 | https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-07-24 | Official doc (Microsoft, upd. 2026-07-02) | WebFetch full | Worked example is a **quota/budget-overrun breaker (Cosmos free tier)**: trips on quota signal, **degrades gracefully (default/cached response)**, alerts operators via Monitor, manual override to reset. "Open state can return a **default value** meaningful to the application" = the fail-open-to-(0,0) analog. "Provide clear observability into both failed and successful requests" |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| https://cloud.google.com/bigquery/pricing | Official BQ | SPA truncated on WebFetch (twice); $6.25/TiB corroborated via source #3 + web-search summary "verified 2026-04-26" |
| https://github.com/BerriAI/litellm/blob/main/docs/my-website/docs/budget_manager.md | Official LiteLLM | 404 on raw + docs host; mechanics covered by source #4 + search summary (BudgetManager: `duration:"daily"` cron-reset, `current_cost` per user, raises `BudgetExceededError`, `litellm.max_budget` global) |
| https://deepwiki.com/BerriAI/litellm/3.3-budget-and-spend-tracking | Community wiki | Redundant with #4 |
| https://www.finout.io/blog/claude-opus-4.8-pricing-2026-everything-you-need-to-know | Industry | Redundant Anthropic price confirm |
| https://openrouter.ai/google/gemini-2.5-flash | Industry | Redundant Gemini price confirm |
| https://www.cloudbees.com/blog/circuit-breaker-... | Industry | Redundant with Fowler/Azure |
| https://docs.litellm.ai/docs/proxy/provider_budget_routing | Official LiteLLM | Provider-level budgets (adjacent, not core) |
| (BQ: nops, yukidata, modern-datatools, cdcalculators, checkthat, costbench) | Industry | Price aggregators, all $5–6.25/TiB |
| (Anthropic: spheron, amnic, pricepertoken, benchlm, totalum, claudefast) | Industry | Price aggregators |
| (Gemini: burnwise, opslyft, tldl, costgoat, lmmarketcap, aipricing) | Industry | Price aggregators |

**Counts:** 6 read-in-full; ~20 catalogued snippet-only; **>40 unique URLs surfaced** across 5 topics.

### Pricing validation table (external ⇄ `cost_tracker.MODEL_PRICING`)
| Model | Repo `MODEL_PRICING` (in/out per 1M) | External source | Match? |
|-------|-----|-----------------|--------|
| claude-opus-4-8 | 5.00 / 25.00 (:29) | $5 / $25 (src #2) | ✅ |
| claude-sonnet-4-6 | 3.00 / 15.00 (:34) | $3 / $15 (src #2) | ✅ |
| claude-haiku-4-5 | 1.00 / 5.00 (:36) | $1 / $5 (src #2) | ✅ |
| gemini-2.5-flash | 0.30 / 2.50 (:26) | $0.30 / $2.50 (src #1) | ✅ |
| gemini-2.5-pro | 1.25 / 10.00 (:27) | $1.25 / $10 ≤200k (src #1) | ✅ (repo has no >200k tier; minor) |
| cache read mult | 0.1× base-in (:200) | 0.1× (src #2) | ✅ |
| cache write mult | 2.0× base-in, 1h TTL (:201) | 2.0× 1h (src #2) | ✅ |
| batch discount | 0.5× (:211) | 50% off (src #2) | ✅ |
| BQ on-demand | $6.25/TiB (spend.py:34) | $6.25/TiB (src #3) | ✅ |

**Verdict: every price feeding the guard is CURRENT and correct.** The defect is NOT the numbers — it is that the guard applies the *BigQuery* number to a cap *named* for LLM spend.

### Recency scan (last 2 years, 2024–2026)
Searched 2026 / 2025 / year-less variants for BQ pricing, Anthropic/Gemini pricing, and LLM-budget prior art. Findings: (a) BQ on-demand held at **$6.25/TiB** since 2023-07-05 through 2026 (no supersession). (b) Opus 4.8 launched 2026-05-28 at $5/$25 (a NEW *Fast Mode* $10/$50 tier exists — not used here; the standard tier the repo prices is unchanged). (c) LiteLLM budget/spend tracking is the current-frontier OSS prior art (2024→2026, still the reference implementation). (d) Circuit-breaker doctrine: Azure doc updated 2026-07-02 adds *adaptive/ML thresholds* and an explicit **budget-overrun** worked example — newer than Fowler's canonical 2014 piece, complements it. **No new finding supersedes the arm-(a) design; the recency scan strengthens it (the Azure budget-overrun breaker is the closest published analog).**

### Queries run (3-variant discipline)
- Frontier (2026): "BigQuery on-demand query pricing per TiB 2026"; "Anthropic Claude Opus 4.8 API pricing per million tokens"; "Google Gemini 2.5 Flash Pro API pricing per million tokens".
- Prior-art / year-less: "LiteLLM budget manager max daily budget circuit breaker LLM spend tracking"; "circuit breaker fail-open vs fail-closed doctrine spend guard".
- (Last-2-year window covered inside the recency scan above.)

---

## Design analysis

### (Q1) Offline/local-first vs BQ query on `llm_call_log`
| Source | Survives process restart? | Cross-process (backend+harness+slack)? | Cost to read | Verdict |
|--------|--------------------------|----------------------------------------|--------------|---------|
| `cost_tracker` in-proc counters | ❌ per-analysis, reset each run | ❌ | $0 | Cannot back a **daily-across-cycles** cap |
| `autonomous_loop._session_cost` gauge | ❌ per-cycle | ❌ | $0 | Per-cycle only; already the `_check_session_budget` source |
| NEW local ledger file (JSONL by UTC date) | ✅ if persisted | ❌ 3 procs → file-lock/drift | $0 | Would **reinvent `llm_call_log`** as a worse, un-shared store |
| **`llm_call_log` BQ query (priced)** | ✅ | ✅ (every rail already writes it) | small: date-partitioned, column-pruned, **60s-cached** like today's guard | **RECOMMENDED** |

The $25/day cap is explicitly "**across all cycles**" (settings.py:384) and this system restarts often + runs 3 separate processes. The durable, cross-process daily total **already exists** — it is `llm_call_log`. A local ledger would have to *become* that shared store and would duplicate/drift from it. The money-guard-costs-money objection is real but bounded: the current guard **already** runs a BQ query every 60s (INFORMATION_SCHEMA.JOBS, cached `_BUDGET_CACHE_TTL_S`); swapping/adding a `llm_call_log` scan keeps the same cadence, honors the 30s-timeout rule (existing sibling queries already pass `timeout=30`), and scans a tiny AI-telemetry table with a `DATE(ts)=CURRENT_DATE()` partition filter + column pruning → sub-TiB → free-tier/pennies.

### (Q2) Is the per-call cost trustworthy post-75.5?
**There is no per-call cost column.** The only dollar field is `session_cost_usd`, a per-cycle cumulative **GAUGE — never sum it** (`api_call_log.py:254-264`; auto-memory `project_return_day_state`). ⇒ price from **raw tokens**. Raw `input_tok/output_tok/cache_*_tok` are **fix-invariant** — the 75.5 cache-double-subtract fix changed COMPUTED cost, not stored tokens — so a token-priced query is correct **across the fix boundary**. Q2's "day-window spanning the fix mixes bad+good rows" concern **does not apply** to a token-priced query; it *would* apply to any design summing `session_cost_usd` (double-wrong: gauge + stale). This is a second reason to price tokens, not the gauge.

### 🔴 CRUX FINDING — flat-fee CC-rail rows must be EXCLUDED (else phantom spend)
The Claude-Code rail is **flat-fee Max** (real metered cost ≈ $0) yet **records tokens** to `llm_call_log` "for volume audits" with "**session_cost_usd delta 0**" (claude_code_client.py:494-496). CC-rail rows are tagged two ways:
- `provider="anthropic"` **+** `agent="cc_rail:<agent>"` (`claude_code_client.py:502-504`), and
- `provider="claude-code"` (`autonomous_loop.py:2297-2298`).

A naïve `SUM(tokens)×MODEL_PRICING` over ALL rows would price these flat-fee tokens at $5/$25 per 1M → **large phantom daily spend → the $25 breaker trips on tokens that cost nothing → false-positive halt of trading.** This is the SAME phantom class as the `session_cost_usd` staircase ($42/day phantom, auto-memory `project_return_day_state`/`project_ph72`; note credits dead since 05-17 → today's real LLM work is CC-rail flat-fee + Gemini/Vertex metered). **arm (a) MUST filter to metered-only:**
```sql
WHERE ok = TRUE
  AND provider != 'claude-code'
  AND (agent IS NULL OR agent NOT LIKE 'cc_rail:%')
```
Post-filter, the daily number is dominated by genuinely-metered **Gemini/Vertex** (real Google Cloud billing) + any direct Anthropic SDK / OpenAI — exactly the metered spend the cap should govern. This is WHY the step mandates a **DARK ON-vs-OFF comparison before any flip**: the comparison will reveal whether the priced query is sane or phantom-inflated.

### (Q3) Fail-open semantic for a MONEY guard
Criteria mandate fail-open preserved + degradation seam firing. Doctrine backing: Fowler — breaker on-failure is an application decision and **state changes must be logged/observable**; Azure — the Open state "can **return a default value** meaningful to the application" and must "provide clear observability into both failed and successful requests" + manual override. pyfinagent's choice (fail-open to (0,0), **but LOUD** via `_record_degradation` counter + one-shot P2 alert) is the textbook realization: never halt trading on a broken meter, but make the silent-open **observable** so an operator can act. arm (a)'s new `fetch_llm_spend()` MUST route its exceptions through the **same** `_record_degradation` seam so arch-04's regression guard (`test_phase_75_llm_rail.py:577-592`) still holds for whichever source is active.

### (Q4) Flag name + DARK default convention
Convention (grep of settings.py): `<name>_enabled: bool = Field(False, description="phase-X.Y: When True ... OFF -> byte-identical ...")`, read via `getattr(settings, flag, False)`. Live exemplars: `sign_safe_overlays`, `kill_switch_peak_reset_enabled`, `paper_atomic_swap_enabled`, `skill_modification_review_enabled`.

---

## PLAN recommendations (arm choice + flag + query + test design)

1. **Choose arm (a)** — add a real LLM-spend source; keep `fetch_spend` (BQ) as a separate metric. Rejecting arm (b): renaming to "BigQuery cap" would abandon the operator's actual intent (an LLM cost ceiling) and leave the metered-spend runaway (the thing credits-era $25 caps existed to stop) ungoverned.
2. **New `fetch_llm_spend() -> (daily_usd, monthly_usd)`** in `backend/services/observability/spend.py` (sibling of `fetch_spend`). Price `llm_call_log` **raw tokens × `MODEL_PRICING`** with the **cache-aware** formula ported from `cost_tracker.py:198-206` (NOT the sovereign_api shortcut, which ignores cache cols and under-counts → trips late). Windows: daily `DATE(ts)=CURRENT_DATE()`, monthly `ts >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)`. **Metered-only filter** (the crux above). `timeout=30`, column-pruned, fail-open to (0,0) **through the shared `_record_degradation` seam**.
3. **Flag:** `cost_budget_use_llm_spend_enabled: bool = Field(False, ...)` (default OFF = byte-identical BQ-spend path). In `_check_cost_budget` (`llm_client.py:429-433`): `source = fetch_llm_spend if getattr(settings,"cost_budget_use_llm_spend_enabled",False) else fetch_spend`. Nothing else in the hot path changes.
4. **Known conservative biases to document (not block on):** `llm_call_log` has no `is_batch` column → batched rows priced at full (over-count, trips EARLY = safe direction); advisor blended-model rows priced at the row's single model (approximation). Both err toward over-counting = safe for a money guard; note them in the module docstring.
5. **Test `backend/tests/test_phase_75_5_1_spend_metric.py` (offline):**
   - Price a **fixture row set against the REAL `MODEL_PRICING`** (import it, don't hardcode) → assert daily LLM $ equals hand-computed cache-aware total.
   - **Metered-scope fixture (measure-don't-assert):** include CC-rail rows (`provider='claude-code'` and `provider='anthropic'`+`agent='cc_rail:x'`) that MUST contribute ~$0, plus metered Gemini/Anthropic-SDK rows that MUST be counted. A fixture that can't represent both categories doesn't count (auto-memory `mutation_test_guards_and_fixtures`).
   - **Flag OFF byte-identical:** monkeypatch both fetchers; assert OFF routes to `fetch_spend` and the trip point is unchanged on identical inputs (ON-vs-OFF $0 diff).
   - **Fail-open + arch-04 regression:** exception in `fetch_llm_spend` → returns (0,0) AND `_record_degradation` fires (`spend_guard_status().degraded_count==1`, `alerted==True`).
   - **Mutation matrix (experiment_results.md):** (i) swap the metric source back to BQ under flag-ON → a test FAILS; (ii) remove the flag gate → a test FAILS; (iii) **drop the CC-rail exclusion** → the phantom-spend test FAILS (this is the highest-value mutation — it guards the crux). Mutate the fixture/stub too, not just the guard.
6. **Queue-don't-fix (separate masterplan steps, per `feedback_queue_discovered_defects`):** (a) the **three-caps disagreement** — hard-block $25/$300 vs tile+Slack-watcher hardcoded $5/$50 (`cost_budget_api.py:53-54`, `cost_budget_watcher.py:33-34`); (b) whether Gemini/Vertex-only metered spend warrants its own sub-cap. Do NOT fold these into 75.5.1.

---

## Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6)
- [x] 10+ unique URLs total (>40)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- Soft: internal exploration covered all consumers + writer + tests + prior art; consensus/contradiction noted (sovereign shortcut vs cache-aware); per-claim citations inline.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 20,
  "urls_collected": 42,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {"audit_class": false, "rounds": 1, "dry_rounds": 0, "K_required": 2, "new_findings_last_round": 0, "dry": false},
  "summary": "Arm (a) is correct: add fetch_llm_spend() pricing llm_call_log RAW tokens x MODEL_PRICING (cache-aware, ported from cost_tracker.py:198-206), flag-gated cost_budget_use_llm_spend_enabled default OFF (byte-identical BQ path), keeping fetch_spend (BQ) as a separate metric. All prices validated current ($6.25/TiB, opus-4-8 $5/$25, gemini-2.5 $0.30/$2.50 & $1.25/$10). CRUX: flat-fee CC-rail rows (provider='claude-code' OR agent LIKE 'cc_rail:%', session_cost_usd delta 0) MUST be excluded or the guard prices free tokens at API rates -> phantom spend -> false halt. Price tokens not session_cost_usd (gauge, never-sum, and fix-invariant so no 75.5-boundary taint). Fail-open preserved through the shared _record_degradation seam (arch-04 regression). No local ledger: llm_call_log is the existing durable cross-process daily total. Test asserts real-pricing math, metered-scope fixture, flag-OFF byte-identical, fail-open+degradation, and a mutation matrix whose highest-value case is dropping the CC-rail exclusion.",
  "brief_path": "handoff/current/research_brief_75.5.1.md",
  "gate_passed": true
}
```

