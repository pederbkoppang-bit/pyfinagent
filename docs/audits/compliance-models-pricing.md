# Compliance Audit: Models / Pricing / Deprecations / Service Tiers / Data Residency

**Phase:** 4.15.11  
**Date:** 2026-04-18  
**Auditor:** Researcher agent (merged)  
**Sources:** Anthropic docs (models/overview, pricing, service-tiers, data-residency, migration-guide); internal `cost_tracker.py`, `llm_client.py`, `model_tiers.py`, `agent_definitions.py`, `autonomous_loop.py`, `mcp_tools.py`, `app_home.py`

---

## Pattern Table (25 patterns)

| # | Pattern / Finding | File(s) | Status | Doc Reference |
|---|---|---|---|---|
| P-01 | `claude-3-5-haiku-20241022` in MODEL_PRICING | `cost_tracker.py:23` | **RETIRED** — will 400 after 2026-04-19 (tomorrow). Haiku 3.5 not listed in Anthropic current pricing page; pricing shown is for Haiku 3.5 ($0.80/$4.00 input/output), which still matches, but the API ID is the concern. | [models/overview — Haiku 3 retirement warning] |
| P-02 | `claude-3-5-haiku-20241022` in GITHUB_MODELS_CATALOG | `llm_client.py:64` | **RETIRED** — GitHub Models routes this via `anthropic/claude-3.5-haiku`; the endpoint may still accept it but the underlying model retires 2026-04-19. Will 400 at Anthropic direct. | [models/overview] |
| P-03 | `claude-3-5-haiku-20241022` in GITHUB_MODELS_ID_MAP, harness_memory.py, settings_api.py | `llm_client.py:171`, `harness_memory.py:53`, `settings_api.py:31,144` | **RETIRED** — same retirement risk, 4 additional sites. | [models/overview] |
| P-04 | `claude-3-5-sonnet-20241022` in MODEL_PRICING, GITHUB_MODELS_CATALOG, ID_MAP, harness_memory.py, settings_api.py | `cost_tracker.py:24`, `llm_client.py:63,170`, `harness_memory.py:52`, `settings_api.py:31,143` | **RETIRED** — Claude 3.5 Sonnet (Oct 2024 snapshot) not listed as available in current docs. Will 400. Migration target: `claude-sonnet-4-6`. | [models/overview — legacy models section] |
| P-05 | `claude-3-7-sonnet-20250219` in MODEL_PRICING, GITHUB_MODELS_CATALOG, ID_MAP, harness_memory.py, settings_api.py | `cost_tracker.py:25`, `llm_client.py:65,172`, `harness_memory.py:54`, `settings_api.py:32,145` | **DEPRECATED** — Claude Sonnet 3.7 is deprecated per pricing page ("deprecated" label on every table row). Retirement date not specified but not in models/overview latest or legacy tables. Will eventually 400. Migration target: `claude-sonnet-4-6`. | [pricing page — Sonnet 3.7 deprecated label] |
| P-06 | `claude-sonnet-4-20250514` hardcoded in autonomous_loop.py | `autonomous_loop.py:438` | **STALE SNAPSHOT** — `claude-sonnet-4-20250514` (alias `claude-sonnet-4-0`) is deprecated, retires 2026-06-15. 59 days from now. Confirmed by models/overview warning. Migration target: `claude-sonnet-4-6`. | [models/overview — deprecation warning] |
| P-07 | `claude-sonnet-4-20250514` in mcp_tools.py and app_home.py | `mcp_tools.py:74,223`, `app_home.py:23` | **STALE SNAPSHOT** — same retirement risk as P-06. Three additional callsites. | [models/overview — deprecation warning] |
| P-08 | `claude-opus-4` (alias, no snapshot) in MODEL_PRICING | `cost_tracker.py:27` | **WRONG PRICING** — `claude-opus-4` is the alias for deprecated `claude-opus-4-20250514`. Anthropic current pricing: $15 input / $75 output. Model_PRICING records $15.00/$75.00 — pricing is correct but the alias is for a deprecated model. The current flagship is `claude-opus-4-7` ($5/$25) or `claude-opus-4-6` ($5/$25). Using the old opus-4 alias will still work until 2026-06-15 but under-charges by 3x vs flagship cost. | [pricing page] |
| P-09 | `claude-opus-4-7` missing from MODEL_PRICING | `cost_tracker.py` | **MISSING** — Claude Opus 4.7 ($5 input / $25 output) is the current recommended model for complex tasks. Not in MODEL_PRICING; any call tracked under this ID falls through to `_DEFAULT_PRICING` ($0.10/$0.40), a 50x / 62.5x under-report. Maps to **MF-7** (cost under-reporting). | [pricing page; models/overview] |
| P-10 | `claude-opus-4-6` missing from MODEL_PRICING | `cost_tracker.py` | **MISSING** — Claude Opus 4.6 ($5/$25) is the current production model used in `_BUILD_TIER` for `mas_main`, `mas_qa`, and `autoresearch_strategic`. Not in MODEL_PRICING; falls through to $0.10/$0.40 — 50x / 62.5x cost under-report on every MAS Opus call. Maps to **MF-7**. | [pricing page] |
| P-11 | `claude-haiku-4-5` missing from MODEL_PRICING | `cost_tracker.py` | **MISSING** — Claude Haiku 4.5 ($1/$5) used in `autoresearch_fast` and `app_home.py` AVAILABLE_MODELS. Falls through to $0.10/$0.40 — 10x / 12.5x cost under-report. Maps to **MF-7**. | [pricing page] |
| P-12 | `claude-sonnet-4-6` pricing correct | `cost_tracker.py:28` | PASS — $3.00/$15.00 matches Anthropic doc. No action needed. | [pricing page] |
| P-13 | `claude-opus-4-6` in _BUILD_TIER (mas_main, mas_qa) | `model_tiers.py:46,48` | PASS — `claude-opus-4-6` is a valid, current (legacy) model per models/overview. Pricing $5/$25. No retirement date announced. | [models/overview — legacy models] |
| P-14 | `claude-haiku-4-5` in _BUILD_TIER (autoresearch_fast) with `anthropic:` prefix | `model_tiers.py:52` | INFO — `anthropic:claude-haiku-4-5` uses a provider-prefix format not recognized by ClaudeClient routing. `make_client()` checks `model_name.startswith("claude-")` after stripping; the `anthropic:` prefix would fail this check and fall through to Gemini. Needs normalization. | [llm_client.py:720] |
| P-15 | `_LIVE_TIER` all sentinel `TODO_DECIDE_AT_LAUNCH` | `model_tiers.py:67` | INTENTIONAL — documented design decision. Will raise RuntimeError if `COST_TIER=live` activated. Must be populated before May 2026 go-live. Maps to **MF-1** (launch readiness). | [model_tiers.py comments] |
| P-16 | `service_tier` never set anywhere in backend | `backend/ grep` | **GAP** — Default is `"auto"` per Anthropic docs, meaning Priority Tier is used when capacity is available. We never log `response.usage.service_tier` so we cannot tell whether we are burning Priority Tier capacity or Standard. Maps to **MF-8** (observability). | [service-tiers doc] |
| P-17 | `response.usage.service_tier` never logged | `cost_tracker.py`, `llm_client.py` | **GAP** — ClaudeClient extracts `input_tokens` and `output_tokens` from usage but does not capture `service_tier`. Same for `OpenAIClient`. This means cost_tracker never records whether a call ran Priority or Standard. Maps to **MF-8**. | [service-tiers doc — response usage object] |
| P-18 | `inference_geo` never set in any ClaudeClient call | `llm_client.py:630` | **GAP** — Default is `"global"`. Since we handle financial data, US-only routing may be preferable for data-residency compliance. The parameter is supported on Opus 4.6+. Not setting it means inference may run outside the US. No action required unless residency policy mandates it, but should be a conscious decision. Maps to **MF-8**. | [data-residency doc] |
| P-19 | `response.usage.inference_geo` never logged | `llm_client.py`, `cost_tracker.py` | **GAP** — Even if inference_geo is left as global, we should log where inference actually ran. The response `usage.inference_geo` field is available. Maps to **MF-8**. | [data-residency doc — response object] |
| P-20 | `anthropic-version` header not explicitly set | `llm_client.py` — ClaudeClient uses `anthropic` SDK | INFO — The Python `anthropic` SDK sets `anthropic-version: 2023-06-01` automatically. No code change needed, but worth noting the pinned version. Current SDK (>=0.49.0 per install comment) is up to date. | [api/versioning doc] |
| P-21 | `claude-3-haiku-20240307` (Haiku 3) mentioned nowhere in backend Python | grep result empty | PASS — No live code references to Haiku 3. Only concern is P-01/P-03 (Haiku 3.5, not Haiku 3). | [models/overview — Haiku 3 retirement 2026-04-19] |
| P-22 | `claude-sonnet-4` (naked alias) in MODEL_PRICING | `cost_tracker.py:26` | INFO — `claude-sonnet-4` is the alias for deprecated `claude-sonnet-4-20250514`. Pricing $3/$15 is correct for the model but this alias maps to a deprecated endpoint retiring 2026-06-15. Any call using the bare alias will break post-retirement. | [models/overview — deprecation warning] |
| P-23 | Opus 4.1 missing from MODEL_PRICING | `cost_tracker.py` | INFO — `claude-opus-4-1` ($15/$75 per pricing page) is not in MODEL_PRICING. Currently not used in code, but if added to `_LIVE_TIER` it would under-report at 187.5x. Preemptive add recommended. | [pricing page] |
| P-24 | `claude-haiku-35-20241022` in app_home.py AVAILABLE_MODELS | `app_home.py:24` | **INVALID ID** — `claude-haiku-35-20241022` is not a recognized Anthropic model ID. The correct ID is `claude-3-5-haiku-20241022` (deprecated, see P-01) or `claude-haiku-4-5`. This will 400 immediately if selected. | [models/overview] |
| P-25 | Prompt caching cost multiplier hardcoded at 0.1 (10%) | `cost_tracker.py:133` | PARTIALLY CORRECT — Anthropic docs show cache reads at 0.1x (10% of base input, i.e., 90% discount). The code comment says "90% discount" and multiplies by 0.1, which is correct for cache reads. However, the code does not account for cache write cost (1.25x for 5-min TTL, 2x for 1-hour TTL). Cache write tokens incur a premium that is never charged in the cost estimator. Minor under-report on cache-write-heavy runs. | [pricing page — prompt caching] |

---

## Critical Findings Summary

### Immediate (retire tomorrow: 2026-04-19)
- **P-01, P-02, P-03**: `claude-3-5-haiku-20241022` in MODEL_PRICING, GITHUB_MODELS_CATALOG, ID_MAP, harness_memory, settings_api. This ID retires **tomorrow**. Any code path that tries to call Anthropic direct with this ID will get a 400.

### Urgent (retire 2026-06-15, 59 days)
- **P-06, P-07**: `claude-sonnet-4-20250514` hardcoded in `autonomous_loop.py:438`, `mcp_tools.py:74,223`, `app_home.py:23`. Four callsites. These are live execution paths, not just catalog entries.
- **P-08, P-22**: `claude-opus-4` and `claude-sonnet-4` bare aliases also map to the deprecated 20250514 snapshots.

### Cost under-reporting (MF-7)
- **P-09**: `claude-opus-4-7` missing from MODEL_PRICING — 50x/62.5x under-report.
- **P-10**: `claude-opus-4-6` missing from MODEL_PRICING — 50x/62.5x under-report on every MAS Opus call in production.
- **P-11**: `claude-haiku-4-5` missing from MODEL_PRICING — 10x/12.5x under-report.

### Observability gaps (MF-8)
- **P-16, P-17**: `service_tier` never sent or logged. We cannot tell if runs consume Priority Tier capacity.
- **P-18, P-19**: `inference_geo` never set or logged. Data residency of inference is unknown.

### Launch readiness (MF-1)
- **P-15**: `_LIVE_TIER` is all sentinels. Must be populated before `COST_TIER=live` is set at May go-live.
- **P-14**: `anthropic:claude-haiku-4-5` prefix in `_BUILD_TIER` will fail `make_client()` routing and silently fall through to Gemini.

---

## Correct Pricing Reference (as of 2026-04-18)

| Model ID | Input $/MTok | Output $/MTok | Cache Read | Status |
|---|---|---|---|---|
| `claude-opus-4-7` | $5.00 | $25.00 | $0.50 | Current — recommended flagship |
| `claude-opus-4-6` | $5.00 | $25.00 | $0.50 | Legacy — current production |
| `claude-opus-4-5` | $5.00 | $25.00 | $0.50 | Legacy |
| `claude-opus-4-1` | $15.00 | $75.00 | $1.50 | Legacy |
| `claude-sonnet-4-6` | $3.00 | $15.00 | $0.30 | Current |
| `claude-sonnet-4-5` | $3.00 | $15.00 | $0.30 | Legacy |
| `claude-haiku-4-5` | $1.00 | $5.00 | $0.10 | Current |
| `claude-haiku-3-5` | $0.80 | $4.00 | $0.08 | Legacy |
| `claude-sonnet-4-20250514` | $3.00 | $15.00 | — | **DEPRECATED** retire 2026-06-15 |
| `claude-opus-4-20250514` | $15.00 | $75.00 | — | **DEPRECATED** retire 2026-06-15 |
| `claude-3-haiku-20240307` | $0.25 | $1.25 | — | **DEPRECATED** retire 2026-04-19 |

---

## Masterplan Finding Cross-Reference

| Finding | MF Tag | Severity |
|---|---|---|
| Opus 4-6 / Haiku 4-5 / Opus 4-7 missing from MODEL_PRICING | MF-7 | High |
| service_tier not sent or logged | MF-8 | Medium |
| inference_geo not set or logged | MF-8 | Medium |
| _LIVE_TIER all sentinels pre-launch | MF-1 | High |
| Retired IDs in GITHUB_MODELS_CATALOG (will 400) | MF-1 | Critical (tomorrow) |
| Stale snapshot IDs in live call sites | MF-1 | High (59 days) |
| anthropic: prefix in _BUILD_TIER silently re-routes | MF-1 | Medium |
