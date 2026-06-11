# Research Brief — phase-60.1: Deep-pipeline restoration + honest-degradation alarm (AW-4, P0)

Tier: COMPLEX (caller-stated). Date: 2026-06-11. Agent: researcher (Layer-3 MAS).
Disclosed overrun: audit tables push past the 1500-word ceiling; prose kept tight.

## 1. Executive summary

- **Migration target:** repin to **`gemini-3.1-flash-lite`** (Google's NAMED replacement for the discontinued gemini-2.0-flash; $0.25/$1.50 per Mtok) with **`gemini-2.5-flash`** ($0.30/$2.50) as the proven-family fallback — final pick is whichever passes the immutable live-smoke on the existing `vertexai` SDK rail; if 2.5-flash is chosen, schedule re-migration before its 2026-10-16 retirement (sources: docs.cloud.google.com gemini/2-0-flash page, accessed 2026-06-11; ai.google.dev/gemini-api/docs/pricing, accessed 2026-06-11; gcpstudyhub.com 2.5-retirement brief).
- **KR design choice:** **honest tagged-skip now, DART deferred** — the single hard abort is one CIK helper in the quant Cloud Function (`functions/quant/main.py:88`); make its SEC-companyfacts leg market-aware (skip for non-US, keep the yfinance leg + all 26 market-agnostic agents) and disclose per cycle; OpenDART (free key, ~10,000 req/day, corpCode.xml ticker mapping, English portal) is a viable FUTURE ingestion path but a new-surface build, not a P0 restoration (sources: engopendart.fss.or.kr/intro/main.do, accessed 2026-06-11; dart-fss docs; xbrl.org Feb-2025 English-platform note).

## 2. External findings

### A. Vertex AI Gemini model lifecycle

1. **gemini-2.0-flash is discontinued.** "As of June 1, 2026, `gemini-2.0-flash-001` and `gemini-2.0-flash-lite-001` are discontinued and are no longer available", including model serving and Provisioned Throughput; recommended replacements named by Google: "Gemini 3.1 Flash-Lite, Gemma 4", or more recent Gemini releases (docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash, read in full 2026-06-11). This matches backend.log exactly: 404s from 06-02, llm_call_log 88/day (05-27) -> 0 (06-02).
2. **Alias semantics caused the 404.** "The auto-updated alias of a Gemini model always points to the latest stable model"; `gemini-2.0-flash` -> `gemini-2.0-flash-001` (originally scheduled retirement 2026-02-05, one year post-release; actually landed 2026-06-01) (blevinscm.github.io/genai-docs mirror of the model-versions page, read in full 2026-06-11; live 2-0-flash page above). Pinning the bare alias of a retired family = guaranteed future 404.
3. **Currently served (June 2026) per Google's models index:** `gemini-3-1-pro`, `gemini-3-pro`, `gemini-3.5-flash`, `gemini-3-flash`, `gemini-3-1-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` (+image/live variants) (docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models, read in full 2026-06-11). NOTE: doc slugs mix `3-1` and `3.5` styles — GENERATE must confirm the exact publisher-path ID via the smoke call, not docs alone.
4. **2.5-family clock is ticking:** Gemini 2.5 Pro / Flash / Flash-Lite retirement updated to **2026-10-16** (earliest; >=6 months notice once Gemini 3 GA dates settle) (gcpstudyhub.com retirement brief + Google model-versions search capture). One capture lists stable `gemini-2.5-flash-lite` retiring **2026-07-22** — CONFLICTING with the Oct-16 date; treat 2.5-flash-lite as short-runway until the live page is re-checked during GENERATE.
5. **Pricing deltas (USD per Mtok in/out, official):** 2.0-flash $0.10/$0.40 (= `cost_tracker.py:22` today); **2.5-flash-lite $0.10/$0.40 (burn-parity)**; **2.5-flash $0.30/$2.50**; **3.1-flash-lite $0.25/$1.50**; 3-flash-preview $0.50/$3.00; 3.5-flash $1.50/$9.00 (too hot for a 28-call pipeline); 2.5-pro $1.25/$10 (<=200k) (ai.google.dev/gemini-api/docs/pricing, read in full 2026-06-11).
6. **Grounding + structured output survive.** Google Search grounding is priced/available for "Gemini 2.5 & 3 Models: 1,500 RPD (free...), then $35/1,000 grounded prompts" (Gemini 3: 5,000/mo free then $14/1k) — grounding exists on both candidate families (same pricing page). Structured output (controlled generation) is a platform capability across served Gemini models; verify on the smoke since the Layer-1 pipeline depends on JSON-schema enforcement (CLAUDE.md).
7. **SDK risk:** the `vertexai.generative_models` SDK is deprecated; "SDK releases after June 24, 2026 won't include the deprecated modules" — migrate to google-genai eventually; Gemini 3 adds thought-signature semantics (docs.cloud.google.com genai-vertexai-sdk migration page + therouter.ai, snippets). The project's Gemini rail uses `GenerativeModel` (orchestrator) — 2.5 family is PROVEN on this stack today (deep_think `gemini-2.5-pro` runs live, settings.py:30), Gemini 3.x through the legacy SDK is UNPROVEN here -> exactly why the criterion demands a live smoke.

### B. Korean filing sources (DART / OpenDART / KRX)

1. **OpenDART open API** (FSS): free authentication key on registration; provides original filings (XML), periodic-report key data, financial statements/major accounts, equity disclosures; English developer portal exists (engopendart.fss.or.kr/intro/main.do, read in full 2026-06-11 — intro page does not publish limits; corroborated: free key, **10,000 requests/day** limit, ~83 endpoint types, per practitioner/integration docs: mcpservers.org OpenDART server; clawhub skill page).
2. **Ticker mapping:** OpenDART keys companies by an 8-digit `corp_code`, NOT the 6-digit KRX ticker; full mapping downloadable as zipped `CORPCODE.xml` via `/api/corpCode.xml?crtfc_key=...` (~120k entries incl. stock_code field) (dart-fss.readthedocs.io corp_code module + github.com/seokhoonj/opendart, snippets). Mature Python lib exists (`dart-fss`, PyPI).
3. **English coverage is expanding but partial:** from 2025-02-10 an English open-data platform serves 83 disclosure types; English disclosure mandatory for large KOSPI firms in phases (2024-25, then post-2026) (xbrl.org news + kedglobal.com, snippets; englishdart.fss.or.kr live portal). Filings BODY text remains largely Korean — an LLM-RAG path would ingest Korean documents.
4. **Implication:** integration cost = new CF/ingestion surface + corp_code mapping + Korean-text RAG; clearly viable later, but the quant CF's SEC leg can be cleanly skipped for non-US NOW with yfinance fundamentals already merged in (orchestrator.py:951-953) — the 50.x multi-market work already validated yfinance for .KS.

### C. Graceful degradation / fallback alarms

1. **Canonical (SRE Workbook ch.5, read in full 2026-06-11):** multiwindow multi-burn-rate is the gold standard (page at 14.4x burn/1h+5m, 6x/6h+30m; ticket at 1x/3d; short window = 1/12 of long). Crucially for pyfinagent: the **low-traffic section** — "If a system receives 10 requests per hour, a single failed request = 10% hourly error rate... a 1,000x burn rate"; remedies: aggregate, lower-severity channels, or adjust the window (sre.google/workbook/alerting-on-slos/).
2. **Application:** the loop runs ~1 cycle/day with 5-15 analyses -> burn-rate windows degenerate; the natural window IS the cycle. Per-cycle fallback-rate >= threshold (default 50% per step spec) as a P1 page + an ALWAYS-ON digest provenance line as the "ticket" tier mirrors the workbook's page-vs-ticket split without fatigue.
3. **2024-26 practice:** LLM-router observability treats **fallback-activation-rate as a first-class alert (>5% of requests baseline)** and logs which model served each request, why primary failed, and cost per step (Portkey/getmaxim.ai + buildmvpfast.com, snippets 2025-26); FutureAGI 2026 field guide adds a quality-floor evaluator gating the fallback route against "silent degradation" — supports persisting per-ticker failure REASONS, not just counts (futureagi.com 2026, snippet).

## 3. Recency scan (last 2 years, 2024-2026)

Performed (all three topics). Findings: (a) the entire topic-A lifecycle evidence is 2025-2026 by nature (discontinuation landed 2026-06-01; Gemini 3 family rolling out since Nov 2025; SDK module removal after 2026-06-24); (b) topic B: 2024-2026 Korean English-disclosure expansion (Feb-2025 platform, 83 types; phased mandates) materially improves the future-DART case; (c) topic C: 2025-2026 LLM-gateway practice (fallback-rate alerts, per-request model provenance, quality-floor gates) complements — does not supersede — the canonical 2018 SRE workbook; the workbook's low-traffic caveat remains the binding constraint for a 1-cycle/day batch system.

## 4. Search queries run (3-variant discipline)

- A: `gemini-2.0-flash retirement Vertex AI 2026 "404 Publisher Model"` (2026) | `Vertex AI Gemini model versions deprecations retirement dates gemini-2.5-flash 2025` (last-2-yr) | `Vertex AI Gemini model lifecycle deprecation migration guide` (year-less).
- B: `OpenDART API Korean filings fundamentals corp code API key rate limit 2026` | `DART open API dart.fss.or.kr English disclosure API 2025` | `map KRX ticker to DART corp_code OpenDART corpCode.xml` (year-less).
- C: `silent degradation LLM pipeline fallback rate alerting observability 2026` | `LLM router fallback monitoring graceful degradation production 2025` | `Google SRE workbook alerting on SLOs burn rate alert fatigue` (year-less).

## 5. Internal audit findings (file:line, HEAD = main 2026-06-11)

1. **Retired pins (live code):** `backend/config/model_tiers.py:71` (`"gemini_enrichment": "gemini-2.0-flash"`) and `:81` (`"layer1_swappable"`) — NOTE: step-context's 63/73 drifted after 59.1 edits; `backend/agents/orchestrator.py:382` (`_GEMINI_FALLBACK = "gemini-2.0-flash"`, +comment :443); `backend/config/settings.py:35` (apply_model_to_all_agents description asserts Gemini-locked roles "still use their hardcoded gemini-2.0-flash"); `backend/agents/multi_agent_orchestrator.py:237,241,243` (MAS Gemini fallback client); `backend/autonomous_loop.py:75` (Layer-3 harness evaluator_model default); `backend/agents/evaluator_agent.py:88`; `backend/agents/harness_memory.py:48,322,503`; `backend/agents/cost_tracker.py:22` (pricing row — keep for history, add new model rows); `backend/meta_evolution/directive_review.py:159`; `backend/meta_evolution/directive_rewriter.py:202`; `backend/slack_bot/mcp_tools.py:204`; `backend/api/agent_map.py:119,131,156`. Full inventory in §6.
2. **Silent fallback:** `backend/services/autonomous_loop.py:1529-1541` (step-context's "1411-1419" is a stale anchor). `except Exception as e: logger.warning("Full orchestrator failed for %s: %s -- falling back to lite Claude analyzer", ...)` then last-resort `_select_lite_analyzer(...)`. Funnel catches EVERYTHING: per-agent `TimeoutError` (orchestrator.py:765), `RuntimeError("orchestrator returned empty report")` (:1489 region), quant/ingestion CF `RuntimeError("ERROR:...")`, Vertex NotFound 404. **Per-ticker failure reason exists only in the log line — never captured into the analysis dict, summary, or BQ.** That is the alarm's missing input.
3. **KR abort:** `functions/quant/main.py:88` — `raise ValueError(f"Ticker {ticker} not found in SEC CIK mapping.")` (CIK map fetch :52-78 from sec.gov company_tickers.json :39). Streamed as `ERROR:` -> `backend/agents/orchestrator.py:936-948` `run_quant_agent` raises RuntimeError; quant is a HARD step-2 dependency (`report["quant"] = await self.run_quant_agent(ticker)` ~:1536). The ingestion CF is also SEC-based but BEST-EFFORT since phase-27.6.6 (orchestrator.py:1511-1533 warn+continue). CIK truly needed by: quant CF's SEC-companyfacts leg + ingestion/RAG filing corpus. Market-agnostic: yfinance merge (orchestrator.py:951-953), market intel, all debate/synthesis/risk/macro LLM agents. Other SEC consumers (not in the abort path): `backend/tools/sec_insider.py:17-46`, `backend/alt_data/f13.py:46-50`.
4. **56.2 guard:** `backend/services/autonomous_loop.py:898-925` (cycle-level, post-gather) + pure predicate `_degraded_scoring_check` `:1669-1696` (fire when ALL degraded or >=3 zero-scored). Alert path: `from backend.services.alerting import raise_cron_alert` — actually resolved at `backend/services/observability/alerting.py:119` — P1, `error_type="degraded_scoring"`, stamps `summary["degraded"]`. **Wire the fallback-rate alarm in the same block**: second predicate over `_path` counts + captured reasons, distinct `error_type="fallback_rate"`, same raise_cron_alert, AlertDeduper precedent at :1417-1428 (drawdown tiers).
5. **Slack plumbing to reuse:** `backend/services/observability/alerting.py:119` `raise_cron_alert` (async) / `:185` `raise_cron_alert_sync`. Already used by the 56.2 guard and the P3 cycle summary (autonomous_loop.py:1398).
6. **Provenance:** persisted but invisible. `_persist_analysis` `backend/services/autonomous_loop.py:2180+` "Reads `_path` ... for honest source tagging in the persisted row (lite vs full)" and writes `standard_model=full_report.get("source")`; persist gate `:875-877` (`_path in ("lite","full")`); full path stamps `"_path": "full"` + `full_report.source/rail` at :1517-1526. NOT surfaced anywhere: digest Recent Analyses renders ticker/score/recommendation only (`backend/slack_bot/formatters.py:384-397`); reports API has zero lite/_path references (`backend/api/reports.py`); frontend `frontend/src/app/reports/page.tsx`. GENERATE: confirm the exact BQ column name for the tag inside `_persist_analysis` (:2180-2240) before wiring UI.
7. **90s timeout:** `backend/agents/orchestrator.py:679` `_generate_with_retry(..., max_retries=3, timeout: int = 90)`; enforcement `future.result(timeout=timeout)` :722; raise `:762-765`.
8. **Vertex smoke script: none exists.** Repo-wide grep for one-shot Gemini callers under `scripts/` matches only `scripts/add_phase_27.py` (a masterplan editor). Precedent: phase-26 ad-hoc live smoke (handoff/harness_log.md:18686). GENERATE must add e.g. `scripts/debug/smoke_vertex_model.py` (shape precedent: `scripts/mcp_servers/smoke_test_bigquery_mcp.py`).

## 6. Full gemini-2.0-flash occurrence inventory (live consumers; 267 raw hits incl. archives/changelogs)

| File:line | Kind | Action at GENERATE |
|---|---|---|
| backend/config/model_tiers.py:71,81 | role pins (enrichment, layer1_swappable) | REPIN |
| backend/agents/orchestrator.py:382 (+443 comment) | `_GEMINI_FALLBACK` | REPIN |
| backend/agents/multi_agent_orchestrator.py:237,241,243 | MAS fallback client | REPIN |
| backend/autonomous_loop.py:75 | harness evaluator default | REPIN |
| backend/agents/evaluator_agent.py:88 | EvaluatorAgent default | REPIN |
| backend/agents/harness_memory.py:48,322,503 | ctx-window map + defaults | REPIN + add new model ctx row |
| backend/meta_evolution/directive_review.py:159; directive_rewriter.py:202 | Layer-4 calls | REPIN |
| backend/slack_bot/mcp_tools.py:204 | bot tool call | REPIN |
| backend/agents/cost_tracker.py:22 | pricing (0.10,0.40) | ADD new model row (keep old for history); memory: 3+ pricing tables exist — patch ALL (settings_api display list, governance estimate) |
| backend/api/agent_map.py:119,131,156 | live_model fallback display | REPIN |
| backend/agents/_inventory.json (~30 `"model"` fields) | roster metadata | bulk-update |
| backend/config/settings.py:35 | flag description text | reword |
| frontend/src/app/settings/page.tsx:79,132,422 | model picker list/default | REPIN + label |
| backend/.env.example:11 | `GEMINI_MODEL=gemini-2.0-flash` | REPIN |
| backend/tests/test_agent_map_live_model.py:6,68,90,96,109; test_apply_model_to_all_agents.py:106; test_evaluator_agent.py:35,37; tests/verify_phase_25_Q.py:298; tests/api/test_paper_trading_deposit.py:85; tests/api/test_settings_api_signal_stack.py:117 | tests asserting the literal | update fixtures/asserts |
| .claude/cron_budget.yaml:35; ARCHITECTURE.md:372; backend/agents/llm_client.py:762 | comments/docs | sweep |

## 7. Risks / gotchas for GENERATE

1. **Alias vs pinned version:** bare `gemini-2.5-flash`-style aliases auto-track stables and die at family retirement — whatever is chosen, record the retirement date next to the pin and consider the `-001`-style pinned form per Google's lifecycle doc.
2. **2.5-flash-lite date conflict** (2026-07-22 vs 2026-10-16): do NOT pick it without re-reading the live model-versions page; the burn-parity is tempting but a 6-week runway repeats AW-4.
3. **Gemini 3.x on the legacy `vertexai` SDK is unproven here** (thought signatures; SDK module removal post 2026-06-24). The live smoke (immutable criterion) is the decider; budget a google-genai SDK follow-up step regardless.
4. **Grounding/structured-output must be smoke-verified** on the chosen ID (Layer-1 depends on both; grounding priced 1,500 RPD free then $35/1k on 2.5/3 families).
5. **deep_think `gemini-2.5-pro` (settings.py:30) also retires 2026-10-16** — same incident class; at minimum log a follow-up step.
6. **Alarm inputs don't exist yet:** the except at autonomous_loop.py:1529 must capture `{ticker, stage, exception}` into the cycle summary before any threshold can report per-ticker reasons. Keep the alarm beside 56.2 (same block, same raise_cron_alert, distinct error_type + dedup key); per-cycle window, default 50%, P1 page + always-on digest provenance line (SRE low-traffic guidance — burn-rate windows degenerate at 1 cycle/day).
7. **KR tagged-skip touches a Cloud Function** (functions/quant/main.py) — deploy needed (phase-27.6.4 precedent: CF redeploys are their own hazard); the backend-only alternative (pre-gate .KS tickers before run_quant_agent) avoids the CF deploy but leaves the CF lying in wait.
8. **Test false-greens:** 6 test files assert the literal `gemini-2.0-flash` — update them with the repin or the suite fails red on a correct fix (memory: name new tests `test_phase_60_1_*`).

## 8. Source table

### Read in full via WebFetch (counts toward gate)
| URL | Accessed | Kind | Key finding |
|---|---|---|---|
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash | 2026-06-11 | official doc | discontinued 2026-06-01; replacement "Gemini 3.1 Flash-Lite, Gemma 4, or recent" |
| https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models | 2026-06-11 | official doc | served lineup incl. gemini-3-flash, gemini-3-1-flash-lite, gemini-2.5-flash |
| https://blevinscm.github.io/genai-docs/deprecations/Model-versions-and-lifecycle/ | 2026-06-11 | doc mirror | alias->latest-stable semantics; 2.0-flash-001 released 2025-02-05, scheduled retirement 2026-02-05 |
| https://ai.google.dev/gemini-api/docs/pricing | 2026-06-11 | official doc | exact $/Mtok for all candidates + grounding pricing |
| https://engopendart.fss.or.kr/intro/main.do | 2026-06-11 | official (FSS) | OpenDART English API: key-based auth, filings/financials endpoints |
| https://sre.google/workbook/alerting-on-slos/ | 2026-06-11 | official (Google SRE) | multiwindow multi-burn-rate numbers; low-traffic degeneration |

### Snippet-only (context; does not count)
| URL | Kind | Why not fetched |
|---|---|---|
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions | official | JS-rendered; 2 fetches returned nav only (mirror used instead) |
| https://docs.cloud.google.com/vertex-ai/generative-ai/docs/release-notes | official | corroborative |
| https://ai.google.dev/gemini-api/docs/deprecations | official | Gemini-API-side dates |
| https://ai.google.dev/gemini-api/docs/models | official | model list corroboration |
| https://firebase.google.com/docs/ai-logic/models | official | corroborative |
| https://gcpstudyhub.com/blog/google-is-retiring-gemini-2-5-on-vertex-ai-what-you-need-to-know-and-do-before-october-2026 | blog | 2.5 family Oct-16-2026 retirement |
| https://gcpstudyhub.com/blog/the-vertex-ai-generative-models-sdk-is-being-deprecated | blog | SDK deprecation |
| https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/deprecations/genai-vertexai-sdk | official | SDK migration (post-2026-06-24 module removal) |
| https://therouter.ai/news/vertex-ai-sdk-migration-gemini-enterprise-agent-platform/ | industry | SDK deadline color |
| https://piunikaweb.com/2026/03/11/gemini-2-5-flash-lite-preview-discontinued-ai-studio-march-31/ | community | flash-lite preview churn |
| https://opendart.fss.or.kr/ + /guide/detail.do?apiGrpCd=DS001&apiId=2019001 | official (KR) | Korean-language API guide |
| https://engopendart.fss.or.kr/guide/detail.do?apiGrpCd=DE002&apiId=AE00032 | official | English endpoint detail |
| https://englishdart.fss.or.kr/ | official | English DART portal |
| https://dart-fss.readthedocs.io/en/latest/_modules/dart_fss/api/filings/corp_code.html | lib doc | corpCode.xml mapping mechanics |
| https://pypi.org/project/dart-fss/ | lib | mature Python client |
| https://www.xbrl.org/news/south-korea-expands-english-disclosure-system-to-boost-foreign-investment/ | industry | Feb-2025 English platform, 83 types |
| https://www.kedglobal.com/regulations/newsView/ked202402190001 | press | phased English mandates |
| https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd?locale=en | official (KRX) | KRX data portal option |
| https://mcpservers.org/servers/songhyojun0228/opendart-mcp-server | community | free key, 10k req/day corroboration |
| https://docs.cloud.google.com/stackdriver/docs/solutions/slo-monitoring/alerting-on-budget-burn-rate | official | burn-rate alerting on GCP |
| https://incident.io/blog/sre-alerting-best-practices | industry | alert-fatigue stats |
| https://futureagi.com/blog/what-is-llm-fallback-strategy-2026/ | industry | quality-floor gate on fallback routes |
| https://www.getmaxim.ai/articles/best-llm-gateway-to-design-reliable-fallback-systems-for-ai-apps/ | industry | fallback-activation-rate >5% alert baseline |
| https://www.buildmvpfast.com/blog/llm-fallback-strategies-primary-model-secondary-model-2026 | industry | per-request model provenance logging |

## 9. JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 25,
  "urls_collected": 31,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "gate_passed": true
}
```

Hard blockers: >=5 read in full (6) [x]; 10+ URLs (31) [x]; recency scan [x]; full pages not abstracts [x]; file:line for every internal claim [x]. Soft: all relevant modules covered (loop, orchestrator, CFs, alerting, formatters, reports API, tests, frontend) [x]; conflicts noted (2.5-flash-lite date; doc-slug vs publisher-ID) [x]; per-claim citations [x].
