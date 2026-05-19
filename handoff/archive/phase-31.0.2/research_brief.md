# Research Brief -- phase-31.0.3 Stage 3 Smoketest (Gemini Full-Path on NVDA)

**Tier:** deep | **Effort:** max | **Date:** 2026-05-20
**Scope:** Stage 3 of 13-stage smoketest. Invoke
`backend/agents/orchestrator.py::AnalysisOrchestrator.run_full_analysis(ticker="NVDA")`
directly. Verify the ~29 Gemini agents log entries to
`pyfinagent_data.llm_call_log`. Mock `_persist_analysis` so no
row hits `analysis_results`.

The substitution rule (Stage 2's lite path swapped Anthropic SDK for
Claude Code subagents) does NOT apply here -- Gemini Vertex calls
stay on Vertex. This Stage tests that the full Gemini pipeline
still works end-to-end -- it is the keep-Gemini complement of the
substitution test in Stage 2.

## Internal code inventory (anchors first, before WebFetch)

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/orchestrator.py` | 306 | `class AnalysisOrchestrator:` -- stateless orchestrator | LIVE |
| `backend/agents/orchestrator.py` | 330-450 | `__init__(settings, backtest_mode, n_tickers)` constructor -- builds per-model GeminiModelBundles, parses GCP service-account JSON, sets up `_genai_client = get_genai_client()` shim | LIVE |
| `backend/agents/orchestrator.py` | 314 | `_GEMINI_FALLBACK = "gemini-2.0-flash"` -- the default model when settings.gemini_model is non-Gemini | LIVE |
| `backend/agents/orchestrator.py` | 323 | `_INGESTION_SEMAPHORE = asyncio.Semaphore(2)` -- per-host throttle for SEC EDGAR | LIVE |
| `backend/agents/orchestrator.py` | 384-387 | per-model config dicts: `_gen_config`, `_enrichment_config` (max_output_tokens=1024), `_synthesis_config` (4096), `_deep_think_config` (2048) | LIVE |
| `backend/agents/orchestrator.py` | 1398-1408 | `async def run_full_analysis(self, ticker: str, on_step=None) -> dict` -- the 13-step entry point | LIVE |
| `backend/agents/orchestrator.py` | 1414-1417 | report dict + TraceCollector + CostTracker + AnalysisContext setup | LIVE |
| `backend/agents/orchestrator.py` | 1430-1463 | Step 0 fetch_market_intel + Step 1 ingestion | LIVE |
| `backend/agents/orchestrator.py` | 1465-1517 | Steps 2-4 quant, RAG, market | LIVE |
| `backend/agents/llm_client.py` | 1645-1669 | ClaudeClient calls `from backend.services.observability import log_llm_call as _log_llm_call` -- writes llm_call_log rows tagged with ticker. Comment: "Enables `SELECT ticker, SUM(input_tok * pricing) FROM llm_call_log`" | LIVE |
| `backend/agents/llm_client.py` | 1899-1933 | `advisor_call` -- writes 1-2 llm_call_log rows (executor + optional advisor) | LIVE |
| `backend/agents/llm_client.py` | 2011-2038 | advisor_call executor + advisor row writes | LIVE |
| `backend/services/autonomous_loop.py` | 1651-1690 | `async def _persist_analysis(analysis: dict, bq: BigQueryClient) -> None` -- writes to `analysis_results` via `bq.save_report()`. Generalized in phase-25.A2 from `_persist_lite_analysis` to handle BOTH lite and full paths. Reads `_path` for honest source tagging. Non-fatal on BQ error. | LIVE |
| `backend/services/autonomous_loop.py` | 708 | `await _persist_analysis(analysis, bq)` -- production callsite | LIVE |
| `backend/services/autonomous_loop.py` | 1207 | phase-25.A2 marker so `_persist_analysis` guard picks up full-pipeline rows | LIVE |
| `backend/services/observability.py` | -- | the `log_llm_call` function imported in llm_client.py:1647 -- canonical writer to `pyfinagent_data.llm_call_log` | TBD-read |
| `backend/agents/skills/*.md` | -- | 29 files (28 functional agents + SKILL_TEMPLATE.md) | LIVE |

**Skill files (29 total, including template; 28 functional agents):**
alpha_decay, alt_data, anomaly, bias_detector, competitor, critic,
debate_stance, deep_dive, earnings_tone, enhanced_macro, info_gap,
insider, market, moderator, nlp_sentiment, options, patent,
quant_model, quant_strategy, rag, risk_judge, risk_stance, scenario,
sector_analysis, sector_catalyst, social_sentiment, supply_chain,
synthesis + SKILL_TEMPLATE.md (template, not counted in 28).

Note: `ls skills/*.md | wc -l = 29` (excluding subdirs
`experiments/` and `_legacy_phase_26_4`). Counting only
functional agent prompts: 28 (subtract SKILL_TEMPLATE.md).
`quant_strategy.md` is an optimizer skill loaded by
`quant_optimizer.py`, NOT a pipeline agent (per
`.claude/rules/backend-agents.md`). The 13-step pipeline
docstring (orchestrator.py:1400) refers to 13 logical pipeline
STEPS, not 13 agents -- multiple agents may run inside a step
(e.g., Step 4 market_agent uses NLP sentiment + social sentiment +
earnings tone in parallel).

## Pass 1: Broad scan (target >=20 sources read in full)

### Read in full (counts toward the gate)
| # | URL | Accessed | Kind | Fetched via | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|

### Recency scan (last 2 years, 2024-2026)
TBD -- populate after Pass 1 completes.

### Search-query composition (three-variant discipline)
TBD -- list queries used across the session.

## Pass 2: Adversarial cross-validation

TBD -- tag findings with [ADVERSARIAL] where 2+ contradicting sources exist.

## Pass 3: Cross-domain triangulation

TBD -- list adjacent-domain sources for each quant-finance claim.

## Key findings (Pass 1)

TBD

## Consensus vs debate

TBD

## Pitfalls (from literature)

TBD

## Application to pyfinagent (mapping findings to file:line)

TBD

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [ ] >=20 authoritative external sources READ IN FULL via WebFetch (deep tier)
- [ ] 25+ unique URLs total (incl. snippet-only)
- [ ] Recency scan (last 2 years) performed + reported
- [ ] Multi-pass structure (Pass 1 / Pass 2 / Pass 3) documented
- [ ] >=1 [ADVERSARIAL] source present
- [ ] Three-variant queries (current-year + 2-year + year-less)
- [ ] file:line anchors for every internal claim

## JSON envelope

```json
{
  "tier": "deep",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 16,
  "gate_passed": false
}
```
