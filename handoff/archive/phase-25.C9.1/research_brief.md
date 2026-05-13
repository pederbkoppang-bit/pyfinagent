# Research Brief: phase-25.C9.1 — Orchestrator BatchClient Routing for Backtest Mode

**Tier:** moderate (stated by caller)
**Date:** 2026-05-13

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://platform.claude.com/docs/en/build-with-claude/batch-processing | 2026-05-13 | Official docs | WebFetch full | "limited to either 100,000 Message requests or 256 MB"; "most batches finishing in less than 1 hour"; 50% flat discount confirmed current; 29-day result retention; no ZDR eligibility |
| https://jangwook.net/en/blog/en/anthropic-message-batches-api-production-guide/ | 2026-05-13 | Practitioner blog 2025 | WebFetch full | "Below 100 requests, the polling overhead often isn't worth it"; tier-1 queue = 100,000; poll 30s for large batches, 10s for <300 requests; expired rows not billed but lost |
| https://stevekinney.com/writing/anthropic-batch-api-with-temporal | 2026-05-13 | Practitioner blog 2025 | WebFetch full | Sequential async masking via Temporal: while(!completed) poll then fetch; result types are succeeded/errored/canceled/expired; recommends per-item error handling |
| https://www.dotzlaw.com/insights/obsidian-notes-02/ | 2026-05-13 | Practitioner blog 2026 | WebFetch full | Dual-mode routing: 1 file -> sync, 2+ files -> batch; silent data corruption from index-reset bug caught only at 122-file boundary test; custom_id must encode position+identity |
| https://www.finout.io/blog/anthropic-api-pricing | 2026-05-13 | Industry pricing guide 2026 | WebFetch full | Batch + 1h-cache combined: ~95% effective discount on reused content; Sonnet 4.6 batch input $1.50/MTok; deploy batch when "latency tolerance exceeds 4 hours AND volume >10,000 items/month" |
| https://aiintransit.medium.com/anthropic-launches-message-batches-api-for-cost-effective-querying-with-claude-ai-3bd4ba3a003e | 2026-05-13 | Blog (original announcement) | WebFetch full | 50% discount confirmed; up to 10,000 requests per batch; 29-day result persistence; partial failure is per-item, others continue |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://temporal.io/code-exchange/using-temporal-with-anthropics-message-batches-api | Code exchange | Duplicate of stevekinney.com Temporal article; sampled via search snippet |
| https://www.ai.moda/en/blog/anthropics-batches-with-caching | Blog | WebFetch returned only schema metadata, no article body |
| https://dasroot.net/posts/2026/02/async-llm-pipelines-python-bottlenecks/ | Blog 2026 | WebFetch returned no content matching the topic |
| https://github.com/cline/cline/discussions/1326 | Community discussion | Snippet only; lower source tier |
| https://www.anthropic.com/news/message-batches-api | Official announcement | Snippet; official docs page already read in full above |

## Recency scan (2024-2026)

Searched for: "Anthropic batch API 2026 orchestrator", "Anthropic batch API 50% discount latency 2025", "async LLM pipeline batch retrofit Python 2026".

**Findings:** The 50% discount is current as of May 2026 -- confirmed against live pricing table fetched from platform.claude.com today. Dotzlaw.com (February 2026) is the most recent practitioner piece; it introduces a production caution not in 2024 sources: a per-batch index-reset bug causing silent data corruption, catchable only via boundary-condition testing (e.g., at batch-size-crossing boundaries). No 2026 source supersedes the canonical Anthropic docs; the core mechanics (submit -> poll -> fetch, 50% discount, 24h max, 29-day retention) are unchanged.

---

## Search queries run (three-variant discipline)

1. **Current-year frontier:** "Anthropic Message Batches API best practices 2026"
2. **Last-2-year window:** "Anthropic batch API 50% discount latency tradeoffs 2025"
3. **Year-less canonical:** "async orchestrator retrofit batch LLM calls submit poll fetch pattern Python"
4. Additional scoped: "Anthropic batch API orchestrator integration backtest pipeline per-call vs window batching 2025 2026"

---

## Key findings

1. **50% discount confirmed current** -- Sonnet 4.6: $1.50 input / $7.50 output per MTok (batch). Combined with 1h-TTL prompt caching the effective discount on reused system prompts is ~95%. (Source: platform.claude.com/docs, accessed 2026-05-13)

2. **Polling overhead crossover** -- Below ~100 requests the poll-wait overhead erodes the latency benefit; above ~100 requests batching is clearly superior. "Sweet spot is workloads where results need to be ready by morning." (Source: jangwook.net, accessed 2026-05-13). 10 tickers x 28 agents = 280 calls: well above the crossover.

3. **Window-batching is the dominant pattern** -- Every practitioner source groups ALL requests into a single batch.submit() call then polls once. Per-call batching (one batch per `_generate_with_retry` call) would require 280 separate poll loops with 5-60s backoff each -- total overhead of 23-280 minutes of mostly-idle polling for zero additional discount benefit. (Source: dotzlaw.com, stevekinney.com, jangwook.net)

4. **Error semantics are per-item, not per-batch** -- A batch completes ("ended") even when individual rows errored or expired. Expired rows are not billed. Errored rows surface an error object; BatchClient.fetch() already handles this by returning `LLMResponse(text="", thoughts="errored: <msg>")`. Callers must check `response.thoughts.startswith("errored:")`. (Source: Anthropic docs; internal: llm_client.py:1677-1683)

5. **custom_id design is safety-critical** -- The dotzlaw.com 2026 production incident was silent data corruption from wrong custom_id scoping. Pattern `{ticker}__{agent_name}` encodes enough identity that mismatches are detectable. (Source: dotzlaw.com, accessed 2026-05-13)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/llm_client.py` | 1574-1684 | `BatchClient`: submit/poll/fetch over Anthropic Message Batch API | Shipped in 25.C9; no orchestrator wiring yet |
| `backend/agents/orchestrator.py` | 302-459 | `AnalysisOrchestrator.__init__`: builds `general_client`, `deep_think_client`, `synthesis_client` -- no `BatchClient` instance, no `backtest_mode` param | Missing batch_mode constructor arg |
| `backend/agents/orchestrator.py` | 488-545 | `_generate_with_retry`: sync wrapper around `model.generate_content()` with ThreadPoolExecutor timeout + exponential retry | Hot-path for all 28 agents |
| `backend/agents/orchestrator.py` | 620-813 | All `run_*_agent()` methods, each calling `self._generate_with_retry(self.general_client, ...)` | 18 candidate batch call sites (general_client calls only; grounded + RAG must stay sync) |
| `backend/agents/cost_tracker.py` | 82-185 | `AgentCostEntry.is_batch: bool`; `CostTracker.record(is_batch=False)` applies 0.5x multiplier when `is_batch=True` | Shipped in 25.C9; ready for use |
| `backend/config/settings.py` | ~71 | `sentiment_haiku_batch_mode: bool = False` -- precedent for per-feature batch toggle in Settings | `backtest_batch_mode` field does NOT yet exist |
| `backend/tasks/analysis.py` | 353-396 | `run_analysis_pipeline()` -- sync wrapper, instantiates `AnalysisOrchestrator(settings)` directly | Backtest entry point; needs `n_tickers` awareness |

---

## Design recommendation

### Batching strategy: WINDOW-BATCH per backtest run, not per-call

For each backtest window, collect all batchable LLM prompts (all `general_client` calls; exclude grounded and RAG which must stay Gemini), aggregate into a single `BatchClient.submit()`, poll once, dispatch results back to each agent's return slot.

Concretely for a 10-ticker backtest:
- 10 tickers x ~18 batchable enrichment agents = 180 requests per submission
- 1 poll loop with exponential backoff (BatchClient.poll already implements 5s->60s cap)
- 1 fetch call
- Results keyed by `custom_id = f"{ticker}__{agent_name}"` dispatched per `run_*_agent()` slot

**Critical constraint:** Synthesis, Critic, and Deep Dive agents form a sequential dependency chain (Synthesis -> Critic -> optional Synthesis revision). These cannot be pre-batched as a group. Recommended scope for window-batch: the 15-18 enrichment agents (Macro, Market, Competitor, Insider, Options, Social Sentiment, Earnings Tone, Alt Data, Sector, Patent, NLP Sentiment, Anomaly, Scenario, Quant Model) whose prompts can be materialized from pre-fetched data before any LLM call fires.

### (a) WHERE to insert the batch gate

1. Add `_run_enrichment_batch(requests: list[dict]) -> dict[str, LLMResponse]` method to `AnalysisOrchestrator` (after `_generate_with_retry` at line 546).
2. In `run_full_analysis()` (line ~1010), branch on `self._backtest_mode and len(tickers) > 3`:
   - `True`: collect all enrichment prompts, call `_run_enrichment_batch()`, inject results into each `run_*_agent()` without calling `_generate_with_retry`.
   - `False`: existing synchronous path unchanged.

### (b) WHAT triggers the batch path

Add a Settings field following the `sentiment_haiku_batch_mode` precedent (settings.py:~71):

```python
backtest_batch_mode: bool = Field(
    False,
    description="Route enrichment agents through Anthropic Batch API when n_tickers>3 in backtest runs (50% discount, async)."
)
```

Add constructor args to `AnalysisOrchestrator.__init__` (orchestrator.py:317):

```python
def __init__(self, settings: Settings, backtest_mode: bool = False, n_tickers: int = 1):
    ...
    self._backtest_mode = backtest_mode and settings.backtest_batch_mode and n_tickers > 3
    if self._backtest_mode:
        self._batch_client = BatchClient(settings.claude_model, settings.anthropic_api_key)
```

Default `backtest_mode=False` keeps the synchronous fast path completely unaffected. Opt-in callers: `backend/tasks/analysis.py:373` (pass `backtest_mode=True, n_tickers=len(tickers)` from backtest engine).

Do NOT use an env var. Settings field is correct (consistent with existing pattern, survives restart, visible to settings endpoint).

### (c) HOW errored entries flow back through the synchronous-style return path

`BatchClient.fetch()` already surfaces errors as `LLMResponse(text="", thoughts="errored: <msg>")` (llm_client.py:1677-1683). The dispatcher in `_run_enrichment_batch()` should:

1. For each `custom_id` in results, check `response.thoughts.startswith("errored:")`.
2. On error: log WARNING, return the same empty-dict fallback each `run_*_agent()` already returns on `_generate_with_retry` failure (e.g., `{"text": ""}` for enrichment agents).
3. On missing custom_id after "ended" status (expired): treat same as errored -- log + fallback.
4. Pass `is_batch=True` to `ct.record(...)` for each successful result (cost_tracker.py:119 already accepts the param, applies 0.5x).

`errored_count` for the 25.C9 verifier: `sum(1 for r in results.values() if r.thoughts and r.thoughts.startswith("errored:"))`.

---

## Consensus vs debate

**Consensus:** Window-batch (single submit for all requests, one poll, one fetch) is the canonical production pattern. No source recommends per-call batching. Anthropic docs state "most batches complete within 1 hour" -- acceptable for overnight backtests.

**Caution note (Feb 2026):** One practitioner migrated back from Batch API to asyncio.TaskGroup parallel sync calls after experiencing 4+ hour completion times with no per-item progress. For pyfinagent this risk is acceptable: the batch path is opt-in (`backtest_batch_mode=False` default), and the synchronous path remains intact as fallback.

## Pitfalls

- **Silent data corruption from custom_id scoping** (dotzlaw.com 2026): encode ticker + agent name in custom_id; never rely on array index.
- **Expired requests are lost** (jangwook.net): if `poll()` returns "timeout", fetch will return partial results; unmatched custom_ids should fall back to synchronous `_generate_with_retry`.
- **Rate limits apply inside batches** (Anthropic docs): 100,000 requests in Tier 1 queue -- 180-request backtests are far below this.
- **No ZDR** (Anthropic docs): batch results retained 29 days. Acceptable for backtest use.
- **Synthesis/Critic chain cannot be batched across tickers**: each ticker's Synthesis depends on that ticker's enrichment results. Exclude from window batch.

## Application to pyfinagent (file:line anchors)

| Recommendation | File:line |
|----------------|-----------|
| Add `backtest_batch_mode: bool` field | `backend/config/settings.py:~71` (after `sentiment_haiku_batch_mode`) |
| Add `backtest_mode: bool, n_tickers: int` to constructor | `backend/agents/orchestrator.py:317` |
| Add `_batch_client: BatchClient` init | `backend/agents/orchestrator.py:~400` (inside `__init__` after client setup) |
| Add `_run_enrichment_batch()` method | `backend/agents/orchestrator.py:~546` (new method after `_generate_with_retry`) |
| Branch in `run_full_analysis()` | `backend/agents/orchestrator.py:1010` |
| Pass `is_batch=True` to `ct.record()` | `backend/agents/cost_tracker.py:119` (already accepts the param) |
| Opt-in call site in backtest entry | `backend/tasks/analysis.py:373` |
| Enrichment agent candidates (18 calls) | `backend/agents/orchestrator.py:620-813` -- all `run_*_agent()` using `self.general_client` |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched in full)
- [x] 10+ unique URLs total incl. snippet-only (11 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (llm_client.py, orchestrator.py, cost_tracker.py, settings.py, tasks/analysis.py)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 5,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
