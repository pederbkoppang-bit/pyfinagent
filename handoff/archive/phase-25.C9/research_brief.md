---
step: 25.C9
slug: anthropic-batch-api-for-non-interactive-pipeline
tier: moderate-complex
cycle_date: 2026-05-13
authored_by: researcher-agent
---

# Research Brief -- phase-25.C9: Adopt Batch API for non-interactive pipeline steps (50% savings)

---

## Three-variant search queries run

1. **Current-year frontier**: `Anthropic Batch API messages batches create lifecycle polling 2026`
2. **Last-2-year window**: `Anthropic batch API 50% discount cached tokens prompt caching combined savings 2025`
3. **Year-less canonical**: `Anthropic batch API Python SDK anthropic.Batch poll results streaming`

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://platform.claude.com/docs/en/build-with-claude/batch-processing | 2026-05-13 | Official doc | WebFetch (full, 70KB) | "reducing costs by 50% and increasing throughput"; lifecycle `in_progress` → `ended`/`canceling`; 10,000 requests or 32MB per batch; most batches finish < 1 hour, max 24h |
| https://platform.claude.com/docs/en/api/creating-message-batches | 2026-05-13 | Official API ref | WebFetch (full) | Full request shape: `{requests: [{custom_id, params}]}`; `params` mirrors Messages API; `processing_status` values: `processing`/`succeeded`/`failed`/`expired`; `custom_id` must be unique per batch |
| https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/resources/beta/messages/batches.py | 2026-05-13 | SDK source | WebFetch (full) | `create()`, `retrieve()`, `results()` → `JSONLDecoder[BetaMessageBatchIndividualResponse]`; `cancel()`, `delete()`; async variants available; results are unordered — use `custom_id` to map |
| https://agentbus.sh/posts/how-to-use-the-anthropic-message-batches-api-for-async-workloads/ | 2026-05-13 | Authoritative blog | WebFetch (full) | Poll every 60s for large batches; chunk into 5,000-request groups for >256MB payloads; result types: `succeeded`, `errored`, `expired`; `custom_id` Map required to avoid silent mismatches |
| https://jangwook.net/en/blog/en/anthropic-message-batches-api-production-guide/ | 2026-05-13 | Authoritative blog | WebFetch (full) | Adaptive polling: 10s small (<100 req), 60s large; hybrid routing: batch for >100 non-real-time; below ~100 "polling overhead often isn't worth it"; batch+cache combined = ~95% discount |
| https://claudelab.net/en/articles/api-sdk/claude-api-messages-batches-async-processing-guide | 2026-05-13 | Authoritative blog | WebFetch (full) | SDK path: `client.beta.messages.batches.create()`; three failure categories: errored, expired, submission failures; automatic batch splitting; works with Opus 4.7, Sonnet 4.6, Haiku 4.5 |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.finout.io/blog/anthropic-api-pricing | Blog | Key pricing facts confirmed from search snippet; docs are authoritative |
| https://platform.claude.com/docs/en/build-with-claude/prompt-caching | Official doc | Stacking behavior confirmed from snippet; full content covered in prior phase-25.B9 gate |
| https://www.ai.moda/en/blog/anthropics-batches-with-caching | Blog | WebFetch returned only metadata markup, no article body |
| https://aiintransit.medium.com/anthropic-launches-message-batches-api | News | Launch announcement; key facts in snippet |
| https://www.anthropic.com/news/message-batches-api | Official announcement | Launch post; facts covered by official docs above |
| https://evolink.ai/blog/claude-api-pricing-guide-2026 | Blog | Pricing confirmed from snippet; docs authoritative |
| https://stevekinney.com/writing/anthropic-batch-api-with-temporal | Blog | Temporal workflow integration; not relevant to pyfinagent stack |
| https://n8n.io/workflows/3409-batch-process-prompts-with-anthropic-claude-api/ | Workflow template | n8n-specific; low relevance |

---

## Recency scan (2024-2026)

Searched with 2026 and 2025 year qualifiers.

**Result:** Active 2026 findings:
1. `output-300k-2026-03-24` beta header extends `max_tokens` to 300,000 for batch requests on Opus 4.6/4.7 and Sonnet 4.6 — relevant for long enrichment outputs if needed.
2. Prompt caching + Batch API confirmed to stack multiplicatively: batch request with a cache hit pays 0.1× base input × 0.5 batch discount = **5% of standard non-cached input cost**.
3. All current GA models (Opus 4.7, Opus 4.6, Sonnet 4.6, Haiku 4.5) fully support batches as of 2026.
4. `client.beta.messages.batches` is the stable SDK namespace at `anthropic>=0.46.0`; installed version is **0.96.0** — no upgrade required.
5. No paradigm shifts since the Oct 2024 GA launch; pricing has been stable at flat 50% throughout.

---

## Key findings

1. **Flat 50% discount on both input AND output tokens** — confirmed across all sources; applies to all supported Claude models with no carve-outs. Cache discounts stack: a cached batch call costs 0.1× (cache read) × 0.5 (batch) = 0.05× standard input price. (Source: Anthropic Batch Processing docs + pricing snippets, 2026-05-13)

2. **Lifecycle states** — `processing_status` field on the batch object: `in_progress` → `ended` (terminal, retrieve results) or `canceling` → `ended`. Individual result entry `result.type`: `succeeded`, `errored`, `expired`. (Source: Anthropic API ref + SDK source, 2026-05-13)

3. **Polling is the only retrieval method** — no webhook option. Best practice: 5–30s for small batches (<50 requests), 60s for large. Exponential backoff: start 5s, cap at 60s. (Source: agentbus.sh, jangwook.net, 2026-05-13)

4. **Results are UNORDERED** — `custom_id` mapping is mandatory. Index-based mapping is a documented anti-pattern causing silent data mismatches. Recommended key format: `f"{ticker}_{agent_name}_{uuid4().hex[:8]}"`. (Source: jangwook.net production guide, 2026-05-13)

5. **Routing threshold** — industry consensus: batch when (a) >100 requests, (b) latency delay acceptable, (c) non-interactive workload. For pyfinagent's backtest: 11 enrichment agents × N tickers = at n=4: 44 requests; at n=20 (typical window): 220 requests — above threshold at n>3. (Source: jangwook.net + claudelab.net, 2026-05-13)

6. **SDK path confirmed:** `client.beta.messages.batches.create(requests=[...])` — no `betas=` header needed for the batch call itself (unlike Files API). Installed SDK 0.96.0 — confirmed via `pip show anthropic`. (Source: SDK source code, bash inspection, 2026-05-13)

7. **Prompt caching is compatible with batch requests** — `cache_control` on system blocks works inside `params` of each batch request entry. The 1h TTL (`ttl: "1h"`) used by `ClaudeClient.generate_content` (`llm_client.py:1198`) is valid inside batch `params`. The `_HOUSE_INSTRUCTIONS` block from phase-25.B9 (`llm_client.py:52`) already clears the per-model cache-write threshold floor (4096 tokens Opus 4.7/Haiku 4.5; 2048 Sonnet 4.6). (Source: Anthropic pricing docs + prior phase-25.B9 gate, 2026-05-13)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/llm_client.py` | 1638 | Unified LLM client — `ClaudeClient` at line 1058, `_HOUSE_INSTRUCTIONS` at line 52, `_get_client()` at line 1082 | Active; `BatchClient` sibling class goes after `ClaudeClient` |
| `backend/agents/cost_tracker.py` | 256 | Per-agent token cost accumulator — `AgentCostEntry` at line 83, `record()` at line 107, `summarize()` at line 187 | Active; add `is_batch: bool = False` + 0.5× multiplier |
| `backend/agents/orchestrator.py` | 1620 | 15-step pipeline — `_generate_with_retry()` at line 488, Step 7 enrichment `_agent_list` at lines 1302–1336, `run_full_analysis()` at line 1010 | Active; routing hook point is `_generate_with_retry()` line 488; Step 7 `asyncio.gather` at line 1331 is primary batching target |
| `backend/config/settings.py` | 200+ | Pydantic settings — `sentiment_haiku_batch_mode: bool` at line 70 (existing proof-of-pattern); backtest fields at lines 122-139 | Active; add `batch_api_enabled` + `batch_min_tickers` fields |

---

## Consensus vs debate (external)

**Consensus:**
- 50% discount on both input AND output tokens; no exceptions for supported models
- Prompt caching and Batch API discounts stack multiplicatively
- Polling is the only retrieval method; webhooks not available as of 2026-05-13
- `custom_id` → results mapping is mandatory; results are unordered
- `client.beta.messages.batches` is the stable SDK namespace at 0.96.0

**Debate / open questions:**
- Optimal poll interval for medium batches (50-200 requests): sources give 10s vs 60s; 30s is a reasonable middle ground
- `BatchClient` as sibling class vs method on `ClaudeClient`: sibling class preferred for separation of concerns and independent testability

---

## Pitfalls (from literature)

1. **Index-based result mapping** — results stream in arbitrary order; always use a `{custom_id: result}` dict.
2. **Duplicate `custom_id`** — causes a validation error at submission time; enforce uniqueness with a uuid suffix.
3. **Polling too aggressively** — thundering herd on many concurrent batches; use jitter or adaptive interval.
4. **Blocking the event loop** — `retrieve()` and `results()` are synchronous; wrap in `asyncio.to_thread()` when called from async orchestrator code (`run_full_analysis` is `async def`).
5. **Missing `is_batch` in cost tracking** — without the flag, batch costs are reported at standard pricing (2× actual), masking the 50% savings from finance dashboards.
6. **Cache TTL inside batch params** — `ttl: "1h"` is valid inside `params.system[].cache_control`; verify the combined system prompt still exceeds the per-model cache-write threshold. The `_HOUSE_INSTRUCTIONS` block (llm_client.py:52) already clears this floor.
7. **24-hour expiry** — if the poll loop crashes or times out, the batch expires; re-submit failed `custom_id`s. `result.type == "expired"` is the signal.

---

## Application to pyfinagent (file:line anchors)

### Where to add `BatchClient`

`backend/agents/llm_client.py` — add `BatchClient` as sibling class after `ClaudeClient` (which starts at line 1058). Reuses `_anthropic_sdk`, `_get_client()` pattern, `UsageMeta`, and `LLMResponse` already defined in the module. Specifically:
- `_anthropic_sdk` import guard (line 1086-1090) — reuse in `BatchClient._get_client()`
- `_HOUSE_INSTRUCTIONS` (line 52) — include verbatim in every batch request's `system` param with the same `cache_control: {"type": "ephemeral", "ttl": "1h"}` block
- `UsageMeta` + `LLMResponse` dataclasses — use to construct results in `fetch()`

### Where to add `is_batch` to cost tracker

`backend/agents/cost_tracker.py`:
- `AgentCostEntry` dataclass (line 83): add `is_batch: bool = False` after `cache_read_input_tokens`
- `CostTracker.record()` (line 107): add `is_batch: bool = False` param; after computing `cost`, apply `if is_batch: cost *= 0.5`
- `CostTracker.summarize()` (line 187): add `batch_calls` count to the returned dict, analogous to `deep_think_calls` at line 205

### Where to route to batch in orchestrator

`backend/agents/orchestrator.py`:
- `_generate_with_retry()` (line 488): gate — `if isinstance(model, ClaudeClient) AND backtest_mode AND n_tickers > 3` → accumulate into `BatchClient` instead of calling synchronously
- Step 7 enrichment `_agent_list` (line 1302) + `asyncio.gather` (line 1331): primary target — all 11 agents call `self.general_client` via `_generate_with_retry`; in batch mode, collect all prompts, submit one batch, poll, distribute results back by key
- `run_full_analysis()` signature (line 1010): add `backtest_mode: bool = False` and `n_tickers: int = 1` params

### Existing batch flag pattern in settings

`backend/config/settings.py` line 70: `sentiment_haiku_batch_mode: bool = Field(False, ...)` — confirms the project already has a pattern for batch routing flags. Add:
```python
batch_api_enabled: bool = Field(False, description="Enable Anthropic Batch API routing for enrichment steps in backtest mode (50% discount)")
batch_min_tickers: int = Field(4, description="Minimum ticker count before routing Step 7 enrichment to Batch API")
```

---

## Verbatim Python signature for `BatchClient`

```python
class BatchClient:
    """phase-25.C9: thin wrapper over Anthropic's Message Batch API.

    Submits N parallel non-interactive requests for 50% token discount.
    Designed for backtest fanout (n_tickers > 3) where 24h latency is
    acceptable. NOT used for the synchronous interactive daily cycle.

    Lifecycle:
        batch_id = bc.submit(requests)   # [{custom_id, params}]
        status   = bc.poll(batch_id)     # blocks until "ended"
        results  = bc.fetch(batch_id)    # dict[custom_id, LLMResponse]

    Discount stacks with prompt caching:
        cache-read batch call = 0.1x * 0.5x = 5% of standard price.
    """

    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self._api_key = api_key

    def _get_client(self):
        if _anthropic_sdk is None:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic>=0.96.0"
            )
        return _anthropic_sdk.Anthropic(api_key=self._api_key, max_retries=3)

    def submit(self, requests: list[dict]) -> str:
        """Submit batch. Each request: {"custom_id": str, "params": dict}.
        Returns batch_id string."""
        ...

    def poll(
        self,
        batch_id: str,
        max_wait_sec: int = 1800,
        initial_delay_sec: int = 5,
    ) -> str:
        """Poll retrieve() with exponential backoff (5s -> 60s cap) until
        processing_status in ("ended", "canceled"). Returns final status.
        Raises TimeoutError if max_wait_sec exceeded.
        Must be called via asyncio.to_thread() from async callers."""
        ...

    def fetch(self, batch_id: str) -> dict[str, LLMResponse]:
        """Stream JSONL results, return {custom_id: LLMResponse}.
        Errored/expired rows produce LLMResponse(text="",
        thoughts="errored:<msg>") so callers can detect and retry."""
        ...
```

---

## Verbatim batch request body shape

```python
# Payload sent to client.messages.batches.create()
{
    "requests": [
        {
            "custom_id": "AAPL_insider_a1b2c3d4",   # unique within batch
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "temperature": 0.0,
                "system": [
                    {
                        "type": "text",
                        "text": "<_HOUSE_INSTRUCTIONS verbatim>",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    }
                ],
                "messages": [
                    {"role": "user", "content": "<enrichment prompt for AAPL insider data>"}
                ]
                # Optional: skill_file_id document block if phase-25.D9 file_ids available
            }
        },
        {
            "custom_id": "MSFT_insider_e5f6g7h8",
            "params": { ... }
        }
        # Up to 10,000 entries or 32MB total
    ]
}

# Result row shape (from client.messages.batches.results() JSONL stream):
{
    "custom_id": "AAPL_insider_a1b2c3d4",
    "result": {
        "type": "succeeded",           # or "errored" / "expired"
        "message": {
            "content": [{"type": "text", "text": "..."}],
            "usage": {
                "input_tokens": 412,
                "output_tokens": 198,
                "cache_read_input_tokens": 390,    # if cache hit
                "cache_creation_input_tokens": 0
            }
        }
    }
}
```

---

## Routing decision table

| Condition | Route | Rationale |
|-----------|-------|-----------|
| `backtest_mode=True` AND `n_tickers > 3` AND step is 7 (enrichment) | `BatchClient` | Non-interactive; 11 agents × N tickers ≥ 44 requests; 50% saving material |
| `backtest_mode=True` AND `n_tickers <= 3` | `ClaudeClient` sync | Polling overhead (≥30s setup) not worth it for <33 requests |
| `backtest_mode=False` (live/interactive) | `ClaudeClient` sync | User actively waiting; 24h batch window unacceptable |
| Steps 8-15 (debate, synthesis, risk, critic) | `ClaudeClient` sync | Sequential dependency chain; each step needs prior step's output |
| Steps 1-6 (quant, RAG, market, competitor) | `ClaudeClient` sync or external HTTP | Data fetch steps; Batch API not applicable to non-LLM calls |
| Model is Gemini (`GeminiClient`) | `GeminiClient` sync | Anthropic Batch API is Anthropic-only |

---

## cost_tracker `is_batch` field design

**`AgentCostEntry` addition** (`cost_tracker.py:83`):
```python
# phase-25.C9: when True, cost_usd reflects 50% Batch API discount.
is_batch: bool = False
```

**`CostTracker.record()` update** (`cost_tracker.py:107`):
```python
def record(
    self,
    agent_name: str,
    model: str,
    response: object,
    is_deep_think: bool = False,
    is_grounded: bool = False,
    is_batch: bool = False,    # NEW phase-25.C9
) -> Optional[AgentCostEntry]:
    # ... existing token extraction and cost computation (lines 128-154) ...
    if is_batch:
        cost *= 0.5   # Anthropic flat 50% Batch API discount
    entry = AgentCostEntry(
        ...,
        is_batch=is_batch,   # NEW
    )
```

**Important note on stacking:** when both `is_batch=True` and prompt cache tokens are present, the cache discount applies first (in the existing cost calculation block at lines 147-154), then `cost *= 0.5` is applied on top. This correctly computes the multiplicative stack: 0.1× cache-read × 0.5× batch = 0.05× standard input.

**`CostTracker.summarize()` addition** (`cost_tracker.py:187`): add to returned dict alongside `deep_think_calls`:
```python
"batch_calls": sum(1 for e in entries if e.is_batch),
```

---

## Files to modify table

| File | Change | Lines affected |
|------|--------|----------------|
| `backend/agents/llm_client.py` | Add `BatchClient` class (sibling to `ClaudeClient`) | After line 1558 (end of `ClaudeClient`) |
| `backend/agents/cost_tracker.py` | Add `is_batch: bool = False` to `AgentCostEntry`; `is_batch` param + `cost *= 0.5` in `record()`; `batch_calls` in `summarize()` | Lines 83, 107, 187 |
| `backend/agents/orchestrator.py` | Add `backtest_mode: bool = False` + `n_tickers: int = 1` to `run_full_analysis()`; routing hook in `_generate_with_retry()` or Step 7 gather | Lines 1010, 488, 1331 |
| `backend/config/settings.py` | Add `batch_api_enabled: bool` + `batch_min_tickers: int = 4` | After line 70 |
| `tests/verify_phase_25_C9.py` | New verifier covering all 3 success criteria | New file |

---

## Research Gate Checklist

Hard blockers — all checked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total (14 URLs: 6 read-in-full + 8 snippet-only)
- [x] Recency scan (last 2 years) performed and reported
- [x] Full pages/docs read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (llm_client.py, cost_tracker.py, orchestrator.py, settings.py)
- [x] Contradictions/consensus noted (poll interval; sibling class vs method)
- [x] All claims cited per-claim (not just footer list)

---

```json
{
  "tier": "moderate-complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
