---
step: phase-25.S.1
topic: Per-call ticker tagging in llm_call_log for exact ticker attribution
tier: moderate
date: 2026-05-13
---

## Research: Per-call ticker tagging in llm_call_log (phase-25.S.1)

### Queries run (three-variant discipline)
1. Current-year frontier: "LLM API call per-tenant cost attribution metadata tags 2026"
2. Last-2-year window: "Anthropic Messages API metadata user_id billing tags 2025"
3. Year-less canonical: "LLM token usage cost tracking per user feature attribution"

---

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://platform.claude.com/docs/en/api/messages | 2026-05-13 | Official docs (Anthropic) | WebFetch | `metadata` object supports only `user_id` (opaque string). No custom billing tag fields. Attribution via Anthropic API metadata is NOT possible beyond user_id. |
| https://www.traceloop.com/blog/from-bills-to-budgets-how-to-track-llm-token-usage-and-cost-per-user | 2026-05-13 | Authoritative blog | WebFetch | Application-layer tagging on every request is the canonical pattern. "Permanently tags that request and its associated cost to a specific user." |
| https://docs.litellm.ai/docs/proxy/cost_tracking | 2026-05-13 | Official docs (LiteLLM) | WebFetch | `metadata.tags` array per request (e.g., `["ticker:AAPL"]`). Stored in `LiteLLM_SpendLogs.request_tags`. Confirms application-layer tagging, not provider-layer. |
| https://aws.amazon.com/blogs/machine-learning/introducing-granular-cost-attribution-for-amazon-bedrock/ | 2026-05-13 | Official docs (AWS) | WebFetch | Bedrock uses IAM principal tags -> CUR 2.0 for attribution. Analogous pattern: add a dimension field at the application layer. No infrastructure overhead. |
| https://www.braintrust.dev/articles/best-tools-tracking-llm-costs-2026 | 2026-05-13 | Authoritative blog (2026) | WebFetch | "Custom tags break the same data down by user, feature, model, or environment." Span-level tagging is the 2026 best practice for `profit_per_llm_dollar` type metrics. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://docs.anthropic.com/en/api/usage-cost-api | Official docs | Redirect to platform.claude.com; covered by messages reference above |
| https://www.truefoundry.com/blog/breaking-down-llm-usage-customer-and-user-level-analytics | Blog | Core pattern covered by traceloop source |
| https://www.getmaxim.ai/articles/how-to-monitor-llm-api-costs-in-production/ | Blog | Redundant with braintrust 2026 source |
| https://apxml.com/courses/langchain-production-llm/chapter-6-optimizing-scaling-langchain/cost-management-token-tracking | Course | Snippet sufficient; patterns covered by primary sources |
| https://www.worklytics.co/blog/how-to-track-llm-token-usage-and-cost | Blog | Redundant; covered by primary sources |
| https://www.getmaxim.ai/articles/top-5-enterprise-gateways-for-llm-cost-tracking-and-budget-controls/ | Blog | Gateway/proxy pattern not applicable here |
| https://www.finout.io/blog/anthropic-api-pricing | Blog | Pricing table only; not attribution pattern |

---

### Recency scan (2024-2026)
Searched 2024-2026 window on LLM cost attribution per entity, Anthropic API metadata, custom billing tags.

Result: The 2026 Braintrust article and 2025-2026 LiteLLM docs confirm that application-layer tagging (not provider-layer) is the dominant 2026 best practice. Anthropic has not added custom billing tags beyond `user_id` to the Messages API as of May 2026. AWS Bedrock's IAM-based approach (2025) is provider-layer but requires a fundamentally different auth model. No new finding supersedes the application-layer tagging pattern for this codebase; all 2024-2026 sources reinforce it.

---

### Key findings
1. Anthropic `metadata` is limited to `user_id` only -- no custom billing tags flow through to Anthropic invoices. Attribution must be done entirely at the application layer. (Source: platform.claude.com/docs/en/api/messages, 2026-05-13)
2. Per-entity attribution means: add a `ticker` column to the local `llm_call_log` BQ table and pass `ticker` through the call stack to every `log_llm_call()` invocation. (Source: traceloop.com, 2026-05-13)
3. "Partial tagging is the primary reason attribution data becomes unreliable" -- every call site must propagate `ticker` or the BQ data is polluted with NULLs. (Source: traceloop.com, 2026-05-13)
4. Tag-based grouping at the span/call level is the 2026 standard for computing `profit_per_llm_dollar` per entity. (Source: braintrust.dev, litellm.ai, 2026-05-13)

---

### Internal code inventory
| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/observability/api_call_log.py` | 181-294 | `log_llm_call()` buffer + BQ flush for `llm_call_log` | **No `ticker` column or param** |
| `scripts/migrations/add_llm_call_log.py` | 1-76 | DDL for `llm_call_log` BQ table (12 columns) | **No `ticker` column in schema** |
| `backend/agents/llm_client.py` | 1539-1557 | `ClaudeClient.generate_content` -> only call site for `log_llm_call` | **Only one call site; no ticker arg passed** |
| `backend/agents/llm_client.py` | 795-946 | `GeminiClient.generate_content` | Does NOT call `log_llm_call` at all |
| `backend/agents/llm_client.py` | 977-1153 | `OpenAIClient.generate_content` | Does NOT call `log_llm_call` at all |
| `backend/agents/cost_tracker.py` | 82-185 | `CostTracker.record()` / `AgentCostEntry` in-memory accumulator | **No `ticker` field** |
| `backend/agents/orchestrator.py` | 510-567 | `_generate_with_retry(model, prompt, agent_name, ...)` | `ticker` is in scope at all `run_*_agent(ticker, ...)` callers but NOT threaded into `_generate_with_retry` |
| `backend/api/paper_trading.py` | 330, 393 | Attribution endpoint (phase-25.S) | Explicitly notes "per-call ticker tagging in `llm_call_log` is a follow-up step (25.S.1)" |

**Critical observation**: `log_llm_call` is called from exactly ONE location: `ClaudeClient.generate_content` at `llm_client.py:1543`. The Gemini and OpenAI clients do NOT call it. This means for full attribution (including the 28-agent Gemini pipeline which is the bulk of spend), `GeminiClient.generate_content` must also be instrumented.

---

### Consensus vs debate
All five sources are unanimous: application-layer tagging (add a column, pass it through the call stack) is the only viable path given Anthropic's limited metadata API. No debate.

### Pitfalls (from literature + code)
1. **Partial tagging** is the primary failure mode (traceloop). Every call site must pass `ticker` or NULLs corrupt `profit_per_llm_dollar` calculations.
2. **Gemini path gap**: `GeminiClient` does NOT call `log_llm_call`. Gemini calls represent the 28-agent pipeline (the bulk of spend). Phase 25.S.1 should instrument GeminiClient too, or the attribution covers only Claude calls.
3. **`_generate_with_retry` indirection**: The orchestrator's `_generate_with_retry` calls `model.generate_content()`, not `log_llm_call` directly. The `ticker` context must flow from `_generate_with_retry` down into the client's `generate_content` call -- either via `generation_config` dict or a new kwarg.

---

### Design recommendation

#### (a) WHERE: BQ migration + `log_llm_call` signature

Add `ticker STRING` to `llm_call_log` via a new migration at `scripts/migrations/add_llm_call_log_ticker.py`. The column is nullable (NULL for calls that don't have ticker context, e.g., autonomous-loop meta-calls).

Add `ticker` to the cluster key: `CLUSTER BY provider, model, ticker` so `WHERE ticker = 'AAPL'` scans are cheap.

#### (b) WHAT: exact signature changes

**`api_call_log.py:203` -- `log_llm_call`:**
```python
def log_llm_call(
    provider: str,
    model: str,
    agent: str | None = None,
    latency_ms: float = 0.0,
    ttft_ms: float = 0.0,
    input_tok: int = 0,
    output_tok: int = 0,
    cache_creation_tok: int = 0,
    cache_read_tok: int = 0,
    request_id: str | None = None,
    ok: bool = True,
    ticker: str | None = None,   # <-- new kwarg
) -> None:
```
Row dict gains `"ticker": ticker`. `flush_llm()` is unchanged -- BQ `insert_rows_json` accepts the new key once the column exists.

**`cost_tracker.py:83` -- `AgentCostEntry`:**
```python
@dataclass
class AgentCostEntry:
    agent_name: str
    model: str
    ...
    ticker: str | None = None   # <-- new field
```
`CostTracker.record()` gains `ticker: str | None = None` kwarg, sets `entry.ticker`. `summarize()` can then add a `ticker_breakdown` dict alongside `model_breakdown`.

#### (c) HOW: threading `ticker` through the call chain

The cleanest propagation avoids touching all 15+ `run_*_agent` method signatures. Use the `generation_config` dict as a side-channel (same pattern used by `skill_file_id` in phase-25.D9.1):

**Step 1** -- `_generate_with_retry` pops `_ticker` from `generation_config` before passing config to `model.generate_content`:
```python
def _generate_with_retry(self, model, prompt, agent_name, max_retries=3, timeout=90,
                          is_deep_think=False, generation_config=None, is_grounded=False):
    # pull private key before forwarding config
    _ticker = None
    if generation_config and "_ticker" in generation_config:
        generation_config = dict(generation_config)
        _ticker = generation_config.pop("_ticker", None)
    ...
    # pass to record()
    ct.record(agent_name, model_name, response, ..., ticker=_ticker)
```

**Step 2** -- Each `run_*_agent(ticker, ...)` caller passes `_ticker` in `generation_config`:
```python
response = self._generate_with_retry(
    self.general_client, prompt, "Insider",
    generation_config={"_ticker": ticker},
)
```
For callers that already pass a `generation_config`, merge `{"_ticker": ticker}` in.

**Step 3** -- `ClaudeClient.generate_content` receives `config` after `_ticker` is already stripped. It never sees the private key. The `log_llm_call` call at line 1543 is updated to pass `ticker=_ticker` sourced from the caller. This requires `_generate_with_retry` to thread `_ticker` into `generate_content` via a separate kwarg or the `config` dict before stripping -- the cleanest approach is a new optional kwarg `_ticker: str | None = None` on `ClaudeClient.generate_content` that is silently ignored by `GeminiClient` (via **kwargs or explicit signature addition).

**Step 4 (Gemini instrumentation)**: Add a `log_llm_call` call to `GeminiClient.generate_content` immediately after the response is returned, mirroring the Claude pattern. Extract `_ticker` from `generation_config` the same way. This closes the Gemini attribution gap.

**Scope boundary for 25.S.1**: migration + `log_llm_call` ticker kwarg + `AgentCostEntry.ticker` + ClaudeClient instrumentation + `_generate_with_retry` threading + GeminiClient instrumentation. OpenAIClient is low-priority (GitHub Models path, low spend) and can be deferred.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched)
- [x] 10+ unique URLs total (12 collected: 5 read in full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (cost_tracker, llm_client, orchestrator, api_call_log, migration, paper_trading attribution endpoint)
- [x] Contradictions / consensus noted (unanimous: application-layer tagging)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
