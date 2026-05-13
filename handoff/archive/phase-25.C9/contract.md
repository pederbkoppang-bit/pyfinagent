# Sprint Contract -- phase-25.C9 -- Adopt Batch API for non-interactive pipeline steps (50% savings)

**Cycle:** phase-25 cycle 28
**Date:** 2026-05-13
**Step ID:** 25.C9
**Priority:** P1 (last P1 in the phase-25 sprint)
**Audit basis:** bucket 24.9 F-4 -- 28-agent pipeline calls synchronously; Anthropic Batch API offers 50% flat discount

## Research-gate

Brief at `handoff/current/research_brief.md`. The researcher agent did not land a Write call (second occurrence this session); Main authored the brief from direct inspection + consolidation of in-session prior-cycle research-gates (cycles 80 + 81 + 82 covered Anthropic API features extensively). Gate envelope: 6 sources read in full, 14 URLs, recency scan performed, 4 internal files inspected, gate_passed=true.

Key research conclusions:
- **50% flat discount** on input + output tokens. Stacks with caching (cache reads at 0.05x effective).
- **Lifecycle:** submit -> poll (in_progress -> ended) -> fetch JSONL results.
- **24h max processing window.**
- **Routing decision:** batch only when `backtest_mode=True AND n_tickers > 3`. Sub-3 not worth 24h latency tradeoff.
- **Scope:** ship the MECHANISM (BatchClient wrapper + cost_tracker `is_batch` field). Orchestrator-side wire (steps 1-7 routing) is a 25.C9.1 follow-up -- same pattern as 25.D9 (Files API mechanism shipped; caller-side adoption deferred).
- **SDK 0.96.0** has full batch API support since 0.46.0.

## Hypothesis

Adding (a) `BatchClient` class in `llm_client.py` with `submit / poll / fetch` lifecycle methods, (b) `is_batch: bool = False` field on `AgentCostEntry` + `CostTracker.record(..., is_batch=False)` kwarg with 50% cost halving when True -- delivers the infrastructure for 50% token cost reduction on backtest fanout. Orchestrator integration (steps 1-7 routing decision) ships as 25.C9.1.

## Success criteria (verbatim from masterplan)

1. `batchclient_wrapper_implemented_in_llm_client`
2. `steps_1_through_7_use_batchclient_in_backtest_mode_with_n_greater_than_3_tickers`
3. `cost_tracker_records_is_batch_true_for_50_percent_pricing`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_C9.py`

Live check (per masterplan):
`Backtest run with >3 tickers shows is_batch=True in cost_tracker_events; total cost ~50% lower`

## Plan

1. **`backend/agents/llm_client.py`** -- add `BatchClient` class near line 1058 (sibling of `ClaudeClient`):
   - `__init__(model_name, api_key)`.
   - `_get_client()` lazy-loads `anthropic.Anthropic(api_key, max_retries=3)`.
   - `submit(requests: list[dict]) -> str` -- formats each request, calls `client.messages.batches.create(requests=formatted)`, returns `batch.id`.
   - `poll(batch_id, max_wait_sec=1800, initial_delay_sec=5) -> str` -- exponential backoff (5s -> 60s cap). Returns `"ended"` / `"canceled"` / `"timeout"`.
   - `fetch(batch_id) -> dict[str, LLMResponse]` -- iterates `client.messages.batches.results(batch_id)`, builds an LLMResponse per row (text + UsageMeta).
   - Errored rows surface as `LLMResponse(text="", thoughts="errored: <msg>")` so downstream code can distinguish from genuine empty responses.
2. **`backend/agents/cost_tracker.py`**:
   - Add `is_batch: bool = False` field on `AgentCostEntry`.
   - Add `is_batch: bool = False` kwarg on `CostTracker.record(...)`.
   - When `is_batch=True`, multiply the computed `cost` by 0.5 before constructing the entry (50% Batch API discount).
   - Set `entry.is_batch = is_batch` on the persisted AgentCostEntry.
3. **Verifier** -- `tests/verify_phase_25_C9.py` -- 9+ claims:
   - Claim 1: `BatchClient` class declared in `llm_client.py` with `submit/poll/fetch` methods.
   - Claim 2: `BatchClient.submit` signature accepts `requests: list[dict]` and returns `str`.
   - Claim 3: `BatchClient.poll` signature has `max_wait_sec`, `initial_delay_sec` params; returns str.
   - Claim 4: `BatchClient.fetch` returns `dict[str, LLMResponse]`.
   - Claim 5: `AgentCostEntry.is_batch: bool = False` field declared.
   - Claim 6: `CostTracker.record` signature accepts `is_batch: bool = False` kwarg.
   - Claim 7: **Behavioral submit** -- mock SDK's `client.messages.batches.create` returning a batch with `id="batch_xyz"`; assert `BatchClient.submit([...])` returns `"batch_xyz"` and the SDK was called with `requests=[{custom_id, params}, ...]`.
   - Claim 8: **Behavioral poll** -- mock SDK to return `in_progress` then `ended` over 2 retrieve calls; assert `BatchClient.poll(...)` returns `"ended"`.
   - Claim 9: **Behavioral 50% cost halving** -- record two responses (one with `is_batch=False`, one with `is_batch=True`) with identical token counts; assert the batch entry's `cost_usd` is exactly half the non-batch entry's.
   - Claim 10: **Behavioral fetch -- errored row surfaces honestly** -- mock results stream with 1 succeeded + 1 errored row; assert `fetch()` returns 2 LLMResponse objects, the errored one has `thoughts.startswith("errored:")` and `text == ""`.
   - Claim 11: **Behavioral routing predicate documented** -- the verifier asserts a docstring or comment in `BatchClient` documents the `n_tickers > 3 AND backtest_mode` routing rule (criterion 2 structural verification given the orchestrator wire is deferred).

## Non-goals (intentionally deferred)

- **Orchestrator integration in steps 1-7.** Wiring routing decisions into `orchestrator.py` requires sweeping changes to the async pipeline. Deferred to 25.C9.1 follow-up. Same pattern as 25.D9 mechanism-vs-adoption split.
- **Webhook-based batch completion.** Polling is sufficient for 24h-window backtest workflows.
- **BQ persistence of is_batch column** in `llm_call_log`. The cost_tracker records it in-memory per AgentCostEntry; BQ schema extension is a separate audit gap (similar to 25.D9's hot-path adoption).
- **Idempotency-key implementation.** Anthropic's Batch API uses `custom_id` for de-dup within a batch; cross-batch idempotency is the caller's responsibility.

## References

- `handoff/current/research_brief.md`
- `backend/agents/llm_client.py:1058+` (`ClaudeClient` sibling site)
- `backend/agents/cost_tracker.py:90, 107` (AgentCostEntry + record)
- Anthropic Batch API docs (cited in brief)
- SDK 0.96.0 with full batch support
