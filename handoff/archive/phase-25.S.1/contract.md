---
step: 25.S.1
slug: per-call-ticker-tagging
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.S.1

## Step ID + masterplan reference

`25.S.1` -- "Per-call ticker tagging in llm_call_log + cost_tracker for
exact per-ticker attribution" (P2, harness_required, depends on
`25.S` done at cycle 88).

## Research-gate summary

Tier=moderate. Brief at `handoff/current/research_brief.md`,
`gate_passed=true` (5 sources fetched in full incl. Anthropic
Messages API docs, Traceloop, LiteLLM, AWS Bedrock attribution, and
Braintrust 2026). Key findings:
- Anthropic API has no per-call billing tag beyond `user_id`;
  application-layer tagging is the canonical pattern (LiteLLM,
  Braintrust, AWS, Traceloop all agree).
- `llm_call_log` table has 12 cols, no `ticker`. Migration needed.
- `log_llm_call` has one call site (ClaudeClient.generate_content).
- `_generate_with_retry` doesn't thread ticker today.
- Use `generation_config["_ticker"]` side-channel (same pattern as
  25.D9.1's `skill_file_id`).

## North-star alignment

Closes the cost-denominator side of the auto-switch goal-c at the
ticker level. With per-call ticker tagging the operator can compute
`SUM(llm_cost_usd) GROUP BY ticker` and join against paper_trades to
get exact `profit_per_llm_dollar` per ticker -- enabling the meta-
evolution layer to auto-prune tickers where LLM cost > realized profit.

## Hypothesis

Adding a `ticker STRING` column to `llm_call_log` + threading
`ticker` through `_generate_with_retry` -> `cost_tracker.record()` ->
`ClaudeClient.generate_content` -> `log_llm_call()` enables exact
per-ticker LLM-cost attribution. The wiring uses the
`generation_config["_ticker"]` side-channel established by 25.D9.1
so the 11+ `run_*_agent` call signatures don't change.

## Success criteria (verbatim from masterplan.json)

> `migration_script_adds_ticker_column_to_llm_call_log`
>
> `cost_tracker_record_accepts_ticker_kwarg`
>
> `generate_with_retry_propagates_ticker_from_generation_config`
>
> `log_llm_call_persists_ticker_in_row_dict`

## Plan steps

1. **NEW migration `scripts/migrations/add_ticker_to_llm_call_log.py`** --
   idempotent `ALTER TABLE ADD COLUMN IF NOT EXISTS ticker STRING` for
   `pyfinagent_data.llm_call_log`. Modeled on the 25.B7 / 25.Q migration
   pattern (dry-run default, --apply flag).
2. **`backend/services/observability/api_call_log.py::log_llm_call`** --
   add `ticker: str | None = None` kwarg + `"ticker": ticker` in row dict.
3. **`backend/agents/cost_tracker.py`** --
   - Add `ticker: str | None = None` to `AgentCostEntry`.
   - Add `ticker: str | None = None` kwarg to `CostTracker.record(...)`.
   - Store on the entry.
4. **`backend/agents/orchestrator.py::_generate_with_retry`** -- when
   `generation_config` carries `"_ticker"`, pluck it (don't mutate
   the dict in place; copy and pop) and pass `ticker=...` to
   `ct.record()`. Continue forwarding the full config to
   `model.generate_content` so ClaudeClient can also pluck it.
5. **`backend/agents/llm_client.py::ClaudeClient.generate_content`** --
   pluck `_ticker` from `generation_config` (if present) and pass
   `ticker=` to `log_llm_call()`.
6. **Verifier** -- `tests/verify_phase_25_S_1.py` with 6 claims:
   - C1: migration script exists with `ADD COLUMN IF NOT EXISTS ticker`.
   - C2: `log_llm_call` signature accepts `ticker` kwarg.
   - C3: `AgentCostEntry` has `ticker` field; `record()` accepts kwarg.
   - C4: `_generate_with_retry` strips `_ticker` from generation_config
     for the cost-tracker call.
   - C5: `ClaudeClient.generate_content` reads `_ticker` from config.
   - C6: behavioral round-trip -- patch `log_llm_call`; call a
     `CostTracker.record(... ticker="AAPL")`; assert ticker reaches
     the row dict.

## Files

| File | Action |
|------|--------|
| `scripts/migrations/add_ticker_to_llm_call_log.py` | NEW (idempotent ALTER TABLE) |
| `backend/services/observability/api_call_log.py` | Add `ticker` kwarg + row field |
| `backend/agents/cost_tracker.py` | Add `ticker` field + record kwarg |
| `backend/agents/orchestrator.py` | Pluck `_ticker` from generation_config in _generate_with_retry |
| `backend/agents/llm_client.py` | ClaudeClient.generate_content plucks `_ticker` for log_llm_call |
| `tests/verify_phase_25_S_1.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_S_1.py
```

## Live-check

`After running migration --apply and a Claude-mode analysis, BQ llm_call_log
table has non-NULL ticker for new rows and can be GROUP BY ticker`.

## Risks + mitigations

- **Risk**: `generation_config["_ticker"]` collides with skill_file_id
  shape (which also lives in generation_config).
  **Mitigation**: `_ticker` is a single key; helpers prefix with `_`
  to distinguish from API-bound keys. Both helpers pop_or_get safely.
- **Risk**: Anthropic API rejects unknown keys.
  **Mitigation**: `ClaudeClient.generate_content` plucks `_ticker`
  BEFORE forwarding the dict to the API request body; the actual
  request payload constructor only sees recognized keys.
- **Risk**: Gemini call sites also need ticker (but GeminiClient
  doesn't call log_llm_call at all today).
  **Mitigation**: This step's criterion is about
  `log_llm_call_persists_ticker_in_row_dict` -- Claude path satisfies
  it. Gemini instrumentation is 25.S.2 follow-up.
- **Risk**: `run_*_agent` callers don't pass `_ticker` yet.
  **Mitigation**: This cycle ships the mechanism. Caller adoption (the
  11+ run_*_agent methods passing `{"_ticker": ticker, "skill_file_id": ...}`)
  is 25.S.1.1 follow-up.

## References

- `handoff/current/research_brief.md` (5 sources, gate_passed=true)
- `backend/services/observability/api_call_log.py:181-294` (log_llm_call)
- `scripts/migrations/add_llm_call_log.py:38-55` (original DDL)
- `backend/agents/cost_tracker.py:82-185` (AgentCostEntry, record)
- `backend/agents/llm_client.py:1539-1557` (ClaudeClient generate_content -> log_llm_call)
- `backend/agents/orchestrator.py:510-567` (_generate_with_retry)
- `.claude/masterplan.json::25.S.1`
