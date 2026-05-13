---
step: phase-25.S.1
cycle: 105
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.S.1

## What was built/changed

Closed the per-call ticker tagging follow-up from 25.S (cycle 88). The
`/attribution` endpoint shipped a proportional-by-trade-count cost
split first-pass; this cycle wires per-call ticker through the LLM
call path so exact `SUM(cost) GROUP BY ticker` becomes possible.

### North-star alignment

Closes the cost-denominator side of the auto-switch goal-c at the
ticker granularity. With per-call ticker tagging the meta-evolution
layer can auto-prune tickers where LLM cost > realized profit -- a
direct ticker-level rendering of "shift strategy to whichever is
making the most money". Compounds with 25.D9.1 (cycle 104) and
25.C9.1 (cycle 103) cost-reductions for tight `profit_per_llm_dollar`
attribution.

### Files changed

| File | Action |
|------|--------|
| `scripts/migrations/add_ticker_to_llm_call_log.py` | NEW idempotent migration (`ALTER TABLE ADD COLUMN IF NOT EXISTS ticker STRING`); modeled on 25.B7 / 25.Q pattern (dry-run default, `--apply` flag) |
| `backend/services/observability/api_call_log.py` | `log_llm_call(..., ticker=None)` kwarg; `"ticker": ticker` added to row dict |
| `backend/agents/cost_tracker.py` | NEW `AgentCostEntry.ticker: Optional[str]` field; `CostTracker.record(..., ticker=None)` kwarg; threaded into entry construction |
| `backend/agents/orchestrator.py` | `_generate_with_retry` plucks `call_ticker = generation_config.get("_ticker")` (read-only, no mutation) and passes `ticker=call_ticker` to `ct.record()` |
| `backend/agents/llm_client.py` | `ClaudeClient.generate_content` passes `ticker=config.get("_ticker")` to `log_llm_call()` |
| `tests/verify_phase_25_S_1.py` | NEW (7 claims) |
| `.claude/masterplan.json` | NEW 25.S.1 step entry (post-25.S) |

### Side-channel design (per research-brief recommendation)

`generation_config["_ticker"]` carries the ticker through the call chain
without changing the 11+ `run_*_agent` method signatures. Same pattern
as 25.D9.1's `skill_file_id`. Underscore prefix marks it as an
orchestrator-private key so API-request builders ignore it.

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_S_1.py

=== phase-25.S.1 verification ===

[PASS] 1. migration_script_adds_ticker_column_to_llm_call_log
[PASS] 2. log_llm_call_persists_ticker_in_row_dict
[PASS] 3. cost_tracker_record_accepts_ticker_kwarg
[PASS] 4. generate_with_retry_propagates_ticker_from_generation_config
[PASS] 5. claude_client_generate_content_passes_ticker_to_log_llm_call
[PASS] 6. behavioral_record_carries_ticker_to_entry
        -> entry.ticker=AAPL
[PASS] 7. behavioral_log_llm_call_buffer_row_carries_ticker
        -> buffered_row.ticker=MSFT

ALL 7 CLAIMS PASS
```

AST clean on all 5 touched .py files.

## Success criteria -> evidence

1. `migration_script_adds_ticker_column_to_llm_call_log` -- Claim 1 PASS.
2. `cost_tracker_record_accepts_ticker_kwarg` -- Claim 3 + 6 PASS (signature regex + live record() invocation with ticker="AAPL" returning entry.ticker=="AAPL").
3. `generate_with_retry_propagates_ticker_from_generation_config` -- Claim 4 PASS (grep for `call_ticker = generation_config.get("_ticker")` AND `ticker=call_ticker` in record() call).
4. `log_llm_call_persists_ticker_in_row_dict` -- Claim 2 + 7 PASS (signature regex + buffered row inspection confirming `ticker` field present with the provided value).

## Out-of-scope / deferred

- `run_*_agent` callers passing `{"_ticker": ticker, ...}` (caller adoption): 25.S.1.1 follow-up. Today the mechanism is shipped; until callers set the key, `_ticker` is None and the column remains NULL (still valid -- pre-existing rows unaffected).
- GeminiClient.generate_content instrumentation: GeminiClient doesn't call `log_llm_call` at all today (researcher noted this gap). 25.S.2 follow-up.
- BQ migration `--apply` execution: operator runs once after pull (the dry-run path is the safe default; verifier checks SQL string).

## References

- `handoff/current/research_brief.md` (5 sources, gate_passed=true)
- `backend/services/observability/api_call_log.py:181-300` (log_llm_call)
- `backend/agents/cost_tracker.py:82-200` (AgentCostEntry, record)
- `backend/agents/orchestrator.py:510-525` (_generate_with_retry pluck)
- `backend/agents/llm_client.py:1539-1573` (ClaudeClient log_llm_call)
- `.claude/masterplan.json::25.S.1`
