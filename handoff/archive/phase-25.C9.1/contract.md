---
step: 25.C9.1
slug: orchestrator-batchclient-routing
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.C9.1

## Step ID + masterplan reference

`25.C9.1` -- "Orchestrator instance-level BatchClient routing (gate +
dispatcher method)" (P2, harness_required, depends on 25.C9 done at
cycle 84).

## Research-gate summary

Tier=moderate. Brief at `handoff/current/research_brief.md`,
`gate_passed=true` (6 sources fetched in full). Key findings:
- 50% discount confirmed against live platform.claude.com pricing.
- Polling crossover ~100 requests; 10 tickers x 28 agents = 280 calls.
- Window-batching is the dominant production pattern.
- custom_id pattern `{ticker}__{agent_name}` is safety-critical.
- Errored entries flow via `LLMResponse.thoughts.startswith("errored:")`.

## North-star alignment

Directly cuts the cost denominator in Net System Alpha
(profit / cost). Backtest runs are the largest LLM-cost line item;
50% off them propagates into the operator's profit-per-LLM-dollar
metric tracked by 25.Q (cycle 77, efficiency_snapshots).

## Hypothesis

Adding `(backtest_mode, n_tickers)` constructor args + a
`backtest_batch_mode` settings flag + a `_run_enrichment_batch(requests)`
dispatcher method gives the orchestrator the instance-level routing
gate AND a clean entry point for the eventual `run_full_analysis()`
refactor. The single-ticker live path is untouched (default
`backtest_mode=False`).

## Success criteria (verbatim from masterplan.json)

> `settings_carries_backtest_batch_mode_flag`
>
> `orchestrator_constructor_accepts_backtest_mode_and_n_tickers`
>
> `orchestrator_run_enrichment_batch_dispatches_via_batchclient`
>
> `gate_evaluates_true_when_backtest_and_n_tickers_above_three`

## Plan steps

1. **`backend/config/settings.py`** -- add
   `backtest_batch_mode: bool = Field(False, description=...)` to the
   Settings class.
2. **`backend/agents/orchestrator.py::AnalysisOrchestrator.__init__`**:
   - Add `backtest_mode: bool = False, n_tickers: int = 1` kwargs.
   - Compute and store
     `self._batch_mode_active = backtest_mode and self.settings.backtest_batch_mode and n_tickers > 3`.
3. **NEW method `_run_enrichment_batch(requests: list[dict]) -> dict[str, "LLMResponse"]`**:
   - Builds custom_ids in `{ticker}__{agent_name}` shape from each request.
   - Calls `BatchClient.submit() -> poll() -> fetch()`.
   - Returns a dict keyed by custom_id; errored/expired rows surface
     `LLMResponse(text="", thoughts="errored: ...")` (already handled by
     the existing BatchClient.fetch path).
4. **Verifier** `tests/verify_phase_25_C9_1.py` with 6 claims:
   - C1: Settings has `backtest_batch_mode` field.
   - C2: orchestrator constructor accepts `backtest_mode`, `n_tickers`.
   - C3: gate is True when `(backtest_mode=True, n_tickers=5,
     settings.backtest_batch_mode=True)`.
   - C4: gate is False when n_tickers <= 3.
   - C5: gate is False when settings.backtest_batch_mode is False.
   - C6: behavioral round-trip -- patch BatchClient with stubbed
     submit/poll/fetch; call `_run_enrichment_batch([request_a, request_b])`;
     assert submit/poll/fetch invoked + dict returned with correct custom_ids.

## Files

| File | Action |
|------|--------|
| `backend/config/settings.py` | Add `backtest_batch_mode` field |
| `backend/agents/orchestrator.py` | Add constructor args + `_run_enrichment_batch` method |
| `tests/verify_phase_25_C9_1.py` | NEW (6 claims) |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_C9_1.py
```

## Live-check

`Instantiate AnalysisOrchestrator(settings, backtest_mode=True, n_tickers=5); _batch_mode_active is True and _run_enrichment_batch dispatches via mocked BatchClient`.

## Risks + mitigations

- **Risk**: existing callers that instantiate `AnalysisOrchestrator(settings)`
  break.
  **Mitigation**: both new kwargs are optional with safe defaults
  (`backtest_mode=False`, `n_tickers=1`). The gate stays False in all
  existing call paths.
- **Risk**: `run_full_analysis()` refactor to actually use the batch
  path is not included here.
  **Mitigation**: documented as 25.C9.2 follow-up. This cycle ships the
  mechanism + gate so 25.C9.2 has a clean surface to wire against
  (mirrors 25.D9 -> 25.D9.1 pattern).

## References

- `handoff/current/research_brief.md`
- `backend/agents/llm_client.py::BatchClient` (lines 1574-1700)
- `backend/agents/cost_tracker.py::CostEvent.is_batch` (25.C9)
- `.claude/masterplan.json::25.C9.1`
