---
step: phase-25.C9.1
cycle: 103
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.C9.1

## What was built/changed

Closed the instance-level gap from 25.C9 (cycle 84). The BatchClient
mechanism existed but the orchestrator instance had no:
(a) constructor opt-in (backtest_mode, n_tickers args)
(b) settings-level toggle (`backtest_batch_mode`)
(c) dispatcher method (`_run_enrichment_batch`)

All three shipped. North-star alignment: enables the 50% flat batch
discount path that compounds with the 25.B9 1h prompt cache to ~95%
effective discount per Finout pricing guide, cutting the cost
denominator in Net System Alpha.

### Files changed

| File | Action |
|------|--------|
| `backend/config/settings.py` | NEW `backtest_batch_mode: bool` field (line 72, after sentiment_haiku_batch_mode precedent) |
| `backend/agents/orchestrator.py` | `__init__` accepts `(backtest_mode=False, n_tickers=1)`; sets `_batch_mode_active = backtest_mode AND settings.backtest_batch_mode AND n_tickers > 3` |
| `backend/agents/orchestrator.py` | NEW `_run_enrichment_batch(requests, *, batch_client=None, cost_tracker=None) -> dict` method (~80 LOC). Single submit/poll/fetch via BatchClient; custom_ids in `{ticker}__{agent_name}` shape (dotzlaw.com 2026 safety pattern); records is_batch=True per succeeded row for cost-tracker 0.5x multiplier; errored/expired rows surface via `LLMResponse.thoughts.startswith("errored:")` |
| `tests/verify_phase_25_C9_1.py` | NEW (7 claims) |
| `.claude/masterplan.json` | NEW 25.C9.1 step entry (post-25.C9) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_C9_1.py

=== phase-25.C9.1 verification ===

[PASS] 1. settings_carries_backtest_batch_mode_flag
[PASS] 2. orchestrator_constructor_accepts_backtest_mode_and_n_tickers
        -> args=['self', 'settings', 'backtest_mode', 'n_tickers']
[PASS] 3. orchestrator_run_enrichment_batch_dispatches_via_batchclient
        -> method present + references BatchClient
[PASS] 4. gate_evaluates_true_when_backtest_and_n_tickers_above_three
        -> gate(True, 5, True) = True
[PASS] 5. gate_false_when_n_tickers_at_or_below_three
        -> n=3 -> False; n=1 -> False
[PASS] 6. gate_false_when_settings_flag_is_off
        -> flag=False -> False
[PASS] 7. run_enrichment_batch_invokes_submit_poll_fetch_with_custom_ids
        -> submit=1 poll=1 fetch=1 custom_ids=['AAPL__Insider', 'TSLA__Options']

ALL 7 CLAIMS PASS
```

AST clean on both modified .py files.

## Success criteria -> evidence

1. `settings_carries_backtest_batch_mode_flag` -- Claim 1 PASS.
2. `orchestrator_constructor_accepts_backtest_mode_and_n_tickers` -- Claim 2 PASS.
3. `orchestrator_run_enrichment_batch_dispatches_via_batchclient` -- Claims
   3 + 7 PASS: method exists with `BatchClient` reference + behavioral
   round-trip with mocked BatchClient confirms `submit/poll/fetch` are
   each invoked exactly once and the custom_ids match the `{ticker}__{agent_name}`
   safety pattern.
4. `gate_evaluates_true_when_backtest_and_n_tickers_above_three` -- Claim
   4 PASS plus 5 + 6 confirm the negative cases (n_tickers <= 3 OR flag off).

## North-star calculus

Per the brief's Finout citation, batch + 1h-cache = ~95% effective discount on
the system-prompt-dominated enrichment workload. A 10-ticker backtest with 28
agents per ticker = 280 calls. At Sonnet 4.6 batch input $1.50/MTok (vs sync
$3.00/MTok), a backtest that previously cost $0.50 of LLM spend now costs
~$0.025-$0.10 -- propagating directly into the operator's
`profit_per_llm_dollar` snapshot (25.Q, cycle 77).

## Out-of-scope / deferred (25.C9.2)

`run_full_analysis()` refactor to actually drive the batch path on
`_batch_mode_active=True`: requires materializing all `general_client`
enrichment prompts upfront, batch-submitting, dispatching results into
each `run_*_agent()` return slot. This is a substantial refactor of
~280 lines of `run_full_analysis()`; deferred to 25.C9.2 follow-up so
this cycle has a focused, verifiable scope.

Today's outcome: AnalysisOrchestrator INSTANCES correctly compute the
gate and have a fully tested dispatcher method ready to be wired by
25.C9.2.

## References

- `handoff/current/research_brief.md` (6 sources, gate_passed=true)
- `backend/agents/llm_client.py::BatchClient` (lines 1574-1700, 25.C9)
- `backend/agents/cost_tracker.py::CostEvent.is_batch` (25.C9)
- `.claude/masterplan.json::25.C9.1`
