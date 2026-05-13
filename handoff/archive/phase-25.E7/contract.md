---
step: 25.E7
slug: yfinance-price-history-guard
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.E7

## Step ID + masterplan reference

`25.E7` -- "yfinance_tool.get_price_history() try/except + counter"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`. Leverages 25.B7's data_source_events table.

## Hypothesis

`get_price_history` is the only completely-unguarded yfinance call in
the tool. Wrapping it in try/except + logging + a data_source_events
row on failure means:
- Callers receive a structured error instead of an unhandled HTTPError.
- Operators see WARNING-level log entries for rate-limits.
- The new `yfinance_price_history` source key on `data_source_events`
  feeds the existing `pct_yfinance_fallback_dominance` aggregation
  query (extended by source value).

## Success criteria (verbatim from masterplan.json)

> `get_price_history_returns_error_dict_on_failure`
>
> `failure_counter_incremented_and_persisted_to_bq`

## Plan steps

1. **`backend/tools/yfinance_tool.py::get_price_history`** -- restructure:
   - Wrap `yf.Ticker(ticker).history(period=period)` in try/except.
   - On exception: log WARNING + persist `data_source_events` row
     (`source="yfinance_price_history"`, `kind="fallback"`, `notes=<exc>`).
   - On empty DataFrame: log WARNING + same persist + return error-list.
   - Return `[{"error": <str>, "ticker": ticker}]` in both failure cases.
2. **Verifier** -- `tests/verify_phase_25_E7.py` with 4 claims:
   - Claim 1: source contains try/except in the function.
   - Claim 2: source references `save_data_source_event` + `yfinance_price_history`.
   - Claim 3: behavioral -- patch `yf.Ticker` to raise; call
     `get_price_history`; assert returns 1-element list with error key.
   - Claim 4: behavioral -- patch the BQ save method to count invocations;
     call get_price_history with failing yf; assert save called once.

## Files

| File | Action |
|------|--------|
| `backend/tools/yfinance_tool.py` | Wrap get_price_history + counter |
| `tests/verify_phase_25_E7.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_E7.py
```

## Live-check

`Inject yfinance rate-limit; verify error returned not propagated`.
Will write `handoff/current/live_check_25.E7.md`.

## Risks + mitigations

- **Risk**: Existing callers expect `list[dict]` happy-path rows; the
  error-shape `[{"error": ..., "ticker": ...}]` is a list-of-one-dict,
  same outer type. Callers iterating won't crash; they'll see one row
  with the `error` key and zero OHLCV fields.
  **Mitigation**: Callers can defensively check `if rows and "error" in rows[0]`.
- **Risk**: BQ persist failure cascades back to the caller.
  **Mitigation**: BQ call is in its own try/except (fail-open WARNING).

## References

- `handoff/current/research_brief.md`
- `backend/tools/yfinance_tool.py:84-88`
- `backend/db/bigquery_client.py::save_data_source_event` (25.B7)
- `.claude/masterplan.json::25.E7`
