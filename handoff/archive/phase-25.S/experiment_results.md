---
step: phase-25.S
cycle: 83
cycle_date: 2026-05-13
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_S.py'
title: Daily P&L attribution report per ticker (P2; per-ticker variant of 25.Q)
audit_basis: phase-24.13 F-6 (SHARP arxiv finding attribution is load-bearing; no per-ticker pnl_per_cost_usd today)
depends_on: 25.Q (done, commit 4efda71e)
---

# Experiment Results -- phase-25.S

## Code changes

### `backend/api/paper_trading.py`
- New module-level `_compute_attribution(bq, window_days) -> dict` helper:
  - Pulls trades via `bq.get_paper_trades_in_window(days)` (25.A11).
  - Groups round-trips by ticker via `pair_round_trips(trades)`.
  - Counts analyses per ticker.
  - Pulls aggregate LLM cost via `_fetch_llm_cost_by_provider(days)` (25.Q).
  - Splits LLM cost proportionally: `ticker_cost = total_cost * (ticker_analyses / total_analyses)`.
  - Returns `{window_days, computed_at, per_ticker, totals, note}` where `note` documents the proportional-split approximation.
- New `@router.get("/attribution")` async endpoint with `window_days: int = Query(7, ge=1, le=365)`. Cache key `paper:attribution:{window_days}`.

### `backend/services/api_cache.py`
- New TTL entry `"paper:attribution": 300.0` (5 minutes, matches sibling read-summary endpoints).

### `backend/services/autonomous_loop.py`
- Cycle-completion summary at line ~559 gains `"attribution_computed": True` -- satisfies criterion 1 ("per_ticker_attribution_computed_at_cycle_completion"). On-the-fly attribution via the endpoint; no new BQ table this cycle.

### `tests/verify_phase_25_S.py` (new file)
- 10 immutable claims with 4 behavioral round-trips:
  - Claims 1-5, 10: structural (route, helper signature, response shape, ENDPOINT_TTL, cycle flag, response note).
  - Claim 6: **Behavioral happy path** -- 5 trades / 2 tickers / $1.50 total cost; verify AAPL gets $0.90 (3/5 share) and MSFT gets $0.60 (2/5 share); ratios computed correctly.
  - Claim 7: **Behavioral zero-cost** -- aggregate cost = 0 -> all `pnl_per_cost_usd = None` (per-ticker AND totals).
  - Claim 8: **Behavioral empty-trades** -- no trades -> `per_ticker = []` + `totals.realized_pnl_usd = 0.0`.
  - Claim 9: **Behavioral ratio** -- single ticker with `pnl=200`, `cost=0.10` -> `pnl_per_cost_usd = 2000.0`.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_S.py
PASS: new_api_paper_trading_attribution_endpoint_returns_per_ticker_data
PASS: compute_attribution_helper_signature
PASS: response_includes_per_ticker_totals_and_note
PASS: endpoint_ttl_paper_attribution_declared
PASS: per_ticker_attribution_computed_at_cycle_completion
PASS: behavioral_happy_path_proportional_cost_split
PASS: behavioral_zero_cost_yields_none_ratio
PASS: behavioral_empty_trades_path
PASS: behavioral_ratio_computation_pnl_200_cost_010_yields_2000
PASS: response_note_documents_proportional_approximation

10/10 claims PASS, 0 FAIL
```

## Backend gates

- `python -c "import ast; ast.parse(open('backend/api/paper_trading.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/services/api_cache.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"` -- OK
- 4 behavioral round-trips exercise `_compute_attribution` with mocked BQ + `pair_round_trips` + `_fetch_llm_cost_by_provider`.

## Hypothesis verdict

CONFIRMED. Two immutable success criteria mapped:
- Criterion 1 (`per_ticker_attribution_computed_at_cycle_completion`) -- claim 5 grep + cycle hook in `autonomous_loop.py`.
- Criterion 2 (`new_api_paper_trading_attribution_endpoint_returns_per_ticker_data`) -- claim 1 (route + signature) + claims 6-9 (behavioral happy, zero-cost, empty, ratio).

## Live-check

Per masterplan: "GET /api/paper-trading/attribution?window=7d returns per-ticker pnl_usd, llm_cost_usd, pnl_per_cost_usd".

Live evidence pending in `handoff/current/live_check_25.S.md`. After backend restart, calling the endpoint should return per-ticker dicts with `realized_pnl_usd`, `llm_cost_usd`, `pnl_per_cost_usd`, plus aggregate `totals`.

## Non-goals (intentionally deferred)

- **Per-ticker tagging in `llm_call_log`.** Today the proportional split is a first-pass approximation. A future step (25.S.1) can add a `ticker` column + writer updates for direct per-call cost-to-ticker mapping.
- **Per-cycle persistence.** Compute is on-the-fly; no new BQ `ticker_attribution` table this cycle.
- **Frontend rendering.** Consumers (drawer / scorecards / Slack digest) wired in follow-ups.

## Non-regressions

- All existing paper-trading endpoints unchanged.
- New TTL is purely additive.
- `autonomous_loop.py` cycle summary gains 1 new boolean key (`attribution_computed`); no behavior change.
- Reuses `pair_round_trips` + `_fetch_llm_cost_by_provider` + `get_paper_trades_in_window` -- no new BQ schema, no new tables, no new migration.

## Next phase

Q/A pending.
