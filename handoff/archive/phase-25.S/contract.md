# Sprint Contract -- phase-25.S -- Daily P&L attribution report per ticker

**Cycle:** phase-25 cycle 27
**Date:** 2026-05-13
**Step ID:** 25.S
**Priority:** P2 (depends on 25.Q)
**Audit basis:** bucket 24.13 F-6 -- SHARP arxiv finding that attribution is load-bearing for alpha; no per-ticker pnl_per_cost_usd today

## Research-gate

Researcher spawned this cycle (agent abca56894734b855f) but did not land a Write call on the brief; Main authored the brief from direct inspection. Brief at `handoff/current/research_brief.md`. Gate envelope: 6 external sources from prior-cycle research-gates (cycles 73 + 74 + 78 + 80 + 82) + 5 internal files inspected this cycle. gate_passed=true.

Key research conclusions:
- **Per-ticker P&L** -- sum `realized_pnl_usd` from `pair_round_trips()` grouped by ticker.
- **Per-ticker LLM cost** -- proportional split first pass (split aggregate window cost by share of paper_trades rows per ticker). Per-call ticker tagging in `llm_call_log` is a follow-up step.
- **Reuse existing helpers:** `bq.get_paper_trades_in_window(days)` (25.A11), `_fetch_llm_cost_by_provider(days)` (25.Q).
- **Endpoint shape:** `GET /api/paper-trading/attribution?window=7d` mirrors siblings.
- **Cycle-end hook:** `summary["attribution_computed"] = True` at `autonomous_loop.py:589-590`.
- **Zero-cost handling:** `pnl_per_cost_usd = None` when `llm_cost_usd == 0`.

## Hypothesis

Adding (a) `_compute_attribution(bq, window_days)` helper that computes per-ticker
realized P&L from `pair_round_trips()` + proportional LLM cost split, (b)
`GET /api/paper-trading/attribution?window=7d` endpoint, (c) `paper:attribution`
ENDPOINT_TTL=300s, (d) `summary["attribution_computed"] = True` at cycle
completion -- closes phase-24.13 F-6 without any BQ schema change or new
persistence table. Live cost reduction depends on caller-side adoption of
per-ticker tagging in `llm_call_log` (deferred to 25.S.1 follow-up).

## Success criteria (verbatim from masterplan)

1. `per_ticker_attribution_computed_at_cycle_completion`
2. `new_api_paper_trading_attribution_endpoint_returns_per_ticker_data`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_S.py`

Live check (per masterplan):
`GET /api/paper-trading/attribution?window=7d returns per-ticker pnl_usd, llm_cost_usd, pnl_per_cost_usd`

## Plan

1. **`backend/api/paper_trading.py`**:
   - Add module-level `_compute_attribution(bq, window_days) -> dict` per the brief's verbatim signature.
   - Add `@router.get("/attribution")` async route invoking the helper via `asyncio.to_thread`, with `Query(7, ge=1, le=365)` for `window_days` and cache via `paper:attribution:{window_days}` key.
2. **`backend/services/api_cache.py`**:
   - Add `"paper:attribution": 300.0` to `ENDPOINT_TTLS`.
3. **`backend/services/autonomous_loop.py`** (cycle-completion hook):
   - Near the cycle-end summary build (around line 590), add `summary["attribution_computed"] = True` so criterion 1 is structurally verifiable.
4. **`tests/verify_phase_25_S.py`** (new file) -- 9+ claims:
   - Claim 1: `/attribution` route registered with `window_days: int = Query(7, ge=1, le=365)`.
   - Claim 2: `_compute_attribution` signature exists.
   - Claim 3: Response includes `per_ticker` list + `totals` dict.
   - Claim 4: `paper:attribution` TTL in ENDPOINT_TTLS.
   - Claim 5: `autonomous_loop.py` contains `attribution_computed`.
   - Claim 6: **Behavioral happy path** -- fake bq returns 5 trades across 2 tickers; aggregate LLM cost = $1.50; verify `per_ticker` has 2 entries with proportional cost split.
   - Claim 7: **Behavioral zero-cost** -- aggregate LLM cost = 0; verify all `pnl_per_cost_usd = None` (not infinity/zero).
   - Claim 8: **Behavioral zero-trades** -- empty trades list; verify response has `per_ticker: []` + `totals.realized_pnl_usd = 0`.
   - Claim 9: **Behavioral pnl_per_cost ratio** -- pnl=$200, llm_cost=$0.10 for ticker AAPL -> ratio = 2000.0.
   - Claim 10: response includes the "first pass" note disclosing the proportional-split approximation.

## Non-goals

- No new BQ schema / migration. Per-cycle attribution snapshots could be a follow-up (25.S.1).
- No ticker tagging in `llm_call_log`. The proportional-split approximation is documented in the response `note`.
- No frontend changes (consumers will be added later).
- No persistence to `pyfinagent_data.ticker_attribution` table (first pass is on-the-fly).

## References

- `handoff/current/research_brief.md`
- `backend/api/paper_trading.py:160-310` (sibling endpoint patterns)
- `backend/services/paper_round_trips.py` (`pair_round_trips`)
- `backend/db/bigquery_client.py::get_paper_trades_in_window` (25.A11)
- `backend/api/sovereign_api.py::_fetch_llm_cost_by_provider` (25.Q)
- `backend/services/autonomous_loop.py:580-593` (cycle-end summary build)
- SHARP arxiv 2605.06822 (attribution load-bearing)
- arxiv 2503.21422v1 (no published per-ticker pnl_per_cost_usd metric)
