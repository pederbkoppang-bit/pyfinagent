---
step: phase-23.1.14
title: Legacy-position sector lookup + live-derived NAV scoreboards
cycle_date: 2026-04-29
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_14.py'
research_brief: handoff/current/phase-23.1.14-external-research.md (also see phase-23.1.14-internal-codebase-audit.md)
---

# Contract — phase-23.1.14

## Hypothesis

Two coordinated bugs from phase-23.1.13:

**A.** `decide_trades` in `backend/services/portfolio_manager.py` reads
`pos.get("sector")` on every existing position to seed `sector_counts`.
The 11 legacy rows in BQ `paper_positions` predate the sector column,
so all of them fall into `"Unknown"`, `sector_counts["Technology"]==0`,
and new Tech BUYs (MU, KEYS today) trivially pass the cap.

**B.** Hero scoreboards (NAV, Total P&L) on the paper-trading page read
`status?.portfolio.nav` / `status?.portfolio.pnl_pct` — both BQ snapshot
fields updated only at end-of-cycle. The position table immediately
below already derives live values from `useLivePrices` on every 30s
yfinance tick. Result: $329.49 discrepancy between scoreboards
($13,952.25) and table ($14,281.74).

If we (1) enrich `current_positions` with their true GICS sector via
the existing `_fetch_ticker_meta` helper before `decide_trades`, and
(2) compute `liveNav = cash + sum(livePrice * qty)` in the page and
pass to `SummaryHero`, then tomorrow's cycle will correctly skip new
Technology candidates and the scoreboard / table values will match on
every tick.

## Research-gate summary

- External brief: `handoff/current/phase-23.1.14-external-research.md`
  — 6 sources read in full (Alpaca, IBKR, Python asyncio, BBC
  Engineering, Sentry FastAPI, Fume Finance), 16 URLs collected,
  recency scan 2024-2026 performed. `gate_passed: true`.
- Internal audit: `handoff/current/phase-23.1.14-internal-codebase-audit.md`
  — 5 files inspected with file:line anchors and concrete patch
  sketches.

Key finding: enrich in the async caller (`autonomous_loop.py`) via
`asyncio.to_thread(_fetch_ticker_meta, ...)` — same pattern already
used at line 179 for top-N candidate enrichment. Keeps `decide_trades`
sync. Live-NAV: lift the `tab === "positions"` gate on `useLivePrices`
and add a `useMemo` for `liveNav` / `liveTotalPnlPct`.

## Plan steps

1. `backend/services/autonomous_loop.py`: between `positions =
   trader.get_positions()` (post-MTM refresh, ~line 317) and
   `orders = decide_trades(...)` (~line 322), add a sector-enrichment
   block that calls `_fetch_ticker_meta` via `asyncio.to_thread` for
   legacy positions whose `pos.get("sector")` is empty. Skips when
   `paper_max_per_sector == 0`. Best-effort: failure is logged
   non-fatal and the cycle continues.

2. `frontend/src/app/paper-trading/page.tsx`:
   - Lift the `tab === "positions"` gate on `useLivePrices` so live
     ticks flow regardless of which tab is open (cap to 25 tickers
     to keep yfinance load bounded).
   - Add `useMemo` for `liveNav` / `liveTotalPnlPct` using the
     `cash + sum(livePrice * qty)` derivation that the position table
     already uses around line 791.
   - Add `liveNav` / `liveTotalPnlPct` props to `SummaryHero` and
     render them instead of the stale BQ snapshot fields. Keep
     Sharpe + Cash + Positions as is.

3. `tests/services/test_sector_concentration.py`: add 2 tests
   covering the enrichment contract (positions with sector populated
   block new same-sector BUYs; positions without sector fall into
   "Unknown" — regression guard documenting Bug A baseline).

4. `tests/verify_phase_23_1_14.py`: immutable verification asserting
   the autonomous_loop enrichment block exists, the page.tsx liveNav
   memo exists, the SummaryHero signature accepts liveNav, and both
   new tests pass.

## Immutable verification command

```bash
source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_14.py
```

Must exit 0 and print an `ok` line covering all four claims.

## Acceptance criteria

- `pytest tests/services/test_sector_concentration.py -q` passes,
  including 2 new tests.
- `python tests/verify_phase_23_1_14.py` exits 0 with one ok-line.
- `cd frontend && npx tsc --noEmit` is silent / exit 0.
- Logs from a manual `/run-now` show the enrichment block running
  (`Enriched N legacy positions with sector ...`).
- Scoreboard NAV matches `cash + sum(live position MV)` on every
  30s tick.

## Backwards compatibility

- `paper_max_per_sector=0` short-circuits the new enrichment block
  (no extra yfinance calls).
- When `livePrices` is empty, `liveNav` falls back to
  `status?.portfolio.nav`.
- 24h `_fetch_ticker_meta` cache means subsequent cycles incur near
  zero overhead.

## References

- `handoff/current/phase-23.1.14-external-research.md`
- `handoff/current/phase-23.1.14-internal-codebase-audit.md`
- `backend/services/portfolio_manager.py:187-251` (existing sector cap)
- `backend/services/autonomous_loop.py:175-192` (existing meta enrichment)
- `frontend/src/app/paper-trading/page.tsx:177-200` (SummaryHero), `:411-415` (gated useLivePrices), `:791-801` (live table derivation)
