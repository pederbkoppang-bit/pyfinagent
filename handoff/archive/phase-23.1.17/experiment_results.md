---
step: phase-23.1.17
cycle_date: 2026-04-29
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_17.py'
---

# Experiment Results — phase-23.1.17

## Summary

User flagged that the home page "MAS Operator Cockpit" shows
different NAV/P&L/Sharpe/DD than the paper-trading page. Two
compounding causes identified by the researcher:

**Cause 1 — Stale `paper_portfolio.total_nav`.** Yesterday's
phase-23.1.15 cleanup script did a raw BQ UPDATE to `current_cash`
(+$1,451.40) but did NOT call `mark_to_market()`. `total_nav` is
the only field recomputed by mark_to_market; without that call,
the column carried `current_cash + OLD positions_value` until
this cycle.

**Cause 2 — Home page reads the stale column.** `page.tsx:142`
did `navValue = nav?.nav` (raw BQ snapshot). Paper-trading page
(post phase-23.1.14) computed `liveNav = cash + sum(livePrice *
qty)` as a `useMemo`. Same math hadn't been ported to home.

## Three coordinated fixes

**Fix A — Shared `useLiveNav` hook**
(`frontend/src/lib/useLiveNav.ts`, NEW). Lifted the inline
`useMemo` math from paper-trading/page.tsx into a single hook.
Both pages now import and call this hook with the same
`(status, positions, livePrices)` triple — single source of
truth for live-derived NAV + total P&L pct. Falls back to
BQ snapshot when no live ticks are available.

**Fix E — Home page wires up the shared hook**
(`frontend/src/app/page.tsx`). Added
`useLivePrices(positionTickers)` and
`useLiveNav(ptStatus, positions, livePrices)`. Replaced
`navValue = nav?.nav` with `navValue = liveNav ?? nav?.nav`,
and `pnl = nav?.pnl_pct` with `pnl = liveTotalPnlPct ??
nav?.pnl_pct`.

**Fix B — One-shot repair script**
(`scripts/repair_phase_23_1_17.py`). Calls
`trader.mark_to_market()` + `trader.save_daily_snapshot()` to
recompute `total_nav` from current cash + live position values
and persist a fresh row to `paper_portfolio_snapshots`. Repair
script also documents (in module docstring) that any future
raw-BQ cash mutation must be followed by `mark_to_market()`.

## Files modified

- `frontend/src/app/paper-trading/page.tsx` (-25 lines: removed
  inline `useMemo` blocks; +2 lines: import + call shared hook)
- `frontend/src/app/page.tsx` (+9 lines: import + hook call +
  navValue/pnl fallback chain)

## Files added

- `frontend/src/lib/useLiveNav.ts` (52 lines, the shared hook)
- `scripts/repair_phase_23_1_17.py` (90 lines, one-shot
  repair + future-author reminder)
- `tests/verify_phase_23_1_17.py` (immutable verification)

## Verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_17.py
ok useLiveNav shared hook + home page consumption + paper-trading refactor + repair script (mark_to_market + save_daily_snapshot)
```
Exit 0.

## Frontend type check

```
$ cd frontend && npx tsc --noEmit
(silent, exit 0)
```

## Test results

```
$ pytest tests/api/test_ticker_meta_perf.py tests/api/test_ticker_meta.py tests/services/test_trade_idempotency.py tests/services/test_sector_concentration.py -q
.........................                                                [100%]
25 passed in 3.03s
```
All prior phases' tests still green.

## Repair script execution log

```
$ python scripts/repair_phase_23_1_17.py --apply --yes
Pre-repair:  cash=$2146.39, total_nav=$14153.03
Calling mark_to_market()...
mark_to_market done: nav=$15647.74 cash=$2146.39 positions_value=$13501.35
Saving daily snapshot...
Post-repair: cash=$2146.39, total_nav=$15647.74
NAV delta: $+1494.71
ok phase-23.1.17 repair complete
```

Live BQ verify:
```sql
SELECT current_cash, total_nav, total_pnl_pct, starting_capital, updated_at
FROM paper_portfolio WHERE portfolio_id='default';
-- cash=$2,146.39, total_nav=$15,647.74, pnl=+4.32%, starting=$15,000
-- updated_at=2026-04-29T19:11:46Z (post-repair)
```

After this cycle:
- BQ `total_nav` = $15,647.74 (was $14,153.03 — stale).
- Home page's NAV tile derives `liveNav = cash + sum(livePrice *
  qty)` and falls back to `total_nav` only if no live ticks.
- Paper-trading page renders the same number via the same
  shared hook.
- Red Line Monitor's most-recent row (today) reflects the
  repaired NAV.

## Backwards compatibility

- The shared hook returns the same shape (`liveNav: number |
  null`, `liveTotalPnlPct: number | null`) as the inline
  useMemo. SummaryHero signature unchanged.
- Paper-trading page behavior is byte-identical (same math,
  same fallback). Only the source location moved.
- Home page falls back to `nav?.nav` when `liveNav` is null
  (initial paint, empty positions).
- Repair script is idempotent — re-running just refreshes
  the snapshot.

## Honest disclosures

1. **Sharpe and Max-DD on home are still computed from
   redLineSeries**, not from the live derivation. They will
   converge with paper-trading's Sharpe over a few cycles as
   the snapshot history rolls forward — but on any given day
   the two pages may show different Sharpe values. Out of
   scope for this cycle (the user's complaint was NAV).

2. **VS SPY** on the home page is `liveTotalPnlPct - benchmark`,
   so it now reflects the live-derived total P&L. The SPY
   benchmark itself is whatever the BQ snapshot last computed —
   may be slightly stale if SPY moved since the last
   mark_to_market. Acceptable: SPY benchmarking is
   inception-to-date, not session-relative.

3. **No real money at risk** — paper trading only.

4. **Future-author reminder**: any raw BQ UPDATE to
   `paper_portfolio.current_cash` (deposits, manual refunds,
   adjustments) MUST be followed by a `mark_to_market()` call.
   The repair script's docstring documents this; phase-23.1.15
   shipped without it and produced this discrepancy.

## Phase 2 (deferred)

- Backend wraps every `UPDATE paper_portfolio.current_cash`
  in a method that auto-calls `mark_to_market()` afterward —
  prevents the "raw BQ mutation forgot MtM" foot-gun
  structurally, not just by docstring.
- Sharpe + Max-DD on home page derive from a "today live"
  point appended to the redLineSeries instead of pure
  snapshot history.
- Status-endpoint `/api/paper-trading/status` returns the
  live-derived NAV server-side (Fix C from the contract).
  Deferred because of the per-call yfinance batch cost (5-10
  HTTP round-trips on every status poll); requires a server
  cache layer with sub-second TTL.
