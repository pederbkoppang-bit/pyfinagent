---
step: phase-23.1.17
title: Home page hero metrics SSOT — share liveNav hook with paper-trading + repair stale total_nav snapshot
cycle_date: 2026-04-29
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_17.py'
research_brief: handoff/current/phase-23.1.17-external-research.md (also see phase-23.1.17-internal-codebase-audit.md)
---

# Contract — phase-23.1.17

## Hypothesis

User reported the home page (MAS Operator Cockpit) shows different
NAV/P&L/Sharpe/DD than the paper-trading page. Two compounding causes
identified by the researcher:

**Cause 1 — Stale `paper_portfolio.total_nav`.** Yesterday's
phase-23.1.15 cleanup script did a raw BQ UPDATE to `current_cash`
(+$1,451.40) but did NOT call `mark_to_market()`. The
`mark_to_market` function in `paper_trader.py` is the only code
path that recomputes `total_nav = current_cash + total_positions_value`
and writes it back. Without that call, `total_nav` is now
`current_cash + OLD positions_value` — off by both the refund and
intervening price drift.

**Cause 2 — Home page reads the stale column directly.**
`frontend/src/app/page.tsx:142` does `navValue = nav?.nav` (raw BQ
snapshot). Paper-trading page (post phase-23.1.14) computes
`liveNav = cash + sum(livePrice * qty)` as a `useMemo`. Home page
was never updated to use this derivation, so it shows the stale
total_nav.

If we (A) extract the live-derive logic into a shared hook
`useLiveNav`, (E) use it on the home page, and (B) run
`mark_to_market()` once to repair the stale `total_nav` column,
then the two pages match and the BQ table reflects reality.

## Research-gate summary

- External brief: `handoff/current/phase-23.1.17-external-research.md`
  — 6 sources read in full (Limina batch-vs-event-driven, Limina
  IBOR guide, TanStack Query overview, SWR mutation docs, Limina
  PMS guide, Bennett NAV calculation). 16 URLs collected.
  Recency scan 2024-2026. `gate_passed: true`.
- Internal audit: `handoff/current/phase-23.1.17-internal-codebase-audit.md`
  — 6 files inspected with file:line anchors and concrete patches.

Key findings:
- Limina IBOR: "live-extract pattern (recompute on demand from
  transactions) outperforms stored snapshot for operator dashboard
  consistency. Cash mutations that bypass revaluation create
  stale state."
- TanStack Query / SWR: shared query key / shared hook = SSOT
  across pages. pyfinagent uses raw `apiFetch`, so a shared
  custom hook is the correct substitute.
- Researcher recommends **A + E + B** with MtM-first sequencing.
  Defers Fix C (backend status endpoint always returns
  live-derived NAV) — too expensive (5-10 yfinance calls per
  status poll on every page load).

## Plan steps

1. **Fix A — Extract `useLiveNav` shared hook**:
   `frontend/src/lib/useLiveNav.ts`. Inputs: `status` (from
   `getPaperTradingStatus`), `positions` (from `getPaperPositions`),
   `livePrices` (from `useLivePrices`). Returns
   `{ liveNav, liveTotalPnlPct }`. Same math currently inline in
   `paper-trading/page.tsx`. Falls back to BQ snapshot when
   `livePrices` is empty.

2. **Refactor paper-trading page** to use the shared hook —
   delete the inline `useMemo` blocks, import the hook, pass
   results unchanged to `SummaryHero`. Behavior unchanged.

3. **Fix E — Home page uses the shared hook**. Add the same
   `useLivePrices(positionTickers, positions.length > 0)` call.
   Replace `navValue = nav?.nav` with the hook's `liveNav`.
   The "P&L (today)" tile keeps its redLine source (it's a
   *daily delta*, not a total — orthogonal). Add a fallback:
   when `liveNav` is null, use `nav?.nav`.

4. **Fix B — Repair stale total_nav**. One-time script
   `scripts/repair_phase_23_1_17.py` that calls
   `PaperTrader(settings, bq).mark_to_market()` once + saves a
   fresh `paper_portfolio_snapshots` row. Idempotent; logs
   before/after total_nav. After this, the redLine series'
   most-recent point reflects post-refund + current MtM.

5. **Fix D (prophylactic)** — add a docstring note in
   `scripts/cleanup_phase_23_1_15.py` and the new repair script
   reminding future authors that ANY cash mutation must be
   followed by `mark_to_market()` (or a comment in
   `bigquery_client.update_paper_portfolio_cash` if such a method
   exists).

6. **Tests** (`tests/lib/test_use_live_nav.test.ts` if Jest set
   up; otherwise a Python sanity test that asserts the hook's
   math matches `cash + sum(livePrice * qty)`).
   Plus `tests/services/test_repair_total_nav.py` mocking
   `mark_to_market` and asserting it's called.

7. **Immutable verification** (`tests/verify_phase_23_1_17.py`):
   asserts `useLiveNav.ts` exists, home page imports it, paper-
   trading page imports it (refactored), repair script exists
   and calls mark_to_market.

## Immutable verification command

```bash
source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_17.py
```

Must exit 0.

## Acceptance criteria

- `useLiveNav.ts` exists in `frontend/src/lib/` and is a single
  file with a clear default-export hook.
- Both `paper-trading/page.tsx` and `page.tsx` import it; no
  duplicated `liveNav = useMemo(...)` logic.
- `cd frontend && npx tsc --noEmit` clean.
- `pytest -q` passes including any new tests.
- `python tests/verify_phase_23_1_17.py` exit 0.
- After running the repair script, BQ
  `paper_portfolio.total_nav` matches the live-derived NAV
  within fee tolerance.
- Home page hero NAV matches paper-trading page NAV (visual
  parity).

## Backwards compatibility

- The shared hook returns the same shape (`liveNav: number |
  null`, `liveTotalPnlPct: number | null`) as the inline
  useMemo. SummaryHero signature unchanged.
- Repair script is one-shot data fix; idempotent re-runs are
  harmless (mark_to_market is itself idempotent).
- Home page falls back to `status.portfolio.nav` when
  `liveNav` is null (consistent with paper-trading's existing
  behavior).

## References

- `handoff/current/phase-23.1.17-external-research.md`
- `handoff/current/phase-23.1.17-internal-codebase-audit.md`
- `backend/services/paper_trader.py:384-395` (mark_to_market)
- `frontend/src/app/page.tsx:141-153` (home hero math)
- `frontend/src/app/paper-trading/page.tsx:430-460` (existing
  inline useLiveNav math)
