---
name: currency-display-61-3
description: phase-61.3 pre-pay brief facts — add-on avg bug = USD-cost/qty stored as LOCAL (1350x KR, +8% EU silent); resolveCurrency explicit-first defeated by base_currency="USD" hardcode; no per-position mark timestamp exists; 07:00 UTC mark-only job design; PAPER_TRADING_HOUR .env unverifiable from sandbox
metadata:
  type: project
---

Pre-pay research brief for masterplan 61.3 written 2026-07-08 at
`handoff/current/research_brief_61.3.md` (gate_passed: true; 7 full reads).
Non-obvious facts a future contract must revalidate:

- **Add-on bug shape**: paper_trader.py:291 `new_avg = USD_cost/new_qty` stored
  as LOCAL avg_entry_price (:297). KR corruption = exactly the fx rate (1350x,
  screams); **EU = +8.0%, silently plausible** — the dangerous case. Post-corruption
  `qty*avg == USD cost`, so cost_basis fallbacks stay accidentally correct; damage
  is confined to LOCAL consumers (realized P&L :393-396/:443, breakeven :1137
  which OVERWRITES a valid stop with an untriggerable USD-scale one, trail
  :1105-1125, backfill :756).
- **Display half**: resolveCurrency (format.ts:161) is explicit-first and every
  pos_row hardcodes base_currency="USD" (:313/:334/:481) -> LOCAL price columns
  always render USD. Fix = market-first `resolveLocalCurrency`, NOT backend column.
- **No mark timestamp exists anywhere per-position** — mark_to_market updates
  dict (:539-546) has no ts; last_analysis_date only stamped on BUY. Criterion 4
  requires a new `marked_at` column (migration + `_POSITION_RT_FIELDS`-style prune).
- **Locale**: NumberFlow USD branches pass `locales={undefined}`
  (positions-columns.tsx:74, cockpit-helpers.tsx:96) = runtime default (nb-NO
  browser vs en-US Node = hydration hazard, MDN/Next.js confirmed). formatCurrency
  already pins en-US — fix is adoption, not new util. 41 repo-wide `$${...}`
  template sites are genuinely-USD dashboards; scope only paper-trading surfaces.
- **Cycle-time discrepancy**: goal doc says "18:00 UTC cycle" but
  settings.py:341 default is 10:00 ET; researcher sandbox DENIED backend/.env —
  Main must read PAPER_TRADING_HOUR at contract time (see
  [[backend-restart-safety]] for the same denial precedent).
- **07:00 UTC KR mark job**: mark-only (mark_to_market does marks+MFE+stop-advance,
  :500-583) gated on is_trading_day(KR-local-date,"KR"); stop EXECUTION lives
  inline in autonomous_loop.py:1052-1105 and extracting it breaches the goal's
  stop-engine-untouchable rule -> recommend defer-with-rationale.
- **live_check risk**: criterion demands a live KR position screenshot; portfolio
  100% cash since 2026-07-03 — may need loop re-entry or seeded position.
- Verification `-k 'addon or avg_entry or currency or 61_3'` -> test files MUST be
  `test_phase_61_3_*.py` with `addon`/`currency` in function names ([[fable5-adoption]]
  -k false-green lesson).
