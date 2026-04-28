---
step: phase-23.1.13
title: Sector concentration enforcement (v1) — populate sector in screener + max-per-sector cap + Risk Monitor fix + portfolio sector breakdown
cycle_date: 2026-04-28
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_13.py'
research_brief: handoff/current/phase-23.1.13-external-research.md (also see phase-23.1.13-internal-codebase-audit.md)
---

# Contract — phase-23.1.13

## Hypothesis

Four coordinated changes close the 11/11-Technology concentration shown in the operator screenshot:

1. Populate `sector` on every screener candidate (currently `None` until after LLM analysis — regime tilt + sector-calendar overlays are no-ops because of this).
2. Hard cap per sector in `portfolio_manager.decide_trades`: skip a BUY when the proposed sector already has `paper_max_per_sector` open positions (default `2` per external research consensus = ≥5 sectors for a 10-position portfolio).
3. Frontend Risk Monitor computes ACTUAL sector concentration (not single-position weight). When >50% of positions are in one sector → "HIGH (N/M Sector)" amber.
4. Backend `/portfolio` endpoint returns a `sector_breakdown` map.

## Plan

### Backend
1. `backend/tools/screener.py::screen_universe` — add `sector_lookup: dict[str, str] | None = None` kwarg; attach `sector` to each result dict at line ~132.
2. `backend/services/autonomous_loop.py` — before `screen_universe`, build `sector_lookup` for top-50 momentum tickers via existing `_fetch_ticker_meta` (BQ-first / yfinance-fallback, already shipped in phase-23.1.10).
3. `backend/config/settings.py` — NEW `paper_max_per_sector: int = Field(2, ge=0, le=20)`.
4. `backend/services/portfolio_manager.py::decide_trades` — build `sector_counts` from `current_positions` (sells reduce); inside buy loop skip when `sector_counts[sector] >= settings.paper_max_per_sector`. Increment after appending. Log skipped candidates.
5. `backend/services/paper_trader.py::execute_buy` — accept `sector: str = ""` kwarg, stash on in-memory position dict.
6. `backend/api/paper_trading.py::get_portfolio` — extend response with `sector_breakdown`.
7. `backend/api/settings_api.py` — expose `paper_max_per_sector` (FullSettings + SettingsUpdate + _FIELD_TO_ENV + _settings_to_full).

### Frontend
8. `frontend/src/lib/types.ts` — `PaperPortfolio.sector_breakdown?` + `FullSettings.paper_max_per_sector?`.
9. `frontend/src/app/paper-trading/page.tsx::RiskMonitor` — replace single-position `concentrationHigh` with sector concentration computation using `tickerMeta`. Render "HIGH (N/M Sector)" when `maxSectorCount / positions.length > 0.5 && positions.length >= 3`.
10. `frontend/src/app/paper-trading/page.tsx::Manage tab` — add `paper_max_per_sector` to Trading Settings via `PaperSettingNum`.

### Tests
11. `tests/services/test_sector_concentration.py` (NEW) — 5 tests on decide_trades sector cap.
12. `tests/services/test_screener_sector_propagation.py` (NEW) — 3 tests on sector_lookup wiring.
13. `tests/verify_phase_23_1_13.py` (NEW) — immutable verification script.

## Out of scope (Phase-2)

- New `sector` column on `paper_positions` BQ schema (--apply migration)
- HRP / risk-parity post-selection optimizer (research ROI rank #7)
- Sector-neutral re-ranking (rank within sector before merge — research ROI rank #5)
- Correlation-cluster deduplication
- Forced rebalance when EXISTING positions exceed cap (cap only blocks NEW buys for v1)
- BQ `sector_concentration_history` table
- Slack alerts on sector cap breach
- Min-sectors enforcement (≥4 distinct sectors)
- 25% NAV-per-sector hard cap (position-count cap of 2 with 10% per-position cap implies ~20% max sector weight; close to canonical 25% for v1)

## Verification

The verification script asserts source-level correctness (settings field, function signature, logic presence). End-to-end behavior is testable via the unit tests + tomorrow's natural cycle.

## What this fixes for the operator

| Before | After |
|---|---|
| 11/11 positions all Technology | Buy loop stops at 2 Tech; remaining slots filled from other sectors |
| "Concentration: OK" misleadingly green | "Concentration: HIGH (N/M Sector)" amber when >50% concentrated |
| No sector breakdown in API | `sector_breakdown` field on `/portfolio` response |
| Operator can't tune cap | Manage tab exposes `paper_max_per_sector` |
| Regime tilt + sector calendars overlays = no-ops | Operate on real sector data through `screen_universe` |

## References

- `handoff/current/phase-23.1.13-external-research.md` (566 lines, 13 sources read in full, gate_passed: true)
- `handoff/current/phase-23.1.13-internal-codebase-audit.md` (351 lines, 13 internal files inspected, 5 critical gaps documented, gate_passed: true)
- SEC 1940 Act 25% concentration threshold (canonical regulatory floor)
- arxiv 2507.20957 — documented LLM tech bias
- DeMiguel et al. 2009 — 1/N beats MVO for small N
