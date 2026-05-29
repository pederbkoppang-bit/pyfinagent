---
name: multimarket-scaffolding-disconnected
description: Multi-market (EU/KR) scaffolding already exists in backtest path but the LIVE loop is hard US-only; phase-50 expansion plan researched 2026-05-29
metadata:
  type: project
---

Multi-market expansion (EU + South Korea) was researched 2026-05-29 (operator ask: agents only buy US Tech; add EU/KR, multi-currency, multi-exchange). Brief: `handoff/current/research_brief_multimarket.md`.

**Key structural fact (non-obvious, verify before acting):** the system is NOT greenfield for multi-market, but the scaffolding is DISCONNECTED from the live money loop.
- `backend/backtest/markets.py` already defines 5 markets (US/NO/CA/EU/KR) with exchange code + currency + timezone + `exchange_calendars` helper.
- `backend/backtest/candidate_selector.py:98` `get_universe_tickers(market=...)` accepts a market arg but returns `[]` for non-US.
- phase-5.1 (broker ABC `backend/markets/broker_base.py`), 5.4 (risk engine), 5.6 (options) are DONE; deferred phase-5 has pending 5.2/5.9/5.13.
- **THE GAP:** the live `backend/services/autonomous_loop.py:310-329` uses `screen_universe`/`get_sp500_tickers`/`get_russell1000_tickers` from `backend/tools/screener.py` — hard US-only Wikipedia scrape, never touches markets.py.

**Why:** operator wants broader sectors + EU/KR exposure. Tech-concentration is momentum-driven (composite score = momentum 0.4 weighted), NOT a structural sector exclusion — sector caps + a `sector_neutral_momentum_enabled` flag (OFF) already exist.

**How to apply:** recommended NEW phase-50 (not the broad asset-class phase-5) to complete the live-loop gap. Order: FX data layer (free, yfinance EURUSD=X/KRW=X) -> multi-currency accounting in `paper_trader.py` (currently currency-blind: `pnl = market_value - cost_basis` at line 446, no FX) -> universe+suffix mapper (KR:005930->005930.KS) -> calendar gating -> backtest -> UI. Biggest risk = yfinance intl data quality (XETRA deviations up to 11%, 10-24 identical-OHLC days/yr); EODHD $19.99/mo is the paid v2 fallback (owner approval). Recommended first market = EU/Germany via .DE, not Korea. Currency decomposition model: arXiv 1611.01463, `r = local + FX`.
