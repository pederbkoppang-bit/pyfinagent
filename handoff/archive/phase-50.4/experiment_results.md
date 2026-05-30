# Experiment results -- phase-50.4: Market-calendar gating

**Date:** 2026-05-30 | **Result: built + verified (byte-identical for paper_markets=['US'])** | $0 LLM | no pip (exchange_calendars already installed).

## What was built
The live loop now gates ENTRY (screening/buying) per market on whether that market is a trading day today (market-local date), so it won't trade a closed EU/KR exchange on stale data. US is never gated (byte-identical). A latent always-True bug in `is_trading_day` was fixed.

## Files changed/added
1. **backend/backtest/markets.py:137** -- rewrote `is_trading_day(date, market)`: `cal.is_session(pd.Timestamp(date).normalize())` (was the broken `date in cal.days` -> AttributeError swallowed -> always True). Fail-open True if calendar unavailable.
2. **backend/services/autonomous_loop.py** (inside the 50.3 `if _intl_markets:` block) -- after building the multi-market universe, drop tickers whose market is closed today (market-LOCAL date via ZoneInfo); US tickers never gated; fail-open on calendar error; log the drop count.
3. **backend/tests/test_phase_50_4_calendar.py** (NEW) -- 7 tests (US weekday/weekend, EU Labour Day closed / US open, KR Seollal + Chuseok closed / US open, normal weekday all-open, unknown-market fail-open, regression guard that it's not always-True, market derivation).

## Verification
- `pytest backend/tests/test_phase_50_4_calendar.py` -> **7 passed**. autonomous_loop imports clean.
- Live across markets (verified vs published Xetra/KRX 2026 calendars): EU 2026-05-01 (Labour Day) closed while US open; KR 2026-02-17 (Seollal) + 2026-09-25 (Chuseok) closed while US open; normal weekday all open; unknown market fail-open True.
- masterplan command: `is_trading_day('2026-01-01','EU') is False` + `('2026-06-15','US') is True` -> "calendar gate OK".

## Success criteria mapping (all 3 met) -- see live_check_50.4.md
1. gates per market, holiday skips that market independently -- YES. 2. dep verify-installed, US path unbroken -- YES (no new dep). 3. live evidence of one market closed / another open -- YES.

## Scope / honesty notes
- **Byte-identical:** the gate is inside `if _intl_markets:` -> default ['US'] never runs it; US tickers are never gated even in a multi-market universe (the live loop never gated US before -- gating it would CHANGE behaviour). `is_trading_day` IS correct for US (a caller can gate US), but the live loop deliberately doesn't (preserves the +20% engine).
- **Exits never gated** -- only the ENTRY universe is filtered; execute_sell / stop-loss paths untouched (a breached stop must always fire).
- Fixed a latent always-True bug (cal.days removed in exchange_calendars 4.0; was dead code with zero live callers).
- exchange_calendars==4.13.2 already installed + imported; recommend an explicit requirements pin (>=4.13,<5) as an owner-flagged follow-up (not a blocker; fail-open).
- $0 LLM; no pip; no spend; no DROP/DELETE.
