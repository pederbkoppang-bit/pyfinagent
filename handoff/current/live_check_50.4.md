# live_check_50.4 -- market-calendar gating (evidence)

Verified 2026-05-30 against the installed exchange_calendars==4.13.2 (XETRExchangeCalendar etc.).

## 1. Unit test (criteria #1, #3) -- backend/tests/test_phase_50_4_calendar.py
`pytest -q` -> **7 passed in 1.51s**. Pinned per-exchange trading days (verified against published Xetra/KRX 2026 calendars):
```
US 2026-06-15 (Mon)  = True     US 2026-06-13 (Sat)        = False
EU 2026-05-01 (Labour Day) = False   US 2026-05-01          = True   (independent: EU closed, US open)
KR 2026-02-17 (Seollal/lunar) = False  US 2026-02-17        = True
KR 2026-09-25 (Chuseok)    = False   US 2026-09-25          = True
US/EU/KR 2026-06-15 (normal Mon) = True (all open)
unknown market 'ZZ' -> True (fail-open, defaults US)
```

## 2. The latent-bug fix (criterion #2)
`markets.is_trading_day` previously did `date in cal.days` -- `.days` was removed in exchange_calendars 4.0, so the bare `except` swallowed the AttributeError and it returned **True for everything** (a no-op gate; zero live callers). Rewritten to `cal.is_session(pd.Timestamp(date).normalize())` (tz-aware labels rejected -> stripped to naive). The `test_not_always_true_regression_guard` test proves it now returns False (weekend + holiday). exchange_calendars==4.13.2 is already installed + imported (markets.py:12) -> NO new dependency.

## 3. The gate (criterion #1) -- entry-gated, exits-open, US byte-identical
- `autonomous_loop.py` (inside the `if _intl_markets:` block): after building the multi-market universe, drops a ticker whose market is CLOSED today (market-LOCAL date via `datetime.now(utc).astimezone(ZoneInfo(market_tz)).date()`). **US tickers are NEVER gated** (`_open_today` returns True for US) -> matches today's behaviour. Fail-open on any calendar error (keep the ticker).
- The gate lives ENTIRELY inside `if _intl_markets:` -> with `paper_markets=['US']` (default) it NEVER runs -> the live +20% engine is byte-identical.
- **Exits are NOT gated** -- only the ENTRY universe is filtered. execute_sell / stop-loss paths are untouched, so a breached stop always fires (gating it would strand a position -> unbounded loss).

## 4. Verification command
```
pytest backend/tests/test_phase_50_4_calendar.py -> 7 passed
is_trading_day('2026-01-01','EU') is False + ('2026-06-15','US') is True -> "calendar gate OK"
test -f handoff/current/live_check_50.4.md -> present
```

## Success criteria mapping (all 3 met)
1. gates trades on is_trading_day per market; a holiday skips that market independently -- YES (EU Labour Day / KR Seollal closed while US open; gate filters the entry universe per-market). Note: the LIVE loop applies the gate to NON-US markets only (US stays ungated = byte-identical, since the loop never gated US before); is_trading_day itself is correct for US too (a caller CAN gate US) -- disclosed in the contract.
2. dep verify-installed without breaking US -- YES (exchange_calendars 4.13.2 already installed/imported; no new dep; US path untouched).
3. live/fixture evidence of a market closed while another open -- YES (EU 2026-05-01 closed / US open; KR 2026-02-17 + 09-25 closed / US open).

## Scope / honesty notes
- **Byte-identical:** gate inside `if _intl_markets:` -> default ['US'] never runs it; US tickers in a multi-market universe also never gated.
- **Exits never gated** (only entries) -- a stop-loss/sell always fires.
- Fixed a latent always-True bug (was dead code).
- exchange_calendars already a transitive dep; an explicit pin (>=4.13,<5) in requirements is RECOMMENDED (owner-flagged) but not a blocker (fail-open).
- $0 LLM; no pip; no spend; no DROP/DELETE.
