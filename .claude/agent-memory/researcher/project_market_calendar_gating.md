---
name: market-calendar-gating
description: phase-50.4 market-calendar gating -- the LATENT is_trading_day bug (cal.days gone in xcals 4.13.2), the tz/date convention, and the entry-gate/exit-open model
metadata:
  type: project
---

phase-50.4 gates the live loop's trades on `is_trading_day(market)` per exchange (XNYS/XETR/XKRX). Researched 2026-05-30.

**THE LATENT BUG (highest-value finding):** `backend/backtest/markets.py:137-152` `is_trading_day` calls `date in cal.days`, but **`cal.days` does NOT exist in exchange_calendars 4.13.2** (removed/renamed in the 4.0 rename wave). It raises AttributeError, caught by the bare `except` at :150 -> **always returns True (never gates)**. Zero live callers today so the bug is latent. 50.4 MUST rewrite to `cal.is_session(pd.Timestamp(naive_date))` -- the canonical 4.x method.

**Why:** verified by direct introspection of the installed dist; `is_session` is the only correct query and it REJECTS tz-aware Timestamps with ValueError (accepts naive Timestamp/str/date). Session labels are tz-naive midnight.
**How to apply:** never trust the pre-existing markets.is_trading_day; any calendar gate must use `cal.is_session(...)` with a tz-NAIVE date. Strip datetimes to `.date()` first.

**TZ/DATE CONVENTION:** the loop fires at `paper_trading_hour` ET (default 10, `settings.py:265`) via APScheduler cron `day_of_week="mon-fri"` (`api/paper_trading.py:1307`). "Is market X open" MUST use market X's LOCAL date: `datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_CONFIG[m]["timezone"])).date()`. A UTC-date or US-date approach MISSES Korean holidays by the ET->KST skew (proven: 10:00 ET day-before-Seollal -> Seoul already on Seollal -> KR correctly closed only under market-local-date).

**SAFETY INVARIANT (US byte-identical):** the live loop has NO in-cycle market-open gate today -- only the cron's mon-fri. It TRADES on US holidays that fall on weekdays (yfinance returns stale close). So adding a US gate WOULD change behaviour. Keep US ungated: wrap the gate in `if _intl_markets:` so `paper_markets==["US"]` never touches calendar code -> provably byte-identical.

**GATING MODEL:** entry-gated, exit-open, per-market. Filter the universe (`autonomous_loop.py:343`) + buy guard (`:1003`) by market-open; ALWAYS allow exits (stop-loss Step 5.6 `:873`, SELL Step 7 `:981`, kill-switch Step 5.5) regardless of market status -- gating an exit strands a breached stop (losses run unbounded). Paper exit fills at last close; log the stale fill. NO whole-cycle skip (would strand open markets incl US).

**DEP STATUS:** `exchange_calendars==4.13.2` installed + ALREADY imported in live path (`markets.py:12`, `autoresearch/monthly_champion_challenger.py:327`) -> NO new dep. NOT in any requirements.txt; recommend pinning `>=4.13,<5` in backend/requirements.txt (1-line declaration, flag for owner but not a blocker; fail-open try/except ImportError means missing install degrades to "always open").

**CROSS-VAL:** library's 2026 XETR/XKRX/XNYS sessions match published Deutsche-Borse/calendarlabs calendars EXACTLY (incl. lunar Seollal Feb16-18, Chuseok Sep24-25, substitute Natl Foundation Oct5). Gate value: KR=15 weekday closures vs US=10, only 4 overlap -> ~11 days/yr the loop would trade a closed KR market on stale data.

Related: [[multimarket_universe_wiring]] (50.3: order.market field, market_for_symbol, INTL_UNIVERSE), [[multimarket_scaffolding_disconnected]] (50-series origin).
