# Research Brief: phase-50.4 -- market-calendar gating

**Tier:** moderate
**Date:** 2026-05-30
**Step:** Gate trades on `is_trading_day(market)` per exchange (XETR/Germany, XKRX/Korea, XNYS/US) so the live loop doesn't try to trade a closed market (holiday/weekend), each with its own holidays + timezone.

## Safety invariant (non-negotiable)
US-only (`paper_markets=['US']`, default) MUST stay byte-identical. **VERDICT (proven below): keep US byte-identical by making the gate a no-op for the US-only default.** The cleanest construction is (b)+(a) combined: when `paper_markets == ["US"]` the gate is bypassed entirely (early `return False`/skip is never reached because the US weekday cron already governs the cadence); when multi-market is active, the gate filters/defers per market but the US slice's open/closed decision is identical to today's "the cron fired, so trade" behaviour. See **§Gate plug-in point** for the byte-identity proof.

---

## TL;DR (the load-bearing findings)

1. **The live loop does NOT gate on market-open today.** `run_daily_cycle` (`backend/services/autonomous_loop.py:124`) runs unconditionally whenever APScheduler fires. The ONLY calendar awareness today is the **cron's `day_of_week="mon-fri"`** (`backend/api/paper_trading.py:1307`) -- weekends are excluded by the *scheduler*, not by any in-cycle check. **US holidays are NOT excluded** -- the cron fires on e.g. 2026-07-03 / Thanksgiving if it's a weekday, and the cycle trades. (In practice yfinance returns the prior close on a US holiday, so it "works" but trades on stale data.) => Adding a US `is_trading_day` gate WOULD change behaviour (it would start skipping US holidays the loop currently runs through). **Therefore the US path must NOT be gated**, or it changes the +20% engine.

2. **`markets.is_trading_day` ALREADY EXISTS (`markets.py:137`) -- but it is SILENTLY BROKEN.** It calls `date in cal.days`, and **`cal.days` does not exist in exchange_calendars 4.13.2** (it was removed/renamed; the attribute raises `AttributeError`). The `except` at `markets.py:150` swallows it and returns `True`. **So `is_trading_day` always returns `True` today -- it never gates anything.** 50.4 MUST rewrite it to use `cal.is_session(...)`. This is the single most important code finding. (Verified by direct introspection of the installed 4.13.2 dist.)

3. **The correct API is `calendar.is_session(date)`** (returns `bool`). It is the canonical 4.x method (the 4.0 rename made `is_session` primary; there is NO `is_trading_day` in the library). It **REJECTS a tz-aware Timestamp** with `ValueError` and accepts a tz-naive `pd.Timestamp` / python `date` / ISO `str`. Session labels are tz-naive (midnight, no timezone).

4. **Date convention MUST be the market's LOCAL calendar date**, computed as `datetime.now(timezone.utc).astimezone(ZoneInfo(market_tz)).date()`. Proven by test: when the loop fires the day before Seollal at 10:00 ET, Seoul has already rolled to Seollal -- the market-local-date approach correctly returns `is_session=False` for KR while a UTC-date or US-date approach would wrongly say "open."

5. **Recommended gating model: per-position/per-candidate by market, NOT whole-cycle skip.** A multi-market universe can have US open while KR is closed (proven: 11 weekdays/yr where they diverge). Gate ENTRIES (filter the candidate set to markets open today) and gate ENTRIES only -- **always allow EXITS** (stop-loss/risk sells) even on a closed market for safety; document that the fill price is stale (paper, no real broker queue). US-only default => no market is ever filtered => byte-identical.

6. **`exchange_calendars==4.13.2` is installed but NOT declared in any `requirements.txt`.** However it is ALREADY imported in the live path (`markets.py:12`, `autoresearch/monthly_champion_challenger.py:327`), so 50.4 introduces **no new runtime dependency**. Recommend ADDING the pin to `backend/requirements.txt` for reproducibility (a 1-line declaration of an already-relied-upon import) -- flag for owner sign-off but it is not a new dep.

---

## Internal code inventory (Q1-Q6 with file:line)

### Q1 -- Does the live loop check market-open TODAY? Where does a per-market gate plug in?

**NO current is-open gate inside the cycle.** Trace:
- **Scheduler** (`backend/api/paper_trading.py:1299-1322`): `_add_scheduler_job` registers `_scheduled_run` as an APScheduler **cron** job: `hour=settings.paper_trading_hour` (default **10**, `settings.py:265`), `minute=0`, **`day_of_week="mon-fri"`**, `timezone=America/New_York`, `misfire_grace_time=3600`, `coalesce=True`. So the ONLY calendar filter is "Mon-Fri ET" -- a scheduler concern, **not** an in-cycle market-open check.
- **`_scheduled_run`** (`:1325`) -> `run_daily_cycle(settings)` (`backend/services/autonomous_loop.py:124`). The cycle body (under `async with asyncio.timeout(...)`, `:226`) runs **Step 1 Screen -> ... -> Step 7 Execute -> Step 8 Snapshot** with **no `is_trading_day` / `is_session` / market-open check anywhere** (grep confirmed: zero hits in the live-loop path).
- **Consequence for the invariant:** the loop trades on US market **holidays** that fall on weekdays today (e.g. Thanksgiving, Jul 4 obs). yfinance returns the last available close, so it executes against stale prices. **If 50.4 adds a US `is_session` gate, the loop would START skipping those US-holiday weekdays -- a behaviour change.** Hence US must stay ungated (gate is a no-op for `["US"]`).

**EXACT gate plug-in points (two, both byte-identical for US-only):**
- **(A) Entry filter -- Step 1, right after the universe is assembled (`autonomous_loop.py:343`, before `screen_universe`).** Filter `universe` to symbols whose market is open today. For `paper_markets==["US"]`, `universe` is all-US (or `None`), and the US filter is a no-op (US is never dropped -- see byte-identity proof). This is where you keep a closed-KR ticker from being screened/bought today.
- **(B) Per-order market guard -- Step 7 buys (`autonomous_loop.py:1003-1014`, before `execute_buy`).** Defense-in-depth: skip a BUY whose `order.market` is closed today. `order.market` already exists (`portfolio_manager.py:33`, set via `markets.market_for_symbol(cand["ticker"])` at `:352`/`:566`) and is already passed at `:1027` (`market=getattr(order, "market", "US")`). A US order is never skipped (US ungated). Keeping this guard at the buy callsite makes the gate robust even if the universe filter is bypassed.

**Recommendation: implement (A) as the primary gate (filter candidates by open-market) and add (B) as a cheap buy-side assertion.** Do NOT add a whole-cycle skip -- see Q3.

### Q2 -- markets.get_trading_calendar return type + is_trading_day API (PARTIALLY PRE-BUILT, BROKEN)

- `markets.get_trading_calendar(market)` (`markets.py:113-134`): returns `xcals.get_calendar(exchange)` where `exchange = MARKET_CONFIG[market]["exchange"]` (US->XNYS, EU->XETR, KR->XKRX; `:21-52`). Returns `None` if `exchange_calendars` is unimportable (fail-open). **Return type: an `ExchangeCalendar` subclass instance** (e.g. `XNYSExchangeCalendar`).
- `markets.is_trading_day(date, market="US")` (`markets.py:137-152`): **EXISTS but BROKEN.** Body:
  ```python
  cal = get_trading_calendar(market)
  if cal is None: return True
  if isinstance(date, str): date = datetime.fromisoformat(date).date()
  try: return date in cal.days          # <-- cal.days does NOT exist in 4.13.2
  except Exception: return True          # <-- always taken -> always True
  ```
  **`cal.days` raises `AttributeError` in 4.13.2** (confirmed by introspection: `'XNYSExchangeCalendar' object has no attribute 'days'`). The bare `except` swallows it -> **`is_trading_day` unconditionally returns `True`**. It has **zero callers in the live path** today (grep), so the bug is latent -- 50.4 is the first real consumer and MUST fix it.
- **Correct 4.x API (verified against the installed dist):**
  - `cal.is_session(date) -> bool` -- the canonical "is this date a trading session" query. Docstring: *"Query if a date is a valid session. ... Return: bool -- True if `date` is a session, False otherwise."*
  - **Argument:** tz-NAIVE only. `cal.is_session(pd.Timestamp("2025-07-07"))` -> OK; `cal.is_session(date(2025,7,7))` -> OK; `cal.is_session("2025-07-07")` -> OK; **`cal.is_session(pd.Timestamp("2025-07-07", tz="America/New_York"))` -> `ValueError`** ("a Date must [be tz-naive]"). So NEVER pass a tz-aware timestamp -- pass `.date()` (a python date) or a naive `pd.Timestamp`.
  - `.sessions -> pd.DatetimeIndex` (all session labels, tz-naive midnight; `= self.schedule.index`). Membership test `pd.Timestamp(d).normalize() in cal.sessions` also works but `is_session` is the intended call.
  - `cal.sessions_in_range(start, end) -> DatetimeIndex` (used for the cross-validation below; useful for tests).
  - Other surface (not needed for date-gating): `is_trading_minute`, `is_open_on_minute`, `next_open`, `previous_close`, `session_open`/`session_close`.
  - **NO `.days` and NO `is_trading_day`** in the library. `is_trading_day` is purely the project's wrapper name.

### Q3 -- Per-position vs per-cycle gating (the model choice)

**Recommended: filter ENTRIES by market-open (cycle continues for whatever markets ARE open); always allow EXITS.** Rationale:
- **Why not whole-cycle skip (option a):** a multi-market universe routinely has US open while KR/EU closed (proven: KR has **15** weekday closures in 2026 vs US's **10**, only **4 overlapping** => ~11 weekdays/yr where US is open but KR is not, and vice versa). Skipping the entire cycle because KR is closed would strand US trading on a normal US session -- unacceptable and a behaviour change for US.
- **Why filter-the-universe (option b) is the right primary gate:** it is the minimal, correct intervention -- a closed market simply contributes no candidates today; the rest of the pipeline (screen/rank/analyze/buy) is unchanged. For `["US"]` the filter drops nothing (US ungated) => byte-identical.
- **Per-order guard at execute_buy (option c)** is a cheap defense-in-depth ADDITION, not a replacement -- `order.market` is already available, so a 2-line "skip BUY if market closed" guard at `:1003` costs nothing and protects against a re-eval candidate slipping past the universe filter.
- **Exits are NEVER gated.** Stop-loss enforcement (Step 5.6, `:873-892`) and Step 7 SELLs (`:981-994`) must run for ALL positions regardless of their market's open status. **Stranding a stop-loss is the dangerous failure mode** (a KR position breaches its stop while KR is "closed" per the gate -> if we gate the exit, the loss runs unbounded until KR reopens). For PAPER trading the sell executes against yfinance's last close immediately; the only caveat is the fill price is stale (the position can't actually trade until KR opens in reality). **Allow the paper exit; log that the fill is at last-known close.** This is the safe default and matches "real markets queue the order for next open" closely enough for a paper sim.

**Summary of the model:** ENTRY = gated by `is_trading_day(market, today_local)`; EXIT (stop-loss + risk sells) = always allowed. US-only => entry filter is a no-op.

### Q4 -- Timezone / date convention (whose "today"?)

**Use each market's LOCAL calendar date: `now_local = datetime.now(timezone.utc).astimezone(ZoneInfo(MARKET_CONFIG[market]["timezone"])).date()`, then `cal.is_session(pd.Timestamp(now_local))`.** Proven necessary:
- The loop fires at `paper_trading_hour` ET (default 10:00). At 10:00 ET the day **before** Seollal (2026-02-15 is a Sunday; take 2025-01-28 10:00 ET as the tested example), Seoul has already rolled to the next calendar day:
  - `2025-01-28 10:00 ET` -> Seoul `2025-01-29 00:00 KST`. KR local date = **2025-01-29 = Seollal => is_session=False** (correct). US local date = 2025-01-28 => is_session=True.
  - The **UTC-date approach** would key KR on 2025-01-28 (a normal KR session) => wrongly "open."
- `MARKET_CONFIG` already carries the IANA tz per market (`markets.py:25,43,49`: `America/New_York`, `Europe/Berlin`, `Asia/Seoul`). Use it.
- **Critical implementation detail:** compute the local date, then pass it tz-NAIVE to `is_session` (`pd.Timestamp(local_date)` or just `local_date`). Do NOT pass a tz-aware datetime -- `is_session` raises `ValueError` on tz-aware input (Q2). The pattern is: `localize -> take .date() -> hand the naive date to is_session`.
- This convention is robust at ANY `paper_trading_hour`; the only thing that changes with the hour is how often the local date differs from the US date, but the rule (use market-local date) is always correct.

### Q5 -- Stop-loss / risk exits on a closed market

**Allow the exit; never gate it. Log that the fill is at last-known close.** Detail:
- Step 5.6 stop-loss enforcement (`autonomous_loop.py:855-892`): `check_stop_losses()` -> `execute_sell(reason="stop_loss_trigger")` per breached position. This MUST run for a KR position even when KR is "closed today" per the gate. Reason: gating the exit lets a breached stop run unbounded until reopen -- the exact opposite of risk management. `execute_sell` reads the position's own `market` (`paper_trader.py:370,477`) for the FX conversion, so a closed-market sell still books correct USD P&L off the last close.
- The kill-switch auto-flatten (Step 5.5, `:818-840`) and Step 7 discretionary SELLs (`:981-994`) similarly must not be gated.
- **The honest caveat (document it):** on a truly closed market a real broker would QUEUE the sell for next open; the paper sim fills immediately at yfinance's last close. So the paper exit is slightly optimistic (it assumes you could transact). For a paper system with no real fills this is acceptable and strictly safer than stranding the stop. Add a one-line log when an exit fires on a market that `is_trading_day==False` so the operator can see it.
- **Net rule:** ENTRY gate = ON for non-US; EXIT gate = always OFF (exits always allowed). This asymmetry is the safe default.

### Q6 -- exchange_calendars dependency status

- **`exchange_calendars==4.13.2` is installed** (confirmed: `pip show` -> Version 4.13.2; matches the 50.3 brief). Released 2026-03-10; supports Python 3.10-3.14; XNYS/XETR/XKRX all present.
- **NOT declared in ANY requirements file** (`backend/requirements.txt`, root `requirements.txt`, the `functions/*` reqs -- all grep-negative for `exchange`/`calendar`).
- **BUT it is already imported in the LIVE backend path:** `backend/backtest/markets.py:12` (`try: import exchange_calendars as xcals`) and `backend/autoresearch/monthly_champion_challenger.py:327`. So the runtime already depends on it being present; 50.4's gate adds **no new dependency** -- it just makes a real (currently-broken) consumer of an already-imported library.
- **`pandas_market_calendars` is NOT installed** (it's the alternative; v2.0+ is merely a mirror of exchange_calendars, so no reason to switch).
- **Recommendation:** ADD `exchange_calendars==4.13.2` (or `>=4.13,<5`) to `backend/requirements.txt`. This is a 1-line *declaration* of an already-relied-upon import, not a new dependency -- but per the CLAUDE.md "owner approval for deps" rule, flag it for sign-off. The `try/except ImportError` in `markets.py:11-14` + the fail-open `is_trading_day` mean a missing install degrades to "always trading day" (no crash), so the gate is safe-by-default even if the pin is deferred. **No owner approval is strictly REQUIRED to ship the gate** (the import already exists); the pin is a reproducibility nicety.

---

## External research

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://raw.githubusercontent.com/gerrymanoim/exchange_calendars/master/exchange_calendars/exchange_calendar.py | 2026-05-30 | official (source) | WebFetch full + dist introspection | **Authoritative `is_session` contract:** `def is_session(self, date: Date, _parse=True) -> bool` -- *"Query if a date is a valid session ... True if `date` is a session."* Parses via `parse_date(date,"date",self)`; `.sessions = self.schedule.index` (tz-NAIVE, built by `pd.date_range(...)` with no tz). **No `.days` property** (the source of the markets.py bug). |
| (installed dist 4.13.2, `inspect`) | 2026-05-30 | official (the actual runtime) | local introspection | `cal.is_session(...)` exists; **`cal.days` raises AttributeError** -> proves markets.py:149 is broken. `is_session` REJECTS tz-aware Timestamp (`ValueError`), accepts naive Timestamp/str/date. `.sessions` is a DatetimeIndex. Methods: is_session / sessions_in_range / is_trading_minute / next_open / previous_close (no is_trading_day, no .days). |
| https://pypi.org/project/exchange_calendars/ | 2026-05-30 | official (PyPI) | WebFetch full | v4.13.2 (2026-03-10), Python 3.10-3.14. **XNYS, XETR (Xetra Germany), XKRX (Korea Exchange) all supported.** Usage: `xcals.get_calendar("XNYS").is_session("2022-01-01") -> False`. `special_closes` property for early closes. Calendars maintained by user PRs. |
| https://github.com/gerrymanoim/exchange_calendars | 2026-05-30 | official (README) | WebFetch full | 50+ exchanges out-of-the-box; `is_session("2022-01-01") -> False` for a holiday; multi-exchange (`get_calendar("XNYS")` + `get_calendar("XHKG")`); integrates with `market_prices` for trading bots respecting hours+holidays. |
| https://www.forexchurch.com/stock-market-holidays/xetra | 2026-05-30 | industry (calendar) | WebFetch full | **Xetra 2026 full closures: Jan1, Apr3 (Good Friday), Apr6 (Easter Monday), May1 (Labour), Dec25.** Early closes: Dec24, Dec30, Dec31. TZ = CET, hours 09:00-17:30. => Xetra observes GERMAN holidays distinct from US (Good Friday/Easter Mon/May1 are not US closures). **Matches the library exactly** (cross-validated below). |
| https://www.calendarlabs.com/krx-market-holidays-2026/ | 2026-05-30 | industry (calendar) | WebFetch full | **KRX 2026 lunar holidays: Seollal Feb16-18, Chuseok Sep24-25, substitute National Foundation Day Oct5** (Oct3 is a Sat -> shifts to Mon Oct5). All "Full Day Off." => lunar dates move yearly and CANNOT be derived by a weekday rule. **Matches the library exactly** (cross-validated below). |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://pypi.org/project/pandas_market_calendars/ | official | The alternative library; v2.0+ MIRRORS exchange_calendars (so no reason to switch); ships calendars as code (no runtime server call) -- corroborates the "static, offline" model. |
| https://pandas-market-calendars.readthedocs.io/en/latest/usage.html | official docs | API style of the mirror lib; `is_session`/`schedule`/`valid_days` equivalents. |
| https://medium.com/@wl8380/mastering-trading-periods-in-python-a-developers-guide-to-market-calendars-75fb9e3fff9f | blog | Dev guide naming exchange_calendars "powerful" for multi-market bots; body truncated on fetch (header only). |
| https://github.com/gerrymanoim/exchange_calendars/issues/61 | community | "todo for release 4.0" -- documents the 4.0 method-rename wave (`session_label`->`session`) that made `is_session` canonical; informs the recency scan. |
| https://www.deutsche-boerse.com/.../xetra-trading-calendar-2026.pdf | official (primary) | Deutsche Borse's own 2026 Xetra calendar PDF -- the authoritative German source; the forexchurch full-read already captured the same dates, cross-validated against the library. |
| https://www.tradinghours.com/markets/krx | industry | KRX hours/holidays; WebFetch 403 (anti-scrape). calendarlabs (full read) + library cross-val cover the same KRX 2026 dates. |
| https://www.tradinghours.com/markets/xetra | industry | Xetra hours/holidays; WebFetch 403. forexchurch (full read) + library cross-val cover it. |
| https://www.tradinghours.com/data | industry | Best-practice "check market status before placing orders"; WebFetch 403; the best-practice WebSearch synthesis captured the guidance. |
| https://kstockguide.com/holidays | industry | Korean 2026 KRX + NexTrade closures; corroborates calendarlabs lunar dates. |
| https://www.timeanddate.com/holidays/south-korea/2026 | reference | Underlying SK public holidays (substitute-holiday mechanics) behind the KRX closures. |
| https://www.quantvps.com/blog/trading-bot-strategies | industry | 2026 trading-bot strategy roundup (recency input); generic market-hours/risk guidance. |

**URLs collected (unique): 17** (6 read-in-full + 11 snippet-only; the dist-introspection row is the runtime itself, not a URL).

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "exchange_calendars python is_session multi-market trading bot gating market open holiday 2026"; "Korea Exchange KRX 2025 2026 holidays lunar new year Seollal Chuseok substitute holiday half day Xetra Germany trading calendar."
2. **Last-2-year window (2025/2024):** "exchange_calendars is_session vs is_trading_day deprecated method rename 2025 2026 changelog migration."
3. **Year-less canonical:** "automated trading bot best practice check market open before placing order defer exit holiday closed market stop loss"; "pandas_market_calendars vs exchange_calendars which library check trading day python comparison" (surfaced the canonical library relationship + the trading_calendars->exchange_calendars->pandas_market_calendars lineage).

### Recency scan (2024-2026)
Searched the last-2-year window on exchange_calendars API changes + the library landscape. **Findings (COMPLEMENT prior art; none change the design):**
1. **exchange_calendars 4.0 method-rename wave (issue #61, surfaced 2025-2026 results):** the 4.0 release renamed methods for consistency (`session_label`->`session`, `start_session_label`->`start`); **`is_session` is the canonical query method in 4.x** and there is **no `is_trading_day`** in the library. This DIRECTLY confirms the fix: rewrite `markets.is_trading_day` to call `cal.is_session(...)`. The project's `cal.days` usage is a pre-4.0 idiom that no longer exists -- a latent breakage the rename caused.
2. **pandas_market_calendars v2.0+ is a MIRROR of exchange_calendars** (2026 docs): the two libraries share the same 50+ calendars; pandas_market_calendars adds a `date_range`/`schedule` convenience layer but the holiday rules are identical. => No reason to switch libraries; staying on exchange_calendars (already imported) is correct.
3. **No 2024-2026 change to the XNYS/XETR/XKRX holiday rules** that would affect a date-level gate. The library's 2026 sessions cross-validate exactly against the published Xetra (Deutsche Borse) and KRX (calendarlabs) 2026 calendars (see below).
4. **No 2024-2026 source contradicts** "gate entries on is_session, always allow exits, use market-local date." Trading-bot best-practice sources (2026 QuantVPS, TradingHours) uniformly say "check market status before placing orders" and "verify the exchange-specific schedule near holidays" -- exactly the gate's purpose.

### DECISIVE cross-validation (library vs published 2026 calendars)
Ran `cal.sessions_in_range("2026-01-01","2026-12-31")` for each exchange and diffed against `pd.bdate_range` (all weekdays) to list weekday closures:
- **XETR 2026 closures = [Jan1, Apr3, Apr6, May1, Dec24, Dec25, Dec31]** -- **exact match** to forexchurch/Deutsche-Borse published Xetra 2026 calendar (incl. the Good Friday/Easter Monday/Labour Day German holidays a US calendar lacks). NB: the library treats Dec24/Dec31 as **full non-sessions** while the published calendar calls them early-closes; for a date-level gate that's conservative (we'd skip entries on those half-days, which is safe).
- **XKRX 2026 closures = [Jan1, Feb16, Feb17, Feb18, Mar2, May1, May5, May25, Aug17, Sep24, Sep25, Oct5, Oct9, Dec25, Dec31]** -- **exact match** to calendarlabs KRX 2026 (Seollal Feb16-18, Chuseok Sep24-25, substitute National Foundation Day Oct5, etc.). The **lunar holidays a naive weekday check would trade through are correctly closed.**
- **XNYS 2026 closures = [Jan1, Jan19 MLK, Feb16 Pres, Apr3 GoodFri, May25 Mem, Jun19 Juneteenth, Jul3, Sep7 Labor, Nov26 Thanks, Dec25]** -- matches the known US 2026 market holiday schedule.
- **The gate's quantified value: KR=15 vs US=10 weekday closures, only 4 overlapping** -> ~11 weekdays/yr where the loop (running on a US weekday) would otherwise try to trade a closed KR market on stale prices. EU=7, 3 overlapping with US.

### Consensus vs debate (external)
- **Consensus:** (a) `is_session(date)` is THE canonical exchange_calendars query for "is this a trading day"; (b) holiday rules differ materially per exchange and lunar (KR) dates move yearly -> a per-exchange calendar library is mandatory, a weekday rule is insufficient; (c) automated systems SHOULD check market status before placing orders and verify exchange-specific schedules near holidays; (d) calendar data should be shipped as code (offline, deterministic), which exchange_calendars/pandas_market_calendars both do.
- **Debate/nuance:** (a) **half-day handling** -- a half-day IS a session (`is_session=True`); a date-level gate correctly KEEPS half-days open (the loop trades on the early-close day, which is right -- the market is open). The library's choice to mark Xetra Dec24/Dec31 as full closures is slightly more conservative than reality but harmless for entries. (b) **defer-vs-execute exits on closed markets** -- real brokers queue for next open; a paper sim can fill immediately at last close. The literature leans "queue," but for paper the immediate fill is safer than stranding a stop. We resolve this by ALWAYS allowing the paper exit + logging the stale fill. (c) **library choice** -- exchange_calendars vs pandas_market_calendars is a non-issue (mirror); stay on the one already imported.

### Pitfalls (from literature + the cross-val) -- applied to the gate
1. **The latent `cal.days` bug** -- the existing `is_trading_day` silently returns `True` forever. If 50.4 reuses it unchanged, the gate does NOTHING (every day "is a trading day"). MUST rewrite to `cal.is_session(...)`. (Found by introspection, not by any external doc -- the highest-value finding.)
2. **tz-aware Timestamp -> ValueError** -- passing `datetime.now(tz)` (tz-aware) directly to `is_session` raises. MUST take `.date()` (naive) first. A naive implementation that forgets this throws at runtime on the first non-US gate check.
3. **Wrong date keying (UTC or US date instead of market-local)** -- misses Korean holidays by the ~9-14h ET->KST skew (proven). MUST localize to the market's tz before `.date()`.
4. **Gating exits** -- stranding a stop-loss on a "closed" market lets losses run unbounded. MUST keep exits ungated.
5. **Whole-cycle skip** -- skipping the cycle because one market is closed strands the open markets (incl. US). MUST gate per-market (filter the universe), not the cycle.
6. **Lunar-holiday drift** -- Seollal/Chuseok move every year; a hardcoded date list rots. The library's maintained calendars handle this (cross-validated for 2026) -- another reason to use the library, not a static list.

---

## Synthesis / deliverable

### (a) Q1-Q6 with file:line + the EXACT gate plug-in point -- see INTERNAL CODE INVENTORY above.
Load-bearing summary:
- **Q1:** no in-cycle market-open gate exists today (only the cron's `mon-fri`). Plug-in = **(A) filter the universe at `autonomous_loop.py:343`** (primary) + **(B) buy-side guard at `:1003`** (defense-in-depth). US-only => both are no-ops.
- **Q2:** `is_trading_day` exists (`markets.py:137`) but is BROKEN (`cal.days` gone in 4.13.2 -> always True). Correct API = `cal.is_session(naive_date)`.
- **Q3:** gate ENTRIES per-market (filter universe + buy guard); always allow EXITS.
- **Q4:** date = market-LOCAL date (`now_utc.astimezone(market_tz).date()`), passed tz-NAIVE to `is_session`.
- **Q5:** allow stop-loss/risk exits on closed markets; log stale fill.
- **Q6:** `exchange_calendars==4.13.2` installed + already imported in live path (NO new dep); recommend pinning in requirements.txt (flag, not blocker).

### (b) Recommended gating model + exit-on-closed-market handling
**ENTRY-gated, EXIT-open, per-market, market-local date.**
- **ENTRY (filter universe, Step 1 `:343`):** keep only symbols whose market `is_trading_day(market, today_local)==True`. Closed market contributes no candidates today. US-only => no-op.
- **ENTRY (buy guard, Step 7 `:1003`):** before `execute_buy`, `if not markets.is_trading_day(order.market, today_local): skip + log`. US order never skipped.
- **EXIT (Step 5.6 stop-loss `:873`, Step 7 SELL `:981`, Step 5.5 kill-switch):** NEVER gated. A closed-market exit fills at last close (paper); log a one-liner noting the market was closed so the operator sees the stale fill. This prevents stranded stops.
- **No whole-cycle skip.** The cycle always runs (US weekday cron); only entries into closed markets are filtered.

### (c) exchange_calendars API calls + `markets.is_trading_day` redesign
**Rewrite `markets.is_trading_day` (markets.py:137-152) to:**
```python
def is_trading_day(date=None, market: str = DEFAULT_MARKET) -> bool:
    """phase-50.4: True if `date` is a trading session for `market`.
    `date` is interpreted as the market's LOCAL calendar date. If None,
    uses 'now' converted to the market's local date. Fail-open: True if
    the calendar is unavailable (preserves pre-50.4 US behaviour)."""
    cal = get_trading_calendar(market)
    if cal is None:
        return True  # xcals missing -> degrade to 'always trading' (no crash)
    import pandas as pd
    from datetime import datetime, date as _date
    from zoneinfo import ZoneInfo
    if date is None:
        tz = get_market_config(market)["timezone"]
        date = datetime.now(timezone.utc).astimezone(ZoneInfo(tz)).date()
    elif isinstance(date, str):
        date = datetime.fromisoformat(date).date()
    elif isinstance(date, datetime):
        date = date.date()  # drop any tz/time -> naive date
    try:
        return bool(cal.is_session(pd.Timestamp(date)))  # NAIVE Timestamp
    except Exception as e:
        logger.warning("is_trading_day(%s,%s) failed: %s; assuming open", date, market, e)
        return True
```
Key changes vs the broken version: **`cal.is_session(...)` not `date in cal.days`**; **tz-naive `pd.Timestamp(date)`**; **`date=None` -> market-local today** (so callers just pass `market`); handles `datetime` by stripping to `.date()` (avoids the tz-aware ValueError).

**Caller-side (autonomous_loop.py), compute `today_local` per market and filter:**
```python
from backend.backtest import markets
# Step 1, after universe assembled (:343):
if _intl_markets:  # only when multi-market is active -> US-only path untouched
    universe = [s for s in universe
                if markets.is_trading_day(market=markets.market_for_symbol(s))]
# Step 7 buy guard (:1003): if not markets.is_trading_day(market=order.market): continue
```
`markets.market_for_symbol(symbol)` (`markets.py:96`) already maps a suffixed symbol -> market code; `is_trading_day(market=...)` defaults the date to that market's local today.

### (d) BYTE-IDENTICAL verification plan (paper_markets=["US"] -> identical to today)
1. **No-op-for-US gate test (the core guarantee).** With `paper_markets==["US"]`, `_intl_markets` is empty -> the Step-1 universe filter block is NEVER entered -> `universe` is byte-identical (`None` or the russell/SP500 list). Assert in a unit test that the universe object is unchanged when `_intl_markets==[]`.
2. **US buy never skipped.** Assert the Step-7 buy guard `markets.is_trading_day(market="US")` -- when reached -- does not drop a US order on a US *trading* weekday (the only days the cron fires). Edge: confirm the guard is wired so US-only runs don't even evaluate it, OR that for a US order on a Mon-Fri the guard returns True identically to "no guard." (Recommended: gate the filter on `if _intl_markets:` so US-only never touches calendar code at all -> provably byte-identical.)
3. **is_trading_day correctness (the fix).** `markets.is_trading_day(date=date(2025,7,4), market="US") == False` (Jul 4, currently returns True due to the bug -> this asserts the bug is fixed); `==True` on a normal weekday; `==False` on a weekend.
4. **tz-naive safety.** `is_trading_day(date=datetime.now(ZoneInfo("Asia/Seoul")), market="KR")` does NOT raise (datetime stripped to date); `is_trading_day("2026-02-16","KR")==False` (Seollal).
5. **Cross-market divergence.** On 2026-10-05 `is_trading_day(market="US")==True` while `is_trading_day(market="KR")==False` (KR substitute holiday) -- proves per-market gating.
6. **Live before/after (US).** With `paper_markets` unset, run one cycle (or the Step-1+Step-7 path) on a US trading weekday; assert the screened set, orders, and trades match a pre-50.4 baseline. NO behaviour change. (Because US-only never enters the calendar branch, this is byte-identical by construction.)
7. **Positive intl smoke.** With `paper_markets=["US","KR"]` in a TEST on a KR holiday (e.g. mock today=2026-02-16), assert KR symbols are filtered out of the universe and any KR buy is skipped, while US symbols pass; assert a KR stop-loss SELL still executes (exit ungated).

### (e) Dependency status
- `exchange_calendars==4.13.2` **installed** + **already imported in the live path** (`markets.py:12`) -> **50.4 adds NO new runtime dependency**; no owner approval is strictly required to ship the gate (the import already exists and is fail-open via `try/except ImportError`).
- **NOT declared** in any `requirements.txt`. **Recommend** adding `exchange_calendars>=4.13,<5` to `backend/requirements.txt` (a 1-line declaration of an existing import, for reproducibility). Per the CLAUDE.md dep-approval rule, flag for owner sign-off -- but this is a declaration of an already-relied-upon library, not a net-new dependency.

### (f) Application mapping (external -> internal file:line)
- exchange_calendars `is_session` contract (source + dist) -> rewrite `markets.is_trading_day` (`markets.py:137-152`) to `cal.is_session(naive_ts)`; the `cal.days` line (`:149`) is the bug.
- tz-aware-rejection (dist test) -> localize-then-`.date()` before `is_session`; date convention in `markets.is_trading_day` + caller (`autonomous_loop.py`).
- Xetra 2026 calendar (forexchurch/Deutsche Borse) + KRX 2026 (calendarlabs) -> validated `MARKET_CONFIG` exchange codes XETR/XKRX (`markets.py:41,47`); confirms the library's holiday rules are correct for the gate.
- Trading-bot best practice "check market status before orders" -> ENTRY gate at universe filter (`:343`) + buy guard (`:1003`).
- "queue exits on closed markets" (best practice) vs paper-sim -> EXIT ungated, fill-at-last-close + log (Step 5.6 `:873`, Step 7 SELL `:981`).
- pandas_market_calendars mirror (recency) -> no library switch; stay on the already-imported exchange_calendars.

## Research Gate Checklist

Hard blockers -- all satisfied:
- [x] >=5 authoritative external sources READ IN FULL (6: exchange_calendars raw source [official], installed-dist introspection [official/runtime], exchange_calendars PyPI [official], exchange_calendars GitHub README [official], forexchurch Xetra 2026 [industry], calendarlabs KRX 2026 [industry]). Hierarchy honored: 4 official/primary + 2 industry calendars.
- [x] 10+ unique URLs total (17 incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported (4.0 rename -> is_session canonical; pandas_market_calendars mirror; no holiday-rule change; library 2026 sessions cross-validate against published calendars)
- [x] Full pages/source read (not abstracts) for the read-in-full set; PLUS decisive empirical validation against the installed 4.13.2 runtime
- [x] file:line anchors for every internal claim (Q1-Q6: cron `:1307`, run_daily_cycle `:124`, universe filter point `:343`, buy guard `:1003`, broken is_trading_day `:137-152`/`cal.days` `:149`, MARKET_CONFIG `:21-52`, market_for_symbol `:96`, order.market `:33`/`:1027`, execute_sell market read `:370,477`, stop-loss `:855-892`, settings hour `:265`, dep import `:12`/`:327`)

Soft checks:
- [x] Internal exploration covered: api/paper_trading (scheduler), services/autonomous_loop (full cycle: universe build, Step 5.5/5.6 exits, Step 7 buys), backtest/markets (calendar helpers + the bug + MARKET_CONFIG + market_for_symbol), services/portfolio_manager (TradeOrder.market), services/paper_trader (execute_buy/sell/mark_to_market market handling), config/settings (paper_trading_hour), requirements files (dep status)
- [x] Contradictions/consensus noted (half-day = session; defer-vs-execute exits; library choice)
- [x] All claims cited per-claim with file:line or URL

## Research-gate JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 11,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```
