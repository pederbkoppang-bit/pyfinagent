# Contract -- phase-50.4: Market-calendar gating

**Step id:** 50.4 | **Priority:** P3 (phase-50) | **depends_on:** 50.3
**Date:** 2026-05-30 | **harness_required:** true | **$0 LLM** | no pip (exchange_calendars 4.13.2 already installed + imported in markets.py:12).

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (gate: **6 sources read in full, recency scan, 17 URLs, 7 internal files, gate_passed=true**). Decisive:
- **LATENT BUG:** `markets.is_trading_day(date, market)` (markets.py:137-152) is silently broken -- `date in cal.days` (`.days` was removed in exchange_calendars 4.0); the bare except swallows the AttributeError + returns True ALWAYS. Zero live callers today -> 50.4 is the first consumer + MUST rewrite to `cal.is_session(pd.Timestamp(naive))`.
- **No in-cycle market-open gate exists today:** run_daily_cycle (autonomous_loop.py:124) runs unconditionally; the only calendar awareness is the cron `day_of_week="mon-fri"` (paper_trading.py:1307). The loop trades on US weekday holidays. So **US MUST stay UNGATED** (gating US would change behaviour). Gate plug-in: filter the universe inside the existing `if _intl_markets:` block (autonomous_loop.py ~:330, added in 50.3) -> US-only never touches calendar code -> byte-identical.
- **Model: entry-gated, exit-ALWAYS-open, per-market.** Filter ENTRY universe to markets open today; NEVER gate exits (gating a stop-loss strands a breached position -> unbounded loss). No whole-cycle skip.
- **Date convention: market-LOCAL date** -- `datetime.now(timezone.utc).astimezone(ZoneInfo(market_tz)).date()`, passed tz-naive (is_session rejects tz-aware). A UTC/US-date approach misses Korean lunar holidays by the ET->KST skew.
- **API:** `cal.is_session(label) -> bool` (4.x); no `.days`. exchange_calendars==4.13.2 installed + already imported (markets.py:12) -> NO new dep (recommend pinning >=4.13,<5; fail-open try/except, not a blocker).
- **Cross-validated:** library 2026 sessions match published Xetra + KRX calendars incl. lunar Seollal/Chuseok. KR has 15 weekday closures vs US 10 (4 overlap) -> ~11 days/yr the loop would otherwise trade a closed KR market on stale data.

## Hypothesis
Rewriting `is_trading_day` to `cal.is_session` (fixing the latent always-True bug) + filtering the multi-market entry universe by each ticker's market-local trading day (US ungated, exits never gated), all inside the `if _intl_markets:` block, gates international entries on market-open while keeping US-only byte-identical.

## Success criteria (IMMUTABLE -- verbatim from masterplan step 50.4)
1. the loop/backtest gates each market's trades on is_trading_day(market, date); a US market holiday skips US, a German holiday skips EU/.DE independently
2. the calendar dep is verify-installed (import works) without breaking the existing US path; if it ships to requirements it carries owner sign-off
3. live/fixture evidence: a known German holiday (or weekend) where EU is correctly marked closed while US logic is unaffected

**Verification command:** pytest backend/tests/test_phase_50_4_calendar.py + is_trading_day('2026-01-01','EU') is False + ('2026-06-15','US') is True + test -f live_check_50.4.md.
**live_check:** REQUIRED -- a date where one market is open and another closed.

NOTE on criterion #1 wording ("a US market holiday skips US"): the research found the live loop has NO US calendar gate today (it trades US weekday holidays on stale data), and adding one would CHANGE US behaviour (break byte-identity). So the SHIPPED behaviour is: `is_trading_day` is correct + AVAILABLE for all markets (US included -- a caller CAN gate US), but the LIVE loop applies the gate ONLY to non-US markets (US stays ungated = byte-identical). The criterion's "skips US" capability is satisfied by the function (is_trading_day('<us holiday>','US')==False); wiring US gating into the live loop is deliberately NOT done (it would regress the +20% engine). This is disclosed, not a skip.

## Plan steps
1. **markets.py:137-152** -- rewrite `is_trading_day(date, market)`: `cal.is_session(pd.Timestamp(date).tz_localize(None if tz-aware).normalize())`; fail-open True if cal None / error. Keep the (date, market) signature.
2. **autonomous_loop.py** (inside the 50.3 `if _intl_markets:` block, after `universe = base + intl`) -- filter: drop a ticker whose `market_for_symbol(ticker)` market is NOT a trading day today (market-local date); US tickers always kept (ungated); log the drop count. Exits (execute_sell path) untouched.
3. **backend/tests/test_phase_50_4_calendar.py** (NEW) -- is_trading_day: US weekday=True, weekend=False, EU New Year 2026-01-01=False, a KR lunar holiday (Seollal 2026-02-17)=False, a normal EU/KR weekday=True; the gate filter keeps US + drops a closed EU/KR ticker (use a known-closed date via monkeypatch or a date assertion); US-only path unaffected.
4. **Verify:** pytest; the masterplan command; a live is_trading_day check across US/EU/KR on a date where they differ (e.g. a German/Korean holiday that's a US trading day, or vice versa). Capture into live_check_50.4.md.
5. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 50.4 -> done.

## Safety / scope notes
- **Byte-identical:** the gate lives inside `if _intl_markets:` -> paper_markets=['US'] (default) never runs it -> the +20% engine is unchanged. US tickers in a multi-market universe are also never gated (matches today).
- **Exits never gated** -- a stop-loss/sell always fires (paper fills at last close on a closed market; log the stale fill). Only ENTRIES are calendar-gated.
- Fixes a latent always-True bug in is_trading_day (was dead code; now correct).
- No pip/spend/DROP-DELETE. exchange_calendars already a transitive dep; recommend an explicit pin (flag, not blocker).

## References
- handoff/current/research_brief.md (50.4 gate)
- backend/backtest/markets.py:12 (xcals import), :113-134 (get_trading_calendar), :137-152 (is_trading_day -- rewrite)
- backend/services/autonomous_loop.py:~330 (the 50.3 _intl_markets block -- add the gate), :124 (run_daily_cycle), :343 (universe)
- backend/api/paper_trading.py:1307 (cron day_of_week)
- exchange_calendars 4.x is_session; Xetra + KRX 2026 holiday calendars
