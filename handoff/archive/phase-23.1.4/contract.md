---
step: phase-23.1.4
title: Sector event calendars (FDA PDUFA scrape + earnings catalyst overlay; EIA + SEMI deferred to Phase 2)
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.sector_calendars import fetch_sector_events, apply_sector_events_to_score, SectorEvent; events = asyncio.run(fetch_sector_events(use_cache=False)); assert isinstance(events, dict); assert all(isinstance(v, SectorEvent) for v in events.values()); from datetime import date; out = apply_sector_events_to_score(10.0, \"AAPL\", \"Information Technology\", {}); assert out == 10.0; print(\"ok events=\" + str(len(events)) + \" sources=\" + str(sorted(set(v.source for v in events.values()))))"'
research_brief: handoff/current/phase-23.1.4-research-brief.md
---

# Contract — phase-23.1.4

## Hypothesis

Sector-specific catalyst calendars (FDA PDUFA dates + upcoming earnings) surface high-conviction event-driven candidates that pure quant momentum + news + macro miss. Tickers with imminent positive catalysts (e.g. earnings in 1-3 days, FDA approval expected) get a +20% boost. Tickers within ±1 day of a binary FDA event get FILTERED OUT (no new positions during binary risk windows). Pure data-pull cycle — $0 LLM cost.

## Plan

1. **NEW `backend/services/sector_calendars.py`** mirroring previous service designs (no LLM call needed, so no schema-strip):
   - `SectorEvent` Pydantic model — `ConfigDict(extra="forbid")`, fields: ticker, event_type, scheduled_date, days_to_event, sector, signal_direction, drug_name, indication, source, confidence, metadata
   - `_fetch_fda_pdufa_events()` — async scrape `rttnews.com/corpinfo/fdacalendar.aspx`, parse the HTML table for ticker / drug / date / indication, return list[SectorEvent]
   - `_fetch_earnings_events()` — async BQ query of `pyfinagent_data.calendar_events` for `event_type='earnings'` in the next 7 days; return list[SectorEvent]
   - `fetch_sector_events(use_cache=True) -> dict[ticker, SectorEvent]` orchestrator
   - 6-hour file cache at `backend/services/_cache/sector_calendars/sector_calendars_<YYYYMMDD>.json`
   - Graceful degradation on RTTNews 4xx/5xx (returns empty list) and BQ failure (returns empty list); still returns the other source's events
2. **`apply_sector_events_to_score(score, ticker, sector, sector_events) -> float | None`** — returns:
   - `None` if ticker has a binary-risk event within ±1 day (FILTER OUT)
   - `score * 1.20` for positive catalyst within 7 days (BOOST)
   - `score * 1.10` for upcoming earnings within 1-3 days (mild boost)
   - `score` unchanged otherwise
3. **Extend `backend/tools/screener.py:151` `rank_candidates`** — add `sector_events: dict[str, SectorEvent] | None = None`. Apply: drop on None return, boost on float. Mirror PEAD's drop-on-None pattern.
4. **Wire into `backend/services/autonomous_loop.py` Step 1** — after the news block, fetch sector_events when `sector_calendars_enabled=True`. Pass to `rank_candidates`. Default-OFF.
5. **NEW settings fields**: `sector_calendars_enabled: bool = False`, `sector_calendars_lookahead_days: int = 7`.
6. **Tests** at `tests/services/test_sector_calendars.py`:
   - Schema validation
   - `apply_sector_events_to_score` paths (binary filter, positive boost, mild earnings boost, no-event identity)
   - `_parse_rttnews_html` on a stubbed HTML fixture (handles missing rows, malformed tickers)
   - Cache roundtrip + freshness window
   - days_to_event calculation (positive/negative based on relative to today)
7. **Verification** — immutable command in front-matter exercises the FULL fetch pipeline (real RTTNews + real BQ) and confirms a valid `dict[str, SectorEvent]` is returned. Empty dict is acceptable on a quiet calendar day.

## Out of scope

- EIA petroleum API integration (Phase 2 — needs free EIA_API_KEY registration)
- SEMI / TSMC monthly billings (Phase 2 — TWSE form-POST is fragile)
- ISM PMI sector signals (covered by macro_regime filter cycle 1)
- BQ table for sector_event_history (file cache covers Phase 1)
- UI surface (phase-23.1.6)

## Files modified

- `backend/services/sector_calendars.py` — NEW (~250 LOC)
- `backend/tools/screener.py` — extend rank_candidates with sector_events kwarg + apply (filter on None / boost on float)
- `backend/services/autonomous_loop.py` — Step 1 sector events fetch block
- `backend/config/settings.py` — 2 new fields
- `tests/services/test_sector_calendars.py` — NEW

## Verification

The front-matter command exercises real RTTNews + real BQ — no mocks. It verifies:
- HTTP scrape works (or returns empty list on outage)
- BQ earnings query works (or returns empty on no-data)
- `SectorEvent` schema validates
- `apply_sector_events_to_score` returns identity when no events
- Returned dict has uppercase ticker keys
- Function reachable from clean `python -c`

## References

- `handoff/current/phase-23.1.4-research-brief.md` — full research brief (401 lines, 7 sources read in full, gate_passed: true)
- `backend/services/news_screen.py` + `pead_signal.py` + `macro_regime.py` — design templates
- `backend/tools/screener.py:151` — rank_candidates extension surface
- `backend/services/autonomous_loop.py` Step 1 — news block (sector slots in after)
- `backend/db/bigquery_client.py` — BQ client for earnings query
