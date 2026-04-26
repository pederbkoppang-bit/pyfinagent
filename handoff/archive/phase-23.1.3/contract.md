---
step: phase-23.1.3
title: Worldwide news idea generator (no-API-key sources + Claude batch event extractor)
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.news_screen import fetch_news_signals; sigs = asyncio.run(fetch_news_signals(use_cache=False, max_headlines=20)); assert isinstance(sigs, dict); assert all(isinstance(t,str) and t==t.upper() for t in sigs.keys()); print(\"ok tickers=\" + str(len(sigs)) + \" sample=\" + str(list(sigs.keys())[:5]))"'
research_brief: handoff/current/phase-23.1.3-research-brief.md
---

# Contract — phase-23.1.3

## Hypothesis

A worldwide news stream from no-API-key public RSS feeds (Google News with multi-country editions, Reuters/BBC/FT Business RSS) + a single batched Claude Haiku 4.5 event-extraction call can surface tradeable single-ticker signals (positive earnings beats, M&A, regulatory actions, analyst upgrades) at near-zero cost. Tickers in news with `impact_polarity="positive"` and `confidence != "low"` get added to the screener candidate pool BEFORE ranking — surfacing tickers that pure quant momentum would have missed.

## Plan

1. **NEW `backend/services/news_screen.py`** mirroring `pead_signal.py` design:
   - Pydantic models `NewsHeadlineSignal` (ticker_mentioned, event_type, impact_polarity, confidence, rationale, skip_reason) and `NewsSignalBatch` — both `ConfigDict(extra="forbid")`
   - `_REGISTERED_FEEDS`: list of 7 no-key RSS endpoints (Google News US/UK/DE/JP BUSINESS topic feeds + Reuters Business + BBC Business + FT Business)
   - `_fetch_rss(url)` — async httpx GET, parse with stdlib `xml.etree.ElementTree`, return list of `{title, url, published_at, source_label}`
   - `_dedup_jaccard(items, threshold=0.4)` — word-3-gram Jaccard self-dedup (no external dep)
   - `_classify_batch_with_claude(headlines)` — single Claude Haiku 4.5 call returning `NewsSignalBatch`, with `_strip_unsupported_schema_keys` applied
   - `fetch_news_signals(use_cache=True, max_headlines=100) -> dict[str, NewsHeadlineSignal]` — orchestrator that returns a `{ticker: NewsHeadlineSignal}` dict
   - 4-hour file cache at `backend/services/_cache/news/news_screen_<YYYYMMDDHH>.json`
2. **Reddit + Alpaca + Finnhub** kept as Phase-2 expansions (only if keys/UA stubs are simple to add). For this cycle, pure RSS-only sources keep the scope tight and verify the architecture.
3. **Extend `backend/tools/screener.py:151` `rank_candidates`** — add `news_signals: dict[str, NewsHeadlineSignal] | None = None` kwarg. Apply: positive_polarity + medium-or-high confidence → `score *= 1.10`. Tickers in news_signals NOT in screen_data → appended as synthetic candidates with `composite_score = 5.0` (mid-tier baseline) so they enter the ranking.
4. **Wire into `backend/services/autonomous_loop.py` Step 1** — after the PEAD block, fetch news_signals when `news_screen_enabled=True`. Pass to `rank_candidates`. Default-OFF.
5. **NEW settings fields**: `news_screen_enabled: bool = False`, `news_screen_model: str = "claude-haiku-4-5"`, `news_screen_max_headlines: int = 100`.
6. **Tests** at `tests/services/test_news_screen.py`:
   - Schema validation (event_type enum, impact_polarity enum, confidence enum)
   - Ticker regex hardening (bad formats stripped)
   - `_dedup_jaccard` paths (identical → 1; near-dups merged; orthogonal kept)
   - `apply_news_to_score` paths (positive boost, low-confidence pass-through, no-signals identity)
   - Cache roundtrip + freshness window
7. **Verification**: immutable command in front-matter calls `fetch_news_signals(max_headlines=20)` end-to-end against real RSS + real Claude. Returns a non-error dict (may be empty if no positive-polarity hits, that's OK).

## Out of scope

- Reddit JSON sources (Phase 2 — needs custom-UA pattern; keep cycle scope tight)
- Alpaca + Finnhub adapters (Phase 2 — already exist in `backend/news/sources/`; could be wired but adds key-availability branching)
- Persisting news_signals to BigQuery (the existing news pipeline does this already; this cycle is a separate fast path)
- Real-time streaming (this is daily-batch only)
- UI surface (phase-23.1.6)

## Files modified

- `backend/services/news_screen.py` — NEW (~280 LOC)
- `backend/tools/screener.py` — extend rank_candidates with news_signals kwarg + apply
- `backend/services/autonomous_loop.py` — Step 1 news fetch block
- `backend/config/settings.py` — 3 new fields
- `tests/services/test_news_screen.py` — NEW

## Verification

The front-matter command exercises real RSS + real Claude — no mocks. It verifies:
- 7 RSS endpoints can be fetched (worst case: graceful degrade if some are 404/timeout)
- Dedup pipeline runs without raising
- Claude batch call returns parseable `NewsSignalBatch`
- Returned dict has uppercase ticker keys
- Function is reachable from clean `python -c`

## References

- `handoff/current/phase-23.1.3-research-brief.md` — full research brief (627 lines, 7 sources read in full, gate_passed: true)
- `backend/services/pead_signal.py` + `backend/services/macro_regime.py` — design templates
- `backend/news/dedup.py` — pre-existing dedup helpers (Stage 1 URL+hash, Stage 3 BQ — used as Phase-2 hardening reference, not required for this cycle)
- `backend/tools/screener.py:151` — extension point
- `backend/services/autonomous_loop.py:127-145` — PEAD block (news slots in after)
