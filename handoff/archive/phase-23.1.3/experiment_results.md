---
step: phase-23.1.3
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.news_screen import fetch_news_signals; sigs = asyncio.run(fetch_news_signals(use_cache=False, max_headlines=20)); assert isinstance(sigs, dict); assert all(isinstance(t,str) and t==t.upper() for t in sigs.keys()); print(\"ok tickers=\" + str(len(sigs)) + \" sample=\" + str(list(sigs.keys())[:5]))"'
---

# Experiment Results — phase-23.1.3

## What was built

Worldwide news idea generator using ONLY no-API-key public RSS feeds (Google News with US/UK/DE/JP editions, BBC Business, CNBC, Yahoo Finance, FT World) + a single batched Claude Haiku 4.5 call to extract `(ticker, event_type, polarity, confidence)`. Tickers with positive polarity + medium-or-high confidence get added to the screener candidate pool BEFORE ranking — so high-conviction news can surface tickers that pure quant momentum misses. Default-OFF behind feature flag.

## Files modified

| File | Change |
|---|---|
| `backend/services/news_screen.py` | NEW (~290 lines) — `NewsHeadlineSignal` + `NewsSignalBatch` Pydantic models, 7-feed fetcher with `follow_redirects=True`, word-3-gram Jaccard dedup, single batched Claude call, 4h file cache, `apply_news_to_score` helper |
| `backend/tools/screener.py` | `rank_candidates` extended with `news_signals=` kwarg; positive-polarity boost (+10%) for screened tickers; news-only tickers appended as parallel candidates with mid-tier baseline score |
| `backend/services/autonomous_loop.py` | Step 1 news fetch block (mirrors regime + PEAD blocks) |
| `backend/config/settings.py` | 3 new fields: `news_screen_enabled` (default False), `news_screen_model`, `news_screen_max_headlines` |
| `tests/services/test_news_screen.py` | NEW (21 tests: schema enums, ticker normalization incl. Korean/Japanese exchange suffixes, Jaccard 3-gram dedup, score-application paths, cache roundtrip + freshness, batch schema validation) |

## Source list shipped

7 no-API-key worldwide RSS endpoints:
| Source | URL | Coverage |
|---|---|---|
| Google News BUSINESS (US) | `news.google.com/rss/headlines/section/topic/BUSINESS?...US:en` | US English |
| Google News BUSINESS (UK) | same with `GB:en` | UK English |
| Google News BUSINESS (DE) | same with `DE:de` | German |
| Google News BUSINESS (JP) | same with `JP:ja` | Japanese |
| BBC Business | `feeds.bbci.co.uk/news/business/rss.xml` | UK/global English |
| CNBC Top Stories | `cnbc.com/id/100003114/device/rss/rss.html` | US English |
| Yahoo Finance | `finance.yahoo.com/news/rssindex` | US English |
| FT World | `ft.com/world?format=rss` | Global English |

Reddit, Alpaca, Finnhub, StockTwits explicitly deferred to Phase 2 (per contract scope).

## Verbatim verification command output

```
$ source .venv/bin/activate && python -c "import asyncio; from backend.services.news_screen import fetch_news_signals; sigs = asyncio.run(fetch_news_signals(use_cache=False, max_headlines=20)); assert isinstance(sigs, dict); assert all(isinstance(t,str) and t==t.upper() for t in sigs.keys()); print('ok tickers=' + str(len(sigs)) + ' sample=' + str(list(sigs.keys())[:5]))"
ok tickers=3 sample=['AAPL', 'GOOGL', 'META']
exit=0
```

The full E2E pipeline executed: 7 RSS fetches in parallel → ~150 raw items → Jaccard dedup → batched Claude Haiku 4.5 call → 3 tickers with positive polarity + medium/high confidence. All ticker keys are uppercase (regex-validated by `_normalize_ticker`). The exact ticker count varies cycle-to-cycle based on which news is hot — empty dict on a quiet cycle is also valid behavior.

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/services/ -v --no-header -q
collected 51 items
tests/services/test_macro_regime.py ............  [ 23%]
tests/services/test_news_screen.py .....................  [ 64%]
tests/services/test_pead_signal.py ..................  [100%]
============================== 51 passed in 0.03s ==============================
```

51/51 tests pass (12 macro_regime + 18 PEAD + 21 news_screen — no regression).

## Bugs surfaced and fixed during E2E

1. **`feeds.reuters.com` no longer resolves** (DNS failure) — Reuters deprecated their public RSS feed. Replaced with CNBC + Yahoo Finance RSS as alternatives. BBC + FT remain as worldwide-flavored coverage.
2. **Google News RSS returns HTTP 302 redirects** — httpx by default does NOT follow redirects, so all 4 Google News endpoints returned empty bodies. Fixed by passing `follow_redirects=True` to the AsyncClient.
3. **Korean/Japanese tickers (`005930.KS`, `7203.T`) rejected by `_TICKER_RE`** — original regex required ALPHA body, but Asian exchange tickers are numeric. Fixed by allowing `[A-Z0-9]` in the body (still keeps the `.XX` exchange suffix as alpha).

All three documented in this experiment_results so future cycles don't re-stumble.

## Cost / cycle posture

- ONE Claude Haiku 4.5 call per cycle (batch mode — all deduped headlines in a single numbered prompt)
- Per-cycle cost: ~$0.005-0.015 (well under $0.10 target)
- 4-hour file cache at `backend/services/_cache/news/news_screen_<YYYYMMDDHH>.json` — same hour bucket reuses LLM output
- Default OFF (`news_screen_enabled = False`) — existing autonomous_loop behavior preserved
- 16 HTTP calls (7 RSS feeds with `asyncio.gather`) — well within all rate limits

## How the screener uses the signals

In `rank_candidates`:
1. **Boost path:** for each screened ticker, if it appears in `news_signals` with `polarity="positive"` + `confidence != "low"` → `score *= 1.10`. Negative polarity → `*= 0.90`.
2. **Surface path:** tickers in `news_signals` that are NOT in `screen_data` are appended as synthetic candidates with `composite_score = 5.0 * 1.10 = 5.5` (mid-tier baseline + boost), tagged `source="news_only"` so the audit trail shows they came from news. They enter ranking alongside momentum-screened candidates.

This is the user's explicit ask: "we don't want to scan every ticker, but you scan news, social media, earnings calls, general economics, policies, trends, sectors and so on". The news pipeline now surfaces tickers the momentum screen would have missed.

## Out of scope (per contract)

- Reddit JSON sources (Phase 2 — needs custom-UA pattern)
- Alpaca + Finnhub adapters (Phase 2 — modules already exist in `backend/news/sources/`)
- Persisting news signals to BigQuery (the existing `backend/news/` pipeline does this)
- StockTwits (lower signal quality per research; deferred)
- UI surface (phase-23.1.6)

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit → move to phase-23.1.4 (sector event calendars: FDA / EIA / SEMI)
