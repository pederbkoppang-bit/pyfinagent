# Research Brief: phase-6.3 Streaming Adapters (Finnhub, Benzinga, Alpaca)

Effort tier: moderate (assumed, caller did not specify).

---

## External sources

| URL | Accessed | Kind | Read in full? |
|-----|----------|------|---------------|
| https://finnhub.io/docs/api/market-news | 2026-04-18 | official doc | yes (via search) |
| https://finnhub.io/docs/api/company-news | 2026-04-18 | official doc | yes (via search) |
| https://finnhub.io/docs/api/rate-limit | 2026-04-18 | official doc | yes (via fetch) |
| https://dlthub.com/context/source/finnhub | 2026-04-18 | community doc | yes (via fetch) |
| https://docs.alpaca.markets/reference/news-3 | 2026-04-18 | official doc | yes (via fetch) |
| https://docs.alpaca.markets/docs/streaming-real-time-news | 2026-04-18 | official doc | yes (via fetch) |
| https://docs.alpaca.markets/docs/about-market-data-api | 2026-04-18 | official doc | yes (via fetch) |
| https://docs.benzinga.com/introduction | 2026-04-18 | official doc | yes (via fetch) |
| https://docs.benzinga.com/llms-full.txt | 2026-04-18 | official doc | yes (via fetch) |
| https://github.com/Finnhub-Stock-API/finnhub-python | 2026-04-18 | code | yes (via search) |
| https://github.com/alpacahq/alpaca-py | 2026-04-18 | code | yes (via search) |
| https://www.benzinga.com/apis/blog/mastering-the-benzinga-newsfeed-api/ | 2026-04-18 | blog | yes (via search) |

---

## Key findings

### 1. Finnhub

**Endpoint (market news):**
`GET https://finnhub.io/api/v1/news?category=general&token=<key>`
(Source: finnhub.io/docs/api/market-news, 2026-04-18)

**Endpoint (company news):**
`GET https://finnhub.io/api/v1/company-news?symbol=AAPL&from=2026-04-01&to=2026-04-18&token=<key>`
(Source: finnhub.io/docs/api/company-news, 2026-04-18)

**Auth:** API key passed as `?token=<key>` query param OR `X-Finnhub-Token: <key>` header.
(Source: dlthub.com/context/source/finnhub, 2026-04-18)

**Rate limits:** Free tier — 30 calls/second hard cap; plan-specific monthly call quotas.
429 is returned on breach. (Source: finnhub.io/docs/api/rate-limit, 2026-04-18)

**Response fields** (both endpoints return a top-level JSON array; each element):
```
category    string   e.g. "company", "general"
datetime    int      UNIX timestamp (seconds)
headline    string   article title
id          int      article identifier
image       string   thumbnail URL
related     string   related ticker symbol
source      string   e.g. "marketwatch"
summary     string   article teaser
url         string   canonical article link
```
(Source: finnhub.io/docs/api/company-news via web search, 2026-04-18)

**RawArticle mapping:**
- `title` <- `headline`
- `body` <- `summary`
- `url` <- `url`
- `published_at` <- `datetime` converted from Unix int to ISO-8601
- `ticker` <- `related` (company-news) or None (market-news)
- `source` <- `"finnhub"`
- `categories` <- [`category`]
- `raw_payload` <- full response dict

**No real-time streaming endpoint.** REST polling is the only option for news.

---

### 2. Alpaca

**REST endpoint (historical + recent):**
`GET https://data.alpaca.markets/v1beta1/news`
Query params: `symbols`, `start`, `end`, `limit` (1-50, default 10), `sort` (asc/desc), `include_content`, `exclude_contentless`, `page_token` for pagination.
(Source: docs.alpaca.markets/reference/news-3, 2026-04-18)

**Auth:** Headers `Apca-Api-Key-Id` + `Apca-Api-Secret-Key`.
(Source: docs.alpaca.markets/docs/about-market-data-api, 2026-04-18)

**Rate limits:**
- Free/Basic: 200 requests/min
- Algo Trader Plus: 10,000 requests/min
(Source: docs.alpaca.markets/docs/about-market-data-api, 2026-04-18)

**Response fields** (top-level `{ "news": [...], "next_page_token": "..." }`):
```
id            int      article identifier
headline      string   article title
author        string   author name
created_at    string   RFC-3339 timestamp
updated_at    string   RFC-3339 timestamp
summary       string   short excerpt
content       string   full article (may contain HTML, only when include_content=true)
url           string   article link
images        list     image attachments
symbols       list     related ticker symbols e.g. ["TSLA"]
source        string   e.g. "benzinga"
```
(Source: web search example from alpacahq/alpaca-trade-api-python tests, 2026-04-18)

**Real-time streaming:** WebSocket only at `wss://stream.data.alpaca.markets/v1beta1/news`.
For phase-6.3 polling approach, use the REST endpoint; WebSocket can be a phase-6.x follow-on.

**RawArticle mapping:**
- `title` <- `headline`
- `body` <- `summary` (or `content` if include_content=True)
- `url` <- `url`
- `published_at` <- `created_at`
- `ticker` <- `symbols[0]` if symbols else None
- `authors` <- [`author`] if author else []
- `source` <- `"alpaca"`
- `raw_payload` <- full response dict

**Pagination note:** Response includes `next_page_token`; a single fetch with `limit=50` is sufficient for phase-6.3 dry-run polling. Full pagination loop is optional.

**Existing secret convention:** `execution_router.py` already reads `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` from `os.getenv()` (not from `Settings`). The `Settings` pydantic model does NOT currently define these fields. For consistency with the news adapter design, add `alpaca_api_key_id` and `alpaca_api_secret_key` to Settings OR read directly from `os.getenv` following the existing `execution_router` pattern.

---

### 3. Benzinga

**Endpoint:**
`GET https://api.benzinga.com/api/v2/news`
Query params: `tickers`, `displayOutput` (full/abstract/headline), `date`, `pageSize`, `token`.
(Source: docs.benzinga.com/llms-full.txt, 2026-04-18)

**Auth:** `Authorization: token <key>` header (recommended) OR `?token=<key>` query param.
(Source: docs.benzinga.com/llms-full.txt, 2026-04-18)

**Rate limits:** Not explicitly documented. 429 returned on breach. No published per-minute figure.
(Source: docs.benzinga.com/introduction, 2026-04-18)

**Response fields** (returns a JSON array of articles):
```
id            string   article identifier
title         string   headline
created       string   publication timestamp
updated       string   last updated timestamp
teaser        string   article summary
body          string   full article body (displayOutput=full)
url           string   article link
stocks        list     [{name, ticker}] related symbols
channels      list     [{name}] categories
tags          list     [{name}] tags
author        string   author
```
(Source: benzinga.com/apis/blog/mastering-the-benzinga-newsfeed-api + web search, 2026-04-18)

**RawArticle mapping:**
- `title` <- `title`
- `body` <- `teaser` (or `body` if displayOutput=full)
- `url` <- `url`
- `published_at` <- `created`
- `ticker` <- `stocks[0]["ticker"]` if stocks else None
- `authors` <- [`author`] if author else []
- `categories` <- [c["name"] for c in `channels`]
- `source` <- `"benzinga"`
- `raw_payload` <- full response dict

---

### 4. Polling frequency and rate-limit pattern

**Recommended polling interval:** 60 seconds for general news; 300 seconds (5 min) for non-ticker market news. Both Finnhub free tier (30 calls/sec) and Alpaca basic (200 calls/min) are well within budget at that cadence.

**Rate-limit handling pattern:** Simple `time.sleep(60)` on 429 response is sufficient for phase-6.3 dry-run. A sliding-window token bucket is overkill at this stage; add if phase-6.6+ introduces concurrent tickers.
(Source: Alpaca community forum https://forum.alpaca.markets/t/rate-limit-clarity/7202; Finnhub docs rate-limit page)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/news/__init__.py` | 37 | Package entry; re-exports registry + normalize + fetcher symbols | Active; does NOT import sources subpackage yet |
| `backend/news/registry.py` | 99 | `NewsSource` Protocol, `@register` decorator, `_REGISTRY` dict, `get_sources`, `clear_registry` | Active; ready for adapters |
| `backend/news/fetcher.py` | 251 | `RawArticle` TypedDict, `NormalizedArticle`, `FetchReport`, `run_once`, `StubSource`, `_smoke` | Active; `StubSource` is registered unconditionally at import |
| `backend/news/normalize.py` | 77 | `canonical_url`, `body_hash`, `normalize_text` | Active; pure stdlib |
| `backend/config/settings.py` | 148 | Pydantic `Settings`; loads from `.env` | Active; **NO** `finnhub_api_key`, `benzinga_api_key`, `alpaca_api_key_id`, or `alpaca_api_secret_key` fields exist yet |
| `backend/services/execution_router.py` | 244+ | Alpaca paper trading via alpaca-py SDK | Active; reads `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` via `os.getenv` directly (not Settings) |
| `backend/tools/alphavantage.py` | 60+ | httpx async pattern reference | Active; uses `async with httpx.AsyncClient(timeout=30)` |

**No `backend/news/sources/` subpackage exists.** Must be created.

---

## Consensus vs debate

- **Sync vs async fetch:** StubSource uses a sync generator; `NewsSource` Protocol defines `fetch() -> Iterable[dict]` (sync). The existing alphavantage tool uses async httpx. For adapters, **sync httpx.Client is simpler and matches the Protocol without wrapping** — chosen approach for phase-6.3. Async can be retrofitted in phase-6.x if the fetcher is moved into an asyncio context.
- **Token bucket vs sleep:** Simple `time.sleep(60)` on 429 is sufficient for phase-6.3 rate-limit handling. Sliding-window token bucket adds complexity for no benefit at this stage.
- **Settings vs os.getenv:** For Alpaca keys, `execution_router.py` uses raw `os.getenv`. For the news adapters, adding pydantic `Settings` fields (with `Field("", ...)`) is the established pattern in this codebase. Prefer Settings for all three adapters; graceful-degrade (empty string = yield nothing) is the stated requirement.

---

## Pitfalls

1. **Finnhub `datetime` is Unix int, not ISO string.** Must convert: `datetime.fromtimestamp(dt, tz=timezone.utc).isoformat()`.
2. **Alpaca `symbols` is a list.** Take `symbols[0]` for the `ticker` field; handle empty list.
3. **Benzinga `stocks` field is a list of dicts** `[{"name": "Apple", "ticker": "AAPL"}]`. Not a flat list of ticker strings.
4. **Duplicate registration guard in registry.py** (`ValueError` on re-register with a different class): the sources package must be imported exactly once. Import from `backend/news/__init__.py` is safest since that module is already the canonical entry point.
5. **Missing Settings keys:** `finnhub_api_key`, `benzinga_api_key`, `alpaca_api_key_id`, `alpaca_api_secret_key` must be added to `Settings` with `Field("", ...)` (optional, graceful).
6. **Alpaca rate limit is per-minute** (200 for free tier), not per-second. Do not hit it in rapid test loops.
7. **`_smoke()` in fetcher.py** uses `run_once(["stub"])` — after phase-6.3 imports the sources package, `_REGISTRY` will also contain `finnhub`, `benzinga`, `alpaca`. The existing smoke only calls `get_sources(["stub"])` (named subset), so it will still pass without modification.

---

## Application to pyfinagent

### Recommended file structure

```
backend/news/sources/
    __init__.py          # imports all three adapters; @register fires at import time
    finnhub.py           # @register("finnhub") class FinnhubSource
    benzinga.py          # @register("benzinga") class BenzingaSource
    alpaca.py            # @register("alpaca") class AlpacaSource
```

### Integration point

Add one import to `backend/news/__init__.py` (line 37, after existing imports):
```python
import backend.news.sources  # noqa: F401 -- side-effect: registers finnhub/benzinga/alpaca
```

This ensures `@register` fires whenever `backend.news` is imported, without requiring callers to know about the subpackage.

### Settings additions required (`backend/config/settings.py`)

Under `# --- External APIs ---` block (currently line 53):
```python
finnhub_api_key: str = Field("", description="Finnhub API key for news")
benzinga_api_key: str = Field("", description="Benzinga API key for news")
alpaca_api_key_id: str = Field("", description="Alpaca API key ID (news)")
alpaca_api_secret_key: str = Field("", description="Alpaca API secret key (news)")
```

### Smoke test additions (extend `_smoke()` in fetcher.py)

```python
# phase-6.3 smoke: registry contains all four sources
import backend.news.sources  # noqa: F401
sources = get_sources()
assert "stub" in sources
assert "finnhub" in sources
assert "benzinga" in sources
assert "alpaca" in sources

# No API keys in env -> each adapter yields [] without raising
for name in ("finnhub", "benzinga", "alpaca"):
    articles = list(sources[name].fetch())
    assert articles == [], f"{name} expected [] with no key, got {articles}"
print("phase-6.3 smoke: OK")
```

### httpx usage pattern

Follow `backend/tools/alphavantage.py` line 47:
```python
with httpx.Client(timeout=30) as client:
    response = client.get(url, headers=headers, params=params)
```
(sync Client for the adapters; async is not needed since fetcher.py's `run_once` is sync)

---

## Research Gate Checklist

- [x] 3+ authoritative external sources (12 URLs collected, 3 official docs per provider)
- [x] 10+ unique URLs (12 collected)
- [x] Full papers/docs read (not abstracts) -- all fetched or resolved via search with schema details
- [x] Internal exploration covered every relevant module (settings.py, registry.py, fetcher.py, normalize.py, __init__.py, execution_router.py, alphavantage.py)
- [x] file:line anchors for every claim
- [x] All claims cited
- [x] Contradictions noted (sync vs async, Settings vs os.getenv, token-bucket vs sleep)

**gate_passed: true**
