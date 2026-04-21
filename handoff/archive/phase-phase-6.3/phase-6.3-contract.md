# Sprint Contract -- phase-6.3
Step: Streaming adapters (Finnhub, Benzinga, Alpaca)

## Research Gate
researcher_63 (tier=moderate) gate_passed=true. Brief: `handoff/current/phase-6.3-research-brief.md`.

Key findings:
- **Finnhub** `GET /api/v1/news?category=general&token=<key>` (market) + `company-news` (per-ticker). Auth via `?token=` query or `X-Finnhub-Token` header. `datetime` is Unix int -> needs conversion.
- **Alpaca** `GET https://data.alpaca.markets/v1beta1/news` with `Apca-Api-Key-Id` + `Apca-Api-Secret-Key` headers. Response top-level `news` list; `symbols` is list (take `[0]` for `ticker`).
- **Benzinga** `GET https://api.benzinga.com/api/v2/news` with `Authorization: token <key>`. `stocks` is list of `{name, ticker}` dicts (not flat strings).
- `Settings` has NO keys for these yet -- must add 4 optional `Field("", ...)` entries.
- Sync `httpx.Client` matches the StubSource + Protocol shape; no async.
- No real-time streaming for news via REST -- polling only (WS exists for Alpaca but out of scope).

## Hypothesis
Adding `backend/news/sources/` subpackage with four modules
(`finnhub.py`, `benzinga.py`, `alpaca.py`, plus `__init__.py` as a
package marker) + importing that subpackage from `backend/news/__init__.py`
to trigger the decorators satisfies phase-6.3. Each adapter registers
via `@register(name)`, exposes a sync `.fetch()` that reads its API
key from `settings`, returns `[]` on missing key (graceful degrade)
or API error (log + swallow), and maps provider fields to `RawArticle`.

## Success Criteria (soft — masterplan has verification=None)
1. `backend/news/sources/__init__.py` + `finnhub.py` + `benzinga.py` +
   `alpaca.py` exist.
2. After `import backend.news`, `get_sources()` returns keys
   `{"stub", "finnhub", "benzinga", "alpaca"}`.
3. Each adapter class matches the `NewsSource` Protocol (has `name`,
   `fetch()`).
4. With NO API keys in env, each real adapter's `.fetch()` returns
   an empty iterable without raising.
5. `from backend.config.settings import Settings; s = Settings()` →
   `s.finnhub_api_key`, `s.benzinga_api_key`, `s.alpaca_api_key_id`,
   `s.alpaca_api_secret_key` all exist and default to empty string.
6. Syntax OK on all new files.
7. Double-import idempotency (`python -m backend.news.fetcher` AND
   direct script) still exits 0 (phase-6.2 contract).

## Plan (PRE-commit)
1. Add 4 `Field("", ...)` entries to `backend/config/settings.py`:
   `finnhub_api_key`, `benzinga_api_key`, `alpaca_api_key_id`
   (may exist), `alpaca_api_secret_key` (may exist). Check first
   to avoid dup.
2. Create `backend/news/sources/__init__.py` that imports each
   adapter so registration side-effects fire on package import.
3. Write `backend/news/sources/finnhub.py`:
   - `@register("finnhub")` class `FinnhubSource`.
   - sync `httpx.Client(timeout=30)` call to the market-news endpoint.
   - Map fields: `headline -> title`, `summary -> body`, `url -> url`,
     `datetime (int) -> iso published_at`, `related -> ticker`,
     `source -> raw_payload.provider_source`.
   - Empty list on missing key or 4xx/5xx.
4. Write `backend/news/sources/benzinga.py`:
   - `@register("benzinga")` class.
   - `Authorization: token <key>` header.
   - Map `title, teaser/body, url, created, stocks[0].ticker,
     channels[].name -> categories`, raw original -> raw_payload.
5. Write `backend/news/sources/alpaca.py`:
   - `@register("alpaca")` class.
   - `Apca-Api-Key-Id` + `Apca-Api-Secret-Key` headers.
   - Map `headline, content (or summary), url, created_at,
     symbols[0], author -> authors[0], source -> categories[0]`.
6. Update `backend/news/__init__.py` to import the sources subpackage.
7. Update `fetcher.py::_smoke` (or add a new smoke path) to assert
   all four source names present and each real adapter returns []
   with no API keys.
8. Run phase-6.2 smoke (must still pass) + new phase-6.3 smoke.

## Scope out
- Real API calls / live polling (deferred; each adapter only RUNS in `run_once` if invoked with keys set).
- Dedup (phase-6.4).
- Sentiment scoring (phase-6.5).
- WebSocket real-time for Alpaca (later if needed).

## References
- Research brief: `handoff/current/phase-6.3-research-brief.md`
- phase-6.2: `backend/news/` package, `NewsSource` Protocol, `RawArticle` TypedDict, `@register` decorator
- `backend/tools/alphavantage.py` httpx pattern
- Official docs (Finnhub market-news, Alpaca news-3, Benzinga llms-full.txt).
