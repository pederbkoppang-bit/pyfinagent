# Experiment Results -- phase-6.3

## What was built
Three production news adapters + the settings entries they read:

- **`backend/news/sources/finnhub.py`** -- `@register("finnhub")`
  class hitting `https://finnhub.io/api/v1/news` with
  `category=general` + `token=<key>`. Converts the Unix-int
  `datetime` to ISO 8601 for `published_at`, carries `related`
  ticker, `category` -> `categories[0]`, `source` / `id` / `image`
  -> `raw_payload`.
- **`backend/news/sources/benzinga.py`** -- `@register("benzinga")`
  class hitting `https://api.benzinga.com/api/v2/news` with
  `Authorization: token <key>`. Handles `stocks[0].ticker` (list of
  dicts, not strings), `channels[].name` -> `categories`, author
  string, fallback `body -> teaser -> ""`.
- **`backend/news/sources/alpaca.py`** -- `@register("alpaca")`
  class hitting `https://data.alpaca.markets/v1beta1/news` with
  `Apca-Api-Key-Id` + `Apca-Api-Secret-Key` headers. Unwraps
  `{"news": [...]}` envelope, takes `symbols[0]` for ticker,
  `content | summary` for body, `source` -> `categories[0]`.

Subpackage `backend/news/sources/__init__.py` imports all three
for decorator side-effects. `backend/news/__init__.py` now imports
that subpackage so a plain `import backend.news` registers them.

Settings gains 4 optional `Field("", ...)` entries in
`backend/config/settings.py`: `finnhub_api_key`, `benzinga_api_key`,
`alpaca_api_key_id`, `alpaca_api_secret_key`. Defaults empty;
adapters fail-soft (`.fetch()` returns `[]`) when keys are missing.

## Files changed
- NEW: `backend/news/sources/__init__.py` (12 lines)
- NEW: `backend/news/sources/finnhub.py` (85 lines)
- NEW: `backend/news/sources/benzinga.py` (103 lines)
- NEW: `backend/news/sources/alpaca.py` (108 lines)
- EDIT: `backend/news/__init__.py` (+3 lines: side-effect import)
- EDIT: `backend/config/settings.py` (+5 lines: 4 api-key fields + section header)

## Verbatim verification
```
$ python -c "
from backend.news import get_sources, run_once
print(sorted(get_sources().keys()))
report = run_once(['finnhub','benzinga','alpaca'], dry_run=True)
print(report.per_source_counts, report.errors)
"
['alpaca', 'benzinga', 'finnhub', 'stub']
{'finnhub': 0, 'benzinga': 0, 'alpaca': 0} []

$ python -c "
from backend.config.settings import get_settings
s = get_settings()
for k in ['finnhub_api_key','benzinga_api_key','alpaca_api_key_id','alpaca_api_secret_key']:
    print(k, repr(getattr(s,k)))
"
finnhub_api_key ''
benzinga_api_key ''
alpaca_api_key_id ''
alpaca_api_secret_key ''

$ python -m backend.news.fetcher
phase-6.2 smoke: OK
  n_articles=3
  per_source_counts={'stub': 3}
```

Syntax check on all 6 touched files: OK.

## Soft-gate coverage
| Gate | Status |
|------|--------|
| 3 adapter modules + `sources/__init__.py` | MET |
| `get_sources()` == {stub, finnhub, benzinga, alpaca} | MET |
| Each adapter matches `NewsSource` Protocol (runtime_checkable) | MET |
| 4 settings fields present + default "" | MET |
| No-keys graceful degrade ([] not raise) | MET |
| Syntax OK | MET |
| phase-6.2 regression smoke still passes | MET |

## Scope honesty
- NO real API calls this cycle (keys empty in dev). Adapters
  compile + register + degrade gracefully; live polling is a phase-6.8
  smoketest concern.
- NO dedup (phase-6.4).
- NO near-dedup / WebSocket real-time (out of scope).
- Benzinga `body` falls back to `teaser` when the full body isn't
  in the response -- acceptable for headline-feed use cases;
  richer content is a follow-on.

## References
- Contract (pre-commit): `handoff/current/phase-6.3-contract.md`
- Research: `handoff/current/phase-6.3-research-brief.md`
- phase-6.2 foundation: `backend/news/{registry,normalize,fetcher}.py`
- phase-6.1 schema: `scripts/migrations/add_news_sentiment_schema.py`
