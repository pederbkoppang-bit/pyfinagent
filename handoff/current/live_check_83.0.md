# live_check evidence — step 83.0 (news corpus persistence)

## BEFORE migration — both tables ABSENT (captured 2026-08-07, prior to any GENERATE work)

Command (read-only, python google-cloud-bigquery under user ADC):

```
from google.cloud import bigquery
c = bigquery.Client(project="sunny-might-477607-p8")
tables = sorted(t.table_id for t in c.list_tables("sunny-might-477607-p8.pyfinagent_data"))
print("pyfinagent_data tables containing 'news' or 'calendar':", [t for t in tables if 'news' in t or 'calendar' in t])
for name in ("news_articles", "news_sentiment"):
    try:
        c.get_table(f"sunny-might-477607-p8.pyfinagent_data.{name}")
        print(f"{name}: EXISTS")
    except Exception as e:
        print(f"{name}: ABSENT ({type(e).__name__}: 404)" if "404" in str(e) else f"{name}: ERROR {e}")
```

Verbatim output:

```
pyfinagent_data tables containing 'news' or 'calendar': ['calendar_events']
news_articles: ABSENT (NotFound: 404)
news_sentiment: ABSENT (NotFound: 404)
```

## Key-presence check — FINNHUB and BENZINGA (criterion 6 disclosure)

A direct grep of `backend/.env` is permission-denied in this session (secrets
file), so presence is checked through the settings object, which loads
`backend/.env` and never prints values:

Command:

```
source .venv/bin/activate && python3 -c "
from backend.config.settings import get_settings
s = get_settings()
print('finnhub_api_key set:', bool(s.finnhub_api_key))
print('benzinga_api_key set:', bool(s.benzinga_api_key))
"
```

Verbatim output:

```
finnhub_api_key set: False
benzinga_api_key set: False
```

Both fields default to `""` (settings.py:129-130) and the adapters return `[]`
when empty (finnhub.py:68-70, benzinga.py:50-52) — so
`backend/news/sources/finnhub.py` and `benzinga.py` are unreachable dead code
in this environment, as the step asserts.

## Migration run (live, 2026-08-07)

`python scripts/migrations/add_news_sentiment_schema.py` — verbatim tail:

```
executing DDL for news_articles...
OK: sunny-might-477607-p8.pyfinagent_data.news_articles ready.
executing DDL for news_sentiment...
OK: sunny-might-477607-p8.pyfinagent_data.news_sentiment ready.
post-condition OK: sunny-might-477607-p8.pyfinagent_data.news_articles carries ['ingested_at', 'provenance', 'published_at'] as required
post-condition OK: sunny-might-477607-p8.pyfinagent_data.news_sentiment carries ['ingested_at', 'provenance', 'scored_at'] as required
```

## AFTER migration — `bq show --schema` verbatim (2026-08-07)

`bq show --schema sunny-might-477607-p8:pyfinagent_data.news_articles`:

```
[{"name":"article_id","type":"STRING","mode":"REQUIRED"},{"name":"published_at","type":"TIMESTAMP","mode":"REQUIRED"},{"name":"ingested_at","type":"TIMESTAMP","mode":"REQUIRED"},{"name":"provenance","type":"STRING","mode":"REQUIRED"},{"name":"source","type":"STRING","mode":"REQUIRED"},{"name":"ticker","type":"STRING"},{"name":"title","type":"STRING"},{"name":"body","type":"STRING"},{"name":"url","type":"STRING"},{"name":"canonical_url","type":"STRING"},{"name":"body_hash","type":"STRING"},{"name":"language","type":"STRING"},{"name":"authors","type":"STRING","mode":"REPEATED"},{"name":"categories","type":"STRING","mode":"REPEATED"},{"name":"raw_payload","type":"JSON"}]
```

`bq show --schema sunny-might-477607-p8:pyfinagent_data.news_sentiment`:

```
[{"name":"article_id","type":"STRING","mode":"REQUIRED"},{"name":"scorer_model","type":"STRING","mode":"REQUIRED"},{"name":"scorer_version","type":"STRING"},{"name":"scored_at","type":"TIMESTAMP","mode":"REQUIRED"},{"name":"ingested_at","type":"TIMESTAMP","mode":"REQUIRED"},{"name":"provenance","type":"STRING","mode":"REQUIRED"},{"name":"sentiment_score","type":"FLOAT"},{"name":"sentiment_label","type":"STRING"},{"name":"confidence","type":"FLOAT"},{"name":"latency_ms","type":"FLOAT"},{"name":"cost_usd","type":"FLOAT"},{"name":"raw_output","type":"STRING"}]
```

(The `bq` CLI also prints Python-3.9 deprecation WARNINGs; elided here as
non-schema noise — full output reproducible with the same command.)

## Deliberately failed write — counter + log line verbatim (2026-08-07)

Fake client returns a BQ error payload through the real `_insert_rows` path
(`patch.object(bq_writer, "_get_client", ...)`; the failure branch, counter,
and log call are all production code):

```
WARNING backend.news.bq_writer bq_writer write FAILED table=news_articles reason=insert_errors failures=1 detail=[{'index': 0, 'errors': [{'reason': 'invalid', 'message': 'deliberate live_check failure'}]}]
counter BEFORE: 0
writer returned: 0 (fail-open, no raise)
counter AFTER: 1
```
