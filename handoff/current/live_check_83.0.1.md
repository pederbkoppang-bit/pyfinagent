# live_check evidence — step 83.0.1 (news timestamp point-in-time integrity)

Captured 2026-08-07.

## Schema ALTER (live, run FIRST while streaming_buffer was None)

```
streaming_buffer: None
executing: ALTER TABLE `sunny-might-477607-p8.pyfinagent_data.news_articles` ALTER COLUMN published_at DROP NOT NULL
OK
executing: ALTER TABLE `sunny-might-477607-p8.pyfinagent_data.news_articles` ADD COLUMN effective_trade_date DATE
OK
  published_at: TIMESTAMP NULLABLE
  ingested_at: TIMESTAMP REQUIRED
  provenance: STRING REQUIRED
  effective_trade_date: DATE NULLABLE
```

## Fetcher output — no-timestamp article → NULL + quarantine increment (verbatim)

```
WARNING backend.news.fetcher news quarantine reason=missing_published_at source=fixture count=1 detail=raw value None url=https://livecheck.example.com/83-0-1
quarantine BEFORE: 0
normalized row (subset): {'published_at': None, 'ingested_at': '2026-08-07T09:41:02.972103+00:00', 'provenance': 'live', 'effective_trade_date': None, 'source': 'fixture'}
quarantine AFTER: 1
```

## 5-row dump — published_at, ingested_at, provenance, effective_trade_date together (verbatim; no partition predicate)

```
{'article_id': 'd9f70fe5...', 'published_at': datetime.datetime(2022, 3, 15, 14, 30, tzinfo=datetime.timezone.utc), 'ingested_at': datetime.datetime(2026, 8, 7, 9, 41, 15, 891519, tzinfo=datetime.timezone.utc), 'provenance': 'backfill', 'effective_trade_date': datetime.date(2022, 3, 16), 'source': 'fixture'}
{'article_id': '97489bcf...', 'published_at': datetime.datetime(2026, 4, 19, 12, 0, tzinfo=datetime.timezone.utc), 'ingested_at': datetime.datetime(2026, 8, 7, 8, 10, 37, 680923, tzinfo=datetime.timezone.utc), 'provenance': 'live', 'effective_trade_date': None, 'source': 'stub'}
{'article_id': '486cc6c8...', 'published_at': datetime.datetime(2026, 4, 19, 13, 30, tzinfo=datetime.timezone.utc), 'ingested_at': datetime.datetime(2026, 8, 7, 8, 10, 37, 680900, tzinfo=datetime.timezone.utc), 'provenance': 'live', 'effective_trade_date': None, 'source': 'stub'}
{'article_id': '58e1db8f...', 'published_at': datetime.datetime(2026, 4, 19, 14, 0, tzinfo=datetime.timezone.utc), 'ingested_at': datetime.datetime(2026, 8, 7, 8, 10, 37, 680849, tzinfo=datetime.timezone.utc), 'provenance': 'live', 'effective_trade_date': None, 'source': 'stub'}
{'article_id': 'a70b02fd...', 'published_at': datetime.datetime(2026, 4, 19, 12, 0, tzinfo=datetime.timezone.utc), 'ingested_at': datetime.datetime(2026, 8, 7, 8, 10, 23, 399423, tzinfo=datetime.timezone.utc), 'provenance': 'live', 'effective_trade_date': None, 'source': 'stub'}
```

**The load-bearing row is the first**: a 2022-03-15-published article whose
`ingested_at` is **2026-08-07** (the backfill-RUN moment — a real ingest event,
not the article's own era), `provenance='backfill'`, and
`effective_trade_date=2022-03-16` (the next session after the publication-day
session). The four `stub` rows are the 83.0-era pollution already queued for
purge under 83.0.7 — their `effective_trade_date` is NULL because they were
written before this step's column existed, which is itself honest: no value was
backfilled silently.

## Quarantined row persisted in the `__NULL__` partition (verbatim)

Query used `WHERE published_at IS NULL` — a `DATE(published_at)` partition
predicate would silently exclude this row (documented in the contract):

```
NULL-partition row: {'article_id': '65a9abd5...', 'published_at': None, 'ingested_at': datetime.datetime(2026, 8, 7, 9, 41, 39, 216377, tzinfo=datetime.timezone.utc), 'provenance': 'backfill', 'effective_trade_date': None, 'source': 'fixture'}
NULL rows visible without partition predicate: 1
```

## Pollution disclosure

This live_check added **2 `source='fixture'` rows** to `news_articles` (the
2022 backfill row + the quarantined NULL row). Both are added to the 83.0.7
purge scope alongside the 9 `stub` rows (operator ask #8 updated). A third
fixture yield was dropped by intra-batch dedup (identical body hash) before
reaching BigQuery — dedup, not quarantine; noted for completeness.
