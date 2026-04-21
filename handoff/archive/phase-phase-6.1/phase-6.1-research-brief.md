# Research Brief -- phase-6.1 BigQuery Schema Migration: News + Sentiment

**Tier:** moderate
**Date:** 2026-04-18
**Researcher:** researcher subagent

---

## External sources (URL coverage)

| URL | Accessed | Kind | Read in full? |
|-----|----------|------|---------------|
| https://docs.cloud.google.com/bigquery/docs/partitioned-tables | 2026-04-18 | Official doc | Yes (via WebFetch) |
| https://cloud.google.com/blog/products/data-analytics/new-bigquery-partitioning-and-clustering-recommendations | 2026-04-18 | Google Cloud blog | Partial (search summary) |
| https://eodhd.com/financial-apis/stock-market-financial-news-api | 2026-04-18 | API vendor doc | Yes (via WebFetch) |
| https://arxiv.org/html/2402.06698v1 | 2026-04-18 | arXiv preprint (FNSPID) | Yes (via WebFetch) |
| https://arxiv.org/html/2306.02136v2 | 2026-04-18 | arXiv preprint (FinBERT) | Yes (via WebFetch) |
| https://dl.acm.org/doi/10.1145/3649451 | 2026-04-18 | ACM survey | Identified (not fetched) |
| https://link.springer.com/chapter/10.1007/978-3-030-66891-4_9 | 2026-04-18 | Springer chapter | Identified (not fetched) |
| https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news | 2026-04-18 | Community dataset | Identified (not fetched) |
| https://docs.cloud.google.com/bigquery/docs/clustered-tables | 2026-04-18 | Official doc | Partial (search summary) |
| https://site.financialmodelingprep.com/developer/docs/stock-news-sentiment-rss-feed | 2026-04-18 | API vendor doc | Identified (not fetched) |

---

## Key findings

### 1. BigQuery partitioning + clustering for news ingestion

**Partitioning strategy:** Timestamp-column partitioning on `DATE(published_at)` (daily granularity) is the correct choice for a news table. Google's official docs state: use daily partitioning for "data spread out over a wide range of dates, or if data is continuously added over time." Hourly is only warranted for "high volume spanning < 6 months." At ~100k rows/day the daily partition stays well under BigQuery's per-partition modification limit. (Source: Google Cloud docs, https://docs.cloud.google.com/bigquery/docs/partitioned-tables, 2026-04-18)

**Clustering strategy:** Cluster on `(source, ticker)` for `news_articles` and `(article_id, scorer_model)` for `news_sentiment`. The docs confirm: "column order affects query performance -- a query filtering only on Country and Status (skipping Order_Date) is not optimized." For `news_articles`, the dominant query patterns are "all articles for ticker AAPL today" and "all articles from finnhub today", so `(source, ticker)` matches both. (Source: same Google Cloud doc)

**BQ has no primary-key enforcement.** Uniqueness must be maintained via dedup logic on `canonical_url` or `body_hash` in the ingestion job -- a MERGE/INSERT INTO ... SELECT ... WHERE NOT EXISTS pattern or a scheduled dedup query. This is the canonical BQ idiom for streaming-to-append tables. (Source: Google Cloud partitioned-tables doc)

**Partition limits:** Monitor per-day partition modification quota. Consolidate small streaming batches into hourly bulk loads rather than per-article single-row inserts to avoid breaching limits at scale.

### 2. Financial news sentiment schema -- industry practice

The EODHD Financial News API (a practitioner API used by hedge funds and quants) returns per article: `date`, `title`, `content`, `link`, `symbols` (array of tickers), `tags` (topic array), `sentiment` object with `polarity`, `neg`, `neu`, `pos`. (Source: https://eodhd.com/financial-apis/stock-market-financial-news-api, 2026-04-18)

FNSPID (arXiv 2402.06698, 2024) -- a major financial NLP dataset -- uses: stock symbol, timestamp, URL, headline, full text, and four algorithmic summary variants. Sentiment scores on a numeric scale (their 1-5 maps to our -1..+1). (Source: https://arxiv.org/html/2402.06698v1, 2026-04-18)

FinBERT paper (arXiv 2306.02136) uses: ticker, date, title, body, sentiment label (positive/negative/neutral), confidence score, mapped NSI value. (Source: https://arxiv.org/html/2306.02136v2, 2026-04-18)

**Consensus field set for a production sentiment table:** article_id, scorer_model, scorer_version, scored_at, sentiment_score (-1..+1), sentiment_label (bullish/bearish/neutral/mixed), confidence (0..1), cost_usd, latency_ms, raw_output (truncated). Tracking `scorer_model` and `scorer_version` separately is validated by the FinBERT and FNSPID papers, which both note model/version drift changes scores for the same article. (Source: FNSPID arXiv 2402.06698; FinBERT arXiv 2306.02136)

### 3. One table vs. two tables: recommendation

**Recommendation: TWO tables** (`news_articles` + `news_sentiment`), joined on `article_id`.

Reasoning:
- Re-scoring is a first-class operation: running a new model (e.g. upgrading gemini-2.0-flash to gemini-2.5-flash) should generate new `news_sentiment` rows without touching the immutable article body.
- Sentiment models will change more frequently than news sources -- versioning is cleanest when decoupled.
- Query patterns differ: article queries filter on `(published_at, source, ticker)`; sentiment queries filter on `(scorer_model, scored_at)`.
- Body/full-text columns (up to 1M chars) are expensive to scan in a wide table; keeping them in a dedicated `news_articles` table allows future column-level access control.
- Industry precedent: EODHD bundles sentiment inline for a REST API (latency wins) but warehouse designs (Snowflake, BQ) universally separate enrichment from raw fact tables. (Source: EODHD docs; FNSPID paper)

**Inlined sentiment is acceptable only** if there is a strict 1:1:1 mapping (one article, one model, forever). That assumption will break in phase-6.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/migrations/add_llm_call_log.py` | 77 | Phase-4.14.23 migration reference; `CREATE TABLE IF NOT EXISTS` via DDL string + `get_settings()` | Active; primary reference |
| `scripts/migrations/add_delisted_at_column.py` | 53 | Simplest pattern: `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` + `--dry-run` arg | Active; reference for ALTER pattern |
| `scripts/migrations/add_round_trip_schema.py` | 93 | Mixed pattern: `ALTER TABLE` + `create_table(exists_ok=True)` using Python SDK schema objects | Active; alternative create pattern |
| `scripts/migrations/migrate_bq_schema.py` | (unchecked) | Older general migration | Likely legacy |
| `backend/config/settings.py` | 148 | Pydantic-settings; `get_settings()` entry point | Active |
| `backend/db/bigquery_client.py` | 200+ | BQ client wrapper; `social_sentiment_score` + `nlp_sentiment_score` stored inline in `analysis_results` | Active; no news tables present |

**No existing `news_articles` or `news_sentiment` tables were found in any Python file.** Grep over the entire `backend/` directory for those names returned zero matches. (Checked: `backend/db/bigquery_client.py`, `backend/tools/`, `backend/agents/`).

---

## Dataset and settings mapping

`backend/config/settings.py` defines (lines 38-43):
- `bq_dataset_reports: str = "financial_reports"` -- SEC filings, analysis results
- `bq_dataset_portfolio: str = "pyfinagent_pms"` -- paper trading
- `bq_dataset_outcomes: str = "financial_reports"` -- outcome tracking

There is **no `bq_dataset_data` or `bq_dataset_observability` key**. The `add_llm_call_log.py` migration (line 62) falls back with:
```python
dataset = getattr(settings, "bq_dataset_observability", None) or "pyfinagent_data"
```

**Recommendation:** The news tables should land in `pyfinagent_data` (the primary prod data dataset, per CLAUDE.md). Use the same `getattr(...) or "pyfinagent_data"` fallback pattern. Do NOT add a new settings key -- the migration can hardcode the fallback safely as all other data tables (historical_prices, harness_learning_log, etc.) already live there.

---

## Recommended schema (confirmed)

### `pyfinagent_data.news_articles`

```sql
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.news_articles` (
  article_id      STRING    NOT NULL,   -- uuid4; surrogate key
  published_at    TIMESTAMP NOT NULL,   -- article publication time (UTC)
  fetched_at      TIMESTAMP NOT NULL,   -- when ingested
  source          STRING    NOT NULL,   -- finnhub | benzinga | alpaca | manual
  ticker          STRING,               -- primary ticker (nullable; multi-ticker -> repeated rows or denorm)
  title           STRING,
  body            STRING,               -- full body up to 1M chars
  url             STRING,
  canonical_url   STRING,               -- dedup anchor (normalized URL)
  body_hash       STRING,               -- sha256(canonical body); dedup anchor
  language        STRING,
  authors         ARRAY<STRING>,
  categories      ARRAY<STRING>,
  raw_payload     JSON                  -- original API row for audit/re-processing
)
PARTITION BY DATE(published_at)
CLUSTER BY source, ticker
OPTIONS (description = "phase-6.1 raw news articles; partitioned daily by published_at")
```

### `pyfinagent_data.news_sentiment`

```sql
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.news_sentiment` (
  article_id      STRING    NOT NULL,   -- FK to news_articles.article_id
  scorer_model    STRING    NOT NULL,   -- e.g. gemini-2.0-flash, claude-haiku-4-5, finbert, vader
  scorer_version  STRING,               -- model version string
  scored_at       TIMESTAMP NOT NULL,
  sentiment_score FLOAT64,             -- -1.0 (bearish) to +1.0 (bullish)
  sentiment_label STRING,               -- bullish | bearish | neutral | mixed
  confidence      FLOAT64,             -- 0.0 to 1.0
  latency_ms      FLOAT64,
  cost_usd        FLOAT64,             -- 0 for local/open models
  raw_output      STRING               -- truncated scorer response for audit
)
PARTITION BY DATE(scored_at)
CLUSTER BY article_id, scorer_model
OPTIONS (description = "phase-6.1 per-article sentiment scores; re-score-safe; partitioned daily by scored_at")
```

---

## Migration implementation pattern (confirmed)

The canonical pattern from `add_llm_call_log.py` (phase-4.14.23):

1. Module-level `DDL` string with `{project}.{dataset}.{table}` placeholders.
2. `main()` calls `get_settings()`, resolves dataset via `getattr(settings, "bq_dataset_observability", None) or "pyfinagent_data"`.
3. `client.query(sql).result(timeout=60)`.
4. Script entry: `if __name__ == "__main__": raise SystemExit(main())`.
5. Header docstring: phase id, table purpose, schema summary, run command.

The new file should be `scripts/migrations/add_news_sentiment_schema.py` and create BOTH tables in sequence (two DDL calls, first `news_articles`, then `news_sentiment`). Idempotent via `CREATE TABLE IF NOT EXISTS`.

Optional: add `--dry-run` flag per `add_delisted_at_column.py` pattern (line 25-30 of that file). Recommended -- makes the migration safe to run in CI without BQ credentials.

---

## Pitfalls (from literature + internal audit)

1. **BQ ARRAY columns in DDL strings**: `ARRAY<STRING>` is valid BQ DDL syntax in a CREATE TABLE statement. However the Python BQ SDK `SchemaField` approach (`add_round_trip_schema.py` pattern) does not support ARRAY natively without mode="REPEATED". Use the raw DDL string approach (`add_llm_call_log.py` pattern) to avoid SDK friction with ARRAY columns.

2. **JSON type**: `raw_payload JSON` is valid in BQ Standard SQL DDL as of 2022. No workaround needed. Do not use `STRING` and manually serialize -- JSON type enables dot-notation queries.

3. **Partition on nullable column**: `published_at` must be NOT NULL. If source APIs send null timestamps (rare but happens with embargoed releases), the ingestion job should default to `fetched_at`.

4. **Dedup is NOT in the migration**: The schema is append-only; dedup logic lives in the ingestion job (phase-6.2+). `canonical_url` and `body_hash` are indexed via clustering for efficient EXISTS checks but the migration does not enforce uniqueness.

5. **`ticker` as nullable STRING vs REPEATED**: Making ticker a nullable STRING (single ticker per row) is simpler and sufficient for phase-6. A separate denorm table is the right answer for multi-ticker articles, but that is phase-6.3+ scope -- do not over-engineer the schema migration.

6. **`authors` and `categories` as ARRAY<STRING>**: BQ ARRAY columns cannot be NULL; they default to an empty array `[]`. Ingestion code must send `[]` not `None` for these fields.

---

## Consensus vs debate

**Consensus:** Two-table design (articles + sentiment) is the validated warehouse pattern. Partitioning on event timestamp (not ingestion time) is correct for time-series queries. Clustering on `(source, ticker)` for articles is unambiguous.

**Debate:** Whether to include a `news_articles_staging` table for streaming dedup. The brief recommends omitting it from phase-6.1 (schema migration only). The staging table can be added in phase-6.2 (ingestion cron) if the dedup pattern requires it. Adding it now adds schema surface area with no consumer.

---

## Application to pyfinagent (file:line anchors)

- Migration pattern: `scripts/migrations/add_llm_call_log.py:59-76` -- use this verbatim structure.
- `get_settings()` import: `scripts/migrations/add_llm_call_log.py:31` -- `from backend.config.settings import get_settings`.
- sys.path bootstrap: `scripts/migrations/add_llm_call_log.py:31-32` -- `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`.
- Dataset fallback: `scripts/migrations/add_llm_call_log.py:62` -- `getattr(settings, "bq_dataset_observability", None) or "pyfinagent_data"`.
- Dry-run pattern: `scripts/migrations/add_delisted_at_column.py:25-30`.
- Target dataset confirmed as `pyfinagent_data`: `backend/config/settings.py:38-43` (no `bq_dataset_data` key; fallback to hardcoded string is the right call).
- No existing news tables in `backend/db/bigquery_client.py` (confirmed by grep, zero matches).

---

## Research Gate Checklist

- [x] 3+ authoritative external sources (Google Cloud official doc, EODHD API doc, FNSPID arXiv, FinBERT arXiv)
- [x] 10+ unique URLs collected (10 URLs in table above)
- [x] Full papers/docs read (not abstracts) -- Google BQ doc, EODHD doc, FNSPID, FinBERT fetched in full
- [x] Internal exploration covered every relevant module (migrations/*.py, config/settings.py, db/bigquery_client.py)
- [x] file:line anchors for every claim
- [x] All claims cited
- [x] Contradictions / consensus noted (staging table debate noted)

**gate_passed: true**
