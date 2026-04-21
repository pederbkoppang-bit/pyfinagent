# Experiment Results -- phase-6.1

## What was built
`scripts/migrations/add_news_sentiment_schema.py` -- idempotent
BigQuery migration for two tables:

- `pyfinagent_data.news_articles` -- 14-col append-only fact table
  with ARRAY<STRING> authors/categories and JSON raw_payload.
  PARTITION BY DATE(published_at), CLUSTER BY source, ticker.
- `pyfinagent_data.news_sentiment` -- 10-col re-scorable enrichment
  table. PARTITION BY DATE(scored_at), CLUSTER BY article_id,
  scorer_model.

Both use `CREATE TABLE IF NOT EXISTS` for idempotency. CLI supports
`--dry-run` to print DDL without executing.

Field set matches FNSPID (arXiv 2402.06698) + FinBERT (arXiv 2306.02136) +
EODHD practitioner consensus per the research brief.

## Files changed
- NEW: `scripts/migrations/add_news_sentiment_schema.py` (140 lines)

## Verification
Syntax OK:
```
$ python -c "import ast; ast.parse(open('scripts/migrations/add_news_sentiment_schema.py').read())"
syntax OK
```

Dry-run prints clean DDL for both tables:
```
$ python scripts/migrations/add_news_sentiment_schema.py --dry-run
== news_articles (dry-run) ==
CREATE TABLE IF NOT EXISTS `sunny-might-477607-p8.pyfinagent_data.news_articles` (
  article_id STRING NOT NULL,
  published_at TIMESTAMP NOT NULL,
  ...
)
PARTITION BY DATE(published_at)
CLUSTER BY source, ticker
OPTIONS (description = "phase-6.1 news ingestion fact table (append-only)")

== news_sentiment (dry-run) ==
CREATE TABLE IF NOT EXISTS `sunny-might-477607-p8.pyfinagent_data.news_sentiment` (
  article_id STRING NOT NULL,
  scorer_model STRING NOT NULL,
  ...
)
PARTITION BY DATE(scored_at)
CLUSTER BY article_id, scorer_model

dry-run: no BigQuery writes executed.
```
Exit: 0.

## Scope honesty
- Migration was NOT executed against live BQ this cycle -- dry-run
  only. Executing requires GCP-auth + would be blocked by the phase-4.14.27
  PreToolUse hook (`MCP_MIGRATE_TOKEN=granted`). Deferred to an operator
  with the MIGRATE token.
- Dedup logic lives in phase-6.2+ ingestion, NOT in this schema step.

## References
- Contract (pre-commit): `handoff/current/phase-6.1-contract.md`
- Research: `handoff/current/phase-6.1-research-brief.md`
- Reference migration: `scripts/migrations/add_llm_call_log.py` (phase-4.14.23)
