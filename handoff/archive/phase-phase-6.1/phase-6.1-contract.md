# Sprint Contract -- phase-6.1
Step: BigQuery schema migration for news + sentiment

## Research Gate
researcher_61 (tier=moderate) gate_passed=true. Brief: `handoff/current/phase-6.1-research-brief.md`.
Key findings:
- DAILY partition on published_at (not hourly) is correct.
- Two tables (news_articles + news_sentiment joined on article_id) per FNSPID / FinBERT + EODHD practitioner consensus -- re-scoring is a first-class operation; body cols are expensive to scan in a wide table.
- DDL-string pattern (like `scripts/migrations/add_llm_call_log.py` I wrote in phase-4.14.23) is the right template -- avoids Python SDK friction with ARRAY columns.
- No existing news tables anywhere; clean slate.
- Target dataset: `pyfinagent_data` (fallback `getattr(settings, 'bq_dataset_observability', None) or 'pyfinagent_data'`).

## Hypothesis
Creating `scripts/migrations/add_news_sentiment_schema.py` with two
idempotent `CREATE TABLE IF NOT EXISTS` DDL blocks -- `news_articles`
(partitioned on `DATE(published_at)`, clustered on `(source, ticker)`)
and `news_sentiment` (partitioned on `DATE(scored_at)`, clustered on
`(article_id, scorer_model)`) -- satisfies phase-6.1.

## Success Criteria
No immutable verification command in masterplan.json. Soft gates:
1. Migration script runs idempotently (`CREATE TABLE IF NOT EXISTS`).
2. Python `ast.parse` clean.
3. Import-smoke the module (no runtime errors).
4. Q/A judges schema against research-cited norms.

## Plan (PRE-commit)
1. Write `scripts/migrations/add_news_sentiment_schema.py` using the
   `add_llm_call_log.py` structure. Two DDL strings, one per table.
2. Columns per research brief (article_id, published_at, fetched_at,
   source, ticker, title, body, url, canonical_url, body_hash,
   language, authors ARRAY<STRING>, categories ARRAY<STRING>,
   raw_payload JSON; sentiment table has scorer_model + version,
   scored_at, sentiment_score, label, confidence, latency_ms,
   cost_usd, raw_output).
3. `--dry-run` flag prints DDL without executing.
4. Syntax check + import smoke.

## Scope honesty
- Does NOT execute the migration against live BQ (requires GCP auth +
  dataset write rights; this is a dev-box context). Prints DDL.
- Dedup logic lives in phase-6.2+ ingestion, NOT in this schema step.

## References
- Research brief: `handoff/current/phase-6.1-research-brief.md`
- Reference migration: `scripts/migrations/add_llm_call_log.py`
- FNSPID (arXiv 2402.06698) + FinBERT (arXiv 2306.02136) for field set
