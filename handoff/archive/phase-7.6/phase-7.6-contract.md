# Sprint Contract — phase-7 / step 7.6 (Twitter/X sentiment scaffold)

**Step id:** 7.6 **Cycle:** 1 **Date:** 2026-04-19 **Tier:** simple

## Research-gate summary

5 sources read in full (X API rate-limits, X API search operators, X API OAuth app-only, FinBERT HF card, FinBERT arXiv 1908.10063), 13 URLs, three-variant queries, recency scan. `gate_passed: true`. Brief at `handoff/current/phase-7.6-research-brief.md`.

## Hypothesis

Scaffold-only module matching `etf_flows.py` house style. 6 functions (stubs returning empties), DDL for `alt_twitter_sentiment`, `_STARTER_CASHTAGS = ("$SPY", "$QQQ", "$AAPL", "$TSLA", "$NVDA")`, `_CASHTAG_RE = r'\$[A-Z]{1,5}\b'`. All live X API + FinBERT wiring deferred to phase-7.12. OAuth advisory `adv_70_oauth_tos` honored by NOT registering a developer app at scaffold time.

## Immutable criterion

- `python -c "import ast; ast.parse(open('backend/alt_data/twitter.py').read())"`

## Plan steps

1. Write `backend/alt_data/twitter.py` scaffold (~200 lines) with:
   - Functions: `fetch_cashtag_tweets`, `extract_cashtags`, `score_sentiment`, `ensure_table`, `upsert`, `ingest_cashtags`, `_cli`.
   - `extract_cashtags` is the ONLY real implementation (regex is cheap + reusable). Others stub.
   - DDL partition `as_of_date`, cluster `cashtag, author_id_hash`.
   - PII: `_hash_author(author_id)` helper using sha256; applied at ingest time per compliance Section 5.5.
2. Immutable check + dry-run CLI + regression.
3. Write results, Q/A, log, flip.

## Out of scope

- No live X API call. No developer-app registration (honors adv_70_oauth_tos).
- No FinBERT model load.
- No BQ table creation (criterion doesn't require).
- ASCII-only.

## References

- `handoff/current/phase-7.6-research-brief.md`
- `backend/alt_data/etf_flows.py` (house style)
- `docs/compliance/alt-data.md` row 7.6 + Section 5.5
- `.claude/masterplan.json` → phase-7 / 7.6
