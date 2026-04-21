# Experiment Results — phase-7 / step 7.6 (Twitter/X sentiment scaffold)

**Step:** 7.6 **Date:** 2026-04-19 **Cycle:** 1.

## What was built

One new file. `backend/alt_data/twitter.py` (~190 lines). Scaffold mirroring `etf_flows.py`. Functions: `extract_cashtags` (real impl — `$[A-Z]{1,5}` regex), `_hash_author` (real impl — sha256 of author_id for PII discipline per compliance Section 5.5), stubs for `fetch_cashtag_tweets`, `score_sentiment`, `ensure_table`, `upsert`, `ingest_cashtags`, `_cli`.

DDL for `alt_twitter_sentiment` baked in: 10 columns, partition `as_of_date`, cluster `cashtag, author_id_hash`.

## Verification command output

```
$ python -c "import ast; ast.parse(open('backend/alt_data/twitter.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -m backend.alt_data.twitter --dry-run
{"ts": "2026-04-19T21:21:43.183183+00:00", "dry_run": true, "ingested": 0, "scaffold_only": true}

$ python -c "from backend.alt_data.twitter import extract_cashtags; print(extract_cashtags('Buying \$AAPL and \$TSLA puts; btw \$lowercase ignored'))"
['$AAPL', '$TSLA']

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

## Contract criterion check

| # | Criterion | Status |
|---|---|---|
| 1 | `ast.parse` | PASS |

## Known caveats

1. **Scaffold only.** `fetch_cashtag_tweets` returns []; `score_sentiment` returns `(0.0, "neutral")`. Live X API call + FinBERT deferred to phase-7.12.
2. **OAuth not performed.** `adv_70_oauth_tos` honored — no developer app registered at scaffold time.
3. **`extract_cashtags` and `_hash_author` are REAL implementations.** Inline unit sanity: `extract_cashtags("Buying $AAPL and $TSLA ...")` → `["$AAPL", "$TSLA"]`. PII hashing uses sha256.
4. **Pricing flag for phase-7.12:** X API Basic tier ~$100-200/mo may BLOCK cashtag operator; Pro $5K/mo confirmed safe. TODO in the module.
5. **Pre-Q/A self-check:** ast.parse OK, dry-run CLI JSON output correct, extract_cashtags behaves, regression unchanged.
