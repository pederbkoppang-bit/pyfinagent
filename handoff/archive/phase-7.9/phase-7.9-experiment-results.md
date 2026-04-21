# Experiment Results — phase-7 / 7.9 (Google Trends scaffold)

**Step:** 7.9 **Date:** 2026-04-19 **Cycle:** 1.

One new file: `backend/alt_data/google_trends.py` (~180 lines). Same scaffold pattern as 7.4/7.6. Functions: `fetch_trend` stub (TODO pytrends-modern ≥0.2.5), `_trend_id` sha256 hash, `ensure_table`, `upsert`, `ingest_keywords`, `_cli`. DDL for `alt_google_trends` baked in (partition as_of_date, cluster keyword). `_STARTER_KEYWORDS` = ("buy stocks", "sell stocks", "recession", "inflation", "bull market", "bear market"). `_RATE_INTERVAL_S = 12.0` per compliance row 7.9.

```
$ python -c "import ast; ast.parse(open('backend/alt_data/google_trends.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -m backend.alt_data.google_trends --dry-run
{"ts": "2026-04-19T21:32:55.164425+00:00", "dry_run": true, "ingested": 0, "scaffold_only": true}

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

Criterion `ast.parse` → PASS.

Caveats: pytrends archived 2025-04-17; scaffold references pytrends-modern for phase-7.12 live-impl. Raw 0-100 stored; rolling-z normalization deferred.
