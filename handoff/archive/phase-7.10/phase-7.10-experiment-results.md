# Experiment Results — phase-7 / 7.10 (Hiring scaffold — LinkUp)

**Step:** 7.10 **Date:** 2026-04-20 **Cycle:** 1.

One new file: `backend/alt_data/hiring.py` (~200 lines). Scaffold mirroring twitter.py/google_trends.py. DDL for `alt_hiring_signals` (12 cols, partition as_of_date, cluster ticker+department). `_STARTER_COMPANIES = ('AAPL','MSFT','NVDA','AMZN','GOOGL')`. Real impls: `_posting_id` sha256 surrogate, `normalize` with is_active derived from last_seen_at recency. Stubs: `fetch_postings` (LinkUp REST + MSA deferred to phase-7.12).

```
$ python -c "import ast; ast.parse(open('backend/alt_data/hiring.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ python -m backend.alt_data.hiring --dry-run
{"ts": "2026-04-19T22:31:03.521706+00:00", "dry_run": true, "ingested": 0, "scaffold_only": true}

$ python -c "open('backend/alt_data/hiring.py','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

Immutable criterion `ast.parse(hiring.py)` → PASS.
