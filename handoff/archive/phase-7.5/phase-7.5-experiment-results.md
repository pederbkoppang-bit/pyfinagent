# Experiment Results — phase-7 / 7.5 (Reddit WSB scaffold + license doc)

**Step:** 7.5 **Date:** 2026-04-20 **Cycle:** 1.

## What was built

Two new files.

1. `backend/alt_data/reddit_wsb.py` (~230 lines) — scaffold with Reddit-specific deltas vs `twitter.py`:
   - `_USER_AGENT = "python:pyfinagent:1.0 (by /u/pederbkoppang)"` (Reddit mandated format)
   - `_CASHTAG_RE = r"\$[A-Z]{2,5}\b"` (2-char floor, avoids $A/$I noise)
   - `_RATE_INTERVAL_S = 0.6` (100 QPM free tier)
   - `_STARTER_SUBS = ("wallstreetbets", "stocks", "investing")`
   - DDL for `alt_reddit_sentiment` (14 cols; partition `as_of_date`, cluster `subreddit, cashtag`)
   - `extract_cashtags` + `_hash_author` live; `fetch_wsb_posts` + `score_sentiment` stubbed for phase-7.12.
   - No `import praw` / `os.environ` / `os.getenv` at import time.
2. `docs/compliance/reddit-license.md` — first per-vendor license doc. 8 sections: Scope / Access / ToS record (RBP pending) / Rate limits / Retention+PII / Permitted use / Review cadence / References. Template for 7.7 Revelio and 7.10 LinkUp.

## Verification

```
$ python -c "import ast; ast.parse(open('backend/alt_data/reddit_wsb.py').read()); print('SYNTAX OK')"
SYNTAX OK

$ test -f docs/compliance/reddit-license.md && echo "LICENSE DOC OK"
LICENSE DOC OK

$ python -m backend.alt_data.reddit_wsb --dry-run
{"ts": "2026-04-19T22:16:49.712902+00:00", "dry_run": true, "ingested": 0, "scaffold_only": true}

$ python -c "from backend.alt_data.reddit_wsb import extract_cashtags; print(extract_cashtags('Long \$AAPL and \$NVDA but not \$A or \$I'))"
['$AAPL', '$NVDA']

$ python -c "open('backend/alt_data/reddit_wsb.py','rb').read().decode('ascii'); open('docs/compliance/reddit-license.md','rb').read().decode('ascii'); print('ASCII OK')"
ASCII OK

$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped
```

Both immutable criteria PASS. 2-char cashtag floor verified (`$A`/`$I` excluded, `$AAPL`/`$NVDA` included).

## Contract criteria

| # | Criterion | Status |
|---|---|---|
| 1 | `ast.parse(reddit_wsb.py)` | PASS |
| 2 | `test -f reddit-license.md` | PASS |

## Caveats

1. **Scaffold only** — PRAW, Reddit app registration, RBP submission all deferred to phase-7.12.
2. **`adv_70_oauth_tos` honored** — no developer-app click-through performed this cycle.
3. **Pushshift defunct; Arctic Shift documented** as the successor for future historical backfill (research-brief finding).
4. **License doc sets precedent** for 7.7 (Revelio) and 7.10 (LinkUp) — same 8-section structure will be reused.
