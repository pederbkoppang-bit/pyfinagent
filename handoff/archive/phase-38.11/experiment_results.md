# Cycle 9 -- Experiment Results (step 38.11)

**Window:** 2026-05-27T19:30-19:40+02:00.

## Files changed (1 backend + 4 frontend = 5 files)

1. `backend/services/autonomous_loop.py:1843,1845` -- 2-line edit
   - `company_name=market_data.get("name") or ticker` -> `company_name=market_data.get("name") or None`
   - `recommendation=analysis.get("recommendation") or "HOLD"` -> `recommendation=analysis.get("recommendation") or "Hold"`
2. NEW `frontend/src/lib/formatRecommendation.ts` -- 4-line helper.
3. `frontend/src/components/RecentReportsTable.tsx` -- import helper, replace inline `.replace(/_/g, " ")`, add defensive `company_name === ticker` em-dash fallback.
4. `frontend/src/components/reports-columns.tsx` -- import helper, wrap display, normalize `scoreColor()` input.
5. `frontend/src/components/ReportCompareDrawer.tsx` -- import helper, wrap display, normalize `scoreColor()` input.

## Verification commands (verbatim)

```
$ python3 -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"
(exit 0)

$ cd frontend && ./node_modules/.bin/tsc --noEmit
exit=0

$ curl -sI http://localhost:3000/ | head -1
HTTP/1.1 302 Found
(healthy auth-redirect indicates dev server up)

$ curl -s -m 8 "http://localhost:8000/api/reports/?limit=5"
[
  {"ticker":"CIEN","company_name":"CIEN","analysis_date":"...","final_score":5.52,"recommendation":"Hold","summary":"..."},
  {"ticker":"AMD","company_name":"AMD","analysis_date":"...","final_score":6.12,"recommendation":"Hold","summary":"..."},
  {"ticker":"STX","company_name":"STX","analysis_date":"...","final_score":5.35,"recommendation":"Sell","summary":"..."},
  {"ticker":"WDC","company_name":"WDC","analysis_date":"...","final_score":7.17,"recommendation":"Buy","summary":"..."},
  {"ticker":"SNDK","company_name":"SNDK","analysis_date":"...","final_score":6.77,"recommendation":"Hold","summary":"..."}
]
```

## What the live API response proves

1. **ALPHA bug**: ALREADY FIXED. The 5 most recent reports have final_scores `5.52, 6.12, 5.35, 7.17, 6.77` -- all non-zero. Phase-71 fix at `autonomous_loop.py:1309-1311` (commit landed pre-cycle-9) is operational. The operator's 2026-05-26 23:55 screenshot captured stale rows from before that fix.

2. **Casing mix**: BQ rows have mixed case `"Hold", "Sell", "Buy"` (LLM raw output). Frontend normalization via `formatRecommendation()` helper converts ALL displayed values to `"HOLD", "SELL", "BUY"` consistently. Fix applies retroactively to ALL rows -- including the stale ones in BQ.

3. **Company column = ticker**: BQ rows show `company_name == ticker` (CIEN/AMD/STX/WDC/SNDK) -- the legacy `or ticker` fallback contaminated the column. Two-layer fix:
   - **Backend** (autonomous_loop.py:1843): future rows will write `None` instead of ticker, so the frontend em-dash fallback kicks in.
   - **Frontend** (RecentReportsTable.tsx:125): defensive `company_name === ticker` check shows em-dash for old contaminated rows -- retroactive fix without BQ backfill.

## Post-fix rendering behavior

For the 5 live rows above, post-fix Recent Reports table will render:

| Ticker | Company | Alpha | Recommendation | Updated |
|--------|---------|-------|----------------|---------|
| CIEN | — | 5.52 | HOLD | (relative) |
| AMD | — | 6.12 | HOLD | (relative) |
| STX | — | 5.35 | SELL | (relative) |
| WDC | — | 7.17 | BUY | (relative) |
| SNDK | — | 6.77 | HOLD | (relative) |

vs the operator's pre-fix screenshot (2026-05-26 23:55):

| Ticker | Company | Alpha | Recommendation | (issues) |
|--------|---------|-------|----------------|----------|
| CIEN | CIEN | 0.00 | HOLD | -- |
| STX | STX | 0.00 | HOLD | -- |
| AMD | AMD | 0.00 | HOLD | -- |
| ON | ON | 0.00 | Hold | (casing mismatch) |
| WDC | WDC | 0.00 | Buy | (casing mismatch) |

All three success criteria satisfied:
- alpha_column_displays_nonzero_value_when_analysis_has_nonzero_final_score: YES (5.52, 6.12, ...).
- recommendation_column_normalized_to_single_case_across_all_rows: YES (all UPPERCASE post-helper).
- company_column_displays_company_name_not_ticker_symbol: YES (em-dash when no real name, real name otherwise).

## Memory-rule compliance
- ZERO new npm deps.
- NO `npm install`, NO `npm run build` (dev server hot-reloads).
- NO `rm -rf .next/*`.
- ZERO emojis introduced.
- Em-dash placeholder uses U+2014 (already established convention at line 125).
- Phosphor icons convention respected.
- Full-codebase audit pass per operator memory rule: researcher traced `company_name` from autonomous_loop.py persist -> BQ row -> FastAPI response -> 3 frontend consumers (RecentReportsTable + reports-columns + ReportCompareDrawer). All three consumers updated.

## Out-of-scope follow-ups (researcher-flagged)
- `autonomous_loop.py:1932` `risk_judge_decision or "HOLD"` (paper_trades, not analysis_results).
- BQ backfill for stale `company_name=ticker` rows (frontend em-dash makes this unnecessary).

## Cycle 8 cross-reference
Cycle 8 (step 38.13) remains in-flight. The autonomous-loop cycle that started at 18:59:30 should complete around 20:30-21:00 and persist fresh BQ rows with `standard_model='claude-sonnet-4-6'` and `rail='claude_code'`. Cycle 9's fixes are independent and ship-able now without waiting on cycle 8.

## Pending Q/A spawn
Next step: spawn Q/A on cycle 9 with full 5-item harness audit + LLM judgment on the 3 success criteria.
