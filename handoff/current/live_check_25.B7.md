# Live-check placeholder -- phase-25.B7

**Step:** 25.B7 -- yfinance fallback counter persisted to BQ + WARNING log promotion
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "BQ data_source_events table grows per cycle with yfinance_fallback rows"

## Pre-deployment evidence
- 5/5 verifier PASS.
- AST clean on all 3 touched .py files (orchestrator.py, bigquery_client.py,
  the new migration script).
- Behavioral round-trip in claim 5 confirms the exact `save_data_source_event`
  call shape (ticker, source='yfinance_fallback', kind='fallback',
  article_count, notes) matches what the orchestrator emits.

## Post-deployment operator workflow
1. Pull main:
   ```
   git pull origin main
   ```
2. Apply the migration once (idempotent CREATE IF NOT EXISTS):
   ```
   source .venv/bin/activate
   python scripts/migrations/create_data_source_events_table.py --apply
   ```
3. Restart backend; run a paper-trading cycle; trigger an analysis on a
   ticker where AV is likely empty (small-cap ETF). Watch the backend log:
   ```
   tail -f handoff/logs/uvicorn.log | grep -i yfinance
   # Expect: WARNING level lines: "AV empty for <TICKER> -- using N yfinance articles as fallback"
   ```
4. Verify rows appear in BQ:
   ```sql
   SELECT event_time, ticker, source, kind, article_count, notes
   FROM `sunny-might-477607-p8.pyfinagent_data.data_source_events`
   ORDER BY event_time DESC
   LIMIT 5;
   ```
5. Confirm the aggregable counter works:
   ```sql
   SELECT
     COUNTIF(source = 'yfinance_fallback') AS fallback_rows,
     COUNT(*) AS total_rows,
     SAFE_DIVIDE(COUNTIF(source = 'yfinance_fallback'), COUNT(*)) AS pct_yfinance_fallback_dominance
   FROM `sunny-might-477607-p8.pyfinagent_data.data_source_events`
   WHERE DATE(event_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY);
   ```

## Closes audit basis
bucket 24.7 F-3 RESOLVED.

**Audit anchor for next bucket:** 25.C (Layer-1 28-skill output surfacing in drawer), 25.D / 25.N / 25.O (P2 backlog).
