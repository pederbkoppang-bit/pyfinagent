---
step: phase-25.H
cycle: 60
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_H.py'
title: Recent-analyses ticker dedup (5x SNDK fix) (P0)
---

# Experiment Results — phase-25.H

## Code change

`backend/db/bigquery_client.py:get_recent_reports`:

```diff
- SELECT ticker, company_name, analysis_date, final_score, recommendation, summary
- FROM `{self.reports_table}`
- ORDER BY analysis_date DESC
- LIMIT @limit
+ WITH ranked AS (
+     SELECT
+         ticker, company_name, analysis_date, final_score, recommendation, summary,
+         ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analysis_date DESC) AS rk
+     FROM `{self.reports_table}`
+ )
+ SELECT ticker, company_name, analysis_date, final_score, recommendation, summary
+ FROM ranked
+ WHERE rk = 1
+ ORDER BY analysis_date DESC
+ LIMIT @limit
```

CTE pattern (BigQuery standard SQL) — ranks per ticker by date, keeps rank=1 only, then applies LIMIT. ScalarQueryParameter preserved (no SQL injection regression).

## Verbatim verifier output

```
=== phase-25.H (recent-analyses ticker dedup) verifier ===
  [PASS] get_recent_reports_uses_row_number_partition_by_ticker
  [PASS] get_recent_reports_filters_to_rk_eq_1
  [PASS] phase_25_H_attribution_comment_present
  [PASS] get_recent_reports_signature_preserved
  [PASS] bigquery_client_py_syntax_clean
  [PASS] get_recent_reports_uses_parameterized_query_for_limit
PASS (6/6) EXIT=0
```

6/6 PASS.

## Hypothesis verdict
CONFIRMED. CTE-based dedup pattern matches phase-24.5 F-3 recommendation. Closes 5x SNDK morning-digest bug.

## Live-check
Per masterplan: "Slack morning digest Recent Analyses shows 5 distinct tickers". To be confirmed on next 06:00 ET digest post-deploy.

## Next phase
Q/A pending.
