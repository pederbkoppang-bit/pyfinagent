# Sprint Contract — phase-25.H — Recent-analyses ticker dedup

**Cycle:** phase-25 cycle 4
**Date:** 2026-05-12
**Step ID:** 25.H
**Priority:** P0

## Research-gate
Reuses phase-24.5 cycle 4 researcher gate (5 sources). Fix verbatim per audit F-3.

## Hypothesis
ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analysis_date DESC) + WHERE rk=1 dedup will return at most one row per ticker before LIMIT.

## Plan
1. Edit `backend/db/bigquery_client.py:257-268` — wrap SELECT in CTE with ROW_NUMBER + WHERE rk=1
2. New verifier `tests/verify_phase_25_H.py` (6 claims incl. SQL injection check)
3. experiment_results.md
4. Q/A
5. harness_log Cycle 60
6. Flip 25.H

## References
- `docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md` F-3
- `backend/db/bigquery_client.py:257-268`
