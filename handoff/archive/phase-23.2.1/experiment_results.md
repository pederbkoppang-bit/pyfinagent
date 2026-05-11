---
step: phase-23.2.1
title: Verify autonomous loop ran daily for 7+ days — experiment results
date: 2026-05-07
verdict_class: HYPOTHESIS_DISCONFIRMED (verification succeeded; system finding to log)
verification_command: 'source .venv/bin/activate && python tests/verify_phase_23_2_1.py'
---

# Experiment Results — phase-23.2.1

## What was done

This is a verification-only step. **No code changes** to backend or
frontend. Two artifacts were produced:

1. `handoff/current/experiment_results.md` (this file).
2. `tests/verify_phase_23_2_1.py` — replayable verifier that runs the
   immutable BQ query via the Python `google.cloud.bigquery` client
   (MCP fallback per CLAUDE.md `BigQuery Access (MCP)::point 6`,
   since the pinned MCP defaults to `US` location and
   `paper_portfolio_snapshots` lives in `us-central1`).

## Verification command — verbatim from `.claude/masterplan.json::23.2.1`

```
bq SELECT DATE(snapshot_date), COUNT(*) FROM paper_portfolio_snapshots
WHERE PARSE_DATE('%Y-%m-%d', snapshot_date) >= DATE_SUB(CURRENT_DATE(),
INTERVAL 9 DAY) GROUP BY 1 ORDER BY 1; expect ~9 rows, no gaps
```

Translated to Python BQ client and executed against
`sunny-might-477607-p8.financial_reports.paper_portfolio_snapshots`
(location `us-central1`):

```sql
SELECT DATE(PARSE_DATE('%Y-%m-%d', snapshot_date)) AS day,
       COUNT(*) AS n
FROM `sunny-might-477607-p8.financial_reports.paper_portfolio_snapshots`
WHERE PARSE_DATE('%Y-%m-%d', snapshot_date) >= DATE_SUB(CURRENT_DATE(),
                                                        INTERVAL 9 DAY)
GROUP BY 1
ORDER BY 1
```

(The translation wraps `snapshot_date` in `PARSE_DATE` for the SELECT
column too, since `DATE(<STRING>)` is not a valid coercion in BQ.
Identical filter; identical row set.)

## Verbatim result (run 2026-05-07, today=2026-05-07 UTC)

```
day          |   n
--------------------
2026-04-28   |   1
2026-04-29   |   1
2026-05-04   |   1
2026-05-05   |   1
2026-05-06   |   1
-- TOTAL ROWS: 5 --
```

Gap analysis (window = 2026-04-28 → 2026-05-07, 10 calendar days):

| Date | DoW | Snapshot present | Notes |
|------|-----|------------------|-------|
| 2026-04-28 | Tue | yes | |
| 2026-04-29 | Wed | yes | |
| 2026-04-30 | Thu | **MISSING** | weekday gap |
| 2026-05-01 | Fri | **MISSING** | weekday gap |
| 2026-05-02 | Sat | **MISSING** | weekend (expected if loop is market-day-aware) |
| 2026-05-03 | Sun | **MISSING** | weekend (expected if loop is market-day-aware) |
| 2026-05-04 | Mon | yes | |
| 2026-05-05 | Tue | yes | (cycle_history.jsonl shows status=running, stale, but snapshot was written) |
| 2026-05-06 | Wed | yes | |
| 2026-05-07 | Thu | **MISSING** | today; cron at `paper_trading_hour:00` may not have fired yet at run time |

Cross-check vs `handoff/cycle_history.jsonl` (researcher table):
- Cycles completed on 04-28, 04-29, 05-06 — all have snapshots.
- 2026-05-05 cycle status was "running" (stale) — snapshot exists,
  meaning Step 8 of `run_daily_cycle` DID execute despite the stale
  state in `cycle_history.jsonl`. The bug is in the heartbeat /
  history-flush, not the snapshot write.
- 2026-05-04 cycle: no entry in `cycle_history.jsonl` but a snapshot
  exists. Either the cycle ran via a different code path (e.g.
  `adjust_cash_and_mtm`'s call site at `paper_trader.py:524`) or
  `cycle_history.jsonl` is missing entries.

## Verdict against the immutable criterion

> "expect ~9 rows, no gaps"

Actual: **5 rows, 5 gaps (2 weekends + 3 weekdays incl. today)**.

The criterion is NOT met. Per the contract's anti-pattern guard
("rewriting immutable criteria"), the criterion stands as written.
The result is a real-world finding: **the autonomous loop did not
run daily for the last 9 days**. There are 2-3 legitimate weekday
gaps (2026-04-30, 2026-05-01, possibly 2026-05-07 if it just hasn't
fired yet) and 2 weekend gaps that may or may not be expected
depending on the loop's intended cadence.

## Findings to surface to the operator

1. **3 weekday gaps in 9 days** (04-30 Thu, 05-01 Fri, 05-07 Thu /
   today). 05-07 may be a false positive if the cron runs late in
   the UTC day — record this run-time and re-check after
   `paper_trading_hour:00 UTC`.
2. **`cycle_history.jsonl` is incomplete vs BQ ground truth.**
   2026-05-04 has a BQ snapshot but no `cycle_history.jsonl` entry;
   2026-05-05 has a snapshot but a stale "running" entry. The
   heartbeat / history-flush path is divergent from the BQ-write
   path. Whichever owns the source of truth for "did the loop run"
   needs reconciling — recommend a follow-up step in 23.2 or 23.3.
3. **Idempotency holds** — every present date has exactly `n=1`,
   confirming the MERGE-on-`snapshot_date` since phase-23.1.18
   prevents duplicates.

## What this step does NOT do

- Diagnose the gaps. Out of scope per the contract.
- Backfill snapshots. Operator action, gated on a separate phase.
- Restart the autonomous loop. Operator action.
- Modify the verification criterion. Forbidden per Anthropic
  immutable-criteria doctrine.

## Artifact files

- `handoff/current/experiment_results.md` — this file.
- `handoff/current/contract.md` — phase-23.2.1 contract.
- `handoff/current/phase-23.2.1-research-brief.md` — researcher output.
- `tests/verify_phase_23_2_1.py` — replayable Python verifier.

## How to re-run the verification

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
source .venv/bin/activate
python tests/verify_phase_23_2_1.py
```

The script exits **0** on a successful BQ query regardless of row
count (the row count is the finding, not a verifier failure mode).
It exits **non-zero** only if the BQ query itself errors (network,
auth, permissions) — that's the only condition that means
"verification could not run." Q/A's deterministic check leg should
treat exit 0 as "the check ran"; the LLM-judgment leg evaluates
whether the data matched the hypothesis.
