# Phase-23.1.18 Internal Codebase Audit
## Red Line Monitor NAV Duplicate Rows

---

## 1. `save_paper_snapshot` — INSERT Pattern Confirmed

**File:** `backend/db/bigquery_client.py:669-685`

```python
def save_paper_snapshot(self, row: dict) -> None:
    """Insert snapshot via DML to avoid streaming buffer conflicts."""
    row = {k: v for k, v in row.items() if v is not None}
    table = self._pt_table("paper_portfolio_snapshots")
    cols = ", ".join(row.keys())
    vals = ", ".join(f"@v_{k}" for k in row.keys())
    query = f"INSERT INTO `{table}` ({cols}) VALUES ({vals})"
    ...
    self.client.query(query, job_config=job_config).result()
```

**Verdict:** Plain `INSERT INTO` with parameterised values. No MERGE, no upsert, no deduplication guard. Every call unconditionally appends a new row regardless of whether one already exists for that `snapshot_date`. This is the root cause of the duplicate rows confirmed in BQ.

---

## 2. Callers of `save_paper_snapshot`

All callers go through `paper_trader.py::save_daily_snapshot`, which is the only direct caller of `bq.save_paper_snapshot`. The call chain is:

| Caller file | Line | Trigger |
|---|---|---|
| `backend/services/paper_trader.py` | 449 | `save_daily_snapshot()` wrapper |
| `backend/services/autonomous_loop.py` | 309 | End-of-day scheduled snapshot (no trades cycle) |
| `backend/services/autonomous_loop.py` | 424 | End-of-loop snapshot (trades cycle) |
| `scripts/repair_phase_23_1_17.py` | 76 | One-shot repair script |

The repair script (`scripts/repair_phase_23_1_17.py:76`) called `save_daily_snapshot` as its fix for phase-23.1.17, which is exactly why 2026-04-29 now has TWO rows (the stale row from the original autonomous loop run + the repair script's fresh row).

The autonomous loop can also call `save_daily_snapshot` twice in a session (lines 309 and 424) if certain code paths overlap, explaining why some dates have THREE rows.

---

## 3. `save_daily_snapshot` — Row Shape

**File:** `backend/services/paper_trader.py:421-450`

The wrapper:
1. Calls `self.get_or_create_portfolio()` and `self.get_positions()` to compute live NAV.
2. Calls `self.bq.get_paper_snapshots(limit=1)` to get `prev_nav` for daily P&L calculation.
3. Constructs the `snap` dict with 11 fields:
   - `snapshot_date` (STRING, `YYYY-MM-DD`, UTC)
   - `total_nav` (FLOAT64)
   - `cash` (FLOAT64)
   - `positions_value` (FLOAT64)
   - `daily_pnl_pct` (FLOAT64)
   - `cumulative_pnl_pct` (FLOAT64)
   - `benchmark_pnl_pct` (FLOAT64)
   - `alpha_pct` (FLOAT64)
   - `position_count` (INT64)
   - `trades_today` (INT64)
   - `analysis_cost_today` (FLOAT64)
4. Calls `self.bq.save_paper_snapshot(snap)`.

No `created_at` field is passed. The migration schema (`scripts/migrations/migrate_paper_trading.py:69-81`) confirms no `created_at` column exists in `paper_portfolio_snapshots` — unlike `paper_trades` which does have `created_at` (line 65). This is the missing de-ordering column.

---

## 4. `_fetch_snapshots` and `_forward_fill_calendar` — ANY_VALUE Bug Confirmed

**File:** `backend/api/sovereign_api.py:122-183`

```python
sql = f"""
  SELECT
    snapshot_date AS d,
    ANY_VALUE(total_nav) AS nav
  FROM `{_GCP_PROJECT}.financial_reports.paper_portfolio_snapshots`
  WHERE PARSE_DATE('%Y-%m-%d', snapshot_date)
        >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
  GROUP BY snapshot_date
  ORDER BY snapshot_date
"""
```

`ANY_VALUE` is documented by Google as non-deterministic — it returns any value from the group with no guarantee of recency. In practice BQ appears to favour the first-inserted row (empirically the stale $14,153.03 for 2026-04-29 rather than the repair's $15,647.74).

`_forward_fill_calendar` (lines 150-183) is clean — it has no pick logic of its own, it only forward-fills calendar gaps from whatever `_fetch_snapshots` returns. The bug is entirely in `_fetch_snapshots`.

**Is `ANY_VALUE` the ONLY bad pick?** Yes. `_forward_fill_calendar` uses a simple dict lookup by date string (line 160: `by_day = {row["d"]: float(row["nav"]) for row in snapshots}`). One row per date after the GROUP BY, so no further non-determinism downstream.

---

## 5. Test Coverage

| File | Covers snapshots? | Notes |
|---|---|---|
| `tests/verify_phase_23_1_17.py` | Indirectly | Asserts `save_daily_snapshot` is called in repair script; no BQ round-trip |
| `tests/test_retired_models.py` | No | Only mentions `snapshot` in passing |
| `tests/slack_bot/test_scheduler_wiring_phase991.py` | No | Scheduler wiring only |

**No test exercises** `save_paper_snapshot`, `_fetch_snapshots`, or the ANY_VALUE pick. The duplicate-row scenario has zero test coverage. This is a gap to fill in phase-23.1.18.

---

## 6. Migration Script Analysis

**File:** `scripts/migrations/migrate_paper_trading.py:68-88`

`PAPER_SNAPSHOTS_SCHEMA` defines 11 columns (snapshot_date through analysis_cost_today). There is:
- No `PRIMARY KEY` constraint (BQ doesn't enforce PKs)
- No `UNIQUE` constraint (BQ has no column-level unique constraint)
- No `created_at` column (unlike `paper_trades` which has `created_at STRING REQUIRED` at line 65)
- No intent comment about uniqueness per snapshot_date

The migration is create-only (lines 102-108): it calls `client.get_table(ref)` and skips if the table exists. There is no ALTER, no constraint DDL, no upsert logic. The schema was never designed with idempotent writes in mind.

---

## 7. Fix Sketches

### Fix A — Convert `save_paper_snapshot` to MERGE on snapshot_date

**Patch location:** `backend/db/bigquery_client.py:669-685`

```python
def save_paper_snapshot(self, row: dict) -> None:
    """Upsert snapshot keyed on snapshot_date — idempotent via MERGE."""
    row = {k: v for k, v in row.items() if v is not None}
    table = self._pt_table("paper_portfolio_snapshots")
    # Build SET clause for WHEN MATCHED UPDATE
    non_key_cols = [k for k in row.keys() if k != "snapshot_date"]
    set_clause = ", ".join(f"T.{k} = S.{k}" for k in non_key_cols)
    cols = ", ".join(row.keys())
    src_vals = ", ".join(f"@v_{k}" for k in row.keys())
    query = f"""
        MERGE `{table}` AS T
        USING (SELECT {src_vals}) AS S({cols})
        ON T.snapshot_date = S.snapshot_date
        WHEN MATCHED THEN
            UPDATE SET {set_clause}
        WHEN NOT MATCHED THEN
            INSERT ({cols}) VALUES ({src_vals})
    """
    params = []
    for k, v in row.items():
        if isinstance(v, float):
            params.append(bigquery.ScalarQueryParameter(f"v_{k}", "FLOAT64", v))
        elif isinstance(v, int):
            params.append(bigquery.ScalarQueryParameter(f"v_{k}", "INT64", v))
        else:
            params.append(bigquery.ScalarQueryParameter(f"v_{k}", "STRING", str(v)))
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    self.client.query(query, job_config=job_config).result()
```

**For:** Eliminates duplicate creation going forward. One row per day guaranteed. Same pattern as the phase-23.1.15 MERGE that fixed `save_paper_position`. Backwards compatible — MERGE behaves identically to INSERT for new dates.

**Against:** MERGE is slightly more expensive than INSERT (scans existing rows for the ON clause match). For a <100-row table this is negligible — sub-second at any reasonable BQ pricing.

---

### Fix B — One-shot cleanup script `scripts/cleanup_phase_23_1_18.py`

**Strategy:** For each `snapshot_date` with duplicate rows, keep the row with the highest `total_nav` (heuristic: the repair/most-recent snapshot always has the highest NAV since it was computed after position re-valuation). Delete all other rows.

**Two-mode skeleton:**

```python
#!/usr/bin/env python3
"""Phase-23.1.18: deduplicate paper_portfolio_snapshots.
Usage:
    python scripts/cleanup_phase_23_1_18.py --dry-run
    python scripts/cleanup_phase_23_1_18.py --apply
"""
import argparse, sys
from google.cloud import bigquery

PROJECT = "sunny-might-477607-p8"
TABLE = f"{PROJECT}.financial_reports.paper_portfolio_snapshots"
DEDUP_SQL = f"""
CREATE OR REPLACE TABLE `{TABLE}` AS
SELECT * EXCEPT(_row_num)
FROM (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY snapshot_date
            ORDER BY total_nav DESC
        ) AS _row_num
    FROM `{TABLE}`
)
WHERE _row_num = 1
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.apply:
        print("Must pass --dry-run or --apply"); sys.exit(1)

    client = bigquery.Client(project=PROJECT)
    # Preview duplicates
    count_sql = f"""
        SELECT snapshot_date, COUNT(*) AS cnt
        FROM `{TABLE}`
        GROUP BY snapshot_date
        HAVING cnt > 1
        ORDER BY snapshot_date
    """
    rows = list(client.query(count_sql).result())
    if not rows:
        print("No duplicates found — table is clean."); return
    print("Duplicate snapshot_dates:")
    for r in rows:
        print(f"  {r['snapshot_date']}: {r['cnt']} rows")

    if args.dry_run:
        print("DRY RUN — no changes made.")
        return
    print("Applying deduplication...")
    client.query(DEDUP_SQL).result()
    print("Done. Table rebuilt with one row per snapshot_date (max total_nav kept).")

if __name__ == "__main__":
    main()
```

**For:** Cleans existing dirty data. Idempotent — safe to re-run. `CREATE OR REPLACE TABLE ... AS SELECT` is the canonical BQ small-table deduplication pattern (no streaming-buffer conflict since we use DML/DDL not streaming). `ORDER BY total_nav DESC` keeps the highest-value (post-repair) row.

**Against:** `CREATE OR REPLACE TABLE` replaces the table object, which loses any table-level metadata (labels, expiry). For this tiny table with no such metadata, this is not a concern.

**ORDER BY heuristic justification:** No `created_at` exists. `total_nav DESC` is a strong proxy because: (a) mark_to_market runs before every snapshot, so the repair snapshot always reflects the most complete position valuation; (b) the stale rows from autonomous_loop runs before phase-23.1.15's cash refund have lower NAV. This may not hold in future if NAV genuinely drops intraday — but Fix A (MERGE going forward) means there will be no future duplicates to worry about.

---

### Fix C — Defensive update of `_fetch_snapshots`

**Patch location:** `backend/api/sovereign_api.py:130-138`

Change:
```sql
ANY_VALUE(total_nav) AS nav
```
To:
```sql
MAX(total_nav) AS nav
```

**For:** Defence in depth. If Fix B is ever missed on a new environment, or if an edge-case duplicate slips through before Fix A is deployed, the chart still shows the highest (most-complete) NAV. `MAX` is deterministic and cheap. One-line change.

**Against:** Technically `MAX` could show an anomalously high intraday snapshot if NAV later corrects downward. In practice this is not a concern for a paper-trading dashboard where we want the "current best estimate." `MAX` is a deliberate semantic choice (latest-known-value is always the highest during normal operation).

**Recommendation:** Include. The defence cost is one SQL token change.

---

### Fix D (deferred) — Add `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()`

**Migration approach** (two-step required per BQ docs):
```sql
-- Step 1: add column without default (BQ restriction)
ALTER TABLE `sunny-might-477607-p8.financial_reports.paper_portfolio_snapshots`
ADD COLUMN created_at TIMESTAMP;

-- Step 2: set default on new rows
ALTER TABLE `...paper_portfolio_snapshots`
ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP();
```

Existing rows get NULL for `created_at`. New rows get the ingestion timestamp automatically. Existing INSERT statements that omit `created_at` are backwards-compatible — BQ applies the default automatically.

**For:** Makes `ROW_NUMBER() OVER (PARTITION BY snapshot_date ORDER BY created_at DESC)` deterministic for any future deduplication. Richer audit trail.

**Against:** Does not help for the existing duplicate rows (they all get NULL created_at). The two-step migration adds complexity. With Fix A (MERGE) deployed, no future duplicates will be created, making this column largely moot. Adds a nullable TIMESTAMP column to every snapshot row forever.

**Recommendation:** Defer. Fix A + B + C solves the problem cleanly. Add created_at only if there's a future audit or compliance requirement for ingestion timestamps.

---

## 8. Recommended Fix Combination

**Apply A + B + C in order:**

1. **Fix B first** (cleanup script) — run `--apply` to deduplicate existing rows before deploying the MERGE change. This gives a known-clean baseline.
2. **Fix A** — convert `save_paper_snapshot` to MERGE. Prevents all future duplicates.
3. **Fix C** — change `ANY_VALUE` to `MAX` in `_fetch_snapshots`. One-line safety net.

This matches the stated preference (A + B + C). Fix D deferred.

---

## Internal File Inventory

| File | Lines read | Role | Status |
|---|---|---|---|
| `backend/db/bigquery_client.py` | 669-696 | `save_paper_snapshot` INSERT + `get_paper_snapshots` | BUG: plain INSERT, no upsert |
| `backend/services/paper_trader.py` | 421-450 | `save_daily_snapshot` wrapper, row shape | No created_at, calls bq.save_paper_snapshot |
| `backend/api/sovereign_api.py` | 115-183 | `_fetch_snapshots` + `_forward_fill_calendar` | BUG: ANY_VALUE non-deterministic |
| `backend/services/autonomous_loop.py` | 309, 424 | Two call sites for `save_daily_snapshot` | Can double-write in same UTC day |
| `scripts/repair_phase_23_1_17.py` | 58, 76 | Phase-23.1.17 repair — added a third row | Root cause of 2026-04-29 duplicate |
| `scripts/migrations/migrate_paper_trading.py` | 68-88 | Schema definition for `paper_portfolio_snapshots` | No unique constraint, no created_at |
| `tests/verify_phase_23_1_17.py` | 1-67 | Verification test for phase-23.1.17 | No BQ snapshot round-trip coverage |
