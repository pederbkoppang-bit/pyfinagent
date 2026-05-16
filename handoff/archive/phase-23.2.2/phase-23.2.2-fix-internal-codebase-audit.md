# phase-23.2.2-fix Internal Codebase Audit

**Date:** 2026-04-29
**Tier:** simple (mechanical cleanup, pattern already proven in phase-23.1.15)
**Scope:** STX orphan trade — BUY with no matching paper_positions row

---

## 1. Phase-23.1.15 Cleanup Script Pattern

**File:** `/Users/ford/.openclaw/workspace/pyfinagent/scripts/cleanup_phase_23_1_15.py` (225 lines)

**Pattern confirmed:**

- Hard-code specific `trade_id` constants at the top of the script (no parameterization)
- Hard-code exact `total_value` and `transaction_cost` (fee) refund amounts
- `apply_changes()` executes three BigQuery DML steps:
  1. `DELETE FROM paper_trades WHERE trade_id = '<id>'`
  2. (additional DELETEs if multiple trades)
  3. `UPDATE paper_portfolio SET current_cash = current_cash + <amount>, updated_at = FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S+00:00', CURRENT_TIMESTAMP()) WHERE portfolio_id = 'default'`
- The `updated_at` column is a STRING field — it must receive `FORMAT_TIMESTAMP(...)`, NOT `CURRENT_TIMESTAMP()` directly (line 154 is the authoritative form; line 112 in the dry-run `show_sql()` uses `CURRENT_TIMESTAMP()` for display only)
- Idempotent: DELETE on an absent row is a no-op; refund is gated on `num_dml_affected_rows > 0` per trade deleted this run
- Modes: `--dry-run` (default), `--apply`, `--yes` (headless bypass)

**Key line anchors:**
- Lines 38-45: hard-coded trade IDs and refund values
- Lines 147-161: `apply_changes()` refund logic, gated on actual deletion
- Line 154: `FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S+00:00', CURRENT_TIMESTAMP())` — the exact string format for `updated_at`

---

## 2. BQ Orphan Scan — Full OUTER JOIN Reconciliation

**Query run:** FULL OUTER JOIN `paper_trades` (grouped by ticker+action) against `paper_positions`, filtering WHERE `cost_basis IS NULL AND action='BUY'`.

**Result:** Exactly ONE orphan ticker found.

| Ticker | Action | Count | total_value | transaction_cost | cost_basis |
|--------|--------|-------|-------------|------------------|------------|
| STX    | BUY    | 1     | $949.48     | $0.9500          | NULL (no position row) |

No other tickers have orphan BUY trades without a position row.

---

## 3. All-Ticker Buy/Sell/Position Cross-Check

All 15 tickers with BUY trades inspected:

| Ticker | buy_n | sell_n | buy_sv    | fee    | Position status |
|--------|-------|--------|-----------|--------|-----------------|
| CIEN   | 1     | 0      | $949.95   | $0.95  | HAS_POSITION    |
| COHR   | 1     | 0      | $1,445.89 | $1.45  | HAS_POSITION    |
| DELL   | 1     | 0      | $949.48   | $0.95  | HAS_POSITION    |
| FIX    | 1     | 0      | $1,319.41 | $1.32  | HAS_POSITION    |
| GEV    | 1     | 0      | $1,396.62 | $1.40  | HAS_POSITION    |
| GLW    | 1     | 0      | $949.95   | $0.95  | HAS_POSITION    |
| INTC   | 1     | 0      | $949.48   | $0.95  | HAS_POSITION    |
| KEYS   | 1     | 0      | $1,396.62 | $1.40  | HAS_POSITION    |
| LITE   | 1     | 0      | $949.95   | $0.95  | HAS_POSITION    |
| MU     | 1     | 0      | $531.55   | $0.53  | HAS_POSITION    |
| ON     | 1     | 0      | $472.36   | $0.47  | HAS_POSITION    |
| SNDK   | 1     | 0      | $949.95   | $0.95  | HAS_POSITION    |
| **STX**| **1** | **0**  | **$949.48** | **$0.95** | **NO_POSITION** |
| TER    | 1     | 0      | $949.48   | $0.95  | HAS_POSITION    |
| WDC    | 1     | 0      | $949.95   | $0.95  | HAS_POSITION    |

No ticker has more BUY trades than SELL trades with a partial-close orphan pattern (all have 0 SELLs, exactly 1 BUY, and exactly 1 position row — except STX). No additional orphans found.

---

## 4. STX Trade Row — Full Detail

Queried directly from `paper_trades`:

| Field             | Value                                  |
|-------------------|----------------------------------------|
| trade_id          | `04c6f356-2a5c-47df-8891-bea686cd444f` |
| ticker            | STX                                    |
| action            | BUY                                    |
| quantity          | 1.619582                               |
| price             | $586.25                                |
| total_value       | $949.48                                |
| transaction_cost  | $0.9500                                |
| created_at        | 2026-04-26T23:44:59.884456+00:00       |
| reason            | new_buy_signal                         |

No matching row in `paper_positions` for STX. Confirmed the pre-MERGE-upsert legacy artifact: cash debit and trade row were written, but the cycle crashed before the position upsert.

---

## 5. Cleanup Actions Required

### A. Trade ID(s) to DELETE

| # | trade_id                               | Ticker | Reason              |
|---|----------------------------------------|--------|---------------------|
| 1 | `04c6f356-2a5c-47df-8891-bea686cd444f` | STX    | no matching position row; orphan BUY from 2026-04-26T23:44 |

### B. Refund per Ticker

| Ticker | total_value | transaction_cost | Total Refund |
|--------|-------------|------------------|--------------|
| STX    | $949.48     | $0.95            | **$950.43**  |

### C. Total Cumulative Refund

**$950.43** (single trade, single ticker)

### D. Current Cash Context

Current `paper_portfolio.current_cash` = **$825.66** (as of 2026-05-04T18:30:43+00:00, after phase-23.1.15 applied its $1,451.40 refund).

Post-cleanup expected cash: $825.66 + $950.43 = **$1,776.09**

---

## 6. Cleanup SQL (to be implemented in `scripts/cleanup_phase_23_2_2.py`)

Following the exact phase-23.1.15 pattern:

```sql
-- Step 1: DELETE the orphan STX trade
DELETE FROM paper_trades
WHERE trade_id = '04c6f356-2a5c-47df-8891-bea686cd444f';

-- Step 2: Refund $950.43 to current_cash (only if deletion occurred)
UPDATE paper_portfolio
SET current_cash = current_cash + 950.43,
    updated_at = FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S+00:00', CURRENT_TIMESTAMP())
WHERE portfolio_id = 'default';
```

Note: `updated_at` is a STRING column — must use `FORMAT_TIMESTAMP(...)`, not raw `CURRENT_TIMESTAMP()`.

---

## 7. Internal Files Inspected

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/cleanup_phase_23_1_15.py` | 225 | Prior orphan cleanup script (WDC+XOM) | Read in full; pattern confirmed |
| `paper_trades` BQ table | N/A | Trade ledger | Queried; STX orphan confirmed |
| `paper_positions` BQ table | N/A | Open positions | Queried; STX absent confirmed |
| `paper_portfolio` BQ table | N/A | Cash ledger | Queried; current_cash=$825.66 |

---

## Summary

- **Exactly 1 orphan:** STX `04c6f356-2a5c-47df-8891-bea686cd444f`, $949.48 + $0.95 fee
- **No other orphans found** across all 15 tickers
- **Refund amount:** $950.43
- **Post-cleanup cash:** $1,776.09
- **Pattern:** identical to phase-23.1.15 (DELETE trade + UPDATE cash with FORMAT_TIMESTAMP)
