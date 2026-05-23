# Internal Codebase Audit — phase-23.2.6-fix
# paper_positions sector column migration

Generated: 2026-04-29

---

## A. Confirm paper_positions has NO `sector` column

**Schema source:** `scripts/migrations/migrate_paper_trading.py` — the canonical table
definition for `paper_positions` (lines 36-51).

```python
PAPER_POSITIONS_SCHEMA = [
    bigquery.SchemaField("position_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("quantity", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("avg_entry_price", "FLOAT64", mode="REQUIRED"),
    bigquery.SchemaField("cost_basis", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("current_price", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("market_value", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("unrealized_pnl", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("unrealized_pnl_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("entry_date", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("last_analysis_date", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("recommendation", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("risk_judge_position_pct", "FLOAT64", mode="NULLABLE"),
    bigquery.SchemaField("stop_loss_price", "FLOAT64", mode="NULLABLE"),
]
```

**Confirmed: no `sector` column.** 14 columns, none named `sector`.

To verify live schema:
```sql
SELECT column_name, data_type, is_nullable
FROM `sunny-might-477607-p8.financial_reports.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'paper_positions'
ORDER BY ordinal_position;
```

---

## B. Existing schema migration pattern

Two canonical patterns observed in the project:

### Pattern 1: Create-if-not-exists (migrate_paper_trading.py lines 102-109)
Used for entire-table creation. Catches exception on `get_table()`, creates on miss.

### Pattern 2: ALTER TABLE ADD COLUMN IF NOT EXISTS (add_delisted_at_column.py — full file)
This is the correct pattern for adding a single column to an existing table.

```python
sql = f"""
ALTER TABLE `{PROJECT}.{DATASET}.{TABLE}`
ADD COLUMN IF NOT EXISTS delisted_at DATE
OPTIONS(description='...')
"""
client.query(sql).result(timeout=30)
```

Key characteristics of Pattern 2:
- Uses `ADD COLUMN IF NOT EXISTS` — idempotent (safe to re-run)
- Includes `OPTIONS(description=...)` for discoverability
- Uses `client.query(sql).result(timeout=30)` — respects the 30s timeout rule
- Supports `--dry-run` argparse flag
- Uses ADC credentials (no SA JSON needed locally — `bigquery.Client(project=PROJECT)`)
- Column must be NULLABLE (BigQuery rejects REQUIRED on existing tables)
- Does NOT backfill — comments explicitly note backfill is a separate step

The `add_delisted_at_column.py` file is the exact template to follow for this migration.

---

## C. execute_buy fields and where to add `sector`

**File:** `backend/services/paper_trader.py`

`execute_buy` signature (lines 69-80):
```python
def execute_buy(
    self, ticker, amount_usd, price, reason="new_buy_signal",
    analysis_id="", risk_judge_decision="", stop_loss_price=None,
    risk_judge_position_pct=None, signals=None,
) -> Optional[dict]:
```

New-position dict built at lines 191-207:
```python
pos_row = {
    "position_id": str(uuid.uuid4()),
    "ticker": ticker,
    "quantity": round(quantity, 6),
    "avg_entry_price": price,
    "cost_basis": round(amount_usd, 2),
    "current_price": price,
    "market_value": round(amount_usd, 2),
    "unrealized_pnl": 0.0,
    "unrealized_pnl_pct": 0.0,
    "entry_date": now,
    "last_analysis_date": now,
    "recommendation": reason,
    "risk_judge_position_pct": risk_judge_position_pct,
    "stop_loss_price": stop_loss_price,
}
self.bq.save_paper_position(pos_row)
```

The add-to-existing-position path (lines 174-190) also calls `save_paper_position`.
Both branches need `sector` added.

`save_paper_position` uses a MERGE on `ticker` (bigquery_client.py lines 553-592).
It drops None values before building the query (line 564: `row = {k: v for k, v in row.items() if v is not None}`).
This means adding `"sector": sector_value` to pos_row will flow through automatically when sector is non-None;
a None sector is silently omitted (column stays NULL in BQ) — safe behavior.

**Where to add `sector` parameter:** add `sector: Optional[str] = None` to `execute_buy`
signature, then include `"sector": sector` in both pos_row dicts (new and existing position).
The None-drop in save_paper_position handles missing sector gracefully.

---

## D. Where sector is ALREADY available at execute_buy call time

**autonomous_loop.py lines 168-192 (Step 1 — candidate enrichment):**
After `rank_candidates()`, `_fetch_ticker_meta` is called on all top-N candidates.
The result is merged into each candidate dict: `c["sector"] = sector` (line 187).

**autonomous_loop.py lines 371-378 (Step 6 — decide_trades call):**
```python
candidates_by_ticker = {c["ticker"]: c for c in candidates if c.get("ticker")}
orders = decide_trades(
    current_positions=positions,
    candidate_analyses=candidate_analyses,
    ...
    candidates_by_ticker=candidates_by_ticker,
)
```

**portfolio_manager.py lines 157-163 (inside decide_trades buy candidates loop):**
```python
cand_sector = ""
if screener_candidate:
    cand_sector = screener_candidate.get("sector") or ""
if not cand_sector:
    full_report = analysis.get("full_report") or {}
    md = full_report.get("market_data") or {}
    cand_sector = md.get("sector") or analysis.get("sector") or ""
buy_candidates.append({
    ...
    "sector": cand_sector or "Unknown",
    ...
})
```

**TradeOrder** (portfolio_manager.py lines 17-30): does NOT currently carry a `sector` field.

**autonomous_loop.py lines 415-426 (execute_buy call):**
```python
trade = await asyncio.to_thread(
    trader.execute_buy,
    ticker=order.ticker,
    amount_usd=order.amount_usd or 0,
    price=price,
    reason=order.reason,
    analysis_id=order.analysis_id,
    risk_judge_decision=order.risk_judge_decision,
    stop_loss_price=order.stop_loss_price,
    risk_judge_position_pct=order.risk_judge_position_pct,
    signals=order.signals,
)
```

**Sector is NOT passed to execute_buy today.** The sector sits in `buy_candidates` list
inside `decide_trades` but is never propagated to `TradeOrder`, so it is lost before
`execute_buy` is called.

**Fix chain required:**
1. Add `sector: str = ""` field to `TradeOrder` dataclass (portfolio_manager.py line 29).
2. Populate it in `decide_trades` at the `orders.append(TradeOrder(...))` call (line 234).
3. Pass `sector=order.sector` in the `execute_buy` call in autonomous_loop.py (line 415).
4. Add `sector: Optional[str] = None` parameter to `execute_buy` and include in both pos_row branches.

---

## E. Dependent code that reads paper_positions.sector

Searched: `grep -rn "sector" backend/` + checked all paper_positions read paths.

**Current readers:**
- `bigquery_client.py get_paper_positions()` (line 535): `SELECT *` — will automatically return `sector` once the column exists.
- `bigquery_client.py get_paper_position()` (line 542): `SELECT *` — same.
- `paper_trading.py GET /portfolio` (lines 175-211): builds `sector_breakdown` by calling `_fetch_ticker_meta` on each position ticker, **ignoring** the BQ row's sector field entirely.
- `autonomous_loop.py lines 330-366` (phase-23.1.14 enrichment block): checks `p.get("sector")` on position dicts returned from `get_paper_positions()`. Once the column is populated in BQ, this block becomes a no-op for enriched rows.
- `portfolio_manager.py decide_trades lines 196-200`: reads `pos.get("sector")` to build sector_counts from current positions.

**No external reader currently depends on sector being present** — all current code
uses the in-memory enrichment fallback from phase-23.1.13/14. The column migration
enables the data-is-already-there path; the in-memory enrichment remains as safety net.

**paper_trading.py sector_breakdown (lines 181-211):** This endpoint reads `_fetch_ticker_meta`
regardless of whether BQ has sector. Post-migration this could optionally read `p.get("sector")`
from the BQ row directly (eliminating the meta call for the breakdown), but that is a
separate optimization, not required for this fix.

---

## Summary of files to touch

| File | Change | Lines |
|------|--------|-------|
| `scripts/migrations/add_sector_to_paper_positions.py` | NEW — ALTER TABLE + backfill | (new file) |
| `backend/services/portfolio_manager.py` | Add `sector` field to `TradeOrder` | ~29 |
| `backend/services/portfolio_manager.py` | Populate sector in `decide_trades` orders.append | ~234 |
| `backend/services/paper_trader.py` | Add `sector` param to `execute_buy`, include in pos_row | ~80, ~191-207 |
| `backend/services/autonomous_loop.py` | Pass `sector=order.sector` in execute_buy call | ~415-426 |
| `backend/services/autonomous_loop.py` | Phase-23.1.14 enrichment block — keep as fallback, add BQ write-back | ~330-366 (optional enhancement) |

No frontend changes required — sector_breakdown endpoint already works.
No BigQuery client changes required — save_paper_position MERGE handles new columns dynamically.
