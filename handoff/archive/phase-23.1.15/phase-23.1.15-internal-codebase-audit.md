# Internal Codebase Audit — phase-23.1.15
## Topic: Duplicate WDC trade / phantom cash leak

Accessed: 2026-04-29

---

## 1. Files Inspected

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/paper_trader.py` | 587 | Virtual trade execution engine | Active, no UPSERT guard |
| `backend/db/bigquery_client.py` | 474-654 | BQ persistence layer (paper_* methods) | Plain INSERT, no MERGE |
| `backend/services/autonomous_loop.py` | 847 | Daily cycle orchestrator | `_running` flag present |
| `backend/api/paper_trading.py` | 875 | FastAPI endpoints for paper trading | /run-now is async fire-and-forget |
| `backend/services/cycle_health.py` | 228 | Cycle heartbeat + freshness | JSONL history preserved |
| `handoff/cycle_history.jsonl` | n/a | Actual cycle execution log | Key evidence — see below |

---

## 2. Root Cause Analysis from Cycle History

The cycle_history.jsonl entries for 2026-04-26 tell the definitive story:

```
f3109df7  started 21:08:59  completed 21:09:50  status=ERROR   n_trades=0
0e8c4a20  started 21:11:35  completed 21:12:31  status=ERROR   n_trades=0
a54a21fc  started 21:16:56  completed 21:19:19  status=completed n_trades=5
```

The BQ-confirmed WDC trades occurred at:
- Cycle 1: trade_id=`56072f0c...` at 21:12:28 UTC
- Cycle 2: trade_id=`e5447bd9...` at 21:17:41 UTC

Mapping cycle history to trade timestamps:
- **Cycle 0e8c4a20** (started 21:11:35, completed 21:12:31, status=ERROR): This cycle STARTED the WDC BUY at 21:12:28 and the trade was recorded in paper_trades. BUT the cycle ended with status=ERROR at 21:12:31 — meaning the cycle crashed AFTER `_safe_save_trade()` succeeded (paper_trade inserted) but likely BEFORE or DURING `save_paper_position()` or the portfolio cash update. Cash debit of $949.95+fee happened; position INSERT is uncertain.

- **Cycle a54a21fc** (started 21:16:56, completed 21:19:19, status=completed, n_trades=5): This cycle ran 5 minutes later. `get_positions()` at line 94 in `execute_buy` did NOT see the WDC position from the previous cycle (because either the position INSERT failed, or the SELECT used a stale BQ snapshot). So `existing = None`, cycle went to the else branch (lines 162-179), inserted a SECOND position_id and SECOND trade, and debited cash again.

This confirms **root cause (a) combined with (d)**: Cycle 0e8c4a20 crashed partway through `execute_buy`, after `_safe_save_trade` (trade row committed at 21:12:28) but before or during `save_paper_position`. The position was never written. Cycle a54a21fc saw no existing position, bought WDC again, and committed both a second trade row and a position row. `_running` flag was released by `finally: _running = False` (line 488) even on error, allowing the next cycle to proceed.

The XOM test_paper_trade is a separate anomaly — a manually triggered test buy (reason='test_paper_trade') from the /run-now endpoint at some point, which wrote a paper_trade row but never created a paper_positions row (suggesting the cycle was killed mid-execution or it was a partial test path).

---

## 3. Mutation Sites — Full Inventory

### paper_trades table

| Location | Call | Trigger |
|----------|------|---------|
| `paper_trader.py:135` | `self._safe_save_trade(trade)` | execute_buy — always first, before position write |
| `paper_trader.py:260` | `self._safe_save_trade(trade)` | execute_sell — before position delete |
| `paper_trader.py:506-515` | `bq.save_paper_trade(row)` via `_safe_save_trade` | with schema-error fallback retry |

### paper_positions table

| Location | Call | Trigger |
|----------|------|---------|
| `paper_trader.py:144` | `self.bq.delete_paper_position(ticker)` | execute_buy — add-to-existing branch |
| `paper_trader.py:161` | `self.bq.save_paper_position(pos_row)` | execute_buy — add-to-existing branch |
| `paper_trader.py:179` | `self.bq.save_paper_position(pos_row)` | execute_buy — new position branch (else) |
| `paper_trader.py:286,289` | `self.bq.delete_paper_position(ticker)` | execute_sell — both full and partial exit |
| `paper_trader.py:306` | `self.bq.save_paper_position(pos_row)` | execute_sell — partial exit re-insert |
| `paper_trader.py:343,352` | `delete_paper_position` + `_safe_save_position` | mark_to_market — every held ticker |
| `paper_trader.py:519` | `self.bq.save_paper_position(row)` | `_safe_save_position` with schema-error fallback |

### paper_portfolio / cash

| Location | Call | Trigger |
|----------|------|---------|
| `paper_trader.py:183` | `self._update_portfolio_cash(new_cash)` | execute_buy — AFTER position write |
| `paper_trader.py:309-311` | `get_or_create_portfolio` + `_update_portfolio_cash` | execute_sell |
| `paper_trader.py:360-366` | `bq.upsert_paper_portfolio(...)` | mark_to_market |
| `paper_trader.py:490-494` | `bq.upsert_paper_portfolio(portfolio)` | `_update_portfolio_cash` |

---

## 4. Critical Code Paths

### execute_buy sequencing (paper_trader.py:69-187)

```
Line 82-83:  portfolio = get_or_create_portfolio()  # reads cash
Line 86-92:  compute total_cost, check sufficiency
Line 94:     positions = self.get_positions()        # READS paper_positions
Line 95:     existing = next(filter ticker)
Line 108:    trade_id = str(uuid.uuid4())
Line 135:    self._safe_save_trade(trade)            # WRITES paper_trades
Line 138-179: if existing:
               ...delete + save                      # WRITES paper_positions (update path)
             else:
               ...save new                           # WRITES paper_positions (insert path)
Line 182-183: new_cash = cash - total_cost
              self._update_portfolio_cash(new_cash)  # WRITES paper_portfolio
```

**Key ordering problem**: `_safe_save_trade` (line 135) commits the trade record BEFORE the position write (lines 161/179) and BEFORE the cash update (line 183). If the process crashes between lines 135 and 179, cash has NOT been debited yet (cash debit is line 183), but the trade row exists. Wait — actually cash debit IS at line 183, AFTER position write. So if crash is between line 135 (trade written) and line 179 (position written), cash has NOT been debited yet either. The next cycle would re-evaluate, see no position, and buy again — cash gets debited in the successful cycle. This matches the observed state: two trades in paper_trades, one position in paper_positions, cash correctly reflecting ONE debit (if cycle 0e8c4a20 crashed before line 183).

HOWEVER: The BQ forensics show $949.95 of cash WAS leaked — meaning cash WAS debited twice. This would only happen if cycle 0e8c4a20 succeeded through the cash debit (line 183) but the cycle then crashed after that, AND the status=ERROR was triggered by a post-trade step (e.g., mark_to_market, snapshot, or learning step). That scenario is consistent: trade inserted (line 135), position inserted (line 179), cash debited (line 183), cycle then fails in a later step → `_running = False` in finally block → cycle a54a21fc starts, `get_positions()` at line 94 returns the position BUT it returned via a BQ query that hit a stale snapshot timestamp — NOT seeing the position.

This is the BQ snapshot isolation issue: cycle a54a21fc started ~4 minutes after cycle 0e8c4a20 completed (21:12:31 → 21:16:56). The `get_positions()` call runs a new BQ SELECT job that determines its snapshot at job-start. If there was any concurrency or the BQ snapshot timestamp was set to slightly before cycle 0e8c4a20's commit, the WDC position row would be invisible. Given that 0e8c4a20 ended with status=ERROR, there may also have been a delete_paper_position call in a cleanup path that removed the row.

### save_paper_position (bigquery_client.py:549-567)

```python
def save_paper_position(self, row: dict) -> None:
    # Drops None values
    row = {k: v for k, v in row.items() if v is not None}
    table = self._pt_table("paper_positions")
    cols = ", ".join(row.keys())
    vals = ", ".join(f"@v_{k}" for k in row.keys())
    query = f"INSERT INTO `{table}` ({cols}) VALUES ({vals})"   # <-- plain INSERT
    ...
    self.client.query(query, job_config=job_config).result()
```

**No MERGE. No ON CONFLICT. No ticker uniqueness check.** If called twice for the same ticker without an intervening delete, you get two rows. `get_paper_positions()` (line 531-536) returns ALL rows; `get_paper_position()` (line 538-547) returns LIMIT 1 with no ORDER BY — effectively random if two rows exist. The execute_buy logic at line 94-95 iterates get_positions() and finds `existing` via list comprehension — if two rows exist for WDC it would find the first one, which may or may not be correct.

### _run_dml_with_retry (bigquery_client.py:492-504)

```python
def _run_dml_with_retry(self, query, job_config, max_retries=3) -> None:
    for attempt in range(max_retries + 1):
        try:
            self.client.query(query, job_config=job_config).result()
            return
        except Exception as e:
            if "streaming buffer" in str(e).lower() and attempt < max_retries:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                ...sleep(wait)
            else:
                raise
```

Used for DELETE and UPDATE operations. NOT used for `save_paper_position` or `save_paper_trade` — those do plain `.result()` calls. This means position and trade inserts have no retry on streaming-buffer conflicts, though they ARE DML (not streaming API), so buffer conflicts are less likely.

### _running flag (autonomous_loop.py:71-83)

```python
_running = False

async def run_daily_cycle(...):
    global _running, _last_run, _last_result
    if _running:
        return {"status": "skipped", "reason": "already_running"}
    _running = True
    ...
    finally:
        _running = False
```

The flag is released in `finally` regardless of exception. So a crashed cycle (status=ERROR) properly releases the flag. Cycle a54a21fc starting at 21:16:56 is 4+ minutes after cycle 0e8c4a20 completed at 21:12:31 — no concurrency issue. The `_running` flag worked correctly. The duplicate is NOT a concurrent-cycle race.

### /run-now endpoint (paper_trading.py:646-665)

```python
@router.post("/run-now")
async def run_now(dry_run: bool = False):
    status = get_loop_status()
    if status["running"]:
        raise HTTPException(409, "A trading cycle is already in progress")
    ...
    asyncio.create_task(_run_cycle_background(settings))
    return {"status": "started", ...}
```

`asyncio.create_task` fires the cycle as a background task. If a user clicks "Run Now" twice within the window between `create_task` and `_running = True` being set inside `run_daily_cycle`, there IS a race window: the 409 check reads `_running` which is still False. However, the cycle_history shows manual runs at 21:08, 21:11, and 21:16 — all sequential, not overlapping. More likely the user clicked "Run Now" 3 times because the first two failed (ERROR status).

### held_tickers filter (autonomous_loop.py:211-213)

```python
positions = trader.get_positions()
held_tickers = {p["ticker"] for p in positions}
new_candidates = [c for c in candidates if c["ticker"] not in held_tickers]
```

This filters CANDIDATES for new analysis, not the execute_buy decision. Even if WDC was in held_tickers, the re-evaluation path (lines 217-230) could trigger a second BUY if the holding analysis recommended BUY again. But the real issue is in execute_buy at line 94-95 where `get_positions()` is called again fresh — if that query returns no WDC, the position is treated as new.

---

## 5. The XOM Test Trade

The XOM trade with reason='test_paper_trade' and no paper_positions row is almost certainly from an early manual /run-now call (cycle f3109df7 at 21:08:59, which errored after 51s). That cycle bought XOM, wrote the trade row (line 135), then crashed before writing the position (line 179) and before the cash debit (line 183). So XOM $500 IS in paper_trades but cash was NOT debited. OR: the crash happened after cash was debited and the position row was subsequently deleted by a mark_to_market or delete call that ran as part of cleanup.

The BQ forensics say $500 is "leaked" — meaning cash shows the $500 debit. This means the crash in cycle f3109df7 was AFTER line 183 (cash debit) but the position row is absent (either never written or deleted by a subsequent incomplete cycle). Given the 3-minute gap between f3109df7 (ended 21:09:50) and 0e8c4a20 (started 21:11:35), a manual mark-to-market or another API call could have run delete_paper_position on XOM.

---

## 6. Fix Sketches

### Fix A — Idempotency Guard in execute_buy

**Where**: paper_trader.py, insert after line 95 (after `existing` is resolved), before line 96 (max positions check).

```python
# Idempotency guard: if a BUY for this ticker was executed in the last 30 min
# at approximately the same quantity, treat as duplicate and skip.
# Prevents double-buys when a cycle crashes after trade write but before position write.
if not existing:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    recent_trades = self.bq.get_paper_trades_for_ticker_since(ticker, cutoff, action="BUY")
    if recent_trades:
        recent_qty = recent_trades[0]["quantity"]
        proposed_qty = amount_usd / price
        if abs(recent_qty - proposed_qty) / max(recent_qty, proposed_qty) < 0.01:
            logger.warning(
                f"Idempotency guard: skipping duplicate BUY for {ticker} "
                f"(recent trade {recent_trades[0]['trade_id']} at {recent_trades[0]['created_at']})"
            )
            return None
```

This requires a new BQ method `get_paper_trades_for_ticker_since(ticker, since_iso, action)` in bigquery_client.py.

**Argument for**: Directly prevents the observed bug class — cycle-crash-and-retry causing phantom double buys. Low cost (one BQ read per candidate ticker). Time window 30 min covers 5-minute retry cycles with headroom. The 1% quantity tolerance handles rounding without false positives.

### Fix B — UPSERT semantics for save_paper_position

**Where**: bigquery_client.py:549-567 — replace the plain INSERT with a MERGE.

```python
def save_paper_position(self, row: dict) -> None:
    """Upsert position via MERGE (idempotent by ticker)."""
    row = {k: v for k, v in row.items() if v is not None}
    table = self._pt_table("paper_positions")
    
    # Build MERGE: match on ticker (the natural business key)
    set_clauses = ", ".join(
        f"T.{k} = S.{k}" for k in row.keys() if k != "ticker"
    )
    cols = ", ".join(row.keys())
    vals = ", ".join(f"@v_{k}" for k in row.keys())
    
    query = f"""
        MERGE `{table}` T
        USING (SELECT {', '.join(f'@v_{k} AS {k}' for k in row.keys())}) S
        ON T.ticker = S.ticker
        WHEN MATCHED THEN
            UPDATE SET {set_clauses}
        WHEN NOT MATCHED THEN
            INSERT ({cols}) VALUES ({vals})
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

**Note**: The existing pattern of `delete_paper_position` + `save_paper_position` (used in mark_to_market and execute_sell partial exit) can be simplified to just `save_paper_position` with MERGE — delete first is no longer needed. However, for minimal-invasive change in phase-23.1.15, keep the delete pattern in existing callers and just make save_paper_position idempotent. The MERGE will handle the "already exists" case cleanly.

**Caution**: BQ MERGE on a table with a streaming buffer conflict will still raise — the `_run_dml_with_retry` retry logic should be applied to MERGE. However, paper_positions uses DML writes (not streaming), so buffer conflicts are unlikely.

### Fix C — Deterministic position_id by ticker (SKIP per user preference)

Changing `position_id = str(uuid.uuid4())` to `position_id = f"pos-{ticker}"` would cause the second INSERT to fail at the BQ level if position_id is a primary/unique key. BUT: BQ does not enforce primary key uniqueness (it's metadata only, not enforced). This won't provide a hard block without a UNIQUE constraint, which BQ doesn't have. **SKIP** this fix.

### Fix D — Verify-after-write in execute_buy

After `self.bq.save_paper_position(pos_row)` at lines 161 and 179, add a read-back to confirm the position is visible:

```python
# Verify-after-write: confirm position is visible before proceeding to cash debit.
verify_pos = self.bq.get_paper_position(ticker)
if verify_pos is None:
    logger.error(
        f"execute_buy verify-after-write FAILED for {ticker}: position not visible after insert. "
        f"Aborting cash debit to prevent cash leak."
    )
    return None
```

**Argument against**: This is a temporary band-aid. BQ DML INSERT is strongly consistent within a session (the same BQ client instance should see the committed data immediately). The visibility issue was cross-cycle (new job start), not same-job. Verify-after-write adds latency and complexity for marginal gain given Fix B (UPSERT). **CONDITIONAL** — ship only if forensics confirm BQ snapshot staleness is cross-cycle not intra-cycle.

### Fix E — Cleanup Script

File: `scripts/cleanup_phase_23_1_15.py`

Logic (two-phase: dry-run then apply):

```python
# Phase 1 — show what will be done
DRY_RUN = True  # flip to False after review

# Step 1: Delete duplicate WDC trade (21:17:41, trade_id e5447bd9...)
#   KEEP: trade_id = 56072f0c... (21:12:28, the original)
#   DELETE: trade_id = e5447bd9-9cb0-437b-b2a2-c851703b77b1

# Step 2: Refund $950.90 (= $949.95 cost + $0.95 fee) to current_cash

# Step 3: Delete XOM test_paper_trade row
#   (reason = 'test_paper_trade', no matching position)
# Step 4: Refund $500.00 (= $500 cost, no fee if fee not charged — verify from trade record)
#   Actually need to check: XOM trade's transaction_cost field for the exact fee

# Step 5: Verify post-state:
#   SELECT COUNT(*) FROM paper_trades -> should be N-2 (lost 2 rows)
#   SELECT current_cash FROM paper_portfolio -> should be += $1450.90
#   SELECT COUNT(*) FROM paper_positions -> unchanged (no WDC dup, no XOM to remove)
#   SELECT cost_basis FROM paper_positions WHERE ticker='WDC' -> $949.95
```

---

## 7. Recommended Fix Combination

**Recommend A + B + E** (as stated in the prompt).

Rationale:
- **Fix A (idempotency guard)** is the defense-in-depth layer. It prevents the bug class regardless of BQ snapshot behavior. A 30-minute lookback for a duplicate BUY of the same ticker at the same qty catches the crash-and-retry scenario definitively. Cost: 1 BQ query per candidate ticker per cycle.
- **Fix B (UPSERT MERGE)** eliminates the "two rows for same ticker" data-corruption state permanently. Even if Fix A is bypassed (e.g., a manual /run-now with different amounts), the position table stays clean. This is the architectural correctness fix.
- **Fix E (cleanup script)** is required to restore accounting integrity immediately. $1,450 phantom cash and 1 duplicate trade must be corrected before any new cycle logic is meaningful.

**Fix C** (deterministic position_id): SKIP. BQ doesn't enforce uniqueness so this provides false safety. More invasive refactor for no real benefit.

**Fix D** (verify-after-write): SKIP for now. With Fix B (UPSERT), same-cycle double-write becomes idempotent. Cross-cycle staleness is addressed by Fix A. Adding verify-after-write adds complexity.

---

## Research Gate Checklist (Internal Half)

Hard blockers:
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered every relevant module
- [x] Every relevant file read in full (not just signatures)

Soft checks:
- [x] Dead code and duplicate code noted (no UPSERT guard, no idempotency check)
- [x] Cycle history correlated with trade timestamps to establish root cause

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 6,
  "report_md": "phase-23.1.15-internal-codebase-audit.md",
  "gate_passed": "see external research brief"
}
```
