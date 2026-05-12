# Live-check placeholder -- phase-25.A3

**Step:** 25.A3 -- Write promoted strategies to pyfinagent_data.promoted_strategies BQ table
**Date:** 2026-05-12

## Live-check field (per masterplan)
> "BQ promoted_strategies row visible after next Friday promotion run"

## Pre-deployment evidence
- 10/10 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_A3.py`)
- Behavioral round-trip (claim 9) calls `run_friday_promotion` with a fake `bq_client`, asserts the mock's `save_promoted_strategy` was invoked once with the correct dict shape (strategy_id, week_iso, params JSON-round-trip, dsr, pbo, status="pending", allocation_pct, promoted_at, sortino_monthly).
- Backward-compat round-trip (claim 10) confirms `bq_client=None` preserves the existing TSV ledger write -- no regression for current callers.
- Migration dry-run prints the full 10-column DDL without touching BQ.
- Backend AST clean for all three touched files.

## Post-deployment operator workflow

### 1. Apply the migration (requires operator approval per CLAUDE.md BQ rules)
```
source .venv/bin/activate
python3 scripts/migrations/create_promoted_strategies_table.py --apply
```
Expected log line:
```
APPLIED: sunny-might-477607-p8.pyfinagent_data.promoted_strategies created/already-exists
```

### 2. Wire the BQ client into the Friday-promotion call site
The new `bq_client` kwarg on `run_friday_promotion` is opt-in. The Friday
routine's caller (the autoresearch scheduler that invokes
`run_friday_promotion`) must pass `bq_client=BigQueryClient(settings)`.
Today's caller(s) don't yet -- 25.B3 (the daily-loop reader) is the
natural place to make that wire mandatory. For 25.A3 itself the writer
is in place and the schema is durable; live BQ rows depend on the
operator wiring or the next 25.B3 step.

### 3. Trigger or wait for the next Friday promotion
```
# manual: invoke run_friday_promotion via a one-off script
# OR wait for the scheduled Friday cron
```

### 4. Verify
```sql
SELECT strategy_id, week_iso, status, dsr, pbo, allocation_pct, promoted_at
FROM `sunny-might-477607-p8.pyfinagent_data.promoted_strategies`
ORDER BY promoted_at DESC
LIMIT 5;
```
Expected: at least one row with `status='pending'` and a valid `params`
JSON value.

## Downstream
Unblocks **25.B3** (daily-loop reader via `load_promoted_params()`),
which unblocks **25.C3** (status state-machine), which unblocks
**25.R** (auto-switching policy / red-line goal-c).

**Audit anchor for next bucket:** 25.B3 (daily loop reads from this
table) is the natural follow-up.
