# Cycle 14 — Experiment Results (DoD-5 closure: freshness probe SQL fix)

**Window:** 2026-05-28T18:00-18:25+02:00 (approx)
**Sub-step of:** phase-43.0 (P1, H) — closes DoD-5 of the 14-criterion gate
**Editor:** Main (Claude Code session)
**Researcher gate:** `a6e333f3743b90f0b` PASSED (6 sources in full / 13 URLs / recency scan / 3-variant queries / 7 internal files)

---

## Files modified

- `backend/services/cycle_health.py` — Pattern C fix to `_bq_max_event_age` + new module-scope `_STRING_DATE_TIMESTAMP_COLS` constant

## Files created

- `handoff/current/research_brief_phase_43_0_dod_5_freshness.md` (researcher output)
- `handoff/current/contract.md` (cycle-14 contract; overwrote cycle-13)
- `handoff/current/live_check_43_0_dod_5.md` (verbatim pre/post curl evidence)
- `handoff/current/experiment_results.md` (this file)

## Bug summary (vs Main's initial premature hypothesis)

**Main's first guess:** `_pt_table()` resolves historical_* tables to wrong dataset (`financial_reports` instead of `pyfinagent_data`). Pattern A or B fix.

**Researcher correction (verified via live `bigquery.Client.get_table()`):** ALL 4 historical/signals tables ARE in `financial_reports`. The bug is the SQL pattern itself.

**Real root cause:**

`cycle_health.py:_bq_max_event_age()` wraps `MAX(time_col)` in `SAFE.TIMESTAMP(...)` to coerce STRING-typed columns (paper_trades.created_at RFC3339, paper_portfolio_snapshots.snapshot_date YYYY-MM-DD) to TIMESTAMP. But this BREAKS for already-TIMESTAMP-typed columns (historical_prices.ingested_at, historical_fundamentals.ingested_at, historical_macro.ingested_at, signals_log.recorded_at) because:
1. `TIMESTAMP()` has no `(TIMESTAMP) -> TIMESTAMP` overload — only STRING/DATE/DATETIME inputs.
2. `SAFE.` prefix is not supported with aggregates — and `MAX()` is one.

Effect: query returns `400 BadRequest: SAFE with function timestamp is not supported`. Broad except at `:445-452` swallows the 400, function returns None, `_band(None, ...)` returns `"unknown"`.

## Fix (Pattern C: type-aware branch)

Added module-scope `_STRING_DATE_TIMESTAMP_COLS = {("paper_trades", "created_at"), ("paper_portfolio_snapshots", "snapshot_date")}` listing the 2 STRING/DATE columns that need the SAFE.TIMESTAMP wrapper. Inside `_bq_max_event_age`, branch:

```python
needs_coerce = (table_logical, time_col) in _STRING_DATE_TIMESTAMP_COLS
max_expr = (
    f"SAFE.TIMESTAMP(MAX({time_col}))" if needs_coerce
    else f"MAX({time_col})"
)
```

All other call sites (historical_prices, historical_fundamentals, historical_macro, signals_log) now use bare `MAX(time_col)` which works against their native-TIMESTAMP columns.

Diff size: 8 LOC + 1 constant. No caller churn. Preserves the working paper-trades path.

## Verification — all 4 commands

```bash
$ python3 -c "import ast; ast.parse(open('backend/services/cycle_health.py').read()); print('OK')"
OK: syntax check passes

$ launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
(backend restarted)

$ curl -sf -m 10 http://localhost:8000/api/health
{"status":"ok","service":"pyfinagent-backend","version":"6.25.3","mcp_servers":{...}}

$ curl -sf http://localhost:8000/api/paper-trading/freshness | python3 -c '
import json, sys
d = json.load(sys.stdin)
src = d["sources"]
unk = [k for k, v in src.items() if (v.get("band") or "").lower() == "unknown"]
print(f"total_sources: {len(src)}, unknown: {len(unk)}, unknown_keys: {unk}")'
total_sources: 6, unknown: 0, unknown_keys: []

$ test -f handoff/current/live_check_43_0_dod_5.md && grep -qE 'PASS|FAIL' handoff/current/live_check_43_0_dod_5.md && echo VERIFY_PASS
VERIFY_PASS
```

## Post-fix band table

| Source | Pre-fix | Post-fix | Ratio | Notes |
|---|---|---|---|---|
| paper_trades | green | green | 1.012x | Working path unchanged |
| paper_portfolio_snapshots | green | amber | 1.641x | Just aged during the audit window; not regression |
| historical_prices | unknown | **red** | 47.98x | Was masking 52-day staleness; real ingestion bug surfaced |
| historical_fundamentals | unknown | **green** | 0.547x | Quarterly cadence; healthy |
| historical_macro | unknown | **amber** | 1.843x | Past 35d threshold; minor staleness |
| signals_log | unknown | **green** | 1.012x | Daily cadence; healthy |

**Unknown count: 4 → 0.** DoD-5 PASS.

## Out-of-scope observations (filed for follow-up cycles, NOT bundled)

1. **`historical_prices` 52-day staleness** — real ingestion-pipeline issue. `overall_band` now correctly red. Should be a separate cycle to investigate the data ingestion job.
2. **`backend/metrics/sortino.py:108`** — hardcodes `pyfinagent_data.historical_macro`; table lives in `financial_reports`; query currently 404'ing. Researcher's separate finding. Should be a separate small-cycle fix.

## Anti-pattern check

- `feedback_no_emojis` — no emojis in code or artifacts.
- `feedback_contract_before_generate` — contract.md written BEFORE this file.
- `feedback_log_last` — harness_log append AFTER Q/A PASS.
- `feedback_qa_harness_compliance_first` — Q/A prompt will open with 5-item audit.
- `feedback_harness_rigor` — DoD-5 verdict was FAIL in cycle 12; closing via real fix (not hand-waving).
- `feedback_full_codebase_audit_before_changes` — researcher overturned my premature hypothesis; honored.
- `feedback_npm_install_requires_launchctl_kickstart` — restarted backend via launchctl kickstart (not pkill, which races the watchdog).
- `feedback_auto_commit_hook_stalls` — will manual-commit; no masterplan flip.

## Step status policy

phase-43.0 STAYS `pending`. Cycle 14 closes DoD-5 only (was FAIL → now PASS). Cumulative tally: **11 most-generous / 7 literal of 14 PASS** (was 10/6 after cycle 13). Remaining open: DoD-1 (phase-39.1, owner-gated), DoD-2 (walk-forward instrumentation), DoD-6 (BQ probe), DoD-7 (Risk Judge runtime), DoD-9 (5-cycle stability).

## References

- Contract: `handoff/current/contract.md`
- Research brief: `handoff/current/research_brief_phase_43_0_dod_5_freshness.md`
- Live evidence: `handoff/current/live_check_43_0_dod_5.md`
- Cycle 12 audit: `handoff/current/production_ready_audit_2026-05-28.md`
- Target: `backend/services/cycle_health.py:_bq_max_event_age` (lines 414-462 after fix)
- BQ TIMESTAMP functions: https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions
- BQ SAFE prefix limitation: https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/conversion_functions
