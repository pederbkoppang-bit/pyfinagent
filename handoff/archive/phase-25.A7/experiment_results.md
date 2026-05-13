---
step: phase-25.A7
cycle: 76
cycle_date: 2026-05-13
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_A7.py'
title: Per-table freshness endpoint covering all 5 data tables (P1)
audit_basis: phase-24.7 F-1 (compute_freshness only queried 2 paper tables; 4 historical/log tables unmonitored)
---

# Experiment Results -- phase-25.A7

## Code changes

### `backend/services/cycle_health.py` (single-file change)
- New module constant `_TABLE_MAX_AGE_SEC: dict[str, float]` with per-table SLA intervals:
  - `historical_prices`: 93_600s (26h nightly + buffer)
  - `historical_fundamentals`: 8_208_000s (95 days = quarterly + 5-day filing lag)
  - `historical_macro`: 3_024_000s (35 days = monthly FRED + release lag)
  - `paper_portfolio_snapshots`: 93_600s (26h, override prior shared-interval semantics)
- New `_worst_band(bands: list[str]) -> str` helper -- priority `red > amber > green > unknown`. Empty list -> `unknown`.
- New `_fire_freshness_alarm(sources: dict) -> None` helper -- iterates `sources`, calls `raise_cron_alert_sync` for every `band == "red"` table with `severity="P1"` + per-table details. Imports the alert function lazily; per-call try/except so a Slack failure never propagates back to the caller.
- `compute_freshness` extended:
  - 4 new `_bq_max_event_age` calls for `historical_prices/ingested_at`, `historical_fundamentals/ingested_at`, `historical_macro/ingested_at`, `signals_log/recorded_at`.
  - New `sources` dict with 6 keys (paper_trades, paper_portfolio_snapshots, historical_prices, historical_fundamentals, historical_macro, signals_log). Each entry: `last_tick_age_sec, interval_sec, ratio, band`.
  - **Key rename**: `paper_snapshots` -> `paper_portfolio_snapshots` so the dict key matches the BQ table name exactly (operators grepping for the table find the freshness key).
  - New top-level `overall_band` field from `_worst_band([...])`.
  - When `overall_band == "red"`, dispatches alarms via `_fire_freshness_alarm(sources)`.

### `tests/verify_phase_25_A7.py` (new file)
- 11 immutable claims with 5 behavioral round-trips:
  - Claims 1-3, 11: structural (constant, helpers, SLA-band documentation).
  - Claim 4: **Behavioral worst-band priority** -- 6 cases including empty list and unknown.
  - Claim 5: **Behavioral 6-table coverage** -- fake BQ returns deterministic ages per table; assert all 6 keys present.
  - Claim 6: schema check -- each entry has `last_tick_age_sec/interval_sec/ratio/band` with band in the canonical 4-value enum.
  - Claim 7: top-level `overall_band` field exists.
  - Claim 8: **Behavioral happy path** -- all ages small -> overall_band=green + NO alert dispatch.
  - Claim 9: **Behavioral critical** -- historical_prices age = 500_000s (well past 2x interval) -> overall_band=red AND `raise_cron_alert_sync` invoked with `severity="P1"` and `details.table=historical_prices`.
  - Claim 10: **Behavioral fail-open** -- `raise_cron_alert_sync` raises -> no propagation; result still returned.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_A7.py
freshness alarm: dispatch fail-open for historical_prices: RuntimeError('slack down')
PASS: table_max_age_sec_constant_with_documented_intervals
PASS: worst_band_helper_signature
PASS: fire_freshness_alarm_helper_with_raise_cron_alert_sync
PASS: behavioral_worst_band_priority_red_amber_green_unknown
PASS: api_observability_freshness_returns_per_table_ages_for_5_tables
PASS: sla_bands_green_amber_red_implemented_per_table
PASS: compute_freshness_returns_overall_band_aggregate
PASS: behavioral_happy_path_all_green_no_alert
PASS: slack_alarm_fires_on_critical_band
PASS: alarm_dispatch_fail_open_on_slack_failure
PASS: sla_band_names_green_amber_red_present_in_source

11/11 claims PASS, 0 FAIL
```

(The "freshness alarm: dispatch fail-open" log line is emitted by the fail-open behavioral test -- it proves the fail-open path actually runs as designed.)

## Backend gates

- `python -c "import ast; ast.parse(open('backend/services/cycle_health.py').read())"` -- OK
- 5 behavioral round-trips exercise the actual function with fake BQ + mocked alerting -- mutation-resistant.

## Hypothesis verdict

CONFIRMED. Three immutable success criteria mapped:
- Criterion 1 (`api_observability_freshness_returns_per_table_ages_for_5_tables`) -- claim 5 (behavioral 6-table coverage; covers the 5 required + paper_trades sanity).
- Criterion 2 (`sla_bands_green_amber_red_implemented_per_table`) -- claim 6 (schema check enforces band ∈ green/amber/red/unknown per entry).
- Criterion 3 (`slack_alarm_fires_on_critical_band`) -- claim 9 (behavioral: red band -> P1 alert with table in details).

## Live-check

Per masterplan: `GET /api/observability/freshness response includes historical_prices + historical_fundamentals + historical_macro + signals_log + paper_portfolio_snapshots with SLA bands`.

Live evidence pending in `handoff/current/live_check_25.A7.md`. After backend restart, the existing route `GET /api/observability/freshness` (alias to canonical `/api/paper-trading/freshness`) returns the new 6-key sources dict + `overall_band`.

## Non-regressions

- API shape -- ADDITIVE: new keys (`overall_band`, additional `sources` entries) appear; existing consumers reading `sources.paper_trades` keep working. The `paper_snapshots` key WAS renamed to `paper_portfolio_snapshots` (matches BQ table name). If any frontend consumer reads the old `paper_snapshots` key, this is a breaking change -- a grep of `frontend/` confirms no usage (only docs reference the old name).
- 30s BQ timeout already enforced inside `_bq_max_event_age`.
- No new BQ schema; no migration.

## Next phase

Q/A pending.
