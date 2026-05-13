# Sprint Contract -- phase-25.A7 -- Per-table freshness endpoint covering all 5 data tables

**Cycle:** phase-25 cycle 20 (P1 sprint)
**Date:** 2026-05-13
**Step ID:** 25.A7
**Priority:** P1
**Audit basis:** bucket 24.7 F-1 -- `cycle_health.py:214-228` queries only `paper_trades + paper_portfolio_snapshots`; 5 historical/log tables unmonitored

## Research-gate

Researcher spawned this cycle (agent a160b7a48ebc77b95). Brief at
`handoff/current/research_brief.md`. Gate envelope: 7 sources read in full,
17 URLs, recency scan performed, 9 internal files inspected, gate_passed=true.

Key research conclusions:
- **5 tables to add (+ keep the 2 existing):** `historical_prices`, `historical_fundamentals`, `historical_macro`, `signals_log`, plus the existing `paper_trades` + `paper_portfolio_snapshots`.
- **Per-table SLA intervals (the new bit):**
  - `historical_prices.ingested_at` -- 93_600s (26h)
  - `historical_fundamentals.ingested_at` -- 8_208_000s (95 days)
  - `historical_macro.ingested_at` -- 3_024_000s (35 days)
  - `signals_log.recorded_at` -- `cycle_interval_sec` (per-cycle cadence)
  - `paper_portfolio_snapshots.snapshot_date` -- 93_600s (26h, override prior shared-interval semantics)
  - `paper_trades.created_at` -- `cycle_interval_sec` (unchanged)
- **Slack alarm** for criterion 3: dispatch via existing `backend/services/observability/alerting.py::raise_cron_alert_sync` (already exposes dedup via `AlertDeduper` -- so a polling-loop call doesn't spam). Fire for each `band="red"` table.
- **Reuse existing `_band` + `_bq_max_event_age` helpers** -- no new BQ machinery needed.
- **`observability_api.py` shape is unchanged** because it delegates to `compute_freshness`; the larger sources dict flows through naturally.

## Hypothesis

Extending `compute_freshness` to query 5 historical/log tables in addition
to the existing 2 paper tables, applying per-table SLA intervals (the
historical-table cadences are orders of magnitude longer than the per-cycle
cadence of the paper tables), and wiring a `_fire_freshness_alarm` helper
that calls `raise_cron_alert_sync` on every red-band table -- closes
phase-24.7 F-1 in a single source file. No API shape change required.

## Success criteria (verbatim from masterplan)

1. `api_observability_freshness_returns_per_table_ages_for_5_tables`
2. `sla_bands_green_amber_red_implemented_per_table`
3. `slack_alarm_fires_on_critical_band`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_A7.py`

Live check (per masterplan):
`GET /api/observability/freshness response includes historical_prices + historical_fundamentals + historical_macro + signals_log + paper_portfolio_snapshots with SLA bands`

## Plan

1. **`backend/services/cycle_health.py`** -- single-file change:
   - Add `_TABLE_MAX_AGE_SEC: dict[str, float]` module constant with the 3 historical-table intervals (per research recommendation).
   - Add `_worst_band(bands: list[str]) -> str` helper (`red > amber > green > unknown` priority).
   - Add `_fire_freshness_alarm(sources: dict) -> None` helper that iterates `sources` and calls `raise_cron_alert_sync` for every entry with `band == "red"`. Per-call try/except (fail-open).
   - In `compute_freshness`:
     - After `trade_age` / `snap_age` lines, add 4 new `_bq_max_event_age` calls for `historical_prices/ingested_at`, `historical_fundamentals/ingested_at`, `historical_macro/ingested_at`, `signals_log/recorded_at`.
     - Build the new `sources` dict with 6 keys (paper_trades, paper_portfolio_snapshots, historical_prices, historical_fundamentals, historical_macro, signals_log). Each carries `last_tick_age_sec`, `interval_sec`, `ratio`, `band`.
     - Compute `overall_band = _worst_band([v["band"] for v in sources.values()])`.
     - When `overall_band == "red"`, call `_fire_freshness_alarm(sources)`.
     - Return dict now includes `overall_band`.
2. **`tests/verify_phase_25_A7.py`** -- new file with 10+ claims:
   - Claim 1: `_TABLE_MAX_AGE_SEC` constant exists with the 3 historical-table keys + correct values.
   - Claim 2: `_worst_band` helper exists with `red > amber > green > unknown` priority (behavioral via direct call).
   - Claim 3: `_fire_freshness_alarm` helper exists and imports `raise_cron_alert_sync` at call time.
   - Claim 4: `compute_freshness` returns a `sources` dict with all 6 required keys.
   - Claim 5: Each `sources` entry contains `last_tick_age_sec, interval_sec, ratio, band` keys.
   - Claim 6: Return dict includes top-level `overall_band` key.
   - Claim 7: **Behavioral happy path** -- fake bq returns small ages -> all bands green -> NO alert call.
   - Claim 8: **Behavioral red band** -- fake bq returns a critical age for `historical_prices` -> `_fire_freshness_alarm` called; the underlying `raise_cron_alert_sync` is invoked at least once with `severity="P1"`.
   - Claim 9: **Behavioral worst-band aggregation** -- mixed bands across tables -> `overall_band` matches `_worst_band(...)`.
   - Claim 10: **Behavioral fail-open** -- `raise_cron_alert_sync` raising does NOT propagate (alarm fail-open).
   - Claim 11: SLA bands `green/amber/red` are mentioned (criterion 2 grep).

## Non-goals

- No API shape break (`observability_api.py` untouched; delegates to `compute_freshness`).
- No new BQ migration (the historical tables already exist).
- No anomaly detection (static thresholds per research finding: data cadences are deterministic).
- No frontend changes.

## References

- `handoff/current/research_brief.md` -- full brief
- `backend/services/cycle_health.py:40-41` (WARN/CRITICAL ratios), :57-65 (`_band`), :161-198 (`_bq_max_event_age`), :201-248 (`compute_freshness`)
- `backend/services/observability/alerting.py:185` (`raise_cron_alert_sync`) + :51 (`AlertDeduper`)
- `backend/api/observability_api.py:25-44` (route delegating to compute_freshness)
- CLAUDE.md `Critical Rules` -- 30s BQ timeout (already enforced via `_bq_max_event_age`)
