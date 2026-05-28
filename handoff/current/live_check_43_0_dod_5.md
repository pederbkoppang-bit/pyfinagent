# Live-check evidence — phase-43.0 DoD-5 (freshness "unknown" bands)

**Cycle:** 14 | **Date:** 2026-05-28 | **Verdict:** PASS

---

## Pre-fix (cycle-12 audit baseline, re-captured immediately before the edit)

```
$ curl -sf http://localhost:8000/api/paper-trading/freshness | python3 -c '...'
total_sources: 6, unknown: 4, unknown_keys: ['historical_prices', 'historical_fundamentals', 'historical_macro', 'signals_log']
```

All 4 `historical_*` + `signals_log` sources show `band: "unknown"` because the BQ query for `_bq_max_event_age` was returning 400 BadRequest (`SAFE with function timestamp is not supported`), silently swallowed by the broad except, returning None, banded as "unknown".

## Post-fix (after the Pattern C edit to cycle_health.py + backend restart)

`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` then `sleep ~6s` until `/api/health` returned 200.

```
$ curl -sf http://localhost:8000/api/paper-trading/freshness | python3 -m json.tool
{
    "sources": {
        "paper_trades":                {"last_tick_age_sec": 87464.0,  "interval_sec": 86400.0,  "ratio": 1.0123, "band": "green"},
        "paper_portfolio_snapshots":   {"last_tick_age_sec": 153574.0, "interval_sec": 93600.0,  "ratio": 1.6407, "band": "amber"},
        "historical_prices":           {"last_tick_age_sec": 4490884.0,"interval_sec": 93600.0,  "ratio": 47.98,  "band": "red"},
        "historical_fundamentals":     {"last_tick_age_sec": 4490600.0,"interval_sec": 8208000.0,"ratio": 0.547,  "band": "green"},
        "historical_macro":            {"last_tick_age_sec": 5573907.0,"interval_sec": 3024000.0,"ratio": 1.843,  "band": "amber"},
        "signals_log":                 {"last_tick_age_sec": 87454.0,  "interval_sec": 86400.0,  "ratio": 1.0122, "band": "green"}
    },
    "overall_band": "red",
    ...
}
```

`unknown_count = 0`.

## DoD-5 criterion mapping

DoD-5 from master_roadmap §6:
> `GET /api/paper-trading/freshness` returns no `band='Unknown'` rows across all source rows.

**Status: PASS.** 0 of 6 sources have `band: "unknown"`.

## Predicted-vs-actual band table

| Source | Predicted | Actual | Match? |
|---|---|---|---|
| historical_prices | red (52d > 26h) | red (47.98x ratio, ~52d) | YES |
| historical_fundamentals | green (52d < 95d quarterly) | green (0.547x ratio) | YES |
| historical_macro | amber (64d > 35d threshold) | amber (1.843x ratio) | YES |
| signals_log | green (24h ~= cycle interval) | green (1.012x ratio) | YES |

4 of 4 predictions match.

## Observations beyond DoD-5 scope (informational, not blocking this cycle)

1. **`overall_band` flipped from "green" to "red"** post-fix. This is the CORRECT operator signal — `historical_prices` is 52 days stale (ingested_at 2026-04-06 area). The pre-fix "green" was a FALSE POSITIVE caused by the Unknown band being treated leniently in `_worst_band`. Post-fix the operator sees the real ingestion-pipeline staleness. SEPARATE follow-up cycle to fix the underlying ingestion freshness.
2. **`paper_portfolio_snapshots` flipped from green to amber.** Not caused by this fix — the snapshot just aged past the green threshold during the audit window. Independent.
3. **`backend/metrics/sortino.py:108`** hardcodes `pyfinagent_data.historical_macro` (researcher's separate finding); the table is in `financial_reports` so the sortino query is currently 404'ing. SEPARATE follow-up.

## Anti-pattern check on the post-fix output

- No emoji in evidence dump.
- Verbatim curl JSON quoted, not paraphrased.
- Pre-fix evidence captured BEFORE the edit (not retroactive).
- Predicted-vs-actual table forces honest "did we know what we'd see" check.

## Verdict

**PASS** — DoD-5 immutable criterion satisfied (0 unknown bands). Fix is type-aware branch in `_bq_max_event_age`, paper_trades / paper_portfolio_snapshots path preserved (verified: both still green/amber respectively, not red/error).
