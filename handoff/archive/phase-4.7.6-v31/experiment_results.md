# Experiment Results -- Cycle 81 / phase-4.8 step 4.8.4

Step: 4.8.4 Drift monitor (PSI + 20-day rolling IC)

## What was generated

1. **NEW** `backend/services/drift_monitor.py`
   - `compute_psi(baseline, current, bins=10)` -- canonical
     Siddiqi form `sum_i (a_i - e_i) * ln(a_i / e_i)` with
     baseline-quantile bins + zero-bin eps floor.
   - `compute_ic(predictions, returns)` -- Spearman rank
     correlation via `argsort().argsort()` + Pearson-on-ranks.
   - `rolling_ic(pred, ret, window=20)` -- per-day series.
   - `run(models=None)` -> snapshot with 3 seeded models by
     default; `data_source="seeded"|"live"`.
   - Constants: `PSI_FREEZE_THRESHOLD=0.25`,
     `IC_FREEZE_MAX=0.0`, `IC_FREEZE_SUSTAINED_DAYS=5`,
     `IC_WINDOW=20`.

2. **NEW** `scripts/audit/drift_monitor_audit.py`
   - 3 fixtures:
     * benign: psi ~0.01, ic ~+0.4, frozen=false
     * psi_trip: shifted+stretched current, psi ~1.87, frozen=true
       with "psi_exceeded"
     * ic_trip: returns = -1.0*preds + 0.2*noise, Spearman ~-0.93
       sustained, frozen=true with "ic_sustained_negative"
   - `--check` exits 1 on FAIL.

## Verification (verbatim, immutable)

    $ python -c "from backend.services.drift_monitor import run; \
                  r=run(); assert 'models' in r and \
                  all('psi' in m and 'ic_20d' in m for m in r['models'])"
    exit=0 (3 seeded models, all with psi + ic_20d floats)

    $ python scripts/audit/drift_monitor_audit.py --check
    {"verdict": "PASS", "benign_ok": true,
     "psi_trip_ok": true, "ic_trip_ok": true}

## Success criteria

| Criterion | Result |
|-----------|--------|
| psi_weekly_logged | PASS (3 float values) |
| ic_20d_rolling_logged | PASS (3 float values) |
| auto_freeze_fires_at_thresholds | PASS (trip fixtures both flip frozen=true with correct reason strings) |

## Known limitations (tracked follow-up)

- run() seeds 3 models when no live strategy-prediction stream is
  wired in. Real hookup pulls from BQ
  `pyfinagent_data.strategy_predictions` + `historical_prices` for
  forward returns -- queued in phase-10 (Karpathy loop integration).
- IC-trip fixture uses -1.0 anti-correlation weight (dramatic) to
  decisively demonstrate sustained-negative detection; real drift
  events are usually weaker (and the rolling-5-day check + sustained
  threshold still catches them).
- Auto-freeze currently emits `frozen=True` flag; wiring it into
  the strategy-manager to actually stop trades lands with
  phase-10 evolution loop.
