# Experiment Results -- Cycle 82 / phase-4.8 step 4.8.5

Step: 4.8.5 Champion-challenger gradual rollout (5 / 25 / 100%)

## What was generated

1. **NEW** `backend/services/promotion_gate.py`
   - `STAGES = [0.05, 0.25, 1.0]`; `MIN_LIVE_DAYS = [14, 30]`;
     `PBO_CEILING = 0.5`.
   - `evaluate_stage(challenger, champion, current_stage,
     days_at_stage, consecutive_failures)` returns decision
     (advance / hold / regress / demote) + next allocation +
     dynamic reason strings.
   - `update_optimizer_best(path, allocation_pct, stage,
     challenger_run_id)` writes IN-PLACE, preserving existing keys.

2. **NEW** `scripts/risk/promotion_gate.py`
   - `--dry-run`: evaluates 3 seeded candidates + ensures
     optimizer_best.json has `allocation_pct`; defaults to
     `STAGES[0]=0.05` (canary) on first deploy; preserves
     existing stage on re-runs.
   - Emits `handoff/promotion_gate_output.json` with
     preserved_keys list proving no overwrite.

3. **NEW** `scripts/audit/promotion_gate_audit.py`
   4 teeth tests:
   (a) allocation_pct field present in optimizer_best.json
   (b) benign: advance to 0.25
   (c) PSR failure: NOT advance, `psr_below_champion` reason
   (d) insufficient days: hold, `days_at_stage_insufficient` reason

## Verification (verbatim, immutable)

    $ python scripts/risk/promotion_gate.py --dry-run && \
      grep -q '"allocation_pct"' backend/backtest/experiments/optimizer_best.json
    {"file_updated": true, "allocation_pct": 0.05, "stage": 0}
    exit=0

    $ python scripts/audit/promotion_gate_audit.py --check
    {"verdict": "PASS", "t_a": true, "t_b": true,
     "t_c": true, "t_d": true}

## optimizer_best.json preservation

Before: 7 keys (params, sharpe, dsr, run_id, kept, discarded,
saved_at).
After: 9 keys (added allocation_pct=0.05 + stage=0). All 7
original keys preserved verbatim.

## Success criteria

| Criterion | Result |
|-----------|--------|
| allocation_pct_field_present | PASS (0.05) |
| promotion_gate_enforced | PASS (PSR failure + days-insufficient both block advance with correct reason strings) |
| initial_live_allocation_5pct_default | PASS (STAGES[0]=0.05 canary) |

## Known limitations (non-blocking)

- Gate is library-layer only this cycle; wiring the daily cron
  that reads live challenger/champion stats and calls
  `evaluate_stage` lives in the next phase-4.8.x step.
- 3 consecutive failures -> demote; failure counter persistence
  (BQ or sidecar JSON) also deferred to the wiring step.
- Seeded candidates in the CLI are for contract verification; real
  decisions will be driven by the autoresearch leaderboard (phase-
  4.7.4) + PSR/PBO values computed from live paper trading.
