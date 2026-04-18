# Contract -- Cycle 82 / phase-4.8 step 4.8.5

Step: 4.8.5 Champion-challenger gradual rollout (5 / 25 / 100%)

## Hypothesis

A promoted challenger strategy should never go from 0% -> 100% live
capital in one step. Canonical staged-rollout pattern
(Google SRE book; McMillan/Ren "Adaptive Production Deployment"):

    Stage 1 (canary):    5% of capital, 14 live days
    Stage 2 (ramp):     25% of capital, 30 live days
    Stage 3 (full):    100% of capital (champion)

Advance to the next stage only when the challenger meets:
- PSR (Probabilistic Sharpe Ratio) >= champion PSR over the stage
- Stage's `min_live_days` elapsed since last_promotion_at
- No kill-switch events during the stage
- PBO < 0.5 (phase-3.7.3 veto preserved)

Regression to previous stage on any gate failure. Demotion to 0
(strategy frozen) after 3 consecutive stage failures.

## Scope

Files created / modified:

1. **NEW** `backend/services/promotion_gate.py`
   - `STAGES = [0.05, 0.25, 1.0]` + `MIN_LIVE_DAYS = [14, 30]`
   - `evaluate_stage(challenger_stats, champion_stats, current_stage,
     days_at_stage) -> {decision: "advance" | "hold" | "regress" |
     "demote", next_allocation_pct, reasons}`
   - `update_optimizer_best(allocation_pct, stage)` -- writes the
     `allocation_pct` + `stage` fields into
     `backend/backtest/experiments/optimizer_best.json` in-place
     (preserves existing keys).

2. **NEW** `scripts/risk/promotion_gate.py`
   - `--dry-run`: reads optimizer_best.json, seeds 3 promotion
     candidates (benign / fails-psr / days-not-elapsed), runs
     `evaluate_stage` on each, updates optimizer_best.json with
     `allocation_pct = 0.05` (initial canary default) when the
     file has no stage yet, emits
     `handoff/promotion_gate_output.json`.

3. **NEW** `scripts/audit/promotion_gate_audit.py`
   Four teeth tests:
   (a) allocation_pct field present in optimizer_best.json
   (b) default 5% when file has no prior stage
   (c) gate blocks advance when PSR below champion
   (d) gate blocks advance when days-at-stage < min

## Immutable success criteria

1. allocation_pct_field_present -- grep optimizer_best.json.
2. promotion_gate_enforced -- audit tests (c) + (d) prove the gate
   actually blocks on failing conditions (not constant "advance").
3. initial_live_allocation_5pct_default -- optimizer_best.json
   default is 0.05 after --dry-run on a no-stage file.

## Verification (immutable)

    python scripts/risk/promotion_gate.py --dry-run && \
    grep -q '"allocation_pct"' backend/backtest/experiments/optimizer_best.json

Plus: `python scripts/audit/promotion_gate_audit.py --check`.

## Anti-rubber-stamp

qa must:
- Trace the advance/hold/regress/demote decision tree; ensure
  reasons are populated dynamically from real comparisons.
- Check that the file update is IN-PLACE and preserves existing
  optimizer_best keys (params, sharpe, dsr, run_id, kept,
  discarded, saved_at).
- Verify 5% default only applies when no prior stage exists; an
  existing allocation_pct should NOT be reset to 5% on re-run.

## References

- Google SRE book chapter on gradual rollouts
- McMillan / Ren "Adaptive Production Deployment" patterns
- Bailey/Lopez de Prado 2012 PSR
- Lopez de Prado AFML ch.14 (champion vs challenger)
