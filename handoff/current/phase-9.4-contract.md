# Sprint Contract — phase-9 / 9.4 (nightly MDA retrain) — REMEDIATION v1

**Step id:** 9.4 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** moderate

## Why remediation

Previous cycle was inline-authored by Main. Fresh MAS re-run per user directive.

## Research-gate summary

Fresh researcher spawn produced `handoff/current/phase-9.4-research-brief.md`:
- 9 sources read in full via WebFetch (exceeds 5 floor)
- 19 URLs collected; 10 snippet-only
- Three-variant query discipline; recency scan performed
- gate_passed = true

Validated design:
- Walk-forward + nightly cadence defensible (arXiv 2025 validation paper + CFA Institute 2025 ensemble chapter)
- Gate-then-commit pattern aligns with SR 11-7 (Fed MRM) + Snowflake/DataRobot/MLflow champion-challenger patterns
- `_default_train` stub intentionally fails gate (DSR=0.80 < min=0.95) — proves rejection path without a full backtest; production wires to `quant_optimizer.py` later

Carry-forwards (NOT in 9.4 scope):
1. `PromotionGate.evaluate()` compares new model to thresholds only, no baseline comparison — must close in phase-10.6 (Champion/Challenger)
2. Retrain should be skipped when `KillSwitchState.is_paused()` — guard absent in job at lines 32-35
3. MDA vs SHAP: MDA still canonical, but SHAP/permutation-importance alternatives are 2025-2026 frontier; reconsider at ML-refresh phase
4. Audit-trail SR 11-7 requirements: store (model_id, promoted_ts, reject_reason) — deferred to harness_learning_log wiring in phase-10.8

## Immutable criterion

`python -c "import ast; ast.parse(open('backend/slack_bot/jobs/nightly_mda_retrain.py').read())" && pytest tests/slack_bot/test_nightly_mda_retrain.py -q`
Expected: exit 0, 3/3 pass.

## Plan

1. Re-verify ast.parse + pytest 3/3.
2. Capture verbatim output.
3. Spawn fresh Q/A.
4. Log and confirm status.

## References

- `handoff/current/phase-9.4-research-brief.md` (9 sources in full, moderate tier)
- `backend/slack_bot/jobs/nightly_mda_retrain.py` (unchanged, 51 lines)
- `tests/slack_bot/test_nightly_mda_retrain.py` (unchanged, 3 tests)
- `backend/autoresearch/gate.py` (PromotionGate from phase-8.5.5)
- `backend/slack_bot/job_runtime.py` (phase-9.1 primitives)
- `.claude/masterplan.json` → phase-9 / 9.4
