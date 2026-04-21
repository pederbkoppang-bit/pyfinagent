# Sprint Contract — phase-9 / 9.6 (nightly outcome rebuild) — REMEDIATION v1

**Step id:** 9.6 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Why remediation

Previous cycle inline-authored. Fresh MAS re-run.

## Research-gate summary

Fresh researcher: `handoff/current/phase-9.6-research-brief.md` — 7 sources in full, 17 URLs, three-variant, recency, gate_passed=true.

Validated design:
- Binary `pnl>0` win/loss classification is canonical (QuantConnect Lean + TraderSync); `pnl==0` → "loss" is defensible
- Full-rebuild (not incremental) correct for outcome tables (OneUptime 2026 BQ guidance)
- 4-field MVP schema (trade_id, ticker, pnl, outcome) valid for this phase

Carry-forwards (NOT in 9.6 scope):
1. **Cross-phase bug to flag:** `job_runtime.py:112-113` marks idempotency key seen even when write fails and `rebuilt == 0` — retry permanently silenced for that day. Real issue; should be addressed in a phase-9.1 hardening follow-up (mark-on-success, not mark-on-heartbeat-exit).
2. Fail-open `logger.warning` has no alert path — silent data loss possible; wire to phase-9.8 cost-budget-watcher alert channel at hardening
3. Schema gaps for production: `mae`, `mfe`, `return_pct`, `holding_period`, `strategy_id` (industry-standard per TraderSync/Tradewink/TradesViz) — deferred to richer outcome-attribution phase
4. Gross-vs-net PnL: must confirm `pyfinagent_pms.paper_trades.pnl` column semantics before wiring `_default_fetch` in production

## Immutable criterion

`python -c "import ast; ast.parse(open('backend/slack_bot/jobs/nightly_outcome_rebuild.py').read())" && pytest tests/slack_bot/test_nightly_outcome_rebuild.py -q`

## Plan

1. Re-verify.
2. Capture output.
3. Spawn fresh Q/A.
4. Log and flip.

## References

- `handoff/current/phase-9.6-research-brief.md`
- `backend/slack_bot/jobs/nightly_outcome_rebuild.py` (58 lines)
- `tests/slack_bot/test_nightly_outcome_rebuild.py` (3 tests)
- `backend/slack_bot/job_runtime.py` (mark-on-success bug carry-forward)
- `.claude/masterplan.json` → phase-9 / 9.6
