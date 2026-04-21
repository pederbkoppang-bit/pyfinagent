# Experiment Results — phase-9 / 9.6 (nightly outcome rebuild) — REMEDIATION v1

**Step:** 9.6 **Remediation cycle:** 1 **Date:** 2026-04-20

## What was done

1. Fresh researcher: 7 sources in full; `handoff/current/phase-9.6-research-brief.md`; gate passed.
2. Contract authored with cross-phase carry-forward for phase-9.1 `job_runtime.py` mark-on-success bug.
3. Re-verified immutable criterion on unchanged artifact.
4. No code changes.

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/nightly_outcome_rebuild.py').read())" && pytest tests/slack_bot/test_nightly_outcome_rebuild.py -q
...                                                                      [100%]
3 passed in 0.01s
(exit 0)
```

Tests: test_outcomes_win_loss_classification, test_idempotent_rebuild, test_bq_write_fail_open.

## Artifact shape

- `nightly_outcome_rebuild.py` — 58 lines, `run()` with DI (`ledger_fetch_fn`, `outcome_write_fn`, `store`, `day`); daily idempotency.
- `_compute_outcomes()` pure function: maps `pnl>0` → "win" else "loss".
- Fail-open: write raising `Exception` is caught + logged.WARNING + `n=0`.

## Carry-forwards (deferred)

1. **Cross-phase:** `job_runtime.py:112-113` marks idempotency key on heartbeat exit even when write fails → retry silenced. Flag for phase-9.1 hardening.
2. Fail-open lacks alert path; wire to phase-9.8 cost-budget channel at hardening.
3. Schema gaps (mae, mfe, return_pct, holding_period, strategy_id) per industry standard.
4. Gross-vs-net PnL: confirm `pyfinagent_pms.paper_trades.pnl` semantics before production wiring.

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse OK | PASS |
| 2 | pytest 3/3 | PASS |
