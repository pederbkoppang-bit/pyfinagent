# Experiment Results — phase-9 / 9.4 (nightly MDA retrain) — REMEDIATION v1

**Step:** 9.4 **Remediation cycle:** 1 **Date:** 2026-04-20

## What was done

1. Fresh researcher subagent (moderate tier): 9 sources in full via WebFetch; brief at `handoff/current/phase-9.4-research-brief.md`; gate passed.
2. Contract authored.
3. Re-verified immutable criterion on unchanged artifact + test suite.
4. No code changes.

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/nightly_mda_retrain.py').read())" && pytest tests/slack_bot/test_nightly_mda_retrain.py -q
...                                                                      [100%]
3 passed in 0.01s
(exit 0)
```

Tests: test_good_model_promotes_and_commits, test_rejected_model_does_not_commit, test_idempotent_same_day.

## Artifact shape

- `nightly_mda_retrain.py` — 51 lines, one `run()`, imports `PromotionGate` from phase-8.5.5, daily idempotency, commits baseline ONLY if `verdict["promoted"]`.
- `_default_train` stub returns DSR=0.80 (intentionally fails gate min=0.95) — rejection-path default.
- DI surface: `train_fn`, `gate`, `commit_fn`, `store`, `day`.

## Carry-forwards (deferred per research brief)

1. `PromotionGate.evaluate()` thresholds-only; no baseline comparison → close in phase-10.6 Champion/Challenger
2. Retrain should skip when `KillSwitchState.is_paused()` — guard absent lines 32-35
3. MDA vs SHAP: MDA canonical, SHAP/permutation frontier — reconsider at ML-refresh
4. SR 11-7 audit trail (model_id, promoted_ts, reject_reason) → wire via phase-10.8 harness_learning_log

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse OK | PASS |
| 2 | pytest 3/3 | PASS |
