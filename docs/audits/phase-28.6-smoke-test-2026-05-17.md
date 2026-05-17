# phase-28.6 Smoke Test — 2026-05-17

**Step:** phase-28.6 (Crude-oil (CL=F) cross-asset trend signal)
**Date:** 2026-05-17
**Outcome:** PASS

## Test 1: Immutable verification

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')" && grep -qE 'CL=F|crude|brent|oil_trend' backend/services/macro_regime.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

Exit 0. **PASS.**

## Test 2: Settings defaults

```
crude_momentum_enabled = False
crude_momentum_window_days = 21
crude_momentum_lookback_days = 252
crude_momentum_zscore_threshold = 1.0
crude_momentum_cache_hours = 24
crude_momentum_sector_etfs = 'XLE'
```

**PASS.**

## Test 3: Live `_fetch_crude_momentum()` — real CL=F data

```
INFO: Crude momentum: current=6.685% zscore=+0.14 above=False

1m momentum: 0.0669 (+6.69%)
z-score: +0.137
rolling mean: 0.0477 std: 0.1394
threshold: 1.00
above_threshold: False
last_date: 2026-05-15 00:00:00
n_observations: 252
```

**REAL DATA:** WTI is up +6.69% over the trailing 21d — but normalized against the highly-volatile 252d distribution (std 13.94%), this is only a z-score of +0.137, well below the 1.0 trigger. **Below threshold today** — the picker would NOT inject XLE via this trigger. Good contrast with phase-28.3 (which IS above its GPR threshold).

## Test 4: `_apply_gpr_tilt` reuse unit

| above_threshold | overweight before | overweight after |
|---|---|---|
| True (synthetic) | ['XLK'] | ['XLK', 'XLE'] |
| False (live current) | ['XLK'] | ['XLK'] (identity) |

**PASS.**

## Test 5: Post-LLM hook ordering

```
line 476: GPR hook (phase-28.3)
line 496: crude hook (phase-28.6)
line 516: _save_cache
```

Both hooks wrapped in `try/except` for non-fatal graceful degradation.

**PASS.**

## Test 6: Q/A subagent verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_before_contract": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last_not_violated": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": [
    "masterplan_immutable_command: exit 0",
    "settings_py_syntax: PASS",
    "settings_defaults: False 21 252 1.0 24 XLE",
    "helper_importable: PASS",
    "phase-28.6_markers: 4 lines",
    "cl_f_ticker_references: 8 occurrences",
    "live_fetch_crude_momentum: current=+0.0669 zscore=+0.137 above=False (verbatim match)",
    "apply_gpr_tilt_reuse_unit_test: both paths exercised",
    "post_llm_hook_ordering: 476 -> 496 -> 516, both wrapped in try/except"
  ],
  "violated_criteria": [],
  "checks_run": 9
}
```

**PASS — no violations.**

## Note on Q/A continuation

Initial Q/A spawn returned mid-execution after confirming hook ordering. Sent a follow-up via `SendMessage` to the same instance with explicit instructions to complete the missing checks + overwrite the critique file + return the JSON verdict. The follow-up completed successfully. NOT verdict-shopping (no prior verdict existed) — continuation of an incomplete run.

## Stack traces

None.

## Conclusion

Crude-oil cross-asset trigger is implemented, tested with real data, and Q/A-verified. Currently below threshold (z-score +0.137 < 1.0); would activate if WTI showed exceptional 1m momentum.

## Related artifacts

- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.6.md`, `phase-28.6-research-brief.md`
- `docs/design/phase-28.6-crude-momentum.md`
- `backend/services/macro_regime.py`, `backend/config/settings.py`
