# phase-28.2 Smoke Test — 2026-05-17

**Step:** phase-28.2 (12-quarter SUE stacking)
**Date:** 2026-05-17
**Outcome:** PASS

## Scope

End-to-end smoke for the `_LOOKBACK_QUARTERS` bump from 8 to 12 in pead_signal.py + parallel settings sync. Goal: confirm constant change, settings sync, docstring update, equal-weight preserved, cache back-compat verified via synthetic-cache test.

## Test 1: Immutable verification

```
$ source .venv/bin/activate && grep -qE '_LOOKBACK_QUARTERS\s*=\s*12' backend/services/pead_signal.py && python -c "import ast; ast.parse(open('backend/services/pead_signal.py').read()); print('PASS')" && echo "MASTERPLAN VERIFICATION: PASS"
PASS
MASTERPLAN VERIFICATION: PASS
```

Exit 0. **PASS.**

## Test 2: Module constant + settings

```
$ python -c "from backend.services.pead_signal import _LOOKBACK_QUARTERS; from backend.config.settings import Settings; print(f'module={_LOOKBACK_QUARTERS}, setting={Settings().pead_signal_lookback_quarters}')"
module=12, setting=12
```

**PASS.**

## Test 3: Description + phase marker grep

```
$ grep -n 'rolling-12Q' backend/services/pead_signal.py
54:        description="sentiment_score - rolling-12Q mean (phase-28.2; was 8Q). Positive = above-trend tone; negative = below-trend.",

$ grep -n 'phase-28.2' backend/services/pead_signal.py
39:# phase-28.2 (2026-05-17): bumped 8 -> 12 per ScienceDirect 2025 ML paper
54:        description="sentiment_score - rolling-12Q mean (phase-28.2; was 8Q). ..."
```

**PASS.**

## Test 4: Synthetic 12-cache smoke (equal-weight + back-compat)

Wrote 12 synthetic cache files (legacy `pead_TESTQ_<YYYY-MM-DD>.json` format) for sentiment progression 0.40 → 0.72. Called `_trailing_mean_from_cache(TESTQ, exclude_quarter="2099-12-31")`:

| Pre-edit (simulated 8Q via monkeypatch) | Post-edit (12Q) |
|---|---|
| n_quarters = 8 | n_quarters = 12 |
| trailing_mean = 0.6025 | trailing_mean = 0.5475 |

Effect on hypothetical current sentiment 0.75:

| | 8Q | 12Q |
|---|---|---|
| surprise_score | +0.1475 | +0.2025 |
| sentiment_tag | positive_surprise | positive_surprise |
| holding_window_days | 28 | **42** |

**PASS** — 12Q correctly identifies a stronger and longer-held positive surprise; equal-weight arithmetic verified.

## Test 5: Q/A subagent verdict

Q/A returned PASS with no violations. Independent Q/A-side synthetic smoke reproduced the exact mean (0.5475 over 12 quarters) reported in experiment_results.md.

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last_not_violated": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": {
    "immutable_verification_command_exit": 0,
    "module_constant_value": 12,
    "settings_default_value": 12,
    "rolling_12Q_in_description": true,
    "phase_28_2_comment_present": true,
    "synthetic_smoke_n_quarters": 12,
    "synthetic_smoke_mean_matches_equal_weight": true
  },
  "violated_criteria": [],
  "checks_run": 11
}
```

## Stack traces / failures

None.

## Conclusion

12-quarter SUE stacking is implemented, tested, and Q/A-verified. The PEAD signal itself is still gated by `pead_signal_enabled=False`, so production behavior is unchanged when that flag is OFF.

## Related artifacts

- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.2.md`, `phase-28.2-research-brief.md`
- `docs/design/phase-28.2-sue-stacking.md`
- `backend/services/pead_signal.py`, `backend/config/settings.py`
