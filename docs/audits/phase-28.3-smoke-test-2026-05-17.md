# phase-28.3 Smoke Test — 2026-05-17

**Step:** phase-28.3 (GPR-triggered energy-sector tilt)
**Date:** 2026-05-17
**Outcome:** PASS

## Test 1: Immutable verification

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')" && grep -qE 'gpr|geopolitical' backend/services/macro_regime.py
syntax OK
```

Exit 0. **PASS.**

## Test 2: Settings defaults

```
gpr_signal_enabled = False
gpr_signal_quantile = 0.9
gpr_signal_cache_hours = 24
gpr_signal_sector_etfs = 'XLE'
```

**PASS.**

## Test 3: `_apply_gpr_tilt` unit (5 cases)

| Case | above_threshold | overweight before | etfs csv | overweight after |
|---|---|---|---|---|
| Below threshold | False | ['XLK'] | "XLE" | ['XLK'] (identity) |
| Above threshold single | True | ['XLK'] | "XLE" | ['XLK', 'XLE'] |
| Above threshold multi | True | ['XLK'] | "XLE,XOM,CVX" | ['XLK', 'XLE', 'XOM', 'CVX'] |
| Dedup | True | ['XLE'] | "XLE" | ['XLE'] (no double-add) |
| Empty CSV | True | ['XLK'] | "" | ['XLK'] (identity) |

**5/5 PASS** (Q/A independently verified).

## Test 4: Live `_fetch_gpr_acts()` (real matteoiacoviello.com data)

```
INFO: GPR Excel downloaded (2705408 bytes) from https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls
GPR-Acts current: 285.35
90th-pct threshold (rolling 60 months): 184.93
last_date: 2026-04-01 00:00:00
above_threshold: True
```

**PASS.** Real GPR-Acts April 2026 reading (285.35) is +54% above the trailing 5y 90th-pct threshold (184.93). Picker would inject XLE if `gpr_signal_enabled=True`.

## Test 5: Q/A subagent verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate_before_contract": true,
    "contract_before_generate": true,
    "results_verbatim": true,
    "log_last_not_violated": true,
    "no_verdict_shopping": true
  },
  "deterministic_checks": {
    "immutable_verification_cmd_exit": 0,
    "settings_defaults_exact": "False 0.9 24 XLE",
    "helpers_importable": true,
    "phase_28_3_markers_count": 5,
    "apply_gpr_tilt_unit_tests": "5/5 PASS",
    "live_fetch_current": 285.35,
    "live_fetch_threshold": 184.93,
    "live_fetch_above_threshold": true,
    "xlrd_version_installed": "2.0.2"
  },
  "violated_criteria": [],
  "checks_run": 14
}
```

## Mid-cycle issue (resolved)

`xlrd` was not installed → first live fetch failed with `Import xlrd failed. Install xlrd >= 2.0.1`. Fixed by `pip install xlrd>=2.0.1` (now 2.0.2 in venv) + persisted in `backend/requirements.txt:20`. Re-smoke succeeded.

## Stack traces

None post-fix.

## Conclusion

GPR-triggered energy-sector tilt is implemented end-to-end, tested with real data, and Q/A-verified. Production rollout (`gpr_signal_enabled=True`) requires `pip install -r backend/requirements.txt` to pick up the xlrd dep.

## Related artifacts

- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.3.md`, `phase-28.3-research-brief.md`
- `docs/design/phase-28.3-gpr-tilt.md`
- `backend/services/macro_regime.py`, `backend/config/settings.py`, `backend/requirements.txt`
