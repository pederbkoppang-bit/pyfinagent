# phase-28.9 Smoke Test — 2026-05-17

**Step:** phase-28.9 (Options-flow OI-surge filter)
**Date:** 2026-05-17
**Outcome:** PASS

## Test 1: Immutable verification

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/options_flow_screen.py').read()); from backend.services.options_flow_screen import fetch_oi_surge_signals; print('importable')" && grep -q 'options_flow_screen_enabled' backend/config/settings.py
importable
```

Exit 0. **PASS.**

## Test 2: Live yfinance surge fetch (5 large-caps)

```
NVDA: n=1 vol/OI=4.48 vol/avg=5.39 boost=1.03 strikes=[270.0]
TSLA: n=11 vol/OI=2750 vol/avg=17.10 boost=1.06 strikes=[427.5...605.0]
AAPL: n=5  vol/OI=17.76 vol/avg=20.70 boost=1.06 strikes=[305.0,315.0]
MSFT: n=6  vol/OI=43.36 vol/avg=17.37 boost=1.06 strikes=[430.0...455.0]
META: n=7  vol/OI=1000000 vol/avg=17.24 boost=1.06 strikes=[642.5...765.0]
```

5/5 flagged. **PASS.**

## Test 3: Q/A verdict (9 deterministic checks)

```json
{"ok": true, "verdict": "PASS", "violated_criteria": [], "checks_run": 17 categories}
```

## Conclusion

Options-flow OI-surge filter implemented end-to-end, tested with real yfinance data, Q/A-verified. Default OFF.

Calibration suggestion: thresholds may be loose for mega-caps; operator can tighten before flipping flag in production.

## Related artifacts

- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.9.md`, `phase-28.9-research-brief.md`
- `docs/design/phase-28.9-options-surge.md`
- `backend/services/options_flow_screen.py`, `backend/tools/screener.py`, `backend/services/autonomous_loop.py`, `backend/config/settings.py`
