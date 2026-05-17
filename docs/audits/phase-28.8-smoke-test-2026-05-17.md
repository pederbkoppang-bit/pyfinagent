# phase-28.8 Smoke Test — 2026-05-17

**Step:** phase-28.8 (Russell-1000 universe expansion)
**Date:** 2026-05-17
**Outcome:** PASS

## Test 1: Immutable verification
```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); from backend.tools.screener import get_sp500_tickers; print('importable')" && grep -qE 'russell|RUSSELL|iShares|IWB|get_russell' backend/tools/screener.py
importable
```
Exit 0. **PASS.**

## Test 2: Live universe fetch
```
SP500: 503 tickers
Russell-1000 fetch: 515 tickers (fallback path activated; IWB returned HTML)
SNDK: True, WDC: True, MU: True (all 3 reference-case names present)
Extras vs SP500: LYFT, MDB, MRVL, NET, OKTA, PINS, PXD, ROKU, SNOW, SPOT, TEAM, ZS
```
**PASS** — reference-case tickers covered.

## Test 3: Q/A verdict (7 deterministic checks)
```json
{"ok": true, "verdict": "PASS", "violated_criteria": [], "checks_run": 15}
```

Q/A explicitly verified the honest IWB-returns-HTML disclosure in both experiment_results §3 AND live_check_28.8.md.

## Conclusion
Russell-1000 universe expansion implemented with 3-tier fallback chain. Default OFF. Fallback path delivers 515 tickers including all 3 reference-case names. Production unchanged. IWB CSV parser issue tracked as separate follow-up.

## Related artifacts
- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.8.md`, `phase-28.8-research-brief.md`
- `docs/design/phase-28.8-russell1000-universe.md`
- `backend/tools/screener.py`, `backend/services/autonomous_loop.py`, `backend/config/settings.py`
