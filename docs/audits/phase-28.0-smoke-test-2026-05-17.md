# phase-28.0 Smoke Test — 2026-05-17

**Step:** phase-28.0 (Candidate Picker Expansion — drift fix)
**Date:** 2026-05-17
**Outcome:** PASS

## Scope

End-to-end harness smoke for the `min_market_cap` parameter removal in `backend.tools.screener.screen_universe()`. Goal: confirm the function still loads, signature is correct, and a real 3-ticker screening run produces expected output.

## Test 1: Immutable verification command (masterplan)

Command (from `.claude/masterplan.json::phase-28.steps[0].verification.command`):

```
source .venv/bin/activate && python -c "import ast,inspect; from backend.tools.screener import screen_universe; src=inspect.getsource(screen_universe); assert ('min_market_cap' in src and 'market_cap' in src.lower().split('def ')[-1]) or 'min_market_cap' not in src, 'param still dead'; print('PASS: min_market_cap is either used or removed')"
```

Output (verbatim):

```
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
PASS: min_market_cap is either used or removed
```

Exit code: 0. **PASS.**

## Test 2: Signature + live 3-ticker screen

Command:

```python
import inspect
from backend.tools.screener import screen_universe
sig = inspect.signature(screen_universe)
params = list(sig.parameters.keys())
print(f'screen_universe signature params: {params}')
assert 'min_market_cap' not in params
assert 'sector_lookup' in params
assert 'period' in params
print('PASS: min_market_cap removed; required params intact')

results = screen_universe(tickers=['AAPL','MSFT','NVDA'], period='1mo')
print(f'Returned {len(results)} results')
if results:
    print(f'First result keys: {list(results[0].keys())}')
    print(f'First ticker: {results[0]["ticker"]} price={results[0]["current_price"]} mom_1m={results[0]["momentum_1m"]}')
```

Output (verbatim):

```
screen_universe signature params: ['tickers', 'min_avg_volume', 'min_price', 'period', 'sector_lookup']
PASS: min_market_cap removed; required params intact

--- Smoke test (3 tickers) ---
Returned 3 results
First result keys: ['ticker', 'current_price', 'avg_volume_20d', 'momentum_1m', 'momentum_3m', 'momentum_6m', 'rsi_14', 'volatility_ann', 'sma_50_distance_pct']
First ticker: AAPL price=300.23 mom_1m=14.09
```

Exit code: 0. **PASS.**

## Test 3: Q/A subagent verdict

Subagent `qa` (Opus 4.7 xhigh per `.claude/agents/qa.md`) was spawned with the full 5-item harness-compliance audit + deterministic checks + LLM judgment. Returned:

```json
{
  "ok": true,
  "verdict": "PASS",
  "audit_items": {
    "researcher_gate": "PASS",
    "contract_before_generate": "PASS",
    "results_verbatim": "PASS",
    "log_last": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "deterministic_checks": [
    {"cmd": "immutable verification from masterplan", "exit": 0, "output_snippet": "PASS: min_market_cap is either used or removed"},
    {"cmd": "ast.parse screener.py", "exit": 0, "output_snippet": "OK"},
    {"cmd": "inspect.signature(screen_universe)", "exit": 0, "output_snippet": "['tickers', 'min_avg_volume', 'min_price', 'period', 'sector_lookup']"},
    {"cmd": "grep -rnE min_market_cap backend/ scripts/ tests/", "exit": 0, "output_snippet": "screener.py:83: docstring note only (documented removal note)"},
    {"cmd": "grep -c screen_universe at caller sites", "exit": 0, "output_snippet": "autonomous_loop=2, backtest=2, test_screener=10, verify_phase=6 (all callers intact)"}
  ],
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": false,
  "checks_run": 5
}
```

**PASS — no violations.**

## Stack traces / failures

None.

## Conclusion

Removing the dead `min_market_cap` parameter is non-breaking and verified end-to-end. The picker continues to operate identically; the brief and code now agree. Phase-28.0 status flipped to `done` in `.claude/masterplan.json`. Cycle 14 entry appended to `handoff/harness_log.md`.

## Related artifacts

- `handoff/current/contract.md` (immutable contract; rolling)
- `handoff/current/experiment_results.md` (verbatim verification + smoke output; rolling)
- `handoff/current/evaluator_critique.md` (Q/A verdict; rolling)
- `handoff/current/live_check_28.0.md` (filter-chain log line; persistent)
- `handoff/current/phase-28.0-research-brief.md` (research gate)
- `docs/design/phase-28.0-drift-fix.md` (design doc — paired with this audit)
- `backend/tools/screener.py` (the actual edit)
