# Experiment Results — phase-28.0 — Drift fix: remove unused `min_market_cap`

**Step ID:** phase-28.0
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

A single one-parameter removal from `screen_universe()` plus a docstring note explaining the change.

### Files modified

| File | Change |
|---|---|
| `backend/tools/screener.py` | Removed `min_market_cap: float = 1e9,` from `screen_universe` signature (line 65). Added phase-28.0 note in the function docstring explaining the removal and the $22.7B S&P 500 inclusion floor (per S&P Dow Jones Indices, July 2025). |

### Files created

| File | Purpose |
|---|---|
| `handoff/current/phase-28.0-research-brief.md` | Research-gate brief; Researcher subagent; 5 sources read in full; `gate_passed: true`. |
| `handoff/current/contract.md` | This step's contract (rolling). |
| `handoff/current/experiment_results.md` | This file (rolling). |
| `handoff/current/live_check_28.0.md` | Live smoke evidence — screener filter chain ran without market_cap. |

### Files NOT modified

- `backend/services/autonomous_loop.py:247` — caller; uses only `period="6mo"`. No change required.
- `backend/api/backtest.py:195` — caller; no kwargs. No change required.
- `tests/services/test_screener_sector_propagation.py` — exercises `sector_lookup`, not `min_market_cap`. No change required.
- `tests/verify_phase_23_1_13.py` — asserts `sector_lookup` in signature, not `min_market_cap`. No change required.
- `scripts/ablation/run_ablation.py:252` — calls a different private method `engine._screen_universe`, not the public `tools.screener.screen_universe`. No change required.

---

## Verification — verbatim output

### 1. Immutable verification command (from `.claude/masterplan.json::phase-28.steps[0].verification.command`)

```
$ source .venv/bin/activate && python -c "import ast,inspect; from backend.tools.screener import screen_universe; src=inspect.getsource(screen_universe); assert ('min_market_cap' in src and 'market_cap' in src.lower().split('def ')[-1]) or 'min_market_cap' not in src, 'param still dead'; print('PASS: min_market_cap is either used or removed')"
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
PASS: min_market_cap is either used or removed
$ echo $?
0
```

EXIT 0. PASS.

### 2. Signature smoke + 3-ticker live screen

```
$ source .venv/bin/activate && python -c "
import inspect
from backend.tools.screener import screen_universe
sig = inspect.signature(screen_universe)
params = list(sig.parameters.keys())
print(f'screen_universe signature params: {params}')
assert 'min_market_cap' not in params, f'param still present: {params}'
assert 'sector_lookup' in params, 'sector_lookup must remain'
assert 'period' in params, 'period must remain'
print('PASS: min_market_cap removed; required params intact')
print()
print('--- Smoke test (3 tickers) ---')
results = screen_universe(tickers=['AAPL','MSFT','NVDA'], period='1mo')
print(f'Returned {len(results)} results')
if results:
    print(f'First result keys: {list(results[0].keys())}')
    print(f'First ticker: {results[0].get(\"ticker\")} price={results[0].get(\"current_price\")} mom_1m={results[0].get(\"momentum_1m\")}')
"
/Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python3.14/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.4.3)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
screen_universe signature params: ['tickers', 'min_avg_volume', 'min_price', 'period', 'sector_lookup']
PASS: min_market_cap removed; required params intact

--- Smoke test (3 tickers) ---
Returned 3 results
First result keys: ['ticker', 'current_price', 'avg_volume_20d', 'momentum_1m', 'momentum_3m', 'momentum_6m', 'rsi_14', 'volatility_ann', 'sma_50_distance_pct']
First ticker: AAPL price=300.23 mom_1m=14.09
```

PASS. New signature is `(tickers, min_avg_volume, min_price, period, sector_lookup)` — `min_market_cap` is gone, all other parameters preserved. Live screener still returns expected fields and momentum.

---

## Artifact shape

Post-edit `screen_universe` signature:

```python
def screen_universe(
    tickers: Optional[list[str]] = None,
    min_avg_volume: int = 100_000,
    min_price: float = 5.0,
    period: str = "6mo",
    sector_lookup: Optional[dict] = None,
) -> list[dict]:
```

Docstring addition (the phase-28.0 note):

> phase-28.0 (2026-05-17): removed unused `min_market_cap` parameter. The parameter was accepted but never applied in the function body (only price + volume filters fire). Zero callers passed it. Re-add via a separate explicit step if market-cap filtering is needed beyond the inherent S&P 500 inclusion floor (currently ~$22.7B per S&P DJI 2024 methodology update).

---

## Success criteria mapping

| Criterion (from masterplan, immutable) | Evidence | Result |
|---|---|---|
| `min_market_cap_parameter_either_applied_or_removed` | New signature has no `min_market_cap`; docstring explains removal | PASS |
| `syntax_OK` | `python -c "import ast; ast.parse(open(...).read())"` exit 0; `import` succeeds | PASS |
| `no_regression_in_existing_screener_callsites` | All 4 callers in repo confirmed not to pass `min_market_cap`; live 3-ticker screen returns 3 results with all expected fields | PASS |

---

## Next

Q/A pass via fresh `qa` subagent (reads this file + contract.md + research brief + live_check_28.0.md). On PASS, append harness_log entry and flip status to `done`.
