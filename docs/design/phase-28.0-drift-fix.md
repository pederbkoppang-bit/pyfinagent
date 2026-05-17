# phase-28.0 — Design: Drift fix for unused `min_market_cap` parameter

**Step:** phase-28.0 (Candidate Picker Expansion — drift fix)
**Date:** 2026-05-17
**Effort:** XS (1-line removal + docstring note)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface (post-change)

`backend.tools.screener.screen_universe()` signature, before and after:

```python
# BEFORE (pre-edit)
def screen_universe(
    tickers: Optional[list[str]] = None,
    min_market_cap: float = 1e9,        # <-- dead, never used in body
    min_avg_volume: int = 100_000,
    min_price: float = 5.0,
    period: str = "6mo",
    sector_lookup: Optional[dict] = None,
) -> list[dict]: ...

# AFTER (post-edit, this step)
def screen_universe(
    tickers: Optional[list[str]] = None,
    min_avg_volume: int = 100_000,
    min_price: float = 5.0,
    period: str = "6mo",
    sector_lookup: Optional[dict] = None,
) -> list[dict]: ...
```

## Inputs / Outputs / Integration points

- **Inputs:** unchanged for all real callers (none pass `min_market_cap`). The function still accepts `tickers`, `min_avg_volume`, `min_price`, `period`, `sector_lookup` with the same defaults and semantics.
- **Outputs:** unchanged — same `list[dict]` shape with all the same fields (`ticker`, `current_price`, `avg_volume_20d`, `momentum_1m/3m/6m`, `rsi_14`, `volatility_ann`, `sma_50_distance_pct`, plus optional `sector`).
- **Integration points (callers):**
  - `backend/services/autonomous_loop.py:247` — `screen_universe(period="6mo")`
  - `backend/api/backtest.py:195` — `screen_universe()`
  - `tests/services/test_screener_sector_propagation.py` — exercises `sector_lookup` and `tickers` kwargs
  - `tests/verify_phase_23_1_13.py` — asserts `sector_lookup` is in the signature
  - None of these pass `min_market_cap`. All continue to work unchanged.

## Feature flag

Not applicable — this is a parameter removal, not a new feature. There is nothing to gate on.

## Test plan

1. **Immutable verification command** (from `.claude/masterplan.json::phase-28.steps[0].verification.command`):
   ```bash
   source .venv/bin/activate && python -c "import ast,inspect; from backend.tools.screener import screen_universe; src=inspect.getsource(screen_universe); assert ('min_market_cap' in src and 'market_cap' in src.lower().split('def ')[-1]) or 'min_market_cap' not in src, 'param still dead'; print('PASS: min_market_cap is either used or removed')"
   ```
2. **Signature smoke** — `inspect.signature(screen_universe).parameters` must include all of `[tickers, min_avg_volume, min_price, period, sector_lookup]` and must NOT include `min_market_cap`.
3. **Live screen smoke** — `screen_universe(tickers=['AAPL','MSFT','NVDA'], period='1mo')` must return 3 results with the expected field set.
4. **Q/A pass** — fresh `qa` subagent reads contract + experiment_results + research-brief + live_check, returns verdict.

All four passed; see `handoff/current/experiment_results.md` for verbatim outputs.

## Why REMOVE rather than WIRE

(From the research brief — see `handoff/current/phase-28.0-research-brief.md` and `docs/research/candidate_picker_improvements_2026-05-16.md`.)

1. **Universe already enforces $22.7B floor** by S&P 500 inclusion rules (effective July 1, 2025, per S&P Dow Jones Indices). A $1B default filter would never exclude anyone.
2. **Wiring would be expensive** — `yfinance.Ticker(t).info["marketCap"]` is a per-ticker HTTP request to a different endpoint than the batch `yf.download()`. With 500 tickers that's 500 extra requests per cycle, hitting Yahoo Finance's tightened 2024 rate limits (multiple GitHub issues #2125, #2288, #2325).
3. **Internal API, zero callers pass the param** — immediate removal is the standard Python practice for non-library code (per PEP 702 + PyAnsys deprecation guide).

If market-cap filtering is needed in the future (e.g., when phase-28.8 expands the universe to Russell 1000), the right path is a new explicit step that uses a cached BQ table of market caps rather than per-cycle yfinance polling.

## References

- `handoff/current/contract.md` — full contract with immutable success criteria
- `handoff/current/experiment_results.md` — verbatim verification output
- `handoff/current/phase-28.0-research-brief.md` — research gate (5 sources read in full, `gate_passed: true`)
- `handoff/current/live_check_28.0.md` — live filter-chain evidence
- `docs/audits/phase-28.0-smoke-test-2026-05-17.md` — smoke-test log
- `.claude/masterplan.json::phase-28.steps[0]` — immutable spec
