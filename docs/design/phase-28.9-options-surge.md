# phase-28.9 — Design: Options-flow OI-surge filter

**Step:** phase-28.9 (Candidate Picker Expansion — post-launch)
**Date:** 2026-05-17
**Effort:** M (new 165-line module + screener kwarg + autonomous_loop wiring + 9 settings fields)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend/services/options_flow_screen.py`:

```python
class OptionsSurgeSignal(BaseModel):
    ticker: str
    n_surges: int
    max_vol_oi_ratio: float
    max_vol_avg_ratio: float
    boost_multiplier: float
    surge_strikes: list[float]

async def fetch_oi_surge_signals(tickers, **thresholds) -> dict[str, OptionsSurgeSignal]: ...
def apply_options_surge_to_score(base, ticker, signals) -> float: ...
```

`screener.py:rank_candidates` extended with `options_surge_signals=None` kwarg.

## Predicate (Wayne State / J. Portfolio Mgmt)

For each strike in each near-expiry expiration:
- `strike > spot * 1.01` (OTM)
- `DTE in [2, 45]`
- `volume > max(5x avg-per-strike, 3x OI)`

Boost: 1.06 if ≥2 surges, 1.03 if 1 surge, 1.0 otherwise.

## Cost guard

Per-ticker `yf.Ticker.option_chain` — slow. Bounded to top 2×paper_screen_top_n (~20 tickers/cycle), not full universe. `Semaphore(4)` + 0.3s throttle.

## Feature flag

`options_flow_screen_enabled = False` default.

## Source rationale

- **Wayne State / J. Portfolio Mgmt** — near-expiry OTM call buys predictive (primary brief item #8)
- **Augustin-Brenner-Subrahmanyam** — informed options trading before M&A (related, supplement Gap 3)
- **Management Science 2026 (Neuhierl et al.)** — adds multi-factor caution; supports moderate boost magnitudes

## Calibration observation

5/5 mega-caps flagged at default thresholds today. Wayne State default is calibrated for median-cap stocks; mega-caps have higher base options activity. Operator may want vol_avg_mult=8.0 or vol_oi_mult=5.0 for mega-cap-heavy universes. Default-OFF protects production.

## References

- `handoff/current/phase-28.9-research-brief.md`
- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.9.md`
- `docs/audits/phase-28.9-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[9]`
