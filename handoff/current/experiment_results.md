# Phase 4.3 Risk Management — Experiment Results

**Date:** 2026-04-14
**Session:** Ford remote, Opus 4.6
**Step:** 4.3 Risk Management (code-only subset)
**Status:** GENERATE complete, all 20 assertions PASS

## What was built

Three new public methods on `SignalsServer`, plus an extension to
`get_risk_constraints()`, plus a localised change to step 5 of `publish_signal`.
Everything in `backend/agents/mcp_servers/signals_server.py`. Stdlib-only.

1. **`size_position(signal, portfolio) -> float`** — hybrid lite-formula
   sizing. Takes the minimum of up to three independent caps:
   (a) hard percent-of-equity cap (`equity * max_position_pct/100`, then
   floored to `max_position_usd`),
   (b) confidence-weighted half-Kelly arm (`0.5 * confidence * equity`,
   degraded-edge form),
   (c) inverse-vol arm (`target_vol_pct/100 * equity / annualized_vol`).
   Only (a) is always computed; (b) and (c) are skipped when their inputs
   are missing / invalid (no fake substitutes). Explicit `signal.size_usd`
   still overrides. Returns 0.0 for non-BUY actions, non-dict input, or
   zero equity. Never raises.

2. **`check_stop_loss(portfolio) -> list[dict]`** — pure detector of
   positions breaching per-position fixed stop (8% from entry, O'Neil
   canonical) or trailing stop (3% off running peak, Chandelier-lite).
   Returns `[{ticker, reason, entry_price, current_price, peak_price,
   loss_pct}, ...]`. Does NOT emit liquidating orders — that wiring is a
   Phase 4.3 follow-up documented in the contract. Malformed positions
   are skipped silently. Never raises.

3. **`track_drawdown(portfolio) -> dict`** — equity-curve drawdown tracker.
   Lazily initialises `self._peak_equity`, updates it on new highs,
   computes `(equity - peak)/peak * 100` and classifies into the
   5%/10%/15% ladder: `ok` / `warning` / `derisk` / `kill`. Mirrors
   QuantConnect's `MaximumDrawdownPercentPortfolio` reference model.
   Returns `{peak, equity, drawdown_pct, tier, kill_switch}`. In-memory
   only this session; durable state is Phase 4.2.

4. **`get_risk_constraints()` extension** — added 6 new keys (the existing
   5 keys are unchanged):
   `max_position_pct=5.0`, `max_position_usd=1000.0`, `stop_loss_pct=8.0`,
   `trail_stop_pct=3.0`, `drawdown_warning_pct=-5.0`, `drawdown_derisk_pct=-10.0`.
   `max_drawdown_pct=-15.0` stays as the kill switch.

5. **`publish_signal` step 5 update** — replaced the inline v1 sizing
   (`min(cash*0.05, 1000.0)`) with a call to `self.size_position(signal,
   portfolio)`. Steps 1-4 and 6-9 are untouched. Explicit `size_usd`
   override path is preserved inside `size_position` itself.

6. **`__init__` state add** — `self._peak_equity: Optional[float] = None`.
   Zero side-effects at construction (lazy init on first `track_drawdown`
   call).

## Deterministic assertions (20/20 PASS)

Run directly against the real `SignalsServer` instance in stub mode (no
LLM, no backend deps, no network).

| # | name | expected | actual | result |
|---|------|----------|--------|--------|
| 1 | size_position_a1 (zero equity) | 0.0 | 0.0 | PASS |
| 2 | size_position_a2 (hard cap at conf=1.0) | 500.0 | 500.0 | PASS |
| 3 | size_position_a3 (kelly arm bounded by hard cap) | 500.0 | 500.0 | PASS |
| 4 | size_position_a4 (vol arm, hard cap still dominant) | 1000.0 | 1000.0 | PASS |
| 5 | size_position_a5 (HOLD action) | 0.0 | 0.0 | PASS |
| 6 | size_position_a6 (non-dict signal) | 0.0 | 0.0 | PASS |
| 7 | size_position explicit size_usd override | 250.0 | 250.0 | PASS |
| 8 | check_stop_loss_a1 (empty positions) | [] | [] | PASS |
| 9 | check_stop_loss_a2 (9% loss triggers fixed_stop) | fixed_stop -9.0% | fixed_stop -9.0% | PASS |
| 10 | check_stop_loss_a3 (7% loss, no trigger) | [] | [] | PASS |
| 11 | check_stop_loss_a4 (peak 120 -> current 116) | trailing_stop -3.33% | trailing_stop -3.33% | PASS |
| 12 | check_stop_loss_a5 (non-dict portfolio) | [] | [] | PASS |
| 13 | track_drawdown_a1 (first call sets peak) | peak=10000 dd=0 ok | peak=10000 dd=0 ok | PASS |
| 14 | track_drawdown_a2 (equity=9500 -> warning) | dd=-5.0 warning | dd=-5.0 warning | PASS |
| 15 | track_drawdown_a3 (equity=8900 -> derisk) | dd=-11.0 derisk | dd=-11.0 derisk | PASS |
| 16 | track_drawdown_a4 (equity=8400 -> kill) | dd=-16.0 kill kill_switch=True | dd=-16.0 kill kill_switch=True | PASS |
| 17 | track_drawdown_a5 (equity=11000 -> new high) | peak=11000 dd=0 ok | peak=11000 dd=0 ok | PASS |
| 18 | constraints_a1 (6 new keys present) | all present | all present | PASS |
| 19 | publish_signal_a1 (stub path preserved) | reason=backend_unavailable | reason=backend_unavailable | PASS |
| 20 | ast_logger_ascii | 0 violations | 0 violations | PASS |

## Static verification

- `python3 -c "import ast; ast.parse(...)"` — CLEAN
- `python3 -m py_compile` — CLEAN
- AST logger ASCII scan — 0 non-ASCII chars in any logger call
- Module import in stub mode — CLEAN (stub banner logged as expected)

## Diff stats

```
 backend/agents/mcp_servers/signals_server.py | 349 +++++++++++++++++++++++++-
 1 file changed, 336 insertions(+), 13 deletions(-)
```

**Soft note on line bound:** Contract budgeted `< 300` added lines; actual
is 336 (+12%). The overage is entirely in docstrings — each new method
carries the research-justification paragraph the anti-leniency rules
require. I flag it here rather than hide it; QA may accept or reject.

## Scope compliance

- Files touched: `backend/agents/mcp_servers/signals_server.py` — the only
  one the contract permits.
- No imports added. Stdlib only. No pandas / numpy / pydantic / backend
  module imports.
- `risk_check`, `validate_signal`, `generate_signal`, `get_portfolio`,
  `get_signal_history`, `_risk_response`, `_signal_id`, `_empty_response`,
  `_remember` — all UNCHANGED.
- `publish_signal` steps 1-4 and 6-9 — UNCHANGED. Step 5's 10-line sizing
  block collapsed to a 1-line `size_position()` call.
- `__init__` gains one attribute (`_peak_equity`); no signature change.
- `get_risk_constraints()` gains 6 keys; no existing key mutated.

## Out-of-scope deferrals (with explicit reasoning)

| Deferred item | Why | Target phase |
|---|---|---|
| BQ persistence of `_peak_equity` | needs schema migration | 4.2 |
| Real mu_hat / var_hat from backtest for full-Kelly | needs backtest query | 3.2 follow-up |
| ATR-based trailing stop (Chandelier full form) | needs historical price series | 3.2 follow-up |
| Emit liquidating orders on stop-loss trigger | needs paper_trader wiring | 4.3 follow-up |
| Consecutive-stop counter for pause | needs persistent state | 4.3 follow-up |
| Event calendar (earnings / FOMC) integration | needs earnings_tone plumbing | 4.3.3 |
| Sector exposure 30% cap | needs sector mapping in portfolio | 4.3.1 follow-up |

## Research citations (justification, not ornament)

- Kaminski & Lo (2014), "When Do Stop-Loss Rules Stop Losses?" — stops
  only add value under momentum; our backtest exhibits the persistence,
  so keeping stops is justified.
- CFA Institute (2018), "The Kelly Criterion: You Don't Know the Half of
  It" — half-Kelly captures ~75% growth at ~50% variance; justifies the
  0.5 coefficient on the Kelly arm.
- William O'Neil, "How to Make Money in Stocks" — 7-8% stop rule as the
  canonical per-position fixed-stop floor.
- QuantConnect LEAN `MaximumDrawdownPercentPortfolio.py` — reference
  implementation for the equity-curve drawdown tracker; our tier cascade
  (ok/warning/derisk/kill) is the tiered-extension convention found across
  QuantifiedStrategies / Robot Wealth / QuantVPS practitioner literature.
- 17 CFR 240.15c3-5 / SEC 34-63241 — stops are post-trade SOFT triggers,
  NOT pre-trade fatal blocks; this justifies their placement outside the
  `risk_check` predicate hierarchy. (`risk_check` keeps the pre-trade
  fatal checks; `check_stop_loss` is called post-fill by the caller.)

## Next

EVALUATE phase — hand off to qa-evaluator (Opus) for independent
cross-verification. Assertions above are reproducible; the QA run should
re-execute them in an isolated Python process, audit the code for the
anti-leniency rules, and return a verdict.
