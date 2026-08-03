# Contract -- step 82.2

**Step id:** 82.2 (phase-82) | **Priority:** P1 | depends_on: ['82.0']
PLAN phase, before GENERATE.

## Research gate

`handoff/current/research_brief_82.2.md` -- **gate_passed: true** (tier=complex,
audit_class=false, 6 read-in-full, 15 snippet-only, 21 URLs, recency scan done,
6 internal files).

Main independently verified every load-bearing claim:

| claim | verified |
|---|---|
| `_compute_quality_momentum_label` (:1124) and `_compute_factor_label` (:1208) carry NO forward information | **CONFIRMED.** Both classify on features at `entry_date` and never fetch a forward price, unlike `_compute_triple_barrier_label` (:753-754) which fetches `entry_date + holding_days*1.5`. Queued as **82.16** |
| `compute_turbulence_index` exists with zero callers | CONFIRMED at `backend/backtest/historical_data.py:281`; grep returns only the definition |
| `cpi_yoy` is the raw CPIAUCSL index LEVEL, not a rate | CONFIRMED at `backend/backtest/historical_data.py:271` (`macro.get("CPIAUCSL").get("value")`, measured 332.57). **Do not threshold it as a percentage** |
| `sma_50_distance` is a fraction, so -0.05/+0.10 are internally consistent | CONFIRMED at `backend/backtest/historical_data.py:105`. My earlier "raw-percent units bug" framing was WRONG -- there is no unit inconsistency. The design critique (a fixed fraction is not vol-scaled) stands separately |
| crowded MOMENTUM has 0.38x crash probability; crowded reversal 1.7-1.8x | accepted from arXiv:2512.11913. **Corrects Main's earlier framing to the operator** -- "the book is in the crowded trade therefore dangerous" is NOT supported for a momentum book. Key the regime on TURBULENCE/co-movement, not on "momentum has run" |

## Hypothesis

Three forward-looking, cost-adjusted, sigma-scaled label methods can be added to
`STRATEGY_REGISTRY` that produce non-degenerate labels, covering the four
overpriced-market lenses, without touching the live funnel.

## Design (all three FORWARD-LOOKING -- the 82.16 defect must not be reproduced)

**`stretch_regime`** (lens a + lens d folded in). Barriers at +/- k*sigma over
the horizon, where sigma comes from `annualized_volatility` in the feature
vector. A market-stretch scalar is computed per entry_date from trailing **SPY**
realized volatility against its own longer-run average (SPY is always preloaded
alongside the universe -- see `.claude/rules/backend-backtest.md`). As stretch
rises the UP barrier widens and the DOWN barrier tightens: fewer +1 labels, so
the trained model buys less and the trader holds more cash. That IS the
cash-timing overlay, expressed inside the label rather than bolted on.
**Deliberately avoids `compute_turbulence_index`** because that needs the
universe list, which is not on `self`; using SPY keeps this step to
`backtest_engine.py` alone and requires no edit to `_run_window`.

**`qarp`** (lens b). Long-only. Gate on fundamentals AT entry_date -- cheap
(`pe_ratio`, `fcf_yield`), quality (`roe`, `profit_margin`), low leverage
(`debt_equity`). A name failing the gate returns **`None`**, not 0, so
non-candidates are EXCLUDED from training rather than flooding the neutral
class. Names passing the gate get a forward sigma-barrier. Does NOT reuse
`quality_score` (its payout leg is the weakest QMJ dimension and rests on a
`fcf_yield` computed with capex=0).

**`reversion_sigma`** (lens c). Replaces the raw-fraction thresholds with
sigma-units: stretch = `sma_50_distance / daily_sigma`. No-signal returns
**`None`** (the existing `mean_reversion` returns 0 on two of three paths --
`:1206` -- which is a principal degeneracy driver). Forward validation that the
reversion actually materialises, cost-adjusted like the triple barrier
(`:763-765`); the existing MR label has NO cost adjustment.

## Immutable success criteria (verbatim from .claude/masterplan.json)

1. three new strategy names are present in STRATEGY_REGISTRY, each mapping to a label method that exists on BacktestEngine
2. each of the three label methods returns a non-degenerate label distribution on a named committed fixture: no single label class exceeds 95 percent of rows
3. a measurement of the existing mean_reversion label distribution on the same fixture is recorded, establishing whether the ~all-neutral report is true
4. the hedged/cash-timing overlay is implemented as an overlay on one named candidate and a test asserts it changes that candidate's exposure on a stretch fixture

**Verification command:** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_2_candidate_strategies.py -q`

## Plan

1. Add the three label methods to `backend/backtest/backtest_engine.py` and
   register them in `STRATEGY_REGISTRY`.
2. Commit a deterministic fixture (no BQ dependency) and MEASURE the label
   distribution of all three PLUS the existing `mean_reversion`.
3. Report the mean_reversion per-stage funnel (how many rows die at the signal
   gate vs the forward-validation gate), not just the final histogram.
4. Tests: registry membership, non-degeneracy (`no class >95%` AND a
   **minimum row count**, or a 3-row set passes trivially), forward-information
   (label must change when post-entry prices are mutated), overlay effect.
5. Fresh Q/A.

## Out of scope

No live-funnel change. No backtest RUN (that is 82.3, and it is blocked on
**82.15** -- the `realtime_start` vintage column still has zero consumers, so a
macro-conditioned backtest would carry ~120d look-ahead). `stretch_regime` uses
SPY prices rather than macro, which sidesteps the vintage issue for THIS step
but does not remove the 82.3 dependency.
