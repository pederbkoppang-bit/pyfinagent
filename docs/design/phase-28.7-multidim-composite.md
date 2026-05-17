# phase-28.7 — Design: Multidimensional momentum composite

**Step:** phase-28.7 (Candidate Picker Expansion — post-launch)
**Date:** 2026-05-17
**Effort:** M (3 files + new pct_to_52w_high field + 2 new helpers + 5 settings fields)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend/tools/screener.py` additions:

```python
# In screen_universe per-ticker loop:
pct_to_52w_high = current_price / close.rolling(252, min_periods=20).max().iloc[-1]
row["pct_to_52w_high"] = round(pct_to_52w_high, 4)

# New module-level helpers:
def _zscore(values: list[float]) -> list[float]: ...

def _apply_multidim_momentum(
    scored: list[dict],
    weights: dict[str, float],
    pead_signals: Optional[dict] = None,
    sector_momentum_ranks: Optional[dict] = None,
) -> None:
    """Z-blends 4 components into composite_score in place. Preserves composite_score_raw."""

# rank_candidates new kwargs:
multidim_momentum: bool = False
multidim_weights: Optional[dict[str, float]] = None
pead_signals_lookup = None
```

## Components & weights

| Component | Source | Weight | Z-score reason |
|---|---|---|---|
| Price momentum | existing composite_score (mom_1m/3m/6m + RSI/vol) | 0.35 | Tilt toward calibrated baseline |
| 52w-high proximity | new `pct_to_52w_high` in screen_universe | 0.25 | George-Hwang 2004 anchoring |
| SUE (PEAD surprise) | `pead_signals[ticker].surprise_score` | 0.20 | Novy-Marx 2014 earnings momentum |
| Sector boost | `sector_momentum_ranks[sector].boost_multiplier - 1.0` | 0.20 | phase-28.12 / Quantpedia |

Cross-sectional z-score normalization across the universe makes scales commensurable. Missing components → 0 (mean). Std=0 → 0 z-score.

## Feature flag

`multidim_momentum_enabled = False`. Production unchanged.

## Test plan

1-8 all passed (immutable verification, 3-file syntax, settings, helper imports, pct_to_52w_high in screen_universe, signature kwargs, _zscore unit, _apply_multidim_momentum 5-candidate unit + mutation test). Q/A pass.

Mutation-test rounding artifact (1.36e-05) is from `round(..., 4)` — not a bug; PASS at 1e-3 tolerance.

## Source rationale

- **CFA Institute Dec 2025** — primary brief item #11 — multidim composite outperforms naive price momentum with materially lower crash risk
- **George-Hwang 2004** — 52w-high anchoring (52w-high proximity predicts continuation)
- **Novy-Marx 2014** — earnings momentum complements price momentum
- **Quantpedia sector momentum** — sector boost component reuses phase-28.12 ranks

## Operator note

Z-blend produces composite_score values roughly in [-3, +3] (z-score range) instead of raw points. `meta_scorer.py` reads composite_score as ranking signal — relative order preserved. `composite_score_raw` is added on each candidate for transparency / downstream inspection.

## References

- `handoff/current/phase-28.7-research-brief.md`
- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.7.md`
- `docs/audits/phase-28.7-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[7]`
