# Contract — phase-28.7 — Multidimensional momentum composite

**Step ID:** phase-28.7
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.7-research-brief.md` (`gate_passed: true`; 5 sources read in full incl. CFA Dec 2025, Stockopedia momentum rank, Review of Finance 2025, George-Hwang 52-week-high SSRN, Novy-Marx fundamental momentum).
- Internal audit: `rank_candidates` (screener.py:236-240) uses `mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25` price-only. `screen_universe` returns mom_1m/3m/6m + RSI + vol but NOT a 52w-high proximity field — need one new line in screen_universe: `pct_to_52w_high = current_price / close.rolling(252, min_periods=20).max().iloc[-1]`. SUE proxy = `pead_signal.surprise_score` (already attached as `pead_signal` dict to candidate when PEAD plug-in is enabled). Sector momentum from phase-28.12 `sector_momentum_ranks`.

## Hypothesis

Per CFA Dec 2025 + George-Hwang 2004 + Novy-Marx 2014: a multidimensional momentum composite that combines:
- Price momentum (the existing signal)
- 52w-high proximity (anchoring effect; stocks near their 52-week highs trend further)
- SUE momentum (earnings-surprise component; complements price)
- Sector/factor momentum (the leading sector ride)

…produces superior risk-adjusted returns vs naive price momentum alone, with materially lower crash risk during regime reversals.

**Weighting (per Researcher recommendation):** 0.35 price + 0.25 52w-high + 0.20 SUE + 0.20 sector. Cross-sectional z-score normalization across the screened universe before blending, to make scales commensurable.

## Immutable success criteria (from masterplan)

1. `composite_momentum_function_added`
2. `weighting_scheme_documented_with_source_citation`
3. `feature_flag_composite_momentum_enabled_default_false`
4. `live_check_compares_naive_vs_composite_top10_for_one_cycle`

Immutable verification command:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')" && grep -qE '52.{0,5}week|fifty.two|composite_momentum' backend/tools/screener.py
```

Immutable live_check shape:
> "live_check_28.7.md: cycle log + scoring diff table (price-only vs composite) for top-10"

## Plan steps

1. **Settings**: 5 new fields (enabled flag + 4 weight knobs).
2. **screener.py:screen_universe**: add `pct_to_52w_high` field to results (one line: `current_price / close.rolling(252, min_periods=20).max().iloc[-1]`).
3. **screener.py: new helper `_apply_multidim_momentum(scored, weights)`**: computes cross-sectional z-scores across the 4 components AND assigns blended composite_score. Preserves `composite_score_raw`. Stocks missing a component get 0 (mean) for that component.
4. **screener.py:rank_candidates**: add `multidim_momentum=False` kwarg + `multidim_weights` kwarg; call `_apply_multidim_momentum` AFTER all overlays (28.1/28.12/etc.) but BEFORE the sector-neutral percentile pass (28.4) — sector-neutral can be layered ON TOP of multidim if both enabled.
5. **autonomous_loop.py**: pass `multidim_momentum=settings.multidim_momentum_enabled` and `multidim_weights={...}` to rank_candidates.
6. Verify + smoke + Q/A + log + flip.

## References

- `handoff/current/phase-28.7-research-brief.md`
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief item #11)
- `.claude/masterplan.json::phase-28.steps[7]`

## Risk / blast radius

- **Default OFF** — `multidim_momentum_enabled = False`.
- **Back-compat** — `multidim_momentum=False` default; existing callers unchanged.
- **Composite score scale changes** under multidim mode (z-blend in roughly [-3, 3] vs raw points). meta_scorer treats composite_score as ranking signal; should still work.
- **52w_high field** added to screen_universe results — additional ~1ms per ticker; no new yfinance call (uses existing OHLCV download).
- **Cost** — zero LLM, no new network.
