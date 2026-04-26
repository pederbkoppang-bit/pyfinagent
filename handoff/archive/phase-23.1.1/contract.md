---
step: phase-23.1.1
title: Daily macro regime filter (LLM-as-judge over FRED) + screener conviction multiplier
cycle_date: 2026-04-26
harness_required: true
verification: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.macro_regime import compute_macro_regime; r = asyncio.run(compute_macro_regime(use_cache=False)); assert r.regime in {\"risk_on\",\"risk_off\",\"mixed\",\"unknown\"}; assert 0.5 <= r.conviction_multiplier <= 1.5; assert isinstance(r.series_used, list) and len(r.series_used) >= 3; assert len(r.rationale) <= 300; print(\"ok regime=\" + r.regime + \" mult=\" + str(r.conviction_multiplier))"'
research_brief: handoff/current/phase-23.1.1-research-brief.md
---

# Contract — phase-23.1.1

## Hypothesis

A single daily LLM call over a 7-FRED-series snapshot produces a stable risk_on/risk_off/mixed/unknown regime tag plus a conviction multiplier that, when applied to `screener.rank_candidates`, raises portfolio Sharpe in risk-off regimes by reducing position-sizing during conditions where momentum signals are noisier. Cost ceiling: <$0.05/day in LLM inference.

## Plan

1. **Add 2 missing FRED series** — extend `backend/tools/fred_data.py:16-24` `SERIES` dict with `VIXCLS` (CBOE VIX) and `BAMLH0A0HYM2` (ICE BofA HY OAS).
2. **New `backend/services/macro_regime.py`:**
   - `MacroRegimeOutput` Pydantic model exactly per the schema in research-brief §"Concrete Schema Recommendation"
   - `compute_macro_regime(use_cache=True) -> MacroRegimeOutput` async function: pulls T10Y2Y/VIXCLS/BAMLH0A0HYM2/FEDFUNDS/CPIAUCSL/UNRATE/INDPRO from FRED → builds a structured prompt with current values, thresholds, and trend → calls Claude Haiku 4.5 via existing `llm_client.generate_content` with structured output → returns `MacroRegimeOutput`.
   - 24-hour file cache at `backend/services/_cache/macro_regime.json` (avoid re-billing the LLM on every screener call).
3. **Extend `backend/tools/screener.py:151` `rank_candidates`:**
   - Add `regime: Optional[MacroRegimeOutput] = None` parameter.
   - After the composite `score` is computed (line ~198), apply `score *= regime.conviction_multiplier` if regime present.
   - Also apply sector tilt: if a candidate's sector ETF is in `regime.sector_hints.overweight`, multiply score by 1.05; if in `underweight`, by 0.95. Use the existing `SECTOR_ETFS` map at `screener.py:20-25`.
4. **Wire into `backend/services/autonomous_loop.py` Step 1:**
   - Before line 113 (`screen_data = screen_universe(period="6mo")`), call `regime = await compute_macro_regime()` (ENABLED via new settings flag, default False so existing behaviour is preserved by default).
   - Pass `regime` to `rank_candidates` call (current call is around line ~117 / `top_candidates = rank_candidates(screen_data, top_n=...)`).
   - Log a single `Macro regime: <tag> conviction=<x> mult=<y>` line so it's visible in cycle logs.
5. **New settings flag** in `backend/config/settings.py`:
   - `macro_regime_filter_enabled: bool = Field(False, description="Apply daily macro regime as a conviction multiplier in screener rank_candidates")`
   - `macro_regime_model: str = Field("claude-haiku-4-5", description="LLM used for daily macro regime classification")`
6. **Tests** at `tests/services/test_macro_regime.py`:
   - Unit: `MacroRegimeOutput` validates conviction in [0,1], multiplier in [0.5,1.5], regime in valid enum.
   - Unit: `_apply_to_score(score=10.0, regime=risk_off mult=0.7)` returns 7.0.
   - Integration (mocked LLM): `compute_macro_regime` with stubbed FRED returns + stubbed Claude response → `MacroRegimeOutput`.
   - Cache: second call within 24h returns cached object without hitting LLM.
7. **Run the immutable verification command** (in front-matter) — must print `ok regime=<tag> mult=<x>` and exit 0.

## Out of scope

- The daily APScheduler job registration (phase-23.1.4-or-later; out of scope here — call lives in `autonomous_loop` Step 1, which already runs daily)
- UI surface in Settings + Signals page (phase-23.1.6)
- Backtest validation of the conviction multiplier (phase-23.2.5, future cycle)
- Existing `regime_detection_enabled` flag at settings.py:84 — different feature (VIXRollingQuantile for spot_checks_harness); not touched.
- Polygon / Benzinga data vendor work — Phase 2.

## Files modified

- `backend/tools/fred_data.py` — add 2 series to dict
- `backend/services/macro_regime.py` — NEW (~150 LOC)
- `backend/tools/screener.py` — extend rank_candidates signature + score multiplier
- `backend/services/autonomous_loop.py` — Step 1 regime call (3 lines)
- `backend/config/settings.py` — 2 new fields
- `tests/services/test_macro_regime.py` — NEW
- `.claude/masterplan.json` — add phase-23.1.1 step entry
- `handoff/current/{contract,experiment_results,evaluator_critique}.md` — rolling

## Verification

The immutable verification command (front-matter) is the contract criterion. It calls the actual LLM (no mocks), so it confirms:
- The FRED pull works
- The Claude structured-output call returns a valid `MacroRegimeOutput`
- The conviction_multiplier is in the [0.5, 1.5] range
- At least 3 FRED series were available (resilient to one-off vendor outages)
- The rationale is bounded at 300 chars (no LLM run-on)

## References

- `handoff/current/phase-23.1.1-research-brief.md` — full research brief (363 lines, 7 sources read in full, gate_passed: true)
- `handoff/current/phase-23.1-external-research.md` — Phase-23.1 master research (40 KB, 11 sources)
- `backend/tools/fred_data.py:16-24` — current FRED series dict
- `backend/services/autonomous_loop.py:109-115` — Step 1 entry point
- `backend/tools/screener.py:151` — rank_candidates extension surface
- `backend/llm/llm_client.py:895-911` — structured output infra
- `backend/config/settings.py:84` — existing `regime_detection_enabled` flag (different feature; not touched)
