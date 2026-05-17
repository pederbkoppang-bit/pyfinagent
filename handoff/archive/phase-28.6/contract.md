# Contract — phase-28.6 — Crude-oil (CL=F) cross-asset trend signal

**Step ID:** phase-28.6
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.6-research-brief.md` (`gate_passed: true`, 6 sources read in full)
- Internal audit: phase-28.3 added `_fetch_gpr_acts` + `_apply_gpr_tilt` + post-LLM hook (lines 100, 184, 368-383 in macro_regime.py). The same `_apply_gpr_tilt` is GENERIC over the trigger info dict (only uses `above_threshold` key), so the crude trigger can REUSE it without duplication.
- yfinance>=0.2.40 is already a dep (no install needed).

## Hypothesis

When WTI crude (`CL=F`) shows elevated upward momentum (z-score > 1.0 over the rolling 12-month percent-change distribution), US energy majors (XOM, CVX, COP, OXY = 39% of XLE) typically follow within 1-3 weeks. Adding this as a SECONDARY trigger to `MacroRegimeOutput.sector_hints.overweight` provides a path to XLE-overweight even when GPR-Acts is below its threshold (e.g., supply-side or demand-side oil rallies that don't show up as geopolitical events).

Per Researcher: GPR and crude momentum are ORTHOGONAL (high GPR + flat oil, and rising oil + low GPR both occur regularly), so this is a genuinely additive signal, not a duplicate.

## Immutable success criteria (from `.claude/masterplan.json::phase-28.steps[6].verification.success_criteria`)

1. `crude_oil_trend_signal_added_to_macro_regime`
2. `threshold_documented`
3. `fallback_when_yfinance_unavailable_does_not_break_cycle`
4. `live_check_shows_oil_trend_value_and_resulting_sector_action`

Immutable verification command:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')" && grep -qE 'CL=F|crude|brent|oil_trend' backend/services/macro_regime.py
```

Immutable live_check shape:
> "live_check_28.6.md: cycle log showing CL=F 1m momentum + threshold check + sector_hints diff"

## Plan steps

1. **Settings additions** (after gpr block):
   - `crude_momentum_enabled: bool = Field(False, ...)`
   - `crude_momentum_window_days: int = Field(21, ...)` (1 trading month)
   - `crude_momentum_lookback_days: int = Field(252, ...)` (1 trading year for z-score normalization)
   - `crude_momentum_zscore_threshold: float = Field(1.0, ...)`
   - `crude_momentum_cache_hours: int = Field(24, ...)`
   - `crude_momentum_sector_etfs: str = Field("XLE", ...)`

2. **macro_regime.py additions**:
   - `_CRUDE_CACHE_PATH = _CACHE_DIR / "crude_momentum.json"` constant
   - New `async def _fetch_crude_momentum(...)` helper:
     - Cache fresh? → return cached dict
     - Else `yf.download("CL=F", period="1y", interval="1d")` (async via `asyncio.to_thread`)
     - Compute daily pct_change, then trailing-window cumulative pct change (~1-month), then z-score over the lookback
     - Return `{current_momentum, zscore, threshold, last_date, above_threshold}` or None on failure
   - Inside `compute_macro_regime()`, ADD a SECOND post-LLM hook AFTER the GPR hook:
     - Call `_fetch_crude_momentum(...)` when `settings.crude_momentum_enabled` is True
     - Reuse `_apply_gpr_tilt(parsed, crude_info, settings.crude_momentum_sector_etfs)` — function is generic over `above_threshold`
     - Log GPR vs crude triggers separately

3. **Run masterplan verification** — must EXIT 0.

4. **Smoke test**:
   - `_fetch_crude_momentum()` returns dict with numeric current/zscore/threshold/above_threshold
   - `_apply_gpr_tilt(parsed, crude_info, "XLE")` injects XLE when above_threshold; identity when below

5. **Write `experiment_results.md`** + `live_check_28.6.md` (with REAL CL=F data).

6. **Spawn Q/A**.

7. **On PASS** — append harness_log Cycle 19, flip status.

## References

- `handoff/current/phase-28.6-research-brief.md`
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief item #6)
- `.claude/masterplan.json::phase-28.steps[6]`

## Risk / blast radius

- **Default OFF** — `crude_momentum_enabled = False`.
- **Reuses existing helper** — `_apply_gpr_tilt` is generic; no new injection code path.
- **Graceful degradation** — yfinance failure → returns None → no tilt → cycle continues. Same pattern as 28.3.
- **Cost** — single yfinance call per 24h cache window. Zero LLM.
- **Orthogonal to 28.3** — both triggers can fire independently; their outputs deduplicate naturally in `_apply_gpr_tilt`'s "preserve order, no duplicate add" logic.
