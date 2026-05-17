# Contract — phase-28.9 — Options-flow OI-surge filter

**Step ID:** phase-28.9
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.9-research-brief.md` (`gate_passed: true`; 5 sources read in full: CXO Advisory, LuxAlgo, Market Rebellion / Najarian, Management Science 2026, OptionsTradingOrg).
- Internal audit: `backend/tools/options_flow.py` already exists as a Layer-1 analysis tool — uses `yf.Ticker.option_chain` per ticker, computes vol > 3*OI on full chain. Phase-28.9 ADDS a NEAR-EXPIRY OTM filter (Wayne State / J. Portfolio Mgmt) on top.
- Per Researcher: surge criteria = `strike > spot * 1.01` (OTM), DTE 2-45 days, `volume > max(5x rolling avg, 3x OI)`. Boost: 6% (strong, 2+ surges), 3% (moderate, 1 surge).

## Hypothesis

The existing options_flow.py looks at all strikes — too noisy. Wayne State research specifies: near-expiry OTM CALL volume surges are predictive; generic large trades are not. A narrow filter at the screener tier (mirror of phase-28.1 pattern) gives pre-rally exposure without rebuilding Layer-1 analysis.

## Immutable success criteria (from masterplan)

1. `options_flow_screen_module_created`
2. `OTM_near_expiry_volume_threshold_documented`
3. `feature_flag_options_flow_screen_enabled_default_false`
4. `wired_into_rank_candidates_or_meta_scorer`
5. `live_check_lists_OI_surge_candidates_for_one_cycle`

Immutable verification command:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/options_flow_screen.py').read()); from backend.services.options_flow_screen import fetch_oi_surge_signals; print('importable')" && grep -q 'options_flow_screen_enabled' backend/config/settings.py
```

Immutable live_check shape:
> "live_check_28.9.md: cycle log showing N tickers flagged with OTM call OI surge + the surge multipliers"

## Plan steps

1. **Settings**: 9 fields — `options_flow_screen_enabled` (False), `options_otm_threshold` (1.01), `options_dte_min` (2), `options_dte_max` (45), `options_vol_avg_multiplier` (5.0), `options_vol_oi_multiplier` (3.0), `options_strong_boost` (0.06), `options_moderate_boost` (0.03), `options_cache_hours` (4).
2. **New module `backend/services/options_flow_screen.py`**: `OptionsSurgeSignal` model + `fetch_oi_surge_signals(...)` async + `apply_options_surge_to_score(...)`. Per-ticker via yfinance, Semaphore(4), 0.3s throttle.
3. **Edit `backend/tools/screener.py:rank_candidates`**: add `options_surge_signals=None` kwarg + apply block after sector_momentum.
4. **Edit `backend/services/autonomous_loop.py`**: fetch surge signals for top 2*paper_screen_top_n candidates when flag is on; pass to rank_candidates.
5. Verify + smoke + Q/A + log + flip.

## References

- `handoff/current/phase-28.9-research-brief.md`
- Primary brief item #8
- `.claude/masterplan.json::phase-28.steps[9]`

## Risk / blast radius

- **Default OFF.** Validated baseline preserved.
- **Cost** — per-ticker yfinance option_chain HTTP call. Bounded to top-N (~20 candidates per cycle). Throttled.
- **Graceful degradation** — yfinance failure → empty dict → no boost. Cycle continues.
- **Signal decay** — Wayne State noted decay risk; threshold knobs let operator tune.
