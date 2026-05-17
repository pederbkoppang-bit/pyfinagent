# Contract — phase-28.10 — Opportunistic insider buying signal

**Step ID:** phase-28.10
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.10-research-brief.md` (`gate_passed: true`; 5 sources read in full: arXiv 2026 insider microcaps, Quant Decoded insider signals, CRA Q2 2025 lit watch, Harvard Law CMP summary, NBER digest).
- Internal audit: `backend/tools/sec_insider.py` has `get_insider_trades(ticker, months=6)` returning `{trades: [...]}` with date/type/insider_name/value per Form 4 transaction. NO existing opportunistic/routine classifier.
- CMP classification (Cohen-Malloy-Pomorski 2012): ROUTINE = insider traded same calendar month in 3 prior consecutive years; OPPORTUNISTIC = all others. Cold-start (insiders with <3y history) → UNKNOWN, NOT opportunistic.
- 82bps/month value-weighted alpha for opportunistic buys; ~0 for routine.

## Hypothesis

Wrapping `sec_insider.get_insider_trades(ticker, months=48)` with the CMP classifier produces a per-ticker "opportunistic-buy dollar value over last 30d" signal. Tickers with material (>$500K aggregate) opportunistic buys get a boost (+7% strong, +4% moderate). Default OFF.

## Immutable success criteria (from masterplan)

1. `insider_signal_screen_module_created`
2. `opportunistic_vs_routine_classifier_documented`
3. `feature_flag_insider_signal_screen_enabled_default_false`
4. `live_check_lists_opportunistic_signals_for_one_cycle`

Immutable verification:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/insider_signal_screen.py').read()); from backend.services.insider_signal_screen import fetch_insider_signals; print('importable')" && grep -q 'insider_signal_screen_enabled' backend/config/settings.py
```

Immutable live_check shape:
> "live_check_28.10.md: cycle log showing N tickers with opportunistic insider buys + insider IDs (anonymized) + aggregate $"

## Plan steps

1. **Settings**: `insider_signal_screen_enabled` (False), `insider_lookback_history_months` (48), `insider_signal_window_days` (30), `insider_signal_min_aggregate_usd` (500_000), `insider_signal_strong_aggregate_usd` (2_000_000), `insider_strong_boost` (0.07), `insider_moderate_boost` (0.04).
2. **New module `backend/services/insider_signal_screen.py`**:
   - `InsiderSignal` Pydantic model
   - `async fetch_insider_signals(tickers, ...)`: per-ticker `get_insider_trades(ticker, months=48)`; CMP classifier; aggregate opportunistic-BUY dollar value over last `window_days`; return one entry per qualifying ticker
   - `apply_insider_signal_to_score(base, ticker, signals)`: multiply by `boost_multiplier`
3. **Edit `backend/tools/screener.py:rank_candidates`**: add `insider_signals=None` kwarg + apply block after options_surge
4. **Edit `backend/services/autonomous_loop.py`**: pre-fetch for top 2*paper_screen_top_n when flag is on; pass through
5. Verify + smoke + Q/A + log + flip.

## References

- `handoff/current/phase-28.10-research-brief.md`
- Primary brief item #9
- `.claude/masterplan.json::phase-28.steps[10]`

## Risk / blast radius

- **Default OFF.**
- **Cost** — per-ticker SEC EDGAR fetch with 48-month lookback (heavy on EDGAR). Throttle via existing `_fetch_form4` semaphore. Bound to ~20 candidates.
- **Cold-start** — UNKNOWN class for insiders with <3y history; doesn't false-positive as opportunistic.
- **Graceful degradation** — EDGAR rate-limits or empty results → no signal → cycle continues.
