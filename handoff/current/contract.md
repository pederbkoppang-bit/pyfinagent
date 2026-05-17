# Contract — phase-28.16 — M&A pre-announcement detection (FINAL phase-28 item)

**Step ID:** phase-28.16
**Date:** 2026-05-18
**Author:** Main (Researcher fell back per user directive)

---

## Research gate
Brief at `handoff/current/phase-28.16-research-brief.md` — Main fallback after Researcher stopped mid-step. 5 sources read in full + 6 failed (SEC EDGAR + most SSRN paywalls), 15 URLs collected.

## Hypothesis
Augustin-Brenner-Subrahmanyam (options pre-M&A) + Duong-Pi-Sapp 2025 (insider buying pre-13D) together identify the PUBLIC FOOTPRINT of informed M&A activity. Combining Legs 1+2 (already implemented at 28.9 and 28.10) as a single aggregator yields a high-confidence pre-announcement boost. Leg 3 (13D EDGAR polling) is stubbed for future implementation (SEC EDGAR full-text-search API returned 403 to direct WebFetch; needs authenticated client).

## Immutable success criteria

1. `ma_preannounce_screen_module_created`
2. `three_legs_present_OTM_options_and_Form_4_cluster_and_13D_polling`
3. `uses_only_public_data_per_legality_boundary_note`
4. `feature_flag_ma_preannounce_enabled_default_false`
5. `live_check_lists_M_A_signal_tickers_for_one_cycle`

Verification:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/ma_preannounce_screen.py').read()); print('syntax OK')" && grep -q 'ma_preannounce_enabled' backend/config/settings.py && grep -qE '13[dg]|SCHEDULE.13' backend/services/ma_preannounce_screen.py
```

Live_check: "cycle log showing N tickers with M&A signal + which legs triggered + signal aggregation"

## Plan

1. Settings: `ma_preannounce_enabled` (False), `ma_preannounce_strong_boost` (0.10 — both legs fire), `ma_preannounce_moderate_boost` (0.05 — single leg fires).
2. New `backend/services/ma_preannounce_screen.py`:
   - `MAPreannounceSignal` Pydantic model: ticker, legs_triggered (list), boost_multiplier
   - `compute_ma_preannounce_signals(tickers, options_surge_signals, insider_signals, schedule_13d_signals)` PURE — three-leg aggregator
   - `_fetch_13d_filings_for(ticker)` STUB returning `[]` with documented TODO for SEC EDGAR full-text-search API (returned 403 in this environment)
   - `apply_ma_preannounce_to_score` helper
3. screener.rank_candidates: `ma_preannounce_signals=None` kwarg + apply after peer_leadlag
4. autonomous_loop: compute via aggregator (reuses already-fetched options_surge + insider signals when both flags on; no extra cost)
5. Verify + smoke + Q/A + log + flip.

## Legality

Picker observes ONLY public market/EDGAR data:
- Options volume + IV (public CBOE/yfinance)
- Form 4 (public SEC filings)
- 13D (public SEC filings, when 13D leg lands)

Never infers / acts on MNPI. The Augustin paper documents what informed traders DO; we observe the PUBLIC FOOTPRINT.

## Risk
- Default OFF.
- ZERO additional network cost — reuses options_surge + insider signals already fetched by phase-28.9 and 28.10.
- 13D leg is currently stubbed (empty list) — documented for Phase-2 follow-up.
