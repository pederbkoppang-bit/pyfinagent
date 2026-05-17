# Contract — phase-28.13 — Earnings-call NLP for firm-level GPR exposure

**Step ID:** phase-28.13
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.13-research-brief.md` (`gate_passed: true`; 5 sources read in full: Fed FEDS Note 2025, API Ninjas docs, arXiv 2503.01886, LSEG insights, Finnhub docs).
- **HONEST CONSTRAINT (Researcher):** Fed 2025 study (R²=0.23 on 240K+ transcripts) is **CONTEMPORANEOUS only — NO forward predictability**. This signal is a defensive RISK FILTER, NOT an alpha source.
- Existing infra: `backend/tools/earnings_tone.py::get_earnings_tone(ticker)` already fetches Yahoo Finance transcripts with GCS caching; returns `transcript_excerpt` (~8000 chars).

## Hypothesis

Per-firm GPR exposure (HIGH/MEDIUM/LOW/NONE) is useful as a defensive risk filter. Firms classified HIGH that ARE NOT in defense-benefiting sectors get a small score penalty (the Fed-documented contemporaneous risk premium). Firms in defense-benefiting sectors (Industrials, Aerospace & Defense, Energy) get EXEMPTION — they benefit from elevated GPR rather than being hurt. Default OFF.

This complements phase-28.3 (sector-level GPR-Acts tilt) by adding the firm-level dimension: a Health Care company explicitly discussing supply-chain risk in their earnings call may merit a small penalty even when sector GPR is low.

## Immutable success criteria (from masterplan)

1. `call_transcript_gpr_module_created`
2. `transcript_data_source_decision_documented`
3. `feature_flag_call_transcript_gpr_enabled_default_false`
4. `live_check_includes_gpr_exposure_classifications_for_5_tickers`

Immutable verification:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/call_transcript_gpr.py').read()); print('syntax OK')" && grep -q 'call_transcript_gpr_enabled' backend/config/settings.py
```

Immutable live_check shape:
> "live_check_28.13.md: cycle log + per-ticker GPR exposure tier (high/medium/low/none) for the cycle's candidate set"

## Plan steps

1. **Settings**: `call_transcript_gpr_enabled` (False), `call_transcript_gpr_model` ("claude-haiku-4-5"), `call_transcript_gpr_high_penalty` (0.97 = -3%), `call_transcript_gpr_exempt_sectors` (csv: "Industrials,Energy"), `call_transcript_gpr_cost_cap_usd` (0.10).
2. **New module `backend/services/call_transcript_gpr.py`**:
   - `GprExposureSignal` Pydantic model: ticker, exposure_tier, key_phrases, rationale, score_multiplier (set when applied)
   - `fetch_gpr_exposure_signals(tickers)` — calls `get_earnings_tone` per ticker, sends transcript_excerpt to Claude Haiku for 4-tier classification
   - `apply_gpr_exposure_to_score(base, ticker, sector, signals, exempt_sectors, high_penalty)` — applies penalty for HIGH unless sector is exempted
3. **Edit screener.py:rank_candidates**: add `gpr_exposure_signals=None` kwarg + apply block after narrative_signals
4. **Edit autonomous_loop.py**: pre-fetch when flag is on; pass through
5. Verify + smoke + Q/A + log + flip.

## References

- `handoff/current/phase-28.13-research-brief.md`
- Primary brief item #12
- `.claude/masterplan.json::phase-28.steps[13]`

## Risk / blast radius

- **Default OFF.**
- **HONEST USE:** defensive filter, NOT alpha source. Fed showed contemporaneous only.
- **Exemption for defense-benefitting sectors** prevents the filter from accidentally penalizing the very stocks that benefit from elevated GPR.
- **Cost** — Claude Haiku per ticker. Bounded to top 2*paper_screen_top_n. Per-cycle target <$0.10.
- **Graceful degradation** — transcript fetch or LLM fail → identity. Cycle continues.
