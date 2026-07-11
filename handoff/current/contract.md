# Contract — Step 69.3 (P1 signal integrity + first $0 free-data alpha lift)

- **Phase / step**: phase-69 → 69.3
- **Date**: 2026-07-11
- **Type**: LIVE signal-integrity fixes + a $0 free-data lift, **all flag-gated default-OFF** (live engine byte-identical until the operator flips the flag).
- **Boundaries**: $0 metered, free APIs (existing FRED key); do-no-harm (flag-gated default-OFF); historical_macro FROZEN (new cached path only, NO BQ write); final IC/ablation/optimizer validation DEFERRED behind the historical_macro un-freeze token; does NOT conflict with phase-68 (overlays ≠ fills).

## Research-gate summary

Brief: `handoff/current/research_brief_69.3.md` — **gate_passed: true**, 5 external sources read in full
(netliquidity.org, macrolighthouse, Anthropic models doc [haiku 64k], eco3min [units confirmed], themarketsunplugged)
+ 69.0's overlay/INDPRO sources. Internal map by the researcher (before the 8th subagent stall); external floor
by Main. Scaffolding already committed (with 69.1): `backend/services/overlay_math.sign_safe_mult` (verified) +
settings flags `sign_safe_overlays` / `regime_net_liquidity` (default-OFF).

**Key design (complete)**: 14 `apply_*_to_score` overlays, ALL `base_score * mult`, applied at ONE chain
(`screener.py:318-411`) → route through the shared flag-gated `sign_safe_mult` helper. INDPRO 1-line fix
(missing from `fred_data.SERIES`). Net-liq `WALCL − WTREGEN − RRPONTSYD*1000` (units confirmed) via a new 24h
file-cache path mirroring `_fetch_gpr_acts`. News cap → chunk ~32 + retry. QMJ reorder. **Do-no-harm
refinement**: the INDPRO + net-liq regime-prompt inclusion is flag-gated behind `regime_net_liquidity` (so the
live regime prompt is byte-identical when OFF), since `_REGIME_SERIES` already lists INDPRO.

## Hypothesis

The sign-inversion, news-truncation, QMJ-dead-Growth, and INDPRO-dead bugs can be corrected — and net-liquidity
added — behind default-OFF flags so the LIVE engine stays byte-identical, with red→green unit tests + a $0
ON-vs-OFF live ranking comparison + a regime-prompt string render proving the fix, and historical_macro
byte-untouched; final IC/ablation validation deferred behind the un-freeze token.

## Immutable success criteria (verbatim from `.claude/masterplan.json` phase-69 → 69.3)

1. Sign-safe overlays (red->green, flag-gated): a test proves a negative-base candidate with a POSITIVE catalyst now ranks ABOVE an equal candidate with a negative catalyst (inversion eliminated) across macro_regime.py:547, news_screen.py:329 and the pead/options/insider/peer_leadlag overlays; the live ranking-behavior change is behind a flag with an ON-vs-OFF live_check comparison.
2. News token cap + parse-retry: the news_screen max_output_tokens min()-inversion is fixed so a 100-headline batch parses instead of truncating to {}, and a parse-fail retry is added. news_screen.py:282.
3. QMJ Growth: revenue_growth_yoy is assigned before quality_score consumes it, so the Asness QMJ Growth dimension fires (test). historical_data.py:202.
4. INDPRO + net-liquidity lift ($0, existing free FRED key, historical_macro untouched): INDPRO is repaired and net-liquidity (WALCL - WTREGEN - RRPONTSYD) is added into _REGIME_SERIES via a NEW cached path (fred_data.SERIES + _REGIME_SERIES + prompt thresholds, 24h file cache) -- NOT the frozen ingestion path; a live_check shows the regime prompt now includes INDPRO (restoring the intended series set) plus the net-liquidity component.
5. Do-no-harm: final IC / ablation / optimizer validation of any ranking change is deferred behind the historical_macro un-freeze token; the live overlay fixes and the FRED-prompt repair do not require it. historical_macro is byte-untouched (git diff). Fresh Q/A PASS with a live ON-vs-OFF ranking comparison.

## Plan (GENERATE)

1. Route the 14 `apply_*_to_score` fns through `sign_safe_mult(base, mult)` (helper reads `sign_safe_overlays`,
   default-OFF = byte-identical). Sites: news_screen:330/332, macro_regime:542/547/549, pead_signal:387/389,
   analyst_revisions:187, call_transcript_gpr:223, sector_momentum:200, options_flow_screen:183,
   analyst_narrative_scorer:242, peer_leadlag_screen:137, insider_signal_screen:224, social_velocity_screen:175,
   ma_preannounce_screen:150, defense_signal:164, sector_calendars:321/323. (The 3 inline base-score penalties
   at screener:306/308/311 are LISTED for operator, NOT changed.)
2. `news_screen`: chunk `deduped` into ~32 batches + parse-fail retry (byte-identical single call for N≤32).
3. `INDPRO` → `fred_data.SERIES`; `QMJ` → reorder `revenue_growth_yoy` before its read in `historical_data.py`.
4. `_fetch_net_liquidity` (WALCL−WTREGEN−RRPONTSYD*1000) 24h file-cache + regime-prompt line, both the INDPRO
   and net-liq regime-prompt inclusion gated behind `regime_net_liquidity` (default-OFF → regime prompt
   byte-identical). NO BQ write.
5. `backend/tests/test_signal_integrity_69.py`: sign-inversion (negative-base positive-catalyst > negative-base
   negative-catalyst; flag-OFF byte-identity), news 100-headline parse, QMJ Growth fires, net-liq unit scaling
   (RRP ×1000). Then experiment_results.md, a $0 live ON-vs-OFF + regime-prompt-string live_check, Workflow Q/A.

## References
- `handoff/current/research_brief_69.3.md` (5 sources + complete internal map) + `research_brief_69.0.md` §3/§4.
- `handoff/current/design_audit_burndown_69.md` §3 (sign-safe algebra) + §6.
- FRED net-liquidity sources; Anthropic models doc (haiku 64k); `backend/services/overlay_math.py` (committed helper).
