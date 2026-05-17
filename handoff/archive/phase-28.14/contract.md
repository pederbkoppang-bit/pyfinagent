# Contract — phase-28.14 — Defense/war-stocks reference case

**Step ID:** phase-28.14
**Phase:** phase-28 — Candidate Picker Expansion
**Cycle:** 1
**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

---

## Research gate summary

- Brief: `handoff/current/phase-28.14-research-brief.md` (`gate_passed: true`; 6 sources read in full).
- Internal audit: `_fetch_gpr_acts` + `_apply_gpr_tilt` from phase-28.3 are generic and reusable. Researcher recommends XAR over ITA (ITA is 19% GE/commercial aviation — too noisy). AND-gate: GPRA above threshold AND XAR 5d momentum > 0.
- Defense ticker list: LMT, NOC, RTX, GD, LHX, BA, LDOS, HII, KTOS (US); BAE.L, RHM.DE, SAAB-B.ST (EU).

## Hypothesis

When BOTH (a) GPR-Acts above threshold (phase-28.3 trigger) AND (b) XAR 5-day momentum > 0 (institutional money confirming the GPR signal via the defense ETF), boost individual defense-ticker composite scores in the screener. Supplement Gap 1 documented +1.00% (−1,−1) anticipatory + +11.65% CAAR (0,3) for European defense in Ukraine event study. Default OFF.

## Immutable success criteria (from masterplan)

1. `defense_signal_module_created_using_GPR_fetcher_from_28.3`
2. `ITA_XAR_flow_delta_implemented`
3. `budget_pledge_headline_keyword_set_documented`
4. `feature_flag_defense_signal_enabled_default_false`
5. `live_check_shows_defense_candidates_when_GPR_above_threshold_AND_flow_positive`

Immutable verification:
```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/defense_signal.py').read()); print('syntax OK')" && grep -q 'defense_signal_enabled' backend/config/settings.py && grep -qE 'ITA|XAR' backend/services/defense_signal.py
```

Immutable live_check shape:
> "live_check_28.14.md: cycle log showing GPR-Acts value + ITA/XAR 5-day flow + any pledge headlines + resulting LMT/NOC/RTX/BAE/RHM score boosts"

## Plan steps

1. **Settings**: `defense_signal_enabled` (False), `defense_xar_window_days` (5), `defense_xar_min_momentum` (0.0), `defense_tickers` ("LMT,NOC,RTX,GD,LHX,BA,LDOS,HII,KTOS,BAE.L,RHM.DE,SAAB-B.ST"), `defense_boost` (0.05), `defense_budget_pledge_keywords` (csv: "NATO budget,defense spending,Zeitenwende,defense pledge,military spending").
2. **New module `backend/services/defense_signal.py`**:
   - `DefenseSignal` Pydantic model: triggered, gpr_above_threshold, xar_5d_momentum, pledge_keyword_hit, boost_multiplier, defense_tickers
   - `fetch_defense_trigger()` async — calls `macro_regime._fetch_gpr_acts` (reuses cached value, free) + computes XAR 5d momentum via yfinance + checks recent news headlines for pledge keywords
   - `apply_defense_boost_to_score(base, ticker, signal, defense_tickers_set)` — multiply by boost when ticker IS defense AND signal.triggered
3. **Edit screener.py:rank_candidates**: add `defense_signal=None` kwarg + apply block after social_velocity
4. **Edit autonomous_loop.py**: pre-fetch ONCE per cycle (not per-ticker — global signal); pass through
5. Verify + smoke + Q/A + log + flip.

## References

- `handoff/current/phase-28.14-research-brief.md`
- Supplement brief Gap 1
- `.claude/masterplan.json::phase-28.steps[14]`

## Risk / blast radius

- **Default OFF.**
- **Reuses 28.3 GPR fetcher** (cached); no extra network cost beyond one XAR yfinance pull.
- **Pledge keyword scan** uses existing news_screen plumbing or simple string match in latest news headlines (optional — design fallback if no news fetcher available).
- **Graceful degradation** — any fetch failure → signal.triggered=False → identity.
