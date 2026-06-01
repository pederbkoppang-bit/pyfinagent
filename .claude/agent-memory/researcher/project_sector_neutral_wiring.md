---
name: project-sector-neutral-wiring
description: phase-51.2 sector-div lever -- the sector_neutral/multidim paths exist+wired but no-op because sectors arrive AFTER ranking; the ONE fix + the long-only caveat + why the ML backtest can't measure it
metadata:
  type: project
---

phase-51.2 sector diversification is the research-recommended near-term money lever
(amplify the WORKING US momentum engine's breadth INSIDE `screener.rank_candidates`,
the live-orders path, vs winner-take-all rotation which is architecturally
disconnected per [[project_strategy_rotation_unbuilt]]).

**Why:** diagnostic finding 4 -- non-tech sectors are OUT-COMPETED (not excluded) by
pure price-momentum ranking (`screener.py:268-282`: mom_1m*.40+mom_3m*.35+mom_6m*.25,
no sector term).

**How to apply (the load-bearing facts, file:line):**

1. **Both levers ALREADY EXIST and are ALREADY WIRED.** Live `rank_candidates`
   (`backend/services/autonomous_loop.py:621`, NOT `backend/autonomous_loop.py` which
   is the harness) passes `sector_neutral=settings.sector_neutral_momentum_enabled`
   (:629) and `multidim_momentum=settings.multidim_momentum_enabled` (:632). Flags
   default OFF (`settings.py:324,334`). The scoring code is correct + tested
   (`screener.py:415-445` within-sector percentile; `:464-523` multidim z-blend).

2. **The ONLY bug is data TIMING.** sector_neutral groups by `s.get("sector")`
   (`screener.py:425`); but `screen_universe` is called at `autonomous_loop.py:369`
   WITHOUT `sector_lookup=` (CONFIRMED 2026-06-01), and sector enrichment runs at
   `:659-676` -- AFTER `rank_candidates` returned. So at rank time every candidate has
   sector=None -> all fall into `_UNKNOWN_` -> single global percentile pool
   (monotone transform of raw score) -> byte-identical to OFF. THAT is the no-op.

3. **THE FIX (one change, no signature change):** build a ticker->sector map for the
   FULL `universe` and pass `sector_lookup=` into `screen_universe` at :369.
   `screen_universe` already accepts it (`screener.py:69`) and attaches sector to rows
   (`:206-213`). Source = `_fetch_ticker_meta` (`api/paper_trading.py:1058`, BQ-FIRST
   via paper_positions+analysis_results UNION, yfinance `.info` fallback, $0, 24h
   cache). Cost wrinkle: enrichment today runs on top-N survivors (~10-30); ranking
   needs the FULL screened set (~400-500), so cold-cache yfinance latency is the only
   real cost. For intl (EU/KR phase-50), use a static DAX-40/KOSPI-200 GICS map.

4. **sector_neutral > multidim as first lever.** sector_neutral directly fixes
   "non-tech out-competed" (ranks within sector). multidim's sector leg is a sector
   TILT (overweight hot sectors via boost_multiplier `screener.py:499-506`) -- AMPLIFIES
   concentration, OPPOSITE of breadth. multidim is a momentum-QUALITY upgrade.

5. **The ML backtest engine CANNOT measure this** -- `BacktestEngine` selects via
   `CandidateSelector._rank_candidates` (`backtest_engine.py:402,464` ->
   `candidate_selector.py:175-206`), a DIFFERENT formula with NO sector_neutral param
   and no sector field. The feature-ablation harness (`scripts/ablation/run_ablation.py`)
   ablates `_NUMERIC_FEATURES`, also wrong path. **Need a NEW screener-level replay**:
   `scripts/ablation/sector_neutral_replay.py` that replays PRODUCTION
   screen_universe+rank_candidates over ~36 monthly dates 2023-2025, OFF vs ON, scoring
   forward-return Sharpe + sector-spread + turnover. $0, free yfinance, real code, no
   live change. This is masterplan-51.2 criterion #2's artifact.

**THE BIG CAVEAT (carry into every contract):** pyfinagent is LONG-ONLY. Harvey et al.
"Is Sector Neutrality in Factor Investing a Mistake?" (Duke,
people.duke.edu/~charvey/.../P165_Is_sector_neutrality.pdf; QuantPedia summary):
"keeping the across [sector] component produces better long-LONG factors in 78% of
trials" -- i.e. FULL sector-neutralization helps long-only only ~22% of the time, and
RAISES turnover. Decision rule: neutralize only if "ratio of across/within Sharpe <
their correlation." => prefer a SOFT/PARTIAL sector tilt or cap over the existing HARD
within-sector-percentile REPLACEMENT (`screener.py:438-440` overwrites composite
entirely); MEASURE the sign on pyfinagent's own universe before any live enable. The
`settings.py:324` description ("Improves Sharpe per CFA Institute Dec 2025") is
optimistic -- CFA supports the multidim composite + vol-scaling, NOT sector-neutral
specifically for long-only.

**Higher-EV adjacent lever flagged:** vol-scaling the momentum book (CFA Dec 2025 /
Barroso-Santa-Clara: nearly DOUBLES Sharpe, halves the -87%/-88% momentum crash) is a
larger + better-evidenced live-path change than sector-neutral. Recommend
sector_neutral for 51.2 (directly answers finding 4 + 2026 breadth case) but note
vol-scaling if the replay disappoints.
