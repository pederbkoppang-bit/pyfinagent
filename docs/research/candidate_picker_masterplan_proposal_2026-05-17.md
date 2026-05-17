# phase-28 — Candidate Picker Expansion (Proposal — READY FOR APPROVAL)

**Date:** 2026-05-17
**Author:** Main (Claude Code session, Opus 4.7 xhigh)
**Linked research:**
- `docs/research/candidate_picker_improvements_2026-05-16.md` (primary brief, 18 sources read in full, `gate_passed: true`)
- `docs/research/candidate_picker_improvements_2026-05-16-supplement.md` (supplement, 9 sources read in full, `gate_passed: true`)

**Goal driver:** catch high-momentum opportunities like Sandisk (memory shortage), oil majors post-conflict, war/defense stocks. Reference cases documented in primary brief Phase 1 + supplement Gap 1.

**Status:** READY FOR APPROVAL — no edits to `.claude/masterplan.json` until Peder approves. Per the user goal: "Show planned diff and wait for approval before editing."

---

## Decisions baked in (justification)

1. **One new top-level phase: `phase-28`.** Rationale: the recommendations form a coherent block of work (picker-side signals), not scattered improvements. A single phase with sub-steps preserves auditability, lets the existing `auto-commit-and-push` + `archive-handoff` hooks operate per item, and matches the precedent set by phase-23.x (a coherent "harness hardening" block).
2. **Pre-go-live items have lower step numbers** (28.1-28.6) so they execute first under harness driver order. Post-launch items are 28.7-28.13. Supplement items (28.14-28.17) come last because they layer on top of the pre-go-live infrastructure (e.g., 28.14 builds on 28.3's GPR fetcher; 28.17 builds on 28.4's sector-neutralization machinery).
3. **Each step is `harness_required: true`** because picker changes touch the autonomous-loop money path. The hard verification command + `live_check` field per step is non-negotiable.
4. **`depends_on_step` reflects technical dependencies**, not arbitrary ordering. Sector-neutral momentum (28.4) depends on 28.0 (the dead `min_market_cap` parameter fix) only insofar as both touch `screener.py`; otherwise the pre-go-live items are independent.
5. **Universe expansion (28.8) is post-launch** because it's the highest-impact change (catches Sandisk-class spinoffs) but also the highest-blast-radius (3x screening cost). It needs careful staging.
6. **Drift fix `28.0`** (unused `min_market_cap` parameter in `screener.screen_universe()` lines 70-83 vs line 121) included as a tiny housekeeping step so the brief and code agree before we layer new logic.
7. **All new screener signals default to OFF behind feature flags** (`backend/config/settings.py`). This preserves the current Sharpe 1.1705 baseline; only when a step's `live_check` shows the flag-on path improves things do we consider switching the default. Same pattern as existing `macro_regime_filter_enabled`, `pead_signal_enabled`, `news_screen_enabled`, `sector_calendars_enabled`, `meta_scorer_enabled`.

---

## Slot-justification table (pre-go-live tier, end of May)

| Step | Name | File(s) touched | Depends on | Effort | Source citation |
|---|---|---|---|---|---|
| 28.0 | Drift fix: actually apply `min_market_cap` in `screen_universe()` (or remove the dead param) | `backend/tools/screener.py` | none | XS | Internal audit during this session |
| 28.1 | Analyst EPS revision-breadth plug-in | new `backend/services/analyst_revisions.py` + wire into `screener.rank_candidates` | 28.0 | S | Mill Street Research 19yr: t=2.93, Sharpe~1.60 combined (primary brief #1) |
| 28.2 | 12-quarter SUE stack in PEAD | `backend/services/pead_signal.py` | 28.0 | S | ScienceDirect 2025: +85% Sharpe lift from stacking (primary brief #2) |
| 28.3 | GPR-triggered energy-sector tilt | `backend/services/macro_regime.py` (consume Iacoviello monthly index) | 28.0 | S | Caldara-Iacoviello AER 2022 + IMF GFSR 2025 (primary brief #3) |
| 28.4 | Sector-neutral momentum scoring | `backend/tools/screener.py:rank_candidates` | 28.0 | S | CFA Institute Dec 2025: multi-D composite outperforms (primary brief #4) |
| 28.5 | Short-interest exclusion filter | `backend/tools/screener.py:screen_universe` filters | 28.0 | S | Boehmer-Jones-Zhang: -1.16%/mo for high-short decile (primary brief #5) |
| 28.6 | Crude-oil cross-asset trend signal | `backend/services/macro_regime.py` (add `CL=F` series) | 28.3 | S | IMF GFSR 2025 + Caldara-Iacoviello asymmetry (primary brief #6) |

## Slot-justification table (post-launch tier)

| Step | Name | File(s) touched | Depends on | Effort | Source citation |
|---|---|---|---|---|---|
| 28.7 | Multidimensional momentum composite (price + SUE + 52-wk hi distance + factor momentum) | `backend/tools/screener.py:rank_candidates` | 28.4 | M | CFA Institute Dec 2025 (primary brief #11) |
| 28.8 | Russell-1000 universe expansion (with screener cost guard) | `backend/tools/screener.py:get_sp500_tickers` + new `get_russell1000_tickers` | 28.4, 28.5 | M | Sandisk/SNDK case — universe miss (primary brief #10) |
| 28.9 | Options-flow plug-in (call OI surge filter) | new `backend/services/options_flow_screen.py` | 28.0 | M | Wayne State / J. Portfolio Mgmt (primary brief #8) |
| 28.10 | Opportunistic insider buying lift Layer-1 → screener | new `backend/services/insider_signal_screen.py` (uses `backend/tools/sec_insider.py`) | 28.8 | M | Cohen-Malloy-Pomorski: 82bps/mo (primary brief #9) |
| 28.11 | LLM analyst-report narrative signal | new `backend/services/analyst_narrative_scorer.py` | 28.1 | L | arXiv 2502.20489v1: 68bps/mo alpha (primary brief #7) |
| 28.12 | Sector-ETF momentum overlay | new `backend/services/sector_momentum.py` (or extend `sector_calendars.py`) | 28.4 | S | Quantpedia: 13.94% annual, Sharpe 0.54 (primary brief #13) |
| 28.13 | Earnings-call NLP for firm-level GPR exposure | new `backend/services/call_transcript_gpr.py` | 28.3, 28.11 | L | Fed Aug 2025 (primary brief #12) |

## Slot-justification table (supplement tier — finalized)

| Step | Name | File(s) touched | Depends on | Effort | Source citation |
|---|---|---|---|---|---|
| 28.14 | Defense/war-stocks reference-case (GPR + ITA/XAR flow + NATO/EU pledge headline) | extends `backend/services/macro_regime.py` with defense branch + new `backend/services/defense_signal.py` for ETF flow | 28.3 | M | Emerald SEF 2023 (+1.00% (−1,−1), +11.65% CAAR (0,3)); PMC11700249 (81.4% firms reacted); supplement Gap 1 |
| 28.15 | Social media velocity in screener (lift `social_sentiment.py` into picker pre-filter) | extends `backend/tools/screener.py:screen_universe` + new `backend/services/social_velocity_screen.py` | 28.0 | S | DailyTickers API guide; SSRN 4947010 (Reddit volatility); 2025 DNUT case (5x StockTwits spike → 90% pre-market); supplement Gap 2 |
| 28.16 | M&A pre-announcement detection (OTM-call spike + Form 4 cluster + 13D/G surveillance) | new `backend/services/ma_preannounce_screen.py` (uses existing `options_flow.py` + `sec_insider.py`) + 13D/G EDGAR polling | 28.9, 28.10 | L | Augustin-Brenner-Subrahmanyam (~25% takeovers, "five to ten days before"); Duong-Pi-Sapp 2025 (7.72% CAR, 14.49% no-prior-discussion insiders); supplement Gap 3 |
| 28.17 | Peer-correlation laggard catch-up (intra-GICS sub-industry lead-lag) | new `backend/services/peer_leadlag_screen.py` extending `screener.py` | 28.4, 28.7 | M | Hou 2007 (intra-industry, low-coverage stocks lag); DeltaLag arXiv 2511.00390 (~10 bpts/day); arXiv 2312.10084 (~10pp 2021 bull); shared-analyst-coverage 1.68%/mo; supplement Gap 4 |

---

## Conflicts and overlaps (with existing masterplan)

- **No conflict with phase-27.6** (Multi-Provider Full-Path Pipeline, blocked on Anthropic credit). The picker work touches `backend/tools/screener.py` and `backend/services/*` — none of phase-27's pending files. Phase-28 can proceed independently if Peder so chooses, even before 27.6.3 unblocks.
- **No conflict with phase-23.8 (Dev-MAS Audit Remediation, pending).** That work touches the harness MAS, not the picker.
- **Partial overlap with the existing `news_screen.py` event-type list** (already includes `merger_acquisition`). Step 28.16 (M&A pre-announcement) explicitly handles the PRE-headline case via insider/options/13D — it does NOT duplicate news_screen.
- **The existing `tools/social_sentiment.py`** is already a Layer-1 enrichment with sentiment + velocity (line 95: `velocity = recent_avg - older_avg`). Step 28.15 LIFTS this signal into the picker (Layer-1 → screener) rather than re-implementing — smaller change than the primary brief implied.
- **The existing `tools/sec_insider.py`** already handles Form 4 (insider trades). Step 28.16 ADDS 13D/G filing polling on top.
- **The existing `tools/options_flow.py`** already analyzes options chains. Step 28.16 ADDS the specific OTM-call-spike + IV term-structure-inversion heuristic.

---

## Sequencing recommendation (for Phase 6 execution)

Per harness convention (one item, one Q/A pass, one merge, then next), execute pre-go-live in this order:

1. **28.0** (drift fix) — tiny, lands first, clears the way
2. **28.5** (short-interest filter) — pure exclusion filter, no scoring change, safest first
3. **28.1** (analyst revisions) — additive plug-in, low blast radius
4. **28.2** (SUE stacking) — extends existing plug-in, isolated
5. **28.3** (GPR sector tilt) — net-new but isolated to macro_regime
6. **28.6** (crude oil cross-asset) — small, builds on 28.3
7. **28.4** (sector-neutral momentum) — touches scoring core; goes LAST in pre-go-live tier so additive items are already in place

Then, after pre-go-live tier completion and Peder go-ahead, post-launch in impact-to-effort order:

8. **28.12** (sector-ETF momentum overlay) — small, complements 28.4
9. **28.7** (multidimensional momentum composite) — extends 28.4 once 28.4 is validated
10. **28.8** (Russell-1000 universe) — high impact, biggest blast radius; needs cost guard
11. **28.9** (options-flow OI surge) — net-new plug-in, isolated
12. **28.10** (insider buying lift) — depends on 28.8 (bigger universe = more signal)
13. **28.11** (LLM analyst narrative) — highest expected lift but L effort + cost
14. **28.13** (earnings-call GPR NLP) — depends on 28.3 + 28.11

Then, supplement-tier in impact-to-effort order:

15. **28.15** (social velocity) — S effort, lift existing tool; quickest supplement win
16. **28.14** (defense GPR + ETF flow) — extends 28.3; M effort
17. **28.17** (peer lead-lag) — extends 28.4 + 28.7; M effort
18. **28.16** (M&A pre-announcement) — L effort, three-leg signal; do last

Each step: Researcher gate → contract.md → implement behind a feature flag (default OFF) → Q/A pass → live_check artifact → flip status to `done` → auto-commit + push → next.

---

## Diff preview (JSON to be appended to `.claude/masterplan.json::phases[]`)

The complete phase-28 block follows. To apply: insert this object as the final element of the `phases` array in `.claude/masterplan.json` (after the existing `phase-27` block; before the closing `]`).

```json
{
  "id": "phase-28",
  "name": "Candidate Picker Expansion",
  "status": "pending",
  "depends_on": ["phase-27"],
  "gate": null,
  "steps": [
    {
      "id": "28.0",
      "name": "Drift fix: apply (or remove) unused min_market_cap parameter in screen_universe()",
      "status": "pending",
      "harness_required": true,
      "priority": "P2",
      "depends_on_step": null,
      "audit_basis": "Internal code audit during phase-28 planning 2026-05-17: backend/tools/screener.py:screen_universe accepts min_market_cap (default 1e9) at line 65 but never uses it inside the function body; only price (line 121) and volume filters apply. Either wire the filter or remove the dead parameter so the picker doc and code agree.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast,inspect; from backend.tools.screener import screen_universe; src=inspect.getsource(screen_universe); assert ('min_market_cap' in src and 'market_cap' in src.lower().split('def ')[-1]) or 'min_market_cap' not in src, 'param still dead'; print('PASS: min_market_cap is either used or removed')\"",
        "success_criteria": [
          "min_market_cap_parameter_either_applied_or_removed",
          "syntax_OK",
          "no_regression_in_existing_screener_callsites"
        ],
        "live_check": "live_check_28.0.md: one cycle log line showing the screener filter chain with market-cap status (applied / removed)"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.1",
      "name": "Analyst EPS revision-breadth plug-in (feature-flagged)",
      "status": "pending",
      "harness_required": true,
      "priority": "P0",
      "depends_on_step": "28.0",
      "audit_basis": "primary brief Phase 4 item #1; Mill Street Research 19yr backtest: revision-breadth top vs bottom decile spread 7.6% annualized, t=2.93, p=0.003; combined with price momentum Sharpe ~1.60. Directly addresses Sandisk/memory and oil-majors reference cases (analysts revise within 1-2 weeks of regime shift, before price momentum is visible).",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/analyst_revisions.py').read()); from backend.services.analyst_revisions import fetch_revision_signals; print('module importable')\" && grep -q 'analyst_revisions_enabled' backend/config/settings.py && grep -q 'analyst_revisions' backend/services/autonomous_loop.py",
        "success_criteria": [
          "analyst_revisions_module_created_and_syntax_OK",
          "feature_flag_analyst_revisions_enabled_default_false",
          "wired_into_rank_candidates_or_meta_scorer",
          "smoke_run_with_flag_on_produces_non_empty_signal_for_recent_reporters",
          "cycle_cost_delta_under_0_05_USD"
        ],
        "live_check": "live_check_28.1.md: cycle log + screener output diff showing N revisions-scored tickers, top-3 conviction shifts vs baseline"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.2",
      "name": "12-quarter SUE stacking in pead_signal.py",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.0",
      "audit_basis": "primary brief Phase 4 item #2; ScienceDirect 2025 ML stacking 12Q SUE history raised Sharpe from 0.34 (latest only) to 0.63 (+85% lift). Existing pead_signal.py at backend/services/pead_signal.py:91 already reads trailing cache via _trailing_mean_from_cache; extend to consume 12 quarters explicitly (currently bounded by _LOOKBACK_QUARTERS=8 — bump to 12 and reweight).",
      "verification": {
        "command": "source .venv/bin/activate && grep -qE '_LOOKBACK_QUARTERS\\s*=\\s*12' backend/services/pead_signal.py && python -c \"import ast; ast.parse(open('backend/services/pead_signal.py').read()); print('PASS')\"",
        "success_criteria": [
          "lookback_quarters_increased_to_12",
          "weighting_scheme_added_or_documented",
          "back-compat_with_existing_cache_files",
          "syntax_OK_and_pead_signal_still_importable"
        ],
        "live_check": "live_check_28.2.md: one ticker's PEAD before/after with 8Q vs 12Q stack, surprise_score diff and resulting holding_window_days"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.3",
      "name": "GPR-triggered energy-sector tilt in macro_regime.py",
      "status": "pending",
      "harness_required": true,
      "priority": "P0",
      "depends_on_step": "28.0",
      "audit_basis": "primary brief Phase 4 item #3 — oil-majors reference case. Caldara-Iacoviello AER 2022 + IMF GFSR 2025 document US-as-net-exporter asymmetry: Middle-East GPR-Acts spikes positively reprice XOM/CVX/COP/OXY. Index public at matteoiacoviello.com monthly. macro_regime.py already has sector_hints.overweight; add a 'gpr_acts > threshold' branch that injects XLE-overweight regardless of base FRED reading.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')\" && grep -qE 'gpr|geopolitical' backend/services/macro_regime.py",
        "success_criteria": [
          "gpr_index_fetcher_implemented_with_caching",
          "sector_tilt_branch_added_to_macro_regime",
          "threshold_documented_in_audit_basis",
          "live_check_shows_XLE_overweight_when_gpr_above_threshold"
        ],
        "live_check": "live_check_28.3.md: one cycle log showing GPR-Acts value + threshold + resulting sector_hints.overweight contents"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.4",
      "name": "Sector-neutral momentum scoring in rank_candidates",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.0",
      "audit_basis": "primary brief Phase 4 item #4; CFA Institute Dec 2025 documented sector-neutral momentum produces superior Sharpe with less regime sensitivity vs absolute momentum. backend/tools/screener.py:rank_candidates currently uses absolute composite; convert to within-sector percentile rank with minimum-per-sector guard.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/tools/screener.py').read()); from backend.tools.screener import rank_candidates; print('importable')\" && grep -qE 'sector.{0,40}rank|percentile' backend/tools/screener.py",
        "success_criteria": [
          "sector_neutral_branch_added_under_a_feature_flag",
          "minimum_per_sector_threshold_documented",
          "absolute_momentum_remains_default_until_validated",
          "live_check_compares_top10_under_both_modes_for_one_cycle"
        ],
        "live_check": "live_check_28.4.md: cycle log + top-10 under absolute vs sector-neutral side-by-side"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.5",
      "name": "Short-interest exclusion filter in screen_universe",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.0",
      "audit_basis": "primary brief Phase 4 item #5; Boehmer-Jones-Zhang: high-short-interest stocks underperform by 1.16%/month on average. yfinance .info['shortRatio'] or FINRA monthly available. Add to screen_universe hard filters as exclusion when shortRatio > top-decile threshold (initial threshold = 10.0).",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')\" && grep -qE 'short.{0,30}(ratio|interest|exclusion)' backend/tools/screener.py",
        "success_criteria": [
          "short_interest_field_collected_in_screen_universe",
          "exclusion_filter_added_with_documented_threshold",
          "feature_flag_short_exclusion_enabled_default_false",
          "live_check_lists_excluded_tickers_for_one_cycle"
        ],
        "live_check": "live_check_28.5.md: cycle log showing N excluded tickers + their shortRatio values"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.6",
      "name": "Crude-oil (CL=F) cross-asset trend signal in macro_regime",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.3",
      "audit_basis": "primary brief Phase 4 item #6; complements 28.3 for the oil-majors reference case. yfinance CL=F gives Brent/WTI futures price; 1-month momentum threshold becomes a secondary trigger for XLE-overweight even absent GPR-Acts spike.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/macro_regime.py').read()); print('syntax OK')\" && grep -qE 'CL=F|crude|brent|oil_trend' backend/services/macro_regime.py",
        "success_criteria": [
          "crude_oil_trend_signal_added_to_macro_regime",
          "threshold_documented",
          "fallback_when_yfinance_unavailable_does_not_break_cycle",
          "live_check_shows_oil_trend_value_and_resulting_sector_action"
        ],
        "live_check": "live_check_28.6.md: cycle log showing CL=F 1m momentum + threshold check + sector_hints diff"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.7",
      "name": "Multidimensional momentum composite (price + SUE + 52-wk hi distance + factor momentum)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.4",
      "audit_basis": "primary brief Phase 4 item #11; CFA Institute Dec 2025 documented multidimensional composite outperforms naive price momentum with materially lower crash risk. Extends 28.4's sector-neutral foundation by combining price momentum + SUE momentum + 52-week high distance + factor momentum into a weighted composite.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')\" && grep -qE '52.{0,5}week|fifty.two|composite_momentum' backend/tools/screener.py",
        "success_criteria": [
          "composite_momentum_function_added",
          "weighting_scheme_documented_with_source_citation",
          "feature_flag_composite_momentum_enabled_default_false",
          "live_check_compares_naive_vs_composite_top10_for_one_cycle"
        ],
        "live_check": "live_check_28.7.md: cycle log + scoring diff table (price-only vs composite) for top-10"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.8",
      "name": "Russell-1000 universe expansion with screener cost guard",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.5",
      "audit_basis": "primary brief Phase 4 item #10; addresses the universe miss that caused the SNDK/Sandisk early-rally gap. Russell 1000 ~3x candidates; needs explicit cost guard (top-N filter tightening or two-pass screen) so cycle cost stays under cap.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/tools/screener.py').read()); from backend.tools.screener import get_sp500_tickers; print('importable')\" && grep -qE 'russell|RUSSELL|iShares|IWB|get_russell' backend/tools/screener.py",
        "success_criteria": [
          "get_russell1000_tickers_function_added",
          "feature_flag_russell1000_enabled_default_false",
          "cost_guard_documented_top_N_or_two_pass_screen",
          "live_check_runs_one_cycle_at_russell1000_size_with_cost_under_cap"
        ],
        "live_check": "live_check_28.8.md: cycle log showing universe size + screening cost + post-screen candidate count"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.9",
      "name": "Options-flow OI-surge filter (new plug-in)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.0",
      "audit_basis": "primary brief Phase 4 item #8; Wayne State / Journal of Portfolio Management documents near-expiry OTM large call buys are predictive of positive abnormal returns. Generic large options trades NOT predictive — filter is specifically near-expiry, OTM, elevated volume vs 30-day average. Existing backend/tools/options_flow.py has data layer; new screener-tier wrapper required.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/options_flow_screen.py').read()); from backend.services.options_flow_screen import fetch_oi_surge_signals; print('importable')\" && grep -q 'options_flow_screen_enabled' backend/config/settings.py",
        "success_criteria": [
          "options_flow_screen_module_created",
          "OTM_near_expiry_volume_threshold_documented",
          "feature_flag_options_flow_screen_enabled_default_false",
          "wired_into_rank_candidates_or_meta_scorer",
          "live_check_lists_OI_surge_candidates_for_one_cycle"
        ],
        "live_check": "live_check_28.9.md: cycle log showing N tickers flagged with OTM call OI surge + the surge multipliers"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.10",
      "name": "Opportunistic insider-buying signal lifted from Layer-1 into screener tier",
      "status": "pending",
      "harness_required": true,
      "priority": "P2",
      "depends_on_step": "28.8",
      "audit_basis": "primary brief Phase 4 item #9; Cohen-Malloy-Pomorski: opportunistic insider trades earn 82bps/month abnormal return; routine trades ~0. backend/tools/sec_insider.py already pulls Form 4; classification logic + screener-tier wrapper required. Most useful once 28.8 expands universe beyond S&P 500 (large-caps heavily scrutinized; effect smaller).",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/insider_signal_screen.py').read()); from backend.services.insider_signal_screen import fetch_insider_signals; print('importable')\" && grep -q 'insider_signal_screen_enabled' backend/config/settings.py",
        "success_criteria": [
          "insider_signal_screen_module_created",
          "opportunistic_vs_routine_classifier_documented",
          "feature_flag_insider_signal_screen_enabled_default_false",
          "live_check_lists_opportunistic_signals_for_one_cycle"
        ],
        "live_check": "live_check_28.10.md: cycle log showing N tickers with opportunistic insider buys + insider IDs (anonymized) + aggregate $"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.11",
      "name": "LLM analyst-report narrative signal (highest expected lift)",
      "status": "pending",
      "harness_required": true,
      "priority": "P0",
      "depends_on_step": "28.1",
      "audit_basis": "primary brief Phase 4 item #7; arXiv 2502.20489v1 documents LLM-extracted Strategic Outlook text from analyst reports generates 68bps/month alpha, IR 0.73-1.41 — strongest signal in the literature, substantially exceeds traditional rec-change/EPS-revision signals. L effort: requires paid analyst-report feed OR EDGAR scraping path + LLM inference cost.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/analyst_narrative_scorer.py').read()); print('syntax OK')\" && grep -q 'analyst_narrative_enabled' backend/config/settings.py",
        "success_criteria": [
          "analyst_narrative_scorer_module_created",
          "data_source_decision_documented_paid_vs_EDGAR_vs_free",
          "feature_flag_analyst_narrative_enabled_default_false",
          "cost_budget_per_cycle_documented",
          "live_check_includes_narrative_score_for_at_least_5_tickers"
        ],
        "live_check": "live_check_28.11.md: cycle log + narrative scores for sample tickers + per-cycle LLM cost"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.12",
      "name": "Sector-ETF momentum overlay (Quantpedia top-3 rotation)",
      "status": "pending",
      "harness_required": true,
      "priority": "P2",
      "depends_on_step": "28.4",
      "audit_basis": "primary brief Phase 4 item #13; Quantpedia documents top-3 sector ETFs by 12-month momentum (monthly rebalance) → 13.94% annual, Sharpe 0.54. Complements 28.4 (sector-neutral momentum) by adding a sector-LEADER preference: stocks in top-3-momentum sectors get a small score boost.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/sector_momentum.py').read()); print('syntax OK')\" && grep -q 'sector_momentum_enabled' backend/config/settings.py",
        "success_criteria": [
          "sector_momentum_module_created",
          "top_3_sector_logic_documented",
          "feature_flag_sector_momentum_enabled_default_false",
          "live_check_lists_winning_sectors_and_boost_recipients"
        ],
        "live_check": "live_check_28.12.md: cycle log showing 11 sector ETF 12-month momentum ranks + which 3 won + N tickers boosted"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.13",
      "name": "Earnings-call NLP for firm-level GPR exposure",
      "status": "pending",
      "harness_required": true,
      "priority": "P2",
      "depends_on_step": "28.11",
      "audit_basis": "primary brief Phase 4 item #12; Fed Aug 2025 documented firm-level GPR exposure via earnings-call NLP on 240K+ transcripts. Note from primary brief: no forward return predictability shown in Fed study (R²=0.23 contemporaneous only). Add as complementary signal to 28.3 (GPR-energy tilt) and 28.11 (analyst narrative), NOT as standalone alpha generator. Re-evaluate priority after 28.11 lands.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/call_transcript_gpr.py').read()); print('syntax OK')\" && grep -q 'call_transcript_gpr_enabled' backend/config/settings.py",
        "success_criteria": [
          "call_transcript_gpr_module_created",
          "transcript_data_source_decision_documented",
          "feature_flag_call_transcript_gpr_enabled_default_false",
          "live_check_includes_gpr_exposure_classifications_for_5_tickers"
        ],
        "live_check": "live_check_28.13.md: cycle log + per-ticker GPR exposure tier (high/medium/low/none) for the cycle's candidate set"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.14",
      "name": "Defense/war-stocks reference-case implementation (GPR-defense branch + ITA/XAR flow + budget-pledge headline)",
      "status": "pending",
      "harness_required": true,
      "priority": "P0",
      "depends_on_step": "28.3",
      "audit_basis": "supplement Gap 1; Emerald SEF 2023 (+1.00% (−1,−1) anticipatory window; +11.65% CAAR (0,3) for European defense); PMC11700249 (81.4% defense firms reacted to Ukraine invasion); PMC11844836 (UK firms most sensitive). 28.3 adds GPR→energy; 28.14 adds GPR→defense branch + ITA/XAR ETF 5-day net-flow delta + NATO/EU budget-pledge headline scrape (Reuters/news_screen.py extension).",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/defense_signal.py').read()); print('syntax OK')\" && grep -q 'defense_signal_enabled' backend/config/settings.py && grep -qE 'ITA|XAR' backend/services/defense_signal.py",
        "success_criteria": [
          "defense_signal_module_created_using_GPR_fetcher_from_28.3",
          "ITA_XAR_flow_delta_implemented",
          "budget_pledge_headline_keyword_set_documented",
          "feature_flag_defense_signal_enabled_default_false",
          "live_check_shows_defense_candidates_when_GPR_above_threshold_AND_flow_positive"
        ],
        "live_check": "live_check_28.14.md: cycle log showing GPR-Acts value + ITA/XAR 5-day flow + any pledge headlines + resulting LMT/NOC/RTX/BAE/RHM score boosts"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.15",
      "name": "Social media velocity in screener (lift social_sentiment.py into picker pre-filter)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.0",
      "audit_basis": "supplement Gap 2; existing backend/tools/social_sentiment.py already computes velocity (line 95: 'velocity = recent_avg - older_avg') but is wired to Layer-1 enrichment only. Lift to screener pre-filter so tickers with social-velocity spikes surface BEFORE momentum filter. Conditions: StockTwits 5x hourly mention spike OR ApeWisdom WSB rank-change into top-20 OR Alpha Vantage cross-source convergence (3+ source types, sentiment + volume > 2x baseline). 2025 DNUT case: 500% StockTwits spike preceded 90% pre-market move.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/social_velocity_screen.py').read()); print('syntax OK')\" && grep -q 'social_velocity_enabled' backend/config/settings.py",
        "success_criteria": [
          "social_velocity_screen_module_created_lifting_existing_alpha_vantage_path",
          "stocktwits_or_apewisdom_data_path_documented",
          "feature_flag_social_velocity_enabled_default_false",
          "rate_limit_handling_documented_per_supplement_pitfalls",
          "live_check_lists_social_velocity_surfaced_tickers_for_one_cycle"
        ],
        "live_check": "live_check_28.15.md: cycle log showing N tickers surfaced by social velocity + the velocity multipliers + final ranking impact"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.16",
      "name": "M&A pre-announcement detection (OTM-call spike + Form 4 cluster + 13D/G surveillance)",
      "status": "pending",
      "harness_required": true,
      "priority": "P2",
      "depends_on_step": "28.9",
      "audit_basis": "supplement Gap 3; Augustin-Brenner-Subrahmanyam document short-dated OTM call volume spike + IV term-structure inversion 5-10 days pre-M&A; Duong-Pi-Sapp 2025: insider buying before 13D filings earns 7.72% CAR (14.49% for no-prior-discussion insiders). Three-leg signal: (a) OTM-call surge via 28.9, (b) Form 4 cluster via existing sec_insider.py, (c) NEW 13D/G EDGAR polling. Picker uses ONLY public market/EDGAR data — see supplement 'legality boundary' pitfall.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/ma_preannounce_screen.py').read()); print('syntax OK')\" && grep -q 'ma_preannounce_enabled' backend/config/settings.py && grep -qE '13[dg]|SCHEDULE.13' backend/services/ma_preannounce_screen.py",
        "success_criteria": [
          "ma_preannounce_screen_module_created",
          "three_legs_present_OTM_options_and_Form_4_cluster_and_13D_polling",
          "uses_only_public_data_per_legality_boundary_note",
          "feature_flag_ma_preannounce_enabled_default_false",
          "live_check_lists_M_A_signal_tickers_for_one_cycle"
        ],
        "live_check": "live_check_28.16.md: cycle log showing N tickers with M&A signal + which legs triggered + signal aggregation"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "28.17",
      "name": "Peer-correlation laggard catch-up signal (intra-GICS sub-industry lead-lag)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "28.7",
      "audit_basis": "supplement Gap 4; Hou 2007 (intra-industry, low-coverage stocks lag; effect strongest when analyst coverage < 5); DeltaLag arXiv 2511.00390 (~10 bpts/day excess in 2022-23); shared-analyst-coverage variant 1.68%/mo. Screen conditions: peer-momentum > +10% trailing 22d AND own-return < +2% AND analyst coverage < 5 AND market cap > $2B. Adds GICS sub-industry grouping to screener.",
      "verification": {
        "command": "source .venv/bin/activate && python -c \"import ast; ast.parse(open('backend/services/peer_leadlag_screen.py').read()); print('syntax OK')\" && grep -q 'peer_leadlag_enabled' backend/config/settings.py && grep -qE 'sub_industry|GICS|industry_group' backend/services/peer_leadlag_screen.py",
        "success_criteria": [
          "peer_leadlag_screen_module_created",
          "GICS_sub_industry_grouping_implemented",
          "screen_conditions_match_audit_basis",
          "feature_flag_peer_leadlag_enabled_default_false",
          "live_check_lists_laggard_candidates_with_their_peer_groups_for_one_cycle"
        ],
        "live_check": "live_check_28.17.md: cycle log showing N laggard candidates with peer-group leaders + the divergence size + analyst counts"
      },
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

---

## Approval gates (Peder)

I am asking for three explicit approvals before any masterplan edit:

**Approval A** — the choice of a single new top-level `phase-28` (vs scattering items into existing phases like phase-23.8 or phase-26).

**Approval B** — the 18 sub-steps as drafted (28.0–28.17) — specifically the source citations, files-touched, dependencies, and `live_check` shapes.

**Approval C** — the three-tier sequencing recommendation:
- Pre-go-live: 28.0 → 28.5 → 28.1 → 28.2 → 28.3 → 28.6 → 28.4
- Post-launch: 28.12 → 28.7 → 28.8 → 28.9 → 28.10 → 28.11 → 28.13
- Supplement: 28.15 → 28.14 → 28.17 → 28.16

AND that **each item is individually gated by your explicit go-ahead** per Phase 6 (no bulk-apply).

---

## What happens when you approve

On "go":
1. I apply the JSON above as a single atomic edit to `.claude/masterplan.json` (insert new `phase-28` block).
2. Append a "Cycle N -- phase=5-masterplan-integration result=PASS" line to `handoff/harness_log.md`.
3. Auto-commit hook fires; `phase-28` is then visible to the harness driver.
4. I pause and wait for your separate "start 28.0" go-ahead before beginning Phase 6.

If you want any change before applying: edit this file in place, or reply with the diff inline, and I'll iterate.
