# Design D2a — Leakage Integrity (phase-73.1, FINAL — specs verbatim from gate `wf_5da65207-39a`, 5 sources read in full)

Scope pivot (73.0 re-score, code-verified twice + re-verified by this gate): the AFML purge + embargo already shipped in phase-69.2 — `_label_overlaps_test` (backtest_engine.py:570-582) implements exactly the canonical AFML overlap condition `(s<=te) and (label_end>=ts)`; horizon 135d covers the 126-135d triple-barrier span; the 5-day embargo (walk_forward.py:36) is a separate, weaker autocorrelation guard. The baseline's `:587` citation is `_build_predict_features`, not a leak. This design covers the four REMAINING components. External grounding: Profit Mirage FactFin (Alg.1/Eq.4/Table 5), Detecting-Lookahead-Bias (LAP confirmed BLOCKED for Claude — needs logprobs the Anthropic API does not expose), Look-Ahead-Bench, AFML/CPCV canon, Time-Machine-GPT principle, The New Quant §7.1 (cutoff-per-fold + embargo + pre-decision-evidence rationales mandated). Full notes: `research_brief_73.1.md`.

## 1. Purge verification + regression test (build step 73.1.1 [executor: sonnet-4.6/high])

Spec (verbatim from the gate):
- Document the shipped mechanics as a named invariant: _label_overlaps_test drops train samples whose label span [s, s+horizon] overlaps test [ts,te]; horizon_days=int(holding_days*1.5)=135d MUST be >= the triple-barrier vertical-barrier span (126-135d); the 5-day embargo is a SEPARATE autocorrelation guard -- the purge is load-bearing for the 135d horizon.
- Net-new regression test (home tests/backtest/test_purge_no_leak.py or tests/regression/test_purge_embargo_invariant.py): (a) truth-table unit test of _label_overlaps_test incl. boundary cases s==te and label_end==ts.
- (b) property test: for a synthetic split, assert NO retained training sample has label_end >= test_start (zero overlap survives purge).
- (c) integration guard: call _build_training_data(...,test_start,test_end); assert purged_count>0 when overlaps exist AND max(retained label_end) < test_start.
- (d) horizon invariant test: assert int(holding_days*1.5) >= vertical_barrier_days.
- Records the stale-F clearance: baseline's cited :587 is now _build_predict_features (predict-time imputation), NOT a leak.

Integration seams: backend/backtest/backtest_engine.py:570-582 (_label_overlaps_test predicate) | backend/backtest/backtest_engine.py:656 (horizon_days=int(holding_days*1.5)) | backend/backtest/backtest_engine.py:659-664 (purge loop) | backend/backtest/backtest_engine.py:428-430 (call site passes test_start/test_end) | backend/backtest/walk_forward.py:36 (embargo_days=5) | backend/backtest/walk_forward.py:61 (test_start=train_end+embargo+1) | tests/regression/ (only test_no_calendar_shadow.py -- new test home)

Cost: $0. Pure code + pytest on the quant-only GBM path; no LLM, no metered spend, no historical_macro. Buildable now.

## 2. Post-knowledge-cutoff eval harness (build step 73.1.2 [executor: sonnet-4.6/high])

Spec (verbatim from the gate):
- Add MODEL_CUTOFFS: dict[str,date] registry in config/model_tiers.py mapping each backbone id (gemini-2.5-flash, claude-opus-4-8, claude-haiku-4-5, claude-sonnet-4-6, ...) to its documented knowledge-cutoff date; unknown -> conservative None = treat all history as suspect.
- Eval-window selector: trusted_start = max(cutoff of every backbone in the path) + purge_embargo_buffer; trustworthy eval window = [trusted_start, now]; LABEL each window with the governing cutoff + which backbones it clears (New Quant §7.1 'strict publication cutoffs per evaluation fold').
- Rely on the Time-Machine principle (E5): our LIVE paper trail since each cutoff is inherently clean -- this is windowing/labeling, NOT retraining (we cannot train PiT models on API-only backbones).
- Scope: quant-only GBM backtest is EXEMPT (no LLM features per rules/backend-backtest.md); C2 governs ONLY the live 20-agent signal-promotion path.
- Grounds: New Quant §7.1 mandates cutoff-per-fold + embargo + rationales citing pre-decision evidence; Detecting-Lookahead-Bias placebo (beta3->0 post-cutoff) validates the pre/post boundary.

Integration seams: backend/config/model_tiers.py:50 (GEMINI_WORKHORSE) + :223-228 (claude model ids) -- registry home; NO cutoff constant exists today | backend/agents/llm_client.py (grep cutoff = empty -- gap) | backend/autoresearch/gate.py (promotion-path window guard bolt-on) | backend/autoresearch/strategy_backtest_adapter.py (eval-window enforcement) | backend/services/outcome_tracker.py EVAL_WINDOWS (labeled substrate)

Cost: $0. Date arithmetic + a constants table; no metered calls. Cutoff dates are operator-sourced (documented per backbone).

## 3. Counterfactual-audit gate — FactFin PC pilot (build step 73.1.3 [executor: opus-4.8/xhigh; METERED-COST PILOT, operator cost approval required])

Spec (verbatim from the gate):
- v1 = FactFin PC ONLY (Eq 4, text-outputs only -> Claude-compatible): perturb the candidate decision agent's context M times (news content-swap strong<->weak per E1 App C; numeric-feature Gaussian noise), re-run the SAME agent, PC = fraction of predictions unchanged; high PC => memorization => REJECT at promotion.
- DEFER CI/IDS to v2: CI needs a confidence scalar (verbalized conviction works) but IDS needs a class-distribution (multi-sample vote-share = M x S calls, ~10x cost).
- NO paper threshold exists (corrects 73.0's PC>0.7/IDS>0.6): calibrate a LOCAL cutoff on our own perturbation set; Table-5 anchor bands leaky PC~0.62/IDS~0.44 vs clean PC~0.31/IDS~0.78.
- Scope = tiny PILOT on a handful of promotion candidates, NOT the full universe; hard-cap M; gate behind D2b/#4 cost meter before any scale-up.
- Design must let the operator pick the CHEAPEST decision surface to perturb (unresolved data-gap: which live-decision agents are on metered vs flat-fee Max rail).

Integration seams: backend/autoresearch/gate.py (promotion-time bolt-on point) | backend/autoresearch/strategy_backtest_adapter.py (candidate-eval hook) | backend/services/meta_scorer.py:203 (unwrap_secret anthropic_api_key -- METERED) | backend/services/meta_scorer.py:221 (ClaudeClient claude-haiku-4-5 -- perturbed re-run executes here) | backend/services/meta_scorer.py:237 (temperature=0.0)

Cost: METERED (direct Anthropic API, CONFIRMED at meta_scorer.py:203/:221 -- NOT flat-fee Max). Refined from 73.0 'M~20-50': PC-only = M single Haiku calls/candidate (~M x ~2.25K tok; Haiku ~$1/$5 per Mtok -> ~$0.05-0.10/candidate at M=20). Full PC+CI+IDS = M x S calls ~10x (low $/candidate). Keep pilot small, cap M, meter before scaling.

## 4. CPCV wiring — robustness complement (build step 73.1.4 [executor: sonnet-4.6/high])

Spec (verbatim from the gate):
- Wire the existing cpcv_folds(n,k) (defined-but-unused at gate.py:42) into a NEW robustness path that replays STORED prices across phi=C(N,k)*k/N paths (E4), applying purge+embargo per fold (reuse _label_overlaps_test).
- Emit an OOS-Sharpe DISTRIBUTION (mean, std, worst-path, %paths Sharpe>0) as a robustness COMPLEMENT reported ALONGSIDE the existing K-variant CSCV PBO scalar the gate consumes -- do NOT replace it (strategy_backtest_adapter.py:38-41; frontier-map REAL REFINEMENT).
- gate.py stays byte-unchanged: the gate keeps reading trial['pbo'] from the K-variant CSCV path; CPCV feeds DSR-style robustness, not the PBO scalar.
- Build under the macro freeze (code-only over cached OHLC, NO historical_macro); validate the distribution at un-freeze.
- Path-count reference (E4): N=6,k=2 -> C(6,2)=15 splits -> 30 test-assignments -> 5 paths; each observation tested exactly once.

Integration seams: backend/autoresearch/gate.py:42-59 (cpcv_folds enumerator, currently only exported :62) | backend/autoresearch/strategy_backtest_adapter.py:38-41 (CPCV-complement note) | backend/autoresearch/strategy_backtest_adapter.py:95-152 (CSCV K-variant column-stack -- add sibling path) | backend/autoresearch/strategy_backtest_adapter.py:247 (compute_pbo call -- unchanged)

Cost: $0. Numpy/pandas over cached prices; no LLM, no macro, no metered spend. Build-dark under freeze, validate at un-freeze.

## Corrections carried (feed 73.1.3's contract)

- FactFin publishes NO formal accept/reject thresholds — 73.0's 'PC>0.7/IDS>0.6' was interpretive. The pilot calibrates a LOCAL cutoff on our own perturbation set (Table-5 anchor bands: leaky PC≈0.62/IDS≈0.44 vs clean PC≈0.31/IDS≈0.78).
- LAP diagnostic is BLOCKED for Claude backbones (requires logprobs); FactFin PC (text-only) is the Claude-compatible substitute; CI possible via verbalized conviction; IDS deferred (~10x cost).
- NO knowledge-cutoff constant exists anywhere in the codebase (grep-confirmed) — the MODEL_CUTOFFS registry is net-new; cutoff dates are operator-sourced per backbone.
- Which rail the live 20-agent decision path uses for perturbation re-runs is an unresolved data-gap — 73.1.3's executor must confirm before spending (hard-cap M=20; PC-only ≈ $0.05-0.10/candidate on metered Haiku).