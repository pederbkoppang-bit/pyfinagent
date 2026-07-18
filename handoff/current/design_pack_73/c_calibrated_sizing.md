# Design D2c — Calibrated Sizing (phase-73.3, FINAL — specs verbatim from gate `wf_95688663-b4a`, 5 sources read in full)

Frontier-map #1 verdict executed at implementation depth. Cornerstones (gate-verified): the raw meta-scorer conviction is exactly the overconfident single-call-T=0 mode 2508.06225 warns about — NEVER a probability; the deliberation stack is already a calibrator at zero extra agents with every input persisted in analysis_results (retro-backfillable); size on the LOWER binomial bound so noise can only shrink a bet (Kelly-under-estimation-error + Brown-Cai-DasGupta + MacLean-Thorp-Ziemba convergence); the sizing seam is exact and non-bypassable in both directions (binding REJECT upstream at portfolio_manager.py:246-263, every cap downstream of :388-392). Single-backbone caveat handled by empirical recalibration; do not claim the paper's ECE gains. Adversarial recency note: 2508.18868 argues shrinkage is necessary-but-insufficient (options-hedge remedy inapplicable to our long-only paper book) — lower-bound shrinkage + DEFER-until-N stands as v1. Full notes: `research_brief_73.3.md`.

## Honest sample-size math (verbatim from the gate)

DENOMINATOR TODAY: ~59 trades / 30 closed round-trips (paper_metrics_v2; the round-trip = the labeled outcome, win = realized_pnl_pct > 0). At 2-3 buckets that is ~10-15 pairs/bucket now. WILSON 95% LOWER BOUND behavior at N=10-20/bucket (z=1.96): n=15,p_hat=0.60 -> CI approx [0.36,0.80], lower ~=0.36 (24pp below the point estimate); n=10,p_hat=0.70 -> [0.40,0.89], lower ~=0.40 (30pp below); n=20,p_hat=0.65 -> [0.43,0.82], lower ~=0.43 (22pp below). So at today's N the lower bound sits 22-30pp under the raw win rate, every bucket's lower-bound scalar collapses toward the pooled base rate p0, and a high-conviction bucket is statistically INDISTINGUISHABLE from base -> the scalar correctly stays ~1.0. Jeffreys/Beta-Binomial behaves equivalently: the posterior Beta(X+a,n-X+b) alpha/2 quantile is the lower bound, and empirical-Bayes shrinkage (prior centered on p0, strength kappa~=20-40) pulls every thin bucket to p0. THRESHOLD WHERE CALIBRATED BEATS UNIFORM: Wilson half-width ~= 0.98/sqrt(n) at p=0.5 (95%) -> +/-15pp needs n~=43/bucket, +/-10pp n~=96, +/-8pp n~=150. With 2-3 buckets and a genuine between-bucket win-rate spread ~15pp (e.g. 45%->60%), the HIGH bucket's 95% lower bound first CLEARS p0 at n~=40-50/bucket ~= 100-150 total closed round-trips -- coinciding with the existing go-live TRADES_THRESHOLD=100 (paper_go_live_gate.py:39). A stable OOS bucket ordering (+/-8pp) wants ~150/bucket, so the A/B promotion floor is a few hundred round-trips. Given the current cadence (30 round-trips whole-window) this is many months out, which is exactly why live calibrated sizing is DEFERRED. EXPLICIT FALLBACK BELOW THRESHOLD: flag paper_calibrated_sizing_enabled OFF, calib_scalar == 1.0 = today's RJ-advisory position_pct byte-identical, while the pipeline logs conviction->outcome pairs so the map can be fit the moment N clears. Belt-and-suspenders: empirical-Bayes shrinkage makes uniform the AUTOMATIC low-N behavior even with the flag ON (posterior -> p0 -> s~=1.0), so the map self-defers and buckets separate only as N grows past ~100-150.

## 1. Elicitation — debate vote-share primary, meta-conviction co-signal (build step 73.3.1)

Spec (verbatim from the gate):
- PRIMARY = debate-derived vote-share on the existing stack (2404.09127 deliberation calibrator, zero extra agents). v1 vote-share = confidence-weighted BUY-side share of {bull, aggressive, neutral} vs {bear, conservative} + da_confidence_adjustment (signed, <=0), clipped to [0,1].
- Inputs are ALL already persisted analysis_results columns: bull_confidence, bear_confidence, aggressive/conservative/neutral_analyst_confidence, da_confidence_adjustment, debate_confidence/consensus_confidence (Moderator=fuser), risk_adjusted_confidence (Risk Judge). Retro-backfillable.
- SECONDARY co-signal = meta_scorer conviction_score 1-10 (MetaScoredCandidate). Live LLM leg only post-72.0.1 reroute (pre = credit-dead composite-rank fallback). Per 2508.06225 it is the single-Haiku-T=0 overconfident mode -> use only as a bucketing input/co-feature, NEVER as a probability.
- Single-backbone diversity caveat (2404.09127 App A.2.2): our stack is one backbone family (Gemini + Haiku); expect attenuated, residually-overconfident vote-share vs the paper's multi-backbone ensemble. No single-vs-multi quantification exists. Mitigate via the empirical recalibration map; do not claim the paper's ECE numbers; prompt-stance/temperature diversity is only a weak substitute.

Integration seams: backend/agents/debate.py:329-360 (consensus + consensus_confidence, DA confidence_adjustment) | backend/agents/risk_debate.py (aggressive/conservative/neutral analyst confidences + Risk Judge) | backend/db/bigquery_client.py save_report (persists bull_confidence, bear_confidence, debate_confidence, da_confidence_adjustment, risk_adjusted_confidence, recommendation_confidence) | backend/services/meta_scorer.py:38-41,222-236 (conviction_score int 1-10, single Claude call temperature=0.0) | backend/services/autonomous_loop.py:907 (meta_scored_top_conviction; 72.0.1 reroute)

Cost: $0. Vote-share is a pure arithmetic reduction over already-stored analysis_results columns -- no extra agents, no extra LLM calls. Meta_scorer is a pre-existing per-cycle Haiku call (post-72.0.1 on a working rail), not new spend.

## 2. Calibration map — 2-bucket Beta-Binomial shrinkage, lower-bound sizing (build step 73.3.1)

Spec (verbatim from the gate):
- 2-3 buckets. v1 = TWO vote-share buckets (LOW < median, HIGH >= median) to maximize per-bucket n; escalate to 3 (LOW/MID/HIGH) only at n>=~120 total. REJECT isotonic/per-integer bins at our N (3-trade bins snap to 0/100%).
- Shrinkage prior = empirical-Bayes Beta-Binomial centered on the POOLED base win rate p0 (not 0.5), prior strength kappa~=20-40; posterior mean (X+kappa*p0)/(n+kappa). Documented fallback = fixed Jeffreys Beta(1/2,1/2) (Brown-Cai-DasGupta small-n recommendation).
- Sizing rule = LOWER bound: per-bucket sizing win rate = alpha/2 quantile of the posterior Beta == Wilson lower bound, so small-sample noise can only shrink a bet.
- Scalar s = clip(lb_winrate / p0, s_min, s_max) with ASYMMETRIC cap per overbet-worse-than-underbet (1701.02814, MacLean-Thorp-Ziemba): s_min~=0.5, s_max~=1.25-1.5 (or s_max=1.0 for a strictly-defensive v1 -- a decision to surface).
- Update cadence = batch/nightly recompute of {p0, per-bucket (X,n), lower bounds, scalars}, NOT per-close (a per-trade refit is noise). Empirical-Bayes auto-defers: at small n every bucket shrinks to p0 -> s~=1.0.

Integration seams: new module e.g. backend/services/conviction_calibration.py (map fit + scalar lookup) | reads the conviction->outcome pairs dataset (see data-pipeline) | p0 = pooled win rate from backend/services/paper_round_trips.py summarize().win_rate | storage: does NOT ride the 73.2.2 agent_memories migration; VIEW over analysis_results + paper_trades + paper_round_trips, or an optional additive conviction_calibration_pairs BQ table (durable/immutable-at-decision)

Cost: $0. O(N) arithmetic (Beta quantiles / Wilson closed form), nightly. No embeddings, no API calls. BQ view is metadata-only; an optional pairs table is additive nullable (no backfill scan).

## 3. Sizing integration — bounded scalar at position_pct, flag dark (build step 73.3.2)

Spec (verbatim from the gate):
- Exact seam: portfolio_manager.py line 388-392 -- replace position_pct = (cand['position_pct'] or 10.0) with the same * calib_scalar(cand), evaluated BEFORE target_amount = nav*(position_pct/100.0) at :392.
- Flag paper_calibrated_sizing_enabled, default False -> calib_scalar == 1.0 -> byte-identical to today (phase-69.3 dark-flag pattern).
- All existing caps clip strictly DOWNSTREAM and stay non-bypassable: $50 floor (:396), min(target_amount, available_cash) (:393), per-sector NAV-pct cap (:406-422), FF3 factor cap (:427-438), position-count cap (:349), available_cash break (:361).
- Binding RiskJudge REJECT gate (:246-263) is strictly UPSTREAM at candidate-build -- REJECT candidates never enter buy_candidates, never sized. REDUCED/HEDGED remain advisory; the scalar multiplies whatever pct the RJ advised.
- Guarantee: s>1 can only REQUEST more (then clipped by NAV-pct/$50/cash), s<1 only shrinks; a buggy s=100 cannot exceed the sector-NAV cap or available_cash. Mutation-resistant by construction.

Integration seams: backend/services/portfolio_manager.py:388-392 (the multiply site) | backend/services/portfolio_manager.py:246-263 (binding RJ REJECT, upstream) | backend/services/portfolio_manager.py:396,406-422,427-438,393,349,361 (downstream caps) | backend/services/portfolio_manager.py:823-839 _extract_position_pct (RJ advisory pct source) | backend/config/settings.py (new flag paper_calibrated_sizing_enabled default False)

Cost: $0. Pure arithmetic on an already-computed position_pct. No new API calls, no metered spend, no new deps.

## 4. A/B evaluation — calibrated vs uniform promotion evidence (build step 73.3.2 live_check + 73.3.1 replay)

Spec (verbatim from the gate):
- Calibrated vs uniform (s==1) on IDENTICAL signals. PREFERRED = counterfactual sizing REPLAY over the pairs history ($0, freeze-safe): same entries/exits/prices from paper_round_trips, only the $ size differs; requires a purged/embargoed (AFML) or forward OOS split (fit in-sample, score OOS). Live paper A/B (two shadow sub-books) deferred to activation.
- Metrics: net P&L (net of ~0.2%/window costs per money_recon_2026-07-18); Brier score B=mean((p_calib-win)^2) of the bucket-assigned calibrated win rate (proves the probability is honest, 2510.00263); max drawdown vs the go-live 20% tolerance; canonical Sharpe/DSR/PSR from perf_metrics.
- Consistency with immutable gates: sizing is a position-size overlay, not a strategy/param change -> does NOT route through the optimizer/PromotionGate, so DSR>=0.95 / PBO<=0.20 stay byte-unchanged. Promotion evidence must show calibrated improves OOS net risk-adjusted P&L WITHOUT worsening DSR or max DD (charter: never maximize a single raw metric); a P&L-only win is rejected.
- Run length: map estimation ~100-150 closed trades; OOS evaluation a further block -> realistically ~200-300 round-trips before promotion, or a purged-split replay to reuse history. At 30 round-trips today this is months out -> DEFER live sizing, log NOW.

Integration seams: backend/services/paper_round_trips.py (realized outcomes for the replay) | backend/services/perf_metrics.py compute_dsr/compute_psr/compute_sharpe (canonical, reuse -- do not recompute) | backend/services/paper_go_live_gate.py:39 TRADES_THRESHOLD=100 (aligns the deferral) | backend/autoresearch/gate.py PromotionGate (byte-unchanged; sizing does not pass through it)

Cost: $0 for the replay path (arithmetic over stored trades). Live paper A/B (deferred) needs a shadow sub-book but still no metered spend on the flat-fee rail.

## 5. Data pipeline — conviction→outcome pairs, START NOW (build step 73.3.1)

Spec (verbatim from the gate):
- Conviction snapshot is ~90% ALREADY persisted: analysis_results/save_report carries final_score, recommendation, debate_consensus, debate_confidence, bull_confidence, bear_confidence, recommendation_confidence, risk_adjusted_confidence, recommended_position_pct, aggressive/conservative/neutral_analyst_confidence, da_confidence_adjustment, price_at_analysis, analysis_date.
- GAP: meta_scorer conviction_score 1-10 is NOT in save_report (only summary['meta_scored_top_conviction'], autonomous_loop.py:907; may ride paper_trades.signals via extract_all_signals). Fix = add vote_share + meta_conviction to extract_all_signals so every BUY durably snapshots its conviction vector into paper_trades.signals (paper_trader.py:483, already json.dumps'd). ZERO migration for v1.
- Join to realized outcomes: paper_round_trips.buy_trade_id -> paper_trades.trade_id -> paper_trades.analysis_id (= analysis_date string, portfolio_manager TradeOrder analysis_id=cand['analysis_id']) -> analysis_results.analysis_date.
- Backfill (retro) = YES for the vote-share leg (historical analysis_results carry all debate/RJ confidence columns; paper_trades BUYs carry analysis_id+signals; round_trips reconstructs outcomes). Meta_scorer leg backfills ONLY where captured in paper_trades.signals at trade time; pre-72.0.1 cycles have the degraded composite-rank fallback conviction -> flag is_pre_reroute=true, prefer the vote-share leg.
- Start logging NOW, before sizing ever activates -- the pairs accumulate against the ~100-150-trade deferral while the flag stays OFF.

Integration seams: backend/services/portfolio_manager.py extract_all_signals (add vote_share + meta_conviction to the signals payload) | backend/services/paper_trader.py:297,483 (signals json.dumps -> paper_trades, already persisted) | backend/db/bigquery_client.py save_report (conviction columns already present; no change needed for the vote-share leg) | backend/services/paper_round_trips.py:99-124 (round-trip rows carry buy_trade_id/sell_trade_id/realized_pnl_pct/holding_days/exit_reason) | optional additive BQ table conviction_calibration_pairs (analysis_date, ticker, buy_trade_id, vote_share, meta_conviction, recommended_position_pct, realized_pnl_pct, win, holding_days, exit_reason, is_pre_reroute)

Cost: $0. Vote-share is derived from stored columns; adding fields to the existing paper_trades.signals JSON needs no migration. An optional dedicated pairs table is additive nullable (metadata-only, no table rewrite). No metered spend.

## Executor note from Q/A wf_10bcde12-835 (binding for 73.3.2)

pm.py:388-392 is now a TWO-WAY branch on `paper_risk_judge_shape_fix_enabled` — the literal `position_pct = (cand['position_pct'] or 10.0)` exists only in the else-branch. Insert `position_pct *= calib_scalar(cand)` AFTER the if/else (before `target_amount` at :392), never a string-replace that would miss the flag-ON branch. The non-bypassability guarantee is branch-independent; the live_check's byte-identical-when-OFF proof enforces the both-branches patch.

## Decisions of record + operator decision surfaced

- v1 = TWO buckets (LOW/HIGH at the median); escalate to 3 only at n≥~120. Nightly batch recompute; empirical-Bayes auto-defers (small n ⇒ every bucket shrinks to p0 ⇒ scalar ≈ 1.0 — the design is safe to ship dark long before the data supports it).
- Activation bar ≈ 100-150 closed round trips (40-50/bucket), coinciding with paper_go_live_gate TRADES_THRESHOLD=100. Today's 30 trips ⇒ Wilson 95% lower bound 22-30pp below point estimate ⇒ scalar correctly ~1.0. Fallback below bar = uniform/advisory-pct with pair logging running.
- Pre-72.0.1 meta-conviction is degraded (credit-dead composite fallback) — backfilled pairs flag is_pre_reroute=true; the vote-share leg was live throughout and is the primary.
- **OPERATOR DECISION SURFACED (for the decision sheet at 73.6/73.7 rollup)**: scalar upper cap s_max — strictly-defensive v1 (s_max=1.0: calibration can only SHRINK bets) vs modest upside (s_max=1.25-1.5). Asymmetric either way (s_min≈0.5); overbetting is worse than underbetting.