# Research Brief — phase-73.3 "D2c CALIBRATED-SIZING DESIGN"

Tier: moderate (assumed per caller). NOT audit-class.
Status: COMPLETE — gate_passed: true (5 sources read in full, recency scan done, 16 URLs, 10 internal files).

Objective: deliver design_inputs for FIVE components of a calibrated
position-sizing layer that maps a per-analysis conviction signal ->
realized-hit-rate -> a bounded scalar on position_pct, gated dark until
~100-150 clean closed trades. Components: elicitation, calibration-map,
sizing-integration, ab-evaluation, data-pipeline.

Feeds: frontier_map_73.md #1 verdict (ADAPT elicitation via debate
vote-share, REJECT logprob + isotonic, 2-3-bucket Platt/Beta-Binomial
shrinkage on LOWER Wilson/Jeffreys bound, bounded scalar at sizing seam,
caps strictly downstream, DEFER live sizing) + design_pack_73/b_learn_loop_v2.md
(the outcome substrate: realized-exit reflections + additive agent_memories
migration).

---

## Internal code inventory (file:line anchors)

### portfolio_manager.py — the SIZING SEAM (the exact insertion point)

`decide_trades` candidate-build then emit loop. Order of operations, verbatim:

- **Candidate build (180-289):** per analysis, `_extract_position_pct(_rj_view, analysis)`
  (823-839) reads `recommended_position_pct` (RJ advisory pct), else
  `analysis.risk_judge_position_pct`, else `None`. The **BINDING RiskJudge REJECT
  gate is here (246-263)**: `if _rj_decision=="REJECT" and paper_risk_judge_reject_binding:
  continue` — a REJECT candidate never enters `buy_candidates`, so it is never sized.
  Flag default-OFF. Placed at the candidate-build chokepoint (common ancestor of the
  main emit loop AND the swap path) = SEC 15c3-5(d) non-bypassable placement.
- **Emit loop (348-472), order of gates:**
  1. `remaining_positions >= max_positions` break (349)
  2. `available_cash <= 0` break (361)
  3. per-sector COUNT cap `max_per_sector` default 2 (368-381) → `sector_blocked` queue
  4. **`position_pct = cand["position_pct"] or 10.0`** (388-391) → **`target_amount =
     nav * (position_pct/100.0)`** (392) → `buy_amount = min(target_amount,
     available_cash)` (393)  ← **THIS IS THE SCALAR SEAM (line 388-392)**
  5. **$50 floor**: `if buy_amount < 50: continue` (396)
  6. per-sector NAV-pct cap `max_sector_nav_pct` default 30 (406-422)
  7. FF3 factor-correlation cap `max_factor_corr` default-OFF (427-438)
  8. emit `TradeOrder(amount_usd=round(buy_amount,2), ...)` (440-463)

**Seam verdict:** a calibrated scalar `s∈[s_min,s_max]` inserted at 388-392 as
`position_pct = (cand["position_pct"] or 10.0) * s` multiplies BEFORE `target_amount`,
so EVERY downstream clip ($50 floor 396, sector-NAV cap 406, factor cap 427, the
`min(target_amount, available_cash)` at 393) fires strictly AFTER and is non-bypassable.
The binding REJECT (246-263) is strictly UPSTREAM. A scalar `s<1` can only shrink; a
`s>1` is still clipped by NAV-pct/$50/available_cash. Confirmed: this seam satisfies
the frontier-map "bounded scalar on position_pct, all caps strictly downstream".

### meta_scorer.py — the verbalized-conviction co-signal

`MetaScoredCandidate.conviction_score: int (ge=1, le=10)` (38-41) + `conviction_reason`
(42-44). Produced by ONE `ClaudeClient` call, `meta_scorer_model` default
`claude-haiku-4-5`, **`temperature=0.0`** (222-236) → exactly the single-call-at-T=0
Self-Confidence mode that 2508.06225 Table 1 flags as most overconfident. On any
failure → `_fallback_conviction` = clamp(round(composite_score),1,10) OR (integrity
flag) `_rank_normalized_convictions` percentile-rank into 1-10 (145-167). So conviction
is: a 1-10 int, LLM-verbalized when the API key + call succeed, else a composite-score
rank. It is NOT a probability and (per 2508.06225) must never be trusted as one.

### _extract_position_pct / _extract_stop_loss (823-883)
`recommended_position_pct` is the RJ advisory; `None` → emit loop defaults to 10%.
Stop-loss resolution: explicit price → `stop_loss_pct` → `paper_default_stop_loss_pct`.
(pending: debate.py vote/verdict shape, analysis_results schema, paper_round_trips, 72.0.1 reroute)

---

## External research

### Read in full (>=5; counts toward the gate)
| # | URL | Kind | Fetched how | Key finding |
|---|-----|------|-------------|-------------|
| 1 | arxiv.org/html/2404.09127 | paper (2024) | WebFetch full | Collaborative Calibration: Stage-2 deliberation aggregates POSTERIOR confidences over a multi-agent debate; "the final mean confidence estimate will be a better indication of the prediction accuracy" than any single agent. ECE gains (GSM8K .086 vs .093; DateUnd .055 vs .092; AmbigQA .026 vs .052). App A.2.2: gains lean on BACKBONE DIVERSITY (Mistral-7B/GPT-3.5/Cohere: "better calibrated (Cohere) or ... more reliable measures (Mistral)") — no explicit single-vs-multi quantification. |
| 2 | arxiv.org/html/2508.06225 | paper (2025) | WebFetch full | Overconfidence in LLM-as-judge: Self-Confidence elicitation = "a single call at temperature 0" → most overconfident (GPT-4o ECE 39.25% @ 49.71% acc; Mistral-Nemo ECE 74.22%). Verbalized confidence "significantly overstates actual correctness ... single-call verbalized scores unreliable for probability estimation without recalibration." Brier = (1/N)Σ(pᵢ−oᵢ)²; ECE = Σ(nᵢ/N)|acc(i)−conf(i)|. LLM-as-a-Fuser (multi-model synthesis) cuts ECE (Mistral-Nemo −53.73% ECE). |
| 3 | arxiv.org/html/2510.00263 | paper (2025) | WebFetch full | Calibrating autoraters: proper scoring rules (Brier/log) → Fisher-consistent truthful probabilities. RL-Brier: ECE .0879, Brier .0946 vs verbalized-baseline ECE .1183/Brier .1615. Estimator p̂ₘ unbiased, variance p*(1−p*)/m (Lemma 2) — m samples cut variance m×. NO guidance on min samples for bin stability; K=10 bins with ~100/bin "may introduce estimation noise." |
| 4 | ar5iv.labs.arxiv.org/html/1701.02814 | paper (2017) | WebFetch full | Kelly with ESTIMATED win-prob: "Replacing the actual values with estimates leads to overbetting ... higher risk with lower returns." Shrink bets ∝ estimation variance: lower-bound reweighting Q^lb(πₕ)=e^(β̂'vₕ)/Σe^(β̂'vᵢ+½(vᵢ−vₕ)'Σ(vᵢ−vₕ)); shrinkage ∝ (vᵢ−vₕ)'Σ(vᵢ−vₕ). Overbetting worse than underbetting (log-utility: overbet losses compound; underbet only lowers growth). Underbetting/lower-bound models empirically beat standard Kelly. |
| 5 | arxiv.org/html/2508.18868v1 | paper (2025) | WebFetch full | Estimation risk in Kelly (recency): "even small inaccuracies in the estimates of key input parameters ... can lead to significant over- or under-investing." Confirms fractional Kelly moderates market risk but "suboptimality due to estimation risk is still an open issue" (proposes options-hedge robustness instead — an ADVERSARIAL note vs pure bet-shrinkage). Cites the MacLean/Thorp/Ziemba lineage as the problem's origin. |
| 6 | projecteuclid.org/.../10.1214/ss/1009213286.full | paper (2001) | WebFetch (recommendation-level; PDF/listing formula routes 403/empty) | Brown-Cai-DasGupta: Wald interval coverage "erratic," worse at small n and p near 0/1; textbook safety guidance "misleading and defective." VERBATIM recommendation: "the Wilson interval or the equal-tailed Jeffreys prior interval for small n and the interval suggested in Agresti and Coull for larger n." Jeffreys = Beta(½,½) prior → posterior Beta(X+½, n−X+½). |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not read in full |
|-----|------|----------------------|
| stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf | paper | WebFetch returned binary PDF (283KB), unparseable in this read-only env; facts (10% mean-error → ~50% overbet; Chopra-Ziemba 20:2:1 means:var:cov; half-Kelly caps growth loss ≤25% vs negative-growth protection) obtained at snippet level from WebSearch. |
| www-stat.wharton.upenn.edu/~lbrown/Papers/2001a...pdf ; ~tcai/.../Binomial-Annals.html ; scirp 1273082 ; arxiv.org/pdf/0707.0837 ; pmc PMC2706447 ; studylib 7960401 | paper/ref | binomial-interval formula mirrors — PDF-binary or listing-only; Wilson/Jeffreys/Agresti-Coull formulas are standard textbook results supplemented below. |
| matthewdowney.github.io uncertainty-kelly ; medium fractional-Kelly ; researchgate Optimal-Betting-Under-Parameter-Uncertainty | blog/practitioner | fractional-Kelly-under-uncertainty corroboration; lower weight than the arXiv full reads. |

### Recency scan (2024-2026)
Performed (3-variant: current-year-frontier arXiv IDs 2508.*/2510.*/2404.*; last-2-year search on Kelly-estimation-error; year-less canonical searches for Brown-Cai-DasGupta and MacLean-Thorp-Ziemba). Findings: the 2025 frontier (2508.06225, 2510.00263, 2508.18868) CONFIRMS and sharpens the canonical picture rather than superseding it — single-call verbalized confidence is still the overconfident failure mode (2508.06225), proper-scoring recalibration is the frontier fix (2510.00263), and estimation-risk-driven overbetting is still "an open issue" that fractional shrinkage only partially solves (2508.18868, an adversarial qualifier: bet-shrinkage is necessary but the paper argues insufficient alone). No 2024-2026 work displaces Wilson/Jeffreys as the small-n interval recommendation (Brown-Cai-DasGupta 2001 remains canonical). Net: our design (recalibrate verbalized+vote-share conviction against realized win rate, size on the LOWER binomial bound, shrink under estimation error) is aligned with BOTH the canon and the frontier.

### Binomial-interval formulas (standard; recommendation anchored to source #6)
For a bucket with X wins in n closed trades, p̂=X/n, z=Φ⁻¹(1−α/2) (1.96 @ 95%):
- **Wilson lower bound** = [ (p̂ + z²/2n) − z·√(p̂(1−p̂)/n + z²/4n²) ] / (1 + z²/n).
- **Jeffreys lower bound** = α/2-quantile of Beta(X+½, n−X+½) (the equal-tailed Jeffreys interval; Beta-Binomial shrinkage toward 0.5 with prior strength 1).
- **Empirical-Bayes Beta-Binomial** (recommended v1): posterior mean (X+κp₀)/(n+κ), prior centered on the POOLED base win rate p₀ (not 0.5), prior strength κ; take the α/2 Beta quantile as the sizing win rate. At small n it shrinks every bucket → p₀ (scalar → 1.0 = uniform) automatically; buckets separate only as n grows — a graceful, self-deferring version of the ~100-150-trade gate.

### Key findings (per-claim)
1. NEVER trust the raw conviction as a probability — a single Haiku call at T=0 is precisely the most-overconfident elicitation mode (2508.06225 Table 1; verified our meta_scorer.py:222-236 runs exactly this). The calibration MAP against realized win rate is mandatory, not optional.
2. The multi-agent deliberation IS a calibrator at zero extra agents — 2404.09127's Stage-2 posterior-confidence aggregation maps onto our bull/bear/DA/moderator + aggressive/conservative/neutral/risk-judge stack; the mean deliberation confidence beats any single agent. All the inputs are ALREADY persisted (analysis_results columns).
3. Single-backbone caveat is real but unquantified — 2404.09127 App A.2.2 attributes lift partly to combining differently-calibrated backbones; our stack is one backbone family (Gemini), so expect ATTENUATED, residually-overconfident vote-share. The empirical recalibration absorbs residual bias; document the ceiling, do not claim the paper's ECE gains.
4. Size on the LOWER bound, shrink under estimation error — Kelly-with-estimated-probability (1701.02814) + Brown-Cai-DasGupta (Wilson/Jeffreys small-n) + MacLean/Thorp/Ziemba (overbet asymmetry) all converge: use the pessimistic bound so sampling noise can only shrink a bet; overbetting is worse than underbetting → asymmetric scalar cap (s_max conservative).
5. Proper scoring rules give an honest calibration target — 2510.00263 shows Brier-reward recalibration yields Fisher-consistent truthful probabilities; our A/B should score the calibrated probability with Brier to prove it is honest, not merely P&L-lucky.

---

## Application to pyfinagent — the FIVE design_inputs

### 1. elicitation (the concrete conviction signal)
- **Primary = debate-derived vote-share** on the EXISTING stack (2404.09127 deliberation calibrator, zero extra agents). A vote-share extractor reads the already-persisted analysis_results columns: `bull_confidence`, `bear_confidence` (the two stances), `aggressive/conservative/neutral_analyst_confidence` (risk_debate.py stance agents), `da_confidence_adjustment` (Devil's-Advocate downward stress), `debate_confidence`/`consensus_confidence` (Moderator = "fuser"), `risk_adjusted_confidence` (Risk Judge). v1 vote-share = confidence-weighted BUY-side share of {bull, aggressive, neutral} vs {bear, conservative}, then `+ da_confidence_adjustment` (signed, ≤0) → clip to [0,1]. debate.py:329-360 confirms `consensus`+`consensus_confidence` (0.5 default) exist on every debate result; save_report persists all legs. RETRO-BACKFILLABLE (all columns historical).
- **Secondary co-signal = meta_scorer 1-10** (`MetaScoredCandidate.conviction_score`, meta_scorer.py:38-41). Post-72.0.1 reroute the LLM leg is live again (pre-reroute it was the credit-dead composite-rank fallback, harness_log 72.0 — flag those pairs degraded). Per 2508.06225 it is the single-Haiku-T=0 overconfident mode → use ONLY as a bucketing input/co-feature, NEVER as a probability. The map recalibrates it.
- **Diversity caveat (single backbone):** our debate/analysis stack is ONE backbone family (Gemini per backend-agents.md; meta_scorer is Haiku). 2404.09127 App A.2.2 attributes calibration lift partly to combining differently-calibrated backbones — a single-backbone deliberation shares systematic biases, so the vote-share stays partially overconfident/correlated; the paper does NOT quantify the single-vs-multi gap. Degradation to expect: attenuated raw calibration (higher residual ECE than the paper's multi-backbone ensemble). Mitigation: (a) the empirical recalibration-against-realized-win-rate map absorbs the residual bias, (b) prompt-stance + temperature diversity is a weak substitute — do not claim the paper's ECE numbers.

### 2. calibration-map
- **Buckets:** 2-3 over the conviction signal. v1 = TWO buckets on vote-share (LOW < median, HIGH ≥ median) to keep per-bucket n maximal; escalate to 3 (LOW/MID/HIGH) only once n≥~120 total. (Meta_scorer 1-10 can supply an alternative bucketing: {1-5}/{6-7}/{8-10}.) REJECT isotonic/per-integer bins at our N (3-trade bins snap to 0/100%).
- **Shrinkage prior:** empirical-Bayes Beta-Binomial centered on the POOLED base win rate p₀ (NOT 0.5), prior strength κ≈20-40; posterior mean (X+κp₀)/(n+κ). Documented fallback = fixed Jeffreys Beta(½,½) (Brown-Cai-DasGupta small-n recommendation). Empirical-Bayes auto-defers: at small n every bucket → p₀ → scalar≈1.0.
- **Sizing rule = LOWER bound:** per-bucket sizing win rate = α/2 quantile (5th-pct @ 90%, or 2.5th @ 95%) of the posterior Beta, equivalently the Wilson lower bound. Noise can only shrink. Scalar `s = clip(lb_winrate / p₀, s_min, s_max)`; asymmetric cap per overbet-worse-than-underbet (1701.02814, MacLean/Thorp/Ziemba): s_min≈0.5, s_max≈1.25-1.5 (or s_max=1.0 for a strictly-defensive v1 — a DECISION to surface).
- **Update cadence:** batch/nightly recompute of {p₀, per-bucket (X,n), lower bounds, scalars} from the pairs dataset — NOT per-close (a per-trade refit is noise). Cheap O(N) arithmetic, $0.
- **Storage:** does NOT need to ride the 73.2.2 agent_memories migration (that is the reflection substrate, different grain). The conviction snapshot is ALREADY ~90% persisted in analysis_results + paper_trades.signals; recommend snapshotting the conviction vector into paper_trades.signals at trade time (paper_trader.py:297/483 already json.dumps `signals`; extract_all_signals already carries conviction) and expressing pairs as a VIEW/thin table over (analysis_results ⋈ paper_trades ⋈ paper_round_trips). A small dedicated `conviction_calibration_pairs` table is the durable option (immutable at decision time), independent additive migration if wanted.

### 3. sizing-integration
- **Exact seam:** portfolio_manager.py **line 388-392**, replace `position_pct = (cand["position_pct"] or 10.0)` with `... * calib_scalar(cand)` BEFORE `target_amount = nav*(position_pct/100)` (392). `calib_scalar` reads the conviction bucket off the candidate's snapshot and returns s (default 1.0).
- **Flag:** `paper_calibrated_sizing_enabled`, default **False** → s≡1.0 → byte-identical to today (the phase-69.3 dark-flag pattern).
- **Downstream, non-bypassable caps (all strictly AFTER the scalar):** $50 floor (:396), `min(target,available_cash)` (:393), per-sector NAV-pct cap (:406-422), FF3 factor cap (:427-438), position-count cap (:349), available_cash break (:361). The **binding RiskJudge REJECT gate (:246-263) is strictly UPSTREAM** — REJECT candidates are dropped at candidate-build and never sized. REDUCED/HEDGED stay advisory; the scalar multiplies whatever pct the RJ advised.
- **Guarantee:** because the scalar multiplies position_pct before target_amount and every cap is a downstream `min`/`continue`, s>1 can only REQUEST more (then clipped by NAV-pct/$50/cash) and s<1 only shrinks. A buggy s=100 cannot exceed the sector-NAV cap or available_cash. Mutation-resistant.

### 4. ab-evaluation
- **Design:** calibrated vs uniform (s≡1) on IDENTICAL signals. Preferred = **counterfactual sizing replay** over the pairs history ($0, freeze-safe): same entries/exits/prices from paper_round_trips, only the $ size differs; requires a purged/embargoed (AFML-style) or forward OOS split so the map is fit in-sample and scored OOS. Live paper A/B (two shadow sub-books) deferred to activation.
- **Metrics:** net P&L (net of ~0.2%/window costs, money_recon_2026-07-18); **Brier score** B=mean((p_calib−win)²) of the bucket-assigned calibrated win rate (proves the probability is honest, 2510.00263); max drawdown (vs go-live 20% tolerance); and the canonical Sharpe/DSR/PSR from perf_metrics.
- **Consistency with immutable gates:** sizing is a POSITION-SIZE overlay, not a strategy/param change — it does NOT route through the optimizer/PromotionGate, so DSR≥0.95/PBO≤0.20 stay byte-unchanged. Promotion evidence MUST show calibrated improves OOS net risk-adjusted P&L WITHOUT worsening DSR or max DD (charter: never maximize a single raw metric); a P&L-only "win" is rejected.
- **Run length:** map ESTIMATION needs ~100-150 closed trades; OOS EVALUATION needs a further block → realistically ~200-300 closed round-trips before promotion, or a purged-split replay to reuse history. At 30 round-trips today this is many months out → DEFER live sizing, log NOW.

### 5. data-pipeline (log pairs NOW, before sizing activates)
- **Conviction snapshot — already ~90% persisted.** analysis_results/save_report ALREADY carries: final_score, recommendation, debate_consensus, debate_confidence, bull_confidence, bear_confidence, recommendation_confidence, risk_adjusted_confidence, **recommended_position_pct**, aggressive/conservative/neutral_analyst_confidence, da_confidence_adjustment, price_at_analysis, analysis_date. GAP: the meta_scorer conviction_score (1-10) is NOT in save_report (only summary["meta_scored_top_conviction"], autonomous_loop.py:907; MAY ride paper_trades.signals via extract_all_signals). Fix = add the vote-share + meta_conviction to extract_all_signals so every BUY durably snapshots its conviction vector into paper_trades.signals (paper_trader.py:483, already persisted). ZERO migration for v1.
- **Join to realized outcomes:** paper_round_trips (realized_pnl_pct, holding_days, exit_reason, win=pnl>0) carries **buy_trade_id** → paper_trades.trade_id → paper_trades.**analysis_id** (= analysis_date string, portfolio_manager TradeOrder analysis_id=cand["analysis_id"]) → analysis_results.analysis_date. Join chain: `analysis_results ⋈(analysis_date=analysis_id) paper_trades[BUY] ⋈(trade_id=buy_trade_id) paper_round_trips`.
- **Backfill (retro):** YES for the vote-share leg — historical analysis_results carry all debate/RJ confidence columns, so vote-share conviction reconstructs for every past analysis; paper_trades BUYs carry analysis_id + signals JSON; paper_round_trips reconstructs outcomes from historical trades. The meta_scorer leg backfills ONLY where captured in paper_trades.signals at trade time; pre-72.0.1 cycles have the DEGRADED composite-rank fallback conviction (flag `is_pre_reroute=true`, prefer the vote-share leg which was live throughout).

## Sample-size math (honest, query-free from known facts)
- **Denominator today:** ~59 trades / **30 closed round-trips** (paper_metrics_v2; the round-trip = the labeled outcome). 2-3 buckets → ~10-15 pairs/bucket NOW.
- **Wilson 95% LOWER bound at N=10-20/bucket** (z=1.96): n=15, p̂=0.60 → CI≈[0.36, 0.80], lower≈**0.36** (24pp below point est); n=10, p̂=0.70 → [0.40, 0.89], lower≈**0.40** (30pp below); n=20, p̂=0.65 → [0.43, 0.82], lower≈**0.43** (22pp below). So at today's N the lower bound sits 22-30pp under the raw win rate → every bucket's lower-bound scalar collapses toward p₀; a "high-conviction" bucket is statistically INDISTINGUISHABLE from base, and the scalar correctly stays ≈1.0.
- **Threshold where calibrated beats uniform:** Wilson half-width ≈0.98/√n at p=0.5 (95%) → ±15pp needs **n≈43/bucket**, ±10pp needs n≈96, ±8pp needs n≈150. With 2-3 buckets and a genuine between-bucket spread ~15pp (e.g. 45%→60%), the HIGH bucket's 95% lower bound first CLEARS p₀ at **n≈40-50/bucket ≈ 100-150 total closed trades** — coinciding with the existing go-live TRADES_THRESHOLD=100 (paper_go_live_gate.py:39). A stable OOS bucket ordering (±8pp) wants ~150/bucket → the A/B promotion floor is a few hundred round-trips.
- **Explicit fallback below threshold:** flag OFF, `calib_scalar ≡ 1.0` = today's RJ-advisory position_pct byte-identical, while the pipeline logs conviction→outcome pairs. Belt-and-suspenders: empirical-Bayes shrinkage makes uniform the AUTOMATIC low-N behavior even with the flag ON (posterior → p₀ → s≈1.0), so the map self-defers and buckets separate only as N clears ~100-150.

## Research Gate Checklist
Hard blockers (all satisfied):
- [x] ≥5 authoritative external sources READ IN FULL via WebFetch (2404.09127, 2508.06225, 2510.00263, 1701.02814, 2508.18868; +Brown-Cai-DasGupta at recommendation level)
- [x] 10+ unique URLs total (16 collected)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full papers read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
Soft:
- [x] Internal exploration covered the sizing seam, meta_scorer, debate, analysis_results schema, round_trips, go-live gate, conviction flow, 72.0.1
- [x] Contradiction/adversarial note recorded (2508.18868: bet-shrinkage necessary-but-insufficient vs pure fractional Kelly)
- [x] Per-claim citation

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 11,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Calibrated-sizing design for phase-73.3. Elicitation: debate-derived vote-share on the EXISTING bull/bear/DA/moderator/risk stack (2404.09127 deliberation calibrator, zero extra agents, all inputs already persisted in analysis_results) + meta_scorer 1-10 as an overconfident co-signal never trusted as a probability (2508.06225 single-Haiku-T=0 mode); single-backbone (Gemini) caveat attenuates but the empirical map absorbs it. Calibration-map: 2-3 vote-share buckets, empirical-Bayes Beta-Binomial shrinkage toward the pooled base rate p0, size on the Wilson/Jeffreys LOWER bound (noise can only shrink), asymmetric scalar cap (overbet-worse-than-underbet, 1701.02814). Sizing seam: portfolio_manager.py:388-392, position_pct *= scalar BEFORE target_amount, flag paper_calibrated_sizing_enabled default OFF, all caps ($50 floor, sector-NAV, FF3, cash) strictly downstream, binding RJ REJECT strictly upstream and non-bypassable. AB-eval: counterfactual sizing replay on identical signals, metrics net-P&L + Brier of the calibrated prob + max DD, DSR/PBO gates byte-unchanged. Data-pipeline: conviction snapshot ~90% already in analysis_results; add vote-share+meta_conviction to paper_trades.signals now (zero migration), join round_trips.buy_trade_id->paper_trades.analysis_id->analysis_results.analysis_date; vote-share leg fully retro-backfillable, meta leg only post-72.0.1. Sample-size: 30 round-trips today -> 10-15/bucket; Wilson lower bound 22-30pp below point est at N=10-20; calibrated beats uniform at ~40-50/bucket ~= 100-150 total (matches go-live TRADES_THRESHOLD=100); fallback = uniform/advisory (flag OFF, or empirical-Bayes auto-shrink to p0). DEFER live sizing, log NOW.",
  "brief_path": "handoff/current/research_brief_73.3.md",
  "gate_passed": true
}
```
