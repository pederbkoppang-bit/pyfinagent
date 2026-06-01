# research_brief -- strategy rotation (HARD-STOP element 2): alt-strategy labeling + DSR selector

**Tier:** complex | **Started:** 2026-06-01 | **Researcher:** Layer-3 Harness
**North-star:** maximize live money; rotate to highest earner from a CITED research basis.
**Constraint:** working triple_barrier/meta_label US path must NOT regress. $0 LLM.

Status: IN PROGRESS (writing as I go).

---

## Q1 ROOT CAUSE (internal) -- WHY do alt strategies emit ~all-neutral labels -> 0 trades?

### Q1.0 -- THE HEADLINE (two distinct root causes, both file:line-pinned)

There are **TWO** root causes, and they sit at different layers. The caller's
framing ("alt strategies emit all-neutral -> 0 trades") is **half the story** --
the more important finding is an ARCHITECTURAL DISCONNECT that means rotation
cannot affect live money TODAY even if the alt strategies traded perfectly.

**RC-A (the BACKTEST labeling problem -- why alt strategies under-trade in backtests):**
A **multi-class neutral-dominance + no-class-balancing** failure. The alt-strategy
label functions emit `{-1, 0, +1}` with the **0 (neutral) class vastly dominant**,
and the `GradientBoostingClassifier` is trained with **NO class weighting**, so it
learns to predict the majority (neutral=0) class. The trade gate only opens a
position on a predicted `label == +1` (`backtest_trader.py:132`), so a model that
predicts mostly 0 fires few/no BUYs -> ~0 trades -> Sharpe 0.0000 (the "crash"
rows in `quant_results.tsv`). triple_barrier escapes this because its barriers are
**cost-shifted and asymmetric** (sl 12.9% vs tp 10%) so a meaningful fraction of
samples hit +1/-1; the alt strategies use **hard double-threshold gates** that put
the overwhelming majority of samples in the 0 bucket.

**RC-B (the LIVE disconnect -- why rotation cannot drive live money AT ALL today,
independent of RC-A):** The BACKTEST strategy taxonomy and the LIVE strategy
taxonomy are **TWO DIFFERENT SYSTEMS that do not share code**. The 5 backtest
strategies (triple_barrier / quality_momentum / mean_reversion / factor_model /
meta_label) are ML-label-classifier functions in `backtest_engine.py`. The LIVE
money loop NEVER calls them. Live ranking is `screener.rank_candidates(...,
strategy="momentum")` (default arg) at `autonomous_loop.py:621` -- and that call
**passes NO `strategy=` argument**, so it is hardwired to the screener's
`"momentum"` branch (`screener.py:268`). The screener only knows two strategies,
`"momentum"` and `"value_momentum"` (`screener.py:225,268,283`) -- NEITHER is one
of the 5 backtest strategies. `best_params["strategy"]` (e.g. `triple_barrier`) IS
loaded into the live cycle (`autonomous_loop.py:211` -> `:1128`) but is used ONLY
to write the `strategy_decisions` heartbeat row (`:1128-1141`); it is **never
threaded into `rank_candidates` or `decide_trades`**. So even a PERFECT rotation
verdict ("switch to mean_reversion") changes a BQ heartbeat row and NOTHING about
which stocks the live engine buys.

**Consequence for the strategic fork:** "fixing alt-strategy labeling so rotation
has something to rotate between" (RC-A) is necessary but NOT sufficient -- without
also bridging the selected strategy into the live ranking/labeling path (RC-B),
rotation remains a backtest-only exercise with **zero live-money effect**. The
DEFERRED notes in the rotation code already concede this: `strategy_registry.py:37-41`
and `rotation_runner.py:37-39` both say "best_params is NOT threaded into
decide_trades/paper_trader ... flipping a promoted_strategies row alone changes
only the heartbeat, not live orders." **This is the load-bearing fact for Q2.**

### Q1.1 -- RC-A mechanism, pinned to the label functions

The label -> trade chain:
1. `_build_training_data` (`backtest_engine.py:551`) loops candidates x biweekly
   dates, calls `_compute_label` (`:587`) per sample, collects `labels_list`.
   `_compute_label` (`:1022`) dispatches via `STRATEGY_REGISTRY` (`:32-38`) to the
   per-strategy function.
2. `_train_model` (`:720-737`) fits `GradientBoostingClassifier(...)` with
   `sample_weight=` (average-uniqueness, temporal) but **NO `class_weight`** --
   and sklearn's `GradientBoostingClassifier` does not even accept `class_weight`,
   so there is NO mechanism here to counter class imbalance. CONFIRMED `:727-734`.
3. `_predict_and_trade` (`:765`) does `pred_label = int(model.predict(X_test)[0])`
   (`:803`). With a neutral-dominant training set the model predicts 0 for almost
   every test ticker.
4. `execute_trades` (`backtest_trader.py:94`) opens a position ONLY for
   `s["label"] == 1` (`:132`); `label == -1` only closes an existing position
   (`:113`). So predicted-0 (and predicted--1 with no open position) -> NO trade.
   This is the gate that turns "mostly-neutral predictions" into "0 trades".

**Why each alt strategy is neutral-dominant (the actual thresholds):**
- `_compute_quality_momentum_label` (`:1030`): returns +1 only if `momentum_6m > 5
  AND quality_score > 0.3`; -1 only if `momentum_6m < -5 AND quality_score < 0.1`;
  **else 0** (`:1045-1049`). The conjunction of a >5% 6-month momentum AND a
  quality_score above 0.3 is rarely BOTH true; the -1 branch needs quality < 0.1
  (almost never). -> overwhelmingly 0.
- `_compute_mean_reversion_label` (`:1051`): Stage-1 requires `sma_dist < -0.05 AND
  rsi < 35` (oversold) or `sma_dist > 0.10 AND rsi > 70` (overbought); if neither,
  **return 0** immediately (`:1082-1083`). Then Stage-2 requires the price to
  actually revert by >= half the SMA gap within `mr_holding_days` or it ALSO
  returns 0 (`:1112`). TWO gates in series -> the vast majority are 0. (Also note:
  in the backtest seed `mr_holding_days` default is 15 but the long shared
  `holding_days` interacts; the 2-stage filter is the dominant neutral driver.)
- `_compute_factor_label` (`:1114`): builds a 5-factor composite in [0,1], returns
  +1 only if `composite > 0.6`, -1 if `composite < 0.3`, **else 0** (`:1187-1191`).
  A sigmoid-normalized weighted average of 5 factors centered on S&P medians
  clusters tightly around 0.5 -> rarely exceeds 0.6 or drops below 0.3 -> mostly 0.
- CONTRAST `_compute_triple_barrier_label` (`:647`): walks forward and returns +1
  on the FIRST TP touch, -1 on the FIRST SL touch, 0 only if time expires
  (`:678-685`). Over a 90-day holding window with a 10% TP / ~13% SL, a large
  fraction of paths touch a barrier -> a workable mix of +1/-1/0. THIS is why
  triple_barrier (and meta_label, which reuses TB labels per `:37`) trade and the
  others do not.

### Q1.2 -- LIVE EVIDENCE the alt strategies don't trade (quant_results.tsv)

`backend/backtest/experiments/quant_results.tsv` (521 rows). Every row that flips
`strategy: triple_barrier -> {quality_momentum | mean_reversion | factor_model}`
shows one of two failure signatures:
- `metric_after = 0.0000`, `status = crash`, `dsr = 0.0000` -- the **0-trades**
  signature (no trades -> empty return series -> Sharpe 0). Examples: exp31/exp44
  factor_model, exp76 quality_momentum (all 2026-04-06).
- `metric_after` strongly NEGATIVE (e.g. `mean_reversion -6.1324`, `factor_model
  -1.2129`, `quality_momentum -0.5942`), `status = discard` -- these DID trade in
  some windows but **lost money** (negative Sharpe). So the alt strategies are not
  uniformly 0-trade; they oscillate between "0 trades" (most windows neutral) and
  "a few bad trades" (occasional threshold crossings that don't predict returns).
  EITHER WAY they never beat the `triple_barrier` baseline of **1.1705** -- every
  alt-strategy row has a NEGATIVE `delta` vs baseline. **NONE cleared DSR; all were
  discarded/crashed.** This is the empirical basis for the "low money value" prior.

### Q1.3 -- The rotation stack (48.x) IS built; it stops at audit-only

- `strategy_registry.py` (48.1): `SEED_STRATEGIES` = 4 seeds (tb_baseline,
  mr_short_horizon, qm_trend_tilt, tb_risk_managed). NOTE 2 of 4 are
  `triple_barrier` variants (tb_baseline, tb_risk_managed); the docstring concedes
  tb_risk_managed is "deliberately a CORRELATED variant" and (`rotation_runner.py:16-29`)
  that its trailing/vol-target overrides are INERT (engine readers reverted in
  commit 9fbd9cd6). So the "diversity" is really **2 distinct types** (TB,
  MR/QM) not 4.
- `strategy_backtest_adapter.py` (48.2) `make_engine_backtest_fn`: real-engine
  per-seed DSR/PBO via param-variants. BUILT.
- `strategy_candidate_producer.py` (48.1) `run_strategy_bakeoff`: scores each seed,
  feeds the selector. BUILT.
- `strategy_selector.py` (47.6) `select_best_strategy`: **the DSR selector** --
  gate-filters on DSR>=0.95 AND PBO<=0.20 (reusing `gate.PromotionGate`), ranks
  passers DSR-desc/PBO-asc, applies anti-churn hysteresis (`min_improvement=0.01`),
  RETAINS incumbent if no passer. WELL-DESIGNED, BUILT, pure-function tested.
- `rotation_runner.py` (48.3) `run_rotation_bakeoff` + `make_rotation_engine`:
  full-ctor-kwarg engine factory + live runner. Persists the verdict to
  `rotation_log.jsonl` at **`allocation_pct=0.0` -- "recorded, NOT deployed"**
  (`:220`). The deployment bridge (params -> `settings.paper_*` + a
  `promoted_strategies` MERGE that actually changes orders) is **explicitly
  DEFERRED** (`:37-39`).

**So the selector exists and is correct; what is missing is (1) alt strategies
that actually trade & clear the gate [RC-A], and (2) a bridge from the verdict to
the live ranking path [RC-B].** Per-strategy DSR is NOT yet populated from 5 live
$0-LLM backtests (`strategy_selector.py:26-31` DEFERRED).

## PART B -- EXTERNAL RESEARCH

### Read in full (>=5 required; 8 read; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.sefidian.com/2021/06/26/labeling-financial-data-for-machine-learning/ | 2026-06-01 | tutorial / AFML Ch.3 walkthrough | WebFetch (full) | **DIRECT RC-A HIT.** "The labels in the original dataset are heavily dominated by 0 values, so if we train on those, we get a degenerate model that predicts 0 every time." Fix = vol-scaled DYNAMIC barriers ("set them based on the prices' approximate daily movement over the last 100 days ... EWMA") + SMOTE rebalancing. "Fixed thresholds make it difficult to accurately label ... low-volatility periods generate mostly neutral labels." |
| https://arxiv.org/html/2402.05272 | 2026-06-01 | peer-reviewed preprint (q-fin) | WebFetch (full, arXiv HTML) | Regime-switching JUMP MODEL beats buy-and-hold OOS: S&P500 Sharpe 0.68 vs 0.48, MDD -26.6% vs -55.2%. **Switch penalty cut turnover to 44% (vs HMM 141-290%)** while improving net-of-cost Sharpe; jump penalty cut regime shifts 9.7/yr -> 0.4-0.5/yr. JM > HMM (HMM makes "numerous short-lived regimes ... difficult to trade"). Warning: regime latency ~15 days misses rebounds. |
| https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-06-01 | peer-reviewed (Bailey & LdP) | WebFetch (full PDF) | DSR deflates SR for N trials, non-normality, sample length. **E[max SR] ~ (2 log N)^0.5 - gamma/(2(2 log N)^0.5)**, gamma=0.5772. "When strategies are correlated rather than independent, practitioners should use the EFFECTIVE number of independent trials rather than the raw count." "selecting the best of many backtested strategies inflates the apparent Sharpe." |
| https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-06-01 | peer-reviewed (Bailey & Borwein) | WebFetch (full PDF) | PBO via CSCV. **"strategies selected for best in-sample performance systematically underperform the median performance across all tested configurations when overfitting is present."** High PBO <=> larger IS-OOS gap. PBO rises non-linearly with the number of configs tested. |
| https://www.vertoxquant.com/p/the-effective-number-of-tested-strategies | 2026-06-01 | practitioner (rigorous) | WebFetch (full) | K_eff: "if all your strategies are perfectly correlated/anticorrelated, then you've effectively only tested a single strategy"; uncorrelated -> K_eff=K; correlated -> K_eff->1. **E[max|Z|] ~ sqrt(2 log K_eff)** -- inflation grows with K_eff NOT raw K. "achieving K_eff near K requires genuine independence rather than mere parameter variation." |
| https://arxiv.org/html/2603.09219 | 2026-06-01 | peer-reviewed preprint (q-fin, 2026) | WebFetch (full, arXiv HTML) | IS->WFA->OOS gated protocol. **"v1 has the highest IS Sharpe but the weakest OOS -- illustrating the risk of selecting alphas by IS peak"** (v1 IS 2.43/OOS 2.19; conservative v3 OOS 2.61). Recommends PLATEAU selection (Omega_stable = {theta: SR(theta) >= 0.9*SR_opt}) over PEAK selection. Purged walk-forward + catastrophic-veto gates. Does NOT ensemble -- locks a single plateau config. |
| https://www.mql5.com/en/articles/19850 | 2026-06-01 | practitioner (AFML Part 4) | WebFetch (full) | Label-concurrency fixes: `max_samples<<1.0`, uniqueness sample weights (tW), sequential bootstrapping; **explicitly endorses `class_weight='balanced'` "to handle imbalanced labels, following King & Zeng (2001)."** (Concurrency is orthogonal to the neutral-dominance problem, but the class_weight fix is the cross-confirming pointer for RC-A.) |
| https://www.ubp.com/en/news-insights/newsroom/systematic-multi-strategy-as-a-portfolio-diversifier | 2026-06-01 | industry (practitioner) | WebFetch (full) | "SMS strategies, by design, have historically exhibited low or near-zero equity correlation." Diversification across alpha strategies is "fundamental." (Supports ENSEMBLE thesis; does NOT quantify the Sharpe uplift or address crisis correlation.) |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.mdpi.com/1911-8074/19/1/70 | peer-reviewed (sector rotation TSX 60, 2000-2025) | MDPI returned HTTP 403 on both `/70` and `/70/htm`; the WebSearch snippet carries the load-bearing numbers (rotation Sharpe 0.922 vs B&H 0.624 = +48%; OOS 2020-2025 16.94% vs 15.53%; **MEDIAN-performer selection + quarterly rebalance beat picking the single best**). |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | reference | DSR formula corroboration; the Bailey primary PDF was read in full instead |
| https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 | peer-reviewed (backtest overfitting OOS methods, 2024) | paywalled abstract; CSCV/DSR comparison covered by the two Bailey PDFs read in full |
| https://www.mdpi.com/2227-9091/11/5/93 | peer-reviewed (diversification framework, multiple pairs strategies) | snippet confirms low cross-spread correlation -> diversification benefit; not load-bearing beyond UBP read |
| https://www.priceactionlab.com/Blog/2024/01/mean-reversion-and-momentum-regime-switching/ | practitioner blog | mean-reversion+momentum regime switch Sharpe>1 (2003-2025); corroborates the jump-model read |
| https://deepwiki.com/quantopian/mlfinlab/6.3-triple-barrier-method | docs (mlfinlab) | confirms 0=neutral when no barrier touched; mechanics covered by sefidian read |
| https://www.morningstar.com/portfolios/these-diversification-strategies-are-winning-2026 | industry (2026) | 2026 concentration/rotation context; recency-scan corroboration |
| https://www.withintelligence.com/insights/hedge-fund-outlook-2026/ | industry (2026) | multi-strategy platforms favored 2026; recency only |
| https://funds.aqr.com/Insights/Strategies/Multi-Strategy | industry (AQR) | multi-strategy = different return sources, low mutual correlation; recency corroboration |

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "ensemble multiple trading strategies versus winner-take-all rotation diversification uncorrelated alpha systematic equity 2026"; "multi-strategy ensemble versus single best strategy live trading correlation diversification benefit 2025 2026 quant".
2. **Last-2-year window (2025/2024):** "strategy rotation regime switching out-of-sample deflated Sharpe ratio overfitting beat buy and hold best strategy 2025" (-> MDPI TSX 2000-2025, jump-model arXiv 2024) -- see Recency scan.
3. **Year-less canonical:** "triple barrier labeling neutral class imbalance financial machine learning Lopez de Prado fix"; "strategy selection multiple testing E[max Sharpe] sqrt(2 log N) best of N backtest overfitting probability" (-> the canonical Bailey DSR + PBO papers, the K_eff piece).

### Recency scan (2024-2026) -- PERFORMED
Searched the last-2-year window on (a) regime-switching/rotation OOS vs buy-and-hold, (b) ensemble-vs-rotation for systematic equity, (c) backtest-overfitting OOS testing methods. **Findings (COMPLEMENT the canonical Bailey/LdP prior art; one materially sharpens the recommendation):**
1. **Jump-model regime switching (arXiv 2402.05272, 2024) is the current frontier over HMM** and quantifies the turnover/switch-penalty tradeoff (44% vs 141-290%) -- this is NEW vs the 2014 DSR/PBO papers and DIRECTLY validates the selector's existing anti-churn hysteresis (`strategy_selector.py:8-11` already cites a "jump-model 2024: switch penalty cut turnover 141% -> 44%" -- CONFIRMED that citation is accurate and load-bearing).
2. **AlgoXpert (arXiv 2603.09219, 2026)** is the most recent and most relevant: it empirically shows the IS-peak alpha (v1) is the WEAKEST OOS and recommends PLATEAU (stable-region) selection over peak selection. This SHARPENS the recommendation: a DSR-ranked winner-take-all selector should prefer a robust plateau, not the raw IS-max -- the project's DSR+PBO gate already approximates this (DSR deflates the peak; PBO rejects overfit peaks), but the plateau idea is an explicit enhancement.
3. **2026 market context (Morningstar, WithIntelligence, AQR):** post-2025 mega-cap concentration unwinding makes low-correlation multi-strategy diversification topical; "there is no single best strategy ... a portfolio of low-correlation strategies is more durable" (2026 practitioner consensus). BUT the same sources warn "correlation can collapse to one during sharp risk-off events" -- diversification is not free insurance.
4. **No 2024-2026 source overturns** the DSR/PBO multiple-testing framework; they EXTEND it (K_eff for correlation, plateau selection, jump-penalty turnover control). The winner-take-all-rotation-vs-ensemble debate is LIVE: rotation works OOS in the sector-ETF setting (MDPI, jump-model) but the practitioner/AQR consensus leans toward RUNNING strategies in parallel (ensemble) rather than rotating winner-take-all, because (a) rotation pays turnover and (b) regime timing has latency (jump-model: ~15-day lag).

### Key findings (per-claim, cited)
1. **Neutral-class dominance is a KNOWN, NAMED failure mode with a standard fix** -- "The labels ... are heavily dominated by 0 values, so if we train on those, we get a degenerate model that predicts 0 every time" (Source: Sefidian AFML labeling, https://www.sefidian.com/2021/06/26/labeling-financial-data-for-machine-learning/, accessed 2026-06-01). The two standard fixes are (a) **volatility-scaled dynamic barriers** instead of fixed-% thresholds and (b) **class balancing** (SMOTE, or `class_weight='balanced'`). This is the textbook remedy for pyfinagent's RC-A.
2. **`class_weight='balanced'` is the canonical class-imbalance lever for these labels** -- "assign class_weight='balanced' to handle imbalanced labels, following King & Zeng (2001)" (Source: MQL5 AFML Part 4, https://www.mql5.com/en/articles/19850). NOTE pyfinagent's `GradientBoostingClassifier` does NOT support `class_weight` -- so the fix is either (i) switch to a classifier that does (HistGradientBoosting/RandomForest/sklearn's `class_weight`), (ii) SMOTE the training set, or (iii) FIX THE BARRIERS so the labels are less imbalanced in the first place (preferred -- attacks the cause not the symptom).
3. **Out-of-sample strategy rotation CAN beat buy-and-hold, but the edge is modest and turnover-sensitive** -- sector rotation OOS 16.94% vs 15.53% buy-and-hold, Sharpe 0.922 vs 0.624 (Source: MDPI TSX 60 2000-2025 snippet, https://www.mdpi.com/1911-8074/19/1/70); regime-switch Sharpe 0.68 vs 0.48 with turnover controlled to 44% (Source: jump-model arXiv https://arxiv.org/html/2402.05272). The uplift is real but single-digit annualized; it REQUIRES a turnover penalty to survive net-of-cost.
4. **Selecting the single best in-sample performer is the canonical overfitting trap; prefer median/plateau** -- "strategies selected for best in-sample performance systematically underperform the median ... when overfitting is present" (Source: Bailey & Borwein PBO, https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf); "v1 has the highest IS Sharpe but the weakest OOS" -> use plateau selection (Source: AlgoXpert arXiv https://arxiv.org/html/2603.09219); MDPI found MEDIAN-performer selection beat picking the best. **=> a winner-take-all DSR selector MUST deflate hard and prefer robustness, exactly what DSR+PBO do.**
5. **Correlated seeds must be deflated by K_eff, not raw N** -- "if all your strategies are perfectly correlated ... you've effectively only tested a single strategy"; E[max] ~ sqrt(2 log K_eff) (Source: VertoxQuant https://www.vertoxquant.com/p/the-effective-number-of-tested-strategies; Bailey DSR "use the effective number of independent trials"). pyfinagent's 4 seeds include 2 triple_barrier variants -> K_eff << 4; the selector's `num_trials=5` (raw) OVER-deflates (the SAFE direction, already noted at `strategy_selector.py:26-31`), but the diversity is illusory.
6. **The 2026 practitioner consensus leans ENSEMBLE over winner-take-all rotation** -- "There is no single best strategy. Instead, a portfolio of low-correlation strategies is more durable" (2026 multi-strategy consensus); "SMS strategies ... near-zero equity correlation" (Source: UBP https://www.ubp.com/en/news-insights/newsroom/systematic-multi-strategy-as-a-portfolio-diversifier; AQR Multi-Strategy). Caveat: "correlation can collapse to one during sharp risk-off events" -- diversification smooths normal regimes, not crashes.

### Consensus vs debate (external)
- **Consensus:** (a) neutral-dominant labels -> degenerate all-0 model is a named, fixable failure (vol-scaled barriers + class balancing); (b) selecting the IS-best of many strategies inflates Sharpe and disappoints OOS -- deflate (DSR) and reject overfit (PBO); (c) correlated trials must be counted as K_eff not N; (d) regime/strategy rotation needs a turnover/switch penalty to survive net-of-cost.
- **Debate:** **winner-take-all ROTATION vs parallel ENSEMBLE.** Rotation beats buy-and-hold OOS in sector-ETF and regime-switch studies (MDPI, jump-model), but the 2026 practitioner consensus (UBP, AQR, Morningstar) and the AlgoXpert plateau finding lean toward running low-correlation strategies in PARALLEL (ensemble/blend) rather than rotating winner-take-all -- because rotation pays turnover, regime timing lags (~15 days), and the IS-best is rarely the OOS-best. The project's own selector already hedges this with anti-churn hysteresis (retain incumbent unless DSR improves by >= 0.01).

### Pitfalls (from literature) -> applied to this step
1. **Don't fix the symptom (class_weight) without fixing the cause (barriers).** Sefidian: low-vol periods produce mostly-neutral labels under FIXED thresholds. pyfinagent's alt strategies use fixed double-threshold gates (momentum_6m>5 AND quality>0.3, etc.). Even with SMOTE/class_weight, a label set that is 95% neutral carries almost no +1/-1 signal -- you'd be oversampling a tiny, noisy minority. **Preferred first move: make ONE alt strategy's barriers/thresholds vol-scaled or looser so the +1/-1 share is materially >0, THEN (if needed) balance.**
2. **Illusory diversity inflates K and PBO.** 2 of 4 seeds are triple_barrier variants (one with INERT overrides per `rotation_runner.py:21-29`). Counting them as distinct trials over-deflates DSR (safe) but gives FALSE confidence the bake-off explored 4 independent strategies. Reseed to genuinely orthogonal types (TB vs MR vs QM vs a true factor/low-vol sleeve) before trusting the verdict.
3. **Rotation overfits to recent winners.** PBO: IS-best underperforms median OOS. A naive "promote whoever earned most last week" loop is exactly this trap. The DSR+PBO gate + hysteresis is the correct guard -- do NOT bypass it with a raw-return ranking.
4. **Turnover eats the rotation edge.** Jump-model: the edge survives only with a switch penalty (44% turnover). The selector's `min_improvement=0.01` hysteresis is the analog; keep it. A winner-take-all flip every week without a penalty would whipsaw.
5. **Regime/selection latency.** Jump-model ~15-day regime lag misses rebounds. A weekly bake-off that switches strategies will always be acting on stale evidence; size the switch conservatively and never go to cash on a weak week (`strategy_selector.py:77` already does this -- RETAIN incumbent, never cash).

## Q2 MONEY VALUE (strategic fork) -- is fixing rotation worth it?

### Q2.0 -- Bottom line recommendation: **REDIRECT. Ensemble-within-the-working-engine > winner-take-all rotation between broken alt strategies.**

The evidence says the highest-EV money path is **NOT** "fix alt-strategy labeling so the
48.x winner-take-all rotation can drive live selection." It is a two-part redirect:

**(1) PRIMARY (highest money, lowest risk): amplify the WORKING momentum engine via
the cheap levers that already touch live orders** -- sector diversification
(phase-51.2) now amplified by the live EU/KR universe, and the alpha overlays. These
operate INSIDE `screener.rank_candidates(strategy="momentum")` (the live path), so
they affect live money TODAY with zero architectural bridge. The 2026 consensus
(low-correlation diversification "more durable"; UBP/AQR) supports broadening the
single working engine's breadth before adding a second strategy TYPE.

**(2) SECONDARY (if a second strategy type is pursued at all): blend/ENSEMBLE, not
winner-take-all rotation.** The practitioner consensus (UBP, AQR, Morningstar 2026)
and the AlgoXpert plateau finding favor running low-correlation strategies in
PARALLEL over rotating to a single winner. Rotation's OOS edge is real but modest
(single-digit annualized) and turnover-fragile (jump-model). Winner-take-all also
maximally exposes you to the PBO trap (IS-best underperforms median OOS).

**Why NOT "fix labeling -> winner-take-all rotation" as the primary money lever:**
- **RC-B is the killer:** even a perfect rotation verdict changes only a BQ heartbeat
  row today (`autonomous_loop.py:1128`); the live engine ignores `best_params["strategy"]`
  and is hardwired to `screener` "momentum" (`:621` passes no `strategy=`). The bridge
  from a backtest-strategy verdict to live orders does NOT exist and is non-trivial
  (the screener taxonomy and the backtest taxonomy are different code -- there is no
  `screener` branch for triple_barrier/mean_reversion/factor_model). Building that
  bridge is multiple cycles BEFORE rotation can earn a dollar.
- **The alt strategies are not just silent, they LOSE** (quant_results.tsv: mean_reversion
  Sharpe -6.13, factor_model -1.21, quality_momentum -0.59; NONE beat the 1.1705
  baseline, NONE cleared DSR). Fixing their labels to TRADE is necessary but there is
  NO evidence they'd trade PROFITABLY -- you'd be investing cycles to make a
  money-LOSING strategy trade more.
- **Genuine uncorrelated alpha is unproven here.** The diversification thesis requires
  the alt strategies to be uncorrelated AND positive-alpha. The seeds are 50% TB
  variants (correlated), and the non-TB seeds lose money in-sample. K_eff << 4.
- **The selector ITSELF is fine** -- it's well-designed and the research VALIDATES its
  DSR+PBO+hysteresis design. The problem is upstream (no tradeable diverse strategies)
  and downstream (no live bridge), not the selection logic.

### Q2.1 -- The minimal first GENERATE step to make ONE alt strategy actually trade
(if the operator chooses to pursue the rotation path despite the redirect)

**Make `mean_reversion` (or `factor_model`) emit a balanced label set by replacing
its fixed double-threshold gate with vol-scaled / looser thresholds, and verify the
+1/-1 share rises from ~0 to a tradeable fraction.** Concretely, the smallest change
with the highest signal:
- Target `_compute_factor_label` (`backtest_engine.py:1114`) -- it's the cleanest
  (single composite score, no 2-stage forward-validation like MR). Widen the
  trade band: change `composite > 0.6 -> +1` / `< 0.3 -> -1` to a PERCENTILE/vol-scaled
  cut (e.g. top-tercile -> +1, bottom-tercile -> -1 across the candidate cross-section
  at each date), so by construction ~1/3 of samples are +1 and ~1/3 are -1 (balanced).
  This directly attacks neutral-dominance at the cause (Sefidian: vol/dynamic barriers).
- **$0 verification** ($0 LLM -- pure quant backtest): run ONE walk-forward backtest
  with `strategy=factor_model` and assert (a) `num_trades > 0` in the window logs
  (`backtest_engine.py:493` BUY count > 0), and (b) the training label distribution is
  no longer >90% neutral (add a one-line log of `np.bincount(labels+1)` in
  `_build_training_data` :640). Compare Sharpe to the prior factor_model crash/0.0000
  rows. **Do NOT promote to live** -- this is a backtest-only proof that the label fix
  makes it trade. Acceptance = "factor_model now trades >0 and its label set is
  balanced," NOT "factor_model beats baseline" (that's a later bar).
- Secondary option if MR is preferred: in `_compute_mean_reversion_label` (`:1051`),
  loosen Stage-1 (`sma_dist < -0.05 AND rsi < 35`) to OR / vol-scaled bands so more
  samples enter Stage-2. Riskier (2-stage) -- factor_model is the cleaner first cut.

**Why this is the right minimal step:** it's $0 (quant-only), it isolates RC-A from
RC-B (proves the label fix WITHOUT touching the live bridge), it attacks the cause
(barrier/threshold construction) not the symptom (class_weight), and it produces a
falsifiable artifact (trade count + label histogram) the Q/A gate can verify
deterministically.

### Q2.2 -- The DSR-selector design (already built; what to ADD)

The selector (`strategy_selector.py::select_best_strategy`) is **already correctly
designed and research-validated**. Design (CONFIRMED against the literature):
- **Gate:** DSR >= 0.95 AND PBO <= 0.20 via `gate.PromotionGate` -- correct (Bailey DSR
  deflates the IS-peak for N trials; PBO rejects configs that underperform the median
  OOS).
- **Rank:** passers DSR-desc, then PBO-asc -- correct.
- **Anti-churn hysteresis:** switch only if `best.dsr - incumbent.dsr >= 0.01` -- correct
  and VALIDATED by the jump-model turnover finding (switch penalty is essential).
- **Never-to-cash:** no passer -> RETAIN incumbent -- correct (avoids latency whipsaw).

**What to ADD (research-backed enhancements, in priority order):**
1. **Deflate by K_eff, not raw N.** Currently `num_trials=5` (raw seed count). With 2
   correlated TB seeds, K_eff < 5. Compute K_eff from the seeds' OOS return
   correlation matrix (VertoxQuant / Bailey) and pass THAT as `num_trials`. (Currently
   over-deflates = SAFE, but K_eff is the principled value and avoids false "tested 5
   independent strategies" confidence.) NOTE: deflating by a SMALLER K_eff makes the
   gate EASIER -- so this must be paired with #2 (reseed) or it weakens the guard.
2. **Reseed to genuinely orthogonal strategies** before trusting any verdict: TB
   (incumbent) + a TRUE mean-reversion + a TRUE quality-momentum + a low-vol/factor
   sleeve -- 4 distinct TYPES, not 2 TB variants. (Required for the diversification
   thesis to hold; current seeds are 50% redundant.)
3. **Plateau over peak (AlgoXpert).** Optionally prefer the seed whose DSR is robust
   across a stability region (params within 90% of best) rather than the raw DSR-max
   -- guards the residual PBO risk the gate doesn't catch.
4. **(Deferred but required for live effect) The RC-B bridge.** A `screener`-side
   strategy router so the selected strategy actually changes live ranking. Until this
   exists, the selector's verdict is audit-only (`rotation_runner.py:220`
   allocation_pct=0). This is the single biggest gap between "rotation works in
   backtest" and "rotation earns a live dollar."

### Q2.3 -- Redirect recommendation (the evidence-based fork)

**RECOMMEND: deprioritize winner-take-all rotation as a near-term money lever.
Reorder the roadmap:**
1. **NOW (highest money/effort ratio):** sector diversification (phase-51.2) +
   alpha-overlay breadth INSIDE the working momentum engine. Touches live orders
   today; 2026 consensus supports breadth; $0-to-cheap; cannot regress the working US
   path (additive scoring overlays).
2. **NEXT (if a 2nd strategy type is wanted):** build it as an ENSEMBLE/BLEND sleeve
   that runs in PARALLEL with momentum (allocate a small capital slice), NOT a
   winner-take-all switch. This captures the low-correlation diversification benefit
   the literature actually supports, with bounded downside.
3. **ONLY IF (1) and (2) are exhausted:** invest the multi-cycle effort to (a) fix
   alt-strategy labeling so >=1 alt strategy trades profitably in backtest [Q2.1], (b)
   reseed to orthogonal types + K_eff deflation [Q2.2], AND (c) build the RC-B live
   bridge. Rotation cannot earn a dollar until all three land.

**One-line verdict:** the selector is good, the rotation plumbing is built, but the
strategies it would rotate between don't trade, half are redundant, and even a
perfect verdict can't reach live orders today -- so the cited-evidence money path is
to broaden the ONE engine that works (sector diversification / ensemble) before
resurrecting winner-take-all rotation.

---
