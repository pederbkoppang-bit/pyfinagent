# Research Brief 55.3 — Synthesis + operator checkpoint (finetune-vs-features + MinTRL + burn)

**Step:** 55.3 (phase-55, review-only, $0). **Tier:** complex. **Date:** 2026-06-10.
**Role of this brief:** it IS the evidence base for the 55.3 strategic chapter, which must
itself pass the research gate (>=5 sources read in full + recency scan). Internal findings
(55.1 B1-B15, 55.2 F-A1..F-I) are the substrate; external literature provides the
finetune-vs-features decision support, the MinTRL framing, the cost-inclusive-evaluation
prognosis, and the turnover-lever evidence. $0; review-only; the chapter's recommendation
must be grounded in findings + literature, not vibes. **53.1's Ledoit-Wolf REJECT is binding
precedent — a naive re-proposal of the no-trade band is an automatic FAIL.**

---

## Internal evidence base (file:line anchored; the findings the chapter synthesizes)

### A. Burn-estimate inputs (EMPIRICAL — BQ-measured 2026-06-10, the decision variable)

The operator-checkpoint block needs $/cycle for lite vs full. The UI claim and the metered
reality DIVERGE — this is itself a finding the chapter should state.

| Source | Claim | Verdict |
|---|---|---|
| `frontend/src/app/paper-trading/manage/page.tsx:228` | lite = "~$0.01/ticker" | **CONFIRMED.** BQ `analysis_results` lite rows: n=120, avg **$0.0094**/ticker (min 0.005, max 0.017). |
| `frontend/src/app/paper-trading/manage/page.tsx:230` | full = "$0.50-2.00/ticker" | **OVERSTATED ~2-10x.** Per-`analysis_results`-row full cost is flat **$0.10** (n=116). Effective full-CYCLE cost (multi-agent calls beyond the single row) measured at day level: 05-22 $4.06/15t=**$0.27/t**, 05-27 $3.00/14t=$0.21/t, 05-26 $1.30/7t=$0.19/t -> full ~**$0.19-0.27/ticker effective**, NOT $0.50-2.00. |
| `backend/config/settings.py:316` (comment) | "Gemini Flash + AI Studio direct keys keep full-cycle costs in the **$1-3 range**" | **CONFIRMED** as the day-level full-cycle figure (and the authoritative internal number; the UI label is stale). |
| 55.2 §2 N* reconciliation | away week metered ~**$1** ($0.40 Gemini + $0.59 lite self-reported) vs churn **-$132** | **CONFIRMED.** Compute burn was ~0.8% of the churn loss; N* drag was Risk/churn, not Burn. |

**Cost-per-cycle table for the operator block (BQ-measured day totals):**

| Mode | Tickers/cycle (observed) | $/cycle (observed range) | $/ticker effective | Source |
|---|---|---|---|---|
| Lite (ran away week) | 5-11 | **$0.05-0.17** | ~$0.01 | BQ `analysis_results` 06-01..06-09 |
| Full (debate/risk-judge/bias-audit) | 7-15 | **$1.08-4.06** | ~$0.19-0.27 | BQ `analysis_results` 05-16/22/26/27 |

For a **1-2 week live window** (the checkpoint horizon): at ~5 trading days/week and one
cycle/day -> **lite: ~$0.25-1.70/week** (~$0.50-3.40 for two weeks); **full: ~$5.40-20/week**
(~$11-40 for two weeks). Both are far inside the `cost_budget_daily_usd=$25` / `$300/mo` caps
(`settings.py:317-318`). The flat-fee Claude-Code rail (55.2 F-A1) carries the Trader/RiskJudge
LLM legs at $0 marginal when its OAuth session is healthy; the metered figures above are the
Gemini-analysis + direct-Anthropic-probe spend only.

### B. The 53.1 BINDING PRECEDENT (Ledoit-Wolf gate + the rejected mechanism)

- **Gate mechanics:** `backend/backtest/analytics.py:239 sharpe_diff_test()` — Ledoit-Wolf
  (2008) Sharpe-DIFFERENCE test via a stationary (Politis-Romano 1994) bootstrap on PAIRED
  per-period returns; one-sided H0: SR(a)<=SR(b); fat-tail + autocorrelation robust; seeded
  deterministic. Returns `{delta, p_one_sided, ci_low, ci_high, se, n}`. Hard floor `n<10 ->
  null result`.
- **The a-priori promote rule (must be reused verbatim by the LEVER candidate):**
  `p_one_sided<0.05 AND delta>=+0.05 AND ci_low>0` (the NET/promote leg) PLUS a do-no-harm
  leg `gross ci_low > -0.05`. Source: `scripts/ablation/no_trade_band_replay.py:133-147` +
  `live_check_53.1.md`.
- **The REJECTED mechanism (`backend/backtest/rebalance_band.py:22 apply_no_trade_band`):**
  a RANK-hysteresis band — RETAIN a held name unless its rank drops below `top_n*(1+band_pct)`;
  ADD a new name only when it clears entry rank `top_n`. **This is the discrete-portfolio
  hysteresis band.** Measured (48 monthly rebalances, S&P-500, 2022-2025, top_n=10): turnover
  0.555->0.489 (-12%), net Sharpe +0.015, maxDD unchanged. **REJECTED:** NET leg dSharpe
  +0.015 < +0.05 floor, p=0.376, CI90=[-0.066,+0.092] (CI_low<0); GROSS do-no-harm also
  FALSE (CI_low -0.071). Honest negative.
- **CRITICAL IMPLICATION for the LEVER spec:** the "score-hysteresis/persistence" candidate is
  the SAME mechanism family as the already-rejected band (rank-hysteresis ≈ score-hysteresis).
  The chapter MUST either (a) argue why a *score-threshold two-band* (act only when |Δscore|
  exceeds a width, vs the rank-rebalance band) is a materially different mechanism on THIS
  book, or (b) honestly conclude the hysteresis family is unpromising here and pick a different
  single lever (min-holding-period or turnover-budget operate on a DIFFERENT axis — time-in-
  position / trade-count cap rather than name-set stability). See External Q6 evidence.

### C. MinTRL / statistical-power inputs (55.1 §7 + analytics.py)

- **55.1 §7 measured:** away-week dailies SR_daily=-0.081 (ann ≈ -1.29), skew -1.05, kurt 3.43
  -> **MinTRL(SR*=0, 95%) ≈ 377 daily observations; we have 7** (35 since inception). DSR=0.0
  after deflating for 5 tested strategies. The displayed Sharpe 4.72 / PSR 0.9993 are NOT
  performance evidence on this window.
- **Code:** `backend/services/paper_metrics_v2.py:33 MIN_OBS_FOR_PSR=30` (PSR/DSR unstable
  below this; Bailey-LdP 2012 footnote). `perf_metrics.py:148` notes n=30 has SE≈0.3 (edge of
  detectability). `analytics.py:292 compute_deflated_sharpe(observed_sr, num_trials, var,
  skew, kurt, T)` is the on-hand DSR primitive — MinTRL is the INVERSE (solve for T given a
  target confidence). No standalone MinTRL function exists; the chapter computes it from the
  Bailey-LdP closed form (External Q4).
- **MinTRL horizon MENU the chapter must state** (compute from Bailey-LdP MinTRL with observed
  skew/kurt, target SR*=0 at 95%): at the OBSERVED away-week SR (~-1.29 ann) MinTRL is
  ill-posed/huge (negative Sharpe can't beat zero) -> use SR_ann assumptions for the menu:
  the backtest Sharpe **1.17** (optimizer_best.json), and intermediate **0.5 / 1.0**. (Numbers
  computed in External Q4 below.)

### D. The finetune-vs-features finding map (55.1/55.2 -> direction)

| Finding (ID) | Shape | Direction it argues for |
|---|---|---|
| B15 / F-I churn: 81.4% weekly turnover, 10 within-week RTs net **-$132**, 35% daily action-flip | **lever-shaped** (a turnover/stability control) | FINETUNE (a single churn lever) — but see 53.1 precedent: the obvious lever (rank/score hysteresis) already failed the gate |
| F-A1 claude-CLI OAuth rail down all week (SignalStack/Trader/RiskJudge LLM legs degraded) | **feature/reliability-shaped** | FEATURES (detect+alert+degraded-mode policy) |
| F-D silent 0.0/10 degraded scoring published | feature/reliability-shaped | FEATURES (degraded-scoring guard / sentinel) |
| F-E llm_call_log blind to the analysis rail (can't answer "which agent fired") | feature/observability-shaped | FEATURES (instrument the rail) |
| F-F RiskJudge REJECT advisory-only (DELL bought at REJECT) | **feature-shaped** (a binding gate) | FEATURES (make REJECT binding = a concentration/risk capability) |
| F-G RiskJudge concentration-blind, 10%-vs-30% prompt divergence | feature-shaped | FEATURES (inject portfolio context) |
| B11 sector-NAV% cap can't constrain intra-book concentration under high cash (book 100% Tech, HHI->0.63) | **feature-shaped** (a working concentration limit) | FEATURES (count-cap or HHI cap that binds) |
| B10 daily-loss kill-switch leg structurally ≈0 under once-daily cadence | feature/risk-shaped | FEATURES (fix the SOD anchor) |
| SignalStack conviction overlay = static fallback all week (the one momentum-damping layer was dead) | feature/reliability-shaped | FEATURES (restore the overlay's live leg) |

**Pre-synthesis read of the map:** the lever-shaped problem (churn) has ALREADY been probed by
53.1 and the obvious hysteresis lever failed the gate; the dominant finding cluster is
feature/reliability/observability-shaped (rail down, REJECT non-binding, scoring silent,
overlay dead, concentration uncapped). This biases the *prior* toward FEATURES — but the
chapter must test that prior against the literature (Q5: does adding capability beat tuning in
agent-trading systems?) and against the do-no-harm constraint (the +20% momentum core must not
regress). Both candidate specs are still required.

### E. Baseline-comparison inputs (passive B&H + US-momentum-core)

- **US-momentum-core baseline:** `optimizer_best.json` — strategy=triple_barrier, Sharpe
  **1.1705**, DSR **0.9526** (backtest 2018-2025, walk-forward). Memory: +20% NAV measured
  2026-05-29, trailing stops + sector caps "functioning" — now NUANCED by 55.2 F-F/F-G (REJECT
  non-binding, concentration-blind) and 55.1 B11 (cap doesn't bind under cash).
- **Passive B&H (away week, 55.1 §7):** SPY +2.49% (the `/performance benchmark_return_pct`),
  vs fund +19.19% pnl since inception (mostly April book) and **away-week NAV -2.26%**. On the
  away week alone the fund UNDERPERFORMED passive SPY (regime: 06-05 semis selloff SOXX -10.4%).
- **Attribution (55.1 §7, n=7 LOW POWER):** β_SPY=1.04 (R²=0.63), β_SOXX=0.19, residual alpha
  +0.28%/day but **statistically meaningless at n=7** — the away-week path is regime (semis
  selloff) + concentration tilt + churn drag; no skill-alpha evidence at this horizon.

---

## External research (in progress below)
</content>
</invoke>
