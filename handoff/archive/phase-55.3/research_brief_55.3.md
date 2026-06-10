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

## External research

### Search-query log (3-variant discipline per .claude/rules/research-gate.md)

| Topic | Frontier (2026) | Recency (2025) | Year-less canonical |
|---|---|---|---|
| Cost-inclusive LLM-trading eval | "LLM trading agent evaluation cost-inclusive longer window backtest 2026" | "LLM trading agent overtrading churn buy-and-hold underperform 2026" | (anchored arXiv ids 2505.07078 / 2510.02209 / 2602.14233) |
| Turnover/no-trade levers | (within cost-inclusive) | "momentum strategy turnover net of costs robust transaction costs 2025" | "turnover regularization momentum strategy transaction costs no-trade region" |
| Min-holding / hysteresis | — | — | "minimum holding period constraint portfolio momentum turnover reduction performance" |
| Capability-vs-tuning ROI | "when adding capabilities beats parameter tuning agent system ablation 2025" | (KTD-Fin 2605.28359 recency) | — |
| MinTRL / DSR | — | — | "deflated Sharpe ratio minimum track record length Bailey Lopez de Prado" |

### Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2505.07078v5 (FINSABER) | 2026-06-10 | paper (peer-adjacent preprint) | WebFetch (full HTML) | LLM 6-month wins "vanish under broader and longer evaluations"; over 2004-2024 Buy-and-Hold "significantly outperforms both LLM strategies across all robust setups". FinMem overtrades (commission 5-9x FinAgent), **negative alpha in all scenarios**. Momentum selection: B&H SPR 0.384 vs FinAgent 0.104. Costs $0.0049/share min $0.99/order. |
| 2 | https://arxiv.org/html/2510.02209v1 (StockBench) | 2026-06-10 | paper (preprint, OpenReview) | WebFetch (full HTML) | Contamination-free, 82 trading days (Mar-Jun 2025). "Most models struggle to outperform a simple buy-and-hold". Best (Kimi-K2) +1.9% vs B&H +0.4%; "all LLM agents underperform the passive baseline during the downturn". **Caveat: does NOT model trading costs/slippage** -> an OPTIMISTIC bound, and they still mostly lose net-zero-cost. |
| 3 | https://arxiv.org/html/2602.14233 | 2026-06-10 | paper (preprint) | WebFetch (full HTML) | Five-bias checklist (look-ahead, survivorship, **narrative**, **objective/over-confidence**, **cost**) + a binary structural-validity GATE: "if any requirement fails, the reported result can only support a stress test interpretation or a proof of concept, and it cannot support a claim of deployable alpha." Five components: Temporal Sanitation, Dynamic Universe, **Rationale Robustness**, **Epistemic Calibration**, Realistic Implementation. Does NOT cover MinTRL/multiple-testing (complement, not substitute, for Bailey-LdP). |
| 4 | https://arxiv.org/html/2605.28359v1 (KTD-Fin) `[ADVERSARIAL to FEATURES]` | 2026-06-10 | paper (preprint, Jun 2026) | WebFetch (full HTML) | **Richer capabilities do NOT reliably improve returns**: open-research (tools) mode returned -2.11% vs memory-only -0.16% for the anchor model — *worse* with more tools. 9/10 frontier LLMs **negative stock-selection alpha**; headline returns "largely explained by passive exposure to market and style factors". LLMs **36-144x slower** than ML baselines with worse alpha. Masking proves "skill" is substantially memorized pretraining, not reasoning. |
| 5 | https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio (canonical Bailey-LdP formulas) | 2026-06-10 | reference (formula source) | WebFetch (full) | Exact MinTRL: `MinTRL = 1 + (1 - g3*SR0 + ((g4-1)/4)*SR0^2) * (Phi^-1(conf) / (SRhat - SR0))^2`, SR per-period (non-annualized), g3=skew, g4=raw kurtosis. PSR = Phi((SRhat-SR0)*sqrt(T-1)/sqrt(1-g3*SR0+((g4-1)/4)*SR0^2)). |
| 6 | https://arxiv.org/html/2509.04541 (Finance-Grounded Optimization) | 2026-06-10 | paper (preprint) | WebFetch (full HTML) | **Band turnover regularization** "maintains strong Sharpe ratios while avoiding the over-suppression of trading dynamics often induced by linear turnover penalties" — objective-level integration beats post-hoc filtering (they "do not apply any post-hoc smoothing, clipping, or turnover-reduction techniques after position generation"). Band turnover 0.17-0.24 vs unregularized 0.22+ at Sharpe >1.2. |
| 7 | https://blog.thinknewfound.com/2018/07/momentums-magic-number/ (ThinkNewfound) | 2026-06-10 | industry blog (named) | WebFetch (full) | Momentum returns "peak when the sum of the formation and holding period is between 14-18 months"; longer holding -> "considerably less trading" and analysis with cost estimates "may exhibit even greater peakedness" — i.e. LONGER HOLDING HELPS NET OF COSTS. 12-month formation -> only 2-6 month hold optimal. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2603.27539 (LLM financial-MAS eval taxonomy) | paper | Already read in full by 55.2 (Coordination Primacy + $0.50-2.00/decision cost-per-decision benchmark) — re-cited, not re-read per the prompt. |
| https://arxiv.org/html/2511.03628 (LiveTradeBench) | paper | Snippet corroborates "real-world alpha with LLMs" theme; pre-anchored trio + KTD-Fin already cover the cost-inclusive thesis with stronger numbers. |
| https://www.kellogg.northwestern.edu/.../korajczyk%20sadka.jf2004.pdf (Are Momentum Profits Robust to Trading Costs?) | peer-reviewed (J. Finance) | Canonical prior art: small-cap momentum trading costs "offset gross profits", large-cap retains some net — surfaced via year-less search; the 2025/2509 sources update it. |
| https://www.sghiscock.com.au/.../SGH_EAM-Investors-Momentum-and-Trading-Costs.pdf | industry | Momentum turnover 150-200%/yr; ~10% of gross eaten by costs over 30y — corroborates the turnover-cost magnitude. |
| https://blogs.cfainstitute.org/investor/2025/12/17/momentum-investing-...resilient-framework/ | industry (CFA, Dec 2025) | Recency corroboration that momentum core remains robust long-run — supports do-no-harm. |
| https://aws.amazon.com/blogs/.../advanced-fine-tuning-techniques-for-multi-agent-orchestration... | vendor blog | Capability-vs-tuning context for agent systems; KTD-Fin is the stronger in-domain source. |
| https://arxiv.org/html/2605.19337v1 (Agentic Trading survey) | paper | 77-study evidence map screened to 2026-03; corroborates "balance window length vs cost vs contamination". |
| https://arxiv.org/pdf/2507.08584 (To Trade or Not to Trade) | paper | Agentic market-risk estimate improves trade decisions — adjacent to the RiskJudge-binding feature direction. |
| https://www.researchaffiliates.com/.../266_the_impact_of_constraints... | industry | Constraint impact on min-var portfolios — turnover constraint "highly effective... modest effect on volatilities". |
| https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf (Boyd multi-period) | peer-reviewed | Canonical no-trade-region / multi-period convex optimization prior art (year-less). |

### Recency scan (2024-2026) — MANDATORY

Searched explicitly for 2024-2026 literature on (a) cost-inclusive LLM-trading evaluation and
(b) momentum turnover/min-holding levers. **Result: found multiple NEW findings that
COMPLEMENT and STRENGTHEN the canonical sources, plus one that is directly adversarial to a
naive FEATURES recommendation.**

1. **KTD-Fin (arXiv:2605.28359, Jun 2026)** — the strongest NEW finding: adding agent
   capabilities/tools did NOT improve returns (open-research mode WORSE than memory-only;
   9/10 negative selection alpha; gains are passive factor exposure). Supersedes the optimistic
   "more agents = better" prior that a naive FEATURES reading would assume. `[ADVERSARIAL]`.
2. **StockBench (arXiv:2510.02209, Oct 2025)** — contamination-free, recent-window confirmation
   that LLM agents mostly lose to passive B&H even WITHOUT modeling costs.
3. **Finance-Grounded Optimization (arXiv:2509.04541, Sep 2025)** — NEW: *band* turnover
   regularization beats naive linear penalties (which over-suppress), and objective-level
   integration beats post-hoc filtering — directly informs why the 53.1 post-hoc band was weak.
4. **2505.07078 v5 (FINSABER, May 2025, v5 update)** — explicit statement of the missing
   guardrail: "without system-level guardrails like **minimum holding periods or cooldown
   windows**, the LLM's linguistic uncertainty is directly executed as trades, magnifying churn."
5. **CFA Institute (Dec 2025)** — momentum remains a resilient long-run framework — supports
   the do-no-harm posture toward the +20% momentum core.

No 2024-2026 source was found that REVERSES the cost-inclusive prognosis (LLM short-window wins
vanish net of costs) or that endorses naive overtrading. The canonical Bailey-LdP MinTRL/DSR
(2012/2014) remains the authoritative statistical-power framework; no newer method supersedes it
for this use case (the Opdyke difference-of-Sharpes MinTRL is an extension, already mirrored by
the repo's `sharpe_diff_test`).

---

## Key findings (external, per-claim cited)

**KF1 — Short-window LLM-trading wins vanish under longer, cost-inclusive evaluation.**
"extending the evaluation horizon significantly diminishes the perceived superiority of LLM
investors" and "the market baseline significantly outperforms both LLM strategies across all
robust setups" (FINSABER, arXiv:2505.07078v5). Corroborated by StockBench: "most models
struggle to outperform a simple buy-and-hold baseline" even without modeling costs
(arXiv:2510.02209). **Application:** pyfinagent's away-week fund UNDERperformed passive SPY
(-2.26% vs +2.49%, 55.1 §7) — fully consistent. The +19% inception level is the April momentum
book (regime + earlier selection), not away-week LLM skill.

**KF2 — Overtrading is the dominant value-destroyer for LLM agents, and the named fix is a
system-level guardrail (minimum holding / cooldown), NOT richer reasoning.** FinMem's
"commission ratio is five to nine times higher" and it shows "negative alpha in all scenarios"
(arXiv:2505.07078v5); "without system-level guardrails like minimum holding periods or cooldown
windows, the LLM's linguistic uncertainty is directly executed as trades, magnifying churn"
(ibid., v5 behavioral analysis). **Application:** this is pyfinagent's exact mechanism — 35%
daily action-flip rate with SignalStack (the damping overlay) dead and RiskJudge REJECT
non-binding (55.2 F-I/F-F), driving 81.4% weekly turnover and -$132 churn (55.1 B15).

**KF3 — Adding capabilities/tools does NOT reliably improve net returns in agent-trading
systems `[ADVERSARIAL]`.** "granting the agent tool access (open-research) consistently lifts
capability above the no-tool baseline... yet returns remain negative across most agents";
open-research -2.11% vs memory-only -0.16% (KTD-Fin, arXiv:2605.28359). 9/10 frontier LLMs had
negative selection alpha; headline returns "largely explained by passive exposure to market and
style factors." **Application:** this is the single most important counterweight to a naive
"add FEATURES (full-mode agents)" recommendation. It does NOT say features are useless — it says
RELIABILITY/CORRECTNESS features (a binding risk gate, a degraded-mode guard, instrumentation)
are higher-EV than ANALYTICAL features (more agents, deeper debate) that chase selection alpha
the literature says is mostly illusory at this horizon.

**KF4 — Turnover control is most robust when (a) band-shaped (two-threshold) rather than a
single linear penalty, (b) integrated into the objective rather than a post-hoc filter, and
(c) realized as a minimum-holding/cooldown time constraint for high-churn signal-followers.**
"band turnover regularization maintains strong Sharpe ratios while avoiding the over-suppression
of trading dynamics often induced by linear turnover penalties" (arXiv:2509.04541); momentum
returns peak with LONGER combined formation+holding (14-18 months) and longer holds exhibit
"considerably less trading" with cost-inclusive analysis showing "even greater peakedness"
(ThinkNewfound 2018). **Application:** distinguishes the candidate levers — see the
finetune-vs-features matrix and the 53.1-differentiation analysis below.

**KF5 — A structural-validity gate (not just a Sharpe number) is the right acceptance test for
"deployable alpha"; short windows can only ever support a stress-test/proof-of-concept claim.**
"if any requirement fails, the reported result can only support a stress test interpretation or
a proof of concept, and it cannot support a claim of deployable alpha" (arXiv:2602.14233). Its
five components map 1:1 onto pyfinagent's findings: Rationale Robustness (rationale-not-consumed,
55.2 §3), Epistemic Calibration (the 0.0/10 confident-failure, F-D), Realistic Implementation
(churn/cost, B15), Temporal Sanitation (the vacuous look-ahead cleanliness, 55.2 §4), Dynamic
Universe (n/a for paper). **Application:** the chapter should frame the operator window as a
stress-test/sanity gate, NOT a deployable-alpha proof — which is exactly what the MinTRL math
forces (KF6).

**KF6 — MinTRL: a 1-2 week live window CANNOT establish skill; multi-year horizons are needed
even at the optimistic backtest Sharpe.** Using the exact Bailey-LdP MinTRL (Wikipedia/DSR;
SR per-period, g3=skew -1.05, g4=raw kurt 3.43, SR*=0, 95% one-sided, 252/yr — the away-week
moments from 55.1 §7):

| Assumed annualized SR | MinTRL (daily obs) | ≈ calendar | Interpretation |
|---|---|---|---|
| **1.17** (backtest, optimizer_best.json) | **~539** | **~2.1 years (≈26 trading-months)** | Even the best-case backtest Sharpe needs ~2y of live dailies for 95% significance. |
| 1.00 | ~730 | ~2.9 years | |
| 0.50 (haircut for live decay) | ~2,820 | ~11 years | A realistic post-haircut Sharpe is effectively unprovable on any practical live horizon. |
| Observed away-week (SR_daily -0.081, ann ≈ -1.29) | ill-posed (negative SR cannot beat 0); 55.1 reported **~377 dailies** at \|SR\| | n/a / ~1.5y | 55.1's 377-daily figure uses the away-week \|SR\| magnitude; our recompute at the same magnitude gives ~450 dailies (convention nuance — same qualitative verdict). |

**The operator takeaway:** a 1-2 week (≈5-10 daily obs) window is **2 orders of magnitude short**
of MinTRL at the backtest Sharpe. The live window's PURPOSE is therefore a **sanity/stress gate**
(does the fixed P&L readout look sane, does the kill switch arm, are there catastrophic
regressions, do non-HOLD trades fire) — NOT a statistical proof of skill. This is the honest
horizon framing for the spend decision and directly seeds the 58.1 "minimum live-window length"
requirement.

---

## Finetune-vs-features evidence matrix (the chapter's spine)

| Direction | What the 55.1/55.2 findings say | Literature support | Verdict |
|---|---|---|---|
| **FINETUNE a churn LEVER** (the lever-shaped problem: B15/F-I churn) | Churn -$132, 81.4% turnover, 35% action-flips is real and lever-shaped. BUT 53.1 already measured the obvious rank-hysteresis band and it FAILED the Ledoit-Wolf gate (dSharpe +0.015<0.05). | KF2 names min-holding/cooldown as the fix; KF4 says band-shaped + time-based + objective-integrated turnover control is most robust; KF7-ThinkNewfound says longer holds help net of costs. The 53.1 band was post-hoc + rank-based (the weak form per KF4). | **PROMISING IF the lever is a DIFFERENT mechanism than the rejected band** (min-holding-period or turnover-budget operate on the time/trade-count axis, not name-set rank). Must clear the SAME gate. |
| **FEATURES: analytical (full-mode agents, deeper debate)** | Lite ran all week; full-mode is the cost decision variable. | KF3 `[ADVERSARIAL]`: adding tools/agents did NOT improve returns (open-research WORSE); selection alpha mostly illusory at this horizon. KF1: longer cost-inclusive eval erases LLM edge. | **LOW-EV** as a profit lever. Full-mode buys richer rationale/reports (operator trust, audit) but the literature does not support it as an alpha source. Worth it for explainability, not for returns. |
| **FEATURES: reliability/correctness** (binding RiskJudge REJECT, degraded-scoring guard, restore SignalStack overlay, instrument the rail, concentration cap that binds, kill-switch SOD fix) | The DOMINANT finding cluster (F-A1 rail down, F-D silent 0.0/10, F-E blind metering, F-F REJECT advisory, F-G concentration-blind, B10 kill-switch ≈0, B11 cap doesn't bind). | KF3: reliability features > analytical features when selection alpha is illusory. KF5 structural-validity gate maps 1:1 to these gaps (Rationale Robustness, Epistemic Calibration, Realistic Implementation). KF2: the missing damping guardrail IS a reliability feature. | **HIGHEST-EV feature direction.** These convert the system from "rationale-quality decent but architecture doesn't consume it" (55.2 §3 verdict) into one whose risk verdicts and scores actually drive behavior. Most are already scoped for phase-56 (F-A1/F-D/F-E/F-F/F-G). |

**Synthesis read (to be argued in the chapter, not pre-decided here):** the evidence points to a
**reliability-features-first** posture with a **single time-axis churn lever** as the phase-57
LEVER candidate. The away week's loss was Risk/churn, not Burn (55.2) and not a failure of
analytical depth — it was a failure of the decision ARCHITECTURE to consume its own (decent)
rationale (the damping overlay was dead, REJECT didn't bind, scoring failed silently). The
literature (KF3) actively warns against treating "more agents/tools" as the profit fix. The
churn lever is worth a measured shot ONLY if it is mechanistically distinct from the 53.1-rejected
band (KF2/KF4 say min-holding/turnover-budget are — they cap TIME/TRADE-COUNT, not name-set rank).

---

## The 53.1-differentiation analysis (binding-precedent compliance)

53.1 REJECTED a **rank-hysteresis no-trade band** (`apply_no_trade_band`: retain held name unless
rank drops below top_n*(1+band_pct)). The phase-57 LEVER MUST be a different mechanism or honestly
concede the family is unpromising. Per-candidate:

- **Score-hysteresis / persistence** — **SAME FAMILY as the rejected band** (rank-hysteresis ≈
  score-threshold-hysteresis; both gate name-set churn on a buffer width). Re-proposing it is
  effectively the 53.1 band renamed -> **AUTO-FAIL risk per the goal constraint.** AVOID unless a
  genuinely novel score-band formulation can be argued; the brief recommends AGAINST it.
- **Minimum holding period** — **DIFFERENT mechanism (time axis).** Forces a held name to remain N
  cycles regardless of rank/score churn — directly attacks the 1-day whipsaw round trips (MU -6.3%
  06-08→06-09; 000660.KS stop-out next day; DELL 4 trades/9 days). KF2 names this exact guardrail;
  KF7 (ThinkNewfound) says longer holds help net of costs. **Mechanistically distinct from the
  rejected band** (caps time-in-position, not name-set rank). **RECOMMENDED LEVER.**
- **Sector-concentration cap** — this is a RISK control (B11), arguably a FEATURE not a churn lever;
  it addresses the 100%-Tech/HHI-0.63 problem, not turnover directly. Better routed to the FEATURE
  variant.
- **Turnover budget** — **DIFFERENT mechanism (trade-count axis).** Caps gross turnover per period
  (e.g. skip the lowest-conviction swaps once a budget is hit). KF4 says band/budget turnover
  control is robust IF not a naive linear penalty. Distinct from the rejected band but harder to
  spec as default-off byte-identical. Secondary recommendation.

**Conclusion:** the evidence-best single lever is **minimum holding period** — it is the literature's
named anti-churn guardrail (KF2), mechanistically orthogonal to the 53.1-rejected rank band, and
maps directly to the observed 1-day whipsaws. An honest caveat for the chapter: even min-holding
must clear the same Ledoit-Wolf gate, and 53.1's prognosis (momentum's gross alpha is intact net of
costs, so turnover levers have a small edge on THIS book) means it MAY also land within noise — in
which case "the lever family is unpromising on the available history" is itself a valid, honest
research conclusion (exactly as 53.1 concluded), and the operator should lean FEATURE.

---

## Draft one-paragraph specs for BOTH phase-57 candidates

**LEVER variant — Minimum holding period (anti-whipsaw time-gate).** Add a config-gated,
default-OFF `min_holding_days` constraint (e.g. 3-5 cycles) that prevents a held paper position
from being sold for churn/rank reasons before it has been held N cycles, while ALWAYS allowing
risk exits (stop-loss, kill-switch, trailing-stop) to override (so it never weakens risk control —
do-no-harm). Mechanism is the time-axis guardrail named in arXiv:2505.07078v5 ("minimum holding
periods or cooldown windows") and is mechanistically DISTINCT from the 53.1-rejected rank-hysteresis
band (it caps time-in-position, not name-set rank), so it is not a renamed re-proposal. Measure
ON-vs-OFF via the $0 replay on the production S&P-500 universe (reusing the 51.2/52.x/53.1 machinery)
reporting Sharpe / return / turnover / maxDD, gross AND net-of-cost, and subject band-vs-baseline to
the SAME Ledoit-Wolf SR-difference robustness gate as 52.3/53.1 (`analytics.sharpe_diff_test`,
n_boot=5000, seed=42): promote ONLY if `p_one_sided<0.05 AND delta>=+0.05 AND ci_low>0` on the
net leg AND the gross leg clears `ci_low>-0.05`; a within-noise result is an honest REJECT (valid
outcome), the helper ships default-OFF + unit-tested for OFF byte-identity, and the +20% US momentum
core stays byte-identical unless the flag is enabled. Cites F-I/B15 (churn) and KF2/KF4/KF7.

**FEATURE variant — Binding RiskJudge gate + concentration-aware sizing (make the risk verdict
drive behavior).** Convert the RiskJudge REJECT from advisory-only (55.2 F-F: DELL was BOUGHT
2026-06-03 at `risk_judge_decision='REJECT'`) into a config-gated, default-OFF BINDING gate that
blocks (or hard-reduces to zero) a BUY when the judge returns REJECT, AND inject the live portfolio
sector breakdown into the judge prompt so it stops reasoning blind (F-G: it cited a phantom "10%
cap" while config is 30% and said "no portfolio sector breakdown was provided"). This is the
highest-EV feature direction per the structural-validity gate (arXiv:2602.14233 Rationale-Robustness
+ Epistemic-Calibration components) and the adversarial capability finding (arXiv:2605.28359:
RELIABILITY features beat ANALYTICAL features when selection alpha is illusory). Acceptance criteria
of 53.1 rigor: a unit test with a REJECT fixture that FAILS on pre-fix (trade executes) and PASSES
on fixed (trade blocked when flag ON); measure the would-have-been-blocked away-week trades and
their realized P&L (the DELL/MU/000660.KS whipsaws) as the ON-vs-OFF evidence; default-OFF so the
US momentum core is byte-identical unless enabled; explicitly NOT a live flag flip in phase-57.
Cites F-F/F-G/B11 and KF3/KF5. (Alternative FEATURE if the operator prefers a pure-construction
capability: per-market benchmark fetch (^KS11/^GDAXI) to make the VS-KOSPI card a true index-excess
readout (55.1 B7) — lower-EV, more cosmetic; the binding-REJECT feature is the recommended one.)

---

## Consensus vs debate (external)

- **Consensus (strong):** LLM-trading short-window wins do not survive longer, cost-inclusive
  evaluation; passive B&H is a hard baseline (FINSABER, StockBench, KTD-Fin all agree). Overtrading
  destroys value and needs system-level guardrails. MinTRL/DSR is the authoritative power framework.
- **Debate / nuance:** whether turnover levers add NET edge on a given book is book-specific —
  arXiv:2509.04541 + arXiv:2412.11575 (cited in 53.1) show real uplift on HIGH-turnover universes,
  but 53.1 found within-noise on THIS lower-turnover momentum book. The phase-57 LEVER measurement
  resolves this for min-holding specifically; a REJECT is a legitimate outcome.
- **Adversarial (KF3):** the strongest dissent against a FEATURES-heavy plan is that adding
  capability did not help returns (KTD-Fin) — which the chapter uses to steer FEATURES toward
  RELIABILITY (binding gate, guards, instrumentation) rather than ANALYTICAL depth (more agents).

## Pitfalls (from literature)

- Treating raw return as skill (KTD-Fin Barra attribution: returns are mostly passive factor
  exposure; 55.1 §7 says the same at n=7) — do NOT read +19% as alpha.
- Reporting a Sharpe/PSR on a sub-MinTRL window as evidence (the displayed Sharpe 4.72 / PSR 0.9993
  are noise; DSR=0.0 is the honest summary).
- Renaming the rejected band as "score hysteresis" and re-proposing it (auto-FAIL).
- Over-suppressing trading with a naive linear turnover penalty (arXiv:2509.04541 warns against;
  prefer band/time-based).
- Assuming full-mode agents = more alpha (KF3 adversarial).

## Application to pyfinagent (external -> internal file:line)

- KF2 churn-guardrail -> `backend/services/portfolio_manager.py` (sell path) + the lite re-eval
  cadence `settings.py:308 paper_reeval_frequency_days=3`; min-holding would gate the SELL decision
  there. The whipsaws live in `paper_round_trips` (55.1 §6).
- KF3 reliability>analytical -> `backend/services/portfolio_manager.py:185-198` (RiskJudge REJECT
  recorded then ignored — F-F) is the single highest-EV feature site.
- KF5 structural-validity components -> F-D guard at `orchestrator.py:2050`/`formatters.py:37`
  (Epistemic Calibration), F-E at `claude_code_client.py` + lite analyzer (instrumentation),
  55.2 §3 (Rationale Robustness — rationale not consumed).
- KF6 MinTRL -> `paper_metrics_v2.py:33 MIN_OBS_FOR_PSR=30`; the 58.1 "minimum live-window length"
  requirement is the ~2-year MinTRL at backtest Sharpe.
- Baseline comparison -> `optimizer_best.json` (Sharpe 1.17 core) + `/performance benchmark_return_pct`
  (passive SPY) — the chapter compares against BOTH per the criterion.

---

## Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 read in full: FINSABER,
      StockBench, 2602.14233, KTD-Fin, DSR-formula, Finance-Grounded-Opt, ThinkNewfound)
- [x] 10+ unique URLs total (7 full + 10 snippet-only = 17)
- [x] Recency scan (last 2 years) performed + reported (5 new findings + adversarial KF3)
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (burn BQ rows, 53.1 precedent, MinTRL, finding map)

Soft checks — gaps noted, no auto-fail:
- [x] Internal exploration covered every relevant module (paper_trader, portfolio_manager,
      analytics, rebalance_band, settings, perf_metrics_v2, manage UI, optimizer_best, BQ cost)
- [x] Contradictions / consensus noted (consensus vs debate vs adversarial section)
- [x] All claims cited per-claim (KF1-KF6 each carry source + file:line application)

### JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/research_brief_55.3.md",
  "gate_passed": true
}
```
</content>
</invoke>
