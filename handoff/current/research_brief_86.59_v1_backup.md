# Research Brief — step 86.59

**Tier:** complex (caller-specified; not self-selected)
**Audit-class:** NO (coverage reported for information only; `coverage.dry` not required)
**Objective:** why pure trailing-return momentum ranking produces near-zero cross-sectional
turnover day-over-day, and what makes a stock screener select a varied candidate set without
weakening it — cross-sectional standardisation, residual/idiosyncratic momentum, short-horizon
reversal, sector-neutralisation, explore-exploit / bandit candidate generation, turnover-aware
and diversity-penalised portfolio construction, and out-of-sample validation of each.

**Brief path:** `handoff/current/research_brief_86.59.md`
**Date:** 2026-08-12

---

## ENVELOPE

```json
{
  "brief_status": "COMPLETE",
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 24,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "coverage": {
    "audit_class": false,
    "rounds": 4,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "gate_passed": true
}
```

---

## METHOD NOTE — WebSearch was unavailable (disclosed, not hidden)

`WebSearch` returned **"this session has used its web search budget (200 of 200 WebSearch
calls)"** on my FIRST search attempt — the budget is session-shared and was exhausted before
this researcher was spawned. `WebFetch` is unaffected.

Search substitutes actually used (each a real search pass, not a snippet harvest):
- **arXiv search UI via WebFetch** — `https://arxiv.org/search/?searchtype=all&query=...`
  returns real result lists with IDs, titles, dates. Four queries run; two produced results,
  two returned arXiv's "produced no results" page (reported below, not hidden).
- **Semantic Scholar Graph API via curl** — rate-limited during this run (1 of 5 queries).
- **arXiv Atom API via curl** — returned empty in this environment; abandoned.

**Three-variant discipline** (required by `.claude/rules/research-gate.md`) is satisfied and
visible in the tables: **year-less canonical** prior art (Novy-Marx & Velikov; Gârleanu &
Pedersen; Daniel & Moskowitz; short-term reversal), **last-2-year** work (arXiv 2024-2025), and
**current-year 2026** frontier (arXiv 2601.*, 2602.*, 2603.*, 2606.*, 2607.*).

PDF handling follows `.claude/rules/research-gate.md`: `WebFetch` was **never** called on an
`arxiv.org/pdf/` URL. NBER PDFs were extracted locally with `pypdf` (project venv) because
WebFetch PDF summarisation is a known quote-fabrication risk here (auto-memory
`reference_webfetch_pdf_summaries_fabricate_quotes`). **Every NBER quote below was
regex-verified against the extracted text**; none is a model summary.

---

## Read in full (6 — counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Verified core finding |
|---|-----|----------|------|-------------|------------------------|
| 1 | https://www.nber.org/system/files/working_papers/w20721/w20721.pdf | 2026-08-12 | peer-reviewed WP | curl + pypdf, 61 pp / 103,894 ch | Novy-Marx & Velikov, *A Taxonomy of Anomalies and Their Trading Costs* — the 50%-turnover survival line |
| 2 | https://www.nber.org/system/files/working_papers/w15205/w15205.pdf | 2026-08-12 | peer-reviewed WP | curl + pypdf, 47 pp / 69,840 ch | Gârleanu & Pedersen, *Dynamic Trading with Predictable Returns and Transaction Costs* — aim portfolio, partial adjustment |
| 3 | https://www.nber.org/system/files/working_papers/w20439/w20439.pdf | 2026-08-12 | peer-reviewed WP | curl + pypdf, 52 pp / 118,626 ch | Daniel & Moskowitz, *Momentum Crashes* — vol-scaled dynamic momentum ~doubles Sharpe |
| 4 | https://arxiv.org/html/2408.09168v1 | 2026-08-12 | paper (CS/IR) | WebFetch, full text | Lichtenberg et al., *Multinomial Blending* — slate composition without touching scores; MMR baseline drifts |
| 5 | https://arxiv.org/html/2601.08717v1 | 2026-08-12 | paper (q-fin) | WebFetch, full text | Garcia & Messud — HHI diversity penalty vs. bounded-degradation form; synthetic energy assets |
| 6 | https://quantpedia.com/strategies/short-term-reversal-in-stocks/ | 2026-08-12 | industry | WebFetch, full page | Short-term reversal: 1-week rank, weekly rebalance, large-cap restriction is the cost fix |

## Identified but snippet-only (24 — context; does NOT count toward the gate)

| URL / ID | Kind | Why not read in full |
|---|---|---|
| https://www.nber.org/system/files/working_papers/w18098/w18098.pdf | WP | **Fetched and REJECTED** — I expected Asness-Moskowitz-Pedersen; the extracted text is *"Market Design in Cap and Trade Programs"* (Holland & Moore). Wrong paper; excluded rather than mis-cited. |
| https://arxiv.org/abs/2408.09168 | abs page | Superseded by the `/html/` full read (#4) |
| https://arxiv.org/abs/2601.08717 | abs page | Superseded by the `/html/` full read (#5) |
| https://www.aqr.com/Insights/Research/Journal-Article/Craftsmanship-Alpha-An-Alternative-to-Factor-Investing | industry | **Dead** — resolves to AQR 404 / fraud-warning page |
| https://alphaarchitect.com/the-limits-of-anomalies-trading-costs/ | blog | HTTP **403**, bot-blocked |
| https://en.wikipedia.org/wiki/Momentum_(finance) | encyclopaedia | Tier-5, deliberately unused |
| arXiv:2601.04062 — Smart Predict-then-Optimize for Portfolio Optimization | paper | "transaction costs, turnover control, and regularization" in the training objective |
| arXiv:2607.21170 — Dynamic Rebalancing via TDA + News Sentiments | paper | Retention mechanism "reducing portfolio turnover" |
| arXiv:2603.16904 — Quantum-Assisted Optimal Rebalancing | paper | Claims 44.5% transaction-cost reduction via fewer rebalances |
| arXiv:2309.10152 — Sparse Index Tracking (l0) | paper | Turnover-sparsity constraint |
| arXiv:2303.12751 — Fast Large-Scale Portfolio Optimization | paper | l1/l2 regularisation to "reduce an excessive turnover" |
| arXiv:2206.14760 — Swarm, cardinality-constrained PO | paper | Exact penalty function on the turnover constraint |
| arXiv:2601.06507 — Emissions-Robust Portfolios | paper | Off-topic (ESG) |
| arXiv:2410.04217 — Improving Portfolio Optimization with Bandit Networks | paper | ADTS/CADTS, "Sharpe ratio 20% higher"; strongest bandit candidate |
| arXiv:2406.06552 — Optimizing Sharpe Ratio in MAB | paper | UCB for risk-adjusted arm choice |
| arXiv:2205.05843 — Survey of Risk-Aware Multi-Armed Bandits | survey | Best entry point if bandits are pursued |
| arXiv:2606.23933 — Flow-Corrected Thompson Sampling, non-stationary | paper | 2026 frontier; portfolio-selection benchmark |
| arXiv:2602.15972 — Hierarchical Unimodal Thompson Sampling | paper | 2026; portfolio management with risky assets |
| arXiv:2512.09850 — Conformal bandits | paper | Statistical validity under weak arm separability |
| arXiv:2211.14768 — Constrained Pure Exploration MAB, fixed budget | paper | Closest bandit formulation to a fixed daily analyse-budget |
| arXiv:2206.12463 — Risk-averse Contextual MAB, linear payoffs | paper | Mean-variance Thompson Sampling |
| arXiv:1709.04415 — Risk-Aware MAB with Application to Portfolio Selection | paper | Coherent risk measures in MAB |
| arXiv:1911.05309 — Adaptive Portfolio via Thompson Sampling | paper | Year-less canonical bandit-portfolio hit |
| arXiv:2312.03294 — Generative Models of Asset Returns | paper | MAB framework for strategy blending/switching |

---

## Recency scan (2024-2026) — PERFORMED

Explicit passes run for the 2024-2026 window (arXiv search UI, three query families:
turnover/rebalancing, bandit/portfolio, diversity/selection).

**Result: 3 findings that COMPLEMENT — and 0 that supersede — the canonical sources.**

1. **Diversity penalties have moved from ad-hoc weights to *bounded-degradation* programs.**
   arXiv:2601.08717 (Jan 2026) formulates diversification as *maximise diversity subject to a
   bounded ROI/CVaR loss* (tolerances Δp, Δr) rather than a free-floating penalty weight. This
   is directly relevant and is a **better template than our current `w`** (see Application §3).
2. **Turnover control is being pushed into the training objective**, not applied post-hoc —
   arXiv:2601.04062 (Jan 2026) embeds "transaction costs, turnover control, and regularization"
   in a decision-focused learning objective; arXiv:2607.21170 (Jul 2026) uses an explicit
   retention mechanism to "preserve high-quality assets across consecutive rebalancing windows".
   Note the direction of travel: recent work is trying to *reduce* turnover, not create it.
3. **Bandit-for-portfolio is an active but non-stationarity-limited literature.** 23 hits, with
   2026 entries (2606.23933, 2602.15972) concentrating on *non-stationary* reward drift — the
   exact regime a stock screener lives in. arXiv:2410.04217 (Oct 2024) reports Sharpe ~20%
   above classical approaches. **Read at snippet level only — lower confidence, flagged.**

Nothing found in the window overturns Novy-Marx & Velikov's cost result, Gârleanu-Pedersen's
partial-adjustment result, or Daniel-Moskowitz's crash/vol-scaling result. They remain load-bearing.

---

## Key findings (per-claim citations)

**F1 — Turnover is not free, and ~50% one-sided monthly turnover is the empirical survival
line.** Verbatim from the extracted text: *"Most of the anomalies that we consider with
one-sided monthly turnover lower than 50% continue to generate statistically significant net
spreads, at least when designed to mitigate transaction costs. Few of the strategies with
higher turnover do."* And: *"while transaction costs dramatically reduce the profitability of
many anomalies, especially those with high turnover, designing strategies to minimize
transaction costs significantly reduces these costs."* (Novy-Marx & Velikov 2014, w20721.)
This is the primary constraint on *any* proposal to make the screener churn more.

**F2 — The single most effective cost mitigation is a banding/abstention rule, not a cleverer
score.** *"...not actively trade into, is the single most effective simple cost mitigation
strategy"* (w20721). The paper also finds *"trading costs are persistent and significantly
positively associated with idiosyncratic volatility"*, and that costs *"for equal-weighted
strategies are generally two to three times as high"* than value-weighted. (Novy-Marx & Velikov 2014.)

**F3 — Optimal response to a moving signal is PARTIAL adjustment toward an "aim", and slow
signals correctly produce low turnover.** *"The optimal strategy is characterized by two
principles: 1) aim in front of the target and 2) trade partially towards the current aim."* The
aim is *"a weighted average of the current Markowitz portfolio (the moving target) and the
expected Markowitz portfolios on all future dates"*, and crucially *"predictors with slower mean
reversion (alpha decay) get more weight in the aim portfolio."* (Gârleanu & Pedersen, w15205.)
**This reframes the whole step:** a 1/3/6-month composite is a *slow* predictor, so low turnover
is the theoretically CORRECT behaviour for it. The defect is not "turnover too low" — it is
"the signal menu contains nothing fast".

**F4 — Volatility-scaling momentum is the best-documented Sharpe improvement, and it is a
SIZING change, not a selection change.** *"An implementable dynamic momentum strategy based on
forecasts of momentum's mean and variance approximately doubles the alpha and Sharpe Ratio of a
static momentum strategy, and is not explained by other factors."* Momentum's payoff is
option-like with asymmetric bear-market betas (*"most of the up- versus down-beta asymmetry in
bear markets is driven by the past losers"*). (Daniel & Moskowitz, w20439.) Our vol leg is a
step function (`vol > 0.6 → ×0.85`), a crude proxy for this.

**F5 — You can add diversity WITHOUT touching the score: change slate composition, not the
ranking function.** Multinomial blending *"first samples a content type according to c∼M(p) and
then selects the highest-scoring remaining candidate from that content type"*, preserving the
personalised order *within* each type. Its decisive property: *"The average exposure guarantees
are independent of the underlying scoring function h and therefore remain stable even after
model re-training or non-stationary user behavior."* (Lichtenberg et al. 2024, arXiv:2408.09168.)
This is the cleanest available answer to "varied without weakening".

**F6 — Score-penalty diversity (MMR-style) needs constant retuning and drifts.** The same paper
rejected MMR operationally because *"various market places required different MMR penalty
parameters and optimal diversification rates would change over time"* — MMR won on one metric
(+13.57% vs +18.82% podcast time; +2.76% vs +2.23% engagement) yet MB *"was globally launched due
to its operational advantages"*. (arXiv:2408.09168.) This is a direct warning about
`paper_soft_sector_diversity_w` as a hand-set constant.

**F7 — Prefer "maximise diversity subject to bounded performance loss" over a free penalty
weight.** Garcia & Messud give both forms; the second *"maximizes diversification while
controlling expected profit and risk degradation"* via explicit tolerances, letting the operator
state the acceptable loss (e.g. 10%) instead of guessing a weight. **Caveat, verbatim: evaluated
*"using synthetic data (energy assets)"* with 100 scenarios** — no equity evidence.
(arXiv:2601.08717, Jan 2026.)

**F8 — Short-horizon reversal is the only surveyed mechanism that creates genuine day-over-day
variation, and its cost problem has a known fix.** Rank on the past *week*, rebalance weekly,
universe = 100 largest caps: 16.25% p.a. net, Sharpe 1.09, maxDD -52.94% (1990-2009). *"the
impact of transaction costs on reversal profits can largely be attributed to excessively trading
in small cap stocks"* — restricting to large caps *"significantly reduces trading costs"*.
Blitz, van der Grient & Honarvar's enhanced version exists specifically to counter reversal's
*"tendency to go against short-term momentum in industry and factor returns"*. (Quantpedia.)

**F9 — Bandit/explore-exploit selection is real but the worst fit here (low confidence,
snippet-only).** The literature is large (23 hits) and 2026 work centres on non-stationary
drift. But a bandit needs a per-pull reward; this screener deep-analyses
`paper_analyze_top_n = 5` names/day (`settings.py:407`) with a delayed, noisy P&L reward. The
closest formulation is fixed-budget pure exploration (arXiv:2211.14768). **I did not read any
bandit paper in full — treat F9 as a pointer, not evidence.**

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/tools/screener.py` | :248-296 | `rank_candidates` signature (24 overlay params) | LIVE |
| `backend/tools/screener.py` | :299-305 | **momentum composite** | **LIVE — the whole ranking** |
| `backend/tools/screener.py` | :306-315 | RSI + vol step penalties | LIVE |
| `backend/tools/screener.py` | :317-405 | 13 `apply_*_to_score` overlays, each `if <signal>:` | all gated |
| `backend/tools/screener.py` | :420-430 | news-only injection at flat `5.0*1.10` | gated |
| `backend/tools/screener.py` | :443-452 | `_apply_multidim_momentum` | dark |
| `backend/tools/screener.py` | :454-484 | sector-neutral percentile re-score | dark |
| `backend/tools/screener.py` | :486-492 | `_apply_52wh_tilt` | dark |
| `backend/tools/screener.py` | :494-499 | `_apply_soft_sector_diversity` | dark |
| `backend/tools/screener.py` | :501-502 | **`sort()` + `[:top_n]`** | **LIVE — structural choke point** |
| `backend/tools/screener.py` | :505-539 | soft-diversity impl `(1-w)^j`, sign-safe | dark |
| `backend/tools/screener.py` | :541-553 | `_zscore` | **defined; unreachable on the live path** |
| `backend/services/autonomous_loop.py` | :169-209 | `_min_k_sector_slice` | dark (k=0) |
| `backend/services/autonomous_loop.py` | :982-1019 | `rank_candidates` call site | LIVE |
| `backend/services/autonomous_loop.py` | :1120-1122 | held-ticker exclusion | LIVE |
| `backend/services/autonomous_loop.py` | :1124-1129 | min-K vs plain top-N branch | LIVE, K=0 |
| `backend/autoresearch/gate.py` | :22-25 | `PromotionGate(min_dsr=0.95, max_pbo=0.20, min_pbo_trials=10)` | LIVE |
| `backend/services/promotion_gate.py` | :53-63 | staging `PBO_CEILING` check | LIVE |
| `backend/config/settings.py` | :407, :468-469, :487-489 | the three dark mitigations + `paper_analyze_top_n=5` | default OFF/0 |

### The composite, verbatim (`backend/tools/screener.py:299-305`)

```python
score = (
    mom_1m * 0.40 +
    mom_3m * 0.35 +
    mom_6m * 0.25
)
```

### ROOT CAUSE — why day-over-day turnover is ~zero (structural, not a data bug)

Every term is a **trailing cumulative return** over ~21 / 63 / 126 trading days. Rolling one day
forward swaps out one observation and swaps in one:

- `mom_1m` changes by O(1/21) of its window; `mom_3m` O(1/63); `mom_6m` O(1/126).
- The composite's day-over-day autocorrelation is therefore very close to 1.
- The **rank** vector is stickier still: an ordering change requires two names' scores to
  *cross*, i.e. a shock exceeding the gap between them.
- `sort()` + `[:top_n]` (`:501-502`) then takes the top slice of a near-static ordering, so
  **set** turnover is even lower than score turnover.

The RSI/vol legs cannot fix this: they are **multiplicative constants on discrete thresholds**
(`rsi>80 → ×0.7`, `rsi<20 → ×0.8`, `vol>0.6 → ×0.85`). They fire rarely and apply the *same*
constant to everyone in a bucket, so they move levels, not ordering, except exactly at a crossing.

**No cross-sectional standardisation exists on the live path.** `_zscore` (`:541-553`) is called
**only** by `_apply_multidim_momentum`, gated on `multidim_momentum_enabled`
(`settings.py:478`, default `False`). So raw returns of three different horizons are summed as if
commensurate. A 6-month return has roughly √6 ≈ 2.4× the dispersion of a 1-month return, so the
6m term's **effective** weight far exceeds its nominal 0.25 — making the composite *more*
persistent than the weights suggest, and meaning **the declared weights 0.40/0.35/0.25 are not
the weights actually in force.**

### Every overlay is default-OFF (measured from `settings.py`)

`pead_signal` :431 · `news_screen` :435 · `meta_scorer` :442 · `analyst_revisions` :450 ·
`sector_neutral_momentum` :468 · `sector_momentum` :471 · `multidim_momentum` :478 ·
`momentum_52wh_tilt` :483 · `paper_soft_sector_diversity` :487 (`w`=0.0 :488) ·
`paper_min_k_sectors_analyzed` :489 (=0) · `options_flow_screen` :506 ·
`insider_signal_screen` :516 · `call_transcript_gpr` :538 · `social_velocity` :544 ·
`defense_signal` :551 · `peer_leadlag` :558 · `ma_preannounce` :565 — **all `False`/0.**

If those defaults hold at runtime, `rank_candidates` computes the composite, skips every
`if <signal>:` block, sorts, truncates — so the **entire live ranking is three trailing returns
plus two step penalties**, and every mechanism that could add cross-sectional variation is dark.

> **LIMITATION (disclosed).** Reading `backend/.env` was **DENIED** by the permission system.
> These are *declared defaults*, not proof of the running process's values. Per this project's
> standing "committed is NOT in force" rule, **Main must re-derive the live values from the
> running backend** before relying on the "all overlays dark" framing.

### The two dark mitigations rest on non-equity citations — VERIFIED BY READING BOTH

- `settings.py:488` cites *"arXiv 2601.08717 shades-never-zeroes"*. The paper is Garcia &
  Messud (13 Jan 2026), HHI diversification evaluated *"using synthetic data (energy assets)"*.
  The *shape* of the claim (penalise concentration in the objective rather than eliminate
  candidates) is genuinely supported; the asset class, data and objective are not ours.
- `autonomous_loop.py:178` cites *"arXiv 2408.09168 multinomial-blend leader-pick"*. The paper
  is an **Amazon Music learning-to-rank** paper about mixing music/podcasts/videos into one
  slate. Legitimate for the *mechanism*; it carries **no return, Sharpe or cost result**.

Cross-domain transfer is a legitimate move — but the contract must not describe these as
validating the *financial* case. **Additionally, a fidelity gap:** MB's headline guarantee
(exposure independent of the scoring function, stable under retraining) derives from
**stochastic sampling** `c∼M(p)`. `_min_k_sector_slice` (`:188-200`) is a **deterministic**
top-k-sectors-by-peak leader pick. It does **not** inherit that guarantee, and it introduces no
exploration whatsoever.

### The one measured equity number already in the tree

`settings.py:487` records that the 2026-06-01 replay measured **-0.166 long-only Sharpe for
hard sector-neutralisation** — this project's own evidence, and it is *negative*. That is why
70.2 chose the soft `(1-w)^j` decay leaving each sector leader untouched. F5/F6 independently
support that ordering: prefer slate-composition changes over score mutation.

---

## Consensus vs debate (external)

**Consensus.** (a) Turnover must be paid for and most high-turnover anomalies do not survive
costs (F1). (b) Optimal trading is *partial* adjustment toward an aim, never full rebalancing to
the target (F3). (c) Slow signals *should* produce low turnover (F3). (d) Diversity is safer
imposed on slate composition than on the score (F5/F6).

**Live debate — cited inside w20721.** Novy-Marx & Velikov record that *"More recently Frazzini
et al. (2014) have argued that 'actual trading costs are less than a tenth as large as, and
therefore the potential scale of these strategies is more than an order of magnitude la[rger]'"*.
So the size of the cost constraint is genuinely contested: N-M&V's estimates are the conservative
end, Frazzini-Israel-Moskowitz's live institutional data the permissive end. **A turnover-raising
proposal that only survives under FIM-style optimistic costs should be treated as unproven.**

**Second tension.** F8 (add a fast reversal signal) pushes turnover *up*; F1/F2 and the entire
2024-2026 recency window push it *down*. These are reconciled only by F3: add the fast signal to
the *forecast*, then let partial adjustment decide how much of it to trade.

---

## Pitfalls (from the literature)

1. **Churn for its own sake destroys value** — few >50%-turnover strategies keep significant net
   spreads (F1). Any picker change must report measured turnover, not assume it is affordable.
2. **Diversity penalties that mutate `composite_score` contaminate the metric the gates read.**
   `_apply_soft_sector_diversity` overwrites `composite_score` (`:538`); DSR/PBO are then computed
   on a score that is part signal, part penalty. `_min_k_sector_slice` does not have this problem.
3. **Hand-set penalty weights drift** and need per-market retuning (F6) — exactly the failure mode
   `paper_soft_sector_diversity_w` invites.
4. **Hard sector-neutralisation is already measured negative here** (-0.166 Sharpe).
5. **Cost is concentrated in small caps and high-idiosyncratic-vol names** (F2, F8) — a naive
   diversity rule that reaches deeper into sectors reaches into *exactly* those names.
6. **Bandit exploration needs dense reward**; 5 analysed names/day with delayed P&L is the
   sparsest plausible feedback (F9).
7. **Every picker variant tried raises the DSR bar.** DSR's N is the *iteration/trial count*
   (project memory `project_dsr_trial_count_reset_82_25`), so sweeping `w` and `K` over many
   values inflates N and makes `min_dsr=0.95` harder — a real cost of a wide search.
8. **Unstandardised weights are not the stated weights** — declaring 0.40/0.35/0.25 while
   summing raw multi-horizon returns misrepresents the model to anyone reading the config.

---

## Application to pyfinagent (external findings → file:line anchors)

**A1. Cross-sectional standardisation is the cheapest true fix, and the helper already exists.**
Z-score each horizon before weighting so nominal weights become effective weights. `_zscore` is at
`screener.py:541-553` but is reachable only through the dark multidim path. Making the *live*
composite standardised is a small, auditable change at `:299-305`. Note it is **not** a turnover
fix on its own (a monotone re-weighting of the same slow state) — it is a **correctness** fix.

**A2. Only a fast-decaying term creates real day-over-day variation.** Per F3, every currently
proposed overlay is another slow signal, so it cannot move day-to-day set membership much.
Short-horizon reversal (F8) is the one surveyed mechanism that does — but it must be
residualised against industry/factor momentum (Blitz-van der Grient-Honarvar) or it fights the
momentum leg, and it must be large-cap-restricted or costs eat it.

**A3. Prefer slate-composition over score mutation.** `_min_k_sector_slice`
(`autonomous_loop.py:169-209`, gated by `paper_min_k_sectors_analyzed`, `settings.py:489`) changes
only *which* names reach the deep-analyse slice and leaves `composite_score` untouched — so DSR/PBO
still measure the signal. `_apply_soft_sector_diversity` (`screener.py:494-499, 505-539`) mutates
the score. **On F5/F6 evidence, prefer the min-K lever; treat `w` as the riskier of the two.**

**A4. If a diversity penalty is used, bound its cost explicitly.** Replace a free-floating `w`
with the F7 form: maximise sector spread **subject to** a stated maximum Sharpe/return
degradation. That converts "how much diversity" into "how much am I willing to pay", which is
directly checkable against the gates instead of being a magic constant.

**A5. Consider making the min-K pick stochastic** to actually inherit the MB guarantee (F5) and to
introduce the only cheap exploration available — sample the sector, then take its leader, rather
than the deterministic top-k-by-peak at `autonomous_loop.py:188-200`. This is a genuine
explore-exploit mechanism that costs no new data feed. It does, however, make cycles
non-reproducible without a seed — a real tradeoff for this project's replay discipline.

**A6. Every change must still clear the immutable gates.** `PromotionGate(min_dsr=0.95,
max_pbo=0.20, min_pbo_trials=10)` at `autoresearch/gate.py:22-25`, plus the staging `PBO_CEILING`
at `promotion_gate.py:53-63`. Per project memory, **PBO — not DSR — has been the binding wall**.
Add a pre-gate sanity check from F1: report **one-sided monthly turnover** next to DSR/PBO, and
treat >50% as requiring explicit justification.

**A7. Do not describe the two arXiv citations as financial validation** in the contract. State
them as cross-domain mechanism transfers, and note the deterministic-vs-stochastic fidelity gap.

---

## Evidence gaps (honest)

1. **Residual / idiosyncratic momentum: NO source read in full.** arXiv search for
   "residual momentum idiosyncratic momentum" returned arXiv's *"produced no results"* page;
   Semantic Scholar was rate-limited; the canonical source (Blitz, Huij & Martens 2011,
   *Journal of Empirical Finance*) is SSRN/paywalled and I could not fetch it. **I therefore make
   no evidenced claim about residual momentum's magnitude or its turnover profile.** This is the
   single largest gap and Main should treat that branch as unresearched.
2. **Bandit/explore-exploit: snippet-level only** (F9). 23 candidates identified, none read.
3. **Sector-neutralisation external evidence is thin** — the strongest datapoint is this
   project's own -0.166 replay, not the external literature.
4. **`backend/.env` not readable** — runtime flag values unverified (see LIMITATION above).
5. **Frazzini et al. (2014)** — the adversarial cost estimate is quoted *via* w20721, not read
   at source.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL — **6** (3 NBER via documented
      curl+pypdf chain, 3 via WebFetch full text)
- [x] 10+ unique URLs total — **30** (6 read-in-full + 24 snippet-only)
- [x] Recency scan (2024-2026) performed + reported — 3 complementary findings, 0 superseding
- [x] Full papers/pages read, not abstracts — arXiv `/html/` full texts used, not `/abs/`;
      NBER full text extracted (61/47/52 pp) and quotes regex-verified
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module in the caller's scope
- [x] Contradictions/consensus noted (Novy-Marx & Velikov vs Frazzini et al. on cost magnitude;
      F8 vs F1 on turnover direction)
- [x] Claims cited per-claim
- [ ] **Residual momentum branch NOT covered** — documented in Evidence gaps §1, not padded over

**gate_passed: true** — the >=5 floor, the >=10 URL floor and the recency scan are all met, with
the residual-momentum gap disclosed rather than concealed.
