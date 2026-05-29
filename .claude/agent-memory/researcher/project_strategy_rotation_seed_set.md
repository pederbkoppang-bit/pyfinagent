---
name: strategy-rotation-seed-set
description: phase-48.1 research — how to design a SEED SET of distinct quant strategies for the rotation producer, and how DSR/PBO/effective-N gate eligibility. Complements the selector that already exists.
metadata:
  type: project
---

phase-48.1 research gate (2026-05-29). The producer half of dynamic rotation:
`strategy_selector.select_best_strategy` is PURE and EXISTS (see
[[strategy-rotation-infra]]); this cycle builds what FEEDS its `per_strategy`
input + a config-driven seed strategy set. Currently only ONE config exists
(optimizer_best.json).

**Seed-set design principles (from literature):**
- Diversify along orthogonal AXES, not parameter tweaks of one idea: strategy
  TYPE (mean-reversion vs trend/momentum vs breakout), MARKET/universe, and
  TIMEFRAME/holding-horizon. Parameter variants of the SAME idea are correlated
  near-duplicates and inflate the trial count without adding independence.
- Canonical pairing: mean-reversion + trend-following are structurally
  anti-correlated (MR wants choppy/range-bound, TF wants sustained moves), so
  their bad periods land at different times — smoother equity curve, lower maxDD.
- Min count: practitioner floor is ">= 2" distinct strategies (Build Alpha). For
  a DSR-deflated bake-off, a 5-strategy seed is reasonable IF the 5 are along
  distinct axes; if they are 5 correlated variants the EFFECTIVE N is ~1-2.
- Never ensemble/seed poor strategies hoping for a miracle — ensembling polishes
  gold, it does not turn dirt into gold (Build Alpha, citing Lopez de Prado).

**DSR/PBO eligibility gate (how it should gate which strategies deploy):**
- DSR formula (Wikipedia/Bailey): DSR = Phi( (SR_hat - SR0)*sqrt(T-1) /
  sqrt(1 - g3*SR0 + (g4-1)/4 * SR0^2) ); SR0 = sqrt(V[SR_n]) *
  ((1-gamma)*Phi^-1[1-1/N] + gamma*Phi^-1[1-1/(N*e)]); gamma=0.5772, e=2.718.
- N = number of INDEPENDENT trials, NOT raw backtest count. Correlated
  strategies have N_eff < N. The project selector docstring already flags that
  plain N over-deflates (the SAFE direction). Effective-N via clustering (ONC /
  hierarchical / spectral) or VertoxQuant's K_eff (root-find on expected max
  chi-square). NOTE: VertoxQuant shows Bailey&LdP's own effective-N measure can
  give nonsense (K_eff=5 for 2 perfectly anticorrelated strats); prefer
  clustering or K_eff with the 5 axioms (1<=Keff<=K; =1 iff fully dependent;
  =K iff independent; non-decreasing; ~sqrt(2 log Keff) asymptotics).
- PBO via CSCV (Bailey/Borwein/LdP/Zhu): partition the T x N perf matrix into S
  disjoint submatrices, form all C(S,S/2) train/complement splits, pick IS-best,
  measure its OOS rank; PBO = fraction of splits where IS-best lands BELOW OOS
  median (logit < 0). High PBO => the selection procedure itself is overfit.
  Project gate: DSR>=0.95 AND PBO<=0.20 (PromotionGate). S typically 8 or 16.
- MinBTL (Bailey&LdP "pseudo-math"): E[max SR] ~ sqrt(2 log N) * sigma_SR, so
  more trials mechanically inflate the best Sharpe; MinBTL ~ log(N)-driven
  sample length needed. Implication: keep the seed set SMALL and pre-registered
  rather than sweeping hundreds of variants.

**Adversarial / qualifying finding:** GT-Score 2026 (MDPI/arXiv 2602.00080)
argues DSR/PBO are POST-HOC (evaluate after the search) and instead bakes
robustness INTO the optimization objective (mu*ln(z)*r^2 / downside_dev). It
positions itself as COMPLEMENTARY, not a replacement — so DSR/PBO-as-eligibility
gate still stands, but a pure rank-by-DSR producer could be complemented by a
robustness-aware objective in the per-strategy backtests that feed it.

**Why:** phase-48.1 builds the per_strategy producer + seed configs.
**How to apply:** seed 4-6 strategies on distinct TYPE/horizon axes; run quant-
only ($0-LLM) walk-forward backtests producing {strategy_id, dsr, pbo, params};
deflate DSR by N_eff (clustering), not raw 5; feed to the existing selector +
PromotionGate. Extend friday_promotion/promoted_strategies, do not fork.
Sources: davidhbailey.com/dhbpapers/deflated-sharpe.pdf + backtest-prob.pdf +
ssrn-id2507040 (MinBTL); en.wikipedia.org/wiki/Deflated_Sharpe_ratio;
vertoxquant.com/p/the-effective-number-of-tested-strategies;
arxiv.org/html/2402.05272v2 (jump model turnover); arxiv 2603.09219 (AlgoXpert
IS/WFA/OOS); mdpi.com/1911-8074/19/1/60 + arxiv 2602.00080 (GT-Score).
