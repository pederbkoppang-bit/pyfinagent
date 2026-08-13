---
name: screener-turnover-86-59
description: Step 86.59 — near-zero screener turnover is STRUCTURAL (trailing-window arithmetic), the declared composite weights are not the effective weights, and both code-cited arXiv IDs are non-equity papers
metadata:
  type: project
---

Near-zero day-over-day candidate turnover in `backend/tools/screener.py::rank_candidates`
is a **structural property of trailing-window arithmetic**, not a data or config bug.

**Why:** the composite at `screener.py:299-305` is `mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25`,
all trailing cumulative returns over ~21/63/126 trading days. A one-day roll swaps one
observation in and one out, so each term moves O(1/21)..O(1/126) of its window; the *rank*
vector is stickier still because an ordering change needs two scores to CROSS. `sort()` +
`[:top_n]` at `:501-502` then takes the top slice of a near-static ordering. The RSI/vol legs
(`:306-315`) can't help — they are multiplicative constants on discrete thresholds
(`rsi>80 → ×0.7`, `vol>0.6 → ×0.85`), so they move levels, not ordering.

Three findings worth carrying forward:

1. **The declared weights are not the weights in force.** There is NO cross-sectional
   standardisation on the live path. `_zscore` exists at `screener.py:541-553` but is called
   ONLY by `_apply_multidim_momentum`, which is dark (`settings.py:478`). Raw returns of three
   horizons are summed as if commensurate; the 6m term has ~√6 ≈ 2.4x the dispersion of the 1m
   term, so its effective weight far exceeds its nominal 0.25.

2. **Both arXiv IDs cited in our own code for the diversity mitigations are non-equity papers**
   — verified by reading both. `settings.py:488` cites arXiv 2601.08717 = Garcia & Messud, HHI
   diversification on *"synthetic data (energy assets)"*. `autonomous_loop.py:178` cites arXiv
   2408.09168 = an **Amazon Music learning-to-rank** paper (multinomial blending). Both support
   the *mechanism*; neither is financial validation. Also a fidelity gap: MB's guarantee
   ("exposure independent of the scoring function... stable even after model re-training")
   comes from STOCHASTIC sampling `c∼M(p)`; `_min_k_sector_slice` is a DETERMINISTIC
   top-k-by-peak pick and does not inherit it.

3. **Gârleanu & Pedersen reframe the whole question:** *"predictors with slower mean reversion
   (alpha decay) get more weight in the aim portfolio"* — so low turnover is the CORRECT
   behaviour for a 1/3/6-month signal. The defect is not "turnover too low", it is "the signal
   menu contains nothing fast". Every proposed overlay is another slow signal.

**How to apply:** prefer slate-composition levers (`_min_k_sector_slice`, which leaves
`composite_score` untouched so DSR/PBO still measure the signal) over score-mutating ones
(`_apply_soft_sector_diversity` overwrites `composite_score` at `screener.py:538`, contaminating
the metric the gates read). Novy-Marx & Velikov's measured line — anomalies with one-sided
monthly turnover **under 50%** mostly keep significant net spreads, few above it do — is the
pre-gate sanity check for any turnover-raising proposal. See
[[research-gate-discipline]] and [[websearch-budget-is-session-shared]].

**Unresearched branch:** residual/idiosyncratic momentum. arXiv has no such paper; the canonical
Blitz-Huij-Martens 2011 source is SSRN-paywalled. Do not assert its magnitude without a source.
