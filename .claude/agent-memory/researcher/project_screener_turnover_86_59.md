---
name: screener-turnover-86-59
description: Step 86.59 — near-zero screener turnover is STRUCTURAL (trailing-window arithmetic); declared composite weights are not the effective weights; _zscore is at :532 NOT :541-553; residual-momentum branch now CLOSED; a summed-vs-per-side turnover convention mismatch reverses the naive conclusion
metadata:
  type: project
---

Near-zero day-over-day candidate turnover in `backend/tools/screener.py::rank_candidates`
is a **structural property of trailing-window arithmetic**, not a data or config bug.

**Why:** the composite at `screener.py:299-305` is `mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25`,
all trailing cumulative returns over ~21/63/126 trading days. A one-day roll swaps one
observation in and one out, so each term moves O(1/21)..O(1/126) of its window; the *rank*
vector is stickier still because an ordering change needs two scores to CROSS. `sort()` +
`[:top_n]` then takes the top slice of a near-static ordering. The RSI/vol legs can't help —
they are multiplicative constants on discrete thresholds, so they move levels, not ordering.

**ANCHORS — re-verified at source 2026-08-13; the earlier version of this memory was WRONG:**
`_zscore` is defined at **`screener.py:532`** (NOT `:541-553`) and called at **exactly four
sites, `:607-610`**, all inside `_apply_multidim_momentum` (**`:564`**, NOT `:443-452`), which
is dark (`settings.py:478`). RSI penalties `:307-310`, vol penalty `:312-313`. `screener.py`
is 759 lines, `autonomous_loop.py` 3,752. **Re-derive before citing again** — these numbers
have now been wrong twice.

1. **The declared weights are not the weights in force.** No cross-sectional standardisation
   exists on the live path, so raw returns of three horizons are summed as if commensurate and
   the 6m term's effective weight far exceeds its nominal 0.25. **Reweighting before
   standardising tunes a dead knob.** Caveat: the "~√6 ≈ 2.4x" dispersion ratio is IID theory,
   never measured on our universe — the *direction* is robust, the *magnitude* is an assumption.

2. **Both arXiv IDs cited in our own code for the diversity mitigations are non-equity papers**
   — `settings.py:488` → arXiv 2601.08717 (Garcia & Messud, HHI on *"synthetic data (energy
   assets)"*); `autonomous_loop.py:178` → arXiv 2408.09168 (an **Amazon Music** learning-to-rank
   paper). Both support the *mechanism*; neither is financial validation. Fidelity gap: MB's
   guarantee comes from STOCHASTIC sampling `c∼M(p)`; `_min_k_sector_slice` is DETERMINISTIC
   top-k-by-peak and does not inherit it.

3. **Gârleanu & Pedersen reframe the question:** *"predictors with slower mean reversion (alpha
   decay) get more weight in the aim portfolio"* — low turnover is the CORRECT behaviour for a
   1/3/6-month signal. The defect is not "turnover too low", it is "the signal menu contains
   nothing fast". Every proposed overlay is another slow signal.

4. **CLOSED (was "unresearched"): residual / idiosyncratic momentum.** arXiv has none of this
   literature — that is why the v1 arXiv-only search found nothing; **WebSearch, not arXiv, is
   the tool for finance factor work.** Blitz-Huij-Martens 2011 is still SSRN-paywalled, but
   Hanauer & Windmüller (`wp.lancs.ac.uk/mhf2019/.../MHF-2019-076-Matthias-Hanauer.pdf`)
   reproduce it on 88 years of US data. Signal = 12-2 cumulative FF3 residual / SD of those
   residuals, 36-month rolling regression, monthly rebalance; a **market-factor-only** variant
   captures *"most of the performance improvement"* (cheap enough for us — no SMB/HML needed).
   **The honest framing is "a better-behaved momentum signal, not a bigger one":** CXO Advisory
   reports gross return LOWER (1.39 vs 1.54%/mo); the Sharpe edge is entirely halved volatility,
   which a long-only screener that never shorts and never vol-targets may not capture at all.

5. **TURNOVER CONVENTION TRAP — a summed figure compared to a per-side threshold inverts the
   answer.** H&W report iMOM turnover 65.32%/mo defined as *"the long leg **plus** the short
   leg"*; Novy-Marx & Velikov's ~50% survival line is *"average over the long and short side"*,
   i.e. PER SIDE. Normalised (÷2), iMOM ≈ 32.7% — inside N-M&V's independently measured 14-35%
   momentum band, NOT a high-turnover strategy. H&W's own paper makes this comparison error in
   three consecutive sentences. **Always read both papers' definitions before comparing two
   turnover numbers.**

**How to apply:** prefer slate-composition levers (`_min_k_sector_slice`, which leaves
`composite_score` untouched so DSR/PBO still measure the signal) over score-mutating ones
(`_apply_soft_sector_diversity` overwrites `composite_score`, contaminating the metric the gates
read). The distinction that actually matters is **signal-vs-penalty**, not score-vs-slate: a
better score is legitimately gate-measurable; an arbitrary diversity penalty is not. Also: the
eight overlay slices at `autonomous_loop.py:749,769,833,860,884,910,938,967` read
`screen_data[:2*paper_screen_top_n]` from an **UNSORTED** `screen_universe` return (`:246`, no
sort anywhere before it), so an overlay can never be an entry path — a new signal must enter the
composite at `:299-305`, not become a ninth overlay. **Step 86.59 is BLOCKED behind 86.69** (81.2%
of analyses persist as an empty placeholder scored 0.0/HOLD): no ranking change can pay off first.

**Rail lesson (cost me two failed gates).** A born-inert envelope's counts go **STALE** when a run
drops mid-way, and the wreck is *internally contradictory*, not merely incomplete: the seed said
`external_sources_read_in_full: 6` while the table below it had **10** rows, `snippet_only: 0`/
`urls_collected: 0` were never filled, and the tail still carried v1's **"30 URLs"** — the exact
over-claim that had already failed the gate once. A re-run must **re-derive every count from the
file on disk under a stated rule** (and run a second, differently-terminated regex as a control),
verify each claimed source has ≥2 content markers in the prose rather than just its URL, and only
then flip the marker. Flipping `brief_status` without re-deriving would have re-shipped v1's
over-claim under a COMPLETE banner. See [[research-gate-discipline]] and
[[websearch-budget-is-session-shared]].
