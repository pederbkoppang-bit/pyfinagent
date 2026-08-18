---
name: slice-vs-entry-path-86-60
description: 86.60 — screen_universe returns UNSORTED so the overlay slice is ALPHABETICAL; news/PEAD are already entry paths; past slices are unreconstructable
metadata:
  type: project
---

**The eight overlay slices are the ALPHABETICALLY-first ~20 names, not the top-20 ranked.**
`screen_universe` appends in `for ticker in tickers` order (`screener.py:147`→`:240`) and
`return results` at `:246` with **no sort** — the module's only `sorted(` is at `:515`, inside
`rank_candidates`. Universe order is the Wikipedia S&P 500 table (`screener.py:56`), i.e.
alphabetical by symbol, with intl appended (`autonomous_loop.py:657`). Nothing between the screen
call (`:715`) and the first slice (`:756`) reorders it.

**The repo already has BOTH architectures, split by accident.** A signal is an ENTRY PATH iff its
producer takes NO ticker argument, because `rank_candidates` sorts the FULL `screen_data` and
truncates (`screener.py:491-492`). PEAD (`autonomous_loop.py:565-569`) and news (`:575-581`) take
none → they can promote from anywhere. The eight overlays (`:756, :776, :840, :867, :891, :917,
:945, :974`) all slice `screen_data[: 2 * settings.paper_screen_top_n]` → they can only re-score
those 20. All thirteen then pass through the SAME `rank_candidates` call, so the difference is
invisible at the call site.

**Live state (2026-08-17):** the two flags that are ON are exactly the two entry paths — so the
alphabetical defect is **latent, not active**, and activates the moment any of the eight is enabled.

**Why:** 86.60's premise was "overlays can't promote a ticker." True — but the reason is the
UNSORTED return, not the slice width, and the fix is not "widen the slice" (arXiv:2601.04618v1
measures naive candidate expansion at −3.3 to −5.8 nDCG@10; only reward-guided expansion helps).

**How to apply:** before proposing a slice change, check whether the cheap producers can just go
full-universe (ma_preannounce at `:756` is explicitly "Pure compute — no extra fetches"). Sorting
`screen_data` by the already-free composite score before slicing converts an arbitrary first stage
into a relevance-ordered one — see [[unmeasurable-running-flags-86-60]] for what you can and
cannot observe while doing it. Related: [[research-gate-depth-86-73]].
