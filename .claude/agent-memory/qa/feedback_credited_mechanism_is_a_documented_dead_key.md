---
name: credited-mechanism-is-a-documented-dead-key
description: A claim citing a config parameter as its mechanism must be checked by grepping the parameter's READER, not its writer; 86.116 credited vol_barrier_multiplier, which the repo itself lists in _DEAD_KEYS
metadata:
  type: feedback
---

When an artifact explains an effect through a **named configuration parameter**
("X sets `barriers = daily_vol * vol_barrier_multiplier`, so a depressed vol
scales every barrier"), grep for the parameter's **READER**. A writer proves
nothing: `engine._strategy_params[k] = v` is a write into a dict, and a dict
accepts keys nobody consumes.

**Why:** phase-86.116 (2026-08-18) made `vol_barrier_multiplier` the quantified
mechanism for criterion 6 and published "measured barrier-width scale 0.7995"
against a 1/sqrt(2) closed form. Measured: the parameter has ZERO readers
repo-wide. `backend/autoresearch/rotation_runner.py` lists it **by name** in a
tuple called `_DEAD_KEYS` with the comment "NO engine reader (reverted in
9fbd9cd6) ... nothing reads them today". The literal
`barriers = daily_vol x multiplier` exists only as a **comment** in
`quant_optimizer.py` describing a search-space bound; the real label function
uses fixed `self.tp_pct`/`self.sl_pct` with no volatility term. The same file,
five lines below the comment relied on, already documented four OTHER params
deleted for exactly this reason -- and a third comment there names
`_compute_vol_target_scale` as a reader, a function that does not exist.
**Stale "read by ..." comments cluster**: one dead pointer in a file predicts more.

Two traps this instance teaches:

1. **The number can be real while the label is an inference.** The script
   computed `vol_ratio = vol_pre / vol_post` and returned the SAME float under
   both `vol_ratio_pre_over_post` and `barrier_width_scale`. The ratio reproduced
   exactly; only the second name was an unjustified claim about dead code.
2. **The live mechanism is often adjacent to the dead one.** `daily_volatility`
   (the dead feed, historical_data.py:132) sits four lines below
   `annualized_volatility` (:128), which flows engine:1251 -> signal dict ->
   `backtest_trader.size_position` -> `vol_scale = min(target_vol/stock_vol, 3.0)`
   -- inverse-volatility POSITION SIZING, a bigger money-path effect than a label
   boundary. So the step's conclusion was right and understated while its
   mechanism was wrong. Trace every hop and name the live one.

**How to apply:** for any mechanism claim naming a parameter, feature key, or
flag: (a) grep the token repo-wide excluding `.venv`; (b) classify each hit as
bound / write / read / comment / test; (c) if there is no READ, the mechanism does
not execute -- `Unjustified_Inference`, and say which hops you traced. Check for a
`_DEAD_KEYS`-style list before concluding; this repo maintains one. Related:
[[check-the-attribution-not-just-the-count]],
[[a-correct-observation-can-credit-the-wrong-mechanism]],
[[a-later-step-bolts-a-mode-on-with-no-guard]].
