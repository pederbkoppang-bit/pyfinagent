# experiment_results -- step 82.2

**GENERATE phase.** Contract: `handoff/current/contract.md`.
Research: `handoff/current/research_brief_82.2.md` (gate_passed=true, 6 read in
full, 21 URLs, 6 internal files).

## What was built

Three FORWARD-LOOKING, cost-adjusted, sigma-scaled label methods added to
`STRATEGY_REGISTRY` (`backend/backtest/backtest_engine.py:32`), covering the
four overpriced-market lenses:

| strategy | lens | rule |
|---|---|---|
| `stretch_regime` | (a) regime gate + **(d) cash-timing overlay folded in** | symmetric sigma barriers MODULATED by market turbulence (trailing SPY realised vol vs its own long-run average). As stretch rises the up-barrier widens and the down-barrier tightens -> fewer +1 -> the model buys less and holds cash by being harder to convince |
| `qarp` | (b) quality-at-reasonable-price, long-only | gate on `pe_ratio`/`roe`/`debt_equity`/`profit_margin` at entry; **non-candidates return `None`, not 0**; survivors get a defensively asymmetric sigma barrier (TP 1.0σ, SL 1.5σ) |
| `reversion_sigma` | (c) mean-reversion on overextension | stretch measured as `sma_50_distance / σ` (a z-score) instead of fixed fractions; **no-signal returns `None`**; cost-adjusted, which the existing label is not |

Helpers: `_sigma_barriers` (de-annualises `annualized_volatility`, applies the
round-trip cost), `_market_stretch` (SPY turbulence proxy, strictly trailing),
`_walk_barriers` (forward walk).

**No live-funnel code was touched.** Only `backtest_engine.py` plus new tests
and a fixture.

## Verification command output (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_2_candidate_strategies.py -q
..................                                                       [100%]
18 passed in 8.03s
```

## MEASURED label distributions (committed seeded fixture, 10 tickers x 88 dates = 880 rows)

```
strategy           labelled   distribution                    top class
stretch_regime          880   {-1: 315, 0: 304, 1: 261}          35.8%
qarp                    528   {-1:  66, 0: 244, 1: 218}          46.2%
reversion_sigma         409   {-1:  87, 0: 196, 1: 126}          47.9%
mean_reversion (old)    880   {-1:   9, 0: 836, 1:  35}          95.0%
```

All three candidates clear the <=95% non-degeneracy bar with wide margin and
emit BOTH directions. `qarp` labels 528 of 880 because 352 rows fail the QARP
gate and are EXCLUDED; `reversion_sigma` labels 409 because 471 rows show no
overextension and are EXCLUDED. That exclusion is the design, not a shortfall.

## The `mean_reversion` degeneracy: MEASURED, not asserted

The "~all-neutral, trades 0" report is **confirmed at 836/880 = 95.0% neutral**,
exactly at the degeneracy boundary. The mechanism is visible in the row counts:
the existing label returns a value for EVERY row (880/880) because it emits 0
rather than `None` on no-signal, so the neutral class is flooded by rows that
were never candidates. `reversion_sigma` on the identical fixture labels 409
rows at 47.9% top class.

CORRECTION to Main's earlier hypothesis: the raw-percent thresholds are NOT a
unit bug. `sma_50_distance` is a fraction (`historical_data.py:105`), so -0.05
and +0.10 mean what they appear to mean. The design critique that survives is
narrower -- a fixed fraction is not volatility-scaled, so it means different
things across the cross-section -- and that is what `reversion_sigma` fixes.

## Seed-robustness: is the fixture RIGGED?

Main re-seeded the fixture and re-measured, because a committed fixture tuned to
pass is the obvious way this criterion could be gamed. Top-class share:

```
    seed   stretch_regime      qarp   reversion_sigma   mean_reversion
     822   35.8% (n=880)   46.2% (528)   47.9% (409)    95.0% (880)
       1   36.1% (n=880)   51.9% (528)   44.9% (376)    94.1% (880)
       7   35.8% (n=880)   53.6% (528)   48.6% (399)    93.6% (880)
   12345   37.5% (n=880)   50.4% (528)   48.9% (419)    94.1% (880)
   99999   38.1% (n=880)   51.3% (528)   44.0% (357)    93.4% (880)
```

All three candidates stay far below the 95% bar on every seed; the incumbent
stays degenerate on every seed. The result is a property of the label logic,
not of the committed seed.

**This exposed a brittle assertion of Main's own and it was FIXED.** The
degeneracy test asserted `top >= 0.95`, which holds only because the committed
seed lands at exactly 95.0%. On four other seeds it is 93.4-94.1%, so the test
would have failed while claiming mean_reversion is "not degenerate" at 93.4% --
which it plainly still is. The bar is now 90%, which is the finding that
actually survives re-seeding.

## Forward-information sweep across the WHOLE registry

Destroying the post-entry price path and re-labelling:

```
strategy          label changes when the future is destroyed?
triple_barrier    (stub lacks tp_pct/sl_pct -- not evaluated)
quality_momentum  NO   <-- carries no forward information
mean_reversion    YES
factor_model      NO   <-- carries no forward information
meta_label        (stub lacks tp_pct/sl_pct -- not evaluated)
stretch_regime    YES
qarp              YES
reversion_sigma   YES
```

This independently reproduces the step-82.16 finding by EXECUTION rather than by
reading: `quality_momentum` and `factor_model` are unchanged when the future is
destroyed, so a model trained on them learns a deterministic function of its own
inputs. All three candidates are forward-looking.

## Guard integrity

- Non-degeneracy is asserted together with a **minimum row count**
  (`MIN_LABELLED_ROWS = 200`); a 3-row label set would otherwise pass the 95%
  bar trivially.
- `test_no_candidate_aliases_an_existing_label_method` blocks the shape where a
  "new" strategy merely re-points at an existing method (as `meta_label` does).
- `test_candidate_produces_both_directions` blocks a set that passes the 95% bar
  while being unable to express a downside view.
- `test_market_stretch_is_backward_looking_only` asserts the regime proxy
  requests prices only through `entry_date`.
- `test_reversion_sigma_fixes_the_degeneracy_it_replaces` compares candidate vs
  incumbent on the SAME fixture, so a rename cannot pass.

## Scope honesty

- No backtest was RUN. That is 82.3, which is blocked on **82.15** (the
  `realtime_start` vintage column still has zero consumers, so a
  macro-conditioned backtest carries ~120d look-ahead).
- `stretch_regime` uses SPY prices rather than macro features, which sidesteps
  the vintage problem for THIS step but does not remove 82.3's dependency.
- `compute_turbulence_index` (`historical_data.py:281`, zero callers) was
  deliberately NOT used: it needs the universe cross-section, which is not
  reachable from a label method, and wiring it would mean editing `_run_window`.
- The fixture is synthetic. It proves the labels are well-formed and
  non-degenerate; it says NOTHING about whether these strategies make money.
  That is exactly what 82.3 is for.


## CYCLE 1 Q/A: PASS -- five advisories, four closed in-cycle

Verdict PASS, 0 violated criteria. The Q/A re-derived every number
independently (exact reproduction), re-seeded the fixture five ways AND widened
the grid 5x to 4400 rows (candidates stayed 35.8-53.6% top class), and verified
the overlay END-TO-END with the UNPATCHED production `_market_stretch`: it spans
0.763-2.490 across the sample and the +1 rate falls **40.3% (calmest third) ->
17.0% (most turbulent third)**. It also re-ran the forward-information probe
with its OWN mutation shape (a fresh random walk rather than Main's 0.5x
collapse): stretch_regime 563/880 rows changed, qarp 297/880, reversion_sigma
226/880, while quality_momentum and factor_model changed **0/880** -- an
independent execution-level confirmation of the 82.16 defect.

It also classified the 32 full-tree failures rather than hand-waving them: it
re-ran the failing files in a HEAD-equivalent process (new registry keys and
methods deleted at runtime) and the SAME tests still failed, across 22 unrelated
files from phases 23/50/57/60/61/64/70/75. Zero failures in any
backtest/label/strategy test.

| # | advisory | disposition |
|---|---|---|
| A1 | surviving mutant: `_market_stretch` -> constant 1.0 left all tests green (the overlay test patches it out) | **FIXED** -- added `test_market_stretch_actually_varies_across_the_sample` (asserts a >=2x spread on the UNPATCHED method) and `test_overlay_changes_exposure_without_patching_market_stretch` (real calm-third vs turbulent-third). M1 now kills 2 tests |
| A2 | surviving mutant: zeroing the round-trip cost left all green, though cost-adjustment is claimed prominently | **FIXED** -- `test_sigma_barriers_include_the_round_trip_cost`. M3 now killed |
| A3 | surviving mutant: `qarp` returning 0 instead of None for non-candidates left all green | **FIXED** -- `test_qarp_excludes_non_candidates_rather_than_labelling_them_neutral`. M4 now killed |
| A4 | knife-edge assertion: `top >= 0.95` passes on an exact 836/880 tie and would go red on any other seed | **ALREADY FIXED** by Main before the verdict arrived, independently, from the same seed-robustness sweep. Bar is now 90% |
| A5 | `IMPLEMENTED_STRATEGY_IDS` (`archetype_library.py:31`) is a hand-maintained "mirror" of STRATEGY_REGISTRY and has drifted in BOTH directions (lists `blend`, which is not in the registry; omits all three new candidates) | **QUEUED as 82.17** -- the durable fix is to DERIVE the set, not to correct the literal and leave the same trap |

Mutation re-proof after the fixes (in-tree, restored from backup, 0 MUTANT
markers remaining):

```
M1 _market_stretch -> constant 1.0   -> 2 failed  (was: all green)
M3 round-trip cost -> 0.0            -> 1 failed  (was: all green)
M4 qarp None -> 0                    -> 1 failed  (was: all green)
RESTORED                             -> 22 passed
```

Suite is now 22 tests (was 18).
