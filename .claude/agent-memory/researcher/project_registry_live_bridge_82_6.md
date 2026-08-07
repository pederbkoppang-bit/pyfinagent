---
name: registry-live-bridge-82-6
description: Phase-82.6 registry-to-live bridge -- BOTH step premises refuted; entry_strategy is a live strategy-name branch that is NULL-inert; the selector already exists and is dark
metadata:
  type: project
---

# Registry-to-live selection bridge (82.6 research gate, 2026-08-06)

Both premises in the step's MEASURED claim were wrong, and the second error is
a live risk-control path.

**Why:** the step name asserted "the registry's EXIT PARAMS cross over; its
SELECTION LOGIC does not". Its source, `docs/strategy/incumbent_live_strategy_spec.md:35`,
actually said exit params cross over *"as display fields"* -- the masterplan
compressed the sentence and dropped the qualifier, turning a telemetry claim
into a behavioural one.

**How to apply:** on any step about "what crosses into the live path", trace the
loaded dict to its LAST reader, not its first. Then sweep for the same *value
domain* (here: strategy names) in the OTHER live services -- the crossing may
exist somewhere the step never looked.

## The findings that generalise

- **A summary-dict write is not a crossing.** `best_params` had exactly 5
  references; `summary["strategy_params"]` had ZERO readers repo-wide (frontend
  included). "Goes into the cycle summary" reads like adoption and means the
  opposite. Live exits came from `settings.paper_default_stop_loss_pct` + the
  Risk Judge instead.
- **The headline was in a file the step did not name.** `paper_trader.py:1427`
  branches on `entry_strategy in {"mean_reversion","pairs"}` and RETURNS EARLY,
  skipping the trailing stop. `mean_reversion` is a STRATEGY_REGISTRY key. It is
  inert ONLY because the column is NULL for every row -- a data fact, not a code
  fact, so no amount of code reading proves inertness. **Query the column.**
- **An inert branch is a loaded gun, not dead code.** The instant something
  populates `entry_strategy`, a risk control silently stops firing. Class:
  a behaviour change disguised as a metadata write.
- **A criterion can be green while the risk lands elsewhere.** Criterion 3
  scoped the test to `autonomous_loop.py`; the dangerous path is in
  `paper_trader.py`. Criteria are immutable -- note the gap, test wider anyway.
- **The design already existed and was dark.** `strategy_selector.py` (phase-47.6)
  is a complete, tested selector whose docstring already specifies the bridge;
  step 48.3's own name says "deployment bridge DEFERRED". Search the repo for
  the design before designing.
- **Gates fail CLOSED on absent inputs, which looks like a working gate.**
  `optimizer_best.json` has no `pbo` key at all and `PromotionGate` rejects on
  missing pbo -- so nothing could ever promote. Meanwhile `promoter.py:134`
  defaults a missing pbo to `0.0`, which PASSES. Same field, opposite failure
  modes, two files apart.
- **Verify a docstring's citation against the primary source.** The
  "141% -> 44% turnover" in `strategy_selector.py:9-11` checked out exactly
  against Shu/Yu/Mulvey (2024). Worth confirming rather than assuming rot.

## Test-predicate traps (for the criterion-3 style guard)

- `STRATEGY_REGISTRY` has 6 keys but only 5 distinct method values
  (`meta_label` shares `triple_barrier`'s) -- `len(set(values)) == len(keys)`
  FAILS on correct code. Use a floor, not equality.
- `autonomous_loop.py` legitimately imports `backend.backtest.universe_lists`
  and `backend.backtest.markets` -- a blanket "no `backend.backtest` import"
  assertion is a false positive. Forbid `backtest_engine` specifically.
- The registry's values are `_compute_*_label` -- distinctive, so a literal
  value sweep is safe here (unlike the 82.59 common-verb sweep). Sweeping the
  KEYS is not: `triple_barrier` legitimately appears in the live params.
- `perf_metrics.py:151` names `backtest_engine` in a COMMENT -- package-wide
  text sweeps need comment handling.

Related: [[project_trial_pool_composition_82_46]], [[project_pbo_level_and_dead_gate_82_27]],
[[project_dsr_trial_count_reset_82_25]], [[project_strategy_rotation_infra]].
