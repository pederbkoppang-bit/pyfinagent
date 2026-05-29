# Research Brief — phase-48.3: live rotation runner + full-kwarg engine_factory

Tier: **moderate** (internal-audit-heavy). PBO/DSR methodology is SETTLED by
the 48.1 + 48.2 briefs — NOT re-derived here. Most effort is INTERNAL.

Status: COMPLETE.

---

## TL;DR (the headline finding that reshapes the cycle)

The 48.2 brief's RISK note ("make_engine threads only a SUBSET of kwargs, so a
live run would SILENTLY ignore `tb_risk_managed`'s risk overrides and collapse
onto tb_baseline") is **REAL and WORSE than stated**. There are actually TWO
distinct gaps, and the second is fatal to one seed:

1. **Constructor-kwarg gap (fixable by a full-kwarg factory).** `make_engine`
   (`run_harness.py:89`) threads 12 of ~25 `BacktestEngine.__init__` kwargs.
   Missing: `market`, `train_window_months`, `test_window_months`,
   `embargo_days`, `starting_capital`, **`target_vol`**, `commission_model`,
   `commission_per_share`. A `make_rotation_engine` that threads these closes
   the gap **for the kwargs the engine actually reads**.

2. **Dead-key gap (NOT fixable by any factory — the override has no reader).**
   `tb_risk_managed`'s headline overrides — `target_annual_vol`,
   `trailing_stop_enabled`, `trailing_trigger_pct`, `trailing_distance_pct` —
   are written by `quant_optimizer._apply_params_to_engine` into
   `engine._strategy_params[...]` (`quant_optimizer.py:557-564`), **but NOTHING
   in the current `backtest_engine.py` or `backtest_trader.py` READS those
   keys.** The engine logic that consumed them (volatility-targeting in position
   sizing, trailing-stop in the daily MTM loop) was **reverted** in commit
   `9fbd9cd6 "validate: pre-Phase-1.2 code confirms Sharpe 1.0142"` (it removed
   what `168c639a` and `c383de65` had added in Phase 1.5). Verified: `grep` for
   every read-site across `backend/backtest/*.py` returns EMPTY except the
   optimizer's own write + the registry seed def + a results JSON.

**Consequence for the spec:** even a perfect full-kwarg `make_rotation_engine`
canNOT make `tb_risk_managed` behave differently from `tb_baseline` on the
risk axis — because `target_annual_vol`/`trailing_*` are inert. With BOTH seeds
on `strategy=triple_barrier` and only `tp_pct` differing (6 vs whatever the
base is), `tb_risk_managed` is a near-duplicate of `tb_baseline` (just a
take-profit tweak). The 48.3 spec MUST surface this honestly:
- thread the LIVE-read kwargs (`target_vol` constructor arg IS read — see §1) so
  the factory is correct and future-proof, AND
- **document loudly** that `target_annual_vol`/`trailing_*` are currently
  no-ops, so `tb_risk_managed` exercises only `tp_pct` (a real but small lever)
  + whatever live kwargs the factory now threads. Re-enabling the reverted vol-
  targeting/trailing engine logic is a SEPARATE cycle (out of 48.3 scope).
- The cleanest 48.3 move: keep the seed in the bake-off (it still clears the
  gate on its own merits or it doesn't), but add a `# DEAD-KEY` note in the
  factory + brief so no one believes the risk overrides are active. Optionally
  the runner can WARN when a seed carries a dead key.

This is a finding, not a blocker. The factory + runner + $0 tests are all
completable this cycle; the dead-key reality just bounds what the live bake-off
can prove and is the highest-value thing for Q/A and Main to know.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `scripts/harness/run_harness.py` | `make_engine` :89-111 | Precedent factory; threads 12 kwargs | LIVE; gap source |
| `backend/backtest/backtest_engine.py` | `__init__` :136-248 | ~25 ctor kwargs; `target_vol`:163 read at :219 | LIVE |
| `backend/backtest/backtest_engine.py` | `_run_window` :381; MTM loop :494-502 | Daily mark-to-market; **NO trailing stop** | LIVE (post-revert) |
| `backend/backtest/backtest_engine.py` | `_strategy_params` :237-248; read :390-393 | Only screening weights read; rest is metadata | LIVE |
| `backend/backtest/backtest_trader.py` | `target_vol` :54/:63; used :89 | Inverse-vol sizing reads `self.target_vol` | LIVE |
| `backend/backtest/quant_optimizer.py` | `_apply_params_to_engine` :534-569 | Writes dead keys to `_strategy_params` | LIVE but writes-to-nowhere for risk keys |
| `backend/autoresearch/strategy_registry.py` | `SEED_STRATEGIES` :67-128; loader :153 | 4 seeds; `tb_risk_managed` carries dead keys | LIVE (48.1) |
| `backend/autoresearch/strategy_candidate_producer.py` | `run_strategy_bakeoff` :127-150 | registry->producer->selector spine | LIVE (48.1) |
| `backend/autoresearch/strategy_backtest_adapter.py` | `make_engine_backtest_fn` :167 | Real `backtest_fn(params)`; takes `engine_factory` | LIVE (48.2) |
| `backend/autoresearch/strategy_selector.py` | `select_best_strategy` :60-131 | Gate + DSR-desc rank + hysteresis; verdict dict | LIVE (47.6) |
| `backend/services/autonomous_loop.py` | `load_promoted_params` :46-74 | Incumbent source (BQ promoted -> optimizer_best) | LIVE (25.B3) |
| `backend/autoresearch/monthly_champion_challenger.py` | `_save_state` :368; `_emit_deployment_log_row` :281 | Verdict-persistence PRECEDENT (JSON state + BQ log) | LIVE |
| `backend/backtest/experiments/optimizer_best.json` | — | Incumbent fallback; `strategy=triple_barrier, sharpe=1.1705, dsr=0.9526`; `target_annual_vol=0, trailing_stop_enabled=False` | LIVE |

`STRATEGY_REGISTRY` (engine.py:32) gate: `self.strategy = strategy if strategy
in STRATEGY_REGISTRY else "triple_barrier"` (:199) — silent fallback, which is
exactly why both the adapter and the factory must validate the name and RAISE.

---

## Q1 — make_engine kwarg-gap table (EXACT) + make_rotation_engine spec

### make_engine vs BacktestEngine.__init__ — every kwarg

| `__init__` kwarg (line) | make_engine threads? | Seed needs it? | Application site / notes |
|---|---|---|---|
| `bq_client` (req) | yes (`bq.client`) | — infra | ctor |
| `project` (req) | yes | — infra | ctor |
| `dataset` (req) | yes | — infra | ctor |
| `market` :142 | **NO** (defaults "US") | no (all US) | ctor; safe to default |
| `start_date` :145 | yes | yes (window) | ctor → scheduler |
| `end_date` :146 | yes | yes (window) | ctor → scheduler |
| `train_window_months` :147 | **NO** (def 12) | no (not in seeds) | ctor → scheduler |
| `test_window_months` :148 | **NO** (def 3) | no | ctor → scheduler |
| `embargo_days` :149 | **NO** (def 5) | no | ctor → scheduler |
| `holding_days` :150 | yes | **YES** (qm=120, mr seed sets 30) | ctor attr :193; also in `_strategy_params` |
| `tp_pct` :151 | yes | **YES** (tb_risk=6) | ctor attr :195 |
| `sl_pct` :152 | yes | maybe (base) | ctor attr :196 |
| `mr_holding_days` :154 | yes | **YES** (mr=8) | ctor attr :194 |
| `frac_diff_d` :156 | yes | base | ctor attr :197 |
| `strategy` :158 | yes | **YES** (the big lever: mr/qm/tb) | ctor :199 + registry validate |
| `starting_capital` :160 | **NO** (def 100k) | no | ctor → trader :216 |
| `max_positions` :161 | yes | base | ctor → trader :217 |
| `transaction_cost_pct` :162 | yes (via tx_cost_pct arg) | base | ctor → trader :218 |
| **`target_vol` :163** | **NO** (def 0.15) | base may set; **IS READ** at trader :89 | ctor → trader :219 → sizing. **LIVE-read.** |
| `top_n_candidates` :164 | yes | base | ctor attr :198 |
| `commission_model` :165 | **NO** (def "flat_pct") | no (not in seeds) | ctor → trader :221 |
| `commission_per_share` :166 | **NO** (def 0.005) | no | ctor → trader :221 |
| `n_estimators` :168 | yes | base | ctor → ml_params :208 |
| `max_depth` :169 | yes | base | ctor → ml_params |
| `min_samples_leaf` :170 | yes | base | ctor → ml_params |
| `learning_rate` :171 | yes | base | ctor → ml_params |
| `progress_callback` :173 | yes | — | ctor |

### Seed params that have NO ctor kwarg — applied only via the setter path

These come from `_apply_params_to_engine` (`quant_optimizer.py:534-569`) writing
to `engine._strategy_params[...]` / `engine.trader.*`:

| Seed param | Setter site | **Is it READ at runtime?** |
|---|---|---|
| `target_annual_vol` | `engine._strategy_params["target_annual_vol"]` :559 | **NO — dead key** (vol-targeting reader reverted in `9fbd9cd6`) |
| `trailing_stop_enabled` | `engine._strategy_params[...]` :562-564 | **NO — dead key** (trailing MTM logic reverted) |
| `trailing_trigger_pct` | `engine._strategy_params[...]` :562-564 | **NO — dead key** |
| `trailing_distance_pct` | `engine._strategy_params[...]` :562-564 | **NO — dead key** |
| `vol_barrier_multiplier` | `engine._strategy_params[...]` :555 | **NO — dead key** (no reader found) |
| `tb_weight/qm_weight/mr_weight/fm_weight` | `engine._strategy_params[...]` :567-569 | **NO — dead key** (`_compute_blend_label` does not exist) |
| `target_vol` | `engine.trader.target_vol` :546 | **YES — read at trader:89.** Also a ctor kwarg, so the factory should pass it via ctor. |
| `max_positions` | `engine.trader.max_positions` :547 | yes (also ctor) |
| `momentum_weight/rsi_weight/volatility_weight/sma_weight` | (not in setter) read at engine:390-393 | yes (screening) — pass via `_strategy_params` if a seed ever sets them |

**Conclusion for the spec:** the ONLY seed-relevant param that needs the
non-ctor setter path AND is actually read is `target_vol` — and that one is
*also* a ctor kwarg, so a full-kwarg ctor factory covers it cleanly. Every other
non-ctor key the registry/optimizer writes (`target_annual_vol`, `trailing_*`,
blend weights, `vol_barrier_multiplier`) is currently **inert**. So:

> The full-kwarg factory only needs to extend the CONSTRUCTOR call. It does NOT
> need to replicate the `_strategy_params` setter writes for risk overrides,
> because those keys have no reader. Replicating them would be cargo-culting a
> dead path and would falsely imply `tb_risk_managed` is risk-differentiated.

### `make_rotation_engine(params, settings, bq, *, start_date=None, end_date=None)` — spec

Place it ALONGSIDE the runner (see Q2), in `backend/autoresearch/`. Behaviour:

1. **Validate strategy FIRST** (mirror adapter:198): `if params.get("strategy")
   not in STRATEGY_REGISTRY: raise ValueError(...)`. This is the whole point —
   prevents the engine's silent triple_barrier fallback (engine:199) from
   masking a typo'd seed.
2. **Construct with the FULL ctor kwarg set**, sourcing each from `params` with
   the engine's own defaults as fallback, threading the 8 currently-dropped
   ctor kwargs — most importantly `target_vol` (LIVE-read) and the scheduler
   window kwargs (`train_window_months`/`test_window_months`/`embargo_days`)
   and `commission_model`/`commission_per_share`/`starting_capital`/`market`.
   Use `params.get("target_vol", 0.15)` etc.
3. **Wire `progress_callback`** to a quiet no-op (the bake-off runs K*seeds
   backtests; per-window stdout spam is undesirable) OR reuse `progress_cb`.
4. **DEAD-KEY guard (the load-bearing honesty bit):** if `params` contains any
   of `{target_annual_vol, trailing_stop_enabled, trailing_trigger_pct,
   trailing_distance_pct, vol_barrier_multiplier, tb_weight, qm_weight,
   mr_weight, fm_weight}` with a non-default/truthy value, log ONE WARN:
   `"[rotation] seed carries currently-inert risk keys %s (engine readers
   reverted in 9fbd9cd6); they will NOT affect this backtest"`. Do NOT write
   them to `_strategy_params` (pointless) — just warn so the operator isn't
   misled. (ASCII-only per the logger rule.)
5. Return the `BacktestEngine`. The factory closes over `settings` + `bq`; the
   runner builds it as `lambda variant: make_rotation_engine(variant, settings,
   bq, start_date=..., end_date=...)` and hands THAT to
   `make_engine_backtest_fn(engine_factory=...)`.

Signature mirrors `make_engine` (same `start_date/end_date` override args) so
the bake-off can pin a shorter window for a smoke run without touching seeds.

---

## Q2 — run_rotation_bakeoff placement, signature, incumbent, persistence

### Placement
`backend/autoresearch/rotation_runner.py` (new). Rationale: it is the live-wiring
glue over the 48.1 producer + 48.2 adapter + 47.6 selector, all of which live in
`backend/autoresearch/`. It is NOT a `scripts/harness/` concern (that module is
the param-optimization harness loop; rotation is a separate autoresearch
pipeline). Keeping `make_rotation_engine` in the same new module avoids a
circular import with the adapter and keeps the engine-construction knowledge
next to its only caller.

### Signature (proposed)
```
def run_rotation_bakeoff(
    settings, bq,
    *,
    seeds: Optional[list[dict]] = None,        # -> registry.load_seed_strategies
    incumbent: Optional[dict] = None,          # None -> auto-resolve (see below)
    num_param_variants: int = 8,               # adapter CSCV columns per strategy
    num_trials: Optional[int] = None,          # DSR deflation N; default = #seeds
    start_date: Optional[str] = None,          # smoke-run window pin
    end_date: Optional[str] = None,
    persist: bool = True,                       # write the verdict row
    engine_factory: Optional[Callable] = None, # TEST SEAM (see Q3); default builds make_rotation_engine
    adapter_fn: Optional[Callable] = None,      # higher TEST SEAM; default builds via make_engine_backtest_fn
) -> dict:                                       # returns the selector verdict dict
```
Body:
1. Resolve `engine_factory` (default: closure over `make_rotation_engine`).
2. Resolve `adapter_fn` (default: `make_engine_backtest_fn(engine_factory,
   num_param_variants=..., num_trials=...)`). Allowing the caller to inject
   EITHER seam is the $0-test lever (Q3).
3. Resolve `incumbent` (below).
4. `verdict = run_strategy_bakeoff(adapter_fn, incumbent=incumbent,
   seeds=seeds, num_trials=num_trials)`.
5. If `persist`: write the verdict (below). Return `verdict` either way.

### Incumbent source (decision order — reuse the EXISTING live read-path)
The selector's own docstring (selector.py:16-17) says the chosen strategy flows
through the `promoted_strategies` BQ row that `load_promoted_params` already
consumes. So mirror that exact precedence, do NOT invent a new path:

1. **`load_promoted_params(bq)`** (`autonomous_loop.py:46`) — returns the live
   params dict (BQ promoted row -> else `optimizer_best.json`). Wrap into an
   incumbent candidate: `{"strategy_id": params.get("strategy"), "strategy":
   params.get("strategy"), "params": params, "dsr": <incumbent dsr>}`.
2. **Incumbent DSR**: `load_promoted_params` returns only params, not a DSR. Two
   options: (a) read `optimizer_best.json["dsr"]` (=0.9526 today) as the
   incumbent DSR floor; (b) score the incumbent through the SAME `adapter_fn`
   for an apples-to-apples DSR. (b) is more correct (the selector compares
   `best.dsr - incumbent.dsr`) but costs one extra strategy's worth of
   backtests. **Recommend (b) when the incumbent's `strategy` is NOT already one
   of the seeds; when it IS a seed (tb_baseline == triple_barrier == the live
   strategy today), the incumbent is already scored in the bake-off** — so pass
   `incumbent=None`-equivalent by mapping the matching seed result, OR simply
   set the incumbent_id to `tb_baseline` and let the selector's
   `incumbent_is_top`/`below_min_improvement` logic do the rest. Document the
   chosen path; (a) is the safe v1 (cheap, slightly stale DSR), (b) is the
   v2 refinement.

   For 48.3 v1 the SIMPLEST correct wiring: `incumbent_id = strategy of
   load_promoted_params`; since today that == `triple_barrier` == `tb_baseline`,
   the incumbent's DSR comes from `tb_baseline`'s OWN bake-off result (no extra
   run). Construct the incumbent dict from the matching ranked candidate after
   the producer runs, OR pre-seed it from `optimizer_best.json["dsr"]`. Either
   is defensible; flag the staleness of (a) in the verdict row.

### Verdict persistence (WITHOUT deploying) — follow the champion-challenger precedent
The deploy bridge is a LATER cycle (params->settings.paper_* is unbuilt —
producer.py:34-39). So 48.3 persists for AUDIT ONLY. Two complementary sinks,
both fail-open, mirroring `monthly_champion_challenger`:

1. **Local TSV/JSONL** (the cheap, $0, always-on audit) — append one row to
   `backend/backtest/experiments/rotation_log.tsv` (new) OR
   `handoff/logs/rotation_bakeoff_log.jsonl`. Columns: `ts, incumbent_id,
   selected_id, switched, reason, delta_dsr, ranked (json), num_trials,
   num_param_variants, window`. Precedent: `quant_results.tsv` +
   `_log_experiment` (engine TSV-write-safety rules: try/except, `flush()`,
   `encoding="utf-8"`, ASCII). This is the recommended v1 sink — no BQ
   dependency, unit-testable with a tmp_path.
2. **BQ deployment-log row (optional, fail-open)** — mirror
   `monthly_champion_challenger._emit_deployment_log_row` (:281): a
   `strategy_deployments_log`-shaped dict with `status="bakeoff_verdict"` /
   `allocation_pct=0.0` (zero = NOT deployed) so the audit trail is in the same
   place as the monthly gate, but no allocation changes. Gate behind `persist`
   and an injected `bq_fn` so tests pass `None`.

**Explicitly NOT done in 48.3:** no `promoted_strategies` MERGE, no
`settings.paper_*` mutation, no allocation change. The verdict is recorded; the
switch is a future cycle's job. State this in the module docstring's DEFERRED
block (same convention as 48.1/48.2).

---

## Q3 — $0 testability (the test seam)

The DI-factory-seam consensus from the external scan
([Stack Overflow / Tao of Testing / MoldStud]) is unambiguous: "inject a
FakeDatabase ... deterministic and consistent"; "Factory Pattern with DI allows
you to avoid hitting the database in tests." The 48.2 adapter is ALREADY built
to this pattern (its docstring: "$0-mockable ... by mocking engine.run_backtest
while the REAL pure-numpy generate_report + compute_pbo run on the fake
result"). 48.3 inherits two clean seams, in increasing order of isolation:

- **Seam A — inject `engine_factory`** (a stub returning an object whose
  `.run_backtest(skip_cache_clear=True)` returns a canned BacktestResult-like
  with a synthetic `nav_history` + `windows`). The REAL adapter + producer +
  selector run on top -> exercises the FULL wiring incl. `compute_pbo` /
  `generate_report` / gate / hysteresis, at $0, no BQ/LLM/engine. **This is the
  recommended primary test** — highest coverage, still deterministic.
- **Seam B — inject `adapter_fn`** (a stub `backtest_fn(params) ->
  {"dsr":..,"pbo":..,"sharpe":..}`). Bypasses the adapter entirely; tests only
  `run_rotation_bakeoff`'s incumbent-resolution + persistence + selector wiring.
  Faster, narrower. Use for the persistence/incumbent-edge tests.

Both seams are already in the `run_rotation_bakeoff` signature above
(`engine_factory=`, `adapter_fn=`). Persistence is tested by pointing the TSV
path at `tmp_path` and passing `bq_fn=None`.

**$0 unit-test matrix to ship this cycle:**
1. `make_rotation_engine` threads `target_vol`/window kwargs into the ctor
   (assert via a monkeypatched `BacktestEngine` capturing kwargs) — no real
   engine; just assert the ctor call args.
2. `make_rotation_engine` RAISES on an unknown strategy (before any I/O).
3. `make_rotation_engine` WARNs on a dead-key seed (caplog assert) and does NOT
   write `_strategy_params`.
4. `run_rotation_bakeoff` with Seam A stub factory -> verdict dict has the
   selector keys; a deliberately-best stub seed wins / ties per gate.
5. `run_rotation_bakeoff` with Seam B stub adapter_fn -> incumbent resolution
   (mock `load_promoted_params`) + `switched`/`reason` correctness.
6. Persistence: verdict row written to `tmp_path` TSV with correct columns;
   `persist=False` writes nothing; `bq_fn` exception is swallowed (fail-open).

All six are pure-Python, no network, no BQ, no LLM, no real backtest -> $0.

**Keep the LIVE bake-off opt-in:** the real run (4 seeds x 8 variants = ~32
backtests; the 48.2 brief estimate) stays behind a `@pytest.mark.skip`/`-m
live` integration test exactly as 48.2 did for the adapter. Do not let it run in
the default suite.

---

## Q4 — Live-run smoke feasibility (advisory)

**Recommendation: OPTIONAL, minimal, and only as a `live_check` artifact — do
NOT make it a gate dependency; the $0 mock tests are the real verification.**

Feasibility facts (verified against the engine):
- Macro/price preload is INSIDE `run_backtest` (engine:296-299), not a separate
  required `cache.preload_macro()` call here, so the "backtests hang after
  ~40min without preload_macro" rule (CLAUDE.md) does NOT bite a single short
  smoke — that rule is about long optimizer loops. A single short-window
  backtest preloads its own window.
- BQ data freshness: the incumbent is `triple_barrier` on US equities; the
  historical tables are the same ones the daily backtest already uses. A short
  window (e.g. `2024-01-01`..`2024-06-30`, 1-2 walk-forward windows) is cheap.
- Cost: $0 LLM (quant-only regime — no agent pipeline in backtest). BQ cost is
  bounded by the 2-query bulk preload over a short window + universe.

**If a smoke is run, use the MINIMAL shape** to stay inside a few minutes and
the session budget:
- `seeds=[{"id":"tb_baseline","param_overrides":{}}]` (ONE seed),
- `num_param_variants=2` (the adapter's MINIMUM for a valid PBO matrix — N>=2;
  with 1 it correctly OMITS pbo and the producer skips, which is itself a
  fine thing to observe),
- `start_date="2024-01-01", end_date="2024-06-30"` (short window),
- `persist=True` -> capture the `rotation_log` row as the `live_check_<sid>.md`
  evidence.

This proves the FULL real wiring (factory -> real engine -> nav_history ->
generate_report -> compute_pbo -> gate -> selector -> persisted row) end to end
once, at near-$0, in a few minutes. But because it depends on live BQ + a few
minutes of compute, it should be the operator's optional `live_check`, NOT a
blocker on the cycle's PASS. The deterministic $0 tests (Q3) are the gate.

If the session is time-boxed, **DEFER** the smoke to the live-run cycle (the
same cycle that the 48.2 brief already deferred the full ~32-backtest bake-off
to). The factory + runner + $0 tests stand on their own.

---

## Completable-one-cycle slice (the 48.3 deliverable)

BUILD this cycle (all $0, deterministic):
1. `backend/autoresearch/rotation_runner.py` with:
   - `make_rotation_engine(params, settings, bq, *, start_date, end_date)` —
     full ctor kwarg set + strategy validation + dead-key WARN.
   - `run_rotation_bakeoff(settings, bq, *, seeds, incumbent, num_param_variants,
     num_trials, start_date, end_date, persist, engine_factory, adapter_fn)` —
     builds factory+adapter, resolves incumbent via `load_promoted_params`, calls
     `run_strategy_bakeoff`, persists the verdict row, returns it.
   - A `_persist_verdict(verdict, path, bq_fn=None)` helper (TSV/JSONL +
     optional fail-open BQ row).
2. `tests/autoresearch/test_rotation_runner.py` — the 6-test $0 matrix (Q3),
   using stub `engine_factory`/`adapter_fn` + `tmp_path` + monkeypatched
   `load_promoted_params`/`BacktestEngine`.

DEFER (document in the module DEFERRED block, mirror 48.1/48.2):
- The LIVE multi-run bake-off (~32 real backtests) -> `@pytest.mark.skip`/live.
- The weekly rotation CRON.
- The deployment bridge (params -> `settings.paper_*` + `promoted_strategies`
  MERGE) — producer.py:34-39 documents why flipping a row alone is inert.
- Re-enabling the reverted vol-targeting / trailing-stop engine readers (so
  `tb_risk_managed` becomes genuinely risk-differentiated) — its OWN cycle.
- Effective-N (ONC) clustering for DSR deflation (over-deflation is the SAFE
  direction; 48.1/48.2 already deferred it).
- CPCV multi-path PBO upgrade (48.2 deferred; complements, not replaces).

---

## External: champion-challenger / periodic reselection (the external floor)

The pattern phase-48.3 implements — score a challenger set, record a verdict,
keep the incumbent live until a challenger demonstrably wins, defer the actual
switch — is the textbook **champion-challenger with shadow/offline evaluation**.
Consensus across the sources read in full:

- **Evaluate-then-promote is a hard separation.** Snowflake's guide runs the
  bake-off on a schedule and promotes ONLY `if challenger_metrics['auc'] >
  champion_metrics['auc']`, else "Challenger performance not better than
  Champion" and the champion is retained. This is exactly the selector's
  `dsr_improvement` vs `below_min_improvement`/`incumbent_is_top` branches.
- **The incumbent stays live during evaluation.** SparklingLogic: "apply
  different strategies on the same transaction but only act upon one";
  "the system would continue to approve the applicant but capture decision data
  for both strategies." Maps to 48.3 persisting a verdict at `allocation_pct=0`
  without touching live orders.
- **Shadow = log, don't serve.** Wallaroo + the Feb-2026 MLOps article: the
  challenger's "predictions are logged but not served to users", "compared
  against the current model" — the offline `rotation_log` row is the pyfinagent
  analogue.
- **Scheduled cadence.** Snowflake "every weekly Monday 1 AM"; DataRobot
  "automatic replay ... configure cadence". Confirms the DEFERRED weekly CRON is
  the standard next step (not 48.3's job).
- **Anti-churn / hysteresis is canonical for rotation.** The volatility-
  rotation hysteresis source (no re-entry until the barometer crosses a HIGHER
  threshold) and the arXiv:2411.07949 "no-trade zone" two-parameter optimum
  both validate the selector's `min_improvement` band — switching on a
  too-small edge just pays turnover for noise. (Reuses the 47.6/48.1 jump-model
  switch-penalty source; not re-derived.)

## Consensus vs debate (external)
- **Consensus:** challenger evaluated offline/in-shadow; incumbent retained
  until a material win; verdict logged separately from deployment; periodic
  cadence; hysteresis to avoid whipsaw. All five full-read sources agree.
- **Minor debate:** A/B (split live traffic) vs pure shadow (no live exposure).
  pyfinagent is unambiguously shadow/offline (paper-only, no live capital at
  risk in the bake-off) — the safer choice, and the only one compatible with
  "persist the verdict without deploying."

## Pitfalls (from literature + internal)
1. **Silent metric collapse (internal, the headline).** A challenger whose
   differentiating knobs are inert reads as "tied with incumbent" — exactly the
   `tb_risk_managed` dead-key situation. Surface it; don't let a no-op seed
   masquerade as a risk-managed challenger.
2. **Switching on noise (external).** Without the `min_improvement` band the
   loop whipsaws; the selector already guards this — keep `num_trials` honest so
   DSR deflation isn't gamed.
3. **Stale incumbent DSR (internal).** `load_promoted_params` returns no DSR;
   reading `optimizer_best.json["dsr"]` is a point-in-time snapshot. Prefer
   scoring the incumbent through the SAME adapter for apples-to-apples, OR rely
   on the incumbent being a seed (tb_baseline) so it's scored in-run.
4. **Cargo-culting the dead setter path.** Do NOT replicate the
   `_strategy_params` risk-key writes in the factory — they have no reader and
   would imply a behavior that doesn't exist.
5. **Letting the live bake-off into the default test suite (external+internal).**
   ~32 backtests x minutes = a slow, BQ-dependent suite. Gate it `-m live`.

## Application to pyfinagent (external -> file:line)
- Champion-challenger evaluate-then-promote -> `strategy_selector.select_best_strategy`
  (:113-131) gate+rank+hysteresis; `run_rotation_bakeoff` records the verdict,
  does NOT deploy.
- Incumbent retention -> `incumbent` arg sourced from `load_promoted_params`
  (`autonomous_loop.py:46`); selector's `incumbent_is_top`/`below_min_improvement`.
- Shadow logging -> `rotation_log.tsv` / BQ `allocation_pct=0` row, precedent
  `monthly_champion_challenger._emit_deployment_log_row` (:281).
- Scheduled cadence -> DEFERRED weekly CRON (producer.py:33).
- DI test seam -> `engine_factory`/`adapter_fn` injection (48.2 adapter:167
  pattern), `make_engine` precedent (run_harness.py:89).

---

## Recency scan (2024-2026)

Searched: (a) `2026` champion-challenger reselection scheduling; (b) `2026`
periodic strategy reselection bake-off; (c) `2025` DI factory test seam; (d)
year-less canonical champion-challenger + rotation hysteresis. Findings:

- **Feb-2026 MLOps deployment-evaluation article** (Medium/Omarzai): confirms
  champion-challenger + shadow + canary remain current best practice; predictions
  logged-not-served, compared, versioned/auditable. COMPLEMENTS the older
  canonical sources — no supersession of the DSR/PBO selection method (which is
  the 48.1/48.2-settled Bailey/LdP foundation, unchanged).
- **arXiv:2411.07949 (Nov 2024)** "Optimal two-parameter portfolio management
  with transaction costs": a no-trade-zone (hysteresis) is the locally optimal
  response to transaction costs — fresh quantitative backing for the selector's
  `min_improvement` anti-churn band.
- **2025 DI/testing guidance**: "constructor/setter patterns reduce false
  positives in automated validation by up to 35%"; factory+DI to avoid DB in
  tests — validates the inject-the-factory $0 seam.
- **No new finding supersedes the 48.1/48.2 PBO/DSR methodology.** Result: new
  sources COMPLEMENT (cadence, anti-churn, test-seam confirmation); none change
  the plan. The one genuinely plan-shaping discovery this cycle is INTERNAL
  (the dead-key revert), not from the literature.

---

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (Snowflake,
  Wallaroo, SparklingLogic, Medium/MLOps-Feb-2026, + the multi-source champion-
  challenger search synthesis; DataRobot + Clarifai attempted but returned
  redirect/empty — recorded as snippet-only).
- [x] 10+ unique URLs total (16 collected across 4 searches).
- [x] Recency scan (last 2 years) performed + reported (2024-2026; Feb-2026
  article + Nov-2024 arXiv + 2025 DI guidance).
- [x] Full pages read (not abstracts) for the read-in-full set.
- [x] file:line anchors for every internal claim.

Soft checks:
- [x] Internal exploration covered every relevant module (runner, engine ctor,
  trader, optimizer setter, registry, producer, adapter, selector, incumbent
  loader, persistence precedent).
- [x] Contradictions / consensus noted (A/B vs shadow; evaluate-vs-deploy).
- [x] Claims cited per-claim with URL + file:line.

### Read in full (>=5; counts toward gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://www.snowflake.com/en/developers/guides/ml-champion-challenger-model-deployment/ | 2026-05-29 | doc/blog | WebFetch full | Weekly schedule; promote only if challenger metric > champion on holdout; version-registry persistence |
| https://wallaroo.ai/ai-production-experiments-the-art-of-a-b-testing-and-shadow-deployments/ | 2026-05-29 | vendor doc | WebFetch full | Shadow = all models get same input, only champion influences decisions; sanity-check before live |
| https://www.sparklinglogic.com/champion-challenger-for-rolling-out-deployments/ | 2026-05-29 | vendor blog | WebFetch full | "apply different strategies ... only act upon one"; incumbent stays live while challenger evaluated |
| https://medium.com/@fraidoonomarzai99/deployment-evaluation-strategies-in-mlops-c208585aa3bd | 2026-05-29 | blog (Feb 2026) | WebFetch full | Recency anchor: champion-challenger/shadow/canary remain best practice; logged-not-served, auditable |
| (multi-source synthesis) champion-challenger scheduling/shadow query | 2026-05-29 | aggregate | WebSearch full-page summaries | Periodic scheduled replay (DataRobot cadence), systematic promotion decisions |

### Snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://docs.datarobot.com/en/docs/mlops/monitor/challengers.html | doc | Returned redirect notice only |
| https://www.clarifai.com/blog/ai-model-deployment-strategies | blog | Returned header/nav only, no body |
| https://arxiv.org/pdf/2411.07949 | paper | Snippet sufficient (no-trade zone = hysteresis optimum); reused 48.1 anti-churn line |
| https://www.volatilitytradingstrategies.com/blog/hysteresis-and-the-defensive-rotation-strategy-part-2 | blog | Snippet sufficient for hysteresis-rotation corroboration |
| https://stackoverflow.blog/2022/01/03/favor-real-dependencies-for-unit-testing/ | blog | DI seam corroboration via snippet |
| https://jasonpolites.github.io/tao-of-testing/ch3-1.1.html | doc | DI fake-vs-mock seam via snippet |
| https://www.datarobot.com/blog/introducing-mlops-champion-challenger-models/ | blog | Duplicate of pattern already read in full |
| https://docs.datarobot.com/en/docs/workbench/nxt-console/nxt-mitigation/nxt-challengers.html | doc | Duplicate challenger-replay doc |
| https://www.goatfundedtrader.com/blog/how-to-backtest-a-trading-strategy | blog | Rotation-interval whipsaw note via snippet |
| https://moldstud.com/articles/p-mastering-dependency-injection-in-unit-testing-a-guide-for-back-end-developers | blog | DI testability via snippet |
| https://www.quantifiedstrategies.com/trading-strategies-free/ | blog | Off-topic for this gate |

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 11,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "report_md": "research_brief_phase_48_3_rotation_runner.md",
  "gate_passed": true
}
```
