# Research Brief — steps 82.22 (optimizer_best provenance) + 82.23 (PBO never computed)

Tier: **moderate**. Audit-class: **no** (`coverage` informational only).

**One gate, two contracts.** 82.22 and 82.23 are a single coupled surface: both
concern `backend/backtest/experiments/optimizer_best.json` and the promotion /
go-live gate that reads metrics out of it. The provenance defect (82.22)
determines *which* numbers the gate is grading; the missing-PBO defect (82.23)
determines *which terms* the gate can grade at all. Fixing one without the other
leaves the gate either grading a mis-attributed metric or unable to compute the
overfitting veto it already declares. This brief feeds BOTH contracts and should
be cited by both.

---

## Part 1 — Internal inventory

### 1.1 Verification of the two measured defects (re-measured, not re-discovered)

**82.22 confirmed.** Run `60617e0b` wrote exactly 10 artifacts under
`backend/backtest/experiments/results/` (`60617e0b-exp01` .. `-exp10`, all with
`parent_run_id=60617e0b`, dated 2026-07-24). Their Sharpes span
0.5383536163549719 .. 0.6505601279884126 and their `deflated_sharpe`
0.006017459920086288 .. 0.6387492887307706. A scan of all **437** artifacts
finds exactly one file carrying the pair recorded in `optimizer_best.json`:

```
20260328T072722Z_52eb3ffe-exp10.json   run_id=52eb3ffe-exp10
  analytics.sharpe            = 1.1704633657934074
  analytics.deflated_sharpe   = 0.9525811126193078
```

i.e. 2026-03-28, four months before the run whose id is stamped on the file.

**Second-order finding (new, worth carrying into the contract).** Six of the ten
`60617e0b` experiments returned the *identical* Sharpe `0.6455483635957818`
(exp01/03/05/06/09/10) — the signature of a perturbation that was a no-op or
reverted, i.e. the incumbent parameter set re-measured under current code+data.
So the honest present-day Sharpe of the params in `optimizer_best.json` is
**≈0.646, not 1.170** — the file overstates the incumbent by ~1.8x. This is an
inference from the recurring value, not a direct baseline measurement (see 1.3:
the run *skipped* its baseline), but it is the strongest available estimate and
the contract should not assume 1.17 is merely mis-attributed-but-still-true.

**82.23 confirmed.** `generate_report` (`backend/backtest/analytics.py:649-760`)
never calls `compute_pbo`; the module docstring of the adapter says so
explicitly (`backend/autoresearch/strategy_backtest_adapter.py:17-18`:
"generate_report does NOT compute PBO -- only Sharpe + DSR").

### 1.2 Where `optimizer_best.json` is WRITTEN

Single writer for the metrics block: `QuantOptimizer._save_best_params()`,
`backend/backtest/quant_optimizer.py:720-735`:

```python
    def _save_best_params(self):
        """Persist best_params + metrics to JSON for warm-start."""
        try:
            payload = {
                "params": self.best_params,
                "sharpe": self.best_sharpe,
                "dsr": self.best_dsr,
                "run_id": self._run_id,          # <-- CURRENT run's uuid
                "kept": self.kept,
                "discarded": self.discarded,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            _BEST_PARAMS_PATH.write_text(json.dumps(payload, default=str, indent=2), ...)
```

Path constant: `quant_optimizer.py:32` `_BEST_PARAMS_PATH = _EXPERIMENTS_DIR /
"optimizer_best.json"`. Called unconditionally at the end of `run_loop()`
(`quant_optimizer.py:381`).

**The exact mechanism of the mis-attribution — a three-line chain:**

1. `__init__` calls `self._load_previous_best()` (`quant_optimizer.py:135`), which
   reads the *previous* `optimizer_best.json` and assigns the previous file's
   metrics onto this instance (`quant_optimizer.py:754-760`):
   ```python
                    prev_sharpe = data.get("best_sharpe", data.get("sharpe"))
                    prev_dsr = data.get("best_dsr", data.get("dsr"))
                    if prev_sharpe is not None:
                        self.best_sharpe = float(prev_sharpe)
                        self.best_dsr = float(prev_dsr) if prev_dsr is not None else 0.0
                        self.num_trials = 1
                        self._warm_started = True
   ```
2. `run_loop()` mints a fresh id — `self._run_id = str(uuid.uuid4())[:8]`
   (`quant_optimizer.py:176-178`) — and then, because `_warm_started` is true,
   **skips the baseline entirely** (`quant_optimizer.py:182`: `"QuantOptimizer:
   skipping baseline (warm-started Sharpe=%.4f)"`). So `best_sharpe`/`best_dsr`
   are never re-measured under the new run.
3. Every trial is compared to that inherited `best_sharpe` (`:293` `delta =
   trial_sharpe - self.best_sharpe`) and only overwrites it on a KEEP (`:299`).
   With `kept=0` the inherited values survive untouched to `_save_best_params()`,
   which stamps them with `self._run_id`.

So the defect is **structural, not a one-off**: any warm-started run with
`kept == 0` re-stamps stale metrics with a fresh, wrong `run_id`. The file
already records the tell — `"kept": 0, "discarded": 10` — but nothing reads it.

Note the reader at `:754` accepts **either** `best_sharpe` **or** `sharpe`
(and `best_dsr` or `dsr`). That tolerance matters for the schema proposal below.

Second, partial writer: `backend/services/promotion_gate.py:125`
`update_optimizer_best(...)` — writes `allocation_pct` + `stage` **in place**
(documented at `:132`), i.e. the file is already treated as an
additively-extensible record by an existing module. Precedent for adding keys.

Third writer (delete-only): `DELETE /api/backtest/optimize/history` removes the
file along with `quant_results.tsv` and `results/*.json`
(`.claude/rules/backend-backtest.md`, "Clear history").

### 1.3 Who READS it — the full consumer list

| # | File:line | Reads | Treats metrics as describing `run_id`? | Breaks if keys are added? |
|---|---|---|---|---|
| 1 | `backend/services/autonomous_loop.py:30,33-44` `load_best_params()` | `params` (falls back to whole dict), logs `sharpe` | logs only | No |
| 2 | `backend/services/autonomous_loop.py:47-74` `load_promoted_params()` → live cycle `:429-436` | delegates to #1; `summary["best_params_sharpe"] = best_params.get("sharpe", "?")` | **surfaces the stale Sharpe into the cycle summary** (note: `load_best_params` returns `data["params"]`, which has no `sharpe`, so this resolves to `"?"` today — a latent inconsistency, not a live bug) | No |
| 3 | `backend/services/perf_metrics.py:354,364-374` `_load_optimizer_best_sharpe()` → `compute_sharpe_gap(backtest_sharpe_source="optimizer_best")` `:419-447` | `sharpe` | **YES — load-bearing.** This is tier 1 of the 3-tier fallback and feeds the gate's reality-gap boolean | No |
| 4 | `backend/services/paper_go_live_gate.py:95-114` `_load_backtest_max_dd()` | `max_drawdown_pct` / `max_dd_pct` / `backtest_max_dd_pct` (all absent today → `None` → 20% cap) | would, if present | No — already a multi-key tolerant reader |
| 5 | `backend/services/promotion_gate.py:125-133` `update_optimizer_best()` | read-modify-write `allocation_pct`, `stage` | No | No |
| 6 | `backend/autoresearch/rotation_runner.py:73,161-166` `_incumbent_dsr_from_optimizer_best()` | `dsr` — "the live strategy's recorded DSR"; fail-open | **YES** — the incumbent DSR the rotation challenger must beat | No |
| 7 | `backend/autoresearch/strategy_registry.py:55-60,132-142` `_load_base_params()` | `params` — the base every seed's `param_overrides` is applied on top of | No | No |
| 8 | `backend/autonomous_loop.py:265-284` (harness planner evidence) | `params` | No | No |
| 9 | `backend/autonomous_harness.py:45,146` | `config_path` default | No | No |
| 10 | `backend/agents/harness_state_reader.py:11` | "current best params" for the MAS state read | No | No |
| 11 | `backend/agents/multi_agent_orchestrator.py:139` | tool description: "Sharpe, DSR, and all parameter values from optimizer_best.json" | **YES — the LLM is told these describe the current best** | No |
| 12 | `backend/autoresearch/proposer.py:25,43` | bundles the file into the LLM proposal context | YES (as prompt context) | No |
| 13 | `scripts/go_live_drills/dsr_oos_test.py:29,71-76` | whole file, DSR gate drill | **YES** | No |
| 14 | `scripts/go_live_drills/paper_runtime_test.py:21,98-99` | whole file | partly | No |
| 15 | Tests: `backend/tests/test_phase_75_promotion_gate.py:42,66`, `backend/tests/test_phase_75_8_1_harness_consumer.py:46,70`, `tests/verify_phase_25_A6.py` (6 patch sites), `tests/verify_phase_25_B3.py`, `tests/autoresearch/test_phase_48_1_*`, `test_phase_48_3_*:169` (`_incumbent_dsr_from_optimizer_best` monkeypatched to `0.9526`) | various | — | Only if a key is **renamed/removed** |

**Blast radius of the mis-attribution:** consumers 3, 6, 11, 12, 13 all consume
`sharpe`/`dsr` as if they were produced by run `60617e0b`. The most consequential
is #3 → `paper_go_live_gate.compute_gate`'s `sr_gap_le_30pct` boolean: live paper
Sharpe is being compared against a March backtest number that is ~1.8x the
present-day measurement of the same params, so the reality gap is measured
against an inflated reference and the boolean is **systematically harder to pass**
in a misleading direction (a live Sharpe near the true 0.646 reads as a ~45% gap
against 1.17). #6 is the mirror hazard on the rotation side: every challenger is
asked to beat an incumbent DSR of 0.9526 that the incumbent itself no longer
achieves.

### 1.4 Question 3 — the SAFE fix shape

Every consumer above uses `dict.get(...)` on a JSON object; **no consumer
enumerates keys, asserts a key count, or round-trips the file through a schema**.
The only read-modify-write consumer (`promotion_gate.update_optimizer_best`)
already adds keys of its own. Therefore:

- **Adding keys is safe** for all 15 consumers.
- **Renaming or removing `params`, `sharpe`, `dsr`, `run_id` is NOT safe** —
  `params` breaks #1/#2/#7/#8 (live cycle param loading and the strategy
  registry base), `sharpe` breaks #3 (silently — `_load_optimizer_best_sharpe`
  returns `None` and `compute_sharpe_gap` falls through to the shadow-curve tier,
  changing gate behaviour without an error), `dsr` breaks #6 (fail-open → `None`,
  so the rotation gate loses its incumbent bar), and `run_id` breaks the tests in
  #15 plus the drills in #13/#14.

Proposed additive schema (all new keys optional; every existing key retains its
current name, type, and meaning):

```jsonc
{
  "params": { ... },                      // UNCHANGED
  "sharpe": 1.1704633657934074,           // UNCHANGED (kept for #3)
  "dsr": 0.9525811126193078,              // UNCHANGED (kept for #6)
  "run_id": "60617e0b",                   // UNCHANGED = the run that WROTE the file
  "kept": 0, "discarded": 10,             // UNCHANGED
  "saved_at": "2026-07-24T11:04:51...",   // UNCHANGED

  // --- new, additive, provenance ---
  "metrics_run_id": "52eb3ffe-exp10",     // the run that MEASURED sharpe/dsr
  "metrics_measured_at": "2026-03-28T07:27:22Z",
  "metrics_source_artifact": "results/20260328T072722Z_52eb3ffe-exp10.json",
  "metrics_are_stale": true,              // metrics_run_id != run_id
  "warm_started_from": "52eb3ffe-exp10",  // what _load_previous_best inherited
  "baseline_skipped": true,               // run_loop:182 took the warm-start branch
  "schema_version": 2
}
```

Rationale for the exact field names: `run_id` must keep meaning "who wrote this
file" because #13/#14/#15 already read it that way; the *new* name carries the
new meaning. `metrics_are_stale` is a derived convenience so a consumer does not
have to know the comparison rule — but it must be **derived at write time from
`metrics_run_id != run_id`**, never hand-set. (Guard against the
absence-becomes-affirmative failure class recorded in
`project_fabricated_safe_80_36`: a *missing* `metrics_run_id` must read as
"unknown provenance", never as "fresh". Prefer `metrics_run_id is None →
unknown` over defaulting `metrics_are_stale` to `false`.)

Two changes are needed at the writer for this to be truthful rather than
decorative:

1. `_load_previous_best()` must capture the provenance it is inheriting
   (`quant_optimizer.py:754-765` already reads `data.get("run_id")` for the log
   line at `:763` — store it on `self` instead of only logging it), and fall
   through to the same treatment on the `result_store.load_latest()` branch
   (`:770-794`, which likewise logs `latest.get("run_id")` at `:793` and drops it).
2. `_save_best_params()` must emit `metrics_run_id = <the run that actually
   produced best_sharpe>` — i.e. the inherited id when no KEEP occurred, and
   `self._run_id` (or the specific `exp_id`) when a KEEP set it at `:299`.

Optional hardening, worth naming in the contract but **not** required for the
minimal fix: `_load_previous_best` currently sets `self.num_trials = 1`
(`:759`, and `:789` on the fallback branch). That resets the DSR trial count on
every warm start, so a long chain of runs is deflated as though only one
configuration had ever been tried — the same family of defect as the provenance
loss, and it inflates DSR. Flag it; do not silently fix it inside 82.22.

### 1.5 Question 4 — can PBO live in `generate_report`? **No.**

`generate_report(result: BacktestResult, num_trials: int = 1, baselines: dict |
None = None) -> dict` (`backend/backtest/analytics.py:649-653`). It receives
**one** `BacktestResult`. Everything it computes is derived from that single
object: `result.windows`, `result.nav_history`, `result.aggregate_sharpe`,
`result.feature_importance_mda`. `num_trials` is a bare integer used only to
deflate the Sharpe (`:674-685`) — it is a *count* of configurations, not the
configurations themselves. There is no path by which `generate_report` can see a
second column of PnL.

`compute_pbo` (`analytics.py:184-236`) needs `(T, N)` where the N columns are
**competing configurations**, and it fails **silently and optimistically**:

```python
    T, N = arr.shape
    if N < 2 or T < S * 2:
        return 0.0
```

`0.0` is the best possible PBO. So a naive `report["pbo"] = compute_pbo(single_
column)` would emit a hard-coded PASS on every run — the exact false-green the
adapter's guard was written to prevent (`strategy_backtest_adapter.py:141-147`:
"the LOAD-BEARING guard against compute_pbo's silent 0.0 (which would false-pass
the pbo<=0.20 gate)"). **Do not put PBO in `generate_report`.**

The 8 call sites confirm it: `quant_optimizer.py:201` and `:263`,
`api/backtest.py:1058`, `scripts/harness/run_harness.py:134,143`,
`run_quick_test.py:59`, `run_validation.py:88`,
`strategy_backtest_adapter.py:162` — every one passes a single result.

**PBO must live at the level that owns N configurations.** Two such levels exist
today, and both are already built:

- **`QuantOptimizer.run_loop()`** (`quant_optimizer.py:176-395`) is the natural
  home for the incumbent: it runs `i` experiments per run, each producing a
  `BacktestResult` with `nav_history`, and currently discards the columns after
  logging scalars. Collect each trial's daily-return column, stack at `:381`
  next to `_save_best_params()`, and stamp `pbo` into the same payload. This is
  where 82.3's 0.7486 measurement can be reproduced in-process.
- **`make_engine_backtest_fn`** (`strategy_backtest_adapter.py:169-256`) is the
  existing per-strategy path used by the rotation producer.

### 1.6 Question 6 — reuse, don't re-implement

`strategy_backtest_adapter.py` is a complete, tested precedent. Its three pieces:

- `_default_param_grid(seed_params, k)` (`:100-129`) — builds K competing configs
  by perturbing one knob (`mr_holding_days` for mean-reversion, else
  `holding_days`) ±8% per step plus a ±5% `tp_pct` co-perturbation.
- `_assemble_pbo_matrix(results, min_rows)` (`:133-153`) — each result's
  `nav_history` → one daily-return column, truncate to the shortest common
  length, `np.column_stack` → `(T, N)`; returns `None` when `<2` usable columns
  or `T < min_rows`.
- `make_engine_backtest_fn(...)` (`:169-256`) — runs the grid with
  `skip_cache_clear=True` (warm BQ cache across variants), drops a failed variant
  as a column rather than failing the strategy (`:207-213`), calls
  `clear_cache` once in a `finally`, and on an undersized matrix **omits `pbo`
  entirely** so the consumer SKIPS rather than reading a false 0.0 (`:236-244`).

`_assemble_pbo_matrix` is directly reusable by the optimizer: the optimizer's
per-trial `BacktestResult` objects expose `nav_history` in the same shape the
helper already handles (it accepts both dicts and objects, `:143`). The one
design decision the contract must make explicit: **the optimizer's N columns are
a sequential greedy search around a moving best, not an independent grid.** That
is still a legitimate CSCV input (Bailey et al. define N as the configurations
tried in the search), but it means the columns are correlated and N is small
(10 here) — see Part 3 for what the literature says about small N.

Also reusable: `pbo_check` at `backend/agents/mcp_servers/risk_server.py:133-158`
already wraps `compute_pbo` with a veto at `DEFAULT_PBO_VETO_THRESHOLD` and an
MCP-native `isError` signal — an existing threshold constant to align with.

### 1.7 Question 5 — the gate surface, including the UI payload

`compute_gate(bq)` (`paper_go_live_gate.py:117-213`) returns
`{booleans{5}, promote_eligible, details{}, thresholds{}, computed_at}` with
`promote_eligible = all(booleans.values())` (`:178`). DSR enters at `:136`
(`dsr = metrics.get("dsr")` from `compute_metrics_v2(bq)` — **live paper NAV**,
not `optimizer_best.json`) and is graded at `:171` against `DSR_THRESHOLD = 0.95`
(`:42`). So the gate's DSR term is already live-derived and is **not** affected
by 82.22; the affected term is `sr_gap_le_30pct` (`:149-150,173`) via
`compute_sharpe_gap` tier 1.

Adding a 6th boolean `pbo_le_50` touches:

1. `paper_go_live_gate.py` — a new loader (analogous to `_load_backtest_max_dd`,
   `:95-114`), a new entry in `booleans` (`:166-177`), a new `details` number,
   a new `thresholds` entry (`:204-211`), and a new module constant
   `PBO_CEILING = 0.5` (mirror `promotion_gate.py:37`, which already defines it —
   two constants for one threshold is the drift hazard `perf_metrics.py:361`
   explicitly calls out for `SR_GAP_THRESHOLD`; import rather than redeclare).
2. `backend/api/paper_trading.py:27,820,830` — passthrough, no change needed.
3. `frontend/src/components/GoLiveGateWidget.tsx:9-15` — the `GoLiveGate`
   TypeScript interface literally enumerates the 5 booleans, so a 6th requires an
   interface edit; `:99-125` builds the checklist rows; `:129-186` drives the
   ELIGIBLE badge and the `disabled={!promote_eligible}` promote button.
4. `frontend/src/components/OpsStatusBar.tsx:274,283-286` — builds a
   PASS/FAIL string array; a 6th boolean must be added or it is silently omitted
   from the operator's status tooltip.
5. `frontend/src/components/OpsStatusBar.test.tsx:96` — fixture carries the
   booleans object.
6. `frontend/src/lib/api.ts:549` `getPaperGate()` — types via the widget's
   exported interface, so it follows (1) automatically.

**Fail-safe direction is the load-bearing choice.** Today PBO does not exist
anywhere in the file. If the new boolean is `bool(pbo is not None and pbo <=
PBO_CEILING)`, then a missing PBO reads FALSE and the gate goes **red until PBO
is plumbed** — correct for a promotion gate (absence ≠ pass) and consistent with
`sr_gap_le_30pct`'s `None → False` at `:172-173`. It also means shipping the
boolean before shipping the producer flips `promote_eligible` to false, which is
an operator-visible behaviour change and must be disclosed in the contract. The
opposite (`pbo is None → True`) is the fabricated-SAFE pattern and should be
rejected outright.

---

## Part 2 — Search queries run (three-variant discipline)

| Variant | Query |
|---|---|
| Year-less canonical | `probability of backtest overfitting CSCV Bailey Borwein Lopez de Prado Zhu` |
| Year-less canonical | `experiment tracking model lineage provenance best model record` |
| Current-year frontier | `deflated Sharpe ratio PBO promotion gate quant strategy 2026` |
| Last-2-year window | `probability of backtest overfitting minimum number of trials 2025` |
| Last-2-year window | `MLflow model registry provenance run id metrics mismatch 2025` |

---

## Part 3 — External sources

### Read in full (8; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| E1 | https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf | 2026-08-04 | peer-reviewed (Bailey, Borwein, López de Prado, Zhu, *J. Computational Finance* 20(4)) | `curl` + `pdfplumber` (64k chars) — the `/pdf` WebFetch path is binary-only per `.claude/rules/research-gate.md` | Algorithm 2.3 defines M; **guided searches must contribute converged outcomes, not intermediate steps**; **`N >> 10 is required`**; S=16 justified |
| E2 | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | 2026-08-04 | peer-reviewed (Bailey & López de Prado, *JPM* 40(5)) | WebFetch (binary) → local `pdfplumber` re-extraction (45k chars) | DSR's five inputs incl. N; "Appendix 3 shows how N can be determined when the trials are not independent" |
| E3 | https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf | 2026-08-04 | peer-reviewed / LBL (Bailey, Ger, López de Prado, Sim, Wu) | WebFetch (binary) → local `pdfplumber` (30k chars) | hold-out "does not control for the number of trials involved in a discovery" |
| E4 | https://cran.r-project.org/web/packages/pbo/vignettes/pbo.html | 2026-08-04 | official package doc (R `pbo`) | WebFetch | "we assemble the trials into an NxT matrix where each column represents a trial and each trial has the same length T"; reference example uses **N=100** |
| E5 | https://arxiv.org/html/2603.20319 | 2026-08-04 | preprint (2026) | WebFetch (HTML) | Implementation risk: "a strategy can pass every multiple-testing correction yet still yield materially different Sharpe ratios depending on the engine that runs it" |
| E6 | https://mlflow.org/docs/latest/ml/model-registry/ | 2026-08-04 | official docs | WebFetch | "Each registered model version is linked to the MLflow run, logged model or notebook that produced it, enabling full reproducibility" |
| E7 | https://github.com/mlflow/mlflow/issues/18489 | 2026-08-04 | issue tracker (industry) | WebFetch | The exact failure mode: registering by artifact URI instead of run URI "breaks the link between the registered model and its source run, losing all context and metadata" — empty `run_id`, metrics not visible |
| E8 | https://arxiv.org/html/2505.14050 | 2026-08-04 | preprint (2025) | WebFetch (HTML) | **[NEGATIVE]** open-source algo-trading reference implementation cites Bailey et al. in the bibliography but implements **neither** PBO nor DSR — reports only Sharpe/Sortino/IR/MaxDD |

### Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not read in full |
|---|---|---|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | canonical PBO landing page | superseded by E1 (same paper, full text) |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | canonical DSR landing page | superseded by E2 |
| https://escholarship.org/uc/item/4w1110bb | repository copy of PBO | **fetch returned empty body** — recorded as an attempt, not a read |
| https://www.semanticscholar.org/paper/b1233b4f5384f003e85c2e0eec1a2dfc08f624c5 | index entry | metadata only |
| https://scholarworks.wmich.edu/math_pubs/42/ | repository entry | duplicate of E1 |
| https://www.pm-research.com/content/iijpormgmt/40/5/94 | publisher | paywalled; duplicate of E2 |
| https://www.researchgate.net/publication/318600389_… , …/286121118_… | aggregator | login wall |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | tertiary | lowest tier; nothing E2 lacks |
| https://arxiv.org/pdf/1509.08248 (Correctness of Backtest Engines) | preprint | adjacent to E5, not to the gate question |
| https://arxiv.org/pdf/1412.5558 | preprint | candle-chart backtests, off-topic |
| https://arxiv.org/pdf/1910.05555 | preprint | asset allocation, off-topic |
| https://www.preprints.org/…/download_pub (Point-in-Time Backtesting) | preprint, not peer-reviewed | tangential |
| https://arxiv.org/pdf/2406.09737 (MLOps multivocal review, 2024) | preprint | recency-scan hit, secondary to E6/E7 |
| https://arxiv.org/pdf/2110.03022 (Tribuo: ML with Provenance) | preprint | provenance-by-construction; interesting but Java-specific |
| https://www.sciencedirect.com/science/article/pii/S0306437924001534 | journal (2024) | paywalled |
| https://neptune.ai/blog/tools-for-ml-model-governance-provenance-lineage | industry blog | vendor comparison |
| https://atlan.com/know/training-data-lineage-for-llms/ , https://agility-at-scale.com/ai/governance/model-lineage-and-reproducibility/ , https://datahub.com/blog/data-lineage-for-ml/ , https://www.nightfall.ai/ai-security-101/data-provenance-and-lineage | industry blogs | tier 4 |
| https://community.fabric.microsoft.com/t5/Data-Science/MLFlow-Problem-with-aliases-and-metrics/m-p/5015181 | forum | tier 5; corroborates E7's "metrics live on the Run, not the registry object" |
| https://mlflow.org/docs/latest/ml/model-registry/tutorial/ , https://docs.databricks.com/aws/en/mlflow/models , https://oneuptime.com/blog/post/2026-01-25-model-registry/view , medium.com/@mlopsomari/… | docs/blog | duplicate of E6 |

**URLs collected: 33** (8 read in full + 25 snippet-only).

### Recency scan (last 2 years, 2024-2026)

Performed. Queries: `deflated Sharpe ratio probability of backtest overfitting strategy selection gate 2025 2026`; `MLflow model registry stale metrics wrong run attributed best model reproducibility failure`; plus the year-less canonical variants listed in Part 2.

Result: **two new findings, neither superseding the canonical sources.**

1. **Nothing supersedes Bailey CSCV (2016) or Bailey/LdP DSR (2014).** The 2025-2026 literature still cites both as the reference method; the search engine itself reported "results primarily reference foundational work from Bailey and López de Prado (2014) rather than very recent 2025-2026 developments." No revised PBO threshold, no replacement estimator, no retraction.
2. **New, orthogonal axis — implementation risk (E5, 2026).** "the variability in backtest outcomes attributable solely to the choice of simulation engine, with strategy logic, input data, and cost specification held fixed." 15 strategies x 5 engines: agreement is exact at zero cost, but "divergence scales monotonically with cost intensity (Spearman rho=0.93, p<0.001)", max divergence 0.2750%-0.4881% for signal strategies and **3.71% in total return for high-turnover rotation**. The authors frame this as "a second, independent source of backtest unreliability" and note "a strategy can pass every multiple-testing correction yet still yield materially different Sharpe ratios depending on the engine that runs it."
3. **Provenance-break is a live, recurring defect class in mainstream tooling (E7, mlflow#18489).** Not a pyfinagent-specific mistake.

### Key findings

**F1 — CSCV's columns are configurations, and pyfinagent's optimizer trials are the WRONG kind of column.** E1, Algorithm 2.3, verbatim:

> "First, we form a matrix M by collecting the performance series from the N trials. In particular, each column n = 1,...,N represents a vector of profits and losses over t = 1,...,T observations associated with a particular model configuration tried by the researcher. M is therefore a real-valued matrix of order (T x N)."

and then, decisively for 82.23:

> "A case in point is guided searches, where an optimization algorithm uses information from prior iterations to decide what direction should be followed next. In this case, the columns of matrix M should be the final outcome of each guided search (i.e., after it has converged to a solution), and not the intermediate steps."

`QuantOptimizer.run_loop` IS a guided search — each trial perturbs from the running best and is kept/discarded against it (`quant_optimizer.py:293-299`), with LLM-proposed directions (`:495-497`). Its 10 per-run trials are precisely "the intermediate steps" the paper excludes. **So the appealing shortcut — "the optimizer already has 10 nav_histories, just stack them" — is NOT CSCV-compliant.** It would produce a number, and the number would be wrong in an unquantified direction.

**F2 — N >> 10, and the existing precedent is under-sized.** E1:

> "N must be large enough to provide sufficient granularity to the values of the relative rank... If N is too small, [omega] will take only a very few values, which will translate into a very discrete number of logits, making f(lambda) too discontinuous, and adding estimation error... if the investor is sensitive to values of [phi] < 1/10, it is clear that the range of values that the logits can adopt must be greater than 10, and so **N >> 10 is required**."

and

> "As a rule of thumb, the researcher should backtest as many theoretically reasonable strategy configurations as possible."

E4's reference example uses N=100. pyfinagent's only PBO producer defaults to `_DEFAULT_K = 8` (`strategy_backtest_adapter.py:70`) — **below the paper's floor**, and its guard only requires `N >= 2` (`_assemble_pbo_matrix`, `:148`). This is a second, independent defect adjacent to 82.23 and should be queued as its own step rather than folded in silently.

**F3 — S=16 is correct and already used; the autocorrelation caveat is not.** E1:

> "For example, S = 16 we will obtain 12,780 logits... and sigma[f(lambda)] < 0.0045, with less than a 0.01 estimation error at 95% confidence level. Also, if M contains 4 years of daily data, S = 16 would equate to quarterly partitions, and the serial correlation structure would be preserved. For these two reasons, we believe that S = 16 is a reasonable value to use in most cases."

`analytics.compute_pbo(..., S=16)` and `_DEFAULT_PBO_S = 16` match. But note the paper's caveat, which applies to our daily-NAV T-axis:

> "if the performance measure as a time series has a strong autocorrelation, then such a division may obscure the characterization especially when S is large."

**F4 — is a gate on a single PBO number sound?** Partly. PBO is defined as

> "the conditional probability that this strategy underperforms the median OOS while remaining optimal IS"

so **0.5 is not an arbitrary threshold — it is the coin-flip point by construction** (Definition 2.1: overfitting means "a strategy with optimal performance IS has an expected ranking below the median OOS"). `promotion_gate.PBO_CEILING = 0.5` and `risk_server.DEFAULT_PBO_VETO_THRESHOLD = 0.5` are therefore correctly calibrated, not folklore. The paper does supply an uncertainty statement, but for the *logit density* rather than for PBO itself: with S=16, `sigma[f(lambda)] < 0.0045`. The paper gives **no** confidence interval on PBO as a function of N — instead it argues the N floor (F2) is what keeps the estimate from being discretization-dominated. Practical reading: a PBO point estimate at compliant N is a sound veto; at N=8-10 it is not, and the honest output is "not computable", never `0.0`.

Also relevant to the gate's shape, E1 lists what PBO does **not** cover:

> "this procedure does nothing to evaluate the correctness of a backtest. If the backtest is flawed due to bad assumptions, such as incorrect transaction costs or using data not available at the moment of making a decision, our approach will be making an assessment based on flawed information."

**F5 — DSR and PBO are complements, and DSR is the one 82.22 silently corrupts.** E2 lists DSR's inputs verbatim:

> "DSR deflates SR by taking into consideration five additional variables: The non-Normality of the returns, the length of the returns series (T), the variance of the SRs tested (V[{SR}]), as well as the number of independent trials involved in the selection of the investment strategy (N)."

and states the disclosure requirement that `optimizer_best.json` currently fails:

> "the most important piece of information missing from virtually all backtests published in academic journals and investment offerings is the number of trials attempted. Without this information, it is impossible to assess the relevance of a backtest. Put bluntly, a backtest where the researcher has not controlled for the extent of the search involved in his or her finding is worthless, regardless of how excellent the reported performance."

This is the literature's direct verdict on `_load_previous_best()` setting `self.num_trials = 1` (`quant_optimizer.py:759`, `:789`): the trial counter is reset to 1 on every warm start, so N is understated and the DSR is inflated. E2 also settles the correlated-trials question raised by F1: "Appendix 3 shows how N can be determined when the trials are not independent" — i.e. the fix for correlated configurations is to compute an *effective* N, not to abandon the deflation.

> **Correction / integrity note.** The first WebFetch pass over E2 returned two fluent "quotes" — *"if the number of trials is understated or reset, the DSR will be artificially inflated, leading to false confidence in strategy performance"* and *"the Deflated Sharpe Ratio should not be used in isolation"* — that do **not** appear anywhere in the paper's extracted text. They were fabricated by the summarising model against a binary PDF. Every E2/E3 quote in this brief was re-verified by local `pdfplumber` extraction. Any downstream contract must not cite those two strings.

**F6 — provenance: the mandatory field is the link from the record to the producing run.** E6 (MLflow, official): "Each registered model version is linked to the MLflow run, logged model or notebook that produced it, enabling full reproducibility"; registry entries carry "lineage (i.e., which MLflow experiment and run produced the model), versioning, aliasing, metadata tagging". E7 documents what our file does today, in another system: registering by artifact URI rather than run URI means "empty `run_id` field", "no connection to the source training run", "evaluation metrics not visible" — and it "breaks the link between the registered model and its source run, losing all context and metadata." pyfinagent's variant is worse in one specific way: our `run_id` is **not empty, it is populated with a different run's id**, so the record is not merely lineage-less but actively misattributing. E6/E7 confirm the fix shape: keep an explicit, machine-readable pointer from the metrics to the run that produced them.

### Consensus vs debate

Consensus: (a) columns of M are configurations, not time windows; (b) N must be substantially greater than 10; (c) S=16 is the default; (d) PBO > 0.5 means the selection process is worse than chance, so 0.5 is a principled ceiling; (e) DSR requires a disclosed, non-reset trial count; (f) a "best model" record must point at the run that produced its metrics.

Debate / open: (a) how to set N for a *guided* search — the paper says use converged outcomes but does not give a recipe for turning a 10-step greedy walk into >>10 converged configurations; (b) how to de-correlate trials (E2 Appendix 3's effective-N is the pointer, not a closed form we have implemented); (c) E5 argues (2026) that even a PBO/DSR-clean strategy carries engine-level implementation risk of up to 3.71% total return for high-turnover strategies — a gap no gate in this repo measures.

### Pitfalls (from the literature, mapped to this repo)

1. **Silent 0.0 on an undersized matrix** — `analytics.py:207-208` returns the most-favourable value on `N < 2 or T < S*2`. Literature offers no such convention; it is a local invention and a false-green generator. The adapter guards it (`:236-244`); the optimizer must too.
2. **Intermediate greedy steps as columns** (F1) — the single most likely way to ship a wrong PBO here.
3. **N = 8 or 10** (F2) — under the paper's floor; the number is discretization-dominated.
4. **`num_trials` reset to 1 on warm start** (F5) — inflates DSR; the same warm-start code path that causes 82.22.
5. **PBO cannot rescue a wrong backtest** (F4) — costs, point-in-time correctness, structural breaks outside T are all outside its scope. Do not let a green PBO be read as "the backtest is right."
6. **Autocorrelated NAV + large S** (F3) — our T-axis is daily NAV; the paper flags exactly this.

### Application to pyfinagent

| Finding | Anchor | Consequence for the contract |
|---|---|---|
| F1 guided-search exclusion | `quant_optimizer.py:176-395` (greedy loop), `:293-299` (keep/discard vs running best) | **Do NOT stack the optimizer's per-run trials into M.** PBO must be produced by an explicit independent-configuration sweep, i.e. the `strategy_backtest_adapter` pattern (`:100-129` grid → `:133-153` matrix → `:246` `compute_pbo`), not by opportunistically reusing `run_loop`'s columns |
| F2 N >> 10 | `strategy_backtest_adapter.py:70` `_DEFAULT_K = 8`; `:148` `len(cols) < 2` guard | The existing precedent is reusable in *shape* but under-sized in *N*. Raise K for the gate path, or explicitly record `n_variants` next to `pbo` so a reader can discount it. Queue the K-floor as its own defect step (per `feedback_queue_discovered_defects_in_masterplan`) |
| F1+F2 vs 82.3's 0.7486 | step 82.3 artifact | The contract must state **what N** produced 0.7486 and whether those columns were converged configurations. If N was ~8-10 greedy steps, the number is directionally suggestive but not gate-grade, and the step should say so rather than promote it into an immutable criterion |
| F4 threshold 0.5 | `promotion_gate.py:37`, `risk_server.py:28` | 0.5 is principled; **import the existing constant, do not declare a third one** (`perf_metrics.py:358-361` already warns about exactly this duplication for `SR_GAP_THRESHOLD`) |
| F4 absence handling | `paper_go_live_gate.py:172-173` (`None → False` precedent) | `pbo_le_50` must be `pbo is not None and pbo <= PBO_CEILING`. Missing PBO → red |
| F5 trial-count reset | `quant_optimizer.py:759`, `:789` | Flag in 82.22's contract as an adjacent, separately-queued defect; do not silently re-arm it (arming a dead guard is a behaviour change — cf. `reference_vacuous_type_guards_on_bq_string_columns`) |
| F5 disclosure | `optimizer_best.json` has no trial count | Add `num_trials` / `n_independent_trials` to the additive schema alongside `metrics_run_id` — the literature treats it as the single most important missing field |
| F6 lineage pointer | `quant_optimizer.py:720-735` writer; `:754-765` + `:770-794` loaders | `metrics_run_id` + `metrics_source_artifact` is the MLflow-equivalent link. Both warm-start branches already *log* the source run id and then discard it — capture it instead |
| F6 misattribution vs absence | `project_fabricated_safe_80_36` | A **missing** `metrics_run_id` must read as "unknown provenance", never "fresh" |
| E5 implementation risk | no anchor — nothing in the repo measures it | Out of scope for both steps. Worth a note in the contract's "known unmeasured risks" only |

### Answers to the six internal questions (short form)

1. **Written at** `backend/backtest/quant_optimizer.py:720-735` (`_save_best_params`, called from `:381`); path constant `:32`. Mis-attribution chain: `:135` → `:754-760` (inherit metrics) → `:176-178` (fresh `run_id`) → `:182` (baseline skipped) → `:299` (only a KEEP overwrites) → `:727` (stamp current id). Partial in-place writer: `promotion_gate.py:125`.
2. **15 consumers**, table in §1.3. The ones that read the metrics as describing the named run: `perf_metrics.py:364-374` → `compute_sharpe_gap` (feeds gate boolean `sr_gap_le_30pct`), `rotation_runner.py:161-166` (incumbent DSR bar), `multi_agent_orchestrator.py:139` + `proposer.py:25,43` (LLM context), `scripts/go_live_drills/dsr_oos_test.py:71-76`. **`paper_go_live_gate.py` does NOT take its DSR from this file** — `compute_gate` reads live paper DSR from `compute_metrics_v2` (`:136`); the file only supplies the optional backtest max-DD (`:95-114`, currently absent) and, indirectly via `perf_metrics`, the backtest Sharpe.
3. **Safe fix = additive only** (§1.4). All 15 consumers are `dict.get`-based; adding keys breaks none. Renaming/removing `params` breaks the live cycle + strategy registry; `sharpe` silently degrades `compute_sharpe_gap` to its shadow-curve tier; `dsr` blinds the rotation gate; `run_id` breaks the two go-live drills and the phase-25/48/75 tests.
4. **PBO cannot live in `generate_report`** (§1.5). It takes one `BacktestResult` (`analytics.py:649-653`); `compute_pbo` returns `0.0` on `N < 2` (`:207-208`); all 8 call sites pass a single run. It must live where N configurations exist — and per F1 those must be *converged, independent* configurations, which points at `strategy_backtest_adapter.make_engine_backtest_fn`, not at `run_loop`'s greedy trials.
5. **Gate wiring** (§1.7): 6th boolean touches `paper_go_live_gate.py` (constant + `booleans` + `details` + `thresholds`), and on the UI side `GoLiveGateWidget.tsx:9-15` (the interface literally enumerates the five), `:99-125`, plus `OpsStatusBar.tsx:283-286` and its test fixture at `:96`. `api/paper_trading.py:830` and `api.ts:549` follow automatically. Shipping the boolean before the producer flips `promote_eligible` false — disclose it.
6. **Reuse `strategy_backtest_adapter`** (§1.6): `_default_param_grid` (`:100-129`), `_assemble_pbo_matrix` (`:133-153`, accepts dicts *or* objects so the optimizer's results drop straight in), `make_engine_backtest_fn` (`:169-256`) with its omit-`pbo`-on-undersized-matrix discipline (`:236-244`). Do not re-implement. Do raise `_DEFAULT_K` (`:70`) toward the paper's `N >> 10`.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (8: E1-E8)
- [x] 10+ unique URLs total (33)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full papers / pages read, not abstracts (E1/E2/E3 via local `pdfplumber` text extraction; E4-E8 full HTML)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (writer, 15 consumers, PBO producer, gate, UI)
- [x] Contradictions / consensus noted (guided-search exclusion contradicts the obvious implementation; fabricated-quote incident recorded)
- [x] All claims cited per-claim
- [ ] **Gap:** the N used to produce step 82.3's PBO 0.7486 was not re-derived here — the contract must state it. Flagged, not silently assumed.

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 25,
  "urls_collected": 33,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "One coupled surface. 82.22: optimizer_best.json's metrics are inherited by _load_previous_best (quant_optimizer.py:754-760), the baseline is skipped when warm-started (:182), and _save_best_params (:720-735) stamps them with the CURRENT run_id -- so any warm-started run with kept==0 re-attributes stale metrics. Verified: run 60617e0b's 10 artifacts span Sharpe 0.538-0.651; the recorded pair belongs to 52eb3ffe-exp10 (2026-03-28). Six of ten trials returned Sharpe 0.6455, so the incumbent's true present-day Sharpe is ~0.646, not 1.170. 15 consumers, all dict.get-based: adding keys is safe, renaming is not. 82.23: PBO cannot live in generate_report -- it takes one BacktestResult and compute_pbo returns a false 0.0 on N<2. Bailey et al. exclude guided-search intermediate steps as columns and require N >> 10, so the optimizer's 10 greedy trials are the wrong columns and the adapter's _DEFAULT_K=8 is under-sized. PBO<=0.5 is principled (coin-flip by construction). A 6th gate boolean touches two frontend components. One fabricated-quote incident recorded and corrected.",
  "brief_path": "handoff/current/research_brief_82.22_82.23.md",
  "gate_passed": true
}
```
