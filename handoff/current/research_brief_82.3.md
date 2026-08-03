# Research Brief — step 82.3 (backtest evidence for 3 candidates vs `triple_barrier`)

Tier: **moderate**. Audit-class: **no**. Write-first: created before any source
was read; sections appended incrementally.

STATUS: COMPLETE. `gate_passed: true` (7 sources read in full, ~28 URLs,
recency scan performed). Envelope at the tail.

---

## 0. CORRECTION to the spawn premise (measured, not asserted)

The prompt states `compute_pbo` has **zero callers**. That is **FALSE**. Measured
via `grep -rn compute_pbo --include="*.py" .` (excluding `.venv`):

| Caller | Line | What it does |
|---|---|---|
| `backend/autoresearch/strategy_backtest_adapter.py:247` | `pbo = float(compute_pbo(matrix, S=pbo_S))` | The REAL production caller — builds the (T,N) matrix from K param variants |
| `backend/agents/mcp_servers/risk_server.py:143` | `pbo = float(compute_pbo(pnl_matrix, S=S))` | MCP risk tool; caller supplies the matrix |
| `tests/autoresearch/test_phase_48_2_backtest_adapter.py:106` | `compute_pbo(mat, S=16)` | Hand-matrix unit test |

What IS true: **no PBO value is ever written into a `results/*.json`** — because
`generate_report()` (`analytics.py:649-745`) does not compute PBO. The
`analytics` block it emits is exactly the 11 keys the prompt lists (verified at
`analytics.py:692-704`): `sharpe, deflated_sharpe, dsr_significant,
total_return_pct, alpha, max_drawdown, hit_rate, n_trades, n_windows,
information_ratio, num_trials`. **No PBO, no turnover.** So the gap is real, but
it is a *plumbing* gap (PBO lives on a different code path — the autoresearch
producer — that never runs during a plain `run_backtest`), not a
*never-implemented* gap. **82.3 should REUSE `strategy_backtest_adapter.py`, not
re-derive the matrix construction.**

---

## 1. PBO matrix construction — the T-axis is DAILY NAV RETURNS, not per-window returns

### What `compute_pbo` actually does (`analytics.py:184-236`)

```
T, N = arr.shape
if N < 2 or T < S * 2:  return 0.0        # :205-206  SILENT degenerate return
size = T // S                              # :209
subsets = [arr[i*size:(i+1)*size, :] for i in range(S)]   # :210  contiguous time slices
for is_idx in combinations(range(S), S//2):               # :213  C(16,8) = 12,870 splits
    IS  = vstack(subsets in is_idx); OOS = vstack(the rest)
    is_sharpe  = IS.mean(axis=0)/IS.std(axis=0)           # :220  per-COLUMN Sharpe
    best = argmax(is_sharpe)                              # :222  IS winner
    rank = sum(oos_sharpe <= oos_sharpe[best])            # :224  its OOS rank
    omega = (rank - 0.5)/N ; logit = log(omega/(1-omega)) # :225-226
PBO = integral of KDE(logits) over (-inf, 0]              # :231-233
```

Reading it line by line settles the question:

- **Columns are trials, rows are time.** `is_sharpe`/`oos_sharpe` are computed
  `axis=0` (down the rows) — so each column must be a time series of PnL for
  ONE configuration, and all columns must share the SAME time axis. This is
  Bailey/Borwein/Lopez de Prado/Zhu Algorithm 2.3 verbatim.
- **Row slicing is positional, not date-keyed** (`arr[i*size:(i+1)*size]`,
  `:210`). There is NO date join. If columns have different lengths you cannot
  even build the array — `np.column_stack` raises. If you pad/truncate them
  differently, row *i* of column A and row *i* of column B are different
  calendar days and every IS/OOS Sharpe is computed on a misaligned panel. The
  existing adapter handles this by truncating all columns to the shortest common
  length (`_assemble_pbo_matrix`, `strategy_backtest_adapter.py:149-152`) with
  the stated justification that variants of one strategy share the `bdate` grid.
- **`T >= S*2 = 32` is a hard floor** at `:205`, and it returns **0.0 silently**
  when unmet. **PBO = 0.0 PASSES the `<= 0.5` gate.** This is a false-pass
  hazard, already documented as LOAD-BEARING at
  `strategy_backtest_adapter.py:20-24`.

### Verdict on the T-axis

**Use the daily NAV return series, NOT `per_window.total_return_pct`.**

Reasons, in order of decisiveness:

1. **T would be ~27, and 27 < 32 → `compute_pbo` returns 0.0 silently.** With
   `S=16` you need T >= 32. Per-window returns on the 2018-2025 sample give
   ~27-31 rows. You would get a fabricated 0.0 that PASSES the gate. Even
   dropping to `S=8` (T>=16) leaves `size = 27//8 = 3` rows per subset — a
   3-observation Sharpe is noise.
2. **Daily NAV gives T ≈ 1,900-2,000.** `result.nav_history` is built at
   `backtest_engine.py:372-375` from `self.trader.snapshots`, and the daily
   mark-to-market loop (`pd.bdate_range`, per `.claude/rules/backend-backtest.md`)
   writes one snapshot per business day per window — ~500+ per backtest on a
   3-year sample, proportionally ~1,900 on 2018-2025. Comfortably above `S*2`.
3. **It matches the DSR side exactly.** `generate_report` computes its own
   `daily_returns = np.diff(navs)/navs[:-1]` at `analytics.py:666-667`, and
   `_daily_returns_from_nav` (`strategy_backtest_adapter.py:75-91`) mirrors it
   verbatim, by explicit design ("so the DSR-side and PBO-side returns are
   computed identically", `:79-81`). Using per-window returns for PBO and daily
   returns for DSR would put the two gate terms on different objects.

### The construction to use for 82.3

For **each of the 4 strategies separately**, build one (T, K) matrix whose K
columns are that strategy's K param variants:

```
matrix_strategy = np.column_stack([
    diff(nav_i)/nav_i[:-1]  truncated to min common length
    for i in 1..K variants of THAT strategy
])
pbo_strategy = compute_pbo(matrix_strategy, S=16)
```

i.e. exactly `_assemble_pbo_matrix(results, min_rows=32)` +
`compute_pbo(matrix, S=16)` from the existing adapter. With the operator's
N=12 plan that is **K=3 columns per strategy** — see §6 for why K=3 is the
weakest link in the plan and what to do about it.

**Do NOT pool all 12 runs into one 12-column matrix across strategies.** Bailey
Algo 2.3 columns are configurations of the SAME model; the adapter enforces this
by holding `strategy` fixed (`_default_param_grid`, `:98-108`, raises on an
unknown name). A cross-strategy 12-column matrix answers "did my *strategy
selection* overfit", which is a different (and also interesting) question — if
82.3 wants that number too, report it as a SEPARATE, clearly-labelled
`pbo_selection`, never as the per-strategy PBO the promotion gate consumes.

**What breaks with unequal column lengths:** `np.column_stack` raises
`ValueError: all input arrays must have same first dimension`. The adapter
pre-empts it by truncating to `min_len` (`:149-152`) — but note truncation is
from the FRONT (`c[:min_len]`), so if one variant's NAV series is shorter
because it *ended early*, truncation silently discards the tail of every other
column. Across strategies with different horizons the bdate grid is in fact the
same (windows come from `WalkForwardScheduler`, not from `holding_days`), so
this is benign here — but assert equal lengths and log any truncation rather
than relying on it.

---

## 2. Turnover — NOT computed anywhere in `backend/`

Measured: `grep -rn "turnover" backend/` returns **zero hits** in Python source.
Every hit is in `scripts/ablation/*` replay scripts and one migration docstring:

| File | Line | Definition used |
|---|---|---|
| `scripts/ablation/residual_momentum_replay.py` | 128 | `1 - len(basket & prev_basket)/max(len(basket),1)` |
| `scripts/ablation/sector_neutral_replay.py` | 205, 231, 246 | same basket-overlap formula |
| `scripts/ablation/no_trade_band_replay.py` | 7 | reports turnover per arm |

So the project's existing precedent is **basket-overlap turnover** (fraction of
the holdings basket replaced per rebalance). That formula is NOT expressible from
a `run_backtest` result — the result carries `nav_history` (date/nav/cash) and
`all_trades` (capped at **500**, `backtest_engine.py:379-386`), not per-date
basket membership.

### Formula to use for 82.3 (expressible from what a run already returns)

`BacktestTrader` records every fill with `quantity`, `price`, `action`,
`commission` (`backtest_trader.py:31`, `Trade` dataclass), and `snapshots`
carry `nav` + `cash`. So define **notional turnover**:

```
traded_notional = sum(t.quantity * t.price for t in trader.trades)      # both legs
avg_nav         = mean(s.nav for s in trader.snapshots)
years           = (last_snapshot_date - first_snapshot_date).days / 365.25

annualized_turnover = traded_notional / (2 * avg_nav) / years
```

The `2 *` converts round-trip notional (buy + sell both counted) into the
conventional one-way turnover ratio, so 1.0 = the portfolio is replaced once per
year. `avg_nav` is already computed by `generate_report` at `analytics.py:744`
(`avg_nav = mean([n["nav"] for n in result.nav_history])`) — reuse it.

Two mandatory cautions:

- **`result.all_trades` is capped at 500** (`backtest_engine.py:380`). Computing
  turnover from `result.all_trades` UNDER-COUNTS any run with >500 fills, and a
  12-run comparison would then rank strategies partly by how badly each was
  truncated. Compute turnover from `engine.trader.trades` (uncapped, in memory)
  in the same process, or from `trader.total_commission` (see below).
- A **commission-implied** cross-check is available with no cap risk:
  `total_commission / (flat_pct/100)` recovers total traded notional exactly
  under the `flat_pct` model, since `_compute_commission` is linear in notional.
  `trader.total_commission` is a running scalar (`backtest_trader.py:71`,
  incremented at `:119, :167, :211`) and is never truncated. Report both; they
  should agree.

---

## 3. Net-of-cost return — `total_return_pct` IS already net. Do not double-count.

Read `backend/backtest/backtest_trader.py`. Commission is deducted from `cash`
on **every** fill path, all three of them:

| Path | Lines | Cash effect |
|---|---|---|
| SELL (close position) | `:117-119` | `cost = _compute_commission(...)`; `self.cash += proceeds - cost` |
| BUY | `:155-167` | `total_needed = notional + transaction_cost`; `self.cash -= total_needed` |
| Liquidate / close-all | `:209-211` | `self.cash += proceeds - cost` |

BUY additionally checks affordability *inclusive* of commission
(`total_needed > self.cash`, `:158`) and re-sizes at 95% of cash if short
(`:159-163`), so commission is a real budget constraint, not a bookkeeping
afterthought.

The chain from cash to the reported number:
`cash` → `nav = cash + positions_value` (`:191, :238`) → `trader.snapshots`
(`:196`) → `result.nav_history` (`backtest_engine.py:372-375`) → window returns →
`result.aggregate_return_pct = (prod(1+returns) - 1) * 100`
(`backtest_engine.py:353`) → `analytics["total_return_pct"]`
(`analytics.py:696`).

**Verdict: `total_return_pct`, `sharpe`, `deflated_sharpe`, `max_drawdown` and
the PBO matrix are ALL net of commission**, because every one of them descends
from the same commission-reduced NAV series. 82.3 must NOT subtract costs again.

Caveat to state in the results: this is net of **commission only**. There is no
spread, no slippage, no market impact, no borrow. The `transaction_cost_pct`
constructor kwarg (default 0.1 in `make_engine`, `run_harness.py:109`) is a
separate knob — confirm which of the two is live before claiming a cost model,
and report the commission model + rate alongside every net return (settings
`backtest_commission_model` / `backtest_commission_per_share`).

---

## 4. Minimum correct headless invocation

`BacktestEngine.run_backtest(universe_tickers=None, skip_cache_clear=False)`
(`backtest_engine.py:275-277`). It is far more self-sufficient than the
CLAUDE.md "always call `cache.preload_macro()`" rule implies — **`run_backtest`
does all three preloads itself** at `:311-315`:

```
cache.preload_prices(universe_tickers + [_benchmark], global_start, global_end)
cache.preload_fundamentals(universe_tickers)
cache.preload_macro()
```

It also calls `self.trader.full_reset()` (`:300`) and
`self._auto_ingest_if_needed(universe_tickers)` (`:296`). So the caller does NOT
need to preload. What the caller MUST do:

1. **Construct the engine via `run_harness.make_engine(params, settings, bq)`**
   (`scripts/harness/run_harness.py:89-111`) — the documented precedent, used by
   `run_backtest`/`run_backtest_full` (`:129-144`). It threads `start_date`,
   `end_date`, `strategy`, `holding_days`, `tp_pct`, `sl_pct`, `frac_diff_d`,
   **`mr_holding_days`**, the 4 ML params, `max_positions`, `top_n_candidates`,
   `transaction_cost_pct`.
   **GAP (already flagged in-repo at `strategy_backtest_adapter.py:49-53`):**
   `make_engine` threads only a SUBSET of constructor kwargs — no
   `target_vol`, no trailing-stop, no blend weights. A run whose params include
   those will SILENTLY ignore them. For 82.3 this is benign *only if* the 12
   configs vary strictly within the threaded subset. Verify that before running;
   otherwise extend the factory.
2. **Pass `skip_cache_clear=True` for every run except the last**, then call
   `cache.clear_cache()` once in a `finally`. This is the warm-cache discipline
   the adapter implements at `strategy_backtest_adapter.py:207-228`. Note
   `run_harness.run_backtest` (`:132`) calls `engine.run_backtest()` with the
   DEFAULT `skip_cache_clear=False`, so using that helper for 12 runs clears and
   re-fetches the BQ cache 12 times. **Do not use `run_harness.run_backtest` for
   a multi-run comparison** — use the adapter's loop shape.
3. `engine.stop_check` is an assignable attribute checked between windows
   (`backtest_engine.py:332-335`), for interrupting a long run. Optional; leave
   unset for headless.
4. `progress_callback` is a constructor kwarg; `make_engine` wires
   `progress_cb`. Harmless headless.

**Minimum correct shape:**

```python
from backend.backtest import cache
from backend.backtest.analytics import generate_report, compute_pbo
from scripts.harness.run_harness import make_engine

results = []
try:
    for cfg in configs:                       # 12 configs
        eng = make_engine(cfg, settings, bq,
                          start_date="2018-01-01", end_date="2025-12-31")
        results.append(eng.run_backtest(skip_cache_clear=True))
finally:
    cache.clear_cache()
```

### What gets persisted

- **`result_store.save_result(run_id, report)`** (`result_store.py:23-40`)
  writes `experiments/results/{UTC %Y%m%dT%H%M%SZ}_{sanitized_run_id}.json`
  containing the WHOLE `report` dict — `analytics`, `per_window`,
  `feature_importance`, `equity_curve`, `nav_history`, `strategy_params`
  (`analytics.py:691-739`). Note it serializes `json.dumps(report, default=str)`
  — no schema validation, and `nav_history` is why each file is ~328 KB
  (measured on the 2026-07-24 run files). Nothing else writes PBO or turnover
  into it; 82.3 must add those keys explicitly if they are to survive to disk.
- **`_log_experiment()`** (quant_optimizer) appends one TSV row to
  `backend/backtest/experiments/quant_results.tsv` with columns
  `timestamp, run_id, param_changed, metric_before, metric_after, delta, status,
  dsr, top5_mda, params_json, parent_run_id` (verified from the file header).
  **There is no PBO column and no turnover column.** It serializes
  `trial_params` (not `best_params`). It is wrapped in try/except and logs at
  ERROR without re-raising, so a failed TSV write is silent — check the row
  count before and after.

---

## 5. RUNTIME — ~20 minutes per walk-forward run, warm cache. 12 runs ≈ 4 hours.

This is **measured**, not estimated. Consecutive optimizer experiment timestamps
in `backend/backtest/experiments/quant_results.tsv` (the optimizer passes
`skip_cache_clear=True`, i.e. WARM cache, per `.claude/rules/backend-backtest.md`):

| Run `60617e0b` (2026-07-24) | Δ to previous |
|---|---|
| `08:02:31` exp01 | (baseline `07:42:28` → **20m 03s**) |
| `08:23:01` exp02 | **20m 30s** |
| `08:43:12` exp03 | **20m 11s** |
| `09:03:30` exp04 | **20m 18s** |
| `09:23:54` exp05 | **20m 25s** |
| `09:44:10` exp06 | **20m 16s** |
| `10:04:20` exp07 | **20m 10s** |
| `10:24:30` exp08 | **20m 10s** |
| `10:44:41` exp09 | **20m 11s** |
| `11:04:50` exp10 | **20m 09s** |

Corroborated by the older run `0083971f` (2026-04-21): exp34→exp62 spacing is
**20m 30s to 21m 00s** throughout, and the `results/*.json` file mtimes for
`60617e0b-exp03..exp10` are exactly 20 minutes apart (10:43, 11:03, 11:23,
11:44, 12:04, 12:24, 12:44, 13:04).

**Estimate: 20.3 ± 0.3 min per run.** Twelve sequential runs ≈ **4 h 04 m**,
plus one cold-cache first run. This is a multi-hour job, not a minutes job — it
must be run detached with logging, and `stop_check` is worth wiring.

Two caveats on transferring this number:
- Those rows are the optimizer's own configuration (whose `start_date`/
  `end_date` I did not verify equal 2018-2025). If the optimizer ran a shorter
  sample, 20 min is a LOWER bound for the 2018-2025 sample. Confirm from
  `params_json` in the TSV before committing to a schedule.
- **`.claude/rules/backend-backtest.md` claims the warm cache drops
  per-experiment time "from ~5-10min to <30s".** That claim is contradicted by
  every measured inter-row gap above (20 min, warm). The doc is stale; do not
  plan against `<30s`. The BQ cache removes the *query* cost, not the
  per-window feature-building + GradientBoosting training + MDA permutation cost,
  which dominates.

---

## 6. Unequal holding periods — a REAL defect, and it biases AGAINST the candidate

The incumbent labels on `holding_days=90`; `reversion_sigma` labels on
`mr_holding_days=15` (`backtest_engine.py:1500, :1517, :1522`). `stretch_regime`
and `qarp` both use `holding_days` (`:1416, :1430, :1469, :1477`), so only
`reversion_sigma` differs.

**The purge is strategy-blind.** `_build_training_data` computes

```python
horizon_days = int(self.holding_days * 1.5)     # backtest_engine.py:665
```

and uses it for BOTH the purge-overlap test and the recorded exit date
(`:662-673`). It never consults `mr_holding_days`. Consequences:

- For `triple_barrier`, `stretch_regime`, `qarp`: horizon = 135d = correct.
- For `reversion_sigma` (and the existing `mean_reversion`): the true label span
  is ~15-30 days, but the purge drops every biweekly sample date within **135
  days** of the test window. That is a ~9x OVER-purge: roughly 9-10 extra
  biweekly sample dates discarded from the end of every training window.

Direction of the bias: over-purging is **conservative on leakage** (it cannot
create look-ahead) but it **starves the short-horizon candidate of its most
recent training data** while the incumbent loses exactly the data it should.
So a `reversion_sigma` loss is confounded — it may be losing on horizon
mis-specification in the harness rather than on signal quality. A
`reversion_sigma` WIN, by contrast, is clean (it won despite the handicap).

**Recommendation for 82.3:** do not silently accept the asymmetry. Either
(a) make `horizon_days` strategy-aware (use `mr_holding_days*1.5` when the
active label function is the MR/sigma family) — a small, well-scoped change that
should be its OWN masterplan step per the queue-discovered-defects rule; or
(b) run `reversion_sigma` and report the result WITH an explicit disclosure that
its purge horizon is 9x its label horizon, and treat a loss as inconclusive.
Option (b) is the honest zero-code path for this step. External evidence on
whether unequal horizons invalidate the comparison per se is in §8.

**Separately: the embargo is 5 days for everyone** (`WalkForwardScheduler`
default, `walk_forward.py:36`), which under-covers a 135-day label horizon on
the incumbent — but the purge (135d) subsumes it, so the embargo is not the
binding control here.

---

## 7. Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/backtest/analytics.py` | 184-236 | `compute_pbo` (CSCV) | LIVE, 2 non-test callers |
| `backend/backtest/analytics.py` | 649-745 | `generate_report` — emits 11 analytics keys | LIVE; no PBO, no turnover |
| `backend/backtest/analytics.py` | 666-667 | daily returns from NAV (DSR input) | LIVE |
| `backend/backtest/analytics.py` | 744 | `avg_nav` — reusable turnover denominator | LIVE |
| `backend/autoresearch/strategy_backtest_adapter.py` | 1-259 | K-variant → (T,N) matrix → PBO. **The thing to reuse.** | LIVE, tested |
| `backend/autoresearch/strategy_backtest_adapter.py` | 75-91 | `_daily_returns_from_nav` | mirrors `analytics.py:666-667` verbatim |
| `backend/autoresearch/strategy_backtest_adapter.py` | 132-152 | `_assemble_pbo_matrix` (min_rows=32 guard) | LIVE |
| `backend/agents/mcp_servers/risk_server.py` | 142-143 | MCP PBO tool | LIVE |
| `backend/backtest/backtest_engine.py` | 275-395 | `run_backtest` — self-preloading | LIVE |
| `backend/backtest/backtest_engine.py` | 353 | `aggregate_return_pct` from NAV | net of commission |
| `backend/backtest/backtest_engine.py` | 372-386 | `nav_history` build; `all_trades` **capped at 500** | LIVE, cap is a turnover hazard |
| `backend/backtest/backtest_engine.py` | 662-673 | purge, `horizon_days = holding_days*1.5` | **strategy-blind — see §6** |
| `backend/backtest/backtest_engine.py` | 1400-1524 | the 3 phase-82.2 label fns | LIVE, no new ctor kwargs |
| `backend/backtest/backtest_trader.py` | 73-76, 117-119, 155-167, 209-211 | commission computed + deducted from cash on all 3 fill paths | LIVE |
| `backend/backtest/result_store.py` | 23-40 | `save_result` — whole report to JSON | LIVE |
| `scripts/harness/run_harness.py` | 89-111 | `make_engine` — threads a SUBSET of kwargs | LIVE, known gap |
| `scripts/harness/run_harness.py` | 114-126 | `_count_experiments` — TSV line count as `num_trials` | LIVE, see §8 |
| `scripts/ablation/sector_neutral_replay.py` | 205 | basket-overlap turnover (the only in-repo definition) | replay-script only |

---

## 8. External research

### Search-query variants run (3-variant discipline)

1. **Year-less canonical** — `probability of backtest overfitting CSCV number of
   trials N required Bailey Borwein Lopez de Prado Zhu`
2. **Year-less canonical** — `purged k-fold embargo different holding periods
   comparing strategies walk-forward backtest`
3. **Current/last-2-year** — `deflated Sharpe ratio number of trials multiple
   testing strategy selection 2025 2026`

### Read in full (7; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://arxiv.org/html/2512.22476 (AutoQuant) | 2026-08-03 | preprint | WebFetch (arXiv HTML) | "PBO = 0.586 under a standard **8-segment CSCV** design with 70 combinatorial splits over the **top-40** Stage I candidates". Effective trials as a RANGE: "N_total ∈ [N_opt × N_windows, N_opt × N_windows × N_scen]" = **[360, 3240]**. Net return defined per bar: `r_net = r_raw − C_fee − C_slip − C_fund`. Turnover proxied by `switch_density_mean ≤ 0.12` as a "turnover/fragility cap". **"Holding-period heterogeneity is absorbed into per-bar net returns before monthly compounding, making direct cross-configuration comparisons feasible under identical strict semantics despite parameter differences."** |
| https://arxiv.org/html/2507.07107 (ML multi-factor, bias correction) | 2026-08-03 | preprint | WebFetch (arXiv HTML) | "With **N≈50 effective configurations** tried during development … the implied Sharpe-selection threshold is SR₀≈0.93. The realised Sharpe of 1.63 yields DSR=0.978" (§4.2). Costs: `r_net = w'r − c·‖w_t − w_{t−1}‖₁`, `c = 5–8 bps per unit of turnover` (§3.6); results reported "net of 8 bps". Turnover reported as a headline number (0.27–0.37) alongside Sharpe. Caveat: "the DSR cannot correct for biases that affect every configuration uniformly". |
| https://arxiv.org/html/2603.13252 (When Alpha Breaks) | 2026-08-03 | preprint | WebFetch (arXiv HTML) | Directly on the unequal-horizon question: evaluates "three forward-return horizons … τ∈{20,60,90} trading days" but applies a **single uniform 90-trading-day embargo** to all folds, and confines comparative claims to τ=20d — "longer horizons used primarily for **diagnostics rather than comparative policy evaluation**". "Overlapping labels are purged to prevent leakage from forward-return overlap" (§3.3). Cost: "10 basis points per rebalance"; turnover reported as "Mean turnover / month: 42.7%". |
| https://cran.r-project.org/web/packages/pbo/vignettes/pbo.html | 2026-08-03 | doc (reference impl.) | WebFetch | Reference CSCV implementation. "each column represents a trial and each trial has the same length T" — confirms columns=trials / rows=time. Worked example uses **N=100 trials**, `s=8`. Sanity check: random data → `p_bo = 1.0000000`. **No minimum-N guidance is given** — a gap the literature does not close explicitly. |
| https://towardsai.com/p/l/the-combinatorial-purged-cross-validation-method | 2026-08-03 | blog (practitioner) | WebFetch | Embargo scales with the LABEL horizon: "embargo_td = pd.Timedelta(days=1) * t_final". Walk-forward critique: "Just one single scenario is tested, which is easily overfit"; k-fold: "Leakage is possible because the training set does not follow the testing set". Path count `φ[6,2] = 5` from `nCr(6,4)=15`. Does NOT discuss PBO or unequal horizons. |
| https://www.risklab.ai/research/financial-modeling/cross_validation | 2026-08-03 | doc (Lopez de Prado's lab) | WebFetch | Purge definition: "delete any observations from the training set whose labels were derived from a time period that **overlaps** with the time period of any label in the testing set." Embargo = "a small buffer (e.g., **1% of the total data**)". `CombinatorialPurged` exposes `event_starts`/`event_ends` per observation — i.e. a **per-sample** label span, not a global constant. Silent on unequal horizons across strategies. |
| https://arxiv.org/html/2505.14050 (PLUTUS) | 2026-08-03 | preprint | WebFetch (arXiv HTML) | **Negative/disconfirming result, recorded honestly:** cites Bailey et al. 2014 in background but "PBO is not directly discussed"; DSR "not mentioned"; number of trials "not specified"; turnover "not explicitly discussed"; uses "simpler in-sample/out-of-sample splits rather than walk-forward or cross-validation". Evidence that even a 2025 open-source *reproducibility*-focused framework ships without these gates — the pyfinagent gap is the field norm, not an outlier. |

### Identified but snippet-only / failed (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 | paper (the canonical PBO source) | SSRN landing page is abstract-only behind a download wall |
| https://escholarship.org/uc/item/4w1110bb | paper (open-access PBO) | WebFetch returned an EMPTY body; PDF-only item |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 | paper (the canonical DSR source) | SSRN abstract-only |
| https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio | encyclopedia | HTTP 404 on fetch (search index stale) |
| https://en.wikipedia.org/wiki/Purged_cross-validation | encyclopedia | HTTP 404 on fetch (search index stale) |
| https://www.tradinginterview.com/.../cross-validation-for-financial-data-purging-and-embargoing/ | course | Landing page only; lesson body gated |
| https://cran.rstudio.com/web/packages/pbo/readme/README.html | doc | Duplicate of the vignette |
| https://www.pm-research.com/content/iijpormgmt/40/5/94 | journal | Paywalled |
| https://grokipedia.com/page/purged_cross_validation | encyclopedia | Low tier; superseded by risklab.ai |
| https://scholarworks.wmich.edu/math_pubs/42/ | repository | PDF-only mirror of the same PBO paper |
| https://www.semanticscholar.org/paper/b1233b4f5384f003e85c2e0eec1a2dfc08f624c5 | index | Metadata only |
| https://arxiv.org/pdf/2407.17645 (Hopfield / asset allocation) | preprint | Off-topic on inspection |
| https://arxiv.org/pdf/2605.07680 (Fermi-LAT causal backtest) | preprint | Cross-domain; purged-CV adjacent but astrophysics |
| https://arxiv.org/pdf/1910.05555, https://arxiv.org/pdf/1412.5558 | preprints | Off-topic |
| https://medium.com/balaena-quant-insights/deflated-sharpe-ratio-dsr-33412c7dd464 | blog | Community tier; superseded by 2507.07107 |
| https://research.mental-momentum.ai/r/backtest-overfitting-trading-strategy-ju55g3 | blog | Community tier |
| https://scispace.com/papers/..., researchgate x2, Google Books | index/paywall | Metadata only |

~28 unique URLs collected.

### Recency scan (last 2 years, 2024-2026)

**Performed.** Result: **3 new findings that materially shape this step**, none
of which supersede Bailey et al. — they operationalise it.

1. **AutoQuant (arXiv:2512.22476, Dec 2025)** — the closest analogue to 82.3:
   an auto-tuning pipeline that runs CSCV/PBO as an explicit multiple-testing
   diagnostic. It uses **S=8** (70 splits) over **40** candidate columns and
   still lands at PBO=0.586. It also reports the effective trial count as a
   **bounded range** rather than a point estimate, and states the
   holding-period-heterogeneity resolution quoted above.
2. **arXiv:2507.07107 (Jul 2025)** — reports DSR with an explicitly
   *effective* N≈50 covering the whole development search, not just the final
   comparison set, and reports turnover as a first-class headline metric next
   to Sharpe.
3. **arXiv:2603.13252 (Mar 2026)** — the only source that confronts the
   unequal-horizon comparison head-on, and its answer is the conservative one:
   uniform longest-horizon embargo, and no cross-horizon *comparative policy*
   claims.

No 2024-2026 source revises the CSCV construction itself; the S=16 / rows=time /
columns=trials shape in `analytics.py:184-236` remains current.

---

## 9. Answers to the four external questions

### 9a. Is PBO over N=12 defensible? — NO as specified, and this is the plan's weakest link

**What PBO estimates:** the probability that the configuration selected as best
IN-SAMPLE ranks below the median OUT-OF-SAMPLE, across all C(S, S/2) symmetric
partitions of the time axis. It is a property of a **search over competing
configurations of one model**, not a property of a single strategy.

The operator's N=12 is **12 runs total = 3 configs per strategy**, and PBO must
be computed per strategy (§1). With **N=3 columns**, the rank statistic
`omega = (rank - 0.5)/N` (`analytics.py:225`) can take exactly three values —
0.167, 0.5, 0.833 — so the logit distribution has a **3-atom support**
(−1.609, 0, +1.609) that a Gaussian KDE then smooths. The resulting PBO is a
coarse, near-quantised number being compared against a 0.5 threshold. It is not
wrong, but its resolution is far below what the gate language implies.

Calibration against the literature and against this repo:

| Source | N (columns) used |
|---|---|
| CRAN `pbo` reference vignette | **100** |
| AutoQuant (arXiv:2512.22476) | **40** |
| arXiv:2507.07107 (DSR N) | **~50** effective |
| **`strategy_backtest_adapter.py:70` `_DEFAULT_K`** | **8** |
| Operator's 82.3 plan | **3** |

The repo's own default is 8 per strategy. **N=3 is below pyfinagent's own
default and an order of magnitude below the literature.** Note also that no
source states an explicit minimum N — so this is an argument from calibration,
not from a cited threshold; say so rather than inventing a rule.

**Recommendation (pick one, state which in the contract):**

- **Preferred: K=8 per strategy** = 32 runs ≈ **10.8 h** at the measured 20.3
  min/run. Matches `_DEFAULT_K` and gets omega granularity to 1/8. Run detached
  overnight.
- **Compromise: K=6** = 24 runs ≈ **8.1 h**.
- **As-specified: K=3** = 12 runs ≈ **4.1 h** — acceptable ONLY if the brief and
  the results file state plainly that PBO at N=3 is directionally indicative and
  is **not** a gate-quality number, and the promotion gate is not asserted to
  have been satisfied on it.

Also set **S=8, not 16**, if T is comfortable: AutoQuant's "standard 8-segment
CSCV" with 70 splits is the current practitioner default, and S=8 lowers the
row floor from T≥32 to T≥16 while keeping ~240 rows/subset on a 1,900-row daily
series. Either is defensible; S=16 is the paper convention and the repo default.
Whichever is used, **assert `T >= S*2` before calling `compute_pbo` and fail
loudly** — a silent 0.0 return (`analytics.py:205-206`) passes the gate.

### 9b. DSR `num_trials` for a 12-configuration comparison

DSR deflates the observed Sharpe by the selection threshold implied by taking
the max of N trials. Both recent sources count **development** trials, not the
final comparison set: N≈50 "effective configurations tried during development"
(2507.07107 §4.2), and a **range** [360, 3240] with the lower bound excluding
scenario evaluations and the upper bound treating each as independent
(AutoQuant §2.6, explicitly "a conservative assumption about correlation
structure").

For 82.3:

- Do **not** pass `num_trials=1`. That is the un-deflated Sharpe wearing a DSR
  label.
- Passing `num_trials=12` (or K per strategy) **over-deflates**, because the
  configs are highly correlated — which is the SAFE direction and is already the
  in-repo position (`strategy_backtest_adapter.py:44-45`: "plain num_trials=K
  over-deflates -- the SAFE direction").
- Do **not** use `run_harness._count_experiments()` (`run_harness.py:114-126`),
  which returns the **TSV line count — currently 531** — mixing every historical
  optimizer experiment across different strategies, samples and dates into one N.
  That is not "all trials for this discovery"; it is a file length.
- **Report a range, AutoQuant-style:** `DSR(num_trials=K_per_strategy)` as the
  headline (conservative), plus `DSR(num_trials=N_total_search)` where
  `N_total_search` is the honest count of configurations evaluated while
  designing 82.2 + 82.3. State both numbers and the assumption behind each.
- Effective-N via ONC clustering (AFML Ch.16) would give the principled middle
  number; it is **not implemented in this repo** and should not be invented
  inside 82.3.

Also remember DSR's blind spot, quoted directly: "the DSR **cannot correct for
biases that affect every configuration uniformly**" (2507.07107 §4.2). All four
strategies here share one feature builder, one screener, one purge horizon and
one universe — so a uniform bias (e.g. the survivorship profile of the S&P 500
constituent list) is invisible to DSR across the whole comparison.

### 9c. Turnover and cost drag in walk-forward backtests

Both recent sources treat turnover as a **reported metric alongside Sharpe**,
not as a pass/fail gate, and both express cost as a **linear function of
turnover**: `c·‖w_t − w_{t−1}‖₁` with `c = 5–8 bps` (2507.07107 §3.6), 10 bps
per rebalance (2603.13252 §3.4), and `switch_density_mean ≤ 0.12` as an explicit
"turnover/fragility cap" (AutoQuant Table 8). The 2507.07107 ablation makes the
point that matters here: Ledoit-Wolf shrinkage **cut turnover 0.33 → 0.27** with
no Sharpe loss, i.e. turnover differences between configs are large enough to
dominate net returns.

Application: pyfinagent's cost model is already inside the NAV (§3), so its
net-of-cost returns are directly comparable to these papers' `r_net`. Report
turnover per §2 as a **descriptive** metric — there is no turnover threshold in
the pyfinagent promotion gate, and 82.3 should not invent one. Its diagnostic
value is specifically for `reversion_sigma`: a 15-day label horizon should
produce visibly higher turnover than a 90-day one, and if it does not, the exit
logic is not honouring the label horizon.

### 9d. VERDICT on unequal holding periods — the comparison is VALID but BIASED

**Unequal holding periods do NOT invalidate a like-for-like comparison here.**
Two independent lines of evidence, plus one internal fact:

1. **AutoQuant §2.4, directly on point:** "Holding-period heterogeneity is
   absorbed into per-bar net returns before monthly compounding, making direct
   cross-configuration comparisons feasible under identical strict semantics
   despite parameter differences." The condition is that the comparison happens
   on a **common time grid of net returns**, not on per-trade or per-signal
   statistics. pyfinagent satisfies this: daily mark-to-market produces one NAV
   per business day for every strategy regardless of horizon
   (`backtest_engine.py:372-375`).
2. **arXiv:2603.13252 §3.3/§5.1:** faced with τ∈{20,60,90}, they apply a
   **single uniform 90-day embargo** — the longest horizon — to every fold.
   pyfinagent does the same thing, though by accident: `horizon_days =
   holding_days * 1.5 = 135d` is applied to all strategies
   (`backtest_engine.py:665`), and 135d ≥ every strategy's true label span. So
   the leakage control is **conservative and uniform**, which is exactly the
   published practice.
3. **Purging is defined per-observation on the label span** (risklab.ai:
   `event_starts`/`event_ends`; towardsai: `embargo_td = 1 day * t_final`), so
   the textbook-correct implementation would use each strategy's own horizon.
   pyfinagent's global constant is a simplification in the SAFE direction.

**But the comparison IS biased, and the bias is one-directional.** The 135-day
purge is ~9x `reversion_sigma`'s true 15-day label span, so `reversion_sigma`
discards ~9-10 extra biweekly training samples per window that it never needed
to discard (§6). It competes with less training data than the incumbent, on data
that is also less recent. Therefore:

- A `reversion_sigma` **WIN is clean** — it won carrying a handicap.
- A `reversion_sigma` **LOSS is inconclusive** — the loss is confounded with
  horizon mis-specification in the harness and must be reported as such, not as
  evidence against the signal.
- `stretch_regime` and `qarp` are **unaffected** (both label on `holding_days`).

The one thing 82.3 must NOT do is what 2603.13252 explicitly declines to do:
promote a cross-horizon result to a **comparative policy claim** without the
disclosure. Report the asymmetry in the results file, and queue the
strategy-aware `horizon_days` fix as its own masterplan step.

---

## 10. Application to pyfinagent — concrete plan for 82.3

1. **Reuse, do not rebuild.** `make_engine_backtest_fn`
   (`strategy_backtest_adapter.py:167-256`) already does: K-variant grid → warm
   -cache loop with `skip_cache_clear=True` → `_assemble_pbo_matrix` →
   `compute_pbo` → `generate_report` DSR. Supply an `engine_factory` and a
   `param_grid_fn`; do not re-derive the matrix.
2. **Verify the factory threads every param the 12 configs vary**
   (`run_harness.py:89-111` omits `target_vol` / trailing / blend — silent
   ignore, flagged at `strategy_backtest_adapter.py:49-53`).
3. **PBO per strategy, on daily NAV returns, with an explicit `T >= S*2`
   assertion.** Never let a 0.0 reach the gate unasserted.
4. **DSR as a range** (§9b), never `num_trials=1`, never the TSV line count.
5. **Turnover from `trader.trades` (uncapped) and cross-checked against
   `trader.total_commission`** — not from `result.all_trades` (500-cap,
   `backtest_engine.py:380`).
6. **State plainly that returns are already net of commission** (§3) and that
   no slippage/spread/impact is modelled.
7. **Budget ~4 h (K=3) to ~11 h (K=8)** of wall clock, detached, with
   `engine.stop_check` wired. Do not plan against the stale "<30s warm" claim
   in `.claude/rules/backend-backtest.md`.
8. **Add `pbo` and `turnover` keys to the saved report explicitly** — nothing in
   `generate_report` or `result_store.save_result` will write them for you, and
   `quant_results.tsv` has no column for either.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**7**)
- [x] 10+ unique URLs total incl. snippet-only (**~28**)
- [x] Recency scan (2024-2026) performed + reported (§8, 3 findings)
- [x] Full pages read, not abstracts, for the read-in-full set (arXiv HTML path
      used for all 4 preprints per `.claude/rules/research-gate.md`; SSRN/PDF
      landing pages explicitly excluded from the read-in-full table)
- [x] file:line anchors for every internal claim (§1-§7)

Soft checks:
- [x] Internal exploration covered every relevant module (§7 inventory)
- [x] Contradictions noted (spawn-premise correction §0; stale `<30s` doc claim
      §5; strategy-blind purge §6; PLUTUS null result §8)
- [x] All claims cited per-claim
- [ ] Canonical Bailey PBO/DSR papers not read in primary form — SSRN and
      escholarship are abstract/PDF-gated. Mitigated by the CRAN reference
      implementation + the in-repo implementation + 2 recent papers that apply
      the method. Disclosed, not papered over.

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 17,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "compute_pbo has 2 non-test callers (strategy_backtest_adapter.py:247, risk_server.py:143), not zero -- 82.3 should reuse the adapter, not rebuild. PBO's T-axis must be DAILY NAV returns (T~1900), not per-window returns (T~27 < S*2=32 triggers a SILENT 0.0 that PASSES the <=0.5 gate). Turnover is computed nowhere in backend/; use traded_notional/(2*avg_nav)/years from trader.trades (result.all_trades is capped at 500). total_return_pct IS already net of commission -- deducted from cash on all 3 fill paths (backtest_trader.py:117,166,210) -- do not double-count. run_backtest self-preloads all 3 caches; use make_engine + skip_cache_clear=True. RUNTIME measured at 20.3 min/run warm (10 consecutive TSV gaps), so 12 runs ~4.1h and the repo's own K=8 default would be ~10.8h; the '<30s warm' doc claim is stale. Unequal holding periods do NOT invalidate the comparison (common daily-NAV grid + uniform longest-horizon purge = published practice) BUT the purge is strategy-blind at holding_days*1.5, over-purging reversion_sigma ~9x, so a loss is inconclusive and a win is clean. N=3 columns/strategy is below the repo default (8) and the literature (40-100); PBO at N=3 is indicative, not gate-quality.",
  "brief_path": "handoff/current/research_brief_82.3.md",
  "gate_passed": true
}
```

