# Research Brief — phase-48.1: Strategy-rotation foundation (registry + per-strategy DSR/PBO producer)

**Produced by a 4-agent Workflow** (run `wf_784c2e77-298`): a research-gate literature agent (`researcher` type) + 2 parallel codebase auditors (backtest-engine interface, deployment/param-space) + a design synthesizer. This brief consolidates their structured outputs.

## Research gate (external literature) — PASSED
`gate_passed: true`, **8 sources read in full** (>= the 5 floor), recency scan performed (2026 + last-2-year + year-less canonical). Source hierarchy honored (peer-reviewed papers + official docs + named-quant practitioners).

Sources read in full:
- davidhbailey.com/dhbpapers/deflated-sharpe.pdf — DSR formula, E[max SR]~sqrt(2 log N)*sigma_SR, gamma=0.5772, DSR>0.95 threshold.
- en.wikipedia.org/wiki/Deflated_Sharpe_ratio — verbatim DSR + SR0 + PSR; effective-N via clustering.
- vertoxquant.com/.../effective-number-of-tested-strategies — K_eff (effective trials), 5 axioms, critique of Bailey&LdP's own effective-N measure.
- arxiv.org/html/2402.05272v2 (Shu/Yu/Mulvey) — statistical jump model; switch penalty cut turnover 141%->44% at 10bps while improving net-of-cost Sharpe (empirical basis for the selector's anti-churn hysteresis).
- arxiv.org/html/2603.09219 (AlgoXpert 2026) — IS/WFA/OOS protocol: plateau selection, purge gap, majority-pass gate; restrict degrees of freedom -> fewer effective trials -> less-inflated DSR.
- buildalpha.com/trading-ensemble-strategies — >=2-strategy ensemble floor; "polishes gold, does not turn dirt into gold" (each seed must clear an individual quality bar).
- cran.r-project.org/.../pbo/readme — CSCV procedure: T x N matrix, S partitions (8/16), PBO = fraction of splits where IS-best lands below OOS median.
- sdm.lbl.gov/.../ssrn-id2507040 (Bailey & LdP) — statistical overfitting, Minimum Backtest Length scales with log(N).
- (snippet-only / blocked: escholarship PBO primary [429], MDPI GT-Score [403] read via arXiv mirror, algomatictrading correlation blog, arXiv 1511.07945 NCO/HRP, SSRN DSR landing.)

**Key findings shaping the design:**
1. **Diversification comes from orthogonal AXES, not parameter tweaks** — strategy TYPE (mean-reversion vs trend/momentum vs breakout), market, timeframe. MR + trend are structurally anti-correlated (different drawdown timing -> lower maxDD). So the seed set spans strategy TYPES (triple_barrier / mean_reversion / quality_momentum), not just risk-knob variants of one model.
2. **Effective-N**: correlated param-variants are NOT independent trials. The selector's plain `num_trials=N` (count) OVER-deflates DSR — the SAFE direction. Principled fix (N_eff clustering) is DEFERRED; the over-conservatism is documented, not a bug.
3. **Keep the seed set small + pre-registered** (E[max SR]~sqrt(2 log N)): recommended >=4, not a sweep of hundreds.
4. **Each seed must clear the individual DSR>=0.95/PBO<=0.20 gate on its own** — never seed a weak strategy expecting ensembling to rescue it; the gate is reused unchanged.
5. **Anti-churn hysteresis is empirically justified** (jump-model turnover result) — the selector's `min_improvement` term is correct.
6. **[Adversarial, qualifying]** GT-Score (2026) argues DSR/PBO are post-hoc and complementary to a robustness-aware objective. Doesn't displace DSR/PBO-as-gate; noted as a future enhancement to the per-strategy objective.

## Internal audit — backtest engine (for the future real-engine adapter)
- Entrypoint: `BacktestEngine.run_backtest(universe_tickers=None, skip_cache_clear=False) -> BacktestResult` (`backend/backtest/backtest_engine.py:266`; constructor `:136`). `strategy` kwarg ∈ {triple_barrier, quality_momentum, mean_reversion, factor_model, meta_label} (+ blend via optimizer).
- Output: `BacktestResult` with `nav_history: [{date, nav, cash}, ...]` (~500+ daily rows = the equity curve). Daily returns: `np.diff(navs)/navs[:-1]`.
- DSR/Sharpe: `backend.backtest.analytics.generate_report(result, num_trials=N)["analytics"]` → `{sharpe, deflated_sharpe (DSR in [0,1]), dsr_significant, ...}` (`analytics.py:536`). PBO: `analytics.compute_pbo(pnl_matrix, S=16)` (`:184`) — requires a (T, N) matrix, T rows, N>=2, T>=2S.
- `macro_preload_needed: false` — `run_backtest` calls `cache.preload_macro()` internally (`:299-302`); the ~40min hang only affects code paths that bypass it.
- **`is_slow_minutes: true`** — a cold single backtest is minutes (walk-forward + GBC training + daily mark-to-market). Warm-cache pattern: one engine, `skip_cache_clear=True` per subsequent run, `cache.clear_cache()` once at end. **This is exactly why the producer takes an injected `backtest_fn` — the slow/real I/O is isolated behind a pure boundary and DEFERRED to the next cycle's adapter.**

## Internal audit — deployment / param space
- Param source: `backend/backtest/experiments/optimizer_best.json` (single config; `params` object has 25 keys incl. strategy, tp_pct, sl_pct, holding_days, mr_holding_days, target_annual_vol, trailing_*, max_positions, top_n_candidates, ML hyperparams).
- Live switch: `backend/services/autonomous_loop.py::load_promoted_params(bq)` — 3-tier (promoted_strategies BQ row → optimizer_best.json → fail-open fallback).
- Differentiating axes (ranked by live impact): (1) **strategy categorical** (biggest lever), (2) holding_days/mr_holding_days (turnover regime), (3) tp/sl/target_vol/trailing (risk-exit), (4) max_positions/top_n (concentration), (5) signal/blend weights.
- **CRITICAL deploy finding (shapes a DEFERRED follow-on):** `best_params` is loaded but **NOT threaded into `decide_trades`/`paper_trader`** — live risk/sizing/turnover is driven entirely by `settings.paper_*`. So a real rotation deployment must ALSO bridge params→settings (map sl_pct→paper_default_stop_loss_pct, max_positions→paper_max_positions, ...). Flipping a `promoted_strategies` row alone changes only the heartbeat label, not live orders. **This cycle does NOT build the deployment switch; it is documented as a hard prerequisite for a later cycle.**

## Verified selector/gate contract (read directly from code, not just the audit)
- `select_best_strategy(per_strategy, incumbent=None, *, gate=PromotionGate(), min_improvement=0.01, num_trials=5)` (`strategy_selector.py:60`). id resolved under `strategy_id|strategy|trial_id` (`:54-57`); ranks DSR-desc/PBO-asc (`:95-98`).
- `PromotionGate.evaluate` (`gate.py:24-39`): **drops** any candidate where `dsr` OR `pbo` is None ("missing_dsr_or_pbo"); promotes iff `dsr>=0.95 AND pbo<=0.20`.
- Producer therefore MUST emit BOTH `dsr` and `pbo` as floats per strategy, under id key `strategy`.

## JSON envelope
```json
{"tier":"complex","external_sources_read_in_full":8,"snippet_only_sources":6,"urls_collected":14,"recency_scan_performed":true,"internal_files_inspected":12,"gate_passed":true}
```
