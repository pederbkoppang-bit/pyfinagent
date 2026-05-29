"""phase-48.2: REAL BacktestEngine adapter for the strategy-rotation producer.

phase-48.1 shipped the producer/registry/selector spine with the per-strategy
scoring isolated behind an injected ``backtest_fn(params) -> {dsr, pbo, sharpe}``
boundary (strategy_candidate_producer.BacktestFn). This module supplies the REAL
implementation of that boundary, backed by ``backtest_engine.BacktestEngine``.

THE PBO METHOD (research-grounded, research_brief_phase_48_2_rotation_adapter.md):
a single backtest's walk-forward WINDOWS cannot be the CSCV columns -- Bailey,
Borwein, Lopez de Prado, Zhu (2016) Algorithm 2.3: "each column n=1..N represents
a vector of profits and losses associated with a particular model CONFIGURATION";
the IS/OOS split is over ROWS (time). One PnL series => analytics.compute_pbo
(N<2 guard) returns 0.0, which PASSES the gate's pbo<=0.20 -- a FALSE pass. So per
strategy we run K param-variant CONFIGURATIONS (same ``strategy`` categorical,
jittered risk knobs), derive each variant's daily returns from its nav_history,
and stack them as the N=K COLUMNS of a (T x K) matrix for compute_pbo. DSR comes
from generate_report(...)["analytics"]["deflated_sharpe"] on the SEED variant
(generate_report does NOT compute PBO -- only Sharpe + DSR).

LOAD-BEARING GUARD: compute_pbo SILENTLY returns 0.0 when N<2 OR T<S*2 (default
needs T>=32, N>=2). A degenerate 0.0 PASSES the pbo<=0.20 gate. So when the
assembled matrix is undersized, this adapter returns a dict WITHOUT a ``pbo`` key
-- the producer (strategy_candidate_producer.py) then SKIPS that strategy rather
than emitting a fake-good 0.0 that the gate would wave through.

The adapter imports NO settings / no BigQuery: the caller supplies an
``engine_factory`` that closes over them (run_harness.make_engine is the
precedent). This keeps the adapter pure of config and $0-unit-testable by mocking
``engine.run_backtest`` while the REAL pure-numpy generate_report + compute_pbo
run on the fake result.

Pure helpers + one factory. Fail-open, ASCII-only.

DEFERRED (NOT built here -- later cycles):
- The LIVE multi-run bake-off (the 4-seed set x K variants = ~32 real backtests,
  tens of minutes even warm-cached) -- gated behind the @pytest.mark.skip opt-in
  integration test + a dedicated live-run cycle.
- CPCV multi-path PBO upgrade (cpcv_folds already exists at gate.py:42; AFML
  Ch.12; Arian/Norouzi/Seco 2024 = lowest PBO / highest DSR). It yields a
  robustness OOS-Sharpe DISTRIBUTION, not the rank-degradation PBO scalar the
  gate consumes, so it complements -- not replaces -- the param-grid.
- The weekly rotation cron; the deployment params->settings.paper_* bridge
  (best_params is NOT threaded into decide_trades -- flipping a promoted_strategies
  row alone changes only the heartbeat, not live orders); effective-N (ONC)
  clustering (plain num_trials=K over-deflates -- the SAFE direction).
- A true date-keyed matrix join (here we truncate variants to the shortest common
  length -- benign because variants of one strategy share the bdate grid).

RISK (flagged, not blocking the mock slice): run_harness.make_engine threads only
a SUBSET of constructor kwargs (no target_vol/trailing/blend), so a live run with
the vanilla factory would SILENTLY ignore the tb_risk_managed seed's risk
overrides. The adapter is engine-factory-agnostic; extending the factory to thread
those kwargs is the live-wiring caller's responsibility.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from backend.backtest.analytics import compute_pbo, generate_report
from backend.backtest.backtest_engine import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)

EngineFactory = Callable[[dict], Any]            # (params) -> BacktestEngine-like with .run_backtest()
ParamGridFn = Callable[[dict, int], list[dict]]  # (seed_params, K) -> K competing-config param dicts

_DEFAULT_K = 8
_DEFAULT_PBO_S = 16
_DEFAULT_MIN_PBO_ROWS = 32


def _daily_returns_from_nav(nav_history: Optional[list[dict]]) -> np.ndarray:
    """Daily returns from a nav_history list of {date, nav, ...}.

    Mirrors analytics.generate_report (analytics.py:553-554) verbatim so the
    DSR-side and PBO-side returns are computed identically. Returns an empty
    array (not a raise) when there are <3 navs or a zero divisor -- fail-open so
    the column is simply dropped from the PBO matrix.
    """
    if not nav_history or len(nav_history) < 3:
        return np.array([])
    try:
        navs = np.array([float(n["nav"]) for n in nav_history], dtype=float)
    except (KeyError, TypeError, ValueError):
        return np.array([])
    if navs.size < 3 or np.any(navs[:-1] == 0):
        return np.array([])
    return np.diff(navs) / navs[:-1]


def _default_param_grid(seed_params: dict, k: int) -> list[dict]:
    """K competing-config variants around a seed (the CSCV columns for ONE strategy).

    The ``strategy`` categorical is held FIXED (Bailey Algo 2.3: columns are
    configs of the SAME model); only the strategy-appropriate risk knob (plus a
    small tp_pct jitter) is stepped. Validates the strategy name against
    STRATEGY_REGISTRY FIRST and RAISES on an unknown name -- so the producer
    skips that strategy rather than the engine silently falling back to
    triple_barrier (backtest_engine.py:199).
    """
    strategy = (seed_params or {}).get("strategy")
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"unknown strategy {strategy!r}; must be one of {sorted(STRATEGY_REGISTRY)}"
        )
    k = max(int(k), 1)
    base = dict(seed_params)
    variants: list[dict] = [dict(base)]  # variant 0 == the seed itself
    if strategy == "mean_reversion":
        knob, default, lo, hi = "mr_holding_days", 8, 5, 30
    else:
        knob, default, lo, hi = "holding_days", base.get("holding_days", 90), 30, 252
    center = int(base.get(knob, default) or default)
    step = max(1, int(abs(center) * 0.08))
    for i in range(1, k):
        v = dict(base)
        sign = 1 if (i % 2) else -1
        mag = (i + 1) // 2
        v[knob] = int(min(hi, max(lo, center + sign * step * mag)))
        if "tp_pct" in base:
            try:
                v["tp_pct"] = round(max(2.0, float(base["tp_pct"]) * (1 + 0.05 * sign * mag)), 3)
            except (TypeError, ValueError):
                pass
        variants.append(v)
    return variants[:k]


def _assemble_pbo_matrix(results: list[Any], min_rows: int) -> Optional[np.ndarray]:
    """Stack K variants' daily-return series into a (T, N) matrix for compute_pbo.

    Each result's nav_history -> a daily-return column. Truncates all columns to
    the shortest common length (benign: variants of one strategy share the bdate
    grid). Returns None when there are <2 usable columns OR T < min_rows -- the
    LOAD-BEARING guard against compute_pbo's silent 0.0 (which would false-pass
    the pbo<=0.20 gate).
    """
    cols: list[np.ndarray] = []
    for r in results:
        nav = r.get("nav_history") if isinstance(r, dict) else getattr(r, "nav_history", None)
        rets = _daily_returns_from_nav(nav)
        if rets.size > 0:
            cols.append(rets)
    if len(cols) < 2:
        return None
    min_len = min(c.size for c in cols)
    if min_len < int(min_rows):
        return None
    return np.column_stack([c[:min_len] for c in cols])  # (T, N)


def _extract_dsr_sharpe(seed_result: Any, num_trials: int) -> tuple[float, float]:
    """(dsr, sharpe) from the SEED variant via the REAL generate_report.

    Mirrors run_harness.run_backtest (run_harness.py:132-135): generate_report
    wraps compute_deflated_sharpe (observed_sr=aggregate_sharpe, variance from
    per-window Sharpes). dsr == analytics["deflated_sharpe"] (in [0,1]).
    """
    rep = generate_report(seed_result, num_trials=max(int(num_trials), 1))
    a = rep.get("analytics", {}) if isinstance(rep, dict) else {}
    return float(a.get("deflated_sharpe", 0.0) or 0.0), float(a.get("sharpe", 0.0) or 0.0)


def make_engine_backtest_fn(
    engine_factory: EngineFactory,
    *,
    num_param_variants: int = _DEFAULT_K,
    param_grid_fn: Optional[ParamGridFn] = None,
    num_trials: Optional[int] = None,
    pbo_S: int = _DEFAULT_PBO_S,
    min_pbo_rows: int = _DEFAULT_MIN_PBO_ROWS,
    clear_cache_fn: Optional[Callable[[], None]] = None,
    log: Optional[logging.Logger] = None,
) -> Callable[[dict], dict]:
    """Build the producer's ``backtest_fn(params) -> {dsr, pbo, sharpe, ...}``.

    Per strategy: validate the name, build K competing-config variants, run each
    via ``engine_factory(variant).run_backtest(skip_cache_clear=True)`` (warm
    cache shared across variants via the module-level BQ cache; macro preload is
    INSIDE run_backtest), read DSR/Sharpe from the seed variant's generate_report,
    assemble the (T x K) matrix and call compute_pbo. ``clear_cache_fn`` is called
    ONCE in a finally (warm-cache discipline); if None it lazily imports
    backend.backtest.cache.clear_cache (kept lazy so module import stays light and
    tests can inject a stub).

    On an undersized PBO matrix the result OMITS ``pbo`` so the producer SKIPS the
    strategy. Raises ValueError on an unknown strategy name (producer catches +
    skips). All real I/O is behind ``engine_factory`` -> $0-mockable.
    """
    _log = log or logger
    grid_fn = param_grid_fn or _default_param_grid

    def backtest_fn(params: dict) -> dict:
        strategy = (params or {}).get("strategy")
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(
                f"unknown strategy {strategy!r}; refusing silent triple_barrier fallback"
            )
        grid = grid_fn(params, num_param_variants)
        if not grid:
            raise ValueError("empty param grid")

        results: list[Any] = []
        try:
            for i, variant in enumerate(grid):
                try:
                    engine = engine_factory(variant)
                    results.append(engine.run_backtest(skip_cache_clear=True))
                except Exception as exc:  # one bad variant drops a column, not the strategy
                    _log.warning(
                        "[adapter] variant %d for strategy '%s' failed (%s); dropping column",
                        i, strategy, exc,
                    )
        finally:
            _clear = clear_cache_fn
            if _clear is None:
                try:
                    from backend.backtest.cache import clear_cache as _clear  # lazy, avoid heavy top import
                except Exception:
                    _clear = None
            if _clear is not None:
                try:
                    _clear()
                except Exception as exc:
                    _log.warning("[adapter] clear_cache failed (%s)", exc)

        if not results:
            raise RuntimeError(f"all {len(grid)} variants failed for strategy '{strategy}'")

        n = num_trials if num_trials is not None else len(grid)
        dsr, sharpe = _extract_dsr_sharpe(results[0], n)
        n_windows = len(getattr(results[0], "windows", []) or [])

        matrix = _assemble_pbo_matrix(results, min_pbo_rows)
        if matrix is None:
            _log.warning(
                "[adapter] strategy '%s': PBO matrix undersized/degenerate "
                "(need >=2 cols and >=%d rows; got %d usable variant(s)); emitting NO pbo "
                "so the producer SKIPS (not a false-good 0.0)",
                strategy, int(min_pbo_rows), len(results),
            )
            return {"dsr": dsr, "sharpe": sharpe, "n_variants": len(results), "n_windows": n_windows}

        pbo = float(compute_pbo(matrix, S=pbo_S))
        return {
            "dsr": dsr,
            "pbo": pbo,
            "sharpe": sharpe,
            "n_variants": len(results),
            "n_windows": n_windows,
        }

    return backtest_fn


__all__ = ["make_engine_backtest_fn", "EngineFactory", "ParamGridFn"]
