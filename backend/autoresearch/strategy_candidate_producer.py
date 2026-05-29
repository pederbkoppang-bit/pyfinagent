"""phase-48.1: per-strategy DSR/PBO PRODUCER + the registry->producer->selector spine.

This is the missing producer half of dynamic strategy rotation (Priority 5). The
47.6 `select_best_strategy` consumes a `per_strategy` list of scored candidates;
nothing produced it. This module maps the seed registry to that list and wires
the full bake-off.

The single dependency is an injected ``backtest_fn(params: dict) -> dict``. This
isolates ALL slow/real I/O (the BacktestEngine, BQ, macro preload) behind a pure
boundary so the producer is unit-testable at $0 today, and the real-engine
adapter is a drop-in next cycle (see DEFERRED below).

Per-strategy contract emitted (verified against strategy_selector.py:54-98 and
gate.py:24-39): one dict per strategy with keys
    {"strategy": <registry id>, "dsr": float, "pbo": float, "params": dict, "sharpe": float}
``dsr`` and ``pbo`` are BOTH MANDATORY -- `PromotionGate.evaluate` (gate.py:28)
silently drops any candidate missing either ("missing_dsr_or_pbo"). The producer
therefore SKIPS (with a warning) any strategy whose backtest_fn raises or omits
dsr/pbo, rather than emitting a partial dict that would vanish from the bake-off
with no error.

Pure, fail-open, ASCII-only.

DEFERRED (NOT built here -- later cycles):
- The REAL ``backtest_fn``: construct ONE warm BacktestEngine, loop seeds calling
  ``run_backtest(skip_cache_clear=True)``, derive daily_returns =
  np.diff(navs)/navs[:-1] from ``result.nav_history``, feed
  ``analytics.generate_report(result, num_trials=N)["analytics"]`` for dsr+sharpe,
  build a per-strategy (T x K-trial) matrix for ``analytics.compute_pbo``, then
  ``cache.clear_cache()`` once at the end. The injected fn's OUT shape here is a
  strict SUBSET of generate_report()["analytics"] + compute_pbo so wiring is an
  adapter, not a rewrite.
- The weekly rotation CRON that calls run_strategy_bakeoff on a schedule.
- The DEPLOYMENT switch + params->settings.paper_* bridge: per the deploy audit,
  best_params is NOT threaded into decide_trades/paper_trader (live behavior is
  settings.paper_* driven), so flipping a promoted_strategies row alone changes
  only the strategy_decisions heartbeat, NOT live orders. A real rotation must
  bridge params->settings (sl_pct->paper_default_stop_loss_pct,
  max_positions->paper_max_positions, ...). Its own cycle.
- Effective-N clustering: plain num_trials=N over-deflates DSR for correlated
  seeds (the SAFE direction); deferred.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from backend.autoresearch.strategy_registry import load_seed_strategies
from backend.autoresearch.strategy_selector import select_best_strategy

logger = logging.getLogger(__name__)

BacktestFn = Callable[[dict[str, Any]], dict[str, Any]]


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_per_strategy_candidates(
    strategy_configs: list[dict[str, Any]],
    backtest_fn: BacktestFn,
) -> list[dict[str, Any]]:
    """Score each strategy config into a selector-ready candidate dict.

    Args:
        strategy_configs: output of ``load_seed_strategies`` -- each
            ``{"id", "rationale", "params"}``.
        backtest_fn: injected ``(params) -> metrics`` where metrics has at least
            ``{"dsr": float, "pbo": float}`` and optionally ``{"sharpe": float}``.
            ALL real backtest/BQ/engine I/O lives behind this boundary.

    Returns one ``{"strategy", "dsr", "pbo", "params", "sharpe"}`` dict per config
    that produced complete metrics. A config whose backtest_fn raises OR returns
    metrics missing/non-numeric dsr|pbo is SKIPPED (warning logged) -- never
    emitted as a partial dict, so the PromotionGate never silently drops it.

    Pure w.r.t. its inputs; the only side effect is logging.
    """
    candidates: list[dict[str, Any]] = []
    for config in strategy_configs or []:
        sid = str(config.get("id") or "").strip()
        if not sid:
            logger.warning("[candidate_producer] skipping config with no id: %r", config)
            continue
        params = config.get("params") or {}
        try:
            metrics = backtest_fn(params)
        except Exception as exc:
            logger.warning(
                "[candidate_producer] backtest_fn raised for strategy '%s' (%s); "
                "skipping (not emitting a partial candidate)", sid, exc,
            )
            continue
        if not isinstance(metrics, dict):
            logger.warning(
                "[candidate_producer] backtest_fn returned non-dict for '%s'; skipping", sid,
            )
            continue
        dsr = _coerce_float(metrics.get("dsr"))
        pbo = _coerce_float(metrics.get("pbo"))
        if dsr is None or pbo is None:
            logger.warning(
                "[candidate_producer] strategy '%s' metrics missing/invalid dsr|pbo "
                "(dsr=%r pbo=%r); skipping so the gate cannot silently drop it",
                sid, metrics.get("dsr"), metrics.get("pbo"),
            )
            continue
        sharpe = _coerce_float(metrics.get("sharpe"))
        candidates.append(
            {
                "strategy": sid,
                "dsr": dsr,
                "pbo": pbo,
                "params": params,
                "sharpe": sharpe if sharpe is not None else 0.0,
            }
        )
    return candidates


def run_strategy_bakeoff(
    backtest_fn: BacktestFn,
    incumbent: Optional[dict[str, Any]] = None,
    *,
    seeds: Optional[list[dict[str, Any]]] = None,
    base_params: Optional[dict[str, Any]] = None,
    num_trials: Optional[int] = None,
) -> dict[str, Any]:
    """Full registry -> producer -> selector spine.

    Loads the seed registry, scores each via ``backtest_fn``, and runs the 47.6
    ``select_best_strategy`` (DSR>=0.95/PBO<=0.20 gate + DSR-desc/PBO-asc rank +
    anti-churn hysteresis). Returns the selector's verdict dict
    ``{selected, selected_id, switched, reason, ranked, incumbent_id, num_trials,
    delta_dsr}``.

    ``num_trials`` defaults to the number of seed configs (the count fed to the
    DSR deflation; the over-deflation for correlated seeds is the SAFE direction
    -- effective-N clustering is deferred).
    """
    configs = load_seed_strategies(seeds=seeds, base_params=base_params)
    per_strategy = build_per_strategy_candidates(configs, backtest_fn)
    n = num_trials if num_trials is not None else len(configs)
    return select_best_strategy(per_strategy, incumbent=incumbent, num_trials=n)


__all__ = ["build_per_strategy_candidates", "run_strategy_bakeoff", "BacktestFn"]
