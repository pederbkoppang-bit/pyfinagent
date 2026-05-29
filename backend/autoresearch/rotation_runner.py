"""phase-48.3: live rotation runner + full-kwarg BacktestEngine factory.

Glue over the rotation stack: registry (48.1) -> real-engine adapter (48.2) ->
producer (48.1) -> selector (47.6). It builds a FULL-constructor-kwarg engine
factory (closing the gap that run_harness.make_engine drops 8 ctor kwargs),
resolves the live INCUMBENT, runs the bake-off, and PERSISTS the verdict for
audit at allocation_pct=0 -- WITHOUT deploying.

KWARG-GAP FIX (research_brief_phase_48_3_rotation_runner.md): make_engine threads
only 12 of ~25 BacktestEngine.__init__ kwargs. make_rotation_engine threads the
full set, most importantly **target_vol** -- which IS read (backtest_trader.py:89
inverse-vol sizing; 0 disables). The 48.1 seeds carry the optimizer's name
``target_annual_vol``; this factory MAPS it onto the live ``target_vol`` ctor arg,
so tb_risk_managed's vol-targeting (0.15) is ACTUALLY live vs tb_baseline (0).

DEAD-KEY HONESTY: the seed keys ``trailing_stop_enabled / trailing_trigger_pct /
trailing_distance_pct / vol_barrier_multiplier`` and the blend weights are written
by quant_optimizer._apply_params_to_engine into engine._strategy_params, but their
engine readers were REVERTED in commit 9fbd9cd6 -- nothing reads them today. This
factory does NOT cargo-cult those writes; it WARNs so the operator is not misled
that tb_risk_managed's TRAILING overrides are active (only its tp_pct + the mapped
target_vol are). Re-enabling the reverted vol-targeting/trailing readers is its own
cycle; until then tb_risk_managed differs from tb_baseline by tp_pct + target_vol
only (trailing inert) -- flagged for a seed-set follow-up.

Pure of config: it imports NO settings/BQ singletons -- the caller passes
``settings`` + ``bq`` (mirrors run_harness.make_engine) so it is $0-mock-testable.

DEFERRED (NOT built here -- later cycles, mirror 48.1/48.2):
- The LIVE multi-run bake-off (~32 real backtests, tens of minutes) -- opt-in.
- The weekly rotation CRON.
- The DEPLOYMENT bridge: params -> settings.paper_* + a promoted_strategies MERGE.
  This runner records the verdict at allocation_pct=0 ONLY; flipping a row alone
  changes the heartbeat, not live orders (strategy_candidate_producer.py:34-39).
- Re-enabling the reverted trailing/vol-target engine readers; effective-N (ONC)
  DSR clustering; CPCV multi-path PBO.

ASCII-only, fail-open.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from backend.autoresearch.strategy_backtest_adapter import make_engine_backtest_fn
from backend.autoresearch.strategy_candidate_producer import run_strategy_bakeoff
from backend.backtest.backtest_engine import STRATEGY_REGISTRY, BacktestEngine

logger = logging.getLogger(__name__)

# Keys written by the optimizer setter but with NO engine reader (reverted in
# 9fbd9cd6). NOT threaded; only WARNed on so a seed isn't believed to be active.
_DEAD_KEYS = (
    "trailing_stop_enabled",
    "trailing_trigger_pct",
    "trailing_distance_pct",
    "vol_barrier_multiplier",
    "tb_weight",
    "qm_weight",
    "mr_weight",
    "fm_weight",
)

_REPO = Path(__file__).resolve().parents[2]
_ROTATION_LOG_PATH = _REPO / "backend" / "backtest" / "experiments" / "rotation_log.jsonl"
_OPTIMIZER_BEST_PATH = _REPO / "backend" / "backtest" / "experiments" / "optimizer_best.json"


def _quiet_progress(*_args, **_kwargs) -> None:
    """No-op progress callback -- a bake-off runs K*seeds backtests; no spam."""
    return None


def make_rotation_engine(
    params: dict,
    settings: Any,
    bq: Any,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> BacktestEngine:
    """Full-ctor-kwarg BacktestEngine for one rotation strategy.

    Validates ``strategy`` against STRATEGY_REGISTRY FIRST (raise on unknown -- no
    silent triple_barrier fallback), threads the full ctor kwarg set (incl. the 8
    make_engine drops), maps ``target_annual_vol`` -> the live ``target_vol`` arg,
    and WARNs (does not cargo-cult) currently-inert risk keys.
    """
    p = params or {}
    strategy = p.get("strategy")
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"unknown strategy {strategy!r}; must be one of {sorted(STRATEGY_REGISTRY)}"
        )

    # target_annual_vol (seed/optimizer name) -> target_vol (the LIVE ctor arg).
    _tv = p.get("target_vol")
    if _tv is None:
        _tv = p.get("target_annual_vol")
    if _tv is None:
        _tv = 0.15

    inert = [k for k in _DEAD_KEYS if p.get(k)]
    if inert:
        logger.warning(
            "[rotation] strategy '%s' carries currently-inert risk keys %s "
            "(engine readers reverted in 9fbd9cd6); they will NOT affect this backtest",
            strategy, inert,
        )

    return BacktestEngine(
        bq_client=getattr(bq, "client", bq),
        project=settings.gcp_project_id,
        dataset=settings.bq_dataset_reports,
        market=p.get("market", "US"),
        start_date=start_date or p.get("start_date", "2018-01-01"),
        end_date=end_date or p.get("end_date", "2025-12-31"),
        train_window_months=p.get("train_window_months", 12),
        test_window_months=p.get("test_window_months", 3),
        embargo_days=p.get("embargo_days", 5),
        holding_days=p.get("holding_days", 90),
        tp_pct=p.get("tp_pct", 10.0),
        sl_pct=p.get("sl_pct", 10.0),
        mr_holding_days=p.get("mr_holding_days", 15),
        frac_diff_d=p.get("frac_diff_d", 0.4),
        strategy=strategy,
        starting_capital=p.get("starting_capital", 100_000.0),
        max_positions=p.get("max_positions", 20),
        transaction_cost_pct=p.get("transaction_cost_pct", 0.1),
        target_vol=_tv,
        top_n_candidates=p.get("top_n_candidates", 50),
        commission_model=p.get("commission_model", "flat_pct"),
        commission_per_share=p.get("commission_per_share", 0.005),
        n_estimators=p.get("n_estimators", 200),
        max_depth=p.get("max_depth", 4),
        min_samples_leaf=p.get("min_samples_leaf", 20),
        learning_rate=p.get("learning_rate", 0.1),
        progress_callback=progress_callback or _quiet_progress,
    )


def _incumbent_dsr_from_optimizer_best() -> Optional[float]:
    """The live strategy's recorded DSR (optimizer_best.json snapshot). Fail-open."""
    try:
        data = json.loads(_OPTIMIZER_BEST_PATH.read_text(encoding="utf-8"))
        dsr = data.get("dsr")
        return float(dsr) if dsr is not None else None
    except Exception:
        return None


def _resolve_incumbent(bq: Any, *, incumbent: Optional[dict] = None) -> Optional[dict]:
    """Build the incumbent candidate the selector compares the best seed against.

    v1 (cheap): strategy NAME from load_promoted_params (BQ promoted -> else
    optimizer_best.json), DSR from the optimizer_best snapshot. NOTE: the incumbent
    is identified by its strategy NAME (e.g. 'triple_barrier'), which may not equal
    a seed ID (e.g. 'tb_baseline'); so the selector treats the best seed as a
    challenger to the incumbent's recorded DSR (it won't return 'incumbent_is_top').
    Refining incumbent->seed-id mapping is a follow-up. Returns None -> first_selection.
    """
    if incumbent is not None:
        return incumbent
    try:
        from backend.services.autonomous_loop import load_promoted_params
        params = load_promoted_params(bq)
    except Exception as exc:
        logger.warning("[rotation] incumbent unresolved (%s); first_selection", exc)
        return None
    if not params:
        return None
    strat = params.get("strategy")
    return {
        "strategy_id": strat,
        "strategy": strat,
        "params": params,
        "dsr": _incumbent_dsr_from_optimizer_best(),
    }


def _persist_verdict(
    verdict: dict,
    *,
    path: Optional[Any] = None,
    bq_fn: Optional[Callable[[dict], None]] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Append one JSONL audit row at allocation_pct=0 (AUDIT ONLY -- NOT deployed).

    Fail-open: never raises. Optional ``bq_fn(row)`` for a BQ audit row (also
    fail-open). Precedent: monthly_champion_challenger._emit_deployment_log_row.
    """
    row = {
        "selected_id": verdict.get("selected_id"),
        "incumbent_id": verdict.get("incumbent_id"),
        "switched": verdict.get("switched"),
        "reason": verdict.get("reason"),
        "delta_dsr": verdict.get("delta_dsr"),
        "ranked": verdict.get("ranked"),
        "num_trials": verdict.get("num_trials"),
        "allocation_pct": 0.0,        # zero == recorded, NOT deployed
        "status": "bakeoff_verdict",
    }
    if extra:
        row.update(extra)
    p = Path(path) if path is not None else _ROTATION_LOG_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
    except Exception as exc:
        logger.error("[rotation] failed to persist verdict row (%s)", exc)
    if bq_fn is not None:
        try:
            bq_fn(row)
        except Exception as exc:
            logger.warning("[rotation] BQ verdict-log write failed (fail-open): %s", exc)
    return row


def run_rotation_bakeoff(
    settings: Any,
    bq: Any,
    *,
    seeds: Optional[list[dict]] = None,
    incumbent: Optional[dict] = None,
    num_param_variants: int = 8,
    num_trials: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    persist: bool = True,
    engine_factory: Optional[Callable[[dict], Any]] = None,
    adapter_fn: Optional[Callable[[dict], dict]] = None,
    log_path: Optional[Any] = None,
    bq_fn: Optional[Callable[[dict], None]] = None,
    clear_cache_fn: Optional[Callable[[], None]] = None,
) -> dict:
    """Run the live rotation bake-off and record (NOT deploy) the verdict.

    registry.load_seed_strategies -> per-strategy real-engine scoring (48.2
    adapter) -> producer -> select_best_strategy. Returns the selector verdict
    dict. Test seams: inject ``engine_factory`` (full wiring) OR ``adapter_fn``
    (narrow). Persists a JSONL audit row at allocation_pct=0 (no deploy).
    """
    if adapter_fn is None:
        if engine_factory is None:
            def engine_factory(variant: dict) -> Any:  # default: full-kwarg real engine
                return make_rotation_engine(
                    variant, settings, bq, start_date=start_date, end_date=end_date
                )
        adapter_fn = make_engine_backtest_fn(
            engine_factory,
            num_param_variants=num_param_variants,
            clear_cache_fn=clear_cache_fn,
        )

    inc = _resolve_incumbent(bq, incumbent=incumbent)
    verdict = run_strategy_bakeoff(adapter_fn, incumbent=inc, seeds=seeds, num_trials=num_trials)

    if persist:
        _persist_verdict(
            verdict,
            path=log_path,
            bq_fn=bq_fn,
            extra={
                "num_param_variants": num_param_variants,
                "window": f"{start_date or 'default'}..{end_date or 'default'}",
            },
        )
    return verdict


__all__ = [
    "make_rotation_engine",
    "run_rotation_bakeoff",
    "_resolve_incumbent",
    "_persist_verdict",
]
