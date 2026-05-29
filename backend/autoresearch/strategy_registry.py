"""phase-48.1: config-driven seed strategy REGISTRY for dynamic rotation.

North star (Priority 5): shift paper capital to whichever strategy is making the
most money. The 47.6 `select_best_strategy` is the SELECTION layer; this module
is the other half of its input -- the enumerated SET of candidate strategies the
producer (`strategy_candidate_producer.build_per_strategy_candidates`) scores and
feeds to the selector.

Design (grounded in research_brief_phase_48_1_rotation_foundation.md, gate PASSED
8 sources):
- **Diversify on orthogonal AXES, not parameter tweaks.** The diversification
  benefit comes from strategy TYPE (mean-reversion vs trend/momentum vs the
  triple-barrier ML baseline), which are structurally anti-correlated (their
  drawdowns occur at different times -> smoother equity, lower maxDD). So the
  seed set spans `strategy` categoricals, not just risk-knob variants of one
  model. (Sources: buildalpha trading-ensemble; algomatictrading correlation;
  Lopez de Prado AFML.)
- **Small + pre-registered** (E[max SR] ~ sqrt(2 log N): more trials mechanically
  inflate the apparent best). Floor is >=2 strategies; we seed 4.
- **Each seed clears the individual DSR>=0.95 / PBO<=0.20 gate on its own** -- a
  seed is never expected to be rescued by ensembling ("polishes gold, does not
  turn dirt into gold"). The gate (`gate.PromotionGate`) is reused unchanged.
- **Config-driven + operator-tunable:** `SEED_STRATEGIES` is a module constant of
  `param_overrides` ON TOP of the live `optimizer_best.params`, and
  `load_seed_strategies` accepts an injected `seeds` list so the operator/tests
  can swap the competing set without touching code.

Pure, fail-open, ASCII-only.

DEFERRED (NOT built here -- later cycles):
- The REAL backtest-engine adapter that turns each seed's params into live
  (dsr, pbo) via `BacktestEngine.run_backtest` (warm-cache loop, skip_cache_clear)
  -> `nav_history` daily returns -> `analytics.generate_report` DSR + a per-strategy
  (T x K-trial) `analytics.compute_pbo` matrix. This cycle injects a `backtest_fn`
  so the producer is pure + $0-testable; the adapter is a drop-in next cycle.
- The weekly rotation CRON that runs the bake-off on a schedule.
- The DEPLOYMENT switch + the params->settings.paper_* bridge: per the deploy
  audit, `best_params` is NOT threaded into `decide_trades`/`paper_trader`; live
  risk/sizing/turnover is driven by `settings.paper_*`. Flipping a
  `promoted_strategies` row alone changes only the heartbeat, not live orders.
  A real rotation MUST bridge params->settings; that is its own cycle.
- Effective-N clustering: the selector's plain `num_trials=N` over-deflates DSR
  for correlated seeds -- the SAFE (too-conservative) direction; deferred.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# backend/autoresearch/strategy_registry.py -> parents[2] == repo root.
_OPTIMIZER_BEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "backend"
    / "backtest"
    / "experiments"
    / "optimizer_best.json"
)

# Seed set: param_overrides applied ON TOP of optimizer_best.params. Distinct
# along the deploy-audit's real axes -- (1) strategy categorical [biggest lever],
# (2) holding/turnover regime, (3) risk/exit profile. Operator-tunable: pass a
# different list to load_seed_strategies(seeds=...).
SEED_STRATEGIES: list[dict[str, Any]] = [
    {
        "id": "tb_baseline",
        "rationale": (
            "Incumbent reference rail -- verbatim optimizer_best.params "
            "(strategy=triple_barrier, the current live config). MUST be in the "
            "bake-off so the selector's anti-churn hysteresis has the live "
            "deployment to retain against. Axis: long-horizon fixed-barrier ML."
        ),
        "param_overrides": {},
    },
    {
        "id": "mr_short_horizon",
        "rationale": (
            "Structurally anti-correlated diversifier: mean-reversion needs "
            "choppy/range-bound regimes while the baseline trend/ML needs "
            "sustained moves, so drawdowns occur at different times (lower "
            "portfolio maxDD). Differentiates on strategy TYPE (mean_reversion) "
            "AND short holding/turnover regime (mr_holding_days in the 5-30 MR "
            "band; shared holding_days=90 is too long for MR)."
        ),
        "param_overrides": {
            "strategy": "mean_reversion",
            "mr_holding_days": 8,
            "holding_days": 30,
        },
    },
    {
        "id": "qm_trend_tilt",
        "rationale": (
            "Quality-momentum / trend sleeve -- third distinct strategy TYPE so "
            "the seed set spans the mean-reversion-vs-trend anti-correlation axis "
            "the literature identifies as the real source of diversification. "
            "Longer holding_days=120 keeps it in the low-turnover trend regime so "
            "it does not collapse onto the baseline's turnover profile."
        ),
        "param_overrides": {
            "strategy": "quality_momentum",
            "holding_days": 120,
        },
    },
    {
        "id": "tb_risk_managed",
        "rationale": (
            "Same strategy TYPE as the baseline (triple_barrier) but "
            "differentiated purely on the risk/exit axis: volatility-targeting on "
            "(target_annual_vol=0.15, off in baseline) + trailing stop on + "
            "tighter take-profit. Deliberately a CORRELATED variant -- it exists "
            "so the selector's anti-churn term has a near-tied challenger to "
            "evaluate, and it must clear the DSR>=0.95/PBO<=0.20 gate on its own "
            "merits like every other seed (no ensembling rescue)."
        ),
        "param_overrides": {
            "strategy": "triple_barrier",
            "target_annual_vol": 0.15,
            "trailing_stop_enabled": True,
            "trailing_trigger_pct": 5,
            "trailing_distance_pct": 3,
            "tp_pct": 6,
        },
    },
]


def load_base_params() -> dict[str, Any]:
    """Read optimizer_best.params (the current live config). Fail-open to {}.

    Mirrors autonomous_loop.load_best_params: returns the inner ``params`` dict
    (not the wrapper). Never raises -- a missing/corrupt file yields {} so the
    registry can still enumerate ids.
    """
    try:
        data = json.loads(_OPTIMIZER_BEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # missing file / bad json -> fail-open
        logger.warning(
            "[strategy_registry] optimizer_best.json unreadable (%s); "
            "seeds will carry only their param_overrides", exc,
        )
        return {}
    if isinstance(data, dict):
        params = data.get("params", data)
        if isinstance(params, dict):
            return dict(params)
    return {}


def load_seed_strategies(
    seeds: Optional[list[dict[str, Any]]] = None,
    base_params: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Enumerate the rotation candidate set.

    For each seed, the deployed params are ``{**base_params, **param_overrides}``
    -- the seed's overrides layered on the current live config. Returns a list of
    ``{"id": str, "rationale": str, "params": dict}``.

    Args:
        seeds: override the module ``SEED_STRATEGIES`` (operator-tunable). Each
            entry needs at least ``id``; ``param_overrides`` defaults to {} and
            ``rationale`` to "".
        base_params: override the optimizer_best.params base (tests inject {}).

    Pure: copies inputs, never mutates ``SEED_STRATEGIES`` or the base.
    """
    base = dict(base_params) if base_params is not None else load_base_params()
    chosen = seeds if seeds is not None else SEED_STRATEGIES
    out: list[dict[str, Any]] = []
    for seed in chosen:
        sid = str(seed.get("id") or "").strip()
        if not sid:
            logger.warning("[strategy_registry] skipping seed with no id: %r", seed)
            continue
        overrides = seed.get("param_overrides") or {}
        out.append(
            {
                "id": sid,
                "rationale": str(seed.get("rationale") or ""),
                "params": {**base, **overrides},
            }
        )
    return out


__all__ = ["SEED_STRATEGIES", "load_base_params", "load_seed_strategies"]
