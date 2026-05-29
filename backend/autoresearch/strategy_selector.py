"""phase-47.6: dynamic strategy SELECTION with anti-churn hysteresis.

North star: shift capital to whichever strategy is making the most money --
but ONLY when the evidence is strong (DSR >= 0.95 AND PBO <= 0.20; Bailey &
Lopez de Prado 2014 -- the Deflated Sharpe Ratio is the correct best-of-N
selection tool, deflating for multiple testing across the strategy set) AND the
improvement over the incumbent is material. The min-improvement gate is the
anti-churn / hysteresis term: without it the loop whipsaws between near-tied
strategies, paying turnover cost for noise (jump-model 2024: a switch penalty
cut turnover 141% -> 44% while improving net-of-cost Sharpe; incumbent-bias
hysteresis is canonical for strategy rotation).

This is the SELECTION layer over the EXISTING promotion infra: it REUSES
`backend/autoresearch/gate.py::PromotionGate` for the DSR/PBO gate and mirrors
`friday_promotion`'s DSR-desc / PBO-asc ranking. The chosen strategy is meant to
flow to the live loop through the existing `promoted_strategies` BQ row that
`autonomous_loop.load_promoted_params` already consumes -- no new read path.

Pure functions, fail-open, ASCII-only.

min_improvement default: gate-passers have DSR in [0.95, 1.0], so the max
possible Delta-DSR between two passers is 0.05. A 0.01 (one DSR-point) default
is the band-appropriate hysteresis -- large enough to ignore noise, small enough
to allow a genuine improvement to win. Parameterized so the operator can tune.

DEFERRED (documented, NOT in this step): live per-strategy DSR population via 5
quant-only ($0-LLM) walk-forward backtests; the weekly cron that drives the
selection; real-capital activation (stays paper-only). v1 carries num_trials for
context (the upstream DSR already deflates); effective-N clustering of the
correlated strategies is deferred -- v1's plain N over-deflates, the SAFE
direction.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from backend.autoresearch.gate import PromotionGate

logger = logging.getLogger(__name__)

# One DSR-point of improvement over the incumbent to justify a switch. See the
# module docstring for why 0.01 (not 0.05) given the [0.95, 1.0] passer band.
_DEFAULT_MIN_IMPROVEMENT = 0.01


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _strategy_id(c: Optional[dict]) -> Optional[str]:
    if not c:
        return None
    return str(c.get("strategy_id") or c.get("strategy") or c.get("trial_id") or "") or None


def select_best_strategy(
    per_strategy: list[dict[str, Any]],
    incumbent: Optional[dict[str, Any]] = None,
    *,
    gate: Optional[PromotionGate] = None,
    min_improvement: float = _DEFAULT_MIN_IMPROVEMENT,
    num_trials: int = 5,
) -> dict[str, Any]:
    """Pick the strategy to deploy from per-strategy scored candidates.

    Each candidate dict carries at least a strategy id (`strategy_id` |
    `strategy` | `trial_id`), `dsr`, and `pbo`; optionally `params`. `incumbent`
    is the currently-deployed strategy dict (or None for first selection).

    Decision order:
      1. Gate-filter candidates (DSR >= 0.95 AND PBO <= 0.20 via PromotionGate).
      2. Rank passers DSR-desc, then PBO-asc.
      3. No passer -> RETAIN incumbent (reason 'no_candidate_passed_gate'); never
         go to cash on a weak research week.
      4. best = ranked[0]. No incumbent -> SELECT best ('first_selection').
      5. best IS the incumbent -> RETAIN ('incumbent_is_top').
      6. best != incumbent: SWITCH only if best.dsr - incumbent.dsr >=
         min_improvement ('dsr_improvement'); else RETAIN incumbent
         ('below_min_improvement') -- anti-churn hysteresis.

    Returns a verdict dict:
      {selected, selected_id, switched, reason, ranked, incumbent_id,
       num_trials, delta_dsr}
    Pure: never mutates inputs or external state.
    """
    g = gate or PromotionGate()
    inc_id = _strategy_id(incumbent)
    inc_dsr = _safe_float(incumbent.get("dsr"), float("-inf")) if incumbent else float("-inf")

    passers = [c for c in (per_strategy or []) if g.evaluate(c).get("promoted")]
    passers_sorted = sorted(
        passers,
        key=lambda c: (-_safe_float(c.get("dsr"), 0.0), _safe_float(c.get("pbo"), 1.0)),
    )
    ranked_ids = [_strategy_id(c) for c in passers_sorted]

    def _result(selected, switched, reason, delta=None):
        return {
            "selected": selected,
            "selected_id": _strategy_id(selected),
            "switched": bool(switched),
            "reason": reason,
            "ranked": ranked_ids,
            "incumbent_id": inc_id,
            "num_trials": int(num_trials),
            "delta_dsr": delta,
        }

    if not passers_sorted:
        return _result(incumbent, False, "no_candidate_passed_gate")

    best = passers_sorted[0]
    best_id = _strategy_id(best)
    best_dsr = _safe_float(best.get("dsr"), 0.0)

    if inc_id is None:
        return _result(best, True, "first_selection")

    if best_id == inc_id:
        return _result(best, False, "incumbent_is_top")

    delta = best_dsr - inc_dsr
    if delta >= min_improvement:
        return _result(best, True, "dsr_improvement", round(delta, 6))
    # Anti-churn: a strictly-better-but-not-by-enough challenger does NOT
    # displace the incumbent -- avoids whipsaw between near-tied strategies.
    return _result(incumbent, False, "below_min_improvement", round(delta, 6))


__all__ = ["select_best_strategy"]
