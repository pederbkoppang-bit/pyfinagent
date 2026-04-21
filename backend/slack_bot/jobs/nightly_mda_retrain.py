"""phase-9.4 Nightly walk-forward MDA retraining with promotion gate.

Retrains the MDA ensemble nightly using a walk-forward window. New model
goes through `PromotionGate` (phase-8.5.5). Rejected models do NOT overwrite
the baseline; the existing `optimizer_best.json` remains authoritative.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from backend.autoresearch.gate import PromotionGate
from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat

logger = logging.getLogger(__name__)
JOB_NAME = "nightly_mda_retrain"


def run(
    *,
    train_fn: Callable[[], dict[str, Any]] | None = None,
    gate: PromotionGate | None = None,
    commit_fn: Callable[[dict], None] | None = None,
    store: IdempotencyStore | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Retrain; evaluate via gate; commit baseline only if promoted."""
    key = IdempotencyKey.daily(JOB_NAME, day=day)
    g = gate or PromotionGate()
    result: dict[str, Any] = {"promoted": False, "key": key, "skipped": False, "reason": None}

    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        new_model = (train_fn or _default_train)()
        verdict = g.evaluate(new_model)
        result["promoted"] = verdict["promoted"]
        result["reason"] = verdict.get("reason")
        result["trial_id"] = new_model.get("trial_id")
        if verdict["promoted"] and commit_fn is not None:
            commit_fn(new_model)
    return result


def _default_train() -> dict[str, Any]:
    """Injected in tests; production invokes backend/backtest/quant_optimizer.py."""
    return {"trial_id": "stub_nightly", "dsr": 0.80, "pbo": 0.30, "sharpe": 1.0}


__all__ = ["run", "JOB_NAME"]
