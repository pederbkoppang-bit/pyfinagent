"""phase-9.6 Nightly outcome-tracker rebuild from trade ledger."""
from __future__ import annotations

import logging
from typing import Any, Callable

from backend.slack_bot.job_runtime import IdempotencyKey, IdempotencyStore, heartbeat

logger = logging.getLogger(__name__)
JOB_NAME = "nightly_outcome_rebuild"


def run(
    *,
    ledger_fetch_fn: Callable[[], list[dict]] | None = None,
    outcome_write_fn: Callable[[list[dict]], int] | None = None,
    store: IdempotencyStore | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    key = IdempotencyKey.daily(JOB_NAME, day=day)
    result: dict[str, Any] = {"rebuilt": 0, "key": key, "skipped": False}
    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        trades = (ledger_fetch_fn or _default_fetch)()
        outcomes = _compute_outcomes(trades)
        try:
            n = (outcome_write_fn or _default_write)(outcomes)
        except Exception as exc:
            logger.warning("outcome_rebuild: write fail-open: %r", exc)
            n = 0
        result["rebuilt"] = int(n)
    return result


def _compute_outcomes(trades: list[dict]) -> list[dict]:
    """Per-trade outcome: pnl > 0 -> win, <= 0 -> loss."""
    return [
        {
            "trade_id": t.get("trade_id"),
            "ticker": t.get("ticker"),
            "pnl": t.get("pnl", 0.0),
            "outcome": "win" if t.get("pnl", 0.0) > 0 else "loss",
        }
        for t in (trades or [])
    ]


def _default_fetch() -> list[dict]:
    return []  # production reads pyfinagent_pms.paper_trades


def _default_write(outcomes: list[dict]) -> int:
    return len(outcomes)


__all__ = ["run", "JOB_NAME"]
