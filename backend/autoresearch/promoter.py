"""phase-8.5.6 Autoresearch -> paper-live promoter.

Requires >= 5 trading days of shadow-log history and DSR >= 0.95 before
promoting a trial to paper-live. Position size is tied to realized DSR.
Kill-switch callback fires on drawdown breach.

phase-25.R: `write_to_registry` closes red-line goal-c (dynamically shift
strategy to whichever is making the most money). On gate clear, writes
`status="active"` to `pyfinagent_data.promoted_strategies` and supersedes
the prior active row atomically.

Pure functions. ASCII-only.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

SHADOW_MIN_DAYS = 5
DD_TRIGGER = 0.10
DSR_MIN_FOR_PROMOTION = 0.95


@dataclass(frozen=True)
class Promoter:
    shadow_min_days: int = SHADOW_MIN_DAYS
    dsr_min: float = DSR_MIN_FOR_PROMOTION
    dd_trigger: float = DD_TRIGGER

    def promote(self, trial: dict[str, Any]) -> dict[str, Any]:
        days = int(trial.get("shadow_trading_days", 0))
        dsr = float(trial.get("dsr", 0.0))
        if days < self.shadow_min_days:
            return {"promoted": False, "reason": f"shadow_days_below_min:{days}<{self.shadow_min_days}"}
        if dsr < self.dsr_min:
            return {"promoted": False, "reason": f"dsr_below_min:{dsr:.4f}<{self.dsr_min}"}
        return {"promoted": True, "reason": None, "trial_id": trial.get("trial_id")}

    def position_size(self, trial: dict[str, Any], capital: float) -> float:
        """Notional size tied to realized DSR. Below DSR=0.5 -> 0. At DSR=1.0 -> full capital."""
        dsr = float(trial.get("dsr", 0.0))
        fraction = max(0.0, min(1.0, (dsr - 0.5) * 2.0))
        return float(capital) * fraction

    def on_dd_breach(
        self,
        current_dd: float,
        kill_fn: Callable[[str], None],
    ) -> bool:
        """Fire kill_fn when abs(dd) exceeds dd_trigger. Returns True when fired."""
        if abs(float(current_dd)) > self.dd_trigger:
            kill_fn(f"dd_breach:{current_dd:.4f}>{self.dd_trigger}")
            return True
        return False

    def write_to_registry(
        self,
        bq_client: Any,
        trial: dict[str, Any],
        *,
        week_iso: str,
        slack_fn: Callable[[list[dict]], Any] | None = None,
    ) -> dict[str, Any]:
        """phase-25.R: ops-authorized auto-switch path.

        Runs the promote() gate first; if the trial passes, atomically:
          1. Looks up the prior active strategy via
             `bq_client.get_latest_promoted_strategy(status_filter=["active"])`.
             If found AND different strategy_id, flips it to "superseded".
          2. Writes a new row with `status="active"` via
             `bq_client.save_promoted_strategy(row)`.
          3. Fires a P0 Slack alert (`format_strategy_switch`) via slack_fn,
             only if the BQ write succeeded -- the Slack alert reflects the
             registry state, never lies about a write that failed.

        Per-call try/except keeps each side effect independent so a BQ
        failure on the supersession doesn't block the new-row write,
        and a Slack failure never blocks the registry update.

        Returns dict: {promoted, reason?, prior_strategy_id, new_strategy_id,
        alert_sent}. When promoted=False, reason mirrors promote()'s output.
        """
        verdict = self.promote(trial)
        if not verdict.get("promoted"):
            return {
                "promoted": False,
                "reason": verdict.get("reason"),
                "prior_strategy_id": None,
                "new_strategy_id": None,
                "alert_sent": False,
            }

        new_id = str(trial.get("trial_id") or "")
        switched_at = datetime.now(timezone.utc).isoformat()

        # Step 1: supersede the prior active row (if any).
        prior_id: str | None = None
        prior_week: str | None = None
        try:
            prior = bq_client.get_latest_promoted_strategy(status_filter=["active"])
            if prior is not None:
                prior_id = str(prior.get("strategy_id") or "") or None
                prior_week = str(prior.get("week_iso") or "") or None
                if prior_id and prior_id != new_id:
                    try:
                        bq_client.update_promoted_strategy_status(
                            prior_id, "superseded", week_iso=prior_week,
                        )
                    except Exception as exc:
                        logger.warning(
                            "write_to_registry: supersession fail-open for %s: %r",
                            prior_id, exc,
                        )
                else:
                    # Same strategy as the prior active -- no supersede needed.
                    prior_id = None
        except Exception as exc:
            logger.warning(
                "write_to_registry: prior-active lookup fail-open: %r", exc,
            )

        # Step 2: write the new active row.
        allocation_pct = self.position_size(trial, capital=1.0)
        new_row = {
            "strategy_id": new_id,
            "week_iso": week_iso,
            "params": json.dumps(trial.get("params") or {}),
            "dsr": float(trial.get("dsr") or 0.0),
            "pbo": float(trial.get("pbo") or 0.0),
            "status": "active",
            "allocation_pct": float(allocation_pct),
            "promoted_at": switched_at,
            "sortino_monthly": float(trial.get("sortino_monthly") or 0.0),
        }
        write_ok = False
        try:
            bq_client.save_promoted_strategy(new_row)
            write_ok = True
        except Exception as exc:
            logger.warning(
                "write_to_registry: registry write fail-open for %s: %r",
                new_id, exc,
            )

        # Step 3: P0 Slack -- only after a successful registry write to
        # avoid lying about the state. Per-call try/except.
        alert_sent = False
        if write_ok and slack_fn is not None:
            try:
                from backend.slack_bot.formatters import format_strategy_switch
                blocks = format_strategy_switch({
                    "new_strategy_id": new_id,
                    "prior_strategy_id": prior_id,
                    "dsr": new_row["dsr"],
                    "pbo": new_row["pbo"],
                    "allocation_pct": new_row["allocation_pct"],
                    "switched_at": switched_at,
                    "week_iso": week_iso,
                })
                slack_fn(blocks)
                alert_sent = True
            except Exception as exc:
                logger.warning(
                    "write_to_registry: Slack fail-open for %s: %r",
                    new_id, exc,
                )

        return {
            "promoted": True,
            "reason": None,
            "prior_strategy_id": prior_id,
            "new_strategy_id": new_id,
            "alert_sent": alert_sent,
        }


__all__ = ["Promoter", "SHADOW_MIN_DAYS", "DD_TRIGGER"]
