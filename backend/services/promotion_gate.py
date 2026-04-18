"""phase-4.8 step 4.8.5 Champion-challenger gradual rollout gate.

Three-stage canary -> ramp -> full promotion:

    STAGES[0] = 0.05   # canary (14 days minimum)
    STAGES[1] = 0.25   # ramp (30 days minimum)
    STAGES[2] = 1.00   # champion

A challenger advances one stage at a time after passing:
- PSR (Probabilistic Sharpe Ratio, Bailey/Lopez de Prado 2012) >=
  champion's PSR
- days_at_stage >= MIN_LIVE_DAYS[current_stage]
- no kill-switch events during the stage (challenger.kill_events == 0)
- PBO < 0.5 (preserves phase-3.7.3 overfitting veto)

Gate outputs one of:
    advance   -- move up a stage; allocation_pct -> STAGES[stage+1]
    hold      -- same stage, re-evaluate later
    regress   -- move down one stage (soft failure)
    demote    -- freeze (0%) after 3 consecutive stage failures
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


STAGES = [0.05, 0.25, 1.0]          # allocation fractions per stage
MIN_LIVE_DAYS = [14, 30]            # min days before advancing stage 0->1, 1->2
PSR_PARITY = 0.0                    # challenger must meet or exceed champion PSR
PBO_CEILING = 0.5


def evaluate_stage(
    *,
    challenger: dict[str, Any],
    champion: dict[str, Any],
    current_stage: int,
    days_at_stage: int,
    consecutive_failures: int = 0,
) -> dict[str, Any]:
    """Decide advance/hold/regress/demote and return the next allocation."""
    reasons: list[str] = []

    ch_psr = float(challenger.get("psr", 0.0))
    cp_psr = float(champion.get("psr", 0.0))
    ch_pbo = float(challenger.get("pbo", 0.0))
    ch_kill = int(challenger.get("kill_events", 0))

    psr_ok = ch_psr >= cp_psr + PSR_PARITY
    pbo_ok = ch_pbo < PBO_CEILING
    kill_ok = ch_kill == 0

    if not psr_ok:
        reasons.append(f"psr_below_champion ({ch_psr:.3f} < {cp_psr:.3f})")
    if not pbo_ok:
        reasons.append(f"pbo_over_ceiling ({ch_pbo:.3f} >= {PBO_CEILING})")
    if not kill_ok:
        reasons.append(f"kill_events_nonzero ({ch_kill})")

    # Days-at-stage only applies when considering ADVANCE; holds/regresses
    # can happen any time.
    can_advance_by_time = (
        current_stage < len(MIN_LIVE_DAYS)
        and days_at_stage >= MIN_LIVE_DAYS[current_stage]
    )
    is_top_stage = current_stage >= len(STAGES) - 1

    # Decision tree
    if consecutive_failures >= 3:
        decision = "demote"
        next_stage = -1
        next_alloc = 0.0
        reasons.append("consecutive_failures>=3")
    elif psr_ok and pbo_ok and kill_ok and can_advance_by_time and not is_top_stage:
        decision = "advance"
        next_stage = current_stage + 1
        next_alloc = STAGES[next_stage]
    elif psr_ok and pbo_ok and kill_ok and is_top_stage:
        decision = "hold"  # already at full allocation
        next_stage = current_stage
        next_alloc = STAGES[current_stage]
    elif psr_ok and pbo_ok and kill_ok:
        # Gates pass but min days not met -> hold at this stage
        decision = "hold"
        next_stage = current_stage
        next_alloc = STAGES[current_stage]
        reasons.append(
            f"days_at_stage_insufficient ({days_at_stage} < {MIN_LIVE_DAYS[current_stage]})"
        )
    else:
        # Any hard fail -> regress one stage (or demote if already at 0)
        if current_stage <= 0:
            decision = "demote"
            next_stage = -1
            next_alloc = 0.0
        else:
            decision = "regress"
            next_stage = current_stage - 1
            next_alloc = STAGES[next_stage]

    return {
        "decision": decision,
        "current_stage": current_stage,
        "next_stage": next_stage,
        "next_allocation_pct": next_alloc,
        "days_at_stage": days_at_stage,
        "min_live_days": MIN_LIVE_DAYS[current_stage] if current_stage < len(MIN_LIVE_DAYS) else None,
        "reasons": reasons,
        "checks": {
            "psr_ok": psr_ok,
            "pbo_ok": pbo_ok,
            "kill_ok": kill_ok,
            "can_advance_by_time": can_advance_by_time,
        },
    }


def update_optimizer_best(
    path: Path,
    *,
    allocation_pct: float,
    stage: int,
    challenger_run_id: str | None = None,
) -> dict[str, Any]:
    """Write allocation_pct + stage into optimizer_best.json IN-PLACE,
    preserving all existing keys. Creates the file with a minimal
    stub if it does not exist (this enables the fresh-deploy case)."""
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
    else:
        blob = {}
    blob["allocation_pct"] = float(allocation_pct)
    blob["stage"] = int(stage)
    if challenger_run_id is not None:
        blob["challenger_run_id"] = challenger_run_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return blob


__all__ = [
    "STAGES",
    "MIN_LIVE_DAYS",
    "PBO_CEILING",
    "evaluate_stage",
    "update_optimizer_best",
]
