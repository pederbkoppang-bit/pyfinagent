"""phase-10.7.1 Alpha Velocity metric.

Sharpe-slope-per-day metric for tracking how fast a strategy's risk-
adjusted performance is improving (or decaying) over a rolling window.

Formula (Candidate B, per research brief):

    alpha_velocity_score = (sharpe_end - sharpe_start) / window_days

Guarded by `MIN_OBSERVATIONS = 20` -- below the floor, the score is set
to None to avoid spurious slopes from thin samples. window_days = 0
raises ValueError on score access (caller must pick a non-zero window).

Persisted to `pyfinagent_pms.alpha_velocity_samples` (BQ table created
by `scripts/migrations/create_alpha_velocity_table.py`). Partitioned by
DATE(window_start), clustered on (strategy_id, macro_regime).

Distinct from the DEPRECATED `backend/agents/meta_coordinator.py` Phase-4
stub. Build all phase-10.7 work in `backend/meta_evolution/`.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIN_OBSERVATIONS = 20

PROJECT = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
DATASET = "pyfinagent_pms"
TABLE = "alpha_velocity_samples"
TABLE_FQN = f"{PROJECT}.{DATASET}.{TABLE}"


@dataclass
class AlphaVelocitySample:
    """One window's alpha-velocity observation.

    `alpha_velocity_score` is computed lazily so we can apply the
    n_obs guard + raise ValueError on zero-window without short-
    circuiting the rest of the row.
    """

    strategy_id: str
    window_start: datetime
    window_end: datetime
    n_obs: int
    sharpe_start: float
    sharpe_end: float
    macro_regime: str = "NEUTRAL"
    components: dict[str, Any] = field(default_factory=dict)
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def window_days(self) -> int:
        return (self.window_end - self.window_start).days

    @property
    def alpha_velocity_score(self) -> Optional[float]:
        """Return Sharpe-slope-per-day or None if observations are insufficient.

        Raises ValueError if window_days is zero or negative -- caller
        must pick a non-degenerate window.
        """
        wd = self.window_days
        if wd <= 0:
            raise ValueError(
                f"window_days must be > 0; got {wd} "
                f"(start={self.window_start.isoformat()}, "
                f"end={self.window_end.isoformat()})"
            )
        if self.n_obs < MIN_OBSERVATIONS:
            return None
        return (self.sharpe_end - self.sharpe_start) / wd

    def to_bq_row(self) -> dict[str, Any]:
        """BQ-canonical row shape for `pyfinagent_pms.alpha_velocity_samples`."""
        wd = self.window_days
        # n_obs guard short-circuits to None without touching window_days
        # error path (so we can still emit a row for thin-sample cases
        # for audit, just with score=None).
        if self.n_obs < MIN_OBSERVATIONS:
            score: Optional[float] = None
        elif wd <= 0:
            # Defensive: caller already had a chance to raise via the
            # property; if they're persisting anyway we record None +
            # leave window_days as-is so the row is still queryable.
            score = None
        else:
            score = (self.sharpe_end - self.sharpe_start) / wd

        return {
            "strategy_id": self.strategy_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "n_obs": int(self.n_obs),
            "sharpe_start": float(self.sharpe_start),
            "sharpe_end": float(self.sharpe_end),
            "alpha_velocity_score": score,
            "window_days": int(wd),
            "macro_regime": self.macro_regime,
            "components_json": json.dumps(self.components, sort_keys=True),
            "computed_at": self.computed_at.isoformat(),
        }


def compute_alpha_velocity(
    *,
    strategy_id: str,
    window_start: datetime,
    window_end: datetime,
    n_obs: int,
    sharpe_start: float,
    sharpe_end: float,
    macro_regime: str = "NEUTRAL",
    components: Optional[dict[str, Any]] = None,
    computed_at: Optional[datetime] = None,
) -> AlphaVelocitySample:
    """Construct an AlphaVelocitySample. Pure function; no I/O."""
    return AlphaVelocitySample(
        strategy_id=strategy_id,
        window_start=window_start,
        window_end=window_end,
        n_obs=n_obs,
        sharpe_start=sharpe_start,
        sharpe_end=sharpe_end,
        macro_regime=macro_regime,
        components=components or {},
        computed_at=computed_at or datetime.now(timezone.utc),
    )


def persist_sample(bq_client: Any, sample: AlphaVelocitySample) -> None:
    """Insert one sample into the `alpha_velocity_samples` table.

    `bq_client` must expose `insert_rows_json(table_fqn, rows)`. Real
    callers pass `google.cloud.bigquery.Client`; tests pass a FakeBQ
    stub. Errors are logged + swallowed -- this is observability, not
    trade-decision-critical, and we do NOT want a transient BQ outage
    to block the daily cycle.
    """
    row = sample.to_bq_row()
    try:
        errors = bq_client.insert_rows_json(TABLE_FQN, [row])
        if errors:
            logger.warning(
                "alpha_velocity persist returned errors: %s (row: %s)",
                errors,
                row,
            )
    except Exception as e:
        logger.warning(
            "alpha_velocity persist failed (fail-open): %s",
            e,
        )
