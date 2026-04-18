"""phase-4.9 step 4.9.0 Immutable risk-limit schema.

The six hardcoded, non-hot-reloadable limits per FINRA risk-
management supervision (Rule 3110) + SEC 15c3-1 haircut precedent
+ AFML bet-sizing caps + QuantConnect LEAN defaults + multi-manager
pod practice (Millennium / Citadel).

DO NOT hot-reload at runtime. `load()` is `lru_cache`-wrapped so
one parse lasts the process lifetime. Any runtime code that
attempts to re-read the YAML file or clear the cache is a violation
of the immutable-core contract (phase-4.9.1 CI lint will block).

Changes to `limits.yaml` require a GPG-signed git tag named
`limits-rotation-YYYYMMDD` (enforced by phase-4.9.1 workflow).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


LIMITS_FILE = Path(__file__).parent / "limits.yaml"


class RiskLimits(BaseModel):
    """Six immutable portfolio-level hard risk limits.

    Pydantic v2 frozen model: attempting to set a field on an
    existing instance raises `ValidationError(type='frozen_instance')`.
    `extra='forbid'` raises on any extra field supplied at
    construction time (typo protection).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_position_notional_pct: float = Field(
        gt=0.0, le=1.0,
        description="Max fraction of NAV in any single symbol "
                    "(SEC 15c3-1 haircut precedent; LEAN default).",
    )
    max_portfolio_leverage: float = Field(
        gt=0.0, le=4.0,
        description="Max gross leverage (long notional / NAV).",
    )
    max_daily_loss_pct: float = Field(
        gt=0.0, le=0.5,
        description="Daily-loss kill-switch threshold.",
    )
    max_trailing_dd_pct: float = Field(
        gt=0.0, le=0.5,
        description="Trailing-drawdown kill-switch threshold "
                    "(peak-to-trough NAV since inception).",
    )
    max_gross_exposure_pct: float = Field(
        gt=0.0, le=4.0,
        description="Max gross exposure as fraction of NAV "
                    "(long-only => 1.0; >1.0 implies leverage).",
    )
    max_sector_weight_pct: float = Field(
        gt=0.0, le=1.0,
        description="Max single-sector concentration.",
    )


@lru_cache(maxsize=1)
def load() -> RiskLimits:
    """Load + validate the immutable limits file. Cached forever.

    The first call parses YAML and validates against the schema;
    every subsequent call returns the SAME object (by id). This is
    intentional -- runtime code must never see a limit change
    mid-process without a full restart + GPG-signed redeploy.
    """
    with open(LIMITS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"limits.yaml root must be a mapping; got {type(data).__name__}"
        )
    return RiskLimits.model_validate(data)


def get_limits_digest() -> str:
    """SHA-256 hex digest of the raw yaml file bytes.

    Used by phase-4.9.2 startup loader to log the committed-limits
    fingerprint; any discrepancy between the expected digest and
    the actual at-boot digest is a deployment-safety signal.
    """
    data = LIMITS_FILE.read_bytes()
    return hashlib.sha256(data).hexdigest()


__all__ = ["LIMITS_FILE", "RiskLimits", "get_limits_digest", "load"]
