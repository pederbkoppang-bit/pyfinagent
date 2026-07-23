"""phase-75.8 (gap3-02): governance-vs-runtime limits divergence checker.

OBSERVABILITY ONLY. The six `governance/limits.yaml` RiskLimits have no
runtime enforcement consumers, while the live kill-switch reads
`settings.paper_daily_loss_limit_pct` / `paper_trailing_dd_limit_pct`.
This module compares the two sources and reports divergent pairs; it is
wired into the `main.py` lifespan as a startup WARNING log and NOTHING
else -- no gating, no enforcement, no mutation. Which value binds is an
operator decision (token GOV-LIMITS-DECIDE, see
handoff/current/governance_limits_divergence_75.md).

Units: settings stores PERCENTS (4.0 == 4%); limits.yaml stores
FRACTIONS (0.02 == 2%). Governed values are normalized (x100) before
comparison, with `math.isclose` so float representation noise can never
manufacture a divergence.

Reads limits via the sanctioned lru-cached `limits_schema.load()` --
never re-reads the YAML (immutable-core contract, phase-4.9.0).
"""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# (pair name, settings attr [percent units], limits.yaml field [fraction])
_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("daily_loss_kill_switch",
     "paper_daily_loss_limit_pct", "max_daily_loss_pct"),
    ("trailing_dd_kill_switch",
     "paper_trailing_dd_limit_pct", "max_trailing_dd_pct"),
)


def compute_divergence(settings_obj: Any | None = None) -> list[dict[str, Any]]:
    """Compare live kill-switch settings against governance/limits.yaml.

    Returns one dict per mapped pair with both values normalized to
    percent units and a `divergent` flag. Pure read -- no I/O beyond the
    cached limits load, no mutation of either source.
    """
    from backend.config.settings import get_settings
    from backend.governance.limits_schema import load as load_limits

    s = settings_obj if settings_obj is not None else get_settings()
    limits = load_limits()
    pairs: list[dict[str, Any]] = []
    for name, settings_attr, governed_attr in _PAIRS:
        settings_pct = float(getattr(s, settings_attr))
        governed_pct = float(getattr(limits, governed_attr)) * 100.0
        pairs.append({
            "name": name,
            "settings_attr": settings_attr,
            "governed_attr": governed_attr,
            "settings_value_pct": settings_pct,
            "governed_value_pct": governed_pct,
            "divergent": not math.isclose(
                settings_pct, governed_pct, rel_tol=1e-9
            ),
        })
    return pairs


def log_divergence_warnings() -> list[dict[str, Any]]:
    """Log-and-return wrapper for lifespan wiring. NEVER raises.

    One ASCII WARNING per divergent pair; a single INFO when all mapped
    pairs match. Any internal failure logs a WARNING and returns [] --
    startup must never be gated on this check (fail-open, mirroring the
    limits_loader block in main.py).
    """
    try:
        pairs = compute_divergence()
    except Exception:
        logger.warning(
            "governance divergence check failed (fail-open; observability "
            "only)", exc_info=True,
        )
        return []
    divergent = [p for p in pairs if p["divergent"]]
    for p in divergent:
        logger.warning(
            "governance divergence: %s -- settings.%s=%.4g%% vs limits.yaml "
            "%s=%.4g%% (observability only; which value binds is operator "
            "token GOV-LIMITS-DECIDE)",
            p["name"], p["settings_attr"], p["settings_value_pct"],
            p["governed_attr"], p["governed_value_pct"],
        )
    if not divergent:
        logger.info(
            "governance divergence: all %d mapped limit pairs match",
            len(pairs),
        )
    return pairs


__all__ = ["compute_divergence", "log_divergence_warnings"]
