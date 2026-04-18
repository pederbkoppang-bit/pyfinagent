"""phase-4.8 step 4.8.10 T+1 funding + real-time margin guards.

Two pre-submit guards on BUY orders:

1. `t1_funding_guard(settled_cash, pending_proceeds, buy_notional)`
   -- blocks BUY when notional exceeds SETTLED cash. Same-day sell
   proceeds are NOT available to fund a same-day BUY under SEC
   Rule 15c6-1 T+1 (effective 2024-05-28).

2. `realtime_margin_guard(gross_long, available_margin,
   buy_notional, deficit_threshold_pct=0.0)` -- blocks BUY when
   `gross_long + buy_notional > available_margin * (1 -
   deficit_threshold_pct)`. Mirrors FINRA SR-2025-017 intraday
   margin-deficit framework (approved 2026-04-17).

Both guards return `(allowed: bool, reason: str)`. `reason` is an
enum-ish string so downstream order routers can log with a
stable code.
"""
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


GuardReason = Literal[
    "OK",
    "UNSETTLED_CASH_INSUFFICIENT",
    "MARGIN_DEFICIT",
    "INVALID_INPUT",
]


def t1_funding_guard(
    *,
    settled_cash: float,
    pending_proceeds: float,
    buy_notional: float,
) -> tuple[bool, GuardReason]:
    """Block a BUY that would be funded from unsettled same-day sells.

    Parameters
    ----------
    settled_cash : cash that has PASSED its T+1 settlement date.
    pending_proceeds : cash from sells executed today that settles
        tomorrow. NOT usable for today's BUYs.
    buy_notional : absolute USD notional of the BUY being checked.
    """
    for name, v in (("settled_cash", settled_cash),
                     ("pending_proceeds", pending_proceeds),
                     ("buy_notional", buy_notional)):
        if v is None or v != v or v < 0:   # NaN or negative
            return False, "INVALID_INPUT"
    if buy_notional > settled_cash + 1e-6:
        return False, "UNSETTLED_CASH_INSUFFICIENT"
    return True, "OK"


def realtime_margin_guard(
    *,
    gross_long: float,
    available_margin: float,
    buy_notional: float,
    deficit_threshold_pct: float = 0.0,
) -> tuple[bool, GuardReason]:
    """Block a BUY that would push projected gross long past available
    margin (FINRA 4210 intraday deficit framework).

    `deficit_threshold_pct` (0..1) tightens the effective margin
    floor: e.g., 0.05 leaves a 5% buffer before triggering.
    """
    for name, v in (("gross_long", gross_long),
                     ("available_margin", available_margin),
                     ("buy_notional", buy_notional)):
        if v is None or v != v or v < 0:
            return False, "INVALID_INPUT"
    if not (0.0 <= deficit_threshold_pct < 1.0):
        return False, "INVALID_INPUT"
    effective_cap = available_margin * (1.0 - deficit_threshold_pct)
    projected = gross_long + buy_notional
    if projected > effective_cap + 1e-6:
        return False, "MARGIN_DEFICIT"
    return True, "OK"


__all__ = ["GuardReason", "t1_funding_guard", "realtime_margin_guard"]
