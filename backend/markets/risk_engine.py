"""phase-5.4 Multi-asset risk engine for position sizing.

Stateless `RiskEngine` class that computes vol-targeted position size
for equity / option / FX / future. Replaces no existing path -- this
is additive net-new code consumed by phase-5.6+ multi-asset execution.

Formulas (research brief evidence):

    base_notional = equity * (target_vol / asset_vol)
                  clamped at  max_leverage * equity

    equity:  return base_notional
    option:  return base_notional * abs(delta)        # delta-adjusted exposure
    fx:      return max(1, round(base_notional / 1000)) * 1000   # micro-lot floor
    future:  return base_notional                     # contract-multiplier table TBD in 5.8

Defaults: target_vol=0.15, max_leverage=3.0 (matches existing
BacktestTrader at backend/backtest/backtest_trader.py:54+80).

Crypto is explicitly rejected (owner directive 2026-04-19): raises
ValueError on `asset_class="crypto"`.

Pure module: no I/O, no env reads, no module-level side effects. Safe
to import in any environment.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_TARGET_VOL = 0.15
MAX_LEVERAGE = 3.0
FX_MICRO_LOT = 1000  # 1 micro lot = 1000 units (OANDA universal convention)
SUPPORTED_ASSET_CLASSES: tuple[str, ...] = ("equity", "option", "fx", "future")
MIN_ASSET_VOL = 1e-6  # floor to avoid div-by-zero


class RiskEngine:
    """Stateless multi-asset position-sizing engine.

    `target_vol` and `max_leverage` are construction-time settings; the
    same instance can be reused across many `compute_position_size`
    calls. No background state, no I/O, thread-safe by design.
    """

    def __init__(
        self,
        *,
        target_vol: float = DEFAULT_TARGET_VOL,
        max_leverage: float = MAX_LEVERAGE,
    ) -> None:
        if target_vol <= 0:
            raise ValueError(f"target_vol must be > 0, got {target_vol}")
        if max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {max_leverage}")
        self.target_vol = float(target_vol)
        self.max_leverage = float(max_leverage)

    def _base_notional(self, equity: float, asset_vol: float) -> float:
        """Vol-targeted notional, clamped at max_leverage * equity."""
        if equity <= 0:
            raise ValueError(f"equity must be > 0, got {equity}")
        v = max(float(asset_vol), MIN_ASSET_VOL)
        raw = equity * (self.target_vol / v)
        return min(raw, self.max_leverage * equity)

    def compute_position_size(
        self,
        symbol: str,
        asset_class: str,
        equity: float,
        asset_vol: float,
        *,
        delta: Optional[float] = None,
        **kwargs: Any,
    ) -> float:
        """Return positive notional position size for the given asset.

        Args:
            symbol: ticker / contract identifier (informational; not used
                in the formula itself but logged on errors)
            asset_class: one of equity / option / fx / future
            equity: account equity in USD
            asset_vol: realised or forecast vol (annualised, decimal,
                e.g. 0.20 for 20%)
            delta: option delta (required for asset_class='option';
                ignored otherwise). Sign is dropped (abs).
            **kwargs: future contract-multiplier hooks (5.8 scope)

        Raises:
            ValueError on unsupported asset_class (incl. 'crypto')
        """
        ac = (asset_class or "").lower().strip()
        if ac == "crypto":
            raise ValueError(
                "asset_class='crypto' is rejected per owner directive 2026-04-19"
            )
        if ac not in SUPPORTED_ASSET_CLASSES:
            raise ValueError(
                f"unsupported asset_class={asset_class!r}; "
                f"supported: {SUPPORTED_ASSET_CLASSES}"
            )

        base = self._base_notional(equity, asset_vol)

        if ac == "equity":
            return base

        if ac == "option":
            d = abs(float(delta)) if delta is not None else 1.0
            return base * d

        if ac == "fx":
            lots = max(1, round(base / FX_MICRO_LOT))
            return float(lots * FX_MICRO_LOT)

        if ac == "future":
            # Placeholder: contract-multiplier table arrives in 5.8 IBKR cycle.
            # For now, return base notional unchanged so the API is stable.
            return base

        # Unreachable -- guarded above. Defensive raise.
        raise ValueError(f"unhandled asset_class={ac!r}")


__all__ = [
    "DEFAULT_TARGET_VOL",
    "FX_MICRO_LOT",
    "MAX_LEVERAGE",
    "RiskEngine",
    "SUPPORTED_ASSET_CLASSES",
]
