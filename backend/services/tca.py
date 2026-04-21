"""phase-4.8 step 4.8.0 Transaction Cost Analysis (implementation shortfall).

Per Perold 1988 / AFML Lopez de Prado ch.20, implementation shortfall
is the cost of turning a paper portfolio into an implemented one.
Normalized to basis points and sign-sensitive (positive = cost):

    IS_bps = side_sign * (fill_price - arrival_price) / arrival_price * 10_000

where side_sign = +1 for buys, -1 for sells.

Arrival price convention: previous-day close ("decision price") per
Wikipedia IS + LSEG's documented fund-manager interpretation. For an
intraday-arrival-mid interpretation see Talos / QB -- both are valid;
we pick decision-price because pyfinagent's backtest driver uses BQ
last-close fills, so using the SAME price for both sides would yield
a degenerate IS=0 for every fill.

Canonical liquid universe: the 20-name S&P set already used by
paper_execution_parity.py (Cycle 64) and virtual_fund_parity.py
(Cycle 67). Keeping the set in one place so the report + harness
agree.
"""
from __future__ import annotations

import json

from backend.utils import json_io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


LIQUID_SYMBOLS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "AVGO", "ORCL", "AMD", "INTC", "IBM", "CRM", "ADBE", "QCOM",
    "CSCO", "NFLX", "PYPL", "SHOP", "UBER",
)


_REPO = Path(__file__).resolve().parents[2]
TCA_LOG_PATH = _REPO / "handoff" / "tca_log.jsonl"


@dataclass
class TCAEvent:
    """One IS measurement. Shape-compatible with handoff/tca_log.jsonl."""
    ts: str
    client_order_id: str
    symbol: str
    side: str                     # "buy" | "sell"
    qty: float
    fill_price: float
    arrival_price: float
    is_bps: float
    notional_usd: float
    liquid: bool
    source: str                   # "bq_sim" | "mock_alpaca" | "alpaca_paper"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def compute_is_bps(fill_price: float, arrival_price: float, side: str) -> float:
    """Implementation shortfall in basis points; positive = cost.

    Raises ValueError on non-positive arrival_price (degenerate case
    that would otherwise silently return inf or sign-inverted noise).
    """
    if arrival_price is None or arrival_price <= 0:
        raise ValueError(f"arrival_price must be positive, got {arrival_price!r}")
    sign = 1.0 if side.lower() == "buy" else -1.0
    return sign * (fill_price - arrival_price) / arrival_price * 10_000.0


def log_tca_event(
    *,
    client_order_id: str,
    symbol: str,
    side: str,
    qty: float,
    fill_price: float,
    arrival_price: float,
    source: str,
    ts: str | None = None,
    meta: dict[str, Any] | None = None,
    log_path: Path | None = None,
) -> TCAEvent:
    """Compute IS + append one jsonl row. Returns the event.

    Caller is expected to supply `arrival_price` -- the previous-day
    close at decision time -- rather than re-using fill_price.
    """
    path = log_path or TCA_LOG_PATH
    is_bps = compute_is_bps(fill_price, arrival_price, side)
    notional = fill_price * qty
    ev = TCAEvent(
        ts=ts or datetime.now(timezone.utc).isoformat(),
        client_order_id=client_order_id,
        symbol=symbol,
        side=side.lower(),
        qty=float(qty),
        fill_price=float(fill_price),
        arrival_price=float(arrival_price),
        is_bps=round(is_bps, 4),
        notional_usd=round(notional, 2),
        liquid=symbol in LIQUID_SYMBOLS,
        source=source,
        meta=meta or {},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append JSONL; one row per fill.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev.to_dict(), default=str) + "\n")
    return ev


def read_log(log_path: Path | None = None) -> list[dict]:
    path = log_path or TCA_LOG_PATH
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json_io.parse_json_line(line))
            except json.JSONDecodeError:
                logger.warning("tca: skipping malformed jsonl row")
    return rows


__all__ = [
    "LIQUID_SYMBOLS",
    "TCA_LOG_PATH",
    "TCAEvent",
    "compute_is_bps",
    "log_tca_event",
    "read_log",
]
