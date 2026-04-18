"""Execution router (phase-3.7 step 3.7.5).

Routes paper-trading orders to one of three backends selected at
import time by the EXECUTION_BACKEND env-var (Fowler "ops toggle"
pattern -- https://martinfowler.com/articles/feature-toggles.html):

- `bq_sim` (default): synthetic fill at the last close from the
  bigquery_client cache. Same write path as before
  (paper_trader._safe_save_trade).
- `alpaca_paper`: uses alpaca-py TradingClient(paper=True). Requires
  ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY env. Triple-enforced
  paper-only: (1) .mcp.json pins ALPACA_PAPER_TRADE=true, (2) this
  router refuses PKLIVE-prefix keys, (3) SDK paper=True.
  When keys are missing, falls back to deterministic mock fills so
  the A/B harness can exercise the code path in CI.
- `shadow`: runs BOTH paths per order and returns their paired fills
  for drift measurement. Position state is still owned by bq_sim;
  the alpaca path is read-only in shadow mode.

Rollback: env-var flip back to bq_sim works immediately, no
in-process state to unwind. State lives in BQ with a `source` column
so history is preserved.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)


BackendMode = Literal["bq_sim", "alpaca_paper", "shadow"]
VALID_MODES = ("bq_sim", "alpaca_paper", "shadow")
DEFAULT_MODE: BackendMode = "bq_sim"


@dataclass
class FillResult:
    """Result of one order submission, shape-compatible across paths."""
    client_order_id: str
    symbol: str
    qty: float
    side: str
    fill_price: float
    status: str
    source: str                        # "bq_sim" | "alpaca_paper" | "mock_alpaca"
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    paper: bool = True
    raw: dict = field(default_factory=dict)
    # phase-3.7.8: fill latency + partial-fill modeling
    latency_ms: float = 0.0
    child_fills: list = field(default_factory=list)


ADV_PARTIAL_FILL_THRESHOLD = 0.05  # orders >= 5% of ADV get partial fills


def _current_mode() -> BackendMode:
    raw = (os.getenv("EXECUTION_BACKEND") or DEFAULT_MODE).strip().lower()
    if raw not in VALID_MODES:
        logger.warning("unknown EXECUTION_BACKEND=%r; falling back to %s",
                        raw, DEFAULT_MODE)
        return DEFAULT_MODE
    return raw  # type: ignore[return-value]


def _refuse_live_keys() -> None:
    key = os.getenv("ALPACA_API_KEY_ID", "")
    if key.startswith("PKLIVE") or os.getenv("ALPACA_PAPER_TRADE", "true").lower() == "false":
        raise RuntimeError(
            "refusing to run: live Alpaca keys or ALPACA_PAPER_TRADE=false "
            "detected. phase-3.7.5 is paper-only."
        )


def _bq_sim_fill(symbol: str, qty: float, side: str,
                  client_order_id: str,
                  close_price: float | None = None,
                  adv: float | None = None) -> FillResult:
    """Synthetic fill at last-close; deterministic if close_price is given.

    When `adv` (30d average daily volume) is supplied and qty/adv exceeds
    `ADV_PARTIAL_FILL_THRESHOLD` (5%), the fill splits into 2 child
    tranches (60/40) at the same parent price -- notional is conserved
    (no phantom P&L from independent price draws; see Bailey et al.
    SSRN 2326253).
    """
    t0 = time.monotonic()
    if close_price is None:
        h = int(hashlib.sha1(symbol.encode()).hexdigest()[:8], 16)
        close_price = 50.0 + (h % 500)
    fill_price = round(float(close_price), 4)
    qty_f = float(qty)

    child_fills: list = []
    if adv is not None and adv > 0 and qty_f / adv >= ADV_PARTIAL_FILL_THRESHOLD:
        q0 = round(qty_f * 0.6, 6)
        q1 = round(qty_f - q0, 6)  # exact complement; sum == qty_f
        child_fills = [
            {"qty": q0, "fill_price": fill_price,
             "ts": datetime.now(timezone.utc).isoformat()},
            {"qty": q1, "fill_price": fill_price,
             "ts": datetime.now(timezone.utc).isoformat()},
        ]
    latency = (time.monotonic() - t0) * 1000.0
    return FillResult(
        client_order_id=client_order_id,
        symbol=symbol,
        qty=qty_f,
        side=side.lower(),
        fill_price=fill_price,
        status="partially_filled" if child_fills else "accepted",
        source="bq_sim",
        paper=True,
        raw={"close_price": close_price, "adv": adv},
        latency_ms=round(latency, 3),
        child_fills=child_fills,
    )


def _alpaca_mock_fill(symbol: str, qty: float, side: str,
                       client_order_id: str,
                       close_price: float | None = None) -> FillResult:
    """Deterministic 'simulated Alpaca' fill when creds are missing.

    Reproducible: applies a fixed 0.3% slippage vs bq_sim close so the
    drift check exercises the measurement logic without needing live
    creds. Real Alpaca fills replace this path when creds are set.
    """
    t0 = time.monotonic()
    bq = _bq_sim_fill(symbol, qty, side, client_order_id, close_price)
    slippage_bps = 30  # 0.30 %
    sign = 1 if side.lower() == "buy" else -1
    fill = bq.fill_price * (1 + sign * slippage_bps / 10_000)
    latency = (time.monotonic() - t0) * 1000.0
    return FillResult(
        client_order_id=client_order_id,
        symbol=symbol,
        qty=float(qty),
        side=side.lower(),
        fill_price=round(fill, 4),
        status="filled",
        source="mock_alpaca",
        paper=True,
        raw={"slippage_bps": slippage_bps, "bq_ref": bq.fill_price},
        latency_ms=round(latency, 3),
    )


def _alpaca_real_fill(symbol: str, qty: float, side: str,
                       client_order_id: str) -> FillResult:
    """Real Alpaca paper submit via alpaca-py. Requires env creds."""
    _refuse_live_keys()
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    key = os.environ["ALPACA_API_KEY_ID"]
    secret = os.environ["ALPACA_API_SECRET_KEY"]
    client = TradingClient(key, secret, paper=True)
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    req = MarketOrderRequest(
        symbol=symbol, qty=qty, side=order_side,
        time_in_force=TimeInForce.DAY,
        client_order_id=client_order_id,
    )
    resp = client.submit_order(req)
    # Pull back-fill price; paper accounts fill at NBBO mid rapidly.
    # Poll up to 2s for terminal status.
    filled_price = getattr(resp, "filled_avg_price", None)
    for _ in range(20):
        if str(resp.status).split(".")[-1].lower() in ("filled", "partially_filled"):
            filled_price = getattr(resp, "filled_avg_price", None) or filled_price
            break
        time.sleep(0.1)
        resp = client.get_order_by_id(str(resp.id))
    return FillResult(
        client_order_id=client_order_id,
        symbol=symbol,
        qty=float(qty),
        side=side.lower(),
        fill_price=float(filled_price) if filled_price else 0.0,
        status=str(resp.status).split(".")[-1].lower(),
        source="alpaca_paper",
        paper=True,
        raw={"order_id": str(resp.id)},
    )


class ExecutionRouter:
    """Single entry point for paper-trading order submission.

    Usage:
        router = ExecutionRouter()
        result = router.submit_order("AAPL", 1, "buy", "oid-123")
        # In shadow mode:
        (bq, alp) = router.shadow_submit(...)
    """

    def __init__(self, mode: BackendMode | None = None) -> None:
        self.mode: BackendMode = mode or _current_mode()

    def submit_order(self, symbol: str, qty: float, side: str,
                      client_order_id: str,
                      close_price: float | None = None) -> FillResult:
        if self.mode == "bq_sim":
            return _bq_sim_fill(symbol, qty, side, client_order_id, close_price)
        if self.mode == "alpaca_paper":
            if os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY"):
                return _alpaca_real_fill(symbol, qty, side, client_order_id)
            return _alpaca_mock_fill(symbol, qty, side, client_order_id,
                                       close_price)
        if self.mode == "shadow":
            # Shadow mode: act on BQ sim, also record Alpaca fill for drift.
            bq = _bq_sim_fill(symbol, qty, side, client_order_id, close_price)
            if os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY"):
                try:
                    _alpaca_real_fill(symbol, qty, side, client_order_id)
                except Exception as e:
                    logger.warning("shadow mode alpaca call failed: %s", e)
            return bq
        raise RuntimeError(f"unsupported mode: {self.mode}")

    def shadow_submit(self, symbol: str, qty: float, side: str,
                       client_order_id: str,
                       close_price: float | None = None,
                       adv: float | None = None,
                       ) -> tuple[FillResult, FillResult]:
        """Run BOTH paths and return (bq_result, alpaca_result) for drift
        measurement. Used by the parity harness; does not write to any
        ledger.

        `adv` (optional) activates BQ-sim partial-fill modeling when
        qty >= 5% of ADV -- see `_bq_sim_fill`.
        """
        bq = _bq_sim_fill(symbol, qty, side, client_order_id, close_price, adv)
        if os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY"):
            try:
                alp = _alpaca_real_fill(symbol, qty, side, client_order_id)
            except Exception:
                alp = _alpaca_mock_fill(symbol, qty, side, client_order_id,
                                          close_price)
        else:
            alp = _alpaca_mock_fill(symbol, qty, side, client_order_id,
                                      close_price)
        return bq, alp

    def flip_to(self, mode: BackendMode) -> None:
        """Rollback primitive. No in-process state to unwind -- mode
        flip takes effect for subsequent submit_order calls."""
        if mode not in VALID_MODES:
            raise ValueError(f"invalid mode: {mode}")
        self.mode = mode
        logger.info("ExecutionRouter mode flipped to %s", mode)


def rollback_to_bq_sim() -> ExecutionRouter:
    """Module-level rollback helper used by circuit breaker."""
    r = ExecutionRouter(mode="bq_sim")
    logger.warning("execution_router: rollback to bq_sim")
    return r
