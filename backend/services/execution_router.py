"""Execution router (phase-3.7 step 3.7.5).

Routes paper-trading orders to one of three backends selected PER
ExecutionRouter CONSTRUCTION by `resolve_execution_mode()` (Fowler "ops
toggle" pattern -- https://martinfowler.com/articles/feature-toggles.html).
(phase-68.1 correction: this docstring used to say "selected at import
time", which was never true -- `__init__` resolves the mode on every
construction, so a mid-process env change affects the next router built.)

- `bq_sim` (default): synthetic fill at the last close from the
  bigquery_client cache. Same write path as before
  (paper_trader._safe_save_trade).
- `alpaca_paper`: uses alpaca-py TradingClient(paper=True). Requires
  ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY env. Paper-only enforcement,
  in order of how much work each part actually does:
    (1) SDK `paper=True` -> base URL pinned to paper-api.alpaca.markets.
        THIS IS THE LOAD-BEARING GUARD -- the paper and live environments
        are separated by DOMAIN.
    (2) ALPACA_PAPER_TRADE=false is refused outright.
    (3) a live-marked key-prefix filter. phase-68.1 correction: Alpaca does
        NOT document any prefix difference between paper and live keys, so
        this is a cheap extra filter, NOT the guarantee the previous wording
        ("triple-enforced ... refuses PKLIVE-prefix keys") implied.
  When keys are missing, falls back to deterministic mock fills so the A/B
  harness can exercise the code path in CI -- and since phase-68.1 says so
  LOUDLY at ERROR instead of silently.
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


def resolve_execution_mode() -> tuple[BackendMode, str]:
    """Resolve the execution backend AND report where the value came from.

    phase-68.1. The provenance is the point, not a nicety: before this existed,
    `EXECUTION_BACKEND` was read only from `os.environ`, while the project's
    configuration channel is `backend/.env` loaded by pydantic-settings -- which
    populates `Settings` but does NOT export to `os.environ`. A value set in .env
    was therefore invisible here, and the router used the default forever with
    nothing in the logs to say so. "bq_sim" logged without its source cannot
    distinguish "deliberately configured" from "your setting was silently dropped".

    Precedence (first hit wins), each reported as `source`:
      "env"     -- os.environ, i.e. the launchd plist or a shell export
      "dotenv"  -- backend/.env via Settings.execution_backend
      "default" -- neither set anything; DEFAULT_MODE

    An unrecognised value from EITHER channel falls back to DEFAULT_MODE with
    source "invalid:<channel>" -- it never escalates and never raises, because a
    typo in a config file must not take the order path down.
    """
    raw_env = os.getenv("EXECUTION_BACKEND")
    if raw_env is not None and raw_env.strip():
        candidate = raw_env.strip().lower()
        if candidate in VALID_MODES:
            return candidate, "env"  # type: ignore[return-value]
        logger.warning("unknown EXECUTION_BACKEND=%r from env; falling back to %s",
                       raw_env, DEFAULT_MODE)
        return DEFAULT_MODE, "invalid:env"

    try:
        from backend.config.settings import get_settings
        candidate = str(getattr(get_settings(), "execution_backend", "") or "").strip().lower()
    except Exception:  # pragma: no cover - settings must never break the order path
        candidate = ""
    if candidate:
        if candidate in VALID_MODES:
            return candidate, "dotenv"  # type: ignore[return-value]
        logger.warning("unknown execution_backend=%r from .env/settings; falling back to %s",
                       candidate, DEFAULT_MODE)
        return DEFAULT_MODE, "invalid:dotenv"

    return DEFAULT_MODE, "default"


def log_resolved_execution_mode() -> tuple[BackendMode, str]:
    """Emit the startup provenance line and return what it reported.

    Called once from the FastAPI startup path so the line lands in the real
    launchd process's log -- the only place that can prove what the running
    service actually resolved (there is no endpoint, and the mode is resolved
    per-construction, not at import).
    """
    mode, source = resolve_execution_mode()
    logger.info(
        "phase-68.1 execution backend: mode=%s source=%s (paper-only enforced; "
        "default=%s)", mode, source, DEFAULT_MODE,
    )
    # phase-68.1 (criterion 3, cycle-2 fix): the missing-credentials error must fire
    # at STARTUP, not at the first order. The first version only warned from the fill
    # path, so an operator who set alpaca_paper without credentials learned about it
    # when the first trade of the day silently became a mock fill -- hours after the
    # misconfiguration, and only if they were reading logs at that moment. Checking
    # here means they learn at configuration time. Fail-open: observability must never
    # take startup down.
    if mode == "alpaca_paper" and not (
        os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY")
    ):
        try:
            _warn_missing_alpaca_creds()
        except Exception:  # pragma: no cover - never break startup
            logger.warning("phase-68.1: missing-creds check failed (fail-open)")
    return mode, source


def _current_mode() -> BackendMode:
    mode, _source = resolve_execution_mode()
    return mode


# phase-68.1: prefixes that must never reach the order path.
#
# HONESTY NOTE, and do not delete it: "PKLIVE" is NOT an Alpaca-documented format.
# The 68.1 research gate read three official Alpaca sources in full and found NO
# prefix or format difference between paper and live API keys -- the environments
# are separated by DOMAIN (paper-api.alpaca.markets vs api.alpaca.markets), not by
# key shape. So this prefix check is a cheap belt-and-braces filter that can catch
# an obviously-mislabelled key, and it is NOT what makes the system paper-only.
# The load-bearing guards are the paper base-URL pin (SDK paper=True) and the
# ALPACA_PAPER_TRADE refusal below. Treating this prefix as the real guard would be
# a false sense of safety.
_LIVE_KEY_PREFIXES = ("PKLIVE", "AKLIVE")


def _refuse_live_keys() -> None:
    key = os.getenv("ALPACA_API_KEY_ID", "")
    if key.upper().startswith(_LIVE_KEY_PREFIXES):
        raise RuntimeError(
            f"refusing to run: Alpaca key begins with a live-marked prefix "
            f"{_LIVE_KEY_PREFIXES}. phase-3.7.5 is paper-only. (Note: Alpaca does "
            f"not actually distinguish paper/live keys by prefix -- the real "
            f"guard is the paper base URL; see _LIVE_KEY_PREFIXES.)"
        )
    if os.getenv("ALPACA_PAPER_TRADE", "true").strip().lower() == "false":
        raise RuntimeError(
            "refusing to run: ALPACA_PAPER_TRADE=false detected. "
            "phase-3.7.5 is paper-only."
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


_MISSING_CREDS_WARNED = False


def _warn_missing_alpaca_creds() -> None:
    """phase-68.1 (criterion 3): say it out loud, once, and name the keys.

    Before this, `mode=alpaca_paper` with no credentials fell through to
    `_alpaca_mock_fill` in SILENCE -- that function has no logging of any kind. An
    operator who set the mode expecting real paper orders got synthetic fills with a
    fixed 0.3% slippage and nothing anywhere to say the credentials never arrived.
    The fills even carry `source="mock_alpaca"`, so the ledger was honest while the
    logs were mute.

    ERROR, not WARNING: the running configuration does not do what it says. Latched
    so an order-rate loop cannot flood the log -- the point is to be unmissable
    once, not to be noisy.
    """
    global _MISSING_CREDS_WARNED
    if _MISSING_CREDS_WARNED:
        return
    _MISSING_CREDS_WARNED = True
    missing = [n for n in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY") if not os.getenv(n)]
    logger.error(
        "phase-68.1: EXECUTION_BACKEND=alpaca_paper but Alpaca credentials are "
        "MISSING (%s). Falling back to deterministic MOCK fills (source=mock_alpaca, "
        "fixed 30bps slippage) -- these are NOT real Alpaca paper orders. Set the "
        "named variables or set EXECUTION_BACKEND=bq_sim to make the intent explicit.",
        ", ".join(missing) or "none-detected",
    )


def _reset_missing_creds_warning() -> None:
    """Test seam for the latch above. Not called by production code."""
    global _MISSING_CREDS_WARNED
    _MISSING_CREDS_WARNED = False


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


_MAX_NOTIONAL_DEFAULT_USD = 10000.0


def _max_notional_usd() -> float:
    """Order-size clamp. Raises on notional > threshold to block
    LLM hallucinations (e.g., 10,000-share buy orders). Default $10,000;
    override via ALPACA_MAX_NOTIONAL_USD env var.
    """
    raw = os.getenv("ALPACA_MAX_NOTIONAL_USD", "")
    if not raw:
        return _MAX_NOTIONAL_DEFAULT_USD
    try:
        val = float(raw)
        return val if val > 0 else _MAX_NOTIONAL_DEFAULT_USD
    except ValueError:
        logger.warning("bad ALPACA_MAX_NOTIONAL_USD=%r; using default", raw)
        return _MAX_NOTIONAL_DEFAULT_USD


def _alpaca_real_fill(symbol: str, qty: float, side: str,
                       client_order_id: str,
                       reference_price: float | None = None) -> FillResult:
    """Real Alpaca paper submit via alpaca-py. Requires env creds.

    Guards:
      1. `_refuse_live_keys()` -- refuses PKLIVE* / ALPACA_PAPER_TRADE=false.
      2. `max_notional_usd` clamp -- raises if qty * reference_price
         (or a quick last-quote lookup) exceeds ALPACA_MAX_NOTIONAL_USD
         (default $10,000). Blocks order-size hallucination before any
         client.submit_order call.
    """
    _refuse_live_keys()
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    key = os.environ["ALPACA_API_KEY_ID"]
    secret = os.environ["ALPACA_API_SECRET_KEY"]

    # ── max_notional_usd clamp (pre-submit) ──────────────────────────
    est_price = reference_price
    if est_price is None or est_price <= 0:
        # Cheap price lookup; fail-open on any error (clamp uses a
        # defensive $1e6 upper-bound so the clamp still trips on
        # absurdly large qty even when price lookup fails).
        try:
            import requests
            r = requests.get(
                f"https://data.alpaca.markets/v2/stocks/{symbol}/snapshot",
                headers={
                    "APCA-API-KEY-ID": key,
                    "APCA-API-SECRET-KEY": secret,
                    "accept": "application/json",
                },
                timeout=5,
            )
            snap = r.json() or {}
            lp = snap.get("latestTrade", {}).get("p") or snap.get("latestQuote", {}).get("ap")
            est_price = float(lp) if lp else 1.0e6
        except Exception:
            est_price = 1.0e6  # defensive: unknown price -> treat as worst-case

    notional = float(qty) * float(est_price)
    cap = _max_notional_usd()
    if notional > cap:
        raise RuntimeError(
            f"max_notional_usd clamp: order {side} {qty} {symbol} @ ~${est_price:.2f} "
            f"= ${notional:,.2f} exceeds ${cap:,.2f}. Raise ALPACA_MAX_NOTIONAL_USD "
            f"explicitly if intended."
        )

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
            _warn_missing_alpaca_creds()
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
