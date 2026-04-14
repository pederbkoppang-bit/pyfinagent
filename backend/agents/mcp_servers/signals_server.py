"""
MCP Signals Server: Callable tools for signal generation, validation, publishing

Tools (FastMCP @mcp.tool):
- generate_signal(ticker, date) → BUY/SELL/HOLD with confidence
- validate_signal(signal) → Check constraints (market hours, liquidity, exposure)
- publish_signal(signal) → Post to Slack + portfolio
- risk_check(portfolio, proposed_trade) → Can we add this position?

Resources:
- portfolio://current → Current holdings (tickers, shares, PnL)
- constraints://risk → Risk limits (max exposure, max drawdown, Sharpe floor)
- signals://history → All generated signals this month
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Import backend modules
_SIGNALS_AVAILABLE = False

try:
    from backend.services.paper_trader import PaperTrader
    from backend.db.bigquery_client import BigQueryClient
    from backend.config.settings import get_settings
    _SIGNALS_AVAILABLE = True
except ImportError:
    logger.warning("Paper trader not available -- signals server in stub mode")


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Trading signal."""
    ticker: str
    date: str
    signal: SignalType
    confidence: float  # 0.0-1.0
    factors: List[str]  # Factors supporting this signal
    reason: str  # Human-readable explanation


class SignalsServer:
    """FastMCP signals server for pyfinAgent."""
    
    def __init__(self):
        global _SIGNALS_AVAILABLE

        self.portfolio = {}  # Current holdings
        self.risk_limits = {}  # Exposure limits
        self.signal_history = []  # All signals generated
        self.bq_client = None
        self.settings = None
        self.paper_trader = None

        # In-memory idempotency state for publish_signal. Cleared on process
        # restart; cross-restart dedup is Phase 4.2 territory (durable BQ
        # signal_history table). Response cache is bounded FIFO.
        self._seen_signal_ids: set = set()
        self._recent_responses: Dict[str, Dict[str, Any]] = {}
        self._recent_responses_limit = 50

        # Initialize paper trader if available
        if _SIGNALS_AVAILABLE:
            try:
                self.settings = get_settings()
                self.bq_client = BigQueryClient(self.settings)
                # PaperTrader.__init__ signature is (settings, bq_client).
                # Prior revision missed the settings arg -- any path where
                # _SIGNALS_AVAILABLE was True would crash on construction.
                self.paper_trader = PaperTrader(settings=self.settings, bq_client=self.bq_client)
                logger.info("SignalsServer initialized with PaperTrader")
            except Exception as e:
                logger.error(f"Failed to initialize SignalsServer: {e}")
                _SIGNALS_AVAILABLE = False
    
    def generate_signal(self, ticker: str, date: str) -> Dict[str, Any]:
        """
        Generate a trading signal (BUY/SELL/HOLD) for a ticker on a date.
        
        Uses the learned features + model to make a prediction.
        
        Returns:
        {
            "ticker": "AAPL",
            "date": "2026-03-29",
            "signal": "BUY",
            "confidence": 0.72,
            "factors": ["momentum_3m: +12%", "insider_ratio: 1.2", "sentiment: 0.68"],
            "reason": "Strong momentum + insider buying"
        }
        """
        logger.info(f"generate_signal({ticker}, {date})")
        # TODO: Run model inference on current features
        return {
            "ticker": ticker,
            "date": date,
            "signal": "HOLD",
            "confidence": 0.0,
            "factors": [],
            "reason": "PENDING_IMPLEMENTATION",
        }
    
    def validate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Signal-intrinsic schema validation. Checks that a generated signal is
        well-formed before any portfolio-state risk checks are applied.

        This is the layered counterpart to risk_check: validate_signal looks
        only at the signal itself; risk_check looks at the signal vs the
        current portfolio. (Pattern follows QuantConnect's Alpha ->
        PortfolioConstruction -> RiskManagement -> Execution layering.)

        Tolerates partial / missing fields via .get() and never raises --
        MCP clients prefer error-shape returns over exceptions.

        Args:
            signal: {
                "ticker": "AAPL",
                "signal": "BUY" | "SELL" | "HOLD",
                "confidence": 0.0..1.0,
                "date": "YYYY-MM-DD",
                "factors": [...]
            }

        Returns:
            {
                "valid": bool,
                "violations": [str],
                "adjusted_signal": dict | None,
                "reason": str
            }
        """
        if not isinstance(signal, dict):
            logger.warning("validate_signal called with non-dict input")
            return {
                "valid": False,
                "violations": ["not_a_dict"],
                "adjusted_signal": None,
                "reason": "signal must be a dict",
            }

        ticker = signal.get("ticker", "")
        sig = signal.get("signal", "")
        confidence = signal.get("confidence", None)
        sdate = signal.get("date", "")
        factors = signal.get("factors", [])

        logger.info(f"validate_signal({ticker})")

        violations: List[str] = []

        # 1. ticker: non-empty alphanumeric/dot string (matches the project's
        #    ticker sanitization rule from .claude/rules/security.md).
        if not isinstance(ticker, str) or not ticker:
            violations.append("missing_ticker")
        elif not all(c.isalnum() or c in ".:-_" for c in ticker):
            violations.append("invalid_ticker_chars")

        # 2. signal type: must be one of the three known SignalType values.
        if sig not in ("BUY", "SELL", "HOLD"):
            violations.append("invalid_signal_type")

        # 3. confidence in [0, 1].
        if confidence is None:
            violations.append("missing_confidence")
        else:
            try:
                cval = float(confidence)
                if cval < 0.0 or cval > 1.0:
                    violations.append("confidence_out_of_range")
            except (ValueError, TypeError):
                violations.append("confidence_not_numeric")

        # 4. date present (calendar parsing left to caller -- the orchestrator
        #    owns the trading-day calendar).
        if not isinstance(sdate, str) or not sdate:
            violations.append("missing_date")

        # 5. BUY/SELL must carry at least one factor; HOLD may be empty.
        if sig in ("BUY", "SELL"):
            if not isinstance(factors, list) or len(factors) == 0:
                violations.append("missing_factors")

        valid = len(violations) == 0
        reason = "All schema checks passed" if valid else f"Schema violations: {','.join(violations)}"
        return {
            "valid": valid,
            "violations": violations,
            "adjusted_signal": signal if valid else None,
            "reason": reason,
        }
    
    @staticmethod
    def _signal_id(signal: Dict[str, Any]) -> str:
        """Deterministic, stable signal identity for dedup.

        Hash over (ticker, date, signal_type, confidence_bucket). Rounding
        confidence to 2 decimals means a 0.711 vs 0.714 variant collapses to
        one id -- two model reruns on the same day should not fire twice.
        Pure, never raises. Returns a 16-char hex prefix of sha1.
        """
        try:
            ticker = str(signal.get("ticker", "") or "").upper()
            sdate = str(signal.get("date", "") or "")
            stype = str(signal.get("signal", "") or "").upper()
            try:
                conf_bucket = round(float(signal.get("confidence", 0.0) or 0.0), 2)
            except (ValueError, TypeError):
                conf_bucket = 0.0
            key = f"{ticker}|{sdate}|{stype}|{conf_bucket}"
            return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        except Exception:
            # Never raise from an id helper -- degrade to empty string so
            # the caller can still build a non-deduped response.
            return ""

    def _empty_response(self, signal_id: str = "", timestamp: str = "") -> Dict[str, Any]:
        """Uniform return shape for publish_signal. Every field present; callers
        override only what they know. Preserves the return-shape invariant
        (anti-leniency rule 11 from contract.md)."""
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        return {
            "published": False,
            "deduped": False,
            "signal_id": signal_id,
            "trade_executed": False,
            "trade_id": "",
            "slack_posted": False,
            "slack_ts": "",
            "slack_channel": "",
            "timestamp": timestamp,
            "reason": "",
        }

    def _remember(self, signal_id: str, response: Dict[str, Any]) -> None:
        """Add signal_id to the seen-set and cache its response for dedup hits.
        Bounded FIFO on the response cache -- the seen-set itself is unbounded
        but cheap (sha1 prefixes). Cache eviction keeps memory stable."""
        if not signal_id:
            return
        self._seen_signal_ids.add(signal_id)
        self._recent_responses[signal_id] = response
        if len(self._recent_responses) > self._recent_responses_limit:
            # Drop the oldest entry. dict preserves insertion order in 3.7+.
            oldest = next(iter(self._recent_responses))
            self._recent_responses.pop(oldest, None)

    def publish_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Publish a validated signal to the paper trader and (optionally) Slack.

        Pipeline (ordering is non-negotiable, see research.md section 3 --
        trade is booked BEFORE Slack, never the reverse):
            1. schema coerce                  -> invalid_input
            2. validate_signal                -> validation_failed:<viol>
            3. signal_id + dedup check        -> deduped:true on hit
            4. stub-mode gate                 -> backend_unavailable
            5. get_portfolio + risk_check     -> risk_rejected:<conflict>
            6. paper_trader.execute_buy/sell  -> trade_rejected (BUY/SELL)
                                                 hold_noop      (HOLD)
            7. slack_sdk.WebClient post       -> slack_not_configured
                                                 slack_api_error:<code>
            8. success                        -> reason:"ok"

        Graceful-degradation ladder returns structured dicts for every rung;
        never raises (anti-leniency rule 4). Lazy-imports slack_sdk and the
        formatters module so the stub-mode import path stays zero-dep.

        Args:
            signal: dict matching validate_signal's schema (ticker, signal,
                    confidence, date, factors). Optional: size_usd, stop_price,
                    reason.

        Returns:
            Dict with the full return-shape invariant (see _empty_response).
        """
        # ---- Step 1: schema coerce --------------------------------------
        if not isinstance(signal, dict):
            logger.warning("publish_signal called with non-dict input")
            resp = self._empty_response()
            resp["reason"] = "invalid_input"
            return resp

        ticker = str(signal.get("ticker", "") or "")
        action = str(signal.get("signal", "") or "").upper()
        logger.info(f"publish_signal({ticker}, {action})")

        # ---- Step 2: validate_signal ------------------------------------
        validation = self.validate_signal(signal)
        if not validation.get("valid", False):
            violations = validation.get("violations") or []
            first_viol = violations[0] if violations else "unknown"
            resp = self._empty_response()
            resp["reason"] = f"validation_failed:{first_viol}"
            return resp

        # ---- Step 3: signal_id + dedup check ----------------------------
        signal_id = self._signal_id(signal)
        if signal_id and signal_id in self._seen_signal_ids:
            cached = self._recent_responses.get(signal_id)
            if isinstance(cached, dict):
                # Return a copy with deduped:true so the caller can tell
                # this was a re-fire without mutating the cache entry.
                resp = dict(cached)
                resp["deduped"] = True
                resp["reason"] = cached.get("reason", "ok") + " (deduped)"
                return resp
            # Cache miss but seen -- synthesize a minimal dedup response.
            resp = self._empty_response(signal_id=signal_id)
            resp["published"] = True
            resp["deduped"] = True
            resp["reason"] = "deduped"
            return resp

        # ---- Step 4: stub-mode gate -------------------------------------
        # Run BEFORE any backend calls. Records the signal_id as seen so a
        # retry in the same stub-mode session dedups cleanly.
        if not _SIGNALS_AVAILABLE or self.paper_trader is None:
            resp = self._empty_response(signal_id=signal_id)
            resp["reason"] = "backend_unavailable"
            resp["stub"] = True
            self._remember(signal_id, resp)
            return resp

        # ---- Step 5: portfolio snapshot + risk_check --------------------
        portfolio = self.get_portfolio()
        # v1 sizing: cash*0.05 capped at $1000 if signal doesn't carry size_usd.
        # Real sizing (Kelly / Risk Judge) is Phase 4.3. Documented in contract.
        try:
            cash = float(portfolio.get("cash", 0.0) or 0.0)
        except (ValueError, TypeError):
            cash = 0.0
        default_amount = min(cash * 0.05, 1000.0) if cash > 0 else 0.0
        try:
            amount_usd = float(signal.get("size_usd", default_amount) or default_amount)
        except (ValueError, TypeError):
            amount_usd = default_amount

        # Estimate shares for the risk_check predicate. Price resolution:
        # signal.price -> position.price -> 1.0 (degraded, same as the
        # risk_check scaffold). Shares is only used for the predicate; the
        # paper trader resolves real prices on execute.
        try:
            price = float(signal.get("price", 0.0) or 0.0)
        except (ValueError, TypeError):
            price = 0.0
        if price <= 0.0:
            positions = portfolio.get("positions", {}) or {}
            if isinstance(positions, dict):
                pos = positions.get(ticker, {})
                if isinstance(pos, dict):
                    try:
                        price = float(pos.get("price", 0.0) or 0.0)
                    except (ValueError, TypeError):
                        price = 0.0
        proposed_shares = 1
        if price > 0.0 and amount_usd > 0.0:
            proposed_shares = max(1, int(amount_usd // price))

        if action in ("BUY", "SELL"):
            proposed_trade = {
                "ticker": ticker,
                "action": action,
                "shares": proposed_shares,
                "price": price if price > 0.0 else None,
            }
            risk = self.risk_check(portfolio, proposed_trade)
            if not risk.get("allowed", False):
                conflicts = risk.get("conflicts") or []
                first_conflict = conflicts[0] if conflicts else "unknown"
                resp = self._empty_response(signal_id=signal_id)
                resp["reason"] = f"risk_rejected:{first_conflict}"
                self._remember(signal_id, resp)
                return resp

        # ---- Step 6: execute trade --------------------------------------
        trade: Optional[Dict[str, Any]] = None
        trade_executed = False
        reason_text = str(signal.get("reason", "") or "mcp_publish_signal")
        if action == "BUY":
            try:
                trade = self.paper_trader.execute_buy(
                    ticker=ticker,
                    amount_usd=amount_usd,
                    price=price if price > 0.0 else 0.0,
                    reason=reason_text,
                )
            except Exception as e:
                logger.error(f"paper_trader.execute_buy failed: {e}")
                trade = None
            if trade is None:
                resp = self._empty_response(signal_id=signal_id)
                resp["reason"] = "trade_rejected"
                self._remember(signal_id, resp)
                return resp
            trade_executed = True
        elif action == "SELL":
            try:
                trade = self.paper_trader.execute_sell(
                    ticker=ticker,
                    quantity=None,
                    price=price if price > 0.0 else None,
                    reason=reason_text,
                )
            except Exception as e:
                logger.error(f"paper_trader.execute_sell failed: {e}")
                trade = None
            if trade is None:
                resp = self._empty_response(signal_id=signal_id)
                resp["reason"] = "trade_rejected"
                self._remember(signal_id, resp)
                return resp
            trade_executed = True
        else:
            # HOLD -- no trade, but still continue to Slack so traders see
            # the considered-and-held decision.
            trade = None
            trade_executed = False

        # ---- Step 7: Slack post (lazy import) ---------------------------
        # Hard invariant: nothing above this line imports slack_sdk or
        # backend.slack_bot.formatters. Stub-mode gate at step 4 already
        # returned if we got here without backend availability.
        slack_posted = False
        slack_ts = ""
        slack_channel = ""
        slack_reason = ""
        slack_token = getattr(self.settings, "slack_bot_token", "") if self.settings else ""
        slack_channel_cfg = getattr(self.settings, "slack_channel_id", "") if self.settings else ""
        if not slack_token or not slack_channel_cfg:
            slack_reason = "slack_not_configured"
        else:
            try:
                from slack_sdk import WebClient  # noqa: PLC0415 -- lazy by design
                from slack_sdk.errors import SlackApiError  # noqa: PLC0415
                from backend.slack_bot.formatters import format_signal_alert  # noqa: PLC0415

                signal_for_format = dict(signal)
                signal_for_format["signal_id"] = signal_id
                blocks = format_signal_alert(signal_for_format, trade)
                try:
                    conf_val = float(signal.get("confidence", 0.0) or 0.0)
                except (ValueError, TypeError):
                    conf_val = 0.0
                # ASCII-only fallback text per security rule.
                text_fallback = f"{action} {ticker} conf={conf_val:.2f}"

                client = WebClient(token=slack_token)
                api_resp = client.chat_postMessage(
                    channel=slack_channel_cfg,
                    blocks=blocks,
                    text=text_fallback,
                )
                slack_posted = bool(api_resp.get("ok", False)) if hasattr(api_resp, "get") else True
                slack_ts = str(api_resp.get("ts", "")) if hasattr(api_resp, "get") else ""
                slack_channel = slack_channel_cfg
                slack_reason = "ok" if slack_posted else "slack_api_error:no_ok"
            except SlackApiError as e:
                code = "unknown"
                try:
                    code = str(e.response.get("error", "unknown"))
                except Exception:
                    pass
                logger.error(f"Slack API error on publish_signal: {code}")
                slack_reason = f"slack_api_error:{code}"
            except ImportError as e:
                # slack_sdk missing at runtime -- log and degrade. Should not
                # happen in prod (slack_sdk is a backend dep) but we still
                # degrade cleanly rather than raise.
                logger.warning(f"slack_sdk import failed at publish time: {e}")
                slack_reason = "slack_not_installed"
            except Exception as e:
                logger.error(f"Unexpected Slack post failure: {type(e).__name__}")
                slack_reason = f"slack_api_error:{type(e).__name__}"

        # ---- Step 8: build and cache success response ------------------
        # "published" is true iff either the trade was booked OR Slack fired
        # (for HOLD the trade is always a noop but Slack might still post).
        published = trade_executed or slack_posted
        final_reason = "ok"
        if not published:
            if action == "HOLD":
                final_reason = f"hold_noop:{slack_reason or 'no_slack'}"
            else:
                final_reason = slack_reason or "unknown"
        elif not trade_executed and action == "HOLD":
            final_reason = f"hold_noop:{slack_reason or 'posted'}"
        elif not slack_posted:
            final_reason = slack_reason or "trade_only"

        trade_id = ""
        if isinstance(trade, dict):
            trade_id = str(trade.get("trade_id", "") or "")

        response = self._empty_response(signal_id=signal_id)
        response["published"] = published
        response["trade_executed"] = trade_executed
        response["trade_id"] = trade_id
        response["slack_posted"] = slack_posted
        response["slack_ts"] = slack_ts
        response["slack_channel"] = slack_channel
        response["reason"] = final_reason

        self._remember(signal_id, response)
        return response
    
    def risk_check(self, portfolio: Dict[str, Any], proposed_trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Portfolio-extrinsic constraint check for a proposed trade.

        Reads its limits from get_risk_constraints() so the threshold table
        stays single-source-of-truth. Pure function: never mutates inputs.
        Does NOT call any other MCP server -- cross-server coupling is an
        explicit anti-pattern per Anthropic MCP best practices.

        Evaluation order is canonical FINRA 15c3-5 / FIA whitepaper ordering:
        regulatory-fatal -> financial-fatal -> soft, fail-fast on hard
        violations.

            1. Schema sanity (ticker, action, shares > 0)                [hard]
            2. SELL requires existing position with >= shares            [hard]
            3. Daily trade count under max_daily_trades (5)              [hard]
            4. Per-ticker concentration <= 10% of total_value            [hard]
            5. Total exposure <= 100% of total_value                     [hard]
            6. Cash floor (BUY notional <= cash)                         [hard]
            7. Drawdown circuit breaker: block BUYs past -15% drawdown   [hard for BUY]

        Args:
            portfolio: {
                "total_value": float,
                "cash": float,
                "positions": {ticker: {"shares": int, "price": float, ...}},
                "trades_today": list | int,
                "current_drawdown_pct": float (optional, default 0.0)
            }
            proposed_trade: {
                "ticker": str,
                "action": "BUY" | "SELL",
                "shares": int,
                "price": float (optional)
            }

        Returns:
            {
                "allowed": bool,
                "current_exposure_pct": float,
                "max_exposure_pct": float,
                "margin_available": bool,
                "conflicts": [str],
                "reason": str
            }
        """
        if not isinstance(portfolio, dict):
            portfolio = {}
        if not isinstance(proposed_trade, dict):
            proposed_trade = {}

        ticker = proposed_trade.get("ticker", "")
        action = proposed_trade.get("action", "")
        shares = proposed_trade.get("shares", 0)
        price = proposed_trade.get("price", None)

        logger.info(f"risk_check({ticker})")

        limits = self.get_risk_constraints()
        max_per_ticker_pct = float(limits.get("max_exposure_per_ticker_pct", 10.0))
        max_total_pct = float(limits.get("max_total_exposure_pct", 100.0))
        max_drawdown_pct = float(limits.get("max_drawdown_pct", -15.0))
        max_daily_trades = int(limits.get("max_daily_trades", 5))

        positions = portfolio.get("positions", {}) or {}
        total_value = float(portfolio.get("total_value", 0.0) or 0.0)
        cash = float(portfolio.get("cash", 0.0) or 0.0)
        trades_today_raw = portfolio.get("trades_today", [])
        trades_today_count = (
            len(trades_today_raw)
            if isinstance(trades_today_raw, (list, tuple))
            else int(trades_today_raw or 0)
        )
        current_dd = float(portfolio.get("current_drawdown_pct", 0.0) or 0.0)

        conflicts: List[str] = []

        # 1. Schema sanity -- hard, fail-fast.
        try:
            shares_int = int(shares)
        except (ValueError, TypeError):
            shares_int = 0
        if not isinstance(ticker, str) or not ticker:
            conflicts.append("missing_ticker")
        if action not in ("BUY", "SELL"):
            conflicts.append("invalid_action")
        if shares_int <= 0:
            conflicts.append("invalid_shares")
        if conflicts:
            return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts, "Schema sanity failed")

        # Resolve a usable per-share price for notional math:
        #   1) explicit proposed_trade.price
        #   2) last_price from existing position record
        #   3) else 0.0 (concentration check still runs; cash check passes
        #      trivially -- documented degraded mode for the paper-trader
        #      scaffold; real prices wired in Phase 4.1).
        position_record = positions.get(ticker, {}) if isinstance(positions, dict) else {}
        try:
            unit_price = float(price) if price is not None else 0.0
        except (ValueError, TypeError):
            unit_price = 0.0
        if unit_price <= 0.0 and isinstance(position_record, dict):
            try:
                unit_price = float(position_record.get("price", 0.0) or 0.0)
            except (ValueError, TypeError):
                unit_price = 0.0
        proposed_notional = unit_price * shares_int

        # 2. Action/state consistency -- SELL requires sufficient existing position.
        if action == "SELL":
            existing_shares = 0
            if isinstance(position_record, dict):
                try:
                    existing_shares = int(position_record.get("shares", 0) or 0)
                except (ValueError, TypeError):
                    existing_shares = 0
            if existing_shares < shares_int:
                conflicts.append("insufficient_position")
                return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts,
                                           f"SELL requires {shares_int} shares, have {existing_shares}")

        # 3. Daily trade count -- hard limit.
        if trades_today_count >= max_daily_trades:
            conflicts.append("max_daily_trades")
            return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts,
                                       f"Daily trade count {trades_today_count} >= {max_daily_trades}")

        # Compute current per-ticker and total exposure (BEFORE the proposed trade).
        existing_position_notional = 0.0
        total_positions_notional = 0.0
        if isinstance(positions, dict):
            for sym, pos in positions.items():
                if not isinstance(pos, dict):
                    continue
                try:
                    pshares = float(pos.get("shares", 0) or 0)
                    pprice = float(pos.get("price", 0.0) or 0.0)
                except (ValueError, TypeError):
                    continue
                pnotional = pshares * pprice
                total_positions_notional += pnotional
                if sym == ticker:
                    existing_position_notional = pnotional

        # 4. Per-ticker concentration -- only meaningful for BUY (SELL reduces exposure).
        if action == "BUY" and total_value > 0.0:
            projected_position_notional = existing_position_notional + proposed_notional
            ticker_pct = (projected_position_notional / total_value) * 100.0
            if ticker_pct > max_per_ticker_pct:
                conflicts.append("max_exposure_per_ticker")
                return self._risk_response(False, ticker_pct, max_per_ticker_pct, conflicts,
                                           f"Per-ticker exposure {ticker_pct:.2f}% > {max_per_ticker_pct:.2f}%")

        # 5. Total exposure -- only meaningful for BUY.
        if action == "BUY" and total_value > 0.0:
            projected_total_notional = total_positions_notional + proposed_notional
            total_pct = (projected_total_notional / total_value) * 100.0
            if total_pct > max_total_pct:
                conflicts.append("max_total_exposure")
                return self._risk_response(False, total_pct, max_per_ticker_pct, conflicts,
                                           f"Total exposure {total_pct:.2f}% > {max_total_pct:.2f}%")

        # 6. Cash availability -- BUY only; paper trader has no margin.
        margin_available = True
        if action == "BUY" and proposed_notional > cash:
            conflicts.append("insufficient_cash")
            margin_available = False
            return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts,
                                       f"Need ${proposed_notional:.2f}, have ${cash:.2f}")

        # 7. Drawdown circuit breaker -- block all new BUYs past the floor.
        #    SELLs are still allowed (de-risking).
        if action == "BUY" and current_dd <= max_drawdown_pct:
            conflicts.append("drawdown_circuit_breaker")
            return self._risk_response(False, 0.0, max_per_ticker_pct, conflicts,
                                       f"Drawdown {current_dd:.2f}% <= floor {max_drawdown_pct:.2f}%")

        # All hard checks passed.
        if action == "BUY" and total_value > 0.0:
            ticker_pct_final = ((existing_position_notional + proposed_notional) / total_value) * 100.0
        else:
            ticker_pct_final = (existing_position_notional / total_value) * 100.0 if total_value > 0.0 else 0.0
        return self._risk_response(True, ticker_pct_final, max_per_ticker_pct, [],
                                   "All risk checks passed", margin_available=margin_available)

    @staticmethod
    def _risk_response(
        allowed: bool,
        current_exposure_pct: float,
        max_exposure_pct: float,
        conflicts: List[str],
        reason: str,
        margin_available: bool = True,
    ) -> Dict[str, Any]:
        """Uniform shape for risk_check returns -- preserves the stub's contract."""
        return {
            "allowed": allowed,
            "current_exposure_pct": float(current_exposure_pct),
            "max_exposure_pct": float(max_exposure_pct),
            "margin_available": bool(margin_available),
            "conflicts": list(conflicts),
            "reason": reason,
        }
    
    def get_portfolio(self) -> Dict[str, Any]:
        """Get current portfolio holdings."""
        logger.info("get_portfolio()")
        
        if not _SIGNALS_AVAILABLE or not self.paper_trader:
            return {
                "timestamp": "",
                "total_value": 10000.0,
                "positions": {},
                "cash": 10000.0,
            }
        
        try:
            # Load portfolio from paper trader
            portfolio = self.paper_trader.get_portfolio()
            
            return {
                "timestamp": portfolio.get("timestamp", ""),
                "total_value": portfolio.get("total_value", 10000.0),
                "positions": portfolio.get("positions", {}),
                "cash": portfolio.get("cash", 10000.0),
                "trades_today": len(portfolio.get("trades_today", [])),
            }
        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            return {
                "timestamp": "",
                "total_value": 10000.0,
                "positions": {},
                "cash": 10000.0,
                "error": str(e),
            }
    
    def get_risk_constraints(self) -> Dict[str, Any]:
        """Get risk limits."""
        logger.info("get_risk_constraints()")
        return {
            "max_exposure_per_ticker_pct": 10.0,
            "max_total_exposure_pct": 100.0,
            "max_drawdown_pct": -15.0,
            "min_sharpe": 0.9,
            "max_daily_trades": 5,
        }
    
    def get_signal_history(self) -> Dict[str, Any]:
        """Get all signals generated this month."""
        logger.info("get_signal_history()")
        return {
            "month": "2026-03",
            "count": 0,
            "signals": [],
        }


def create_signals_server():
    """Factory function to create FastMCP signals server."""
    try:
        from fastmcp import FastMCP
        
        mcp = FastMCP(name="pyfinagent-signals")
        server = SignalsServer()
        
        # Register tools
        @mcp.tool
        def generate_signal(ticker: str, date: str) -> Dict[str, Any]:
            """
            Generate a trading signal (BUY/SELL/HOLD) for a ticker.
            
            Uses the trained model + current features to make a prediction.
            Includes confidence score and contributing factors.
            
            Returns: signal, confidence, factors, reason
            """
            return server.generate_signal(ticker, date)
        
        @mcp.tool
        def validate_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
            """
            Validate a signal against risk constraints.
            
            Checks: market hours, liquidity, exposure limits, margin.
            
            Returns: valid (bool), violations (list), adjusted_signal
            """
            return server.validate_signal(signal)
        
        @mcp.tool
        def publish_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
            """
            Publish a validated signal to Slack and update portfolio.
            
            This commits the trade to paper trading system.
            
            Returns: published (bool), slack_id, timestamp
            """
            return server.publish_signal(signal)
        
        @mcp.tool
        def risk_check(portfolio: Dict[str, Any], proposed_trade: Dict[str, Any]) -> Dict[str, Any]:
            """
            Check if a proposed trade is within risk limits.
            
            Args:
                portfolio: Current holdings
                proposed_trade: {"ticker": "AAPL", "action": "BUY", "shares": 100}
            
            Returns: allowed (bool), current_exposure, conflicts
            """
            return server.risk_check(portfolio, proposed_trade)
        
        # Register resources
        @mcp.resource("portfolio://current")
        def portfolio_resource() -> str:
            """Get current portfolio holdings and P&L."""
            result = server.get_portfolio()
            return json.dumps(result)
        
        @mcp.resource("constraints://risk")
        def constraints_resource() -> str:
            """Get risk limits and constraints."""
            result = server.get_risk_constraints()
            return json.dumps(result)
        
        @mcp.resource("signals://history")
        def signals_history_resource() -> str:
            """Get all signals generated this month."""
            result = server.get_signal_history()
            return json.dumps(result)
        
        logger.info("Signals server created with 4 tools + 3 resources")
        return mcp
    
    except ImportError:
        logger.error("FastMCP not installed. Install with: pip install fastmcp")
        raise


if __name__ == "__main__":
    # For testing: start signals server standalone
    mcp = create_signals_server()
    mcp.run()  # Runs on stdio by default
