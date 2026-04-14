"""
MCP Signals Server: Callable tools for signal generation, validation, publishing

Tools (FastMCP @mcp.tool):
- generate_signal(ticker, date) -> BUY/SELL/HOLD with confidence
- validate_signal(signal) -> Check constraints (market hours, liquidity, exposure)
- publish_signal(signal) -> Post to Slack + portfolio
- risk_check(portfolio, proposed_trade) -> Can we add this position?

Resources:
- portfolio://current -> Current holdings (tickers, shares, PnL)
- constraints://risk -> Risk limits (max exposure, max drawdown, Sharpe floor)
- signals://history -> All generated signals this month
"""

import copy
import hashlib
import json
import logging
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple
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

        # Phase 4.3: trailing drawdown state. Running peak-equity high-water
        # mark for the drawdown tier computation in track_drawdown(). In-memory
        # only this session; durable cross-restart persistence is Phase 4.2.
        self._peak_equity: Optional[float] = None

        # Phase 4.2.2: signal accuracy tracking. `signal_history` (inherited
        # from the stub init above) is the append-only time series populated
        # by publish_signal step 9. `_signals_by_id` is the O(1) lookup index
        # used by track_signal_accuracy. Both point at the same dict records
        # (list entry IS dict mirrored in index), so in-place outcome updates
        # are visible from both views. In-memory only this session; durable
        # BQ `signals_log` persistence is Phase 4.2.4.
        self._signals_by_id: Dict[str, Dict[str, Any]] = {}

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
        # Phase 4.3 sizing: hybrid lite-formula via size_position(), which
        # takes the min of (hard pct cap, half-Kelly confidence-weighted,
        # inverse-vol) with graceful degradation. Explicit signal.size_usd
        # still overrides (preserves the 4.1 contract).
        amount_usd = self.size_position(signal, portfolio)

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

        # ---- Step 9: append to signal_history for accuracy tracking ----
        # Phase 4.2.2. Only successful publishes (published=True) get a
        # history entry; rejected / degraded paths are captured in
        # _recent_responses but not in the accuracy log. Deepcopy guards
        # against caller mutation of the stored record. The dict entry
        # is simultaneously the list element and the _signals_by_id value
        # -- track_signal_accuracy mutates it in place, so both views
        # stay consistent without any re-sync step.
        if published:
            try:
                self._append_signal_history(signal_id, signal, trade)
            except Exception as e:
                # Anti-leniency: never let a history append failure
                # break the publish path. Log and continue.
                logger.warning(f"signal_history append failed: {type(e).__name__}")

        return response

    def _append_signal_history(
        self,
        signal_id: str,
        signal: Dict[str, Any],
        trade: Optional[Dict[str, Any]],
    ) -> None:
        """Append a published signal to signal_history + _signals_by_id index.

        Phase 4.2.2 helper. Pure append; never mutates the input signal.
        Record shape matches what track_signal_accuracy + get_accuracy_report
        expect. Price resolution: signal.price -> trade.price -> 0.0.
        """
        if not signal_id:
            return
        # Dedup: if we've already recorded this signal_id, skip. The
        # dedup path in publish_signal step 3 already returns before
        # reaching step 9, so this is defense in depth.
        if signal_id in self._signals_by_id:
            return

        entry_price = 0.0
        try:
            raw_price = signal.get("price", 0.0) if isinstance(signal, dict) else 0.0
            entry_price = float(raw_price or 0.0)
        except (ValueError, TypeError):
            entry_price = 0.0
        if entry_price <= 0.0 and isinstance(trade, dict):
            try:
                entry_price = float(trade.get("price", 0.0) or 0.0)
            except (ValueError, TypeError):
                entry_price = 0.0

        record = {
            "signal_id": signal_id,
            "ticker": str(signal.get("ticker", "")) if isinstance(signal, dict) else "",
            "signal_type": str(signal.get("signal", "")).upper() if isinstance(signal, dict) else "",
            "confidence": 0.0,
            "date": str(signal.get("date", "")) if isinstance(signal, dict) else "",
            "entry_price": entry_price,
            "factors": [],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            # Outcome fields -- populated later by track_signal_accuracy.
            "outcome": "pending",
            "scored": False,
            "hit": None,
            "exit_price": None,
            "exit_date": None,
            "forward_return_pct": None,
            "holding_days": None,
        }
        # Fill confidence + factors defensively.
        if isinstance(signal, dict):
            try:
                record["confidence"] = float(signal.get("confidence", 0.0) or 0.0)
            except (ValueError, TypeError):
                record["confidence"] = 0.0
            factors = signal.get("factors", [])
            if isinstance(factors, list):
                # Deepcopy to decouple from caller's list.
                record["factors"] = copy.deepcopy(factors)

        self.signal_history.append(record)
        self._signals_by_id[signal_id] = record
    
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

    def size_position(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> float:
        """Hybrid lite-formula position sizing (Phase 4.3).

        Returns the recommended USD notional for a proposed BUY, as the
        minimum of up to three independent caps:

          (a) Hard percent-of-equity cap: min(equity * max_position_pct/100,
              max_position_usd). Always computed. Preserves the v1 worst-case
              bound.
          (b) Confidence-weighted half-Kelly arm: 0.5 * confidence * equity.
              Only considered when signal.confidence is a numeric in [0, 1].
              Degrades gracefully when mu_hat / var_hat are not available --
              we don't synthesise fake edge estimates (anti-leniency rule 6).
          (c) Inverse-vol arm: (target_vol_pct/100) * equity / annualized_vol.
              Only considered when signal.annualized_vol is a positive float.
              Target vol defaults to max_position_pct if not supplied.

        Justification (research.md sections 1-2):
          * Half-Kelly captures ~75% of full-Kelly growth at ~50% variance
            (CFA Institute 2018).
          * Inverse-vol sizing decouples size from edge estimate quality
            (Alvarez Quant Trading).
          * Hard percent cap is the regulatory/15c3-5-style fatal bound
            (pre-trade credit-threshold equivalent for a paper trader).

        Pure, never raises, never mutates inputs. Returns 0.0 on any degraded
        path (non-dict, non-BUY, zero equity, all arms skipped).

        Args:
            signal: dict matching validate_signal's schema. Optional extra
                keys: size_usd (hard override, bypasses formula), confidence
                (float in [0,1]), annualized_vol (positive float), mu_hat
                and var_hat (reserved for future full-Kelly arm).
            portfolio: dict matching get_portfolio's shape. Uses total_value
                if present, else cash.

        Returns:
            float USD amount. 0.0 means "do not trade".
        """
        if not isinstance(signal, dict) or not isinstance(portfolio, dict):
            return 0.0

        action = str(signal.get("signal", "") or "").upper()
        if action != "BUY":
            return 0.0

        # Explicit override bypasses the formula -- preserves the 4.1 contract
        # where callers can pass an already-sized signal through publish_signal.
        try:
            explicit = signal.get("size_usd", None)
            if explicit is not None:
                explicit_val = float(explicit)
                if explicit_val > 0.0:
                    return explicit_val
        except (ValueError, TypeError):
            pass

        try:
            equity = float(portfolio.get("total_value", 0.0) or 0.0)
        except (ValueError, TypeError):
            equity = 0.0
        if equity <= 0.0:
            try:
                equity = float(portfolio.get("cash", 0.0) or 0.0)
            except (ValueError, TypeError):
                equity = 0.0
        if equity <= 0.0:
            return 0.0

        limits = self.get_risk_constraints()
        try:
            max_pos_pct = float(limits.get("max_position_pct", 5.0))
        except (ValueError, TypeError):
            max_pos_pct = 5.0
        try:
            max_pos_usd = float(limits.get("max_position_usd", 1000.0))
        except (ValueError, TypeError):
            max_pos_usd = 1000.0

        # (a) Hard percent cap -- always included in the min().
        hard_cap = min(equity * (max_pos_pct / 100.0), max_pos_usd)
        candidates: List[float] = [hard_cap]

        # (b) Confidence-weighted half-Kelly arm -- include only when
        # confidence is a valid numeric in [0, 1]. We use the degraded-edge
        # form 0.5 * confidence * equity since real mu_hat/var_hat are not
        # plumbed yet (Phase 3.2 follow-up, per contract).
        confidence_val: Optional[float] = None
        try:
            raw_conf = signal.get("confidence", None)
            if raw_conf is not None:
                confidence_val = float(raw_conf)
                if confidence_val < 0.0 or confidence_val > 1.0:
                    confidence_val = None
        except (ValueError, TypeError):
            confidence_val = None
        if confidence_val is not None:
            kelly_arm = 0.5 * confidence_val * equity
            if kelly_arm > 0.0:
                candidates.append(kelly_arm)

        # (c) Inverse-vol arm -- only when annualized_vol is a positive
        # float. Target-vol defaults to max_position_pct (conservative).
        try:
            ann_vol_raw = signal.get("annualized_vol", None)
            ann_vol = float(ann_vol_raw) if ann_vol_raw is not None else 0.0
        except (ValueError, TypeError):
            ann_vol = 0.0
        if ann_vol > 0.0:
            try:
                target_vol_pct = float(signal.get("target_vol_pct", max_pos_pct))
            except (ValueError, TypeError):
                target_vol_pct = max_pos_pct
            vol_arm = (target_vol_pct / 100.0) * equity / ann_vol
            if vol_arm > 0.0:
                candidates.append(vol_arm)

        sized = min(candidates) if candidates else 0.0
        if sized < 0.0:
            sized = 0.0
        return float(sized)

    def check_stop_loss(self, portfolio: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect positions at the per-position fixed or trailing stop.

        Post-trade SOFT check (FINRA 15c3-5 hierarchy -- stops generate
        liquidating orders, they are NOT pre-trade fatal blocks). Returns
        the list of positions the caller should exit; does NOT emit trades
        itself (caller wiring is a Phase 4.3 follow-up -- see contract).

        Triggers, per research.md section 2:
          * fixed_stop: (current_price - entry_price) / entry_price <= -stop_loss_pct/100
                        Default 8% -- O'Neil / CAN SLIM canonical.
          * trailing_stop: (current_price - peak_price) / peak_price <= -trail_stop_pct/100
                        Default 3% -- Chandelier-lite.

        Pure function over the portfolio snapshot; never mutates input;
        never raises. Malformed position records are skipped (not raised).

        Args:
            portfolio: dict matching get_portfolio's shape. Each position in
                portfolio["positions"][ticker] must have entry_price (or
                price) and current_price (or mark_price) for the checks to
                fire. peak_price is optional -- defaults to max(entry,
                current) if missing.

        Returns:
            List of dicts, one per triggered stop. Each has keys: ticker,
            reason ("fixed_stop" | "trailing_stop"), entry_price,
            current_price, peak_price, loss_pct.
            Empty list means "no action".
        """
        if not isinstance(portfolio, dict):
            return []

        positions = portfolio.get("positions", {})
        if not isinstance(positions, dict) or not positions:
            return []

        limits = self.get_risk_constraints()
        try:
            stop_pct = float(limits.get("stop_loss_pct", 8.0))
        except (ValueError, TypeError):
            stop_pct = 8.0
        try:
            trail_pct = float(limits.get("trail_stop_pct", 3.0))
        except (ValueError, TypeError):
            trail_pct = 3.0

        # Convert to negative fractional bounds once (e.g. -0.08, -0.03).
        fixed_floor = -(stop_pct / 100.0)
        trail_floor = -(trail_pct / 100.0)

        triggered: List[Dict[str, Any]] = []
        for ticker, pos in positions.items():
            if not isinstance(ticker, str) or not ticker:
                continue
            if not isinstance(pos, dict):
                continue
            try:
                entry = float(pos.get("entry_price", pos.get("price", 0.0)) or 0.0)
            except (ValueError, TypeError):
                entry = 0.0
            try:
                current = float(pos.get("current_price", pos.get("mark_price", 0.0)) or 0.0)
            except (ValueError, TypeError):
                current = 0.0
            if entry <= 0.0 or current <= 0.0:
                # Missing price data -- can't evaluate, skip silently.
                continue
            try:
                peak = float(pos.get("peak_price", 0.0) or 0.0)
            except (ValueError, TypeError):
                peak = 0.0
            if peak <= 0.0:
                peak = max(entry, current)

            # Fixed stop: loss vs entry.
            fixed_loss = (current - entry) / entry
            if fixed_loss <= fixed_floor:
                triggered.append({
                    "ticker": ticker,
                    "reason": "fixed_stop",
                    "entry_price": entry,
                    "current_price": current,
                    "peak_price": peak,
                    "loss_pct": fixed_loss * 100.0,
                })
                continue  # one reason per position

            # Trailing stop: loss vs running peak (only relevant when the
            # position has run up past entry).
            if peak > entry and peak > 0.0:
                trail_loss = (current - peak) / peak
                if trail_loss <= trail_floor:
                    triggered.append({
                        "ticker": ticker,
                        "reason": "trailing_stop",
                        "entry_price": entry,
                        "current_price": current,
                        "peak_price": peak,
                        "loss_pct": trail_loss * 100.0,
                    })

        return triggered

    def track_drawdown(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Update the trailing drawdown state and return the current tier.

        The canonical equity-curve drawdown, mirroring QuantConnect's
        MaximumDrawdownPercentPortfolio model (research.md section 3):

            peak_t      = max(peak_{t-1}, equity_t)
            drawdown_t  = (equity_t - peak_t) / peak_t       (always <= 0)
            tier        = ok | warning | derisk | kill

        Tier thresholds read from get_risk_constraints(). Industry-standard
        5/10/15 ladder: -5% log-only warning, -10% halve sizes, -15% full
        kill switch (liquidate + manual reset required).

        Stateful: mutates self._peak_equity. Lazily initialised on first
        call. In-memory only; durable cross-restart persistence is Phase 4.2
        territory.

        Args:
            portfolio: dict matching get_portfolio's shape. Uses total_value.

        Returns:
            {
                "peak":         float,
                "equity":       float,
                "drawdown_pct": float,   # signed, 0 or negative
                "tier":         str,     # "ok" | "warning" | "derisk" | "kill"
                "kill_switch":  bool,    # True iff tier == "kill"
            }
        """
        if not isinstance(portfolio, dict):
            return {
                "peak": 0.0,
                "equity": 0.0,
                "drawdown_pct": 0.0,
                "tier": "ok",
                "kill_switch": False,
            }

        try:
            equity = float(portfolio.get("total_value", 0.0) or 0.0)
        except (ValueError, TypeError):
            equity = 0.0

        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity
        peak = self._peak_equity or 0.0

        if peak > 0.0:
            drawdown_pct = ((equity - peak) / peak) * 100.0
        else:
            drawdown_pct = 0.0

        limits = self.get_risk_constraints()
        try:
            warn_pct = float(limits.get("drawdown_warning_pct", -5.0))
        except (ValueError, TypeError):
            warn_pct = -5.0
        try:
            derisk_pct = float(limits.get("drawdown_derisk_pct", -10.0))
        except (ValueError, TypeError):
            derisk_pct = -10.0
        try:
            kill_pct = float(limits.get("max_drawdown_pct", -15.0))
        except (ValueError, TypeError):
            kill_pct = -15.0

        # Canonical tier cascade: check most severe first.
        if drawdown_pct <= kill_pct:
            tier = "kill"
        elif drawdown_pct <= derisk_pct:
            tier = "derisk"
        elif drawdown_pct <= warn_pct:
            tier = "warning"
        else:
            tier = "ok"

        return {
            "peak": float(peak),
            "equity": float(equity),
            "drawdown_pct": float(drawdown_pct),
            "tier": tier,
            "kill_switch": tier == "kill",
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
        """Get risk limits.

        Phase 4.3 extension: adds 6 new keys for position sizing, stop-loss,
        and trailing-drawdown tiers. Existing keys are preserved unchanged so
        risk_check and any prior callers keep working.
        """
        logger.info("get_risk_constraints()")
        return {
            # Phase 3.0 risk_check predicates (unchanged)
            "max_exposure_per_ticker_pct": 10.0,
            "max_total_exposure_pct": 100.0,
            "max_drawdown_pct": -15.0,
            "min_sharpe": 0.9,
            "max_daily_trades": 5,
            # Phase 4.3 position sizing + stop-loss + drawdown ladder
            "max_position_pct": 5.0,           # hard cap arm: equity * 5%
            "max_position_usd": 1000.0,        # absolute $ cap on a single trade
            "stop_loss_pct": 8.0,              # O'Neil 7-8% canonical per-position stop
            "trail_stop_pct": 3.0,             # Chandelier-lite trailing stop (peak - 3%)
            "drawdown_warning_pct": -5.0,      # log-only warning tier
            "drawdown_derisk_pct": -10.0,      # halve new sizes tier (kill at -15%)
        }
    
    @staticmethod
    def _parse_iso_date(s: Any) -> Optional[date]:
        """Parse an ISO-8601 calendar date, tolerating unpadded month/day.

        Closes SN4: lex compare of mixed padded/unpadded ISO date strings
        diverges from chronological order. Accepts canonical "YYYY-MM-DD"
        via ``date.fromisoformat``; also tolerates unpadded "YYYY-M-D" by
        re-padding each component. Returns None on any parse failure or
        non-string input. Never raises.
        """
        if not isinstance(s, str) or not s:
            return None
        try:
            return date.fromisoformat(s)
        except (ValueError, TypeError):
            pass
        try:
            yr, mo, dy = s.split("-")
            return date.fromisoformat(f"{int(yr):04d}-{int(mo):02d}-{int(dy):02d}")
        except (ValueError, TypeError):
            return None

    def get_signal_history(
        self,
        limit: Optional[int] = None,
        since_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the in-memory signal history with optional filters.

        Phase 4.2.2: replaces the prior stub. Return shape is additively
        compatible with the stub's `{month, count, signals}` keys -- existing
        callers continue to work; new callers can pass `limit` and/or
        `since_date` for tail slices or date-range filters.

        Args:
            limit: Optional tail slice. If set, return only the last N
                signals (most-recent-N, in insertion order). None = all.
            since_date: Optional ISO date string "YYYY-MM-DD". Only signals
                with `date >= since_date` are returned. Non-string or
                invalid-date values degrade to no filter (never raises).

        Returns:
            {
                "month": "YYYY-MM" (the current month),
                "count": int (len of filtered signals),
                "signals": list[dict] (the filtered records),
                "total_count": int (len of the full history),
            }
        """
        logger.info(f"get_signal_history(limit={limit}, since_date={since_date})")

        signals = list(self.signal_history)  # shallow copy, decouples view

        # since_date filter -- parse both sides to datetime.date to avoid
        # the SN4 lexicographic-compare trap (mixed padded/unpadded ISO
        # strings diverge from chronological order). Tolerate non-string,
        # invalid date, missing date. Never raise from a read API.
        since_dt = self._parse_iso_date(since_date)
        if since_dt is not None:
            filtered: List[Dict[str, Any]] = []
            for sig in signals:
                sdate = sig.get("date", "") if isinstance(sig, dict) else ""
                sig_dt = self._parse_iso_date(sdate)
                if sig_dt is not None and sig_dt >= since_dt:
                    filtered.append(sig)
            signals = filtered

        # Tail limit.
        if isinstance(limit, int) and limit > 0:
            signals = signals[-limit:]

        now = datetime.now(timezone.utc)
        month = f"{now.year:04d}-{now.month:02d}"
        return {
            "month": month,
            "count": len(signals),
            "signals": signals,
            "total_count": len(self.signal_history),
        }

    def track_signal_accuracy(
        self,
        signal_id: str,
        exit_price: Any,
        exit_date: Optional[str] = None,
        neutral_band_pct: float = 0.20,
    ) -> Dict[str, Any]:
        """Record the outcome of a previously-published signal.

        Phase 4.2.2 accuracy tracker. Idempotent in-place update: calling
        twice with the same signal_id overwrites the prior outcome and
        never creates a duplicate history entry.

        Hit/miss semantics (research.md D5, contract D12):
          * BUY hit:  forward_return_pct > +neutral_band_pct
          * BUY miss: forward_return_pct < -neutral_band_pct
          * BUY neutral: |forward_return_pct| <= neutral_band_pct
          * SELL hit: forward_return_pct < -neutral_band_pct
          * SELL miss: forward_return_pct > +neutral_band_pct
          * SELL neutral: |forward_return_pct| <= neutral_band_pct
          * HOLD: always scored=False, outcome='unscored'

        Args:
            signal_id: sha1-16 prefix from SignalsServer._signal_id().
            exit_price: numeric exit price (coerced via float()).
            exit_date: optional ISO "YYYY-MM-DD"; used to compute
                holding_days. None or invalid -> holding_days=None.
            neutral_band_pct: dead-band percent for neutral classification.
                Default 0.20% (below typical round-trip transaction cost).

        Returns:
            {
                "ok": bool,
                "reason": str,
                "updated": bool,
                "signal_id": str,
                "outcome": "hit"|"miss"|"neutral"|"unscored"|"error",
                "scored": bool,
                "hit": bool|None,
                "forward_return_pct": float|None,
                "holding_days": int|None,
            }
        """
        logger.info(f"track_signal_accuracy(signal_id={signal_id})")

        # ---- Guard 1: non-string / empty signal_id ----------------------
        if not isinstance(signal_id, str) or not signal_id:
            return {
                "ok": False,
                "reason": "invalid_signal_id",
                "updated": False,
                "signal_id": str(signal_id) if signal_id is not None else "",
                "outcome": "error",
                "scored": False,
                "hit": None,
                "forward_return_pct": None,
                "holding_days": None,
            }

        # ---- Guard 2: signal not in history -----------------------------
        record = self._signals_by_id.get(signal_id)
        if record is None:
            return {
                "ok": False,
                "reason": "signal_not_found",
                "updated": False,
                "signal_id": signal_id,
                "outcome": "error",
                "scored": False,
                "hit": None,
                "forward_return_pct": None,
                "holding_days": None,
            }

        # Detect idempotent update (already scored).
        already_scored = record.get("scored", False) is True or record.get("outcome") in ("hit", "miss", "neutral", "unscored")

        # ---- Guard 3: coerce exit_price --------------------------------
        try:
            exit_price_f = float(exit_price)
        except (ValueError, TypeError):
            exit_price_f = 0.0

        entry_price = 0.0
        try:
            entry_price = float(record.get("entry_price", 0.0) or 0.0)
        except (ValueError, TypeError):
            entry_price = 0.0

        # ---- HOLD short-circuit: unscored ------------------------------
        signal_type = str(record.get("signal_type", "")).upper()
        if signal_type == "HOLD":
            record["outcome"] = "unscored"
            record["scored"] = False
            record["hit"] = None
            record["exit_price"] = exit_price_f if exit_price_f > 0.0 else None
            record["exit_date"] = exit_date if isinstance(exit_date, str) else None
            record["forward_return_pct"] = 0.0
            record["holding_days"] = self._compute_holding_days(record.get("date", ""), exit_date)
            return {
                "ok": True,
                "reason": "hold_unscored",
                "updated": already_scored,
                "signal_id": signal_id,
                "outcome": "unscored",
                "scored": False,
                "hit": None,
                "forward_return_pct": 0.0,
                "holding_days": record["holding_days"],
            }

        # ---- Compute forward return ------------------------------------
        # Degraded path: missing entry_price -> can't compute return.
        if entry_price <= 0.0 or exit_price_f <= 0.0:
            record["outcome"] = "error"
            record["scored"] = False
            record["hit"] = None
            record["exit_price"] = exit_price_f if exit_price_f > 0.0 else None
            record["exit_date"] = exit_date if isinstance(exit_date, str) else None
            record["forward_return_pct"] = None
            return {
                "ok": False,
                "reason": "missing_prices",
                "updated": already_scored,
                "signal_id": signal_id,
                "outcome": "error",
                "scored": False,
                "hit": None,
                "forward_return_pct": None,
                "holding_days": None,
            }

        forward_return_pct = ((exit_price_f - entry_price) / entry_price) * 100.0

        # Classify per D5.
        band = abs(float(neutral_band_pct))
        if abs(forward_return_pct) <= band:
            outcome = "neutral"
            hit: Optional[bool] = None
            scored = False
        elif signal_type == "BUY":
            if forward_return_pct > band:
                outcome = "hit"
                hit = True
            else:
                outcome = "miss"
                hit = False
            scored = True
        elif signal_type == "SELL":
            if forward_return_pct < -band:
                outcome = "hit"
                hit = True
            else:
                outcome = "miss"
                hit = False
            scored = True
        else:
            outcome = "unscored"
            hit = None
            scored = False

        record["outcome"] = outcome
        record["scored"] = scored
        record["hit"] = hit
        record["exit_price"] = exit_price_f
        record["exit_date"] = exit_date if isinstance(exit_date, str) else None
        record["forward_return_pct"] = forward_return_pct
        record["holding_days"] = self._compute_holding_days(record.get("date", ""), exit_date)

        return {
            "ok": True,
            "reason": "recorded",
            "updated": already_scored,
            "signal_id": signal_id,
            "outcome": outcome,
            "scored": scored,
            "hit": hit,
            "forward_return_pct": forward_return_pct,
            "holding_days": record["holding_days"],
        }

    @staticmethod
    def _compute_holding_days(entry_date: Any, exit_date: Any) -> Optional[int]:
        """Pure helper: parse two ISO "YYYY-MM-DD" strings and return the day
        delta. Returns None on any parse failure. Never raises."""
        if not isinstance(entry_date, str) or not isinstance(exit_date, str):
            return None
        try:
            d1 = datetime.strptime(entry_date, "%Y-%m-%d")
            d2 = datetime.strptime(exit_date, "%Y-%m-%d")
            return (d2 - d1).days
        except (ValueError, TypeError):
            return None

    def get_accuracy_report(
        self,
        group_by: Optional[str] = None,
        neutral_band_pct: float = 0.20,
    ) -> Dict[str, Any]:
        """Aggregate signal accuracy statistics over the in-memory history.

        Phase 4.2.2 aggregator. Pure read -- does NOT mutate state or
        re-classify existing records (they are classified at track time,
        against the band used then). The `neutral_band_pct` arg is kept in
        the signature for API parity with track_signal_accuracy but does
        NOT re-score records.

        Args:
            group_by: Optional "signal_type" or "ticker" for sub-aggregation.
                None -> single aggregate only. Unknown values ignored.
            neutral_band_pct: forwarded for API parity; unused in aggregation.

        Returns:
            {
                "total_count": int,         # all signals in history
                "scored_count": int,        # hit + miss only
                "hits": int,
                "misses": int,
                "neutral": int,
                "unscored": int,            # HOLD + error + pending
                "hit_rate": float,          # hits / scored_count (0.0 if 0)
                "hit_rate_ci_low": float,   # Wilson 95% lower
                "hit_rate_ci_high": float,  # Wilson 95% upper
                "mean_forward_return_pct": float,
                "median_forward_return_pct": float,
                "groups": dict[str, dict],  # sub-aggregates if group_by set
            }
        """
        logger.info(f"get_accuracy_report(group_by={group_by})")

        def _aggregate(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
            total = len(signals)
            hits = 0
            misses = 0
            neutral = 0
            unscored = 0
            returns: List[float] = []
            for s in signals:
                outcome = s.get("outcome", "pending")
                if outcome == "hit":
                    hits += 1
                elif outcome == "miss":
                    misses += 1
                elif outcome == "neutral":
                    neutral += 1
                else:
                    unscored += 1
                # Return aggregation: include any record with a numeric
                # forward_return_pct, regardless of scored/unscored. This
                # matches pyfolio's "all-positions" P&L view.
                ret = s.get("forward_return_pct", None)
                if isinstance(ret, (int, float)) and ret is not None:
                    returns.append(float(ret))

            scored = hits + misses
            hit_rate = (hits / scored) if scored > 0 else 0.0
            ci_low, ci_high = self._wilson_ci(hits, scored)
            mean_ret = statistics.mean(returns) if returns else 0.0
            median_ret = statistics.median(returns) if returns else 0.0

            return {
                "total_count": total,
                "scored_count": scored,
                "hits": hits,
                "misses": misses,
                "neutral": neutral,
                "unscored": unscored,
                "hit_rate": float(hit_rate),
                "hit_rate_ci_low": float(ci_low),
                "hit_rate_ci_high": float(ci_high),
                "mean_forward_return_pct": float(mean_ret),
                "median_forward_return_pct": float(median_ret),
            }

        overall = _aggregate(self.signal_history)
        overall["groups"] = {}

        if group_by in ("signal_type", "ticker"):
            buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for s in self.signal_history:
                key = str(s.get(group_by, "") or "unknown")
                buckets[key].append(s)
            for key, bucket_signals in buckets.items():
                overall["groups"][key] = _aggregate(bucket_signals)

        return overall

    @staticmethod
    def _wilson_ci(hits: int, n: int, z: float = 1.96) -> Tuple[float, float]:
        """Wilson Score Interval (two-sided) for a binomial proportion.

        Pure stdlib (math.sqrt). Safe for small n and extreme proportions
        where the Wald interval gives negative or >1 bounds. Default z=1.96
        is the 95% critical value.

        Handles edge cases:
          * n=0 -> (0.0, 0.0)   (no data, collapse to zero)
          * n=1, hits=0 -> (0.0, upper>0)
          * n=1, hits=1 -> (lower<1, 1.0)

        Formula (Wilson 1927):
            center = p_hat + z^2/(2n)
            half   = z * sqrt( (p_hat*(1-p_hat) + z^2/(4n)) / n )
            denom  = 1 + z^2/n
            (low, high) = ((center - half)/denom, (center + half)/denom)

        Args:
            hits: number of successes (non-negative int).
            n: total trials (non-negative int).
            z: two-sided critical value. Default 1.96 (95%).

        Returns:
            (ci_low, ci_high) clamped to [0.0, 1.0].
        """
        try:
            n_int = int(n)
            hits_int = int(hits)
        except (ValueError, TypeError):
            return (0.0, 0.0)
        if n_int <= 0 or hits_int < 0:
            return (0.0, 0.0)
        if hits_int > n_int:
            hits_int = n_int

        p_hat = hits_int / n_int
        z2 = z * z
        denom = 1.0 + z2 / n_int
        center = p_hat + z2 / (2.0 * n_int)
        radicand = (p_hat * (1.0 - p_hat) + z2 / (4.0 * n_int)) / n_int
        if radicand < 0.0:
            radicand = 0.0
        half = z * math.sqrt(radicand)
        low = (center - half) / denom
        high = (center + half) / denom
        # Clamp.
        if low < 0.0:
            low = 0.0
        if high > 1.0:
            high = 1.0
        return (float(low), float(high))


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
