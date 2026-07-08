"""
Paper Trading Engine — executes virtual trades, tracks positions, computes NAV.

All persistence through BigQueryClient. No real money involved.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import yfinance as yf

from backend.config.settings import Settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.execution_router import ExecutionRouter


def _parse_iso_date(s: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; return None if unparseable."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

logger = logging.getLogger(__name__)


def _fx_local_to_usd(market: Optional[str], date: Optional[str] = None) -> Optional[float]:
    """phase-50.2: FX rate to convert 1 unit of the market's local currency into USD.
    Returns 1.0 for US/USD (keeps the USD-only path byte-identical). Returns None when
    a non-USD rate is genuinely unavailable -- callers fail-soft to last-known, NEVER
    silently to 1.0 (that would mis-value a non-USD position as if it were USD)."""
    from backend.services import fx_rates
    ccy = fx_rates.market_currency(market or "US")
    if ccy == "USD":
        return 1.0
    return fx_rates.get_fx_rate(ccy, "USD", date)


def fx_pnl_attribution(qty: float, entry_price_local: float, current_price_local: float,
                       entry_fx: float, current_fx: float) -> tuple[float, float]:
    """phase-50.2: decompose a position's USD P&L into local-return + FX-return
    (arXiv 1611.01463 / CFA / CFI). entry_fx/current_fx = USD per 1 unit of the
    local currency (1.0 for USD). By construction, with no residual:
      local_pnl + fx_pnl == qty*(Pc*Fc - Pe*Fe) == market_value_usd - cost_usd.
    For a USD position (Fe==Fc==1.0): local_pnl = qty*(Pc-Pe), fx_pnl = 0.0."""
    local_pnl = qty * (current_price_local - entry_price_local) * entry_fx
    fx_pnl = qty * current_price_local * (current_fx - entry_fx)
    return round(local_pnl, 2), round(fx_pnl, 2)


def _fx_usd_to_local(market: Optional[str], date: Optional[str] = None) -> Optional[float]:
    """phase-50.2: FX rate to convert 1 USD into the market's local currency (to size a
    non-USD buy from a USD budget). 1.0 for US/USD."""
    from backend.services import fx_rates
    ccy = fx_rates.market_currency(market or "US")
    if ccy == "USD":
        return 1.0
    return fx_rates.get_fx_rate("USD", ccy, date)


class PaperTrader:
    """Virtual trade execution engine backed by BigQuery."""

    def __init__(self, settings: Settings, bq_client: BigQueryClient,
                 trade_notifier: "Optional[Callable[[dict], None]]" = None):
        self.settings = settings
        self.bq = bq_client
        # phase-25.J: optional hook invoked after every successful trade.
        # Receives the persisted trade dict. Used by Slack/operator-alert
        # wiring. None by default (no behavior change for callers that
        # don't need notification). Failures inside the hook are caught
        # and logged so they never break trade execution.
        self.trade_notifier = trade_notifier

    def _maybe_notify_trade(self, trade: dict) -> None:
        """phase-25.J: best-effort dispatch to the optional trade_notifier."""
        if self.trade_notifier is None:
            return
        try:
            self.trade_notifier(trade)
        except Exception as e:
            logger.exception("phase-25.J: trade_notifier hook failed (non-fatal): %s", e)

    # ── Portfolio State ──────────────────────────────────────────

    def get_or_create_portfolio(self) -> dict:
        """Load portfolio from BQ, or initialize with starting capital."""
        portfolio = self.bq.get_paper_portfolio("default")
        if portfolio:
            return portfolio
        now = datetime.now(timezone.utc).isoformat()
        row = {
            "portfolio_id": "default",
            "starting_capital": self.settings.paper_starting_capital,
            "current_cash": self.settings.paper_starting_capital,
            "total_nav": self.settings.paper_starting_capital,
            "total_pnl_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "inception_date": now,
            "updated_at": now,
        }
        self.bq.upsert_paper_portfolio(row)
        logger.info(f"Initialized paper portfolio with ${self.settings.paper_starting_capital:,.0f}")
        return row

    def get_positions(self) -> list[dict]:
        return self.bq.get_paper_positions()

    def get_position(self, ticker: str) -> Optional[dict]:
        return self.bq.get_paper_position(ticker)

    # ── Trade Execution ──────────────────────────────────────────

    def execute_buy(
        self,
        ticker: str,
        amount_usd: float,
        price: float,
        reason: str = "new_buy_signal",
        analysis_id: str = "",
        risk_judge_decision: str = "",
        stop_loss_price: Optional[float] = None,
        risk_judge_position_pct: Optional[float] = None,
        signals: Optional[list[dict]] = None,
        sector: Optional[str] = None,  # phase-23.2.6-fix: persist GICS sector
        market: str = "US",  # phase-50.2: market code (US/EU/KR/...) -> local currency via fx_rates
        # phase-30.6: analysis-time price reference for the pre-trade
        # price-tolerance gate. None disables the gate (fail-open) so the
        # lite-Claude path (which can lack a written price) still trades.
        price_at_analysis: Optional[float] = None,
        # phase-40.8.1 (P3): FF3 factor loadings carried IN-MEMORY only.
        # BQ persistence deferred to phase-40.8.2 (Step 7 schema window).
        factor_loadings: Optional[dict] = None,
        # phase-61.2 (criterion 5): the ANALYSIS recommendation (BUY/STRONG_BUY).
        # Persisted into paper_positions.recommendation instead of the trade
        # mechanism `reason` when paper_position_recommendation_fix_enabled is
        # ON, so the signal_downgrade rule (portfolio_manager.py:127) can match.
        analysis_recommendation: str = "",
    ) -> Optional[dict]:
        """Buy shares of a ticker. Returns the trade record or None if can't execute."""
        # phase-25.6: no-stop-on-entry HARD BLOCK. If stop_loss_price is None
        # at entry, synthesize one from settings.paper_default_stop_loss_pct
        # (8% default per O'Neil canonical + arxiv 2604.27150) so every new
        # position has a stop in BQ. Closes phase-24.1 audit finding F-4
        # (_extract_stop_loss fallback only applied to NEW buys; positions
        # bought before phase-23.1.8 had stop_loss_price=None forever).
        # Defense-in-depth alongside 25.2 backfill: prevents regression
        # if an upstream resolution chain (risk_judge -> portfolio_manager
        # -> execute_buy) ever passes None again.
        if stop_loss_price is None:
            default_pct = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))
            if price > 0:
                stop_loss_price = round(price * (1.0 - default_pct / 100.0), 4)
                logger.warning(
                    "phase-25.6: no stop_loss_price provided for %s; defaulting to %.4f (%.1f%% below entry %.4f)",
                    ticker, stop_loss_price, default_pct, price,
                )

        # phase-30.6: price-tolerance pre-trade gate (FIA WP July 2024 Sec 1.3
        # canonical pre-trade control). Reject when the live fill price
        # diverges from the analysis-time price by more than
        # `paper_price_tolerance_pct` (default 5 per SEC LULD Tier 1 band).
        # Fail-open when price_at_analysis is None (lite-Claude path can lack
        # it). Placed BEFORE the ExecutionRouter call so the non-bypassable-
        # invariants pattern from arXiv 2603.10092 holds (gate cannot be
        # circumvented by routing). Closes phase-30.0 cross-val 6.1 / P2-4.
        price_tolerance_pct = float(
            getattr(self.settings, "paper_price_tolerance_pct", 0.0) or 0.0
        )
        if (
            price_tolerance_pct > 0
            and price_at_analysis is not None
            and price_at_analysis > 0
            and price > 0
        ):
            divergence_pct = abs(price - price_at_analysis) / price_at_analysis * 100.0
            if divergence_pct > price_tolerance_pct:
                logger.warning(
                    "phase-30.6: rejecting BUY %s -- live fill price $%.4f diverges %.2f%% from "
                    "analysis-time price $%.4f (tolerance %.2f%%). Likely stale analysis or news-driven move.",
                    ticker, price, divergence_pct, price_at_analysis, price_tolerance_pct,
                )
                return None

        portfolio = self.get_or_create_portfolio()
        cash = portfolio["current_cash"]

        # Transaction cost
        tx_cost = amount_usd * (self.settings.paper_transaction_cost_pct / 100.0)
        total_cost = amount_usd + tx_cost

        if total_cost > cash:
            logger.warning(f"Insufficient cash for {ticker}: need ${total_cost:.2f}, have ${cash:.2f}")
            return None

        # Check max positions
        positions = self.get_positions()
        existing = next((p for p in positions if p["ticker"] == ticker), None)
        if not existing and len(positions) >= self.settings.paper_max_positions:
            logger.warning(f"Max positions ({self.settings.paper_max_positions}) reached, skipping {ticker}")
            return None

        # phase-50.2: size shares from the USD budget converted to the stock's
        # local currency. `price` is in the local currency. x1.0 for US/USD ->
        # quantity == amount_usd/price (byte-identical). FX-unavailable for a
        # non-USD market -> skip the buy (never silently treat as USD).
        _usd_to_local = _fx_usd_to_local(market)
        _local_to_usd = _fx_local_to_usd(market)
        if _usd_to_local is None or _local_to_usd is None:
            logger.warning("phase-50.2: FX unavailable for market=%s; skipping BUY %s", market, ticker)
            return None
        quantity = (amount_usd * _usd_to_local) / price

        # phase-23.1.15: idempotency guard. Crash-and-retry between
        # autonomous-loop cycles can produce a phantom double-buy: cycle 1
        # books the trade + debits cash but errors before the position write
        # lands visibly; cycle 2 sees no existing position and books again.
        # Guard: when no `existing` position is in our snapshot, look back
        # 30 minutes in paper_trades for a matching BUY at near-identical
        # quantity (1% tolerance for rounding). If found, skip.
        if not existing:
            try:
                cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
                recent_trades = self.bq.get_paper_trades_for_ticker_since(
                    ticker, cutoff, action="BUY",
                )
                if recent_trades:
                    recent_qty = float(recent_trades[0].get("quantity") or 0)
                    if recent_qty > 0:
                        delta = abs(recent_qty - quantity) / max(recent_qty, quantity)
                        if delta < 0.01:
                            logger.warning(
                                "Idempotency guard: skipping duplicate BUY for %s "
                                "(recent trade %s at %s qty=%.4f vs proposed %.4f)",
                                ticker, recent_trades[0].get("trade_id"),
                                recent_trades[0].get("created_at"),
                                recent_qty, quantity,
                            )
                            return None
            except Exception as e:
                logger.warning("Idempotency guard query failed (non-fatal): %s", e)
        now = datetime.now(timezone.utc).isoformat()

        # phase-17.5: route every buy through ExecutionRouter so the
        # bq_sim / alpaca_paper / shadow mode switch is honored. In
        # bq_sim mode the router returns the same fill_price we passed
        # in (behavior-preserving). In alpaca_paper the returned
        # fill_price is Alpaca's actual fill + `source` tracks the path.
        trade_id = str(uuid.uuid4())
        router = ExecutionRouter()
        fill = router.submit_order(
            symbol=ticker, qty=quantity, side="buy",
            client_order_id=trade_id, close_price=price,
        )
        exec_price = float(fill.fill_price) if fill and fill.fill_price else price
        exec_source = fill.source if fill else "bq_sim"

        # Record trade
        trade = {
            "trade_id": trade_id,
            "ticker": ticker,
            "action": "BUY",
            "quantity": round(quantity, 6),
            "price": exec_price,
            # phase-56.1 (55.1 F-2): persist USD, not local notional (x1.0 for US).
            "total_value": round(quantity * exec_price * _local_to_usd, 2),
            "transaction_cost": round(tx_cost, 2),
            "reason": reason,
            "analysis_id": analysis_id,
            "risk_judge_decision": risk_judge_decision,
            "created_at": now,
            # 4.5.5: serialized agent-signal attribution; see signal_attribution.py.
            # Stored as JSON text (not ARRAY<STRUCT>) to keep the dynamic INSERT
            # path parameterizable.
            "signals": json.dumps(signals or []),
        }
        self._safe_save_trade(trade)
        # phase-40.8.1 (P3): attach FF3 loadings AFTER BQ save so the in-memory
        # caller sees them but the dynamic INSERT path (which would reject an
        # unknown column) is not impacted. BQ persistence is phase-40.8.2
        # (column add inside Step 7 schema window).
        if factor_loadings is not None:
            trade["factor_loadings"] = factor_loadings

        # phase-61.2 (criterion 5): choose what paper_positions.recommendation
        # stores. Legacy = the trade mechanism (`reason`), which never matches
        # _BUY_RECS, leaving signal_downgrade structurally dead. Flag ON + a
        # non-empty analysis verdict = store the verdict. Flag OFF or empty
        # verdict = byte-identical legacy.
        _pos_rec = reason
        if (
            getattr(self.settings, "paper_position_recommendation_fix_enabled", False)
            and analysis_recommendation
        ):
            _pos_rec = analysis_recommendation

        # Update or create position
        if existing:
            old_qty = existing["quantity"]
            old_cost = existing["cost_basis"] or (old_qty * existing["avg_entry_price"])
            new_qty = old_qty + quantity
            new_cost = old_cost + amount_usd
            new_avg = new_cost / new_qty
            self.bq.delete_paper_position(ticker)
            pos_row = {
                "position_id": existing["position_id"],
                "ticker": ticker,
                "quantity": round(new_qty, 6),
                "avg_entry_price": round(new_avg, 4),
                "cost_basis": round(new_cost, 2),
                "current_price": price,  # phase-50.2: LOCAL price; market_value below is USD
                "market_value": round(new_qty * price * _local_to_usd, 2),
                "unrealized_pnl": round(new_qty * price * _local_to_usd - new_cost, 2),
                "unrealized_pnl_pct": round(((new_qty * price * _local_to_usd - new_cost) / new_cost) * 100, 2),
                "entry_date": existing["entry_date"],
                "last_analysis_date": now,
                "recommendation": _pos_rec,  # phase-61.2 (criterion 5)
                "risk_judge_position_pct": risk_judge_position_pct,
                "stop_loss_price": stop_loss_price,
                # phase-23.2.6-fix: prefer the new sector arg; preserve existing
                # if the new BUY didn't carry one (None-drop in save_paper_position
                # leaves the existing column untouched via MERGE).
                "sector": sector or (existing.get("sector") or None),
                "market": market,  # phase-50.2
                "base_currency": "USD",  # phase-50.2: NAV/cost_basis are in USD
            }
            self.bq.save_paper_position(pos_row)
        else:
            pos_row = {
                "position_id": str(uuid.uuid4()),
                "ticker": ticker,
                "quantity": round(quantity, 6),
                "avg_entry_price": price,
                "cost_basis": round(amount_usd, 2),
                "current_price": price,
                "market_value": round(amount_usd, 2),
                "unrealized_pnl": 0.0,
                "unrealized_pnl_pct": 0.0,
                "entry_date": now,
                "last_analysis_date": now,
                "recommendation": _pos_rec,  # phase-61.2 (criterion 5)
                "risk_judge_position_pct": risk_judge_position_pct,
                "stop_loss_price": stop_loss_price,
                "sector": sector or None,  # phase-23.2.6-fix
                "market": market,  # phase-50.2
                "base_currency": "USD",  # phase-50.2: cost_basis=amount_usd is already USD
            }
            self.bq.save_paper_position(pos_row)

        # Update cash
        new_cash = cash - total_cost
        self._update_portfolio_cash(new_cash)

        logger.info(f"BUY {quantity:.4f} x {ticker} @ ${exec_price:.2f} "
                    f"(source={exec_source}) = ${quantity*exec_price:.2f} (fee: ${tx_cost:.2f})")
        # phase-25.J: fire trade_notifier hook (Slack confirmation when wired)
        self._maybe_notify_trade(trade)
        return trade

    def execute_sell(
        self,
        ticker: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        reason: str = "signal_flip",
        signals: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """Sell shares. If quantity is None, sells entire position. Returns trade record."""
        position = self.get_position(ticker)
        if not position:
            logger.warning(f"No position in {ticker} to sell")
            return None

        if price is None:
            price = _get_live_price(ticker) or position.get("current_price", 0)

        sell_qty = quantity or position["quantity"]
        sell_qty = min(sell_qty, position["quantity"])

        # phase-50.2: local-currency -> USD rate for this position (x1.0 for US/USD
        # -> byte-identical). FX unavailable for a non-USD exit -> last-resort 1.0
        # with a WARN (never block an exit; non-USD only, the live engine is USD).
        _l2u = _fx_local_to_usd(position.get("market"))
        if _l2u is None:
            logger.warning("phase-50.2: FX %s->USD unavailable on SELL %s; crediting at 1.0",
                           position.get("market"), ticker)
            _l2u = 1.0

        # phase-17.5: route sell through ExecutionRouter (mirrors execute_buy).
        trade_id = str(uuid.uuid4())
        router = ExecutionRouter()
        fill = router.submit_order(
            symbol=ticker, qty=sell_qty, side="sell",
            client_order_id=trade_id, close_price=price,
        )
        price = float(fill.fill_price) if fill and fill.fill_price else price
        exec_source = fill.source if fill else "bq_sim"

        sell_value = sell_qty * price
        tx_cost = sell_value * (self.settings.paper_transaction_cost_pct / 100.0)
        net_proceeds = sell_value - tx_cost
        now = datetime.now(timezone.utc).isoformat()

        # Round-trip enrichment (4.5.2): holding_days, realized_pnl_pct, MFE/MAE, capture_ratio.
        entry_price = float(position.get("avg_entry_price") or 0.0)
        realized_pnl_pct = (
            ((price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        )
        entry_dt = _parse_iso_date(position.get("entry_date", ""))
        now_dt = datetime.now(timezone.utc)
        holding_days = int((now_dt - entry_dt).days) if entry_dt else 0
        mfe_pct = float(position.get("mfe_pct") or 0.0)
        mae_pct = float(position.get("mae_pct") or 0.0)
        # Capture ratio = realized / MFE (how much of the max favorable excursion we kept).
        # Undefined when MFE <= 0 (never printed a gain); use 0.0 for that edge.
        capture_ratio = realized_pnl_pct / mfe_pct if mfe_pct > 0 else 0.0
        round_trip_id = position.get("position_id") or str(uuid.uuid4())

        # Record trade
        trade = {
            "trade_id": trade_id,
            "ticker": ticker,
            "action": "SELL",
            "quantity": round(sell_qty, 6),
            "price": price,
            # phase-56.1 (55.1 F-2): persist USD (x1.0 for US). Row-level only --
            # net_proceeds/sell_value stay LOCAL for the cash credit at *_l2u below.
            "total_value": round(sell_value * _l2u, 2),
            "transaction_cost": round(tx_cost * _l2u, 2),
            "reason": reason,
            "analysis_id": "",
            "risk_judge_decision": "",
            "created_at": now,
            "round_trip_id": round_trip_id,
            "holding_days": holding_days,
            "realized_pnl_pct": round(realized_pnl_pct, 4),
            "mfe_pct": round(mfe_pct, 4),
            "mae_pct": round(mae_pct, 4),
            "capture_ratio": round(capture_ratio, 4),
            "signals": json.dumps(signals or []),
        }
        self._safe_save_trade(trade)

        # Persist canonical round-trip row for exit-quality analysis (4.5.9 consumes this).
        rt_row = {
            "round_trip_id": round_trip_id,
            "ticker": ticker,
            "buy_trade_id": position.get("position_id", ""),
            "sell_trade_id": trade["trade_id"],
            "entry_date": position.get("entry_date"),
            "exit_date": now,
            "entry_price": entry_price,
            "exit_price": price,
            "quantity": round(sell_qty, 6),
            "realized_pnl_usd": round((price - entry_price) * sell_qty * _l2u, 2),  # phase-50.2: LOCAL pnl -> USD
            "realized_pnl_pct": round(realized_pnl_pct, 4),
            "holding_days": holding_days,
            "mfe_pct": round(mfe_pct, 4),
            "mae_pct": round(mae_pct, 4),
            "capture_ratio": round(capture_ratio, 4),
            "exit_reason": reason,
        }
        self._safe_save_round_trip(rt_row)

        remaining = position["quantity"] - sell_qty
        if remaining < 0.0001:
            # Full exit
            self.bq.delete_paper_position(ticker)
        else:
            # Partial - delete and re-insert with reduced quantity
            self.bq.delete_paper_position(ticker)
            # phase-50.2: remaining cost_basis = proportional USD cost (byte-identical
            # for USD); market_value is USD (current_price stays LOCAL); x1.0 for US/USD.
            _orig_cb = position.get("cost_basis") or (position["quantity"] * position["avg_entry_price"])
            _rem_cb = _orig_cb * (remaining / position["quantity"]) if position["quantity"] else 0.0
            _rem_mv = remaining * price * _l2u
            pos_row = {
                "position_id": position["position_id"],
                "ticker": ticker,
                "quantity": round(remaining, 6),
                "avg_entry_price": position["avg_entry_price"],
                "cost_basis": round(_rem_cb, 2),
                "current_price": price,
                "market_value": round(_rem_mv, 2),
                "unrealized_pnl": round(_rem_mv - _rem_cb, 2),
                "unrealized_pnl_pct": round(((_rem_mv - _rem_cb) / _rem_cb) * 100, 2) if _rem_cb else 0.0,
                "entry_date": position["entry_date"],
                "last_analysis_date": position.get("last_analysis_date", ""),
                "recommendation": position.get("recommendation", ""),
                "risk_judge_position_pct": position.get("risk_judge_position_pct"),
                "stop_loss_price": position.get("stop_loss_price"),
                "market": position.get("market") or "US",  # phase-50.2
                "base_currency": "USD",  # phase-50.2
            }
            self.bq.save_paper_position(pos_row)

        # Update cash -- phase-50.2: net_proceeds is in the stock's LOCAL currency;
        # convert to the USD base before crediting (x1.0 for US/USD).
        portfolio = self.get_or_create_portfolio()
        new_cash = portfolio["current_cash"] + net_proceeds * _l2u
        self._update_portfolio_cash(new_cash)

        logger.info(f"SELL {sell_qty:.4f} x {ticker} @ ${price:.2f} = ${sell_value:.2f} (fee: ${tx_cost:.2f})")
        # phase-25.J: fire trade_notifier hook (Slack confirmation when wired).
        # Stop-loss-trigger sells (from autonomous_loop Step 5.6 via 25.1) also
        # flow through this notifier since they use the same execute_sell path.
        self._maybe_notify_trade(trade)
        return trade

    # ── Mark-to-Market ───────────────────────────────────────────

    def mark_to_market(self) -> dict:
        """Update all positions with live prices. Returns portfolio summary."""
        positions = self.get_positions()
        portfolio = self.get_or_create_portfolio()
        total_positions_value = 0.0

        for pos in positions:
            ticker = pos["ticker"]
            live_price = _get_live_price(ticker)
            if live_price is None:
                live_price = pos.get("current_price", pos["avg_entry_price"])

            # phase-50.2: live_price is in the position's LOCAL currency; convert
            # market value to the USD base (x1.0 for US/USD -> byte-identical). FX
            # unavailable for a non-USD position -> keep the last-known USD mark.
            _l2u = _fx_local_to_usd(pos.get("market"))
            if _l2u is None:
                logger.warning(
                    "phase-50.2: FX %s->USD unavailable; keeping last-known market_value for %s",
                    pos.get("market"), ticker,
                )
                market_value = float(pos.get("market_value") or (pos["quantity"] * live_price))
            else:
                market_value = pos["quantity"] * live_price * _l2u
            cost_basis = pos.get("cost_basis") or (pos["quantity"] * pos["avg_entry_price"])
            pnl = market_value - cost_basis  # both USD
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0

            # 4.5.2: MFE/MAE tracked monotonically across the position's holding period.
            # MFE = best unrealized_pnl_pct seen; MAE = worst (lowest). Reset only when
            # the position is fully closed (handled by execute_sell).
            prev_mfe = float(pos.get("mfe_pct") or 0.0)
            prev_mae = float(pos.get("mae_pct") or 0.0)
            new_mfe = max(prev_mfe, pnl_pct)
            new_mae = min(prev_mae, pnl_pct)

            new_stop, advance_iso = self._advance_stop(pos, new_mfe)

            self.bq.delete_paper_position(ticker)
            updates: dict = {
                "current_price": live_price,
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(pnl, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "mfe_pct": round(new_mfe, 4),
                "mae_pct": round(new_mae, 4),
            }
            if new_stop is not None:
                updates["stop_loss_price"] = new_stop
                # phase-32.2: only set stop_advanced_at_R when the breakeven
                # branch (one-shot) fired -- advance_iso is None for
                # trailing updates so we do not overwrite the original
                # breakeven timestamp.
                if advance_iso is not None:
                    updates["stop_advanced_at_R"] = advance_iso
            pos.update(updates)
            self._safe_save_position(pos)
            total_positions_value += market_value

        nav = portfolio["current_cash"] + total_positions_value
        starting = portfolio["starting_capital"]
        pnl_pct = ((nav - starting) / starting) * 100 if starting > 0 else 0.0
        # phase-38.7: anchor SPY to first-funded snapshot (where positions_value
        # transitioned from 0 to >0), NOT to inception_date (set at row-creation
        # time before any capital injection). Fall back to inception_date when
        # no funded snapshot exists yet (cold-start grace).
        first_funded = self.bq.get_first_funded_snapshot_date()
        benchmark_ret = _get_benchmark_return(
            portfolio.get("inception_date", ""),
            first_funded_date=first_funded,
        )

        self.bq.upsert_paper_portfolio({
            **portfolio,
            "total_nav": round(nav, 2),
            "total_pnl_pct": round(pnl_pct, 2),
            "benchmark_return_pct": round(benchmark_ret, 2) if benchmark_ret else portfolio.get("benchmark_return_pct", 0.0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "nav": round(nav, 2),
            "cash": portfolio["current_cash"],
            "positions_value": round(total_positions_value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "benchmark_return_pct": round(benchmark_ret, 2) if benchmark_ret else 0.0,
            "position_count": len(positions),
        }

    # ── Stop Loss Check ──────────────────────────────────────────

    def check_stop_losses(self) -> list[str]:
        """Return tickers where current price is at or below stop loss."""
        positions = self.get_positions()
        triggered = []
        for pos in positions:
            stop = pos.get("stop_loss_price")
            current = pos.get("current_price", 0)
            if stop and current and current <= stop:
                triggered.append(pos["ticker"])
        return triggered

    # phase-36.1: scale-out take-profit ladder (50% close at +2R, remainder at +3R).
    # Gated by settings.paper_scale_out_enabled (default False per /goal gate 3).
    # R = paper_default_stop_loss_pct (the initial stop distance, e.g. 8%); 2R = 16%
    # MFE, 3R = 24% MFE. Idempotent via scale_out_levels_hit column on
    # paper_positions (JSON-encoded list of strings; NULL/empty -> not yet fired).
    # Closes phase-31.0 audit P1.3 (the only OPEN code BLOCK on profit-protection
    # per closure_roadmap.md §2 OPEN-2).
    def check_scale_out_fires(self) -> list[dict]:
        """Fire scale-out partial closes for positions whose MFE crossed +2R / +3R.

        Should be called after mark_to_market in autonomous_loop Step 5.5 area
        (before stop-loss enforcement) so the fires use freshly-updated MFE.

        Returns:
            List of fire records, one per partial close executed. Empty list
            when flag is OFF or no position crossed a threshold.
        """
        if not getattr(self.settings, "paper_scale_out_enabled", False):
            return []

        R_pct = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))
        if R_pct <= 0:
            logger.warning("phase-36.1: paper_default_stop_loss_pct=%s is non-positive; skipping scale-out", R_pct)
            return []

        threshold_2r = 2.0 * R_pct
        threshold_3r = 3.0 * R_pct

        fires: list[dict] = []
        for pos in self.get_positions():
            ticker = pos.get("ticker", "?")
            mfe = float(pos.get("mfe_pct") or 0.0)
            quantity = float(pos.get("quantity") or 0.0)
            if quantity <= 0:
                continue

            # Parse existing scale_out_levels_hit (NULL/missing -> [] -- pre-migration positions).
            raw = pos.get("scale_out_levels_hit")
            if raw is None or raw == "":
                hit: set[str] = set()
            elif isinstance(raw, str):
                try:
                    hit = set(json.loads(raw))
                except Exception:
                    hit = set()
            elif isinstance(raw, (list, tuple)):
                hit = set(str(x) for x in raw)
            else:
                hit = set()

            # +2R: 50% close (one-shot per position).
            if mfe >= threshold_2r and "2R" not in hit:
                qty_to_sell = round(quantity * 0.5, 6)
                if qty_to_sell > 0:
                    trade = self.execute_sell(
                        ticker, quantity=qty_to_sell, reason="take_profit_2R",
                    )
                    if trade:
                        hit.add("2R")
                        self._persist_scale_out_levels(ticker, hit)
                        fires.append({
                            "ticker": ticker, "level": "2R",
                            "qty": qty_to_sell, "trade_id": trade.get("trade_id"),
                            "mfe_pct": round(mfe, 4),
                        })
                        logger.info(
                            "phase-36.1: scale-out 2R fired for %s -- sold %s qty at mfe=%.4f%% (threshold %.4f%%)",
                            ticker, qty_to_sell, mfe, threshold_2r,
                        )

            # +3R: close remainder (one-shot per position). Re-fetch latest
            # position state -- the 2R fire above may have reduced quantity.
            if "3R" not in hit and mfe >= threshold_3r:
                latest = self.get_position(ticker)
                if not latest:
                    # Already fully closed by the 2R partial above (rare edge);
                    # nothing to do.
                    continue
                latest_qty = float(latest.get("quantity") or 0.0)
                latest_mfe = float(latest.get("mfe_pct") or 0.0)
                if latest_qty <= 0 or latest_mfe < threshold_3r:
                    continue
                trade = self.execute_sell(
                    ticker, quantity=latest_qty, reason="take_profit_3R",
                )
                if trade:
                    hit.add("3R")
                    # Position is fully closed; no scale_out_levels_hit row to
                    # update (delete_paper_position fired inside execute_sell).
                    fires.append({
                        "ticker": ticker, "level": "3R",
                        "qty": latest_qty, "trade_id": trade.get("trade_id"),
                        "mfe_pct": round(latest_mfe, 4),
                    })
                    logger.info(
                        "phase-36.1: scale-out 3R fired for %s -- closed remainder %s qty at mfe=%.4f%% (threshold %.4f%%)",
                        ticker, latest_qty, latest_mfe, threshold_3r,
                    )

        return fires

    def _persist_scale_out_levels(self, ticker: str, levels: set[str]) -> None:
        """Idempotency support: update scale_out_levels_hit column on the
        (already-reduced) position row. Fail-open: if the column or row is
        missing, log WARN and continue (next cycle will re-detect via
        quantity-vs-cost-basis derivation if the column path fails)."""
        try:
            pos = self.get_position(ticker)
            if not pos:
                return
            pos["scale_out_levels_hit"] = json.dumps(sorted(levels))
            self._safe_save_position(pos)
        except Exception as exc:
            logger.warning(
                "phase-36.1: failed to persist scale_out_levels_hit for %s: %r",
                ticker, exc,
            )

    def backfill_missing_stops(self, default_pct: float | None = None) -> dict:
        """phase-25.2: backfill stop_loss_price for positions where it is None.

        For each open position with `stop_loss_price` None/missing, compute
        `stop = avg_entry_price * (1 - default_pct / 100)` and persist via
        `save_paper_position`. Closes phase-24.1 audit finding F-5: 6 of 11
        current positions (ON, INTC, TER, DELL, GLW, CIEN) pre-date the
        phase-23.1.8 entry-path fallback and have stop_loss_price=None.

        Args:
            default_pct: stop percentage below entry. Defaults to
                `settings.paper_default_stop_loss_pct` (8.0 per O'Neil
                canonical + arxiv 2604.27150 finding).

        Returns:
            {
              "backfilled": [list of {ticker, entry_price, stop_loss_price}],
              "skipped":    [list of tickers that already had stops],
              "count_backfilled": N,
              "count_skipped": M,
            }
        """
        if default_pct is None:
            default_pct = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))

        positions = self.get_positions()
        backfilled: list[dict] = []
        skipped: list[str] = []

        for pos in positions:
            ticker = pos.get("ticker")
            if not ticker:
                continue
            if pos.get("stop_loss_price"):
                skipped.append(ticker)
                continue
            entry_price = float(pos.get("avg_entry_price") or 0.0)
            if entry_price <= 0:
                logger.warning(
                    "backfill_missing_stops: %s has avg_entry_price=%s; cannot compute stop -- skipping",
                    ticker, entry_price,
                )
                skipped.append(ticker)
                continue
            stop_loss_price = round(entry_price * (1.0 - default_pct / 100.0), 4)
            # Preserve existing position row and only mutate the stop field.
            updated = {**pos, "stop_loss_price": stop_loss_price}
            try:
                self.bq.save_paper_position(updated)
                backfilled.append({
                    "ticker": ticker,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                })
                logger.warning(
                    "phase-25.2: backfilled stop_loss_price=%.4f for %s (entry %.4f, %.1f%% below)",
                    stop_loss_price, ticker, entry_price, default_pct,
                )
            except Exception as e:
                logger.exception("backfill_missing_stops save_paper_position failed for %s: %s", ticker, e)
                skipped.append(ticker)

        return {
            "backfilled": backfilled,
            "skipped": skipped,
            "count_backfilled": len(backfilled),
            "count_skipped": len(skipped),
        }

    def backfill_missing_company_names(self, force: bool = False) -> dict:
        """phase-32.4: backfill company_name for positions where it is missing.

        For each open position whose `company_name` is None, empty, or
        equal to the ticker (the legacy fallback sentinel), resolve via
        yfinance (`shortName` then `longName` per the canonical chain at
        `backend/api/paper_trading.py:958-968`) and persist via
        `_safe_save_position`. Fail-open on any yfinance error -- log a
        warning and continue.

        Audit basis: dashboard observation 2026-05-20 -- 9 of 11 current
        paper_positions rows showed ticker-as-company (MU, KEYS, GEV, COHR,
        ON, DELL, GLW, LITE, WDC). Same legacy pattern as phase-25.2
        stop_loss_price backfill. Cosmetic gap, not safety-critical.

        Args:
            force: when True, refresh ALL positions even when company_name
                already differs from the ticker. Default False keeps the
                helper idempotent on repeat runs.

        Returns:
            {
              "backfilled": [list of {ticker, old, new}],
              "skipped":    [list of tickers that already had real names],
              "count_backfilled": N,
              "count_skipped": M,
            }
        """
        positions = self.get_positions()
        backfilled: list[dict] = []
        skipped: list[str] = []

        for pos in positions:
            ticker = pos.get("ticker")
            if not ticker:
                continue
            current_name = (pos.get("company_name") or "").strip()
            needs_backfill = (
                force
                or not current_name
                or current_name == ticker
            )
            if not needs_backfill:
                skipped.append(ticker)
                continue
            # yfinance lookup -- fail-open so a network / rate-limit error
            # never blocks the autonomous cycle.
            try:
                import yfinance as yf
                info = yf.Ticker(ticker).info or {}
                resolved = info.get("shortName") or info.get("longName") or ticker
            except Exception as e:
                logger.warning(
                    "phase-32.4: yfinance lookup failed for %s (fail-open): %s",
                    ticker, e,
                )
                skipped.append(ticker)
                continue
            if not resolved or resolved == ticker:
                # No real name available (yfinance returned the ticker or
                # nothing). Skip rather than persist the sentinel.
                skipped.append(ticker)
                continue
            updated = {**pos, "company_name": resolved}
            try:
                self.bq.save_paper_position(updated)
                backfilled.append({
                    "ticker": ticker,
                    "old": current_name or None,
                    "new": resolved,
                })
                logger.info(
                    "phase-32.4: backfilled company_name for %s: %r -> %r",
                    ticker, current_name or None, resolved,
                )
            except Exception as e:
                logger.exception(
                    "phase-32.4: save_paper_position failed for %s: %s",
                    ticker, e,
                )
                skipped.append(ticker)

        return {
            "backfilled": backfilled,
            "skipped": skipped,
            "count_backfilled": len(backfilled),
            "count_skipped": len(skipped),
        }

    # ── Snapshot ─────────────────────────────────────────────────

    def save_daily_snapshot(
        self,
        trades_today: int = 0,
        analysis_cost_today: float = 0.0,
        external_flow_today: float = 0.0,
    ) -> dict:
        """Save daily NAV snapshot. Call after mark_to_market.

        phase-30.4: `external_flow_today` records net external cash flow
        (deposits positive, withdrawals negative) on the snapshot_date.
        Default 0.0 covers normal cycles. Operator-driven mutations via
        `adjust_cash_and_mtm` pass `delta` through here so the snapshot
        carries the explicit flow. `paper_metrics_v2._nav_to_returns`
        subtracts this from V_t before differencing -> GIPS-canonical TWR.
        Without this field, a +$5K deposit would appear as a phantom +32%
        daily return (phase-30.0 Anomaly A; 2026-05-13 incident)."""
        portfolio = self.get_or_create_portfolio()
        positions = self.get_positions()
        positions_value = sum(p.get("market_value", 0) for p in positions)
        nav = portfolio.get("total_nav", portfolio["current_cash"])
        starting = portfolio["starting_capital"]
        cum_pnl = ((nav - starting) / starting) * 100 if starting > 0 else 0.0
        benchmark = portfolio.get("benchmark_return_pct", 0.0)

        # Get yesterday's snapshot for daily P&L
        snapshots = self.bq.get_paper_snapshots(limit=1)
        prev_nav = snapshots[0].get("total_nav", starting) if snapshots else starting
        daily_pnl = ((nav - prev_nav) / prev_nav) * 100 if prev_nav > 0 else 0.0

        snap = {
            "snapshot_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_nav": round(nav, 2),
            "cash": round(portfolio["current_cash"], 2),
            "positions_value": round(positions_value, 2),
            "daily_pnl_pct": round(daily_pnl, 2),
            "cumulative_pnl_pct": round(cum_pnl, 2),
            "benchmark_pnl_pct": round(benchmark, 2),
            "alpha_pct": round(cum_pnl - benchmark, 2),
            "position_count": len(positions),
            "trades_today": trades_today,
            "analysis_cost_today": round(analysis_cost_today, 4),
            # phase-30.4: external_flow_today persisted so
            # paper_metrics_v2._nav_to_returns can subtract it before
            # differencing (GIPS-canonical TWR). 0.0 covers normal cycles;
            # adjust_cash_and_mtm threads operator deposits through here.
            "external_flow_today": round(external_flow_today, 2),
        }
        self.bq.save_paper_snapshot(snap)
        return snap

    # ── 4.5.7 Kill-switch ────────────────────────────────────────

    def flatten_all(self, reason: str = "manual_flatten") -> dict:
        """
        Close every open position at last-known market price and cancel any
        pending intent. Returns a summary dict.

        Per FINRA Rule 15c3-5 hard-block pattern: this is a terminal action
        that drives the portfolio to a known-quiet state (zero positions)
        without requiring further confirmation from the caller.

        Audited: every execute_sell() already writes to paper_trades; callers
        should also record to the kill_switch audit log.
        """
        positions = self.get_positions()
        closed: list[dict] = []
        for pos in positions:
            ticker = pos["ticker"]
            px = _get_live_price(ticker) or pos.get("current_price") or pos.get("avg_entry_price")
            trade = self.execute_sell(
                ticker=ticker,
                quantity=pos.get("quantity"),
                price=px,
                reason=reason,
            )
            if trade:
                closed.append({"ticker": ticker, "quantity": pos.get("quantity"), "price": px})
        logger.info(f"flatten_all closed {len(closed)} positions (reason={reason})")
        return {"closed_count": len(closed), "closed": closed, "reason": reason}

    def adjust_cash_and_mtm(
        self, delta: float, reason: str = "manual_adjustment",
    ) -> dict:
        """phase-23.2.17: helper for raw cash mutations that MUST be followed
        by mark_to_market. Bug class history:

        - phase-23.1.15 cleanup script bumped current_cash without MtM ->
          stale total_nav silently broke the home-cockpit Red Line and the
          paper_portfolio_snapshots row stayed wrong for 5 days
          (phase-23.1.17 had to add a separate repair script).
        - phase-23.2.2 STX cleanup repeated the same pattern + manual
          repair invocation.

        Going forward, ANY raw cash mutation (refunds, deposits via DML,
        manual corrections) should call this helper. It:
        1. Reads current_cash, applies delta, writes back via
           upsert_paper_portfolio.
        2. Calls mark_to_market() so total_nav is recomputed from current
           live position values + new cash.
        3. Saves a daily_snapshot so paper_portfolio_snapshots reflects
           the post-mutation state immediately (Red Line stays correct).

        Logs an audit line so operators can grep for `cash_mtm_adjust`.
        """
        portfolio = self.get_or_create_portfolio()
        old_cash = float(portfolio.get("current_cash") or 0.0)
        new_cash = round(old_cash + delta, 2)
        # Use upsert_paper_portfolio (already wraps current_cash + updated_at).
        self._update_portfolio_cash(new_cash)
        logger.info(
            "cash_mtm_adjust: cash %.2f -> %.2f (delta %+.2f) reason=%s",
            old_cash, new_cash, delta, reason,
        )
        # Recompute total_nav from live positions + new cash.
        mtm = self.mark_to_market()
        # Persist a snapshot so the Red Line + dashboard reflect this state.
        # phase-30.4: thread the cash delta through as external_flow_today so
        # the snapshot records the explicit operator-driven flow.
        # paper_metrics_v2._nav_to_returns subtracts this before computing
        # daily returns -> GIPS-canonical TWR. Without this, a +$5K deposit
        # appears as a +32% phantom daily return (phase-30.0 Anomaly A).
        self.save_daily_snapshot(
            trades_today=0,
            analysis_cost_today=0.0,
            external_flow_today=float(delta),
        )
        return {
            "old_cash": old_cash,
            "new_cash": new_cash,
            "delta": delta,
            "reason": reason,
            "post_nav": mtm.get("nav"),
        }

    def check_and_enforce_kill_switch(self) -> dict:
        """
        Evaluate daily-loss + trailing-DD limits. If either breached, auto-
        flatten all positions and pause new-order generation. Call this at the
        top of every autonomous cycle BEFORE deciding trades.
        """
        from backend.services.kill_switch import evaluate_breach, get_state, check_auto_resume
        portfolio = self.get_or_create_portfolio()
        nav = float(portfolio.get("total_nav") or portfolio.get("starting_capital") or 0.0)
        state = get_state()
        # Ratchet the peak upward (monotonic).
        state.update_peak(nav)
        # phase-23.2.19: idempotent daily SOD roll. Re-anchor when either
        # (a) we have never written SOD before, or (b) the SOD anchor date
        # is older than today's UTC date. Same-day re-calls are no-ops at
        # the audit-log level because the comparison evaluates to False.
        # Restart-idempotent: boot replay restores _sod_date, so the first
        # cycle after a mid-day restart sees the morning's SOD and skips.
        snap = state.snapshot()
        today = datetime.now(timezone.utc).date().isoformat()
        if snap.get("sod_nav") is None or snap.get("sod_date") != today:
            state.update_sod_nav(nav, date=today)

        breach = evaluate_breach(
            current_nav=nav,
            daily_loss_limit_pct=self.settings.paper_daily_loss_limit_pct,
            trailing_dd_limit_pct=self.settings.paper_trailing_dd_limit_pct,
        )
        if breach["any_breached"] and not state.is_paused():
            logger.warning(f"kill_switch: breach detected -- flatten+pause. details={breach}")
            flatten_result = self.flatten_all(reason="kill_switch_auto_flatten")
            state.pause(trigger="limit_breach", details={"breach": breach, "flatten": flatten_result})
            return {"triggered": True, "breach": breach, "flatten": flatten_result}

        # phase-38.1.1: hysteresis evaluation -- only fires when paused + no breach.
        # Default-OFF; operator opts in via settings.kill_switch_auto_resume_enabled.
        auto_resume = check_auto_resume(
            current_nav=nav,
            daily_loss_limit_pct=self.settings.paper_daily_loss_limit_pct,
            trailing_dd_limit_pct=self.settings.paper_trailing_dd_limit_pct,
            enabled=bool(getattr(self.settings, "kill_switch_auto_resume_enabled", False)),
        )
        if auto_resume["action"] in ("alert", "resume"):
            logger.warning(
                "kill_switch auto-resume action=%s reason=%s seconds_paused=%s",
                auto_resume["action"], auto_resume["reason"], auto_resume.get("seconds_paused"),
            )
        return {"triggered": False, "breach": breach, "auto_resume": auto_resume}

    # ── Internal ─────────────────────────────────────────────────

    def _update_portfolio_cash(self, new_cash: float) -> None:
        portfolio = self.get_or_create_portfolio()
        portfolio["current_cash"] = round(new_cash, 2)
        portfolio["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.bq.upsert_paper_portfolio(portfolio)

    def _advance_stop(
        self, pos: dict, new_mfe: float
    ) -> tuple[Optional[float], Optional[str]]:
        # phase-32.1: breakeven ratchet at +1R (one-shot, idempotent).
        # phase-32.2: after breakeven, switch to HWM-trailing branch with a
        # Kaminski-Lo Proposition 2 adversarial guard on mean-reversion/pairs
        # entries.
        #
        # Returns (new_stop_loss_price, advance_iso_or_None):
        #   * (None, None)          -- no change this cycle
        #   * (entry, ISO_now)      -- breakeven fired first time (32.1)
        #   * (new_trail_stop, None) -- trailing fired post-breakeven (32.2);
        #                              advance_iso stays None so we do NOT
        #                              overwrite the existing
        #                              stop_advanced_at_R timestamp.
        entry_price = float(pos.get("avg_entry_price") or 0.0)
        if entry_price <= 0:
            return (None, None)
        ticker = pos.get("ticker", "?")
        current_stop = pos.get("stop_loss_price")
        current_stop_f = float(current_stop) if current_stop is not None else None

        # ── Phase-32.2 trailing branch (fires post-breakeven) ──────────────
        # Active only after the breakeven ratchet has already advanced
        # stop_advanced_at_R. Kaminski-Lo Proposition 2: mean-reverting
        # strategies (and cointegrated pairs) lose expected return when
        # trailing-stop cumulative-loss thresholds fire; SKIP for those.
        # Fail-CLOSED-conservative: when entry_strategy is None/unknown,
        # treat as momentum (trail IS applied) -- forgetting to flag a
        # mean-reversion entry should err toward "more protection", not
        # "no protection".
        if pos.get("stop_advanced_at_R"):
            entry_strategy = (pos.get("entry_strategy") or "").lower().strip()
            if entry_strategy in {"mean_reversion", "pairs"}:
                return (None, None)
            trail_pct = float(getattr(self.settings, "paper_trailing_stop_pct", 8.0))
            peak_price = entry_price * (1.0 + max(new_mfe, 0.0) / 100.0)
            if peak_price <= entry_price:
                # MFE went non-positive (shouldn't happen post-breakeven but
                # defensive); nothing to trail.
                return (None, None)
            new_trail = peak_price * (1.0 - trail_pct / 100.0)
            if current_stop_f is None or new_trail <= current_stop_f:
                # Monotonic: never lower the stop.
                return (None, None)
            logger.info(
                "phase-32.2: trail fired for %s -- advanced stop from %.4f to %.4f "
                "(peak=%.4f, trail_pct=%.4f, mfe_pct=%.4f, entry_strategy=%s)",
                ticker, current_stop_f, new_trail, peak_price, trail_pct, new_mfe,
                entry_strategy or "unknown",
            )
            return (new_trail, None)

        # ── Phase-32.1 breakeven branch (one-shot) ─────────────────────────
        threshold = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))
        if new_mfe < threshold:
            return (None, None)
        if current_stop_f is not None and current_stop_f >= entry_price:
            return (None, None)
        now_iso = datetime.now(timezone.utc).isoformat()
        old_stop_str = f"{current_stop_f:.4f}" if current_stop_f is not None else "None"
        logger.info(
            "phase-32.1: ratchet fired for %s -- advanced stop from %s to %.4f "
            "at mfe_pct=%.4f (threshold %.4f)",
            ticker, old_stop_str, entry_price, new_mfe, threshold,
        )
        return (entry_price, now_iso)

    # 4.5.2: Writes tolerate pre-migration schemas by retrying without the new
    # round-trip columns if BigQuery complains. Run scripts/migrations/
    # add_round_trip_schema.py once per environment to land them.

    _ROUND_TRIP_FIELDS = {
        "round_trip_id", "holding_days", "realized_pnl_pct",
        "mfe_pct", "mae_pct", "capture_ratio", "signals",
    }
    _POSITION_RT_FIELDS = {"mfe_pct", "mae_pct", "stop_advanced_at_R", "entry_strategy", "company_name"}

    def _safe_save_trade(self, row: dict) -> None:
        try:
            self.bq.save_paper_trade(row)
        except Exception as e:
            if self._looks_like_schema_error(e):
                logger.warning("paper_trades missing round-trip columns, retrying without")
                pruned = {k: v for k, v in row.items() if k not in self._ROUND_TRIP_FIELDS}
                self.bq.save_paper_trade(pruned)
            else:
                raise

    def _safe_save_position(self, row: dict) -> None:
        try:
            self.bq.save_paper_position(row)
        except Exception as e:
            if self._looks_like_schema_error(e):
                logger.warning("paper_positions missing MFE/MAE columns, retrying without")
                pruned = {k: v for k, v in row.items() if k not in self._POSITION_RT_FIELDS}
                self.bq.save_paper_position(pruned)
            else:
                raise

    def _safe_save_round_trip(self, row: dict) -> None:
        """Insert into paper_round_trips. Silently skipped if the table doesn't exist."""
        try:
            table = self.bq._pt_table("paper_round_trips")
            from google.cloud import bigquery
            cols = ", ".join(row.keys())
            vals = ", ".join(f"@v_{k}" for k in row.keys())
            query = f"INSERT INTO `{table}` ({cols}) VALUES ({vals})"
            params = []
            for k, v in row.items():
                if isinstance(v, float):
                    params.append(bigquery.ScalarQueryParameter(f"v_{k}", "FLOAT64", v))
                elif isinstance(v, int):
                    params.append(bigquery.ScalarQueryParameter(f"v_{k}", "INT64", v))
                else:
                    params.append(bigquery.ScalarQueryParameter(
                        f"v_{k}", "STRING", str(v) if v is not None else None
                    ))
            job_config = bigquery.QueryJobConfig(query_parameters=params)
            self.bq.client.query(query, job_config=job_config).result()
        except Exception as e:
            logger.warning(f"paper_round_trips insert skipped: {e}")

    @staticmethod
    def _looks_like_schema_error(e: Exception) -> bool:
        msg = str(e).lower()
        return any(tok in msg for tok in (
            "no such column", "unrecognized name", "does not exist",
            "not found", "invalid field",
        ))


def _get_live_price(ticker: str) -> Optional[float]:
    """Fetch latest price from yfinance. Returns None on error.

    phase-50.5 (L2 data-quality door): for INTERNATIONAL tickers (.DE/.KS/...),
    drop an unambiguous bad bar (identical-OHLC+zero-vol / impossible OHLC) by
    returning None -- callers already fall back to last-known price. US bars are
    NEVER validated here (byte-identical: market_for_symbol -> "US" -> skip)."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if not hist.empty:
            from backend.backtest.markets import market_for_symbol
            if market_for_symbol(ticker) != "US":
                from backend.tools.price_quality import is_bad_bar
                row = hist.iloc[-1]
                if is_bad_bar(row.get("Open"), row.get("High"), row.get("Low"),
                              row.get("Close"), row.get("Volume")):
                    logger.debug("price_quality: dropped bad live bar for %s -> fallback", ticker)
                    return None
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.debug(f"Could not get live price for {ticker}: {e}")
    return None


def _get_benchmark_return(
    inception_date: str,
    first_funded_date: Optional[str] = None,
) -> Optional[float]:
    """SPY return since portfolio's first-funded snapshot.

    phase-38.7 (closes closure_roadmap.md section 3 OPEN-9):
    Previously anchored to inception_date, but that's set at row-creation
    time before any capital injection -- per industry taxonomy
    (PerformanceMeasurementSolutions / GIPS), it's the "Initialization Date"
    and is a documented anti-pattern for performance reporting because the
    strategy had no money to invest yet.

    Correct anchor: first_funded_date (earliest snapshot where
    positions_value > 0). Falls back to inception_date when no funded
    snapshot exists (cold-start grace), preserving original behavior.
    """
    anchor = first_funded_date or inception_date
    if not anchor:
        return None
    try:
        spy = yf.Ticker("SPY")
        start = anchor[:10]
        hist = spy.history(start=start)
        if len(hist) >= 2:
            first = float(hist["Close"].iloc[0])
            last = float(hist["Close"].iloc[-1])
            return ((last - first) / first) * 100
    except Exception as e:
        logger.debug(f"Could not compute benchmark return: {e}")
    return None
