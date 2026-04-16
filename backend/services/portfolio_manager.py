"""
Portfolio Manager — decides which trades to execute based on analysis results.

Implements sell-first-then-buy logic with Risk Judge position sizing.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class TradeOrder:
    ticker: str
    action: str  # "BUY" or "SELL"
    amount_usd: Optional[float] = None   # For buys
    quantity: Optional[float] = None      # For sells (None = full exit)
    reason: str = ""
    analysis_id: str = ""
    risk_judge_decision: str = ""
    stop_loss_price: Optional[float] = None
    risk_judge_position_pct: Optional[float] = None
    price: Optional[float] = None


# Recommendations that imply selling
_SELL_RECS = {"SELL", "STRONG_SELL"}
# Recommendations that indicate downgrade from a prior buy
_DOWNGRADE_RECS = {"HOLD", "SELL", "STRONG_SELL"}
# Recommendations that trigger buying
_BUY_RECS = {"BUY", "STRONG_BUY"}


def _normalize_rec(raw: str) -> str:
    """Normalize recommendation strings to the canonical underscore-uppercase form.

    The Gemini synthesis schema uses "Strong Buy" / "Strong Sell" (space, mixed case)
    while the lookup sets use "STRONG_BUY" / "STRONG_SELL" (underscore, uppercase).
    """
    return raw.strip().upper().replace(" ", "_")


def decide_trades(
    current_positions: list[dict],
    candidate_analyses: list[dict],
    holding_analyses: list[dict],
    portfolio_state: dict,
    settings: Settings,
) -> list[TradeOrder]:
    """
    Decide which trades to execute.

    Args:
        current_positions: List of open position dicts from BQ
        candidate_analyses: Analysis results for new candidates (with recommendation, risk_assessment)
        holding_analyses: Analysis results for current holdings (re-evaluation)
        portfolio_state: Dict with keys: nav, cash, positions_value, position_count
        settings: App settings

    Returns:
        List of TradeOrders (sells first, then buys)
    """
    orders: list[TradeOrder] = []
    nav = portfolio_state.get("nav", settings.paper_starting_capital)
    cash = portfolio_state.get("cash", nav)
    min_cash = nav * (settings.paper_min_cash_reserve_pct / 100.0)
    held_tickers = {p["ticker"] for p in current_positions}

    # ── 1. Sell decisions (process first to free up cash) ────────

    # Build lookup: ticker -> re-evaluation result
    holding_lookup = {}
    for analysis in holding_analyses:
        ticker = analysis.get("ticker", "")
        if ticker:
            holding_lookup[ticker] = analysis

    for pos in current_positions:
        ticker = pos["ticker"]
        analysis = holding_lookup.get(ticker)

        # Stop loss check (already priced in mark-to-market)
        stop = pos.get("stop_loss_price")
        current = pos.get("current_price", 0)
        if stop and current and current <= stop:
            orders.append(TradeOrder(
                ticker=ticker, action="SELL", reason="stop_loss",
                price=current,
            ))
            continue

        # If we have a fresh re-evaluation
        if analysis:
            rec = _normalize_rec(analysis.get("recommendation", "HOLD"))
            old_rec = _normalize_rec(pos.get("recommendation") or "")

            # Explicit sell signal
            if rec in _SELL_RECS:
                orders.append(TradeOrder(
                    ticker=ticker, action="SELL", reason="sell_signal",
                    analysis_id=analysis.get("analysis_date", ""),
                    price=pos.get("current_price"),
                ))
                continue

            # Downgrade: was a buy, now hold/sell
            if old_rec in _BUY_RECS and rec in _DOWNGRADE_RECS:
                orders.append(TradeOrder(
                    ticker=ticker, action="SELL", reason="signal_downgrade",
                    analysis_id=analysis.get("analysis_date", ""),
                    price=pos.get("current_price"),
                ))
                continue

    # Tickers being sold
    selling_tickers = {o.ticker for o in orders if o.action == "SELL"}

    # Estimate cash freed from sells
    estimated_freed_cash = 0.0
    for pos in current_positions:
        if pos["ticker"] in selling_tickers:
            estimated_freed_cash += pos.get("market_value", 0)

    available_cash = cash + estimated_freed_cash - min_cash

    # ── 2. Buy decisions ─────────────────────────────────────────

    # Count positions after sells
    remaining_positions = len(current_positions) - len(selling_tickers)

    buy_candidates = []
    for analysis in candidate_analyses:
        ticker = analysis.get("ticker", "")
        rec = _normalize_rec(analysis.get("recommendation", "HOLD"))

        # Skip if already held (and not being sold)
        if ticker in held_tickers and ticker not in selling_tickers:
            continue

        if rec not in _BUY_RECS:
            continue

        # Extract Risk Judge sizing
        risk_assessment = analysis.get("risk_assessment", {})
        position_pct = _extract_position_pct(risk_assessment, analysis)
        stop_loss = _extract_stop_loss(risk_assessment, analysis)
        final_score = analysis.get("final_score", 0)

        buy_candidates.append({
            "ticker": ticker,
            "recommendation": rec,
            "position_pct": position_pct,
            "stop_loss_price": stop_loss,
            "risk_judge_decision": risk_assessment.get("decision", ""),
            "analysis_id": analysis.get("analysis_date", ""),
            "final_score": final_score,
            "price": analysis.get("price_at_analysis"),
        })

    # Sort by final_score descending (best opportunities first)
    buy_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    for cand in buy_candidates:
        if remaining_positions >= settings.paper_max_positions:
            break

        if available_cash <= 0:
            break

        # Position size: min(risk_judge_pct * NAV, available_cash)
        position_pct = cand["position_pct"] or 10.0  # Default 10% if Risk Judge didn't specify
        target_amount = nav * (position_pct / 100.0)
        buy_amount = min(target_amount, available_cash)

        # Skip tiny positions (less than $50)
        if buy_amount < 50:
            continue

        orders.append(TradeOrder(
            ticker=cand["ticker"],
            action="BUY",
            amount_usd=round(buy_amount, 2),
            reason="new_buy_signal",
            analysis_id=cand["analysis_id"],
            risk_judge_decision=cand["risk_judge_decision"],
            stop_loss_price=cand["stop_loss_price"],
            risk_judge_position_pct=cand["position_pct"],
            price=cand.get("price"),
        ))
        available_cash -= buy_amount
        remaining_positions += 1

    n_sells = len([o for o in orders if o.action == "SELL"])
    n_buys = len([o for o in orders if o.action == "BUY"])
    logger.info(f"Trade decisions: {n_sells} sells, {n_buys} buys")

    if not orders:
        # Diagnostic: explain why zero orders were generated
        recs = [_normalize_rec(a.get("recommendation", "HOLD")) for a in candidate_analyses]
        rec_counts: dict[str, int] = {}
        for r in recs:
            rec_counts[r] = rec_counts.get(r, 0) + 1
        logger.warning(
            f"Zero orders generated. Diagnostics: "
            f"candidate_analyses={len(candidate_analyses)}, "
            f"holding_analyses={len(holding_analyses)}, "
            f"positions={len(current_positions)}, "
            f"cash={cash:.2f}, nav={nav:.2f}, "
            f"recommendations={rec_counts}"
        )

    return orders


def _extract_position_pct(risk_assessment: dict, analysis: dict) -> Optional[float]:
    """Extract recommended position % from risk assessment."""
    # Try risk_judge output
    pct = risk_assessment.get("recommended_position_pct")
    if pct:
        try:
            return float(pct)
        except (ValueError, TypeError):
            pass
    # Fall back to analysis-level field
    pct = analysis.get("risk_judge_position_pct")
    if pct:
        try:
            return float(pct)
        except (ValueError, TypeError):
            pass
    return None


def _extract_stop_loss(risk_assessment: dict, analysis: dict) -> Optional[float]:
    """Extract stop loss price from risk assessment."""
    limits = risk_assessment.get("risk_limits", {})
    if isinstance(limits, dict):
        stop = limits.get("stop_loss")
        if stop:
            try:
                return float(stop)
            except (ValueError, TypeError):
                pass
    # Maybe encoded as % below entry price
    stop_pct = limits.get("stop_loss_pct") if isinstance(limits, dict) else None
    price = analysis.get("price_at_analysis")
    if stop_pct and price:
        try:
            return float(price) * (1 - float(stop_pct) / 100.0)
        except (ValueError, TypeError):
            pass
    return None
