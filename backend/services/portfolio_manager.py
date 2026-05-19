"""
Portfolio Manager — decides which trades to execute based on analysis results.

Implements sell-first-then-buy logic with Risk Judge position sizing.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.config.settings import Settings
from backend.services.signal_attribution import extract_signals_from_analysis, extract_all_signals

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
    signals: list[dict] = field(default_factory=list)  # 4.5.5 agent attribution
    sector: str = ""  # phase-23.2.6-fix: persisted to paper_positions.sector at execute_buy


# Recommendations that imply selling
_SELL_RECS = {"SELL", "STRONG_SELL"}
# Recommendations that indicate downgrade from a prior buy
_DOWNGRADE_RECS = {"HOLD", "SELL", "STRONG_SELL"}
# Recommendations that trigger buying
_BUY_RECS = {"BUY", "STRONG_BUY"}


def decide_trades(
    current_positions: list[dict],
    candidate_analyses: list[dict],
    holding_analyses: list[dict],
    portfolio_state: dict,
    settings: Settings,
    candidates_by_ticker: dict[str, dict] | None = None,
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
            rec = analysis.get("recommendation", "HOLD").upper()
            old_rec = (pos.get("recommendation") or "").upper()

            # Explicit sell signal
            if rec in _SELL_RECS:
                orders.append(TradeOrder(
                    ticker=ticker, action="SELL", reason="sell_signal",
                    analysis_id=analysis.get("analysis_date", ""),
                    price=pos.get("current_price"),
                    signals=extract_signals_from_analysis(analysis),
                ))
                continue

            # Downgrade: was a buy, now hold/sell
            if old_rec in _BUY_RECS and rec in _DOWNGRADE_RECS:
                orders.append(TradeOrder(
                    ticker=ticker, action="SELL", reason="signal_downgrade",
                    analysis_id=analysis.get("analysis_date", ""),
                    price=pos.get("current_price"),
                    signals=extract_signals_from_analysis(analysis),
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
        rec = analysis.get("recommendation", "HOLD").upper()

        # Skip if already held (and not being sold)
        if ticker in held_tickers and ticker not in selling_tickers:
            continue

        if rec not in _BUY_RECS:
            continue

        # Extract Risk Judge sizing
        risk_assessment = analysis.get("risk_assessment", {})
        position_pct = _extract_position_pct(risk_assessment, analysis)
        stop_loss = _extract_stop_loss(risk_assessment, analysis, settings=settings)
        final_score = analysis.get("final_score", 0)

        # phase-23.1.7: pull the screener candidate dict (contains momentum / RSI /
        # composite_score / conviction / signal-stack tags) so the rationale captures
        # ALL inputs that drove this decision, not just the LLM's verdict.
        screener_candidate = (candidates_by_ticker or {}).get(ticker)
        # phase-23.1.13: capture sector for the per-sector cap. Resolution order:
        # screener candidate (preferred -- enriched in autonomous_loop via
        # _fetch_ticker_meta) -> analysis full_report.market_data.sector ->
        # analysis.sector -> "Unknown" sentinel.
        cand_sector = ""
        if screener_candidate:
            cand_sector = screener_candidate.get("sector") or ""
        if not cand_sector:
            full_report = analysis.get("full_report") or {}
            md = full_report.get("market_data") or {}
            cand_sector = md.get("sector") or analysis.get("sector") or ""
        buy_candidates.append({
            "ticker": ticker,
            "recommendation": rec,
            "position_pct": position_pct,
            "stop_loss_price": stop_loss,
            "risk_judge_decision": risk_assessment.get("decision", ""),
            "analysis_id": analysis.get("analysis_date", ""),
            "final_score": final_score,
            "price": analysis.get("price_at_analysis"),
            "sector": cand_sector or "Unknown",
            "signals": extract_all_signals(analysis, candidate=screener_candidate),
        })

        decision = risk_assessment.get("decision", "") or ""
        if decision and decision != "APPROVE_FULL":
            logger.info(
                "buy_candidate risk_judge decision=%s ticker=%s position_pct=%s final_score=%s",
                decision, ticker, position_pct, round(float(final_score or 0), 3),
            )

    # Sort by final_score descending (best opportunities first)
    buy_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # phase-23.1.13: build sector_counts from current positions NOT being sold,
    # plus a meta lookup for legacy positions that lack a sector field. The cap
    # is `settings.paper_max_per_sector` (0 disables). We block a candidate's
    # BUY when its sector has already reached the cap; non-blocked sectors go
    # through normally. Increments after each accepted BUY so 3rd+ candidates
    # in the same sector skip cleanly.
    #
    # phase-30.5: ALSO build sector_market_values for the NAV-percentage cap
    # (`settings.paper_max_per_sector_nav_pct`, 0 disables, default 30). The
    # two caps fire independently: count handles "many small positions",
    # NAV-pct handles "one fat position dominating the sector". Sound source:
    # arXiv 2512.02227 Dec 2025 Orchestration Framework `sectorLimit: 0.30`.
    # Edge cases per research_brief.md Section 4: missing market_value -> 0
    # (conservative, no crash); cap=0 disables; candidate self-exceeds gets
    # blocked because existing_sector_value + buy_amount is checked; already-
    # over sector keeps existing semantics (no force-divest).
    max_per_sector = int(getattr(settings, "paper_max_per_sector", 0) or 0)
    max_sector_nav_pct = float(
        getattr(settings, "paper_max_per_sector_nav_pct", 0.0) or 0.0
    )
    sector_counts: dict[str, int] = {}
    sector_market_values: dict[str, float] = {}
    if max_per_sector > 0 or max_sector_nav_pct > 0:
        for pos in current_positions:
            if pos["ticker"] in selling_tickers:
                continue
            s = (pos.get("sector") or "").strip() or "Unknown"
            sector_counts[s] = sector_counts.get(s, 0) + 1
            sector_market_values[s] = (
                sector_market_values.get(s, 0.0)
                + float(pos.get("market_value", 0) or 0)
            )

    for cand in buy_candidates:
        if remaining_positions >= settings.paper_max_positions:
            # phase-23.2.22: emit a diagnostic line so 0-trade cycles are
            # diagnosable from logs without forensic analysis. Without this
            # the loop silently `break`s and the cycle reports "Executing 0
            # trades" with no explanation -- making the position cap look
            # like a bug.
            logger.info(
                "Position cap reached: %d held >= %d max -- skipping all BUY candidates",
                remaining_positions, settings.paper_max_positions,
            )
            break

        if available_cash <= 0:
            break

        # phase-23.1.13: per-GICS-sector cap (default 2). 0 disables the check.
        if max_per_sector > 0:
            cand_sector = cand.get("sector") or "Unknown"
            current_in_sector = sector_counts.get(cand_sector, 0)
            if current_in_sector >= max_per_sector:
                logger.info(
                    "Skipping BUY %s: sector %s at cap (%d/%d)",
                    cand["ticker"], cand_sector,
                    current_in_sector, max_per_sector,
                )
                continue

        # Position size: min(risk_judge_pct * NAV, available_cash)
        position_pct = cand["position_pct"] or 10.0  # Default 10% if Risk Judge didn't specify
        target_amount = nav * (position_pct / 100.0)
        buy_amount = min(target_amount, available_cash)

        # Skip tiny positions (less than $50)
        if buy_amount < 50:
            logger.warning(
                "Skipping BUY %s: buy_amount=%.2f below $50 minimum (nav=%.2f position_pct=%s available_cash=%.2f)",
                cand["ticker"], buy_amount, nav, position_pct, available_cash,
            )
            continue

        # phase-30.5: per-sector NAV-percentage cap. Independent of the
        # count cap above. Cap=0 disables. Check AFTER buy_amount is known
        # so we know exactly how much this BUY would push the sector past.
        if max_sector_nav_pct > 0 and nav > 0:
            cand_sector_nav = cand.get("sector") or "Unknown"
            existing_sector_value = sector_market_values.get(cand_sector_nav, 0.0)
            projected_sector_pct = (
                (existing_sector_value + buy_amount) / nav * 100.0
            )
            if projected_sector_pct > max_sector_nav_pct:
                logger.info(
                    "Skipping BUY %s: sector %s would hit NAV-pct cap "
                    "(%.2f%% projected > %.2f%% cap; existing=$%.2f buy=$%.2f nav=$%.2f)",
                    cand["ticker"], cand_sector_nav,
                    projected_sector_pct, max_sector_nav_pct,
                    existing_sector_value, buy_amount, nav,
                )
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
            signals=cand.get("signals", []),
            sector=cand.get("sector", ""),  # phase-23.2.6-fix
        ))
        available_cash -= buy_amount
        remaining_positions += 1
        # phase-23.1.13: increment sector count after a BUY clears the cap
        # phase-30.5: ALSO increment sector_market_values so the next
        # candidate in the same sector sees the updated NAV-pct.
        if max_per_sector > 0 or max_sector_nav_pct > 0:
            cs = cand.get("sector") or "Unknown"
            sector_counts[cs] = sector_counts.get(cs, 0) + 1
            sector_market_values[cs] = sector_market_values.get(cs, 0.0) + buy_amount

    logger.info(f"Trade decisions: {len([o for o in orders if o.action == 'SELL'])} sells, "
                f"{len([o for o in orders if o.action == 'BUY'])} buys")
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


def _extract_stop_loss(
    risk_assessment: dict,
    analysis: dict,
    settings: Optional["Settings"] = None,
) -> Optional[float]:
    """Extract stop loss price from risk assessment.

    Resolution order:
      1. Explicit `risk_assessment.risk_limits.stop_loss` (absolute price)
      2. Encoded `risk_assessment.risk_limits.stop_loss_pct` (% below entry)
      3. phase-23.1.8 fallback: `settings.paper_default_stop_loss_pct` * entry
         price. Activates on lite-Claude-analyzer BUYs where risk_assessment
         is just `{"reason": "..."}` and risk_limits is absent.

    Returns None when no price_at_analysis is available (cannot derive any
    stop without an entry reference).
    """
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
    # phase-23.1.8: settings-driven default (Option B from research brief)
    if settings is not None and price:
        default_pct = getattr(settings, "paper_default_stop_loss_pct", None)
        if default_pct is not None:
            try:
                return float(price) * (1 - float(default_pct) / 100.0)
            except (ValueError, TypeError):
                pass
    return None
