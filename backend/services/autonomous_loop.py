"""
Autonomous Paper Trading Loop — daily cycle orchestrator.

Screen → Analyze → Decide → Trade → Snapshot → Learn.
Designed to run as an APScheduler cron job.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.agents.meta_coordinator import MetaCoordinator
from backend.agents.orchestrator import AnalysisOrchestrator
from backend.config.settings import Settings, get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.paper_trader import PaperTrader
from backend.services.portfolio_manager import decide_trades
from backend.tools.screener import screen_universe, rank_candidates

logger = logging.getLogger(__name__)

# Path to optimizer best parameters
_OPTIMIZER_BEST_PATH = Path(__file__).parent.parent / "backtest" / "experiments" / "optimizer_best.json"


def load_best_params() -> dict:
    """Load the best backtest parameters from optimizer_best.json."""
    if not _OPTIMIZER_BEST_PATH.exists():
        logger.warning("optimizer_best.json not found, using defaults")
        return {}
    with open(_OPTIMIZER_BEST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    params = data.get("params", data)
    sharpe = data.get("sharpe", "?")
    logger.info(f"Loaded best params (Sharpe {sharpe}): {list(params.keys())}")
    return params


# Module-level state
_running = False
_last_run: Optional[str] = None
_last_result: Optional[dict] = None
_coordinator = MetaCoordinator()


async def run_daily_cycle(settings: Optional[Settings] = None) -> dict:
    """
    Execute one full paper trading cycle:
    1. Screen universe (free)
    2. Analyze top candidates (lite mode)
    3. Re-evaluate holdings due for refresh
    4. Decide trades
    5. Execute trades
    6. Mark to market
    7. Save snapshot
    8. Learn from closed trades

    Returns summary dict.
    """
    global _running, _last_run, _last_result

    if _running:
        logger.warning("Paper trading cycle already running, skipping")
        return {"status": "skipped", "reason": "already_running"}

    _running = True
    settings = settings or get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)
    total_analysis_cost = 0.0
    trades_executed = 0
    summary = {"status": "running", "steps": []}

    # Load optimizer best params for strategy decisions
    best_params = load_best_params()
    if best_params:
        summary["best_params_sharpe"] = best_params.get("sharpe", "?")
        summary["strategy_params"] = {
            k: best_params[k] for k in ["tp_pct", "sl_pct", "holding_days"]
            if k in best_params
        }

    try:
        # ── Step 1: Screen universe (free) ───────────────────────
        logger.info("Paper trading: Step 1 -- Screening universe")
        summary["steps"].append("screening")

        screen_data = screen_universe(period="6mo")
        candidates = rank_candidates(screen_data, top_n=settings.paper_screen_top_n)
        summary["screened"] = len(screen_data)
        summary["candidates"] = len(candidates)

        # ── Step 2: Filter candidates ────────────────────────────
        positions = trader.get_positions()
        held_tickers = {p["ticker"] for p in positions}
        new_candidates = [c for c in candidates if c["ticker"] not in held_tickers]
        analyze_tickers = [c["ticker"] for c in new_candidates[:settings.paper_analyze_top_n]]

        # Determine holdings due for re-evaluation
        reeval_tickers = []
        now = datetime.now(timezone.utc)
        for pos in positions:
            last_date = pos.get("last_analysis_date", "")
            if not last_date:
                reeval_tickers.append(pos["ticker"])
                continue
            try:
                last_dt = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
                days_since = (now - last_dt).days
                if days_since >= settings.paper_reeval_frequency_days:
                    reeval_tickers.append(pos["ticker"])
            except (ValueError, TypeError):
                reeval_tickers.append(pos["ticker"])

        summary["new_to_analyze"] = len(analyze_tickers)
        summary["reeval_tickers"] = len(reeval_tickers)

        # ── Step 3: Analyze candidates (lite mode) ───────────────
        logger.info(f"Paper trading: Step 3 -- Analyzing {len(analyze_tickers)} new + {len(reeval_tickers)} re-evals")
        summary["steps"].append("analyzing")

        # Force lite mode for paper trading (cost control)
        original_lite = settings.lite_mode
        settings.lite_mode = True

        candidate_analyses = []
        for ticker in analyze_tickers:
            if total_analysis_cost >= settings.paper_max_daily_cost_usd:
                logger.warning(f"Daily cost cap (${settings.paper_max_daily_cost_usd}) reached, stopping analysis")
                break
            try:
                analysis = await _run_single_analysis(ticker, settings)
                if analysis:
                    candidate_analyses.append(analysis)
                    cost = analysis.get("total_cost_usd", 0.1)
                    total_analysis_cost += cost
            except Exception as e:
                logger.error(f"Failed to analyze candidate {ticker}: {e}")

        # ── Step 4: Re-evaluate holdings ─────────────────────────
        holding_analyses = []
        for ticker in reeval_tickers:
            if total_analysis_cost >= settings.paper_max_daily_cost_usd:
                logger.warning(f"Daily cost cap reached, stopping re-evaluation")
                break
            try:
                analysis = await _run_single_analysis(ticker, settings)
                if analysis:
                    holding_analyses.append(analysis)
                    cost = analysis.get("total_cost_usd", 0.1)
                    total_analysis_cost += cost
            except Exception as e:
                logger.error(f"Failed to re-evaluate {ticker}: {e}")

        # Restore lite mode setting
        settings.lite_mode = original_lite

        # ── Step 5: Mark to market ───────────────────────────────
        logger.info("Paper trading: Step 5 -- Mark to market")
        summary["steps"].append("mark_to_market")
        portfolio_state = trader.mark_to_market()

        # ── Step 6: Decide trades ────────────────────────────────
        logger.info("Paper trading: Step 6 -- Deciding trades")
        summary["steps"].append("deciding")
        positions = trader.get_positions()  # Refresh after MTM
        orders = decide_trades(
            current_positions=positions,
            candidate_analyses=candidate_analyses,
            holding_analyses=holding_analyses,
            portfolio_state=portfolio_state,
            settings=settings,
        )

        # ── Step 7: Execute trades ───────────────────────────────
        logger.info(f"Paper trading: Step 7 -- Executing {len(orders)} trades")
        summary["steps"].append("executing")
        closed_tickers = []

        # Sells first
        for order in orders:
            if order.action != "SELL":
                continue
            trade = trader.execute_sell(
                ticker=order.ticker,
                quantity=order.quantity,
                price=order.price,
                reason=order.reason,
            )
            if trade:
                trades_executed += 1
                closed_tickers.append(order.ticker)

        # Then buys
        for order in orders:
            if order.action != "BUY":
                continue
            price = order.price
            if price is None:
                from backend.services.paper_trader import _get_live_price
                price = _get_live_price(order.ticker) or 0
            if price <= 0:
                continue
            trade = trader.execute_buy(
                ticker=order.ticker,
                amount_usd=order.amount_usd or 0,
                price=price,
                reason=order.reason,
                analysis_id=order.analysis_id,
                risk_judge_decision=order.risk_judge_decision,
                stop_loss_price=order.stop_loss_price,
                risk_judge_position_pct=order.risk_judge_position_pct,
            )
            if trade:
                trades_executed += 1

        # ── Step 8: Final mark-to-market + snapshot ──────────────
        logger.info("Paper trading: Step 8 -- Final snapshot")
        summary["steps"].append("snapshot")
        final_state = trader.mark_to_market()
        snapshot = trader.save_daily_snapshot(
            trades_today=trades_executed,
            analysis_cost_today=total_analysis_cost,
        )

        # ── Step 9: Learn from closed trades ─────────────────────
        if closed_tickers:
            summary["steps"].append("learning")
            try:
                await _learn_from_closed_trades(closed_tickers, bq, settings)
            except Exception as e:
                logger.error(f"Learning step failed (non-fatal): {e}")

        # ── Step 10: MetaCoordinator health check ────────────────
        try:
            snapshots = bq.get_paper_snapshots(limit=60)
            from backend.services.perf_tracker import get_perf_tracker
            health = MetaCoordinator.gather_health(
                bq_client=bq,
                perf_tracker=get_perf_tracker(),
                paper_snapshots=snapshots,
            )
            decision = _coordinator.decide(health)
            summary["coordinator"] = {
                "action": decision.action,
                "reason": decision.reason,
                "target_agents": decision.target_agents,
                "priority": decision.priority,
                "health": {
                    "sharpe": round(health.sharpe_ratio, 4),
                    "accuracy": round(health.agent_accuracy, 4),
                    "p95_latency_ms": round(health.p95_latency_ms, 1),
                },
            }
            logger.info(
                f"MetaCoordinator decision: {decision.action} "
                f"(reason={decision.reason})"
            )
        except Exception as e:
            logger.warning(f"MetaCoordinator step failed (non-fatal): {e}")

        # ── Done ─────────────────────────────────────────────────
        summary.update({
            "status": "completed",
            "nav": final_state["nav"],
            "pnl_pct": final_state["pnl_pct"],
            "trades_executed": trades_executed,
            "analysis_cost": round(total_analysis_cost, 4),
            "closed_tickers": closed_tickers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        _last_run = summary["timestamp"]
        _last_result = summary
        logger.info(f"Paper trading cycle complete: NAV=${final_state['nav']:.2f}, "
                     f"P&L={final_state['pnl_pct']:.2f}%, trades={trades_executed}, "
                     f"cost=${total_analysis_cost:.4f}")
        return summary

    except Exception as e:
        logger.error(f"Paper trading cycle failed: {e}", exc_info=True)
        summary.update({"status": "error", "error": str(e)})
        _last_result = summary
        return summary
    finally:
        _running = False


async def _run_single_analysis(ticker: str, settings: Settings) -> Optional[dict]:
    """Run a single analysis and extract key fields for trade decisions."""
    orchestrator = AnalysisOrchestrator(settings)
    report = await orchestrator.run_full_analysis(ticker)
    if not report:
        return None

    synthesis = report.get("final_synthesis", {})
    rec = synthesis.get("recommendation", {})
    quant = report.get("quant", {})
    risk = synthesis.get("risk_assessment", {})
    cost_summary = report.get("cost_summary", {})

    return {
        "ticker": ticker,
        "recommendation": rec.get("action", "HOLD") if isinstance(rec, dict) else str(rec),
        "final_score": synthesis.get("final_score", 0),
        "risk_assessment": risk,
        "price_at_analysis": quant.get("yf_data", {}).get("valuation", {}).get("currentPrice") if isinstance(quant.get("yf_data"), dict) else None,
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "total_cost_usd": cost_summary.get("total_cost_usd", 0.1) if isinstance(cost_summary, dict) else 0.1,
        "full_report": report,
    }


async def _learn_from_closed_trades(tickers: list[str], bq: BigQueryClient, settings: Settings):
    """Feed closed trades into outcome tracking for reflection generation."""
    from backend.services.outcome_tracker import OutcomeTracker

    tracker = OutcomeTracker(settings)

    # Get recent sell trades to find analysis_date, recommendation, and entry price
    recent_trades = bq.get_paper_trades(limit=50)
    sell_by_ticker = {}
    for t in recent_trades:
        if t.get("action") == "SELL" and t.get("ticker") in tickers:
            sell_by_ticker.setdefault(t["ticker"], t)

    for ticker in tickers:
        try:
            trade = sell_by_ticker.get(ticker)
            if not trade:
                logger.debug(f"No sell trade found for {ticker}, skipping outcome eval")
                continue
            analysis_date = trade.get("analysis_id") or trade.get("created_at", "")
            if hasattr(analysis_date, "isoformat"):
                analysis_date = analysis_date.isoformat()
            recommendation = trade.get("risk_judge_decision", "HOLD")
            price_at_rec = trade.get("price", 0.0)
            tracker.evaluate_recommendation(ticker, str(analysis_date), recommendation, price_at_rec)
        except Exception as e:
            logger.debug(f"Outcome evaluation failed for {ticker}: {e}")


def get_loop_status() -> dict:
    """Return current status of the autonomous loop."""
    return {
        "running": _running,
        "last_run": _last_run,
        "last_result": _last_result,
    }


def get_coordinator() -> MetaCoordinator:
    """Return the module-level MetaCoordinator instance."""
    return _coordinator
