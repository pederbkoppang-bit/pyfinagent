"""
Outcome tracker service — evaluates past recommendations against actual price changes.
This is the core of the 'learning' capability.
"""

import logging
from datetime import datetime, timedelta

from backend.config.settings import Settings
from backend.db.bigquery_client import BigQueryClient
from backend.tools.yfinance_tool import get_comprehensive_financials

logger = logging.getLogger(__name__)

# Evaluation windows (in days)
EVAL_WINDOWS = [7, 30, 90, 180, 365]


class OutcomeTracker:
    """Evaluates historical recommendations against actual price performance."""

    def __init__(self, settings: Settings):
        self.bq = BigQueryClient(settings)

    def evaluate_recommendation(self, ticker: str, analysis_date: str,
                                recommendation: str, price_at_rec: float) -> dict | None:
        """
        Evaluate a single historical recommendation.
        Compares price at recommendation time to current price.
        Returns outcome dict or None if evaluation not possible.
        """
        current_data = get_comprehensive_financials(ticker)
        current_price = current_data.get("valuation", {}).get("Current Price")
        if not current_price or not price_at_rec:
            return None

        rec_date = datetime.fromisoformat(analysis_date)
        holding_days = (datetime.utcnow() - rec_date).days
        return_pct = ((current_price - price_at_rec) / price_at_rec) * 100

        # Simple benchmark: SPY average annual return ~10%
        expected_benchmark_return = (10.0 / 365) * holding_days
        beat_benchmark = return_pct > expected_benchmark_return

        # Determine if recommendation was directionally correct
        is_buy = recommendation in ("Strong Buy", "Buy")
        is_sell = recommendation in ("Strong Sell", "Sell")
        directionally_correct = (is_buy and return_pct > 0) or (is_sell and return_pct < 0)

        outcome = {
            "ticker": ticker,
            "analysis_date": analysis_date,
            "recommendation": recommendation,
            "price_at_recommendation": price_at_rec,
            "current_price": current_price,
            "return_pct": round(return_pct, 2),
            "holding_days": holding_days,
            "beat_benchmark": beat_benchmark,
            "directionally_correct": directionally_correct,
        }

        # Persist to BigQuery
        self.bq.save_outcome(
            ticker=ticker,
            analysis_date=analysis_date,
            recommendation=recommendation,
            price_at_rec=price_at_rec,
            current_price=current_price,
            return_pct=return_pct,
            holding_days=holding_days,
            beat_benchmark=beat_benchmark,
        )

        return outcome

    def evaluate_all_pending(self) -> list[dict]:
        """
        Fetch all past reports and evaluate those that haven't been evaluated yet
        or need re-evaluation (new eval window reached).
        """
        reports = self.bq.get_recent_reports(limit=100)
        results = []

        for report in reports:
            rec_date = datetime.fromisoformat(report["analysis_date"])
            days_since = (datetime.utcnow() - rec_date).days

            # Only evaluate if at least 7 days have passed
            if days_since < 7:
                continue

            # Get the price at recommendation time from the stored report
            stored = self.bq.get_report(report["ticker"], report["analysis_date"])
            if not stored or not stored.get("full_report_json"):
                continue

            full = stored["full_report_json"]
            if isinstance(full, str):
                import json
                full = json.loads(full)

            price_at_rec = (
                full.get("quant", {})
                .get("yf_data", {})
                .get("valuation", {})
                .get("Current Price")
            )
            if not price_at_rec:
                continue

            outcome = self.evaluate_recommendation(
                ticker=report["ticker"],
                analysis_date=report["analysis_date"],
                recommendation=report["recommendation"],
                price_at_rec=price_at_rec,
            )
            if outcome:
                results.append(outcome)

        return results

    def get_performance_summary(self) -> dict:
        """Get aggregated performance stats from BigQuery."""
        return self.bq.get_performance_stats()
