"""
BigQuery client wrapper for report persistence and outcome tracking.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from google.cloud import bigquery
from google.oauth2 import service_account

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class BigQueryClient:
    """Thin wrapper around google.cloud.bigquery.Client."""

    def __init__(self, settings: Settings):
        self.settings = settings
        credentials = None
        if settings.gcp_credentials_json:
            creds_info = json.loads(settings.gcp_credentials_json)
            credentials = service_account.Credentials.from_service_account_info(creds_info)

        self.client = bigquery.Client(project=settings.gcp_project_id, credentials=credentials)
        self.reports_table = f"{settings.gcp_project_id}.{settings.bq_dataset_reports}.{settings.bq_table_reports}"
        self.outcomes_table = f"{settings.gcp_project_id}.{settings.bq_dataset_outcomes}.{settings.bq_table_outcomes}"

    # ── Reports ──────────────────────────────────────────────────────

    def save_report(self, ticker: str, company_name: str, final_score: float,
                    recommendation: str, summary: str, full_report: dict) -> None:
        row = {
            "ticker": ticker,
            "company_name": company_name,
            "analysis_date": datetime.utcnow().isoformat(),
            "final_score": final_score,
            "recommendation": recommendation,
            "summary": summary,
            "full_report_json": json.dumps(full_report),
        }
        errors = self.client.insert_rows_json(self.reports_table, [row])
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            raise RuntimeError(f"Failed to save report: {errors}")
        logger.info(f"Report saved for {ticker}")

    def get_recent_reports(self, limit: int = 20) -> list[dict]:
        query = f"""
            SELECT ticker, company_name, analysis_date, final_score, recommendation, summary
            FROM `{self.reports_table}`
            ORDER BY analysis_date DESC
            LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        )
        rows = self.client.query(query, job_config=job_config).result()
        return [dict(row) for row in rows]

    def get_report(self, ticker: str, analysis_date: Optional[str] = None) -> Optional[dict]:
        if analysis_date:
            query = f"""
                SELECT * FROM `{self.reports_table}`
                WHERE ticker = @ticker AND analysis_date = @analysis_date
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                bigquery.ScalarQueryParameter("analysis_date", "STRING", analysis_date),
            ])
        else:
            query = f"""
                SELECT * FROM `{self.reports_table}`
                WHERE ticker = @ticker
                ORDER BY analysis_date DESC
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
            ])
        rows = list(self.client.query(query, job_config=job_config).result())
        if rows:
            row = dict(rows[0])
            if row.get("full_report_json") and isinstance(row["full_report_json"], str):
                row["full_report_json"] = json.loads(row["full_report_json"])
            return row
        return None

    # ── Outcome Tracking ─────────────────────────────────────────────

    def save_outcome(self, ticker: str, analysis_date: str,
                     recommendation: str, price_at_rec: float,
                     current_price: float, return_pct: float,
                     holding_days: int, beat_benchmark: bool) -> None:
        row = {
            "ticker": ticker,
            "analysis_date": analysis_date,
            "recommendation": recommendation,
            "price_at_recommendation": price_at_rec,
            "current_price": current_price,
            "return_pct": return_pct,
            "holding_days": holding_days,
            "beat_benchmark": beat_benchmark,
            "evaluated_at": datetime.utcnow().isoformat(),
        }
        errors = self.client.insert_rows_json(self.outcomes_table, [row])
        if errors:
            logger.error(f"Outcome insert errors: {errors}")

    def get_performance_stats(self) -> dict:
        query = f"""
            SELECT
                COUNT(*) as total_recommendations,
                COUNTIF(return_pct > 0) as wins,
                COUNTIF(return_pct <= 0) as losses,
                AVG(return_pct) as avg_return,
                COUNTIF(beat_benchmark) as beat_benchmark_count
            FROM `{self.outcomes_table}`
        """
        rows = list(self.client.query(query).result())
        if rows:
            row = dict(rows[0])
            total = row.get("total_recommendations", 0)
            row["win_rate"] = row["wins"] / total if total > 0 else 0
            row["benchmark_beat_rate"] = row["beat_benchmark_count"] / total if total > 0 else 0
            return row
        return {"total_recommendations": 0, "win_rate": 0, "avg_return": 0}
