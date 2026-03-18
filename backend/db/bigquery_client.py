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
            credentials = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=["https://www.googleapis.com/auth/bigquery",
                        "https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            logger.warning("GCP_CREDENTIALS_JSON not set, falling back to Application Default Credentials")

        self.client = bigquery.Client(project=settings.gcp_project_id, credentials=credentials)
        self.reports_table = f"{settings.gcp_project_id}.{settings.bq_dataset_reports}.{settings.bq_table_reports}"
        self.outcomes_table = f"{settings.gcp_project_id}.{settings.bq_dataset_outcomes}.{settings.bq_table_outcomes}"

    # ── Reports ──────────────────────────────────────────────────────

    def save_report(self, ticker: str, company_name: str, final_score: float,
                    recommendation: str, summary: str, full_report: dict,
                    pillar_scores: Optional[dict] = None,
                    recommendation_justification: str = "",
                    debate_consensus: str = "",
                    debate_confidence: Optional[float] = None,
                    enrichment_signals_summary: Optional[dict] = None,
                    critic_review: str = "",
                    bias_flags: str = "",
                    # ── Phase 1: Financial fundamentals ──
                    price_at_analysis: Optional[float] = None,
                    market_cap: Optional[float] = None,
                    pe_ratio: Optional[float] = None,
                    peg_ratio: Optional[float] = None,
                    debt_equity: Optional[float] = None,
                    sector: str = "",
                    industry: str = "",
                    # ── Phase 2: Risk metrics ──
                    annualized_volatility: Optional[float] = None,
                    var_95_6m: Optional[float] = None,
                    var_99_6m: Optional[float] = None,
                    expected_shortfall_6m: Optional[float] = None,
                    prob_positive_6m: Optional[float] = None,
                    anomaly_count: Optional[int] = None,
                    # ── Phase 3: Debate & reasoning ──
                    bull_confidence: Optional[float] = None,
                    bear_confidence: Optional[float] = None,
                    bull_thesis: str = "",
                    bear_thesis: str = "",
                    contradiction_count: Optional[int] = None,
                    dissent_count: Optional[int] = None,
                    recommendation_confidence: Optional[float] = None,
                    key_risks: str = "",
                    # ── Phase 4: Enrichment signals ──
                    insider_signal: str = "",
                    options_signal: str = "",
                    social_sentiment_score: Optional[float] = None,
                    nlp_sentiment_score: Optional[float] = None,
                    patent_signal: str = "",
                    earnings_confidence: Optional[float] = None,
                    sector_signal: str = "",
                    # ── Phase 5: Bias & conflict audit ──
                    bias_count: Optional[int] = None,
                    bias_adjusted_score: Optional[float] = None,
                    conflict_count: Optional[int] = None,
                    overall_reliability: str = "",
                    decision_trace_count: Optional[int] = None,
                    # ── Phase 6: Macro context ──
                    fed_funds_rate: Optional[float] = None,
                    cpi_yoy: Optional[float] = None,
                    unemployment_rate: Optional[float] = None,
                    yield_curve_spread: Optional[float] = None,
                    # ── Phase 7: Multi-round debate, DA, info-gap, risk assessment ──
                    debate_rounds_count: Optional[int] = None,
                    devils_advocate_challenges: Optional[int] = None,
                    info_gap_count: Optional[int] = None,
                    info_gap_resolved_count: Optional[int] = None,
                    data_quality_score: Optional[float] = None,
                    risk_judge_decision: str = "",
                    risk_adjusted_confidence: Optional[float] = None,
                    aggressive_analyst_confidence: Optional[float] = None,
                    conservative_analyst_confidence: Optional[float] = None,
                    # ── Phase 8: Cost tracking ──
                    total_tokens: Optional[int] = None,
                    total_cost_usd: Optional[float] = None,
                    deep_think_calls: Optional[int] = None,
                    # ── Phase 9: Reflection loop + quality gates ──
                    synthesis_iterations: Optional[int] = None,
                    ) -> None:
        row = {
            "ticker": ticker,
            "company_name": company_name,
            "analysis_date": datetime.utcnow().isoformat(),
            "final_score": final_score,
            "recommendation": recommendation,
            "summary": summary,
            # ── Pillar scores ──
            "pillar_1_corporate": (pillar_scores or {}).get("pillar_1_corporate", 0.0),
            "pillar_2_industry": (pillar_scores or {}).get("pillar_2_industry", 0.0),
            "pillar_3_valuation": (pillar_scores or {}).get("pillar_3_valuation", 0.0),
            "pillar_4_sentiment": (pillar_scores or {}).get("pillar_4_sentiment", 0.0),
            "pillar_5_governance": (pillar_scores or {}).get("pillar_5_governance", 0.0),
            # ── Recommendation reasoning ──
            "recommendation_justification": recommendation_justification,
            # ── Debate results ──
            "debate_consensus": debate_consensus,
            "debate_confidence": debate_confidence,
            # ── Enrichment signal summary (JSON string of signal_name -> signal value) ──
            "enrichment_signals_summary": json.dumps(enrichment_signals_summary or {}),
            # ── Validation ──
            "critic_review": critic_review,
            "bias_flags": bias_flags,
            # ── Phase 1: Financial fundamentals ──
            "price_at_analysis": price_at_analysis,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "peg_ratio": peg_ratio,
            "debt_equity": debt_equity,
            "sector": sector,
            "industry": industry,
            # ── Phase 2: Risk metrics ──
            "annualized_volatility": annualized_volatility,
            "var_95_6m": var_95_6m,
            "var_99_6m": var_99_6m,
            "expected_shortfall_6m": expected_shortfall_6m,
            "prob_positive_6m": prob_positive_6m,
            "anomaly_count": anomaly_count,
            # ── Phase 3: Debate & reasoning ──
            "bull_confidence": bull_confidence,
            "bear_confidence": bear_confidence,
            "bull_thesis": bull_thesis,
            "bear_thesis": bear_thesis,
            "contradiction_count": contradiction_count,
            "dissent_count": dissent_count,
            "recommendation_confidence": recommendation_confidence,
            "key_risks": key_risks,
            # ── Phase 4: Enrichment signals ──
            "insider_signal": insider_signal,
            "options_signal": options_signal,
            "social_sentiment_score": social_sentiment_score,
            "nlp_sentiment_score": nlp_sentiment_score,
            "patent_signal": patent_signal,
            "earnings_confidence": earnings_confidence,
            "sector_signal": sector_signal,
            # ── Phase 5: Bias & conflict audit ──
            "bias_count": bias_count,
            "bias_adjusted_score": bias_adjusted_score,
            "conflict_count": conflict_count,
            "overall_reliability": overall_reliability,
            "decision_trace_count": decision_trace_count,
            # ── Phase 6: Macro context ──
            "fed_funds_rate": fed_funds_rate,
            "cpi_yoy": cpi_yoy,
            "unemployment_rate": unemployment_rate,
            "yield_curve_spread": yield_curve_spread,
            # ── Phase 7: Multi-round debate, DA, info-gap, risk assessment ──
            "debate_rounds_count": debate_rounds_count,
            "devils_advocate_challenges": devils_advocate_challenges,
            "info_gap_count": info_gap_count,
            "info_gap_resolved_count": info_gap_resolved_count,
            "data_quality_score": data_quality_score,
            "risk_judge_decision": risk_judge_decision,
            "risk_adjusted_confidence": risk_adjusted_confidence,
            "aggressive_analyst_confidence": aggressive_analyst_confidence,
            "conservative_analyst_confidence": conservative_analyst_confidence,
            # ── Phase 8: Cost tracking ──
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "deep_think_calls": deep_think_calls,
            # ── Phase 9: Reflection loop + quality gates ──
            "synthesis_iterations": synthesis_iterations,
            # ── Full report ──
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

    # ── Cost History ─────────────────────────────────────────────────

    def get_cost_history(self, limit: int = 50) -> list[dict]:
        query = f"""
            SELECT ticker, analysis_date, total_tokens, total_cost_usd, deep_think_calls
            FROM `{self.reports_table}`
            WHERE total_cost_usd IS NOT NULL
            ORDER BY analysis_date DESC
            LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        )
        rows = self.client.query(query, job_config=job_config).result()
        return [dict(row) for row in rows]

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

    # ── Agent Memories ───────────────────────────────────────────

    def save_agent_memory(self, agent_type: str, ticker: str,
                          situation: str, lesson: str):
        """Persist an agent memory (situation + lesson) to BigQuery."""
        row = {
            "agent_type": agent_type,
            "ticker": ticker,
            "situation": situation[:2000],
            "lesson": lesson[:1000],
            "created_at": datetime.utcnow().isoformat(),
        }
        table = f"{self.settings.gcp_project_id}.{self.settings.bq_dataset_reports}.agent_memories"
        errors = self.client.insert_rows_json(table, [row])
        if errors:
            logger.error(f"Memory insert errors: {errors}")

    def get_agent_memories(self, agent_type: str | None = None, limit: int = 200) -> list[dict]:
        """Retrieve agent memories from BigQuery."""
        table = f"{self.settings.gcp_project_id}.{self.settings.bq_dataset_reports}.agent_memories"
        try:
            where = f"WHERE agent_type = '{agent_type}'" if agent_type else ""
            query = f"""
                SELECT agent_type, ticker, situation, lesson, created_at
                FROM `{table}`
                {where}
                ORDER BY created_at DESC
                LIMIT {int(limit)}
            """
            rows = list(self.client.query(query).result())
            return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Could not load agent memories: {e}")
            return []
