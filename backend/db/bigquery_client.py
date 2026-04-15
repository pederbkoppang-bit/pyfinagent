"""
BigQuery client wrapper for report persistence and outcome tracking.
"""

import json
import logging
import time
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
                    # ── Phase 10: Model tracking ──
                    standard_model: str = "",
                    deep_think_model: str = "",
                    # ── Phase 11: Autoresearch – FEATURE_TO_AGENT bridge ──
                    consumer_sentiment: Optional[float] = None,
                    revenue_growth_yoy: Optional[float] = None,
                    quality_score: Optional[float] = None,
                    momentum_6m: Optional[float] = None,
                    rsi_14: Optional[float] = None,
                    # ── Phase 11: Autoresearch – enrichment signal parity ──
                    alt_data_signal: str = "",
                    alt_data_momentum_pct: Optional[float] = None,
                    anomaly_signal: str = "",
                    monte_carlo_signal: str = "",
                    quant_model_signal: str = "",
                    quant_model_score: Optional[float] = None,
                    social_sentiment_velocity: Optional[float] = None,
                    nlp_sentiment_confidence: Optional[float] = None,
                    # ── Phase 11: Autoresearch – risk assessment parity ──
                    risk_level: str = "",
                    recommended_position_pct: Optional[float] = None,
                    neutral_analyst_confidence: Optional[float] = None,
                    risk_debate_rounds_count: Optional[int] = None,
                    # ── Phase 11: Autoresearch – debate parity ──
                    groupthink_flag: Optional[bool] = None,
                    da_confidence_adjustment: Optional[float] = None,
                    # ── Phase 11: Autoresearch – cost parity ──
                    grounded_calls: Optional[int] = None,
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
            # ── Phase 10: Model tracking ──
            "standard_model": standard_model,
            "deep_think_model": deep_think_model,
            # ── Phase 11: Autoresearch – FEATURE_TO_AGENT bridge ──
            "consumer_sentiment": consumer_sentiment,
            "revenue_growth_yoy": revenue_growth_yoy,
            "quality_score": quality_score,
            "momentum_6m": momentum_6m,
            "rsi_14": rsi_14,
            # ── Phase 11: Autoresearch – enrichment signal parity ──
            "alt_data_signal": alt_data_signal,
            "alt_data_momentum_pct": alt_data_momentum_pct,
            "anomaly_signal": anomaly_signal,
            "monte_carlo_signal": monte_carlo_signal,
            "quant_model_signal": quant_model_signal,
            "quant_model_score": quant_model_score,
            "social_sentiment_velocity": social_sentiment_velocity,
            "nlp_sentiment_confidence": nlp_sentiment_confidence,
            # ── Phase 11: Autoresearch – risk assessment parity ──
            "risk_level": risk_level,
            "recommended_position_pct": recommended_position_pct,
            "neutral_analyst_confidence": neutral_analyst_confidence,
            "risk_debate_rounds_count": risk_debate_rounds_count,
            # ── Phase 11: Autoresearch – debate parity ──
            "groupthink_flag": groupthink_flag,
            "da_confidence_adjustment": da_confidence_adjustment,
            # ── Phase 11: Autoresearch – cost parity ──
            "grounded_calls": grounded_calls,
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

    def get_latest_report_json(self) -> Optional[dict]:
        """Get the ticker, analysis_date, and full_report_json from the most recent report.

        Single-query replacement for the two-call pattern (get_recent_reports + get_report).
        """
        query = f"""
            SELECT ticker, analysis_date, full_report_json
            FROM `{self.reports_table}`
            ORDER BY analysis_date DESC
            LIMIT 1
        """
        rows = list(self.client.query(query).result())
        if not rows:
            return None
        row = dict(rows[0])
        if row.get("full_report_json") and isinstance(row["full_report_json"], str):
            row["full_report_json"] = json.loads(row["full_report_json"])
        return row

    def get_report(self, ticker: str, analysis_date: Optional[str] = None) -> Optional[dict]:
        if analysis_date:
            # analysis_date column is TIMESTAMP in BQ — use a 1-second window
            # to match the exact record regardless of microsecond formatting
            from datetime import datetime, timedelta, timezone
            # Parse ISO 8601 string (handles both "Z" and "+00:00" suffixes)
            clean = analysis_date.replace("Z", "+00:00")
            try:
                ts = datetime.fromisoformat(clean)
            except ValueError:
                ts = None

            if ts:
                ts_start = ts - timedelta(seconds=1)
                ts_end = ts + timedelta(seconds=1)
                query = f"""
                    SELECT * FROM `{self.reports_table}`
                    WHERE ticker = @ticker
                      AND analysis_date BETWEEN @ts_start AND @ts_end
                    ORDER BY ABS(TIMESTAMP_DIFF(analysis_date, @ts_target, MICROSECOND))
                    LIMIT 1
                """
                job_config = bigquery.QueryJobConfig(query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                    bigquery.ScalarQueryParameter("ts_start", "TIMESTAMP", ts_start),
                    bigquery.ScalarQueryParameter("ts_end", "TIMESTAMP", ts_end),
                    bigquery.ScalarQueryParameter("ts_target", "TIMESTAMP", ts),
                ])
            else:
                # Fallback: treat as latest
                query = f"""
                    SELECT * FROM `{self.reports_table}`
                    WHERE ticker = @ticker
                    ORDER BY analysis_date DESC
                    LIMIT 1
                """
                job_config = bigquery.QueryJobConfig(query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
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
            SELECT ticker, analysis_date, total_tokens, total_cost_usd, deep_think_calls,
                   standard_model, deep_think_model
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

    # -- Signals log (Phase 4.2.4 durable persistence) ---------------

    def save_signal(self, record: dict) -> None:
        """Append a single signal-publish event to the signals_log table."""
        table = f"{self.settings.gcp_project_id}.{self.settings.bq_dataset_reports}.signals_log"
        errors = self.client.insert_rows_json(table, [record])
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")

    def query_latest_signal_state(self, signal_id: str) -> Optional[dict]:
        """Project the latest observed state for one signal_id from signals_log.

        Uses a QUALIFY ROW_NUMBER window function to pick the event with the
        largest recorded_at across publish + outcome (+ future revision) rows
        for the given signal_id. Returns the single-row dict, or None when
        the signal_id is absent from the table. Raises on BQ network / auth
        errors -- callers at the MCP boundary wrap this in a never-raise
        try/except and return None on failure.
        """
        table = f"{self.settings.gcp_project_id}.{self.settings.bq_dataset_reports}.signals_log"
        query = f"""
            SELECT *
            FROM `{table}`
            WHERE signal_id = @signal_id
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY signal_id ORDER BY recorded_at DESC
            ) = 1
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("signal_id", "STRING", signal_id),
        ])
        rows = list(self.client.query(query, job_config=job_config).result())
        if not rows:
            return None
        return dict(rows[0])

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

    # ── Paper Trading ────────────────────────────────────────────

    def _pt_table(self, name: str) -> str:
        return f"{self.settings.gcp_project_id}.{self.settings.bq_dataset_reports}.{name}"

    # -- Portfolio --

    def get_paper_portfolio(self, portfolio_id: str = "default") -> Optional[dict]:
        query = f"""
            SELECT * FROM `{self._pt_table("paper_portfolio")}`
            WHERE portfolio_id = @pid LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("pid", "STRING", portfolio_id),
        ])
        rows = list(self.client.query(query, job_config=job_config).result())
        return dict(rows[0]) if rows else None

    def _run_dml_with_retry(self, query: str, job_config, max_retries: int = 3) -> None:
        """Run a DML query with retry for BQ streaming buffer conflicts."""
        for attempt in range(max_retries + 1):
            try:
                self.client.query(query, job_config=job_config).result()
                return
            except Exception as e:
                if "streaming buffer" in str(e).lower() and attempt < max_retries:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    logger.warning(f"BQ streaming buffer conflict (attempt {attempt + 1}/{max_retries}), retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

    def upsert_paper_portfolio(self, row: dict) -> None:
        table = self._pt_table("paper_portfolio")
        pid = row["portfolio_id"]
        delete_query = f"DELETE FROM `{table}` WHERE portfolio_id = @pid"
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("pid", "STRING", pid),
        ])
        self._run_dml_with_retry(delete_query, job_config)
        # Use DML INSERT (not streaming) to avoid buffer conflicts
        cols = ", ".join(row.keys())
        vals = ", ".join(f"@v_{k}" for k in row.keys())
        insert_query = f"INSERT INTO `{table}` ({cols}) VALUES ({vals})"
        params = []
        for k, v in row.items():
            if isinstance(v, float):
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "FLOAT64", v))
            elif isinstance(v, int):
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "INT64", v))
            else:
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "STRING", str(v) if v is not None else None))
        insert_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(insert_query, job_config=insert_config).result()

    # -- Positions --

    def get_paper_positions(self) -> list[dict]:
        query = f"""
            SELECT * FROM `{self._pt_table("paper_positions")}`
            ORDER BY entry_date DESC
        """
        return [dict(r) for r in self.client.query(query).result()]

    def get_paper_position(self, ticker: str) -> Optional[dict]:
        query = f"""
            SELECT * FROM `{self._pt_table("paper_positions")}`
            WHERE ticker = @ticker LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        ])
        rows = list(self.client.query(query, job_config=job_config).result())
        return dict(rows[0]) if rows else None

    def save_paper_position(self, row: dict) -> None:
        """Insert position via DML (not streaming) to avoid buffer conflicts with UPDATE/DELETE."""
        table = self._pt_table("paper_positions")
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
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "STRING", str(v) if v is not None else None))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(query, job_config=job_config).result()

    def delete_paper_position(self, ticker: str) -> None:
        table = self._pt_table("paper_positions")
        query = f"DELETE FROM `{table}` WHERE ticker = @ticker"
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
        ])
        self._run_dml_with_retry(query, job_config)

    def update_paper_position(self, ticker: str, updates: dict) -> None:
        """Update specific fields of a paper position."""
        table = self._pt_table("paper_positions")
        set_clauses = ", ".join(f"{k} = @val_{k}" for k in updates)
        params = [bigquery.ScalarQueryParameter("ticker", "STRING", ticker)]
        for k, v in updates.items():
            if isinstance(v, float):
                params.append(bigquery.ScalarQueryParameter(f"val_{k}", "FLOAT64", v))
            elif isinstance(v, int):
                params.append(bigquery.ScalarQueryParameter(f"val_{k}", "INT64", v))
            else:
                params.append(bigquery.ScalarQueryParameter(f"val_{k}", "STRING", str(v)))
        query = f"UPDATE `{table}` SET {set_clauses} WHERE ticker = @ticker"
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self._run_dml_with_retry(query, job_config)

    # -- Trades --

    def save_paper_trade(self, row: dict) -> None:
        """Insert trade via DML to avoid streaming buffer conflicts."""
        table = self._pt_table("paper_trades")
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
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "STRING", str(v) if v is not None else None))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(query, job_config=job_config).result()

    def get_paper_trades(self, limit: int = 100) -> list[dict]:
        query = f"""
            SELECT * FROM `{self._pt_table("paper_trades")}`
            ORDER BY created_at DESC
            LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ])
        return [dict(r) for r in self.client.query(query, job_config=job_config).result()]

    # -- Snapshots --

    def save_paper_snapshot(self, row: dict) -> None:
        """Insert snapshot via DML to avoid streaming buffer conflicts."""
        table = self._pt_table("paper_portfolio_snapshots")
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
                params.append(bigquery.ScalarQueryParameter(f"v_{k}", "STRING", str(v) if v is not None else None))
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(query, job_config=job_config).result()

    def get_paper_snapshots(self, limit: int = 365) -> list[dict]:
        query = f"""
            SELECT * FROM `{self._pt_table("paper_portfolio_snapshots")}`
            ORDER BY snapshot_date DESC
            LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ])
        return [dict(r) for r in self.client.query(query, job_config=job_config).result()]
