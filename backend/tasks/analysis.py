"""
Celery app configuration and async analysis task.

Settings are loaded lazily so the module can be imported without requiring
all environment variables to be present (e.g. during FastAPI startup checks).
"""

import asyncio
import json
import logging
import os

from celery import Celery

logger = logging.getLogger(__name__)

# Create Celery app with env-based defaults; the worker will pick up real
# values from the environment at runtime.
_broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery("pyfinagent", broker=_broker, backend=_backend)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="run_analysis")
def run_analysis_task(self, ticker: str):
    """
    Celery task that runs the full analysis pipeline.
    Updates task state with step-by-step progress.
    """
    from backend.agents.orchestrator import AnalysisOrchestrator
    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient
    from backend.tools.slack import send_notification

    settings = get_settings()
    steps_completed = []

    def on_step(step_name, status, message):
        if status == "completed":
            steps_completed.append(step_name)
        self.update_state(
            state="PROGRESS",
            meta={
                "current_step": step_name,
                "step_status": status,
                "message": message,
                "steps_completed": steps_completed,
            },
        )

    try:
        orchestrator = AnalysisOrchestrator(settings)

        # Run the async orchestrator in a sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            report = loop.run_until_complete(orchestrator.run_full_analysis(ticker, on_step=on_step))
        finally:
            loop.close()

        # Save to BigQuery
        synthesis = report.get("final_synthesis", {})
        quant = report.get("quant", {})
        try:
            bq = BigQueryClient(settings)

            # Extract reasoning fields for expanded BQ columns
            scoring_matrix = synthesis.get("scoring_matrix", {})
            rec_obj = synthesis.get("recommendation", {})
            debate_result = synthesis.get("debate_result", {})
            enrichment_sigs = synthesis.get("enrichment_signals", {})

            # Build compact signal summary: {signal_name: signal_value}
            sig_summary = {}
            for sig_name, sig_data in enrichment_sigs.items():
                if isinstance(sig_data, dict):
                    sig_summary[sig_name] = sig_data.get("signal", "N/A")

            # Extract critic and bias info
            critic_text = ""
            if isinstance(synthesis.get("critic_review"), str):
                critic_text = synthesis["critic_review"][:2000]
            elif isinstance(synthesis.get("critic_review"), dict):
                critic_text = json.dumps(synthesis["critic_review"])[:2000]

            bias_text = ""
            bias_report = synthesis.get("bias_report", {})
            if isinstance(bias_report, dict) and bias_report.get("bias_flags"):
                bias_text = json.dumps(bias_report["bias_flags"])[:2000]
            elif isinstance(bias_report, str):
                bias_text = bias_report[:2000]

            # ── Phase 1: Financial fundamentals ──
            yf_data = quant.get("yf_data", {})
            valuation = yf_data.get("valuation", {}) if isinstance(yf_data, dict) else {}
            health = yf_data.get("health", {}) if isinstance(yf_data, dict) else {}

            # ── Phase 2: Risk metrics (Monte Carlo + Anomalies) ──
            risk_data = synthesis.get("risk_data", {})
            mc_raw = risk_data.get("monte_carlo", {}) if isinstance(risk_data, dict) else {}
            mc_6m = mc_raw.get("horizons", {}).get("6M", {}) if isinstance(mc_raw, dict) else {}
            anomalies_raw = risk_data.get("anomalies", {}) if isinstance(risk_data, dict) else {}
            anomaly_list = anomalies_raw.get("anomalies", []) if isinstance(anomalies_raw, dict) else []

            # ── Phase 3: Debate details ──
            bull_case = debate_result.get("bull_case", {}) if isinstance(debate_result, dict) else {}
            bear_case = debate_result.get("bear_case", {}) if isinstance(debate_result, dict) else {}
            contradictions = debate_result.get("contradictions", []) if isinstance(debate_result, dict) else []
            dissents = debate_result.get("dissent_registry", []) if isinstance(debate_result, dict) else []
            risks_list = synthesis.get("key_risks", [])

            # ── Phase 4: Individual enrichment signal data ──
            insider_raw = report.get("insider", {})
            insider_data_dict = insider_raw.get("data", {}) if isinstance(insider_raw, dict) else {}
            options_raw = report.get("options", {})
            options_data_dict = options_raw.get("data", {}) if isinstance(options_raw, dict) else {}
            social_raw = report.get("social_sentiment", {})
            social_data_dict = social_raw.get("data", {}) if isinstance(social_raw, dict) else {}
            nlp_raw = report.get("nlp_sentiment", {})
            nlp_data_dict = nlp_raw.get("data", {}) if isinstance(nlp_raw, dict) else {}
            earnings_raw = report.get("earnings_tone", {})
            earnings_data_dict = earnings_raw.get("data", {}) if isinstance(earnings_raw, dict) else {}

            # ── Phase 5: Bias & conflict audit ──
            conflict_report = synthesis.get("conflict_report", {})
            if not isinstance(conflict_report, dict):
                conflict_report = {}
            if not isinstance(bias_report, dict):
                bias_report = {}

            # ── Phase 6: Macro context (FRED indicators) ──
            fred_indicators = {}
            macro_data = report.get("macro", {})
            if isinstance(macro_data, dict):
                fred_indicators = macro_data.get("fred_data", {})
                if not isinstance(fred_indicators, dict):
                    fred_indicators = macro_data.get("indicators", {})
                    if not isinstance(fred_indicators, dict):
                        fred_indicators = {}
            fred_enrichment = risk_data.get("fred_macro", {}) if isinstance(risk_data, dict) else {}
            if not fred_indicators and isinstance(fred_enrichment, dict):
                fred_indicators = fred_enrichment.get("indicators", {})

            # ── Phase 7: Multi-round debate, DA, info-gap, risk assessment ──
            info_gap = synthesis.get("info_gap_report", {})
            if not isinstance(info_gap, dict):
                info_gap = {}
            risk_assessment = synthesis.get("risk_assessment", {})
            if not isinstance(risk_assessment, dict):
                risk_assessment = {}
            risk_judge = risk_assessment.get("judge", {})
            if not isinstance(risk_judge, dict):
                risk_judge = {}

            # ── Phase 8: Cost tracking ──
            cost_summary = synthesis.get("cost_summary", {})
            if not isinstance(cost_summary, dict):
                cost_summary = {}

            def _fred_val(series_id: str):
                entry = fred_indicators.get(series_id, {})
                return entry.get("current") if isinstance(entry, dict) else None

            # ── Phase 11: Autoresearch – additional enrichment data ──
            qm_raw = report.get("quant_model", {})
            qm_data_dict = qm_raw.get("data", {}) if isinstance(qm_raw, dict) else {}
            qm_features = qm_data_dict.get("data", {}).get("features", {}) if isinstance(qm_data_dict, dict) else {}
            if not isinstance(qm_features, dict):
                qm_features = {}

            alt_raw = report.get("alt_data", {})
            alt_data_dict = alt_raw.get("data", {}) if isinstance(alt_raw, dict) else {}
            if not isinstance(alt_data_dict, dict):
                alt_data_dict = {}

            anomaly_raw_tool = report.get("anomaly", {})
            anomaly_data_dict = anomaly_raw_tool.get("data", {}) if isinstance(anomaly_raw_tool, dict) else {}
            if not isinstance(anomaly_data_dict, dict):
                anomaly_data_dict = {}

            scenario_raw = report.get("scenario", {})
            scenario_data_dict = scenario_raw.get("data", {}) if isinstance(scenario_raw, dict) else {}
            if not isinstance(scenario_data_dict, dict):
                scenario_data_dict = {}

            da_result = debate_result.get("devils_advocate", {}) if isinstance(debate_result, dict) else {}
            if not isinstance(da_result, dict):
                da_result = {}

            neutral_analyst = risk_assessment.get("neutral", {}) if isinstance(risk_assessment, dict) else {}
            if not isinstance(neutral_analyst, dict):
                neutral_analyst = {}

            bq.save_report(
                ticker=ticker,
                company_name=quant.get("company_name", "N/A"),
                final_score=synthesis.get("final_weighted_score", 0),
                recommendation=rec_obj.get("action", "N/A") if isinstance(rec_obj, dict) else str(rec_obj),
                summary=synthesis.get("final_summary", ""),
                full_report=report,
                pillar_scores=scoring_matrix,
                recommendation_justification=rec_obj.get("justification", "") if isinstance(rec_obj, dict) else "",
                debate_consensus=debate_result.get("consensus", "") if isinstance(debate_result, dict) else "",
                debate_confidence=debate_result.get("consensus_confidence") if isinstance(debate_result, dict) else None,
                enrichment_signals_summary=sig_summary,
                critic_review=critic_text,
                bias_flags=bias_text,
                # Phase 1: Financial fundamentals
                price_at_analysis=valuation.get("Current Price") or mc_raw.get("current_price"),
                market_cap=valuation.get("Market Cap"),
                pe_ratio=valuation.get("P/E Ratio"),
                peg_ratio=valuation.get("PEG Ratio"),
                debt_equity=health.get("Debt/Equity Ratio"),
                sector=yf_data.get("sector", "") if isinstance(yf_data, dict) else "",
                industry=yf_data.get("industry", "") if isinstance(yf_data, dict) else "",
                # Phase 2: Risk metrics
                annualized_volatility=mc_raw.get("annualized_volatility") if isinstance(mc_raw, dict) else None,
                var_95_6m=mc_6m.get("var_95"),
                var_99_6m=mc_6m.get("var_99"),
                expected_shortfall_6m=mc_6m.get("expected_shortfall_95"),
                prob_positive_6m=mc_6m.get("prob_positive"),
                anomaly_count=len(anomaly_list) if isinstance(anomaly_list, list) else None,
                # Phase 3: Debate & reasoning
                bull_confidence=bull_case.get("confidence") if isinstance(bull_case, dict) else None,
                bear_confidence=bear_case.get("confidence") if isinstance(bear_case, dict) else None,
                bull_thesis=(bull_case.get("thesis", "") if isinstance(bull_case, dict) else "")[:2000],
                bear_thesis=(bear_case.get("thesis", "") if isinstance(bear_case, dict) else "")[:2000],
                contradiction_count=len(contradictions) if isinstance(contradictions, list) else None,
                dissent_count=len(dissents) if isinstance(dissents, list) else None,
                recommendation_confidence=rec_obj.get("confidence") if isinstance(rec_obj, dict) else None,
                key_risks=json.dumps(risks_list)[:2000] if risks_list else "",
                # Phase 4: Enrichment signals
                insider_signal=sig_summary.get("insider", ""),
                options_signal=sig_summary.get("options", ""),
                social_sentiment_score=social_data_dict.get("avg_sentiment") if isinstance(social_data_dict, dict) else None,
                nlp_sentiment_score=nlp_data_dict.get("aggregate_score") if isinstance(nlp_data_dict, dict) else None,
                patent_signal=sig_summary.get("patent", ""),
                earnings_confidence=earnings_data_dict.get("management_confidence") if isinstance(earnings_data_dict, dict) else None,
                sector_signal=sig_summary.get("sector", ""),
                # Phase 5: Bias & conflict audit
                bias_count=bias_report.get("bias_count"),
                bias_adjusted_score=bias_report.get("adjusted_score"),
                conflict_count=conflict_report.get("conflict_count"),
                overall_reliability=conflict_report.get("overall_reliability", ""),
                decision_trace_count=len(synthesis.get("decision_traces", [])),
                # Phase 6: Macro context
                fed_funds_rate=_fred_val("FEDFUNDS"),
                cpi_yoy=_fred_val("CPIAUCSL"),
                unemployment_rate=_fred_val("UNRATE"),
                yield_curve_spread=_fred_val("T10Y2Y"),
                # Phase 7: Multi-round debate, DA, info-gap, risk assessment
                debate_rounds_count=debate_result.get("total_rounds") if isinstance(debate_result, dict) else None,
                devils_advocate_challenges=len(debate_result.get("devils_advocate", {}).get("challenges", [])) if isinstance(debate_result, dict) else None,
                info_gap_count=len(info_gap.get("gaps", [])) if isinstance(info_gap, dict) else None,
                info_gap_resolved_count=len([g for g in info_gap.get("gaps", []) if g.get("status") == "SUFFICIENT"]) if isinstance(info_gap, dict) else None,
                data_quality_score=info_gap.get("data_quality_score") if isinstance(info_gap, dict) else None,
                risk_judge_decision=risk_judge.get("decision", "") if isinstance(risk_judge, dict) else "",
                risk_adjusted_confidence=risk_judge.get("risk_adjusted_confidence") if isinstance(risk_judge, dict) else None,
                aggressive_analyst_confidence=risk_assessment.get("aggressive", {}).get("confidence") if isinstance(risk_assessment, dict) else None,
                conservative_analyst_confidence=risk_assessment.get("conservative", {}).get("confidence") if isinstance(risk_assessment, dict) else None,
                # Phase 8: Cost tracking
                total_tokens=cost_summary.get("total_tokens") if isinstance(cost_summary, dict) else None,
                total_cost_usd=cost_summary.get("total_cost_usd") if isinstance(cost_summary, dict) else None,
                deep_think_calls=cost_summary.get("deep_think_calls") if isinstance(cost_summary, dict) else None,
                # Phase 9: Reflection loop + quality gates
                synthesis_iterations=synthesis.get("synthesis_iterations"),
                # Phase 10: Model tracking
                standard_model=cost_summary.get("standard_model", "") if isinstance(cost_summary, dict) else "",
                deep_think_model=cost_summary.get("deep_think_model", "") if isinstance(cost_summary, dict) else "",
                # Phase 11: Autoresearch – FEATURE_TO_AGENT bridge features
                consumer_sentiment=_fred_val("UMCSENT"),
                revenue_growth_yoy=qm_features.get("revenue_growth_yoy"),
                quality_score=qm_features.get("quality_score"),
                momentum_6m=qm_features.get("momentum_6m"),
                rsi_14=qm_features.get("rsi_14"),
                # Phase 11: Autoresearch – enrichment signal parity
                alt_data_signal=alt_data_dict.get("signal", ""),
                alt_data_momentum_pct=alt_data_dict.get("momentum_pct"),
                anomaly_signal=anomaly_data_dict.get("signal", ""),
                monte_carlo_signal=scenario_data_dict.get("signal", ""),
                quant_model_signal=qm_data_dict.get("signal", ""),
                quant_model_score=qm_data_dict.get("score"),
                social_sentiment_velocity=social_data_dict.get("sentiment_velocity"),
                nlp_sentiment_confidence=nlp_data_dict.get("confidence"),
                # Phase 11: Autoresearch – risk assessment parity
                risk_level=risk_judge.get("risk_level", "") if isinstance(risk_judge, dict) else "",
                recommended_position_pct=risk_judge.get("recommended_position_pct") if isinstance(risk_judge, dict) else None,
                neutral_analyst_confidence=neutral_analyst.get("confidence"),
                risk_debate_rounds_count=risk_assessment.get("total_risk_rounds") if isinstance(risk_assessment, dict) else None,
                # Phase 11: Autoresearch – debate parity
                groupthink_flag=da_result.get("groupthink_flag"),
                da_confidence_adjustment=da_result.get("confidence_adjustment"),
                # Phase 11: Autoresearch – cost parity
                grounded_calls=cost_summary.get("grounded_calls") if isinstance(cost_summary, dict) else None,
            )
            # Invalidate cached report lists so home/reports pages refresh
            from backend.services.api_cache import get_api_cache
            get_api_cache().invalidate("reports:*")
        except Exception as e:
            logger.error(f"Failed to save report to BigQuery: {e}")

        # Slack notification
        try:
            loop2 = asyncio.new_event_loop()
            loop2.run_until_complete(send_notification(
                settings.slack_webhook_url,
                f"✅ Analysis Complete: {ticker}",
                {
                    "Score": f"{synthesis.get('final_weighted_score', 'N/A')}/10",
                    "Verdict": synthesis.get("recommendation", {}).get("action", "N/A"),
                },
                "success",
            ))
            loop2.close()
        except Exception:
            pass

        return {
            "ticker": ticker,
            "status": "completed",
            "final_synthesis": synthesis,
            "steps_completed": steps_completed,
        }

    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}", exc_info=True)
        # Try to notify on failure
        try:
            loop3 = asyncio.new_event_loop()
            loop3.run_until_complete(send_notification(
                settings.slack_webhook_url,
                f"❌ Analysis Failed: {ticker}",
                {"Error": str(e)[:200]},
                "error",
            ))
            loop3.close()
        except Exception:
            pass
        raise
