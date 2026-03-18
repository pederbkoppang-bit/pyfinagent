"""
Analysis API routes — start analysis, check status, get results.

Supports two execution modes:
- Celery (USE_CELERY=true): dispatches to a Celery worker via Redis.
- Sync  (USE_CELERY=false): runs in a background asyncio task in-process.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.api.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    AnalysisStatusResponse,
    StepLogEntry,
    SynthesisReport,
)
from backend.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# ── In-memory task store (used when USE_CELERY=false) ────────────
_tasks: dict[str, dict[str, Any]] = {}


async def _run_sync_analysis(task_id: str, ticker: str, settings: Settings):
    """Run the full analysis pipeline in-process and update _tasks."""
    from backend.agents.orchestrator import AnalysisOrchestrator
    from backend.db.bigquery_client import BigQueryClient
    from backend.tools.slack import send_notification

    steps_completed: list[str] = []
    step_log: list[dict] = []

    def on_step(step_name: str, status: str, message: str = ""):
        ts = datetime.now(timezone.utc).isoformat()
        step_log.append({"step": step_name, "status": status, "message": message, "timestamp": ts})
        if status == "completed":
            steps_completed.append(step_name)
        _tasks[task_id].update(
            status=AnalysisStatus.RUNNING,
            current_step=step_name,
            step_status=status,
            message=message,
            step_log=list(step_log),
            steps_completed=list(steps_completed),
        )

    try:
        orchestrator = AnalysisOrchestrator(settings)
        report = await orchestrator.run_full_analysis(ticker, on_step=on_step)

        synthesis = report.get("final_synthesis", {})
        quant = report.get("quant", {})

        # Save to BigQuery (best-effort)
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
            # Also check direct fred enrichment data
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
            )
        except Exception as e:
            logger.error(f"Failed to save report to BigQuery: {e}")

        # Slack notification (best-effort)
        try:
            if settings.slack_webhook_url:
                await send_notification(
                    settings.slack_webhook_url,
                    f"Analysis Complete: {ticker}",
                    {
                        "Score": f"{synthesis.get('final_weighted_score', 'N/A')}/10",
                        "Verdict": synthesis.get("recommendation", {}).get("action", "N/A"),
                    },
                    "success",
                )
        except Exception:
            pass

        _tasks[task_id].update(
            status=AnalysisStatus.COMPLETED,
            steps_completed=list(steps_completed),
            result={"ticker": ticker, "final_synthesis": synthesis, "steps_completed": steps_completed},
        )

    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}", exc_info=True)
        _tasks[task_id].update(
            status=AnalysisStatus.FAILED,
            error=str(e),
        )


# ── Endpoints ────────────────────────────────────────────────────

@router.post("/", response_model=AnalysisResponse)
async def start_analysis(req: AnalysisRequest, settings: Settings = Depends(get_settings)):
    """Start a new analysis for the given ticker. Returns a task ID for polling."""
    ticker = req.ticker.upper()

    if settings.use_celery:
        from backend.tasks.analysis import run_analysis_task
        task = run_analysis_task.delay(ticker)
        task_id = task.id
    else:
        task_id = str(uuid.uuid4())
        _tasks[task_id] = {
            "ticker": ticker,
            "status": AnalysisStatus.PENDING,
            "current_step": "Queued",
            "steps_completed": [],
        }
        # Launch as a fire-and-forget asyncio task
        asyncio.create_task(_run_sync_analysis(task_id, ticker, settings))

    logger.info(f"Analysis started for {ticker}, task_id={task_id}")
    return AnalysisResponse(analysis_id=task_id, ticker=ticker, status=AnalysisStatus.PENDING)


@router.get("/{analysis_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(analysis_id: str, settings: Settings = Depends(get_settings)):
    """Poll for analysis status and progress. Returns final report when complete."""

    if settings.use_celery:
        return _poll_celery(analysis_id)

    # ── Sync / in-memory mode ──
    task = _tasks.get(analysis_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    status = task["status"]
    report_model = None

    if status == AnalysisStatus.COMPLETED:
        synthesis = task.get("result", {}).get("final_synthesis", {})
        try:
            report_model = SynthesisReport(**synthesis)
        except Exception:
            logger.warning("Could not parse synthesis into SynthesisReport model")

    return AnalysisStatusResponse(
        analysis_id=analysis_id,
        ticker=task.get("ticker", ""),
        status=status,
        current_step=task.get("current_step", ""),
        steps_completed=task.get("steps_completed", []),
        message=task.get("message", ""),
        step_log=[StepLogEntry(**e) for e in task.get("step_log", [])],
        report=report_model,
        error=task.get("error"),
    )


def _poll_celery(analysis_id: str) -> AnalysisStatusResponse:
    """Poll Celery AsyncResult for task status."""
    from celery.result import AsyncResult

    result = AsyncResult(analysis_id)

    if result.state == "PENDING":
        return AnalysisStatusResponse(
            analysis_id=analysis_id, ticker="", status=AnalysisStatus.PENDING,
            current_step="Queued", steps_completed=[],
        )
    if result.state == "PROGRESS":
        meta = result.info or {}
        return AnalysisStatusResponse(
            analysis_id=analysis_id, ticker="", status=AnalysisStatus.RUNNING,
            current_step=meta.get("current_step", ""),
            steps_completed=meta.get("steps_completed", []),
        )
    if result.state == "SUCCESS":
        data = result.result or {}
        synthesis = data.get("final_synthesis", {})
        report = None
        try:
            report = SynthesisReport(**synthesis)
        except Exception:
            logger.warning("Could not parse synthesis into SynthesisReport model")
        return AnalysisStatusResponse(
            analysis_id=analysis_id, ticker=data.get("ticker", ""),
            status=AnalysisStatus.COMPLETED,
            steps_completed=data.get("steps_completed", []),
            report=report,
        )
    # FAILURE or REVOKED
    return AnalysisStatusResponse(
        analysis_id=analysis_id, ticker="", status=AnalysisStatus.FAILED,
        error=str(result.info) if result.info else "Unknown error",
    )
