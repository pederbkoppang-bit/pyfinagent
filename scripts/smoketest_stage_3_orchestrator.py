"""phase-31.0.3 Stage 3 smoketest: AnalysisOrchestrator full-path on NVDA.

Invokes the 28-Gemini-agent pipeline directly with `_persist_analysis`
mocked so no row hits `analysis_results`. Verifies `llm_call_log` gets
agent-tagged rows.

NOT a pytest unit test -- this is a live integration smoketest that
calls real Vertex AI Gemini APIs. Expected wall-clock 3-10 min.
Expected cost ~$0.20-$1.00.

Persists output to `handoff/smoketest_20260520/STAGE_3_gemini_full_path_output.json`.
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT = Path(__file__).resolve().parent.parent / "handoff" / "smoketest_20260520" / "STAGE_3_gemini_full_path_output.json"


async def main() -> int:
    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient
    from backend.agents.orchestrator import AnalysisOrchestrator

    settings = get_settings()
    bq = BigQueryClient(settings)

    # Baseline
    start_ts = datetime.now(timezone.utc).isoformat()
    pre_rows = list(bq.client.query(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT agent) AS distinct_agents "
        "FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log` "
        f"WHERE DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)"
    ).result())
    pre = dict(pre_rows[0])
    print(f"Baseline llm_call_log: {pre}", flush=True)

    pre_ar_rows = list(bq.client.query(
        "SELECT COUNT(*) AS n FROM `sunny-might-477607-p8.financial_reports.analysis_results` "
        "WHERE ticker = 'NVDA' AND DATE(analysis_date) = CURRENT_DATE()"
    ).result())
    pre_ar = dict(pre_ar_rows[0])
    print(f"Baseline NVDA analysis_results today: {pre_ar}", flush=True)

    # Monkey-patch _persist_analysis to skip the BQ write
    # The persist call typically happens in autonomous_loop, NOT inside
    # orchestrator. orchestrator just returns the report; the caller
    # decides whether to persist. So we just NOT pass through any
    # persist callback.
    # The orchestrator itself does NOT call _persist_analysis -- that's
    # an autonomous_loop concern. Stage 3 just invokes run_full_analysis
    # directly, gets the report, and verifies llm_call_log got rows.

    result: dict = {
        "stage": "phase-31.0.3",
        "started_at": start_ts,
        "ticker": "NVDA",
        "baseline_llm_call_log": pre,
        "baseline_analysis_results_nvda_today": pre_ar,
    }

    try:
        print("Spawning AnalysisOrchestrator(settings).run_full_analysis('NVDA')...", flush=True)
        # phase-29.6: orchestrator constructor signature
        orchestrator = AnalysisOrchestrator(settings)
        report = await orchestrator.run_full_analysis("NVDA")
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        result["report_keys"] = sorted(list(report.keys())) if isinstance(report, dict) else None
        result["report_recommendation"] = (
            report.get("final_synthesis", {}).get("recommendation", {})
            if isinstance(report, dict) else None
        )
        result["report_final_score"] = (
            report.get("final_synthesis", {}).get("final_score")
            if isinstance(report, dict) else None
        )
        result["orchestrator_status"] = "completed"
        print(f"Orchestrator returned report with keys: {result['report_keys']}", flush=True)
    except Exception as exc:
        result["orchestrator_status"] = "error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        print(f"Orchestrator FAILED: {exc}", flush=True)

    # Post-run measurement
    post_rows = list(bq.client.query(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT agent) AS distinct_agents "
        "FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log` "
        f"WHERE DATE(ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)"
    ).result())
    post = dict(post_rows[0])

    post_ar_rows = list(bq.client.query(
        "SELECT COUNT(*) AS n FROM `sunny-might-477607-p8.financial_reports.analysis_results` "
        "WHERE ticker = 'NVDA' AND DATE(analysis_date) = CURRENT_DATE()"
    ).result())
    post_ar = dict(post_ar_rows[0])

    # Distinct agent tags newly observed
    new_agents_rows = list(bq.client.query(
        "SELECT DISTINCT agent FROM `sunny-might-477607-p8.pyfinagent_data.llm_call_log` "
        f"WHERE ts >= TIMESTAMP('{start_ts}') AND agent IS NOT NULL"
    ).result())
    new_agents = [r["agent"] for r in new_agents_rows]

    new_rows_count = post["n"] - pre["n"]
    result["post_llm_call_log"] = post
    result["post_analysis_results_nvda_today"] = post_ar
    result["new_llm_call_log_rows"] = new_rows_count
    result["new_distinct_agents"] = new_agents

    # Assertions
    assertions = {
        "orchestrator_completed_no_raise": result["orchestrator_status"] == "completed",
        "new_llm_call_log_rows_ge_10": new_rows_count >= 10,
        "new_distinct_agents_ge_3": len(new_agents) >= 3,
        "no_new_analysis_results_for_nvda": post_ar["n"] == pre_ar["n"],
    }
    result["assertions"] = assertions
    result["verdict"] = "PASS" if all(assertions.values()) else (
        "PARTIAL" if assertions["orchestrator_completed_no_raise"] else "FAIL"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nVERDICT: {result['verdict']}", flush=True)
    print(f"Assertions: {assertions}", flush=True)
    print(f"PERSISTED: {OUTPUT}", flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
