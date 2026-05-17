#!/usr/bin/env python3
"""One-off: add phase-27 to .claude/masterplan.json.

phase-27 is born from the 2026-05-16 pre-prod smoke audit (phase-4.9
companion artifact: docs/audits/smoke_test_preprod_2026-05-16.md).

Goal: the full 28-skill pipeline succeeds end-to-end on BOTH Gemini
(direct AI Studio API key) AND Claude (direct Anthropic API key). Lite
fallback is provider-aware. BQ persistence works for the standard
analysis_results columns the lite analyzer emits.

Each step is harness-required; verification commands are immutable.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

MP = Path(".claude/masterplan.json")
data = json.loads(MP.read_text(encoding="utf-8"))

# Guard against double-apply.
existing_ids = {p["id"] for p in data["phases"]}
if "phase-27" in existing_ids:
    print("phase-27 already present; nothing to do")
    raise SystemExit(0)

PHASE_27 = {
    "id": "phase-27",
    "name": "Multi-Provider Full-Path Pipeline (Gemini + Claude)",
    "status": "pending",
    "harness_required": True,
    "depends_on": ["phase-26"],
    "summary": (
        "Make the full 28-skill orchestrator pipeline succeed end-to-end on "
        "BOTH Gemini (direct AI Studio API key) AND Claude (direct Anthropic "
        "API key). Born from the 2026-05-16 smoke audit which surfaced 3 "
        "bugs hidden behind the GitHub-Models-routing failure: C1 Gemini "
        "null-response NoneType.strip(); C2 lite fallback hardcoded to Claude "
        "only; C3 Anthropic structured-output schema requires "
        "additionalProperties:false on every object type. Plus B-2 BQ schema "
        "drift (5 columns the lite analyzer emits but the table is missing). "
        "North-star: dynamic provider selection per cost+latency, so the live "
        "cycle can route to whichever provider currently offers the best "
        "$ / quality ratio."
    ),
    "description": (
        "Source bug report: docs/audits/smoke_test_preprod_2026-05-16.md. "
        "Each step has its own immutable verification command. Both providers "
        "must complete a full cycle and persist >= 14 of 15 analyses without "
        "lite-fallback for the step to PASS."
    ),
    "steps": [
        {
            "id": "27.0",
            "name": (
                "Research gate: provider parity — Anthropic structured output + "
                "Gemini response shape + dual-provider fallback patterns"
            ),
            "status": "pending",
            "harness_required": True,
            "priority": "P0",
            "depends_on_step": None,
            "audit_basis": (
                "Smoke audit 2026-05-16 §A B-1..B-7 + this-session findings "
                "C1/C2/C3. Each downstream fix needs canonical provider docs "
                "as ground truth — not LLM-recalled API shapes."
            ),
            "verification": {
                "command": (
                    "test -f handoff/current/research_brief.md && "
                    "grep -q 'gate_passed.*true' handoff/current/research_brief.md && "
                    "grep -cE 'https?://' handoff/current/research_brief.md | awk '$1>=10{exit 0} {exit 1}'"
                ),
                "success_criteria": [
                    "research_brief.md present in handoff/current/",
                    "json_envelope_shows_gate_passed_true",
                    "min_5_sources_fetched_in_full",
                    "min_10_urls_collected",
                    "recency_scan_section_present_for_2025_2026",
                    "covers_both_anthropic_strict_schema_and_gemini_null_response_behavior",
                    "covers_provider_fallback_patterns_canonical"
                ],
                "live_check": (
                    "research_brief.md cites Anthropic structured-output docs "
                    "URL, Gemini generateContent response shape docs URL, and "
                    "at least one third-party multi-provider fallback pattern"
                )
            },
            "retry_count": 0,
            "max_retries": 3
        },
        {
            "id": "27.1",
            "name": "Fix C3 — Anthropic structured-output schema additionalProperties:false",
            "status": "pending",
            "harness_required": True,
            "priority": "P0",
            "depends_on_step": "27.0",
            "audit_basis": (
                "Live observation 2026-05-16 cycle 756a19c7: every claude-direct "
                "structured-output call rejected with "
                "'output_config.format.schema: For object type, additionalProperties "
                "must be explicitly set to false'. Anthropic's strict-mode validator "
                "requires this on every nested object node."
            ),
            "verification": {
                "command": (
                    "source .venv/bin/activate && python -c \""
                    "from backend.agents.llm_client import _ensure_additional_properties_false; "
                    "s={'type':'object','properties':{'x':{'type':'object','properties':{'y':{'type':'string'}}}, "
                    "'arr':{'type':'array','items':{'type':'object','properties':{'z':{'type':'string'}}}}}}; "
                    "r=_ensure_additional_properties_false(s); "
                    "assert r['additionalProperties'] is False, 'top'; "
                    "assert r['properties']['x']['additionalProperties'] is False, 'nested'; "
                    "assert r['properties']['arr']['items']['additionalProperties'] is False, 'array items'; "
                    "print('PASS')\""
                ),
                "success_criteria": [
                    "_ensure_additional_properties_false_helper_exists",
                    "recursive_application_on_nested_objects",
                    "recursive_application_on_array_items_of_object_type",
                    "applied_in_ClaudeClient_generate_content_before_send",
                    "unit_test_added_for_helper",
                    "live_claude_sonnet_call_with_structured_output_returns_200"
                ],
                "live_check": (
                    "after fix, one claude-sonnet-4-6 generate call with a "
                    "nested-object response_format schema returns HTTP 200 "
                    "(captured via curl -i or logged Anthropic request_id)"
                )
            },
            "retry_count": 0,
            "max_retries": 3
        },
        {
            "id": "27.2",
            "name": "Fix C1 — Gemini null-response safety (.text never None downstream)",
            "status": "pending",
            "harness_required": True,
            "priority": "P0",
            "depends_on_step": "27.0",
            "audit_basis": (
                "Live observation 2026-05-16 cycle 7c... at 22:40:48: "
                "POST gemini-2.5-pro:generateContent returned HTTP 200, but "
                "the downstream orchestrator parser crashed with "
                "'NoneType' object has no attribute 'strip'. Gemini returns "
                "null .text on safety-filter blocks, MAX_TOKENS truncation, "
                "or empty candidates."
            ),
            "verification": {
                "command": (
                    "source .venv/bin/activate && python -c \""
                    "from backend.agents.llm_client import LLMResponse; "
                    "r=LLMResponse(text=None, input_tokens=0, output_tokens=0); "
                    "assert r.text == '' or r.text is None, 'text should be empty or None'; "
                    "from backend.agents.llm_client import safe_text; "
                    "assert safe_text(None) == '', 'safe_text(None)'; "
                    "assert safe_text('  hi  ').strip() == 'hi', 'safe_text str'; "
                    "print('PASS')\" && "
                    "! grep -rE '\\.text\\.strip\\(\\)' backend/agents/orchestrator.py | grep -v 'safe_text' | head -1"
                ),
                "success_criteria": [
                    "safe_text_helper_exists_in_llm_client",
                    "LLMResponse_text_coerced_to_str_at_construction_or_safe_text_used_everywhere",
                    "no_raw_response_text_strip_in_orchestrator_without_guard",
                    "no_raw_response_text_lower_in_orchestrator_without_guard",
                    "unit_test_added_for_None_response_path",
                    "live_gemini_call_with_null_text_doesnt_crash_pipeline"
                ],
                "live_check": (
                    "after fix, simulating a None .text response (or running "
                    "real cycle that previously crashed) does not raise "
                    "AttributeError; cycle proceeds to next ticker"
                )
            },
            "retry_count": 0,
            "max_retries": 3
        },
        {
            "id": "27.3",
            "name": "Fix C2 — Provider-aware lite fallback (Gemini lite + Claude lite)",
            "status": "pending",
            "harness_required": True,
            "priority": "P0",
            "depends_on_step": "27.0",
            "audit_basis": (
                "Live observation 2026-05-16: when standard model set to "
                "gemini-2.5-flash, the lite fallback refuses: 'standard model "
                "gemini-2.5-flash is not a Claude model; _run_claude_analysis "
                "is Claude-only'. This leaves the system with no safety net — "
                "if full path fails, nothing runs. Lite path must adapt to "
                "the selected provider."
            ),
            "verification": {
                "command": (
                    "source .venv/bin/activate && python -c \""
                    "from backend.services.autonomous_loop import _select_lite_analyzer; "
                    "g=_select_lite_analyzer('gemini-2.5-flash'); "
                    "c=_select_lite_analyzer('claude-sonnet-4-6'); "
                    "assert callable(g) and callable(c), 'both return callables'; "
                    "assert g is not c, 'distinct implementations'; "
                    "print('PASS')\""
                ),
                "success_criteria": [
                    "_select_lite_analyzer_helper_exists",
                    "gemini_branch_uses_genai_Client_direct",
                    "claude_branch_preserves_existing_run_claude_analysis",
                    "either_branch_returns_dict_with_required_keys_recommendation_risk_assessment_total_cost_usd",
                    "no_double_LLM_round_trip_in_either_lite_path",
                    "live_cycle_with_gemini_standard_completes_without_Both_full_and_lite_paths_failed_errors"
                ],
                "live_check": (
                    "run-now with standard=gemini-2.5-flash completes a cycle "
                    "without any 'Both full and lite paths failed' log lines"
                )
            },
            "retry_count": 0,
            "max_retries": 3
        },
        {
            "id": "27.4",
            "name": "Fix B-2 — BQ schema migration: add 5 missing analysis_results columns",
            "status": "pending",
            "harness_required": True,
            "priority": "P0",
            "depends_on_step": "27.0",
            "audit_basis": (
                "Live observation 2026-05-16: 14 of 15 lite-path persists "
                "failed with rotating 'no such field' errors for "
                "consumer_sentiment, revenue_growth_yoy, quality_score, "
                "momentum_6m, rsi_14. Python writer (BigQueryClient.save_report) "
                "emits these (Phase-11 Autoresearch FEATURE_TO_AGENT bridge); "
                "BQ table financial_reports.analysis_results is missing them. "
                "Partial Phase-11 migration."
            ),
            "verification": {
                "command": (
                    "source .venv/bin/activate && python -c \""
                    "from google.cloud import bigquery; "
                    "c=bigquery.Client(project='sunny-might-477607-p8'); "
                    "t=c.get_table('sunny-might-477607-p8.financial_reports.analysis_results'); "
                    "names={f.name for f in t.schema}; "
                    "missing={'consumer_sentiment','revenue_growth_yoy','quality_score','momentum_6m','rsi_14'} - names; "
                    "assert not missing, f'still missing: {missing}'; "
                    "print('PASS, all 5 columns present')\""
                ),
                "success_criteria": [
                    "idempotent_migration_script_in_scripts_migrations",
                    "all_5_columns_added_as_FLOAT64_NULLABLE",
                    "re-running_migration_is_noop",
                    "downstream_save_report_call_no_longer_raises_no_such_field"
                ],
                "live_check": (
                    "run-now after migration produces >= 1 row in "
                    "financial_reports.analysis_results with non-null values "
                    "for at least one of the 5 newly-added columns"
                )
            },
            "retry_count": 0,
            "max_retries": 3
        },
        {
            "id": "27.5",
            "name": "End-to-end smoke verify: full path on Gemini",
            "status": "pending",
            "harness_required": True,
            "priority": "P0",
            "depends_on_step": "27.4",
            "audit_basis": (
                "Goal: confirm 27.1+27.2+27.3+27.4 together unblock the full "
                "28-skill pipeline when standard=gemini-2.5-flash."
            ),
            "verification": {
                "command": (
                    "test -f handoff/current/live_check_27.5.md && "
                    "grep -q 'cycle_id' handoff/current/live_check_27.5.md && "
                    "grep -q 'lite_mode.*[Ff]alse' handoff/current/live_check_27.5.md && "
                    "grep -qE 'analyses_persisted.*1[4-9]|analyses_persisted.*2[0-9]' handoff/current/live_check_27.5.md"
                ),
                "success_criteria": [
                    "model_set_to_gemini-2.5-flash_via_settings_api",
                    "full_cycle_completed_status_completed",
                    "lite_mode_false_observed_in_step_3_log",
                    "zero_Full_orchestrator_failed_lines_for_the_cycle",
                    "min_14_of_15_analyses_persisted_to_BQ_analysis_results",
                    "OutcomeTracker_step_9_attempted_at_minimum_logged"
                ],
                "live_check": (
                    "handoff/current/live_check_27.5.md captures verbatim "
                    "cycle_id + step list + per-ticker analysis status + BQ "
                    "row count delta on analysis_results"
                )
            },
            "retry_count": 0,
            "max_retries": 3
        },
        {
            "id": "27.6",
            "name": "End-to-end smoke verify: full path on Claude",
            "status": "pending",
            "harness_required": True,
            "priority": "P0",
            "depends_on_step": "27.5",
            "audit_basis": (
                "Goal: confirm same fixes also work when standard=claude-sonnet-4-6. "
                "Closes the dual-provider requirement from the goal."
            ),
            "verification": {
                "command": (
                    "test -f handoff/current/live_check_27.6.md && "
                    "grep -q 'cycle_id' handoff/current/live_check_27.6.md && "
                    "grep -q 'lite_mode.*[Ff]alse' handoff/current/live_check_27.6.md && "
                    "grep -qE 'analyses_persisted.*1[4-9]|analyses_persisted.*2[0-9]' handoff/current/live_check_27.6.md"
                ),
                "success_criteria": [
                    "model_set_to_claude-sonnet-4-6_via_settings_api",
                    "full_cycle_completed_status_completed",
                    "lite_mode_false_observed_in_step_3_log",
                    "zero_Full_orchestrator_failed_lines_for_the_cycle",
                    "min_14_of_15_analyses_persisted_to_BQ_analysis_results",
                    "OutcomeTracker_step_9_attempted_at_minimum_logged"
                ],
                "live_check": (
                    "handoff/current/live_check_27.6.md captures verbatim "
                    "cycle_id + step list + per-ticker analysis status + BQ "
                    "row count delta on analysis_results"
                )
            },
            "retry_count": 0,
            "max_retries": 3
        }
    ]
}

data["phases"].append(PHASE_27)
data["updated_at"] = datetime.now(timezone.utc).isoformat()

MP.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"OK — phase-27 added with {len(PHASE_27['steps'])} steps")
