# Sprint Contract — phase-24.5 — Slack Notifications + Operator Alerting Audit

**Cycle:** phase-24 cycle 4
**Date:** 2026-05-12
**Step ID:** 24.5
**Priority:** P0 (third and final P0 bucket; closes all operator-reported Slack bugs)
**Depends on:** 24.0 (charter — DONE)

## Research-gate summary

`gate_passed: true` (tier=complex). 5 sources read in full: Anthropic harness-design, Slack Alert block (April 2026), oneuptime monitoring + alert-dedup (Feb 2026), stockalarm trading alerts.

```json
{"tier":"complex","external_sources_read_in_full":5,"snippet_only_sources":10,"urls_collected":15,"recency_scan_performed":true,"internal_files_inspected":10,"gate_passed":true}
```

## Hypothesis

Digest builder queries the wrong BQ table for portfolio P&L. "Recent Analyses" lacks ticker dedup (5x SNDK). Morning digest scheduler uses wrong TZ. Trade/kill-switch/drawdown/error notifications are unimplemented.

**Researcher verdict: CONFIRMED with refinements.** Four discrete bugs (all with file:line anchors):

1. **Wrong endpoint:** `scheduler.py:235` calls `/api/portfolio/performance` (legacy in-memory, always empty post-restart). Correct: `/api/paper-trading/portfolio`.
2. **Wrong field key:** `formatters.py:322` reads `total_return_pct`; endpoints return `total_pnl_pct`. Double bug — wrong endpoint AND wrong key.
3. **No ticker dedup:** `bigquery_client.py:258-268` `ORDER BY analysis_date DESC LIMIT 5` returns 5x SNDK if SNDK was analyzed 5 times.
4. **`.env` config wrong (not code bug):** `MORNING_DIGEST_HOUR=14`, `EVENING_DIGEST_HOUR=23`. TZ code is correct (`scheduler.py:144` uses `ZoneInfo("America/New_York")`).
5. **Missing notifications:** infrastructure exists (`send_trading_escalation`, `format_escalation_alert` at `scheduler.py:369-423` + `formatters.py:624-686`) but zero call sites for trade-confirm / kill-switch / drawdown / cycle-completion / error-escalation. `pause_signals()` at `scheduler.py:353-366` does NOT call `send_trading_escalation()`.

## Success criteria (verbatim from masterplan)

1. findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_5_slack_notifications_findings_md
2. research_gate_envelope_present_with_gate_passed_true
3. external_sources_count_at_least_5
4. canonical_url_cited_verbatim_slack_bot
5. recency_scan_2024_2026_section_present
6. at_least_three_phase_25_candidate_steps_proposed
7. each_candidate_step_has_files_list_with_absolute_paths
8. each_candidate_step_has_draft_verification_command
9. harness_log_has_phase_24_24_5_cycle_entry
10. executive_summary_section_present
11. findings_documents_wrong_pnl_data_source_bug_with_file_line_anchor
12. findings_documents_5x_sndk_recent_analyses_bug_with_query_anchor
13. findings_audits_morning_digest_2pm_schedule_bug
14. findings_enumerates_missing_notification_types_trade_killswitch_drawdown_error
15. findings_audits_slash_commands_portfolio_analyze_report

**Verifier:** `source .venv/bin/activate && python3 tests/verify_phase_24_5.py`

## Plan steps

1. Write findings doc at `docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md`
2. Write experiment_results.md
3. Spawn Q/A
4. Append `## Cycle 45 -- 2026-05-12 -- phase=24.5 result=PASS` to harness_log.md
5. Write live_check_24.5.md
6. Flip masterplan 24.5 to done

## References

External: 5 read-in-full URLs above.
Internal: `backend/slack_bot/{app.py,scheduler.py,formatters.py,commands.py}`; `backend/slack_bot/jobs/cost_budget_watcher.py`, `_production_fns.py`; `backend/api/{portfolio.py,paper_trading.py}`; `backend/db/bigquery_client.py:258-268`; `backend/config/settings.py:199`.
