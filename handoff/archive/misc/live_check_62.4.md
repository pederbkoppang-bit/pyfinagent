# live_check -- phase-62.4: guardrail/budget sentinel

Date: 2026-06-12. Status: COMPLETE.

Healthy (real BQ, verbatim): {"metered_llm_usd_today": 0.0, "baseline_usd": 8.0,
"kill_switch_paused": false, "flags_match_tokens": true, "ok": true, "gates_failed":
[], "warnings": []} exit=0

Tamper transcripts + iteration log: handoff/current/experiment_results_62.4.md
(metered 99 -> exit 1 metered_budget; unauthorized flag -> exit 1 flags_match_tokens
naming the flag; BQ fail -> exit 2 metered_source_unavailable; wrapper breach ->
PREFLIGHT_PROMPT=digest_only with the session.log downgrade line; healthy wrapper ->
PREFLIGHT_PROMPT=am). pytest: 8 offline + 1 requires_live, all green.

Live self-discoveries (the sentinel auditing its own environment on first run): BQ
schema mismatch vs brief (no rail column -> session_cost_usd SUM, flat-fee rows 0 by
design, verified via 7-day provider breakdown) + 2 pre-away operational flags
grandfathered with provenance (PAPER_TRADING_ENABLED, PAPER_USE_CLAUDE_CODE_ROUTE).
