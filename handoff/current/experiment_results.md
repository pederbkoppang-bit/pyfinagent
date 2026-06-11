# Experiment Results -- Step 60.4 (GENERATE)

**Step:** 60.4 -- Observability + ops residuals (AW-7, AW-1/AW-2 residuals, AW-10, hygiene). **Date:** 2026-06-11.

## What was built (per criterion)

1. **CC-rail llm_call_log writer:** `ClaudeCodeClient._log_cc_call` wired into `generate_content` success AND error paths (claude_code_client.py); agent/ticker from the orchestrator's `_role`/`_ticker` generation_config side-channel; latency measured around the invoke; tokens from the CLI envelope; request_id = CLI session id; flat-fee cost untouched. Live smoke -> BQ row (live_check §A). 3 unit tests.
2. **Ingestion silence + ticket failure:** `TicketsDB.get_last_ticket_age_days()` (fail-open) + watchdog wiring with the standard state-transition gate + `settings.ticket_ingestion_silence_days` (default 7); `QueueNotificationService.send_ticket_failure_notification` (slack channel/thread + imessage routes) called at the max-retries close, replacing the literal `TODO` (the #5101 silent death). 5 unit tests.
3. **Event-loop + watchdog:** `_fetch_yf_market_data` via `asyncio.to_thread` in BOTH lite analyzers (names preserved -- 60.3 wiring untouched, its 13 tests still pass); unused in-function `yfinance` imports removed; `_cycle_state_line()` (cycle_lock.inspect_lock) APPENDED to watchdog unreachable-alerts (busy-vs-down; never suppresses). 3 tests incl. a structural no-naked-yfinance assert.
4. **Operator-gated decisions (recorded verbatim in live_check §D):** cost budget RE-SPEC 0.50 -> 5.00 (settings description carries the verbatim decision + rationale); PEAD migration RUN -- which surfaced the ROOT CAUSE of the 404 class: the DDL's unquoted `window` column is a GoogleSQL reserved keyword, so the script had NEVER succeeded; fixed (backticked) and run, table BQ-confirmed. Meta-scorer fallback surfaced: `meta_scorer_degraded` persisted on cycle-ledger rows -> morning digest cron-health line appends a DEGRADED warning (healthy = byte-identical). 2 tests.
5. **Secret hygiene:** `backend/services/observability/log_redaction.py` (`redact_secrets` + `SecretRedactionFilter`) attached to the ROOT HANDLER in setup_logging (handler-level per the Python-docs descendant-logger trap the researcher flagged); gateway escalation quoted in live_check §E. 3 tests.

## Files changed

backend/agents/claude_code_client.py, backend/services/{queue_notification,ticket_queue_processor,cycle_health,autonomous_loop}.py, backend/services/observability/log_redaction.py (NEW), backend/db/tickets_db.py, backend/slack_bot/scheduler.py, backend/config/settings.py (2 fields + cost re-spec), backend/main.py, scripts/migrations/add_calendar_events_schema.py (reserved-keyword fix), backend/tests/test_phase_60_4_observability.py (NEW, 16 tests).

## Verification command output (verbatim)

```
$ python -m pytest backend/tests -k 'cc_rail_log or ingestion_silence or ticket_failure or redact or 60_4' -q
16 passed, 823 deselected, 1 warning in 2.24s     (exit 0)
$ test -f handoff/current/live_check_60.4.md -> OK
```
FULL suite: **821 passed, 12 skipped, 6 xfailed exit 0** (805 post-60.3 + 16 new).

## Live verification (live_check_60.4.md)

- CC-rail smoke -> llm_call_log BQ row (agent cc_rail:cc_rail_smoke_60_4, 3482/15 tok, ok=true).
- calendar_events table created + BQ-confirmed (creation_time 2026-06-11).
- Operator decisions verbatim: "RE-SPEC to $5.00 (Recommended)" / "RUN the migration (Recommended)" -- both ENACTED.
- Gateway escalation one-liner with the verbatim auth error.
- Restart note: backend + slack bot pick up the changes at next restart (no live restart forced by this step).

## Artifact shape

llm_call_log rows with `agent="cc_rail:<role>"`; Slack Ingestion Silence Alarm block; ticket-failure channel notice; watchdog alerts with a "Cycle state:" line; digest cron-health with an optional Meta-scorer DEGRADED line; `api_key=***REDACTED***` log lines.
