# live_check_60.4 -- Observability + ops residuals (AW-7, AW-1/AW-2 residuals, AW-10, hygiene)

**Step:** 60.4 (phase-60, P1). **Date:** 2026-06-11. **Burn:** one CC-rail smoke call ($0 flat-fee) + BQ reads + one DDL job. ~$0 metered.

## A. BQ MCP llm_call_log row from the CC-rail smoke (criterion 1; job_Z7DetBNE7MOWpesxVtesQoCo94Id)

Live smoke: `ClaudeCodeClient("claude-sonnet-4-6").generate_content("Reply with the single word OK.", generation_config={"_role": "cc_rail_smoke_60_4", "_ticker": "SMOKE"})` -> text "OK" -> `flush_llm()` -> 1 row:

| field | value |
|---|---|
| provider / model | anthropic / claude-sonnet-4-6 |
| agent | **cc_rail:cc_rail_smoke_60_4** (the side-channel `_role` label) |
| ticker | SMOKE |
| latency_ms | 44535.8 |
| input_tok / output_tok | 3482 / 15 (from the CLI envelope) |
| ok / request_id | true / 5cad6fbe-dcb2-4be1-a2fd-f543f80e4c78 (the CLI session id) |

The rail that did ALL the away-week deciding (zero rows 06-02..06-09) now writes on BOTH success and error paths; cost stays flat-fee (no session_cost delta). Unit tests: writer field mapping, error path, generate_content wiring.

## B. Alarm/notification test transcripts (criterion 2, verbatim)

```
$ python -m pytest backend/tests -k 'cc_rail_log or ingestion_silence or ticket_failure or redact or 60_4' -q
16 passed, 823 deselected, 1 warning in 2.24s     (exit 0)
$ python -m pytest backend/tests -q
821 passed, 12 skipped, 6 xfailed, 1 warning in 83.27s   (exit 0)
```

- Ingestion silence: `TicketsDB.get_last_ticket_age_days()` (real schema, synthetic rows: 10d -> 10.0, 1d -> under threshold, empty -> None) + watchdog wiring with the standard state-transition spam gate; threshold `settings.ticket_ingestion_silence_days` (default 7; the 04-24 -> 06-10 six-week class is caught in one week).
- Ticket failure notice (the #5101 case): the max-retries close now posts to the ticket's channel/thread (`send_ticket_failure_notification`; replaces the literal `TODO: Implement follow-up trigger` at the close site). Tests: channel+thread+text assertions; fail-open without a client.

## C. Event-loop + watchdog (criterion 3)

- Both lite analyzers fetch yfinance via `asyncio.to_thread(_fetch_yf_market_data, ...)`; structural test asserts no `stock.info` / `stock.history(` in either async body (and the 60.3 suite still passes -- names preserved).
- Watchdog unreachable-alerts now APPEND `_cycle_state_line()` from `cycle_lock.inspect_lock()`: "cycle IN PROGRESS ... likely BUSY, not down" vs "lockfile absent -- looks DOWN". Context never suppresses the alert. Tests: busy + down branches.

## D. Operator decisions (criterion 4 -- recorded VERBATIM, AskUserQuestion 2026-06-11)

> Cost budget: **"RE-SPEC to $5.00 (Recommended)"** -- ENACTED: `settings.max_analysis_cost_usd` 0.50 -> 5.00 with the rationale in the field description (the 0.50 limit predated the restored full pipeline; the breached number is cost-tracker NOMINAL pricing flat-fee CC-rail tokens at API rates; the real circuit breaker is the $25/day hard cap).
> PEAD: **"RUN the migration (Recommended)"** -- ENACTED: `scripts/migrations/add_calendar_events_schema.py` run; **the script itself had the latent bug that explains the whole 404 class**: `window` is a GoogleSQL RESERVED KEYWORD, the unquoted column made the DDL fail (`400 Syntax error: Expected ")" or "," but got keyword WINDOW`), so the table never existed. Fixed (backticked) and re-run: `OK: sunny-might-477607-p8.pyfinagent_data.calendar_events ready.` BQ-confirmed (job_rzlfO42XlosytAyjTygSdIKMFIdf, creation_time 2026-06-11). Table starts empty -- an empty result is honest; daily 404s end.

- Meta-scorer fallback surfaced: cycle ledger rows carry `meta_scorer_degraded` (from the 56.2 summary stamp) and the morning digest's cron-health line appends "*Meta-scorer:* DEGRADED last cycle -- conviction values were no-LLM fallbacks (raw composite scores), not LLM judgments" when the latest cycle degraded; healthy cycles render byte-identical. (Repair of the LLM leg itself = the metered-Anthropic-credit class, routed to the candidate list; the criterion's OR-branch chosen.)

## E. Secret hygiene (criterion 5)

- `SecretRedactionFilter` attached to the ROOT HANDLER in `setup_logging` (handler-level, NOT logger-level -- descendant-logger records like httpx's bypass logger filters per the Python docs; the away week left 2,101 plaintext `api_key=` lines in backend.log via the httpx logger). Unit tests: synthetic FRED-style URL redacted through a real handler (`api_key=***REDACTED***`, non-secret params untouched), token/access_token classes, short-value false-positive guard, never-drops-records.
- **OpenClaw gateway escalation (out of repo scope -- ONE-LINE OPERATOR ACTION):** the #ford-approvals gateway agent has had unresolvable Anthropic auth since 2026-04-09; error verbatim: `No API key found for provider "anthropic". Auth store: /Users/ford/.openclaw/agents/pyfinagent/agent/auth-profiles.json` (thrown at `/opt/homebrew/lib/node_modules/openclaw/dist/model-auth-CElc27BR.js:310`). Fix: add an anthropic key/profile to that auth store (or rebind the channel), then send a test "Approve".

## F. Residual notes

- The redaction filter sanitizes NEW log lines; the existing 397MB repo-root backend.log retains historical plaintext keys -- rotating/scrubbing it is an operator hygiene item (one-line: rotate the FRED key at stlouisfed.org and truncate the log).
- Restart required for the backend (redaction filter, cost re-spec, analyzers) and the slack bot (silence alarm, watchdog context, digest line) to take effect -- next restart picks them up; no urgency beyond normal cadence.
