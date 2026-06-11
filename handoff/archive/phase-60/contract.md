# Contract -- 60.4 Observability + ops residuals (AW-7, AW-1/AW-2 residuals, AW-10, hygiene)

**Step:** 60.4 (phase-60, P1, harness_required, depends_on 60.3 done). **Date:** 2026-06-11.
**AW basis:** 59.3 report §1-2,4,7 (handoff/archive/phase-59.3/59.3-harness-free-output.md).

## Research-gate summary (researcher a67122f8, tier=complex, gate_passed:true)

6 sources in full (OWASP logging cheat sheet, Python logging docs, Better Stack sensitive-data, healthchecks.io, LiteLLM budgets, FinOps-for-AI), 50 URLs, recency scan. Brief: handoff/current/research_brief.md.

- **C1 gap pinned:** `ClaudeCodeClient.generate_content` (claude_code_client.py:338-394) writes ZERO log_llm_call rows on success or error; agent/ticker labels already arrive via the `_role`/`_ticker` generation_config side-channel (SDK clients consume them at llm_client.py:1090/:1099, :1766/:1775) -- mirror 56.2's lite-path helper, no signature changes.
- **C2:** tickets created ONLY by ticket_ingestion.py:129/:184 into repo-root tickets.db (indexed created_at); the max-retries close (ticket_queue_processor.py:344-359) has a literal `TODO: Implement follow-up trigger` at :357 and never posts to the ticket's channel_id. Silence-alarm home: the slack-bot watchdog block (scheduler.py:549-583) with its state-transition spam gate, reading tickets.db DIRECTLY (not via backend HTTP).
- **C3:** naked `stock.info`/`stock.history` inside async at autonomous_loop.py:1906-1908 + :2191-2193 (post-60.3 lines); ONE to_thread helper preserving `info`/`hist` names keeps the 60.3 integrity wiring byte-identical. Busy-vs-down: `cycle_lock.inspect_lock()` (:62-81; lock handoff/.autonomous_loop.lock, 90-min TTL, pid-alive) is importable from the slack-bot process -- APPEND state to the alert text at scheduler.py:519-523, never suppress the alert.
- **C4:** the "$4.3262 > $0.50" is cost_tracker.check_budget (cost_tracker.py:275-280) called at orchestrator.py:2254-2256 vs `settings.max_analysis_cost_usd` (settings.py:203) whose description says "does not abort" BY DESIGN -- ENFORCE-vs-respec is genuinely operator-gated; the $25/day hard layer (llm_client.py:396) already bounds runaway. PEAD: the migration ALREADY EXISTS (scripts/migrations/add_calendar_events_schema.py:36-52, --dry-run, schema matches pead_signal.py:337-341) -- the gated choice is RUN it vs DISABLE the overlay. Meta-scorer: the 56.2 `meta_scorer_degraded` alert exists (autonomous_loop.py:744-754) but has ZERO digest/signals consumers -- surface it.
- **C5:** the FRED leak emitter is the **httpx library logger** (2,101 `api_key=` lines in the 397MB repo-root backend.log; URL built at fred_data.py:37). THE TRAP: a redaction filter must attach to root HANDLERS, not the root logger (descendant-logger records bypass logger-level filters -- Python docs). Gateway escalation quote pinned from 59.3.
- External consensus: handler-level redaction (OWASP/Better Stack/Python docs); dead-man's-switch period+grace (healthchecks.io); both ENFORCE (LiteLLM 429-block) and alert-first (FinOps Foundation) are literature-backed -- the operator picks.
- Test net collects 0 today; name tests `test_phase_60_4_*` with the -k terms embedded.

## Hypothesis

Making the working rail visible in llm_call_log, alarming on ingestion silence and ticket death, fixing the event-loop blockers, surfacing the meta-scorer fallback, and redacting secrets at the handler level closes the AW-7/AW-10/hygiene residuals -- with the two genuinely operator-owned decisions (cost ENFORCE-vs-respec; PEAD migrate-vs-disable) prepared as ready-to-execute options and recorded verbatim.

## Immutable success criteria (verbatim from .claude/masterplan.json step 60.4)

**Command:** `cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'cc_rail_log or ingestion_silence or ticket_failure or redact or 60_4' -q && test -f handoff/current/live_check_60.4.md`

1. "the Claude Code CLI rail writes llm_call_log rows (provider/model/agent-label/ticker/latency/token counts from the CLI envelope, cost tagged 0 flat-fee) so burn and firing audits see the working rail; proven by a BigQuery MCP row in pyfinagent_data.llm_call_log from a single live smoke invocation, plus a unit test on the writer"
2. "an inbound-ingestion silence alarm exists: if the operator channel produces zero ingested tickets for a configurable N days (default 7), a Slack alert fires (the 6-week 04-24->06-10 outage class becomes impossible to miss); AND a ticket closed on max-retries posts a failure notice to its channel instead of dying silently (the #5101 case); both covered by unit tests"
3. "event-loop hygiene + watchdog semantics: the sync yfinance calls in _run_claude_analysis/_run_gemini_analysis are wrapped in asyncio.to_thread (test asserts no direct .info/.history call on the loop), and the slack-bot watchdog alert distinguishes busy-vs-down by including cycle-in-progress state (from the cycle lock or /api/paper-trading/status) in the alert text"
4. "cost-budget decision recorded and enacted: the per-analysis budget either ENFORCES (abort/flag the analysis at breach) or its limit is re-specced with rationale -- an OPERATOR-GATED choice recorded verbatim in the live_check; the PEAD calendar_events dependency is fixed via an operator-gated migration under scripts/migrations/ OR the overlay is disabled with a logged rationale (silent daily 404s are a FAIL); the meta-scorer LLM leg is repaired or its fallback state is surfaced in the digest/signals instead of masquerading as conviction 10"
5. "secret hygiene: third-party API keys no longer appear in plaintext in backend.log request-URL lines (redaction at the logging layer, unit-tested with a synthetic key); the OpenClaw gateway anthropic auth repair is escalated as a one-line operator action in the live_check (out of repo scope), with the gateway error text quoted as the remediation pointer"

**live_check:** "REQUIRED -- BQ MCP llm_call_log row from the CC-rail smoke call, alarm/notification test transcripts, the operator's verbatim cost-budget decision, the migration-or-disable evidence for calendar_events, and the redaction test output."

### Operator-gate handling (recorded BEFORE GENERATE)

Criterion 4 embeds TWO operator-owned choices. GENERATE builds BOTH options ready-to-execute, then asks the operator in-session (AskUserQuestion; the operator answered one this morning). If no reply lands, the verbatim-decision sub-criterion cannot be satisfied -> 60.4 stays in-progress at a SOFT STOP with the crisp ask in cycle_block_summary.md (honest blocking beats fabricated decisions).

## Plan

1. **C1:** `_log_cc_call` writer inside claude_code_client (or reuse backend/services/observability/api_call_log.log_llm_call) wired into ClaudeCodeClient.generate_content success+error paths: provider anthropic, model, agent=_role, ticker=_ticker (generation_config side-channel), latency_ms, input/output tokens from the envelope, session_cost untouched (flat-fee 0 delta). Unit test with a fake envelope + captured writer. Live smoke: one ClaudeCodeClient.generate_content call live -> BQ MCP row.
2. **C2:** `check_ingestion_silence(db_path, n_days=7)` helper (reads MAX(created_at) from tickets.db) + wiring in the scheduler watchdog block with the existing state-transition gate; max-retries close posts a failure notice to the ticket's channel_id (replacing the :357 TODO) via the processor's existing Slack send path. Unit tests for both (synthetic sqlite + captured poster).
3. **C3:** `_fetch_yf_info_hist(ticker)` sync helper called via `await asyncio.to_thread(...)` in both analyzers (names preserved -> 60.3 wiring untouched); structural test asserts no naked `.info`/`stock.history(` inside the async bodies. Watchdog: import cycle_lock.inspect_lock, append `cycle: IN PROGRESS (started <ts>)` or `cycle: not running` to the alert text; unit test on the text builder.
4. **C4:** meta-scorer fallback surfaced: morning digest line when the latest cycle summary carried meta_scorer_degraded (data path researched: scheduler digest assembly) -- plus the two PREPARED operator options: (a) cost budget: ENFORCE (flag-gated abort at check_budget breach) vs RE-SPEC (raise max_analysis_cost_usd to the measured full-path envelope $1.08-4.06 with rationale); (b) PEAD: run add_calendar_events_schema.py (--dry-run output attached) vs disable the overlay flag with logged rationale. AskUserQuestion -> verbatim into live_check.
5. **C5:** `SecretRedactionFilter` (regex api_key=/token=/key= query-param classes) attached to ROOT HANDLERS in main.py setup_logging (the documented trap); unit test: synthetic key through a handler -> redacted. Gateway escalation line quoted in live_check.
6. live_check_60.4.md -> fresh Q/A -> log -> flip (or SOFT STOP if operator-gated sub-items lack replies).

## Do-no-harm

All additive observability; the only decision-path candidates (cost enforce; PEAD disable) ship ONLY as operator options; redaction filter touches log RECORDS not behavior; watchdog alerts append context, never suppress. No live flag flips.

## References

Brief source table; 56.2 precedents (_log_claude_code_call, meta_scorer_degraded alert, watchdog); healthchecks.io DMS; OWASP logging; LiteLLM/FinOps budget patterns.
