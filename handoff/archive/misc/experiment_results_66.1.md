# Experiment results -- 66.1 Restore the decision path (Cycle 68, 2026-07-06/07)

## What was built (commit `fix(rail): phase-66.1 cc_rail guard ...`, pushed)

1. **The zero-pages root cause, fixed.** `backend/services/alerting.py` DOES NOT EXIST;
   four in-cycle P1 sites imported it (`backend/services/autonomous_loop.py` rail-probe,
   conviction-degraded, degraded-scoring, fallback-rate sites) and the
   ModuleNotFoundError died inside each fail-open except -- every page of the
   06-15..07-06 outage was silently dropped. All four now import
   `backend.services.observability.alerting` (test-enforced: the suite asserts
   find_spec("backend.services.alerting") is None AND >=4 corrected imports).
2. **RailGuard** (`backend/agents/claude_code_client.py`): module-level, thread-safe,
   per-cycle state.
   - Probe gate (criterion 1): `rail_guard_reset(cycle_id)` at cycle start BEFORE the
     existing phase-56.2 probe; on probe failure the loop calls `rail_guard_disable`
     -> every `ClaudeCodeClient.generate_content` call returns an empty LLMResponse
     immediately: no subprocess spawn, no llm_call_log row (skips are cycle-level
     state, not per-call rows -- avoids re-creating the 06-17/18 row spam).
   - Circuit breaker (criterion 2): consecutive-failure counter beside the existing
     failure path; at `settings.claude_rail_breaker_threshold` (new bounded Field,
     default 20) the breaker opens and pages EXACTLY ONCE on the closed->open
     transition (caller-side latch -- P1s bypass the AlertDeduper by design,
     phase-62.7) via `raise_cron_alert_sync` -> `_bot_token_fallback`. The probe-gate
     path consumes the latch so a rail-down incident never double-pages.
   - Success resets the consecutive count; `rail_guard_reset` per cycle is the
     circuit-breaker window reset (probe doubles as the half-open check).
3. **Funnel observability** (feeds 66.2): `rail_skipped` / `breaker_tripped` stamped
   into the cycle summary and persisted via `record_cycle_end` (new kwargs,
   `backend/services/cycle_health.py`) into cycle_history.jsonl.
4. **Degraded-mode policy documented** (criterion 4):
   `docs/runbooks/claude-rail-degraded-mode.md` -- rail-down => HOLD (fail-safe,
   current behavior); Gemini fallback NOT implemented; any future implementation must
   be config-gated DEFAULT OFF + operator token.

## Verbatim verification output (immutable command)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_1_rail_guard.py -q
........                                                                 [100%]
8 passed in 0.25s
```
Regression: `pytest backend/tests/test_phase_60_4_observability.py
backend/tests/test_phase_62_4_sentinel.py -q` -> `25 passed, 1 warning`.

## Live drill (criterion 2 evidence)

Stub `claude` binary (exit 1, "drill 66.1: simulated auth failure (401)") placed first
on PATH; REAL client, REAL breaker, REAL paging chain:
```
resolved binary: .../scratchpad/stubbin/claude
drill_id=drill-66.1-1783377034
25 calls in 2.49s
breaker_tripped=True consecutive_failures=20 skipped_calls=5
alert bot-token fallback delivered=True source=claude_code_rail title='Claude Code rail breaker OPEN -- 20 consecutive failures; ...'
```
Exactly ONE P1 delivered; server-side read-back (62.8 doctrine) confirmed message
ts=1783377037.178179 in C0ANTGNNK8D; permalink:
https://pyfinagent.slack.com/archives/C0ANTGNNK8D/p1783377037178179

## Deploy (criterion 3 precondition)

Import smoke test IMPORT_OK -> commit pushed -> `launchctl kickstart -k
gui/$(id -u)/com.pyfinagent.backend` -> single uvicorn instance, new lstart
2026-07-07 00:31:39 (postdates the commit), `curl /docs` -> HTTP 200.
Criterion 3 (ok=true cc_rail rows from a SCHEDULED cycle) is WALL-CLOCK-GATED to the
next scheduled cycle, 2026-07-07 18:00 UTC. Expected verdict this cycle: CONDITIONAL
with criteria 1/2/4 closed; a fresh Q/A closes criterion 3 on the BQ evidence after
the cycle (sanctioned cycle-2 flow -- evidence will have changed).

## Honest disclosures

- **Drill side effects in prod telemetry:** the successful drill wrote 20
  ok=false rows labeled `cc_rail:drill_66_1` (cost 0). A FIRST drill attempt
  failed because `_resolve_claude_binary` prefers `shutil.which` over the
  CLAUDE_CODE_BINARY env override (docstring claims the reverse order -- latent
  doc/behavior mismatch, noted for the defect register, NOT fixed here: out of
  contract scope); that attempt invoked the REAL CLI ~14x with 8-10s timeouts,
  writing ~14 unlabeled/short-prompt ok=false cc_rail rows (cost 0, Max flat-fee).
  All rows are 2026-07-06T22:1x-22:3xZ and predate the deploy.
- The breaker counts only REAL attempted failures (subprocess attempted); probe-gated
  skips do not increment it. Skipped calls are counted separately
  (rail_guard_status.skipped_calls) and never write llm_call_log rows.
- `raise_cron_alert_sync` inside a running event loop is fire-and-forget
  (create_task, returns True optimistically) -- in the drill (no loop) it ran
  synchronously and returned the true delivery result.

## File list

backend/agents/claude_code_client.py (RailGuard + gate), backend/services/
autonomous_loop.py (4 import fixes + guard wiring + funnel stamps),
backend/services/cycle_health.py (2 kwargs + row fields), backend/config/settings.py
(claude_rail_breaker_threshold), backend/tests/test_phase_66_1_rail_guard.py (NEW, 8),
docs/runbooks/claude-rail-degraded-mode.md (NEW), handoff artifacts.
