# Research Brief — Step 75.7: Slack assistant streaming await-correctness + P0 pager integrity

**Tier:** complex (P0; async-correctness + LAST-RESORT safety pager). NOT audit-class.
**Status:** IN PROGRESS (write-first skeleton)
**Researcher:** Layer-3 Harness MAS Researcher
**Date:** 2026-07-23

---

## 0. Step summary (from masterplan / spawn prompt)

Four sub-items:
- (a) pysvc-01: `chat_stream`/`append`/`stop` are coroutines on live AsyncWebClient but called synchronously in `_stream_simple_response` (:177) and `_stream_complex_task_plan` (:203) → every non-DIRECT assistant message dies into broad except (:141-149). Fix = await them.
- (b) pysvc-02: agent fan-out at :260 blocks the single event loop with `concurrent.futures.as_completed`/`future.result` → replace with `asyncio.create_task(asyncio.to_thread(...))` + `asyncio.as_completed`.
- (c) gap1-06: `app_home.update_app_home`'s sync `_get_live_data` (2x httpx.get + 3x subprocess.run, ~41s worst case), `commands._read_status`, reaction-handler git push must run via `await asyncio.to_thread` (scheduler.py:487 idiom).
- (d) gap1-02: `scheduler.py:963` `send_trading_escalation`'s L2 iMessage leg (LAST-resort pager for 'Kill Switch Activated' P0s) never checks imsg exit code, logs 'sent' unconditionally → capture CompletedProcess, on returncode!=0 log ERROR+stderr AND post Slack fallback; move phone literal to settings. Non-overlap with queued 72.0.4 (autonomous_loop.py:360-398/902-958).

Immutable verification command:
`cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_slack_streaming.py -q`

---

## 1. Queries run (three-variant discipline)

_(to be filled)_

## 2. Source table

### Read in full (>=5 required; counts toward gate)
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|

## 3. Recency scan (last 2 years, 2024-2026)

_(to be filled)_

## 4. Internal evidence table (verbatim file:line quotes)

_(to be filled)_

## 5. Per-item verdicts (a)-(d)

_(to be filled)_

## 6. slack_sdk-version-confirmed await-shape for (a)

_(to be filled)_

## 7. 72.0.4 non-overlap proof for (d)

_(to be filled)_

## 8. PAGER ANALYSIS (mandatory)

_(to be filled)_

## 9. Caller list (every caller of the changed functions)

_(to be filled)_

## 10. Async-test-tooling finding

_(to be filled)_

## 11. Risks

_(to be filled)_

## 12. JSON envelope

_(to be filled)_
