# Contract -- masterplan step 75.7

**Step id**: `75.7` | **Name**: Audit75 S7 -- Slack assistant streaming await-correctness + P0 pager integrity
**Phase**: phase-75 | **Priority**: P0 | **Cycle**: 1 | **Date**: 2026-07-23
**Executor**: sonnet-4.6/high (executed this session)

---

## 1. Research gate

**PASSED.** Run `wf_cfc4bd83-679`, two legs. Envelope: `tier=complex`,
`external_sources_read_in_full=7`, `urls_collected=30`, `recency_scan_performed=true`,
`internal_files_inspected=15`, `gate_passed=true`. Brief:
`handoff/current/research_brief_75.7.md`. Grounded on **installed slack_sdk 3.41.0
introspection** (`iscoroutinefunction` True for chat_stream/append/stop), not docs alone.

### 1a. Corrections the gate forced (all stale anchors -- edit by semantic search, not line #)
- **(a)** helpers at `:186`/`:228` (not :177/:203); the catching `except` is `:176-183`
  (the step's `:141-149` is the classification try/except, wrong block). `chat_stream`
  returns a coroutine; the `AttributeError` fires when `.append()` is called on it.
  DIRECT messages (`say()`) are unaffected -- only non-DIRECT assistant messages die.
- **(b)** fan-out at `:292-341` (not :260).
- **(c) is OVER-COUNTED**: only TWO calls need `to_thread` -- `_get_live_data`
  (`app_home.py:364`) and `_read_status` (`commands.py:481`). **The reaction-handler git
  push (`commands.py:567`) is ALREADY `asyncio.to_thread`-wrapped since phase-75.2.1 --
  it must NOT be touched** (double-wrap regression). `app_home.py` had no top-level
  `import asyncio` (added); `commands.py:7` already has it.
- **(d)** `send_trading_escalation` def at `:915`, imsg call `:963`, phone `:955`. Settings
  had no escalation-phone field.
- **72.0.4 non-overlap PROVEN**: it anchors `autonomous_loop.py:360-398` (a P1 via Slack
  webhook) -- different file, severity, transport. The phone literal never appears in
  `autonomous_loop.py`. Zero overlap.

### 1b. Test-tooling decisive finding
`pytest-asyncio` is **NOT installed** (pytest 9.0.3; anyio present). Repo precedent =
`asyncio.run(coro)` inside a plain sync `def` test (`test_phase_75_2_slack_control_plane.py:227`).
**The un-awaited-coroutine RuntimeWarning fires at GC time (non-deterministic; pytest-asyncio
#184)** -- a test relying only on `-Werror` can silently PASS. So criterion 1 is anchored on
the **deterministic AttributeError** that a sync-called coroutine's `.append()` raises, with
`gc.collect()` inside `warnings.catch_warnings()` if the warning is also asserted. Filter is
**module-scoped**, never a global `pytest.ini error::RuntimeWarning` (would break unrelated
suite tests emitting benign RuntimeWarnings).

---

## 2. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

> 1. New backend/tests/test_phase_75_slack_streaming.py passes offline and drives both _stream_* helpers with a stub AsyncWebClient whose chat_stream/append/stop are async -- asserting completion with zero un-awaited-coroutine RuntimeWarnings (warnings filter error::RuntimeWarning)
> 2. Source/AST assert: _stream_complex_task_plan contains no concurrent.futures.as_completed and no future.result() call; agent fan-out goes through asyncio.to_thread/create_task with results awaited
> 3. Test (or AST assert) proves app_home._get_live_data, commands._read_status, and the reaction-handler push are invoked via asyncio.to_thread from their async handlers
> 4. Pager test: with subprocess.run stubbed to returncode=1, send_trading_escalation logs ERROR (not the success line) AND invokes the Slack fallback post; with returncode=0 the success path is unchanged
> 5. No phone-number literal remains in scheduler.py -- the escalation recipient resolves from a settings field (source scan)
> 6. python -m py_compile passes on streaming_integration.py, app_home.py, commands.py, scheduler.py

**Command**: `cd /Users/ford/.openclaw/workspace/pyfinagent && .venv/bin/python -m pytest backend/tests/test_phase_75_slack_streaming.py -q`
**live_check**: `handoff/current/live_check_75.7.md` -- verbatim command output (exit 0) +
`git diff --stat`. No UI surface (Slack bot backend); no live bot run, no real iMessage.

*Criterion 3 note: the reaction-handler push is ALREADY to_thread-wrapped (75.2.1); the
test asserts all three are dispatched via to_thread -- which is TRUE today for the push and
newly true for the other two. Not a change to the push.*

---

## 3. What was built (GENERATE -- done)

- **streaming_integration.py**: `import asyncio` hoisted to module top; removed
  `from concurrent.futures import ThreadPoolExecutor, as_completed`; awaited all 2
  `chat_stream` + 7 `append` + 2 `stop` streamer coroutines (safe -- the Python-list
  `.append` calls use different variable names: `initial_chunks`/`progress_chunks`/
  `synthesis_parts`). Fan-out rewritten: `_run_agent` returns a **3-tuple**
  `(agent_type, result, err)` so `await done` never raises;
  `tasks = [asyncio.create_task(asyncio.to_thread(_run_agent, at)) ...]`;
  `for done in asyncio.as_completed(tasks): agent_type, result, err = await done`. This
  preserves per-agent identity AND per-agent error isolation (the gate's flagged risk).
- **app_home.py**: `import asyncio` added; `_get_live_data()` -> `await asyncio.to_thread(_get_live_data)`.
- **commands.py**: `_read_status()` -> `await asyncio.to_thread(_read_status)`. Reaction
  push untouched (already wrapped).
- **settings.py**: new `escalation_phone_e164` field, default `+4794810537` (byte-identical).
- **scheduler.py**: pager captures the `CompletedProcess`; on `returncode != 0` logs ERROR
  (not the success line) AND posts a Slack fallback (`P0 iMessage pager FAILED: ...`); the
  **exception path ALSO posts the fallback** (imsg-missing/timeout is equally silent -- the
  gate's added risk); empty phone -> ERROR + fallback + skip; phone from settings.

## 4. Mutation matrix (mandatory)
M1 revert an awaited streamer call to sync -> criterion-1 test fails (AttributeError).
M2 restore the ThreadPoolExecutor/as_completed fan-out -> criterion-2 AST assert fails.
M3 revert `_get_live_data`/`_read_status` to bare sync -> criterion-3 assert fails.
M4 make the pager ignore `returncode` (log 'sent' unconditionally) -> criterion-4 test fails.
M5 restore a phone literal in scheduler.py -> criterion-5 scan fails.
M6 drop the exception-path Slack fallback -> a test for the imsg-missing case fails.

## 5. Risks (from the gate)
- Edit-by-stale-line-number would misedit -- done by semantic search instead.
- Global RuntimeWarning filter would break unrelated tests -- module-scoped only.
- GC-time warning non-determinism -- anchored on the deterministic AttributeError.
- Fan-out could lose per-agent card mapping/isolation -- 3-tuple preserves both.
- Reaction push already wrapped -- NOT re-touched.
- the literal `+4794810537` at 4 occurrences across 2 files (`backend/services/sla_monitor.py:20`, `backend/services/queue_notification.py:34/63/164`)
  are OUT of scope -> **queue as their own step**, do not silently expand 75.7.

## 6. References
`research_brief_75.7.md` (`wf_cfc4bd83-679`); Slack Bolt async + assistant-streaming docs;
installed slack_sdk 3.41.0; asyncio.to_thread/as_completed docs; `.claude/agents/qa.md` §4b.
