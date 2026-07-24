# Research Brief — Step 75.7: Slack assistant streaming await-correctness + P0 pager integrity

**Tier:** complex (P0; async-correctness + LAST-RESORT safety pager). NOT audit-class.
**Status:** COMPLETE — gate_passed: true (7 sources read in full, recency scan done)
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

- Current-year frontier (2026): `Slack Bolt Python async chat_stream assistant streaming chat.appendStream 2026`
- Last-2-year window (2025): `asyncio.to_thread run blocking code in event loop 2025`
- Year-less canonical: `slack_sdk AsyncWebClient chat_stream await streaming assistant agents`; `python coroutine was never awaited RuntimeWarning pytest filterwarnings error test`
- Ground-truth (strongest evidence, beats any web doc): direct introspection of the INSTALLED `slack_sdk 3.41.0` in `.venv` (see §4 header) — this is the pinned version the executor's test runs against.

## 2. Source table

### Read in full (>=5 required; counts toward gate) — 7 read
| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| S1 | https://docs.slack.dev/reference/methods/chat.appendStream/ | 2026-07-23 | Official API doc | WebFetch | Streaming is 3 separate API calls: `chat.startStream` → `chat.appendStream` (repeatable) → `chat.stopStream`; Tier-4 rate limit; supports markdown/task/plan/block chunks. Grounds the start/append/stop lifecycle the code models. |
| S2 | https://docs.slack.dev/changelog/2025/10/7/chat-streaming/ | 2026-07-23 | Official changelog | WebFetch | Chat streaming introduced **Oct 7 2025**; Python + Node SDK helper utilities (`chat_stream`) added. Recency anchor. |
| S3 | https://docs.slack.dev/tools/bolt-python/concepts/adding-agent-features/ | 2026-07-23 | Official Bolt doc | WebFetch | Documents `say_stream`/`chat_stream` helpers with append/stop. NB: page shows **sync** examples (`streamer = say_stream(); streamer.append(...)`) — but that is the sync-Bolt surface; the async app uses `await` (see §4 installed-package proof). |
| S4 | https://docs.python.org/3/library/asyncio-dev.html | 2026-07-23 | Official Python doc | WebFetch | "Blocking … code should not be called directly … all concurrent asyncio Tasks and IO operations would be delayed"; use `run_in_executor`/thread. **"When a coroutine function is called, but not awaited (e.g. `coro()` instead of `await coro()`) … asyncio will emit a `RuntimeWarning`."** Grounds (b)(c) + criterion 1. |
| S5 | https://docs.python.org/3/library/asyncio-task.html | 2026-07-23 | Official Python doc | WebFetch | `asyncio.to_thread(func,…)` "Return a coroutine that can be awaited"; primarily for IO-bound blocking calls. `create_task` schedules a coro. `as_completed` iterated `async for x in as_completed(tasks): await x`. Grounds (b)(c) idioms exactly. |
| S6 | https://superfastpython.com/asyncio-coroutine-was-never-awaited/ | 2026-07-23 | Authoritative blog | WebFetch | "calling the coroutine function does not run the coroutine. Instead, it creates a coroutine object." Confirms the un-awaited-coroutine object semantics that make `.append()` on it an AttributeError. |
| S7 | https://github.com/pytest-dev/pytest-asyncio/issues/184 | 2026-07-23 | Official project issue | WebFetch | **Critical for test design:** the RuntimeWarning fires at **GC time (non-deterministic)**; plain `-Werror` can silently PASS; use `filterwarnings = error::RuntimeWarning` AND `gc.collect()` + `warnings.catch_warnings()` for reliability. Confirms the AttributeError should be the deterministic primary assertion. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://docs.slack.dev/reference/methods/chat.startStream/ | Official API | Lifecycle sibling of S1; covered by S1. |
| https://docs.slack.dev/tools/bolt-python/concepts/ai-apps/ | Official Bolt | Higher-level AI-apps overview; S3 more specific. |
| https://docs.slack.dev/tools/python-slack-sdk/reference/web/async_client.html | Official SDK ref | Superseded by direct introspection of the installed 3.41.0 package. |
| https://slack.dev/slack-developer-changelog-recap-april-june-2026/ | Official changelog | 2026 recap; recency corroboration only. |
| https://github.com/slackapi/bolt-js/issues/2696 | Community issue | JS, not Python; setStatus+stream interaction (tangential). |
| https://anyio.readthedocs.io/en/stable/threads.html | Official lib doc | anyio thread offload; project uses stdlib `asyncio.to_thread`. |
| https://dev.to/cbornet/introducing-blockbuster-...-3487 | Practitioner blog | BlockBuster event-loop-block detector (tooling idea; out of scope). |
| https://docs.python.org/3/library/asyncio-eventloop.html | Official Python | run_in_executor detail; S4/S5 sufficient. |
| https://github.com/pytest-dev/pytest/issues/10404 | Official project issue | `pytest.raises` swallows unawaited-coroutine warning (same GC caveat as S7). |
| https://www.geeksforgeeks.org/python/python-runtimewarning-coroutine-was-never-awaited/ | Community | Lower-tier; S6 covers it. |

URLs collected across searches: **~30 unique** (10 tables above + additional search hits). Read-in-full: 7. Snippet-only recorded: 10.

## 3. Recency scan (last 2 years, 2024-2026)

**Findings (material):**
1. **Slack chat streaming is NEW (introduced 2025-10-07, S2).** The `chat.startStream`/`appendStream`/`stopStream` methods and the `slack_sdk` `chat_stream` streamer helper did not exist before Oct 2025. The installed `slack_sdk 3.41.0` is a post-introduction build that ships both sync `ChatStream` and async `AsyncChatStream`. This is why the code is even able to import `slack_sdk.models.messages.chunk` — a 2025+ API. No breaking change to the append/stop shape found between 3.27 (repo floor) and 3.41.0 (installed); the async methods are coroutines in the installed build (measured).
2. **2026 changelog recap (Apr–Jun 2026, snippet)** notes continued streaming-block expansion (`feedback_buttons`/`icon_button`/`context_actions`) but no removal/rename of start/append/stop.
3. **asyncio guidance is stable (S4/S5, 2025 blogs).** `asyncio.to_thread` (3.9+) remains the recommended one-liner over `run_in_executor` for IO-bound blocking offload; no supersession.
4. **pytest un-awaited-coroutine detection (S7, ongoing):** the GC-timing non-determinism of the RuntimeWarning is a known, still-open limitation — the test must not rely on the warning alone.

**No finding that supersedes the fix direction.** The step's prescribed fixes (await the streamer methods; `asyncio.create_task(asyncio.to_thread(...))` + `as_completed`; `await asyncio.to_thread(sync_fn)`; capture `CompletedProcess`) are all consistent with the current (2025-2026) official guidance.

## 3b. External key findings (per-claim, cited)

1. Streaming lifecycle is 3 discrete API calls, each a coroutine on AsyncWebClient — "Use `chat.appendStream` to append text to a stream started with `chat.startStream` … stop the stream with `chat.stopStream`" (S1). Installed `AsyncChatStream.append`/`.stop` are coroutines (§4).
2. Calling a coroutine without await creates a coroutine object and emits a RuntimeWarning — "When a coroutine function is called, but not awaited (e.g. `coro()` instead of `await coro()`) … asyncio will emit a `RuntimeWarning`" (S4); "calling the coroutine function … creates a coroutine object" (S6). → `.append()` on that object is an AttributeError (deterministic, fires before the GC-time warning).
3. Blocking calls must be offloaded — "Blocking … code should not be called directly … all concurrent asyncio Tasks and IO operations would be delayed" (S4); `asyncio.to_thread` "Return a coroutine that can be awaited … primarily … for executing IO-bound functions/methods that would otherwise block the event loop" (S5). Grounds (c) `await asyncio.to_thread(_get_live_data)` / `_read_status`.
4. The (b) idiom is canonical — `asyncio.create_task(coro)` "schedule its execution"; `async for earliest in as_completed(tasks): … await earliest` (S5). So: `tasks=[asyncio.create_task(asyncio.to_thread(_run_agent, at)) for at in agents]; async for done in asyncio.as_completed(tasks): agent_type, result = await done`.
5. Test criterion 1 must anchor on the AttributeError, not only the warning — the RuntimeWarning "is generated at garbage collection time … non-deterministic … `-Werror` … silently passes the test"; use `filterwarnings = error::RuntimeWarning` + `gc.collect()` + `warnings.catch_warnings()` (S7).

## 4. Internal evidence table (verbatim file:line quotes)

**Environment ground truth (measured, not asserted):**
- `slack_sdk` **3.41.0**, `slack_bolt` **1.27.0** installed (satisfies `slack-sdk>=3.27.0`, `slack-bolt[async]>=1.18.0`). App is built with `AsyncApp` (`app.py:12,29`), so every listener receives an **`AsyncWebClient`** at runtime.
- `AsyncWebClient.chat_stream` → `iscoroutinefunction: True` (MUST be awaited). Returns `AsyncChatStream`.
- `AsyncChatStream.append` → `iscoroutinefunction: True`. `AsyncChatStream.stop` → `iscoroutinefunction: True`.
- Sync `WebClient.chat_stream/append/stop` → all `iscoroutinefunction: False` (the sync twin; NOT what runs here).
- Installed docstring example (verbatim from `.venv/.../slack_sdk/web/async_client.py`):
  `streamer = await client.chat_stream(...)` / `await streamer.append(markdown_text="...")` / `await streamer.stop()`.

| # | File:line | Verbatim | Meaning |
|---|-----------|----------|---------|
| E1 | streaming_integration.py:25 | `from slack_sdk import WebClient` | Type annotation only imports the SYNC class; runtime object is `AsyncWebClient` (misleading annotation, not the runtime type). |
| E2 | streaming_integration.py:212-217 | `streamer = client.chat_stream(channel=..., recipient_team_id=..., recipient_user_id=..., thread_ts=...)` | **No `await`.** On AsyncWebClient this returns an un-awaited coroutine. (step said :177 — actual **:212**) |
| E3 | streaming_integration.py:220 | `streamer.append(markdown_text=chunk)` | `.append` on a coroutine object → **AttributeError: 'coroutine' object has no attribute 'append'** (fires immediately, before any RuntimeWarning). |
| E4 | streaming_integration.py:224 | `streamer.stop()` | Never reached (dies at :220), but also un-awaited. |
| E5 | streaming_integration.py:238-244 | `streamer = client.chat_stream(..., task_display_mode="plan")` | Complex path; no `await`. (step said :203 — actual **:238**) |
| E6 | streaming_integration.py:261 | `streamer.append(chunks=initial_chunks)` | First `.append` in complex path → AttributeError. |
| E7 | streaming_integration.py:176-183 | `except Exception as e: logger.error(f"[FAIL] Streaming handler failed: {e}") ... await say("⚠️ Something went wrong ...")` | The broad except that ACTUALLY catches the AttributeError. (step said :141-149 — **WRONG**: :141-149 is the *classification* try/except `except Exception as cls_err`, unrelated.) |
| E8 | streaming_integration.py:21 | `from concurrent.futures import ThreadPoolExecutor, as_completed` | The blocking primitive (b) targets. |
| E9 | streaming_integration.py:292-341 | `with ThreadPoolExecutor(max_workers=len(agents)) as pool: futures = {pool.submit(_run_agent, at): at for at in agents}` then `for future in as_completed(futures):` … `_, result = future.result()` | Blocking fan-out. `as_completed`/`future.result()` block the single asyncio loop until each agent returns. (step said :260 — actual **:292 (submit) / :295 (as_completed) / :302 (result)**) |
| E10 | app_home.py:364 | `data = _get_live_data()` | Sync blocking call inside `async def update_app_home` (:357-358). NOT wrapped. |
| E11 | app_home.py:57,64,78,84,106 | `httpx.get(...,timeout=3)`×2 + `subprocess.run(...,timeout=10/15/10)`×3 | Worst case **3+3+10+15+10 = 41s** blocking the loop. CONFIRMED "~41s". |
| E12 | commands.py:481 | `status_text = _read_status()` | Sync blocking call inside `async def handle_any_message`. NOT wrapped. `_read_status` (:103-157) does file reads + `subprocess.check_output(git, timeout=5)` + `urllib.request.urlopen(timeout=5)`. |
| E13 | commands.py:567-572 | `result = await asyncio.to_thread(subprocess.check_output, ["git","push","origin","main"], ...)` | **The reaction-handler git push is ALREADY wrapped in `asyncio.to_thread`** (phase-75.2.1). Also :231 (`_pending_push_payload`) and :543 (`_resolve_head_sha`) already use to_thread. |
| E14 | scheduler.py:487 | `d["commits_today"] = await _asyncio.to_thread(_git_today)` | The `asyncio.to_thread` idiom the step cites for (c). CONFIRMED at **:487** (uses local `import asyncio as _asyncio` at :453). |
| E15 | scheduler.py:915 | `async def send_trading_escalation(app, severity, title, details, actions=None):` | The P0 pager fn. (step said :963 — that line is the `subprocess.run` INSIDE it; def is **:915**) |
| E16 | scheduler.py:954-955 | `if severity == "P0":` / `_ESCALATION_PHONE = "+4794810537"` | Hardcoded phone literal. CONFIRMED. |
| E17 | scheduler.py:961-969 | `try: import subprocess; subprocess.run(["imsg","send","--to",_ESCALATION_PHONE,"--text",imsg_text], capture_output=True, text=True, timeout=10,) logger.warning("iMessage escalation sent for %s: %s", severity, title) except Exception: logger.exception("Failed to send iMessage escalation")` | **THE BUG**: return value discarded; `subprocess.run` w/o `check=True` does NOT raise on non-zero exit → "sent" logged whenever `imsg` runs at all, even returncode=1. |

## 5. Per-item verdicts (a)-(d)

**(a) pysvc-01 — CONFIRMED (line anchors corrected).** On the runtime `AsyncWebClient` (proven: `AsyncApp`→`AsyncWebClient`; `chat_stream` is a coroutine in 3.41.0), the un-awaited `client.chat_stream(...)` at :212 / :238 yields a coroutine, and `.append()` at :220 / :261 raises `AttributeError: 'coroutine' object has no attribute 'append'`, caught by the broad except at **:176-183** (NOT :141-149). Every SIMPLE/MODERATE and COMPLEX assistant message dies there and the user gets "⚠️ Something went wrong". DIRECT replies (`say()` at :159) are unaffected — matches "every non-DIRECT message". **Corrections: helper lines are :186/:228 (not :177/:203); the sync calls are :212/:220/:224 and :238/:261/…/:374/:377; the catching except is :176-183 (the step's :141-149 is the wrong block).** Fix per installed docstring: `streamer = await client.chat_stream(...)`, `await streamer.append(...)`, `await streamer.stop()` at all sites (:212,:220,:224,:238,:261,:278,:315,:333,:343,:374,:377).

**(b) pysvc-02 — CONFIRMED (line anchor corrected).** The `ThreadPoolExecutor` + `as_completed(futures)` + `future.result()` block is at :292-341 (step said :260). `as_completed`/`future.result()` synchronously block the Socket-Mode event loop until agents finish. Idiomatic fix: `tasks=[asyncio.create_task(asyncio.to_thread(orchestrator.call_single_agent_sync, ...)) for at in agents]` then `for coro in asyncio.as_completed(tasks): agent_type, result = await coro`, preserving per-agent `TaskUpdateChunk` updates via `await streamer.append(...)`. NOTE: `_run_agent` currently returns `(agent_type, result)` so the completion-order mapping survives the port (keep that tuple so you can still resolve `meta`/`task_id` per completed task). Criterion 2 AST-asserts no `concurrent.futures.as_completed` and no `future.result()` remain in `_stream_complex_task_plan` — note the module-level `from concurrent.futures import ... as_completed` at :21 may also need removal or the AST scan must be scoped to the function body.

**(c) gap1-06 — PARTIAL (one of three sub-targets is already done).**
- `_get_live_data` (app_home.py:364) — CONFIRMED unwrapped; fix `data = await asyncio.to_thread(_get_live_data)`.
- `_read_status` (commands.py:481) — CONFIRMED unwrapped; fix `status_text = await asyncio.to_thread(_read_status)`.
- reaction-handler git push (commands.py:567-572) — **ALREADY WRAPPED** in `await asyncio.to_thread(...)` since phase-75.2.1 (E13). This sub-claim is already satisfied; the executor must NOT "re-fix" it. Criterion 3 asks to "prove … the reaction-handler push are invoked via asyncio.to_thread" — an assertion on the current code already PASSES. The genuinely-new work is the two above. **"measure, don't assert" flag: the step's (c) overstates the change surface by one item.**

**(d) gap1-02 — CONFIRMED.** scheduler.py:961-969 discards the `subprocess.run` CompletedProcess and logs "sent" unconditionally; phone literal `+4794810537` at :955. settings.py has **no** escalation-phone field today (must be added). Fix: `proc = subprocess.run([...settings.escalation_phone_e164...], capture_output=True, text=True, timeout=10)`; `if proc.returncode != 0: logger.error("iMessage pager FAILED rc=%s stderr=%s", proc.returncode, proc.stderr); await app.client.chat_postMessage(channel=settings.slack_channel_id, text=f"P0 iMessage pager FAILED: rc={proc.returncode} {proc.stderr[:200]}")`; else keep the existing success log. Keep the existing `except Exception` (covers `FileNotFoundError` if `imsg` is absent — that path should ALSO post the Slack fallback). See PAGER ANALYSIS §8.

## 6. slack_sdk-version-confirmed await-shape for (a)

Installed **slack_sdk 3.41.0**. `AsyncWebClient.chat_stream(*, buffer_size=256, channel, thread_ts, recipient_team_id=None, recipient_user_id=None, task_display_mode=None, **kwargs) -> AsyncChatStream` is `async def`. The three streaming methods map to Slack Web API `chat.startStream` / `chat.appendStream` / `chat.stopStream`. Correct await-shape (verbatim from the installed docstring):
```python
streamer = await client.chat_stream(channel="C…", thread_ts="…", recipient_team_id="T…", recipient_user_id="U…")
await streamer.append(markdown_text="…")   # or append(chunks=[…])
await streamer.stop()
```
`AsyncChatStream.append` and `.stop` are BOTH coroutines and must each be awaited (confirmed by introspection). The existing code awaits neither.

## 7. 72.0.4 non-overlap proof for (d)

| Axis | 75.7(d) | 72.0.4 |
|------|---------|--------|
| File | `scheduler.py:953-969` | `autonomous_loop.py:360-398` & `:902-958` |
| Severity | **P0** ("Kill Switch Activated") | **P1** (degraded-scoring / meta-scorer / rail-down) |
| Escalation layer | **L2 iMessage** (`imsg` CLI subprocess) | **L1 Slack** bot-token path (`raise_cron_alert` → `send_notification`) |
| Concern | imsg exit-code integrity (silent pager failure) | whether degraded-cycle P1s DELIVER to Slack at all |
| Verify cmd | `pytest test_phase_75_slack_streaming.py` | `grep -Eq "degraded|no-LLM" autonomous_loop.py` + delivered-Slack live_check |

**Dispositive proof of disjointness:** the phone literal `+4794810537` and any `imsg` invocation appear in `scheduler.py`, `services/sla_monitor.py`, `services/queue_notification.py` — but **NEVER in `autonomous_loop.py`** (grep confirmed). autonomous_loop.py's alert seam (read at :360-398) fires `raise_cron_alert(severity="P1", …)` over the async Slack webhook/bot-token path — it has no iMessage leg and no exit-code-unchecked subprocess. Therefore 72.0.4 (verify L1 Slack delivery of autonomous-loop degraded P1s) and 75.7(d) (fix the exit-code integrity of scheduler.py's L2 iMessage P0 pager) touch **different files, different severity tiers, different transports** — zero overlap. The step's own claim is CONFIRMED.

## 8. PAGER ANALYSIS (mandatory — LAST-RESORT P0 kill-switch pager)

**Escalation chain (`send_trading_escalation`, scheduler.py:915-969):**
- **L1 (all severities, :942-951):** `await app.client.chat_postMessage(...)` to `slack_channel_id`, then `logger.warning("Trading escalation posted to Slack…")`. Wrapped in its own try/except (Slack failure logged, does not abort L2).
- **L2 (P0 only, :954-969):** builds `imsg send --to +4794810537 --text …` and runs it via `subprocess.run(capture_output=True, timeout=10)`.

**Exact current behaviour when the iMessage subprocess "fails":**
1. `imsg` runs but exits non-zero (recipient unreachable, iMessage not signed in, number not registered on iMessage, transient send error) → `subprocess.run` (no `check=True`) returns a `CompletedProcess(returncode=1, stderr=…)` and **does not raise**. The return value is discarded. `logger.warning("iMessage escalation sent…")` fires. **Operator is told the pager succeeded when it did not.**
2. `imsg` binary absent (`FileNotFoundError`) or timeout (`TimeoutExpired`) → raises → caught by `except Exception` → `logger.exception("Failed to send iMessage escalation")`. This path at least logs a failure, but still posts NO Slack fallback line, so L1 does not record that L2 failed.

**Why "logs 'sent' unconditionally" is dangerous:** this is the LAST-RESORT pager on the operator's live money. It fires exactly for `notify_kill_switch_activated` → title `"Kill Switch Activated"` (scheduler.py:849-859) — i.e. the autonomous trader has hit a daily-loss / trailing-DD breach and halted. If the operator is away (the entire away-ops premise) and the iMessage silently fails at returncode=1, they are **never paged and never know**, while the log falsely reads "sent". The codebase has already been bitten by this exact bug-class once: `alerting.py:15-21` records that a prior version "called `send_trading_escalation` without `await` … so every alert raised TypeError into the fail-open except and was silently dropped." Silent alert-delivery failure is a recurring failure mode here, which is why exit-code capture + a Slack fallback (so L1 records the L2 failure) matters.

**Fix is fully testable OFFLINE (no real iMessage):** stub `subprocess.run` (monkeypatch `scheduler.subprocess.run`) to return a fake `CompletedProcess(args, returncode=1, stdout="", stderr="delivery failed")`; assert (i) `logger.error` fired (not the "sent" warning) and (ii) `app.client.chat_postMessage` was called with a body containing "P0 iMessage pager FAILED". Second case: stub returncode=0 → assert the success log fires and NO fallback post. `app` is a stub with an async `client.chat_postMessage`. No network, no `imsg`, no money touched. The pager fires only for P0, so the test must pass `severity="P0"`.

## 9. Caller list (every caller of the changed functions)

| Changed fn | Callers | Await-correct? | Signature change? |
|-----------|---------|----------------|-------------------|
| `handle_user_message_with_streaming` | `assistant_lifecycle.py:153` (`await`), `test_phase_75_2_slack_control_plane.py:227` (`asyncio.run`) | Yes | No — (a) edits are internal to the two helpers |
| `_stream_simple_response` | `streaming_integration.py:171` (`await`) — sole caller | Yes | No |
| `_stream_complex_task_plan` | `streaming_integration.py:164` (`await`) — sole caller | Yes | No |
| `_get_live_data` | `app_home.py:364` — sole caller | (currently sync) | No — stays sync; only the CALL is wrapped in `to_thread` |
| `_read_status` | `commands.py:481` — sole caller | (currently sync) | No — stays sync; call wrapped |
| `send_trading_escalation` | `scheduler.py:849` (`notify_kill_switch_activated`, `await`), `scheduler.py:906` (`notify_kill_switch_deactivated`, `await`) | Yes | No — (d) edits are internal to the L2 block; signature unchanged. `alerting.py` only NAMES it in a historical docstring; it does NOT call it. |

**Conclusion:** none of the four fixes changes a public signature; every existing caller already awaits the async ones. Adding `await` to the streamer calls, porting the fan-out to `asyncio.as_completed`, wrapping two sync calls in `to_thread`, and capturing the CompletedProcess are all internal-body changes → no caller breaks.

## 10. Async-test-tooling finding (decisive for the test design)

- **pytest 9.0.3** installed. **pytest-asyncio is NOT installed.** `anyio 4.13.0` IS installed AND registered as a `pytest11` plugin (`['pytest_cov','timeout','langsmith_plugin','anyio']`), so `@pytest.mark.anyio` would work (needs an `anyio_backend` fixture → `"asyncio"`). `trio` absent.
- **Established repo pattern = `asyncio.run(coro)` inside a plain sync `def test_…`** — used at `test_phase_75_2_slack_control_plane.py:227,263,265,287` and elsewhere. This needs NO plugin and is the lowest-risk choice. The new test SHOULD follow it (do NOT introduce `@pytest.mark.asyncio` — the marker exists only with pytest-asyncio, which is absent, so such tests would silently not-run/emit warnings).
- **`pytest.ini` (549 B)** registers only the `requires_live` marker — **no `filterwarnings`, no `asyncio_mode`.** Criterion 1 wants `error::RuntimeWarning`. Prefer a **module/function-scoped** filter (`pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")` or `with warnings.catch_warnings(): warnings.simplefilter("error", RuntimeWarning)`), NOT a global pytest.ini change (a global `error::RuntimeWarning` could fail unrelated suite tests that emit benign RuntimeWarnings). See §11 risks.
- **`conftest.py`** sets `PYFINAGENT_TEST_NO_BQ=1` at import → the new test inherits BQ isolation automatically (offline-safe).
- **Test-design note for criterion 1 (deterministic RuntimeWarning):** the un-awaited-coroutine `RuntimeWarning` is emitted at GC time (non-deterministic). To make the assertion deterministic, either (i) rely on the AttributeError path (stub `chat_stream` as `async def` returning a stub streamer with `async def append/stop`; if prod fails to await, `.append` on the coroutine raises immediately — the test asserts the handler completes WITHOUT that error), and/or (ii) force `gc.collect()` inside a `warnings.catch_warnings(record=True)`/`simplefilter("error")` block so a lingering un-awaited coroutine surfaces. A robust stub: `chat_stream` is `async def` (so a missing outer `await` leaves a coroutine), and its returned streamer records that `append`/`stop` were awaited (set a flag inside the `async def`), letting the test assert positive await-completion, not just absence of warning.
- **Existing precedent that the (a) bug is currently UNTESTED:** `test_deploy_request_refused_before_any_llm_call` (:206-231) calls the handler with `client=None` but a *deploy* message, which returns at :120-125 BEFORE any `chat_stream` — so no existing test exercises the streaming path. That is why the AttributeError ships silently.

## 11. Risks / executor pitfalls

1. **Stale line anchors in the step text (a)(b)(c)(d).** Every cited line is off: (a) helpers are :186/:228 not :177/:203; the sync streamer calls are :212/:220/:224 and :238/:261/:278/:315/:333/:343/:374/:377; the catching except is :176-183 not :141-149 (:141-149 is the classification block). (b) fan-out is :292-341 not :260. (d) `send_trading_escalation` def is :915, the imsg call :963. Edit by SEMANTIC search, not by line number.
2. **Misleading type annotation (a).** `client: WebClient` (streaming_integration.py:25, assistant_lifecycle.py:16, app_home.py:15) imports the SYNC class, but `AsyncApp` injects `AsyncWebClient`. Do NOT let the annotation argue the calls are sync. The test MUST stub an **async** client (`async def chat_stream` returning a streamer with `async def append`/`async def stop`).
3. **`import asyncio` missing at module top (c).** `app_home.py` has NO top-level `import asyncio` (only logging+WebClient at :14-15) — the executor MUST add it. `commands.py:7` already has it. `streaming_integration.py` only has LOCAL `import asyncio` at :221/:263 — hoist to module top since (a)/(b) now use `asyncio` throughout.
4. **(b) leftover `concurrent.futures` import.** `from concurrent.futures import ThreadPoolExecutor, as_completed` (:21) becomes unused after the port. Criterion 2 forbids `concurrent.futures.as_completed`/`future.result()` in `_stream_complex_task_plan`; a naive source-grep for `as_completed` would still hit :21. **Remove the unused import** (and scope any AST assert to the function body) so the check is clean and not gamed.
5. **(b) preserve per-agent card mapping + per-agent error isolation.** Keep `_run_agent` returning `(agent_type, result)` so completed tasks still resolve `meta`/`task_id`. `await done` re-raises that agent's exception — keep the per-agent `try/except` so one agent's failure renders its error `TaskUpdateChunk` and does not abort the fan-out (as_completed does not cancel siblings; but the await raises). `call_single_agent_sync` is a plain sync `def` (multi_agent_orchestrator.py:388) → correct `to_thread` target.
6. **(c) the reaction-handler git push is ALREADY wrapped (commands.py:567-572, phase-75.2.1).** Do NOT re-wrap. Only `_get_live_data` (app_home.py:364) and `_read_status` (commands.py:481) are new work. Criterion 3's assertion on the reaction push already passes on current code. The step's "3 sync calls" over-counts the change surface by one.
7. **(d) cover BOTH failure modes, not just returncode=1.** returncode!=0 (imsg ran, delivery failed) does NOT raise → must be caught by capturing the CompletedProcess. But `imsg` absent (`FileNotFoundError`) / `timeout` (`TimeoutExpired`) DO raise → the existing `except Exception` must ALSO post the Slack fallback line, else a missing `imsg` binary still fails L1-recording. Criterion only tests returncode=1; the safe fix covers the except path too.
8. **(d) Slack fallback transport.** Reuse `await app.client.chat_postMessage(channel=settings.slack_channel_id, text="P0 iMessage pager FAILED: …")` (mirrors L1 at :944). Inherent limit: if Slack itself is down, the fallback can't deliver either (iMessage WAS the Slack backup) — this is a best-effort L1 record, acceptable and worth a code comment.
9. **(d) settings field.** No escalation-phone field exists in `settings.py` today (add e.g. `escalation_phone_e164: str = Field("+4794810537", …)`, keeping the current value as default so behavior is byte-identical). Note the SAME literal is duplicated at `sla_monitor.py:20`, `queue_notification.py:34/63/164` — OUT of scope for 75.7 (criterion only requires scheduler.py clean) but a real latent-duplication defect: candidate for its OWN research-gated masterplan step (per `feedback_queue_discovered_defects_in_masterplan`). Do NOT silently expand 75.7 to touch them.
10. **Criterion 1 test reliability.** The un-awaited-coroutine RuntimeWarning fires at **GC time** (non-deterministic; S7) — `-Werror` alone can silently PASS. Anchor the assertion on the deterministic AttributeError (stub `chat_stream` as `async def`; if prod fails to await, `.append` on the coroutine raises immediately → assert the handler completes WITHOUT that error and that the stub's async append/stop were actually awaited). If also asserting the warning, force `gc.collect()` inside `warnings.catch_warnings()` + `simplefilter("error", RuntimeWarning)`.
11. **Do NOT set a global `filterwarnings = error::RuntimeWarning` in pytest.ini.** That would fail unrelated suite tests emitting benign RuntimeWarnings. Scope it to the new test module/functions (`pytestmark`/`@pytest.mark.filterwarnings`).
12. **Test import chain / offline.** Importing `streaming_integration` pulls `multi_agent_orchestrator` + `agent_definitions`; `conftest.py` sets `PYFINAGENT_TEST_NO_BQ=1` (offline-safe) and the existing test monkeypatches `get_orchestrator` — follow that to avoid constructing the real orchestrator. Use the repo's `asyncio.run(coro)`-in-sync-`def` pattern (pytest-asyncio is NOT installed).

## 12. JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 10,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "slack_sdk 3.41.0 introspection is dispositive: AsyncApp injects AsyncWebClient whose chat_stream/append/stop are coroutines, so (a) pysvc-01 is CONFIRMED -- un-awaited chat_stream at :212/:238 makes .append() at :220/:261 raise AttributeError into the broad except at :176-183 (step's :141-149 is the wrong block; every non-DIRECT message dies). (b) CONFIRMED, fan-out is :292-341 not :260; port to asyncio.create_task(asyncio.to_thread(_run_agent))+async-for-as_completed. (c) PARTIAL: _get_live_data (app_home:364, ~41s) and _read_status (commands:481) are unwrapped, but the reaction git push (commands:567-572) is ALREADY asyncio.to_thread'd (75.2.1) -- step over-counts by one. (d) CONFIRMED: scheduler:961-969 discards the imsg CompletedProcess and logs 'sent' whenever returncode!=0 (no raise) -- a silent LAST-RESORT P0 kill-switch pager failure; phone literal :955; NO settings field exists yet. 72.0.4 non-overlap PROVEN: it anchors autonomous_loop.py P1 bot-token Slack (no imsg leg; phone literal grep-absent there), disjoint file/severity/transport. Test tooling: pytest-asyncio ABSENT, use asyncio.run-in-sync-def (repo precedent) + scoped filterwarnings; RuntimeWarning is GC-nondeterministic so anchor on AttributeError. Executor must fix stale line anchors, add import asyncio to app_home, remove unused concurrent.futures import, and NOT re-wrap the reaction push.",
  "brief_path": "handoff/current/research_brief_75.7.md",
  "gate_passed": true
}
```

### Research Gate Checklist
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7: S1-S7, mostly Tier-1/2 official)
- [x] 10+ unique URLs total (~30)
- [x] Recency scan (last 2 years) performed + reported (§3)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (§4-§9)
- [x] Internal exploration covered every relevant module (streaming_integration, app, assistant_lifecycle, app_home, commands, scheduler, settings, autonomous_loop, alerting, tests, conftest, pytest.ini, installed slack_sdk)
- [x] Contradictions/consensus noted (Bolt sync-example vs installed async coroutine — resolved by introspection)
- [x] Per-claim citations (§3b external; §4 internal)
