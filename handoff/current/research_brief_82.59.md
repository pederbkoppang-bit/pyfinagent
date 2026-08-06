# Research Brief -- step 82.59 (Slack assistant-lifecycle handler binding defect)

TIER: moderate | AUDIT_CLASS: true (loop-until-dry, K=2) | Started 2026-08-06

STATUS: IN PROGRESS (write-first skeleton; sections filled incrementally)

## Research: slack_bolt AsyncAssistant listener kwarg binding, error swallowing, and
## test-fixture shapes for assistant middleware handlers

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://docs.slack.dev/reference/events/assistant_thread_started/ | 2026-08-06 | official doc | WebFetch | Full envelope: `{token, team_id, api_app_id, event:{type, assistant_thread:{user_id, context:{channel_id,team_id,enterprise_id}, channel_id, thread_ts}, event_ts}, type:"event_callback", authorizations, event_id, event_time}`. "No scopes required!" -- but the app must be configured as an assistant app. |
| https://docs.slack.dev/reference/events/assistant_thread_context_changed/ | 2026-08-06 | official doc | WebFetch | Identical envelope shape to `assistant_thread_started`, only `event.type` differs; `context.channel_id` is the NEW channel the user switched to (`C333` in Bolt's own fixture vs `C222` on start). |
| https://raw.githubusercontent.com/slackapi/bolt-python/main/tests/scenario_tests_async/test_events_assistant.py | 2026-08-06 | official source (vendor tests) | WebFetch | Bolt's OWN async assistant scenario test. Gave all three canonical fixture bodies verbatim, the listener signatures Bolt's maintainers use, and the drive pattern: `AsyncBoltRequest(body=..., mode="socket_mode")` + `await app.async_dispatch(request)`, verified with `assert response.status == 200` AND `asyncio.Event` flags via `asyncio.wait_for(listener_called.wait(), timeout=0.1)`. The Event flag is the tell that status alone is insufficient. |
| https://docs.python.org/3/library/inspect.html | 2026-08-06 | official doc | WebFetch | `Signature.bind(*args, **kwargs)` "Raises a TypeError" if args do not match; `bind_partial` "allows the omission of some required arguments". `BoundArguments.arguments` holds only explicitly bound args -- "Arguments for which Signature.bind() ... relied on a default value are skipped." `follow_wrapped=True` by default follows `__wrapped__` (relevant: Bolt's decorators use `functools.wraps`). |
| https://github.com/slackapi/bolt-python/releases | 2026-08-06 | official release notes | WebFetch | Latest is **1.30.0**; 1.28.0 "introduced `say_stream` ... plus agent-specific enhancements like the `set_suggested_prompts` helper and improved thread context defaults"; 1.29.0 added authorship args to `say_stream` + fixed Socket-Mode retry propagation; 1.30.0 "widened `set_suggested_prompts` initialization scope to any direct message". |
| https://pypi.org/pypi/slack-bolt/json | 2026-08-06 | official registry | WebFetch | Confirms current published version `1.30.0`; corroborates the release-page reading independently of GitHub. Installed here is **1.27.0** (`.venv/.../slack_bolt/version.py:3`). |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://docs.slack.dev/ai/developing-agents/ | official doc | The reference cited in `assistant_lifecycle.py:9`; superseded for our purposes by the two event-reference pages + Bolt's own test, which are more precise. |
| https://docs.slack.dev/tools/bolt-python/concepts/assistant/ | official doc | **FETCH FAILED TWICE** -- both `tools.slack.dev/...` (301) and the redirect target returned the docs HOMEPAGE, not the concept page (JS-rendered nav shell, same failure mode as `feedback_gcloud_docs_fetch`). Substituted with the installed package source, which is authoritative anyway. |
| https://raw.githubusercontent.com/slackapi/bolt-python/main/tests/scenario_tests_async/test_assistant.py | vendor tests | 404 -- wrong filename; the real path is `test_events_assistant.py` (found via `gh api search/code`), which IS read in full above. |
| https://docs.slack.dev/reference/events/message/assistant_app_thread/ | official doc | Message subtype in assistant threads; out of scope (site #3 already binds OK). |
| https://docs.slack.dev/tools/bolt-js/concepts/using-the-assistant-class/ | official doc | Bolt **JS**, not Python -- different kwarg-injection model. |
| https://tools.slack.dev/bolt-python/api-docs/slack_bolt/listener/listener_error_handler.html | official API ref | Generated API stub; the installed `async_listener_error_handler.py` source is strictly better evidence. |
| https://docs.slack.dev/tools/bolt-python/reference/middleware/index.html | official API ref | Same -- superseded by reading the installed middleware. |
| https://docs.slack.dev/tools/bolt-python/reference/listener/ | official API ref | Same. |
| https://slack.dev/bolt-python/api-docs/slack_bolt/middleware/async_middleware_error_handler.html | official API ref | Middleware (not listener) error path; the assistant runs as a listener. |
| https://docs.slack.dev/tools/bolt-python/reference/logger/messages.html | official API ref | Confirms the `warning_did_not_call_ack` / warning message catalogue; not load-bearing. |
| https://docs.slack.dev/changelog/ | official changelog | Platform-wide; the Bolt release page is the targeted source. |
| https://github.com/slackapi/python-slack-sdk/releases | official release notes | SDK, not Bolt; no bearing on listener binding. |
| https://api.slack.com/events/assistant_thread_started | official doc (legacy host) | Duplicate of the `docs.slack.dev` page read in full. |
| https://api.slack.com/events | official index | Index page. |
| https://1oglop1.github.io/slack-bolt-python-api-docs/app/app.html | third-party mirror | Unofficial mirror of the API docs; community tier, superseded. |
| https://ai-sdk.dev/cookbook/guides/slackbot | vendor blog | Vercel AI SDK slackbot guide -- different framework. |
| https://flueframework.com/docs/ecosystem/channels/slack/ | community doc | Unrelated framework. |
| https://api.slack.com/changelog/slack-connect-api-2025 | official changelog | Slack Connect, unrelated. |

Queries run (three-variant discipline): **year-less canonical** `slack assistant_thread_started event payload example`;
**current-year frontier** `slack bolt python Assistant middleware listener error handler 2026`;
**last-2-year window** `slack-bolt-python releases 2025 assistant AsyncAssistant changelog`. Plus a
`gh api search/code` pass over `repo:slackapi/bolt-python` to locate the vendor test file by content
rather than by guessed filename.

### Recency scan (2024-2026)

Searched the 2024-2026 window for changes to Bolt's Assistant middleware and assistant listener
kwargs. **Result: TWO new findings, both of which qualify the fix rather than change it.**

1. **A version gap exists and it is unpinned.** `backend/requirements.txt:55` declares
   `slack-bolt[async]>=1.18.0` -- a FLOOR, not a pin. Installed is `1.27.0`; current published is
   `1.30.0` (PyPI + GitHub releases, both read in full). So any `pip install -U` silently moves this
   code three minor versions. Relevant releases: **1.28.0** added the `set_suggested_prompts` agent
   helper and "improved thread context defaults"; **1.29.0** "fixed Socket Mode retry propagation";
   **1.30.0** "widened `set_suggested_prompts` initialization scope to any direct message".
2. **The fix is version-robust anyway.** `set_suggested_prompts` is already present in the installed
   1.27.0 injection table (`kwargs_injection/async_utils.py:58`) and in the assistant utilities
   (`context/assistant/async_assistant_utilities.py:63-65`), and the listener at
   `assistant_lifecycle.py:180` already receives it successfully today. Nothing in the 1.28-1.30
   notes removes or renames any kwarg the fix depends on. **No 2024-2026 finding supersedes the
   installed-source reading**; the `>=1.18.0` floor is worth flagging as a separate hardening item
   but is out of scope for 82.59.

### Key findings

1. **The defect is SILENT.** Bolt's `AsyncioListenerRunner` catches `Exception`
   (`asyncio_runner.py:70` and `:119`) and hands it to `AsyncDefaultListenerErrorHandler`, whose whole
   body is `self.logger.exception(message)` (`async_listener_error_handler.py:66-67`). Slack still gets
   its 200 (auto-ack, `async_assistant.py:316`; and in the async branch the ack precedes the listener,
   `asyncio_runner.py:104-106`). Impact = blank assistant panel + one stderr line, not a visible error.
2. **The call site drifted, not the handlers.** Handler signatures: `0da5a907` (2026-04-05), never
   edited since. Registration block: `cb77f065` (2026-04-06). The one kwarg set
   `(body, client, say, set_status, logger)` -- which is exactly `handle_user_message`'s correct
   signature -- was copy-pasted onto all three inner calls. Fix the call site; leave the handlers alone.
3. **Exactly 3 listeners / 3 inner calls / 2 broken.** Derived by AST walk + `inspect.signature().bind()`,
   then re-derived a second way by reading `AsyncAssistant._*_listeners` off the live middleware. The
   step's count of two is correct. No fourth site.
4. **A SECOND defect at site #1 the step does not mention:** `body={}` is hardcoded
   (`assistant_lifecycle.py:182`) and `on_thread_started` does not declare `body` (`:180`), so
   `channel_id`/`thread_ts` at `:45-46` resolve to `None` even after the kwarg fix.
5. **Zero test coverage, and the bot is LIVE.** No test references `assistant_lifecycle`;
   `register_assistant_lifecycle` is called unconditionally at `app.py:33`; `pgrep` shows pid 658 under
   launchd `com.pyfinagent.slack-bot`.
6. **Bolt never validates at registration time**, and even at invocation it is lenient: an unknown
   listener kwarg becomes `None` + a warning (`kwargs_injection/async_utils.py:109-111`). Nothing could
   have caught this at import.
7. **`>=1.18.0` is a floor, not a pin** (`backend/requirements.txt:55`); installed 1.27.0, current 1.30.0.
   The fix is version-robust, but the unpinned floor is a separate hardening item.
8. **Audit rounds found no further instances** in `backend/slack_bot/` (22 files, 176 defs, 45
   resolvable kwarg calls, 20 decorated Bolt listeners) -- see the coverage log below.

### Internal code inventory
| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/assistant_lifecycle.py` | 202 | The 3 handler methods + `register_assistant_lifecycle` | **DEFECTIVE at :181 and :188** (+ hardcoded `body={}` at :182). Handlers at :28-88, :90-113, :115-169 are correct. |
| `backend/slack_bot/app.py` | 78 | Bolt entry; imports at :18, invokes at :33 inside `create_app()` | Production-wired, unconditional, no flag |
| `backend/slack_bot/streaming_integration.py` | ~560 | `handle_user_message_with_streaming(body, client, say, set_status, logger)` at :85-91 | OK -- binds cleanly from `assistant_lifecycle.py:153-159` |
| `backend/slack_bot/assistant_guards.py` | ~130 | Guard helpers imported by `streaming_integration.py:40` and `app_home.py:404` | Not on the two broken paths |
| `backend/tests/test_phase_75_2_slack_control_plane.py` | ~250 | Nearest existing test; covers `assistant_guards` (:23) + `handle_user_message_with_streaming` (:227) | Does NOT cover `register_assistant_lifecycle` |
| `backend/requirements.txt` | -- | `slack-bolt[async]>=1.18.0` at :55 | Unpinned floor (see finding 7) |
| `.venv/.../slack_bolt/middleware/assistant/async_assistant.py` | 321 | `AsyncAssistant`: decorators :52/:98/:144/:190, dispatch :248-278, listener build :280-320 | Vendor, read in full |
| `.venv/.../slack_bolt/listener/asyncio_runner.py` | 191 | The swallow points :70 and :119 | Vendor, read in full |
| `.venv/.../slack_bolt/listener/async_listener_error_handler.py` | 67 | `AsyncDefaultListenerErrorHandler` logs only, :60-67 | Vendor, read in full |
| `.venv/.../slack_bolt/kwargs_injection/async_utils.py` | 112 | Injection table :32-66; lenient unknown-arg path :109-111 | Vendor, read in full |
| `.venv/.../slack_bolt/context/assistant/async_assistant_utilities.py` | ~95 | Builds `say`/`set_status`/`set_suggested_prompts`; **raises `ValueError` at :49** if the payload lacks `assistant_thread.{channel_id,thread_ts}` | Vendor -- constrains the fixture payload |
| `.venv/.../slack_bolt/app/async_app.py` | -- | :1434-1446 injects the assistant utilities into `req.context` only when `is_assistant_event(req.body)` | Vendor, region read |
| `.venv/.../slack_bolt/request/payload_utils.py` | -- | Matchers :44-53 gate on `body["event"]["type"]` | Vendor, region read |
| `.venv/.../slack_bolt/version.py` | 3 | `__version__ = "1.27.0"` | Vendor |

### Q1-Q7 answers

#### Q1 (a) BLAST RADIUS -- **THE DEFECT IS SILENT.** Bolt swallows the TypeError.

Read from the INSTALLED package (`slack_bolt==1.27.0`, `.venv/lib/python3.14/site-packages/slack_bolt/version.py:3`), not from docs.

Chain: `AsyncAssistant.async_process` (`async_assistant.py:248-278`) matches the event and
delegates to `listener_runner.run(...)` at `async_assistant.py:268-273`. Inside
`AsyncioListenerRunner.run`:

- `process_before_response=True` branch: `try:` at `asyncio_runner.py:63`, listener invoked at
  `:65`, and **`except Exception as e:` at `:70`** -> `await self.listener_error_handler.handle(...)`
  at `:76-80`.
- `process_before_response=False` branch (the Socket-Mode default): `ack()` fires FIRST at
  `asyncio_runner.py:104-106` (before the listener runs), the listener body runs inside
  `run_ack_function_asynchronously` at `:111-135`, with **`except Exception as e:` at `:119`**
  -> `listener_error_handler.handle(...)` at `:128-132`.

The default handler is `AsyncDefaultListenerErrorHandler.handle`
(`async_listener_error_handler.py:60-67`), whose entire body is:

```
message = f"Failed to run listener function (error: {error})"
self.logger.exception(message)
```

It logs and returns None. It does not re-raise. The `raise` inside
`handle_thread_started`'s own `except` (`assistant_lifecycle.py:88`) is therefore also
absorbed one frame up.

**Operator-visible impact:** nothing crashes; Slack still gets its 200 ack (auto-ack is set at
`async_assistant.py:316` `auto_acknowledgement=True`, and in the async branch the ack precedes the
listener entirely). The user opening the assistant container sees **no welcome message and no
suggested prompts** -- a blank agent panel -- and the only trace is a
`Failed to run listener function (error: ... missing a required argument: 'set_suggested_prompts')`
line in the bot's stderr. So the correct framing for criterion 3 is: silent feature death, not a
visible error. This makes the "assert it was invoked" style of guard useless here -- the listener
IS invoked; it just dies one frame in.

#### Q2 (b) WHICH SIDE DRIFTED -- **the CALL SITE drifted, one day later. Do NOT touch the handler signatures.**

`git blame` on `backend/slack_bot/assistant_lifecycle.py`:

| Region | Commit | Date | Subject |
|--------|--------|------|---------|
| handler sigs :28-34, :90-94, :115-122 | `0da5a907` | 2026-04-05 | Phase 2 GENERATE: Assistant lifecycle handlers |
| registration bodies :173-199 (all three listeners + all three inner calls) | `cb77f065` | 2026-04-06 | Fix Slack bot startup: AsyncAssistant import + assistant lifecycle |
| :201 logger string only | `38768f75` | 2026-05-23 | phase-38.5.1 ASCII-logger sweep |

The three handler methods have **never been edited since they were written** on 2026-04-05. The
registration block was rewritten wholesale the next day by `cb77f065` when the code was ported onto
`AsyncAssistant`. So the call site is the side that drifted, and the fix belongs there.

**Mechanism (visible in the blame, and it explains why it is exactly two sites):** `cb77f065`
wrote ONE kwarg set -- `(body=, client=, say=, set_status=, logger=)` -- and pasted it into all
three inner calls. That set is the *correct and complete* signature of the THIRD handler,
`handle_user_message(self, body, client, say, set_status, logger)` (`assistant_lifecycle.py:115-122`),
which binds cleanly. It was then copy-pasted onto the two handlers whose signatures differ. This is
a copy-paste drift, not a deliberate interface change -- further evidence that the handlers are
authoritative and the call site is wrong.

#### Q5 (e) FULL ENUMERATION -- exactly THREE listeners, THREE inner call sites, TWO broken. The step's count of two is CORRECT (derived, not assumed).

Derived structurally by AST-walking `register_assistant_lifecycle` and then binding each inner call
against the live method via `inspect.signature().bind()`:

| # | Bolt decorator | Listener (line) | Listener params Bolt injects | Inner call (line) | Bind result |
|---|----------------|-----------------|------------------------------|-------------------|-------------|
| 1 | `@assistant.thread_started` | `on_thread_started` :180 | `say, set_suggested_prompts, get_thread_context, logger` | `handler.handle_thread_started(body, client, say, set_status, logger)` :181 | **TypeError: missing a required argument: 'set_suggested_prompts'** -- MISSING `['set_suggested_prompts']`, EXTRA `['client','set_status']`; real sig `(self, body, say, set_suggested_prompts, logger)` |
| 2 | `@assistant.thread_context_changed` | `on_context_changed` :187 | `body, logger` | `handler.handle_context_changed(body, client, say, set_status, logger)` :188 | **TypeError: got an unexpected keyword argument 'client'** -- MISSING `[]`, EXTRA `['client','say','set_status']`; real sig `(self, body, logger)` |
| 3 | `@assistant.user_message` | `on_user_message` :194 | `body, client, say, set_status, logger` | `handler.handle_user_message(body, client, say, set_status, logger)` :195 | **BINDS OK** -- real sig `(self, body, client, say, set_status, logger)` |

No fourth registration exists. `@assistant.bot_message` is available on the middleware
(`async_assistant.py:144-188`) but is not used here. Note `AsyncAssistant.async_process` auto-installs
a `default_thread_context_changed` listener **only if none was registered**
(`async_assistant.py:255-256`) -- since site #2 registers one, that default is suppressed, so the
broken listener also silently disables Bolt's built-in thread-context persistence. Downstream, the
one non-listener call `handle_user_message_with_streaming(body, client, say, set_status, logger)`
(`assistant_lifecycle.py:153-159`) also binds cleanly against
`streaming_integration.py:85-91` -- no third defect there.

**Criterion-4 note:** the derived set must be asserted non-empty AND asserted to equal 3. An AST
walk that silently matches 0 nodes (e.g. if the code is later refactored to `self.handler.x()` or a
module-level function) would pass a naive "all bound calls are OK" assertion vacuously. This is the
82.12/82.16 vacuous-guard class.

#### Q3 (c) IS THE FIX SIMPLY "PASS WHAT BOLT PROVIDES"? -- Yes, but there is a SECOND defect at the same site.

The handler bodies genuinely use their declared params, so the feature is **not** dead code:
`handle_thread_started` awaits `say({...})` at `assistant_lifecycle.py:52-60` AND awaits
`set_suggested_prompts({"prompts": [...4 prompts...]})` at `:63-82`. So `set_suggested_prompts` is
really used -- dropping it is not an option; the feature is live-but-dead-on-arrival.
`handle_context_changed` uses only `body` + `logger` (`:104-113`) and is a pure log/no-op
("In production, store in Redis or thread-local state", `:110-111`).

Bolt supplies every name the handlers need, from `kwargs_injection/async_utils.py:32-66`:
`say` (:53), `set_suggested_prompts` (:58), `logger` (:33), `body` (:38), `client` (:34),
`set_status` (:56), `get_thread_context` (:59). So the minimal fix is:

- site #1 -> `handle_thread_started(body=body, say=say, set_suggested_prompts=set_suggested_prompts, logger=logger)`
- site #2 -> `handle_context_changed(body=body, logger=logger)`

**SECOND DEFECT AT SITE #1 (the step does not mention it; latent even after the kwarg fix):** the
call hardcodes `body={}` (`assistant_lifecycle.py:182`) and the listener `on_thread_started` does
not even declare `body` in its params (`:180`). `handle_thread_started` reads
`body["assistant_thread"]["channel_id"]` and `["thread_ts"]` at `:45-46`, so with `body={}` both
resolve to `None` and the log line at `:48` would print `channel=None, thread_ts=None` forever. The
fix must therefore ALSO add `body` to the `on_thread_started` listener params so Bolt injects the
real payload. A test that only asserts "no TypeError" would pass while `channel_id` stays `None` --
so criterion 2's "completes without raising" should be paired with an assertion that the handler saw
the fixture's real channel_id (otherwise the guard stops one seam short).

Note `on_thread_started` also declares `get_thread_context` (`:180`) and never uses it -- harmless
dead param, but worth removing or wiring while in the file.

#### Q6 (f) EXISTING TESTS + IS IT LIVE? -- ZERO tests, and the bot IS RUNNING.

- **Zero test coverage of `assistant_lifecycle`.** `grep -rn "assistant_lifecycle|register_assistant|handle_thread_started|handle_context_changed" --include="*.py" backend/tests/ scripts/` returns exactly one hit, and it is not a test of this module: `scripts/qa/sweep_ascii_logger_v3.py:37` merely lists the file path. The nearest neighbour,
  `backend/tests/test_phase_75_2_slack_control_plane.py`, covers `assistant_guards` (`:23`) and calls
  `streaming_integration.handle_user_message_with_streaming` (`:227`) -- it never touches
  `register_assistant_lifecycle`. So `backend/tests/test_phase_82_59_assistant_lifecycle_binding.py`
  is a greenfield file.
- **The feature is NOT dark.** `register_assistant_lifecycle` is imported at
  `backend/slack_bot/app.py:18` and invoked unconditionally inside `create_app()` at `:33` -- no
  feature flag, no env gate. And the bot process is live right now: `pgrep -fl slack_bot` ->
  `pid 658 ... -m backend.slack_bot.app`, supervised by launchd label `com.pyfinagent.slack-bot`
  (`launchctl list`, pid 658, exit 0). So every assistant-container open in the workspace today hits
  site #1 and dies silently.
- Caveat I could not settle from the repo: whether the Slack **app manifest** has the Agents &
  Assistants feature toggled on. If it is off, Slack never sends `assistant_thread_started` and the
  defect is latent rather than active. The code-side wiring is unconditional either way, so
  correctness is unaffected; only the urgency framing is. State this as an assumption in the
  contract rather than asserting live user impact.

#### Q4 (d) CRITERION-2 FIXTURE -- PROVEN, not proposed. Runs with no Slack token and no network.

I built and RAN the fixture. Output reproduces the defect exactly: 2 TypeErrors, 1 pass.

**Realistic payloads** -- take them verbatim from Bolt's own scenario test
(`tests/scenario_tests_async/test_events_assistant.py`, read in full), which matches the official
event reference. `assistant_thread_started`:

```python
{"token":"verification_token","team_id":"T111","enterprise_id":"E111","api_app_id":"A111",
 "event":{"type":"assistant_thread_started",
   "assistant_thread":{"user_id":"W222",
     "context":{"channel_id":"C222","team_id":"T111","enterprise_id":"E111"},
     "channel_id":"D111","thread_ts":"1726133698.626339"},
   "event_ts":"1726133698.665188"},
 "type":"event_callback","event_id":"Ev111","event_time":1599616881}
```

`assistant_thread_context_changed` is identical except `event.type` and
`context.channel_id: "C333"` (the channel switched TO). The user-message body is the third dict in
that file (`event.type:"message"`, `channel:"D111"`, `channel_type:"im"`, `thread_ts` matching).

**Getting the registered handlers structurally** -- `register_assistant_lifecycle` needs only a stub
app; `AsyncApp` (and therefore a Slack token) is NOT required:

```python
class FakeClient: pass
class FakeApp:
    client = FakeClient()
    used = []
    def use(self, m): self.used.append(m)

app = FakeApp(); register_assistant_lifecycle(app)
assistant = app.used[0]                      # the AsyncAssistant instance
buckets = {"thread_started": assistant._thread_started_listeners,
           "thread_context_changed": assistant._thread_context_changed_listeners,
           "user_message": assistant._user_message_listeners,
           "bot_message": assistant._bot_message_listeners}
registered = {k: v for k, v in buckets.items() if v}   # -> 3 buckets, 3 listeners
fn = registered["thread_started"][0].ack_function       # the decorated listener itself
```

`AsyncCustomListener.ack_function` is the handle on the real listener (built at
`async_assistant.py:310-318`). Measured output: `thread_started -> on_thread_started(say,
set_suggested_prompts, get_thread_context, logger)`, `thread_context_changed -> on_context_changed(body,
logger)`, `user_message -> on_user_message(body, client, say, set_status, logger)`; `bot_message`
empty.

**Minimal fakes** -- every assistant utility Bolt injects is an awaitable callable, so one recorder
class covers `say` / `set_suggested_prompts` / `set_status` / `get_thread_context`:

```python
class Recorder:
    def __init__(self, name): self.name, self.calls = name, []
    async def __call__(self, *a, **kw): self.calls.append((a, kw))
```

`logger` -> a real `logging.getLogger(...)` (the handlers only call `.info`/`.error`); `client` -> a
bare stub object (never dereferenced on these two paths). Then build kwargs by introspecting each
listener -- `{p: fakes[p] for p in inspect.signature(fn).parameters}` -- and `await fn(**kwargs)`.

**Measured result on current code:**
```
thread_started:         TypeError: AssistantLifecycleHandler.handle_thread_started() got an unexpected keyword argument 'client'
thread_context_changed: TypeError: AssistantLifecycleHandler.handle_context_changed() got an unexpected keyword argument 'client'
user_message:           COMPLETED (no raise)
```

**Why direct listener invocation beats full `app.async_dispatch`:** Bolt's own tests use
`AsyncBoltRequest(body=..., mode="socket_mode")` + `await app.async_dispatch(request)`, but that path
(a) needs a real `AsyncApp` + token + Bolt's mock web-api server, and (b) **runs the listener
fire-and-forget** (`asyncio.ensure_future` at `asyncio_runner.py:137`) with the exception swallowed --
which is precisely why the vendor test needs `asyncio.Event` + `asyncio.wait_for` on top of
`assert response.status == 200`. Direct invocation lets the TypeError propagate to the test, which is
what criterion 2 actually wants.

#### Q7 (g) REGISTRATION-TIME vs INVOCATION-TIME -- invocation-time only, and even then Bolt does not error.

Bolt's kwarg injection is `build_async_required_kwargs`
(`kwargs_injection/async_utils.py:21-112`). It reads the listener's declared arg names and picks
matching entries out of `all_available_args` (`:32-66`). Two consequences:

1. It runs at **invocation**, per-request (called from the listener's `run_ack_function`), never
   at decoration/registration. `AsyncAssistant.thread_started` (`async_assistant.py:52-96`) only
   appends a built listener to a list -- it never inspects the function signature. So nothing about
   this defect could have been caught at import time.
2. Even an *unknown* kwarg on the listener is not an error: `async_utils.py:109-111` logs
   `f"{name} is not a valid argument"` and injects `None`. Bolt is deliberately lenient.

But note the defect here is NOT in Bolt's injection at all. The decorated listeners
(`on_thread_started(say, set_suggested_prompts, get_thread_context, logger)` at
`assistant_lifecycle.py:180`) bind fine -- all four names exist in `all_available_args`
(`say` :53, `set_suggested_prompts` :58, `get_thread_context` :59, `logger` :33). The TypeError
comes from the **plain Python call one frame deeper**, `handler.handle_thread_started(...)` at
`assistant_lifecycle.py:181-184`, which Bolt has no visibility into. That is exactly why a
signature-binding test is the right instrument.


### Adaptive-coverage log (audit_class=true, K_required=2)

| Round | Method | Hits | NEW findings |
|-------|--------|------|--------------|
| 1 | Targeted: read `assistant_lifecycle.py` + `app.py` in full; AST-enumerate `register_assistant_lifecycle`; `inspect.signature().bind()` each inner call; git blame; vendor source read | 2 broken sites (+1 OK) | many (baseline) |
| 2 | Package-wide static bind sweep over `backend/slack_bot/` -- 22 files, 176 defs indexed, 45 kwarg-bearing calls resolved to a unique local def | 2 | **0 (both already known)** -> DRY |
| 3 | Bolt-injectable-param audit: all 20 decorated Bolt listeners in the package, each declared param checked against the injection table read from `kwargs_injection/async_utils.py:32-66` | 0 non-injectable params | **0** -> DRY |
| 4 | Ambiguous-name closure: the 10 kwarg calls round 2 skipped as multi-def, re-bound against EVERY candidate | 10 flagged, **all 10 verified FALSE POSITIVES** (`subprocess.run` colliding with `jobs/*.run`; confirmed by reading `direct_responder.py:120/134/147/266`, `app_home.py:79/85/107`, `scheduler.py:482/984/1046`) | **0** -> DRY |

`rounds = 4`, `dry_rounds = 3` (>= K_required 2) -> `coverage.dry = true`.

Scope honesty: rounds 2-4 cover `backend/slack_bot/` only. 150 kwarg-bearing calls resolve to
definitions outside the package (stdlib / other `backend.*` modules / third-party) and were NOT
bind-checked -- that is a deliberate boundary matching criterion 4's scope
("every handler registered by `register_assistant_lifecycle`"), which IS exhaustively covered (3/3).

### Consensus vs debate (external)

**Consensus.** Slack's own docs, Bolt's shipped tests, and the installed source agree on the payload
shape and on which kwargs each assistant listener receives; there is no ambiguity to adjudicate.
Bolt's design philosophy is explicitly lenient -- unknown listener kwargs become `None` with a warning
rather than an error (`async_utils.py:109-111`) -- which is a deliberate trade of fail-fast for
forward-compatibility.

**Debate / tension.** Bolt's own scenario test asserts `response.status == 200` *and* an
`asyncio.Event` flag. Those two assertions disagree about what "the handler worked" means: the status
is 200 even when the listener raised (auto-ack precedes the listener body). Bolt's maintainers
resolve it by adding the Event; this project should resolve it by invoking the listener directly so the
exception propagates. There is no third-party literature contradicting the swallow behaviour -- it is
read straight from the installed source, which is the strongest available evidence.

### Pitfalls (from literature + measurement)

1. **`assert response.status == 200` is a vacuous guard here.** Proven by reading
   `asyncio_runner.py:104-106` (ack before listener) + `:119` (swallow). It is green against today's
   broken code.
2. **Asserting on the TypeError *message* is instrument-dependent.** Measured: `inspect.signature().bind()`
   says `missing a required argument: 'set_suggested_prompts'` for site #1, while the real runtime call
   says `got an unexpected keyword argument 'client'` -- Python reports whichever it hits first. Assert on
   the DERIVED `missing` / `extra` sets instead, so the test names each offending parameter (criterion 1)
   without being brittle.
3. **One shared payload across all three handlers is vacuous for `user_message`.** Measured: driving
   `on_user_message` with the *thread_started* body still printed `COMPLETED (no raise)` because every
   field resolved to `None` and the streaming path no-ops. Each handler needs its own realistic body.
4. **A name-based AST resolver produces false positives.** Measured: 10/10 in round 4. Resolve
   `handler.<attr>` through the known local binding, not by bare function name.
5. **The derived handler set can silently go empty** (82.12/82.16 class). Assert it is non-empty AND
   equals 3, or a later refactor makes "all bound calls OK" trivially true.
6. **"No raise" is not "works."** Because of finding 4 (hardcoded `body={}`), the fix can pass a
   raise-free assertion while `channel_id` stays `None`. Pair criterion 2 with an assertion that the
   handler observed the fixture's real `channel_id` / `thread_ts`, and that `set_suggested_prompts` was
   actually awaited with 4 prompts.
7. **`AsyncAssistantUtilities` raises `ValueError`** (`async_assistant_utilities.py:37-49`) if the payload
   lacks `assistant_thread.{channel_id, thread_ts}` -- a stripped-down fixture body fails in Bolt before
   it reaches the handler. Use the vendor fixture verbatim.
8. **Registering site #2 suppresses Bolt's built-in `default_thread_context_changed`**
   (`async_assistant.py:255-256`), so the broken listener also disables thread-context persistence.

### Application to pyfinagent (external findings -> file:line)

- Fix `backend/slack_bot/assistant_lifecycle.py:181-184` ->
  `handle_thread_started(body=body, say=say, set_suggested_prompts=set_suggested_prompts, logger=logger)`
  and add `body` to the listener params at `:180` (Bolt injects it -- `async_utils.py:38`).
- Fix `backend/slack_bot/assistant_lifecycle.py:188-191` -> `handle_context_changed(body=body, logger=logger)`.
- Leave `:195-198` and all three handler signatures untouched (Q2).
- New test `backend/tests/test_phase_82_59_assistant_lifecycle_binding.py`: stub-app registration ->
  read `AsyncAssistant._*_listeners` -> assert the derived set is non-empty and == 3 -> per-listener
  `inspect.signature().bind()` reporting missing/extra -> per-listener `await fn(**fakes)` with the
  vendor payloads -> assert side effects (`say` awaited once, `set_suggested_prompts` awaited with 4
  prompts, observed `channel_id == "D111"`).
- Record the swallow finding (criterion 3) citing `asyncio_runner.py:70/:119` +
  `async_listener_error_handler.py:66-67`, and state the operator impact as *silent blank panel*.
- Queue separately (out of 82.59 scope): pin `slack-bolt` above the `>=1.18.0` floor
  (`backend/requirements.txt:55`).

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **6** (2 official Slack event refs,
      Bolt vendor scenario test, CPython inspect docs, Bolt GitHub releases, PyPI registry JSON)
- [x] 10+ unique URLs total -- **6 read in full + 18 snippet-only = 24**
- [x] Recency scan (2024-2026) performed + reported -- 2 findings (version gap; fix is version-robust)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
- [x] Three-variant query discipline (year-less / 2026 / 2025) -- listed above
- [x] audit_class -> `coverage.dry == true` (3 dry rounds >= K=2)

Soft checks:
- [x] Internal exploration covered every relevant module (both repo-side and vendor-side)
- [x] Contradictions / consensus noted (status-200 vs Event-flag tension)
- [x] Claims cited per-claim
- [~] GAP: could not confirm from the repo whether the Slack **app manifest** has the Agents &
      Assistants feature enabled -- that gates whether the defect is *active* today or *latent*.
      Correctness and the fix are unaffected; only the urgency framing is. Flag as an assumption in
      the contract rather than asserting live user impact.

### JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 18,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": true,
    "rounds": 4,
    "dry_rounds": 3,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_82.59.md",
  "gate_passed": true
}
```

STATUS: COMPLETE.
