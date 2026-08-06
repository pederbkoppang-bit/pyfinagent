# Contract -- phase-82.59

**Step:** 82.59 (P1) -- two production-wired Slack handler call sites bind
arguments that do not exist.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.59.md`,
`gate_passed: true`, **audit_class** with `dry: true` after 4 rounds / 3 dry,
6 sources read in full, 24 URLs, 14 internal files.

---

## 1. The defect, re-derived by me and independently by the gate

`register_assistant_lifecycle` (`backend/slack_bot/assistant_lifecycle.py`)
registers **three** Bolt listeners. Runtime `inspect.signature().bind()`:

```
MISMATCH handle_thread_started: missing a required argument: 'set_suggested_prompts'
         real: (self, body, say, set_suggested_prompts, logger)
MISMATCH handle_context_changed: got an unexpected keyword argument 'client'
         real: (self, body, logger)
OK       handle_user_message(self, body, client, say, set_status, logger)
```

Production-wired: `register_assistant_lifecycle` imported at
`backend/slack_bot/app.py:18`, invoked unconditionally at `:33` with no flag.
**The bot is live** (launchd `com.pyfinagent.slack-bot`). Zero test coverage.

## 2. Blast radius: SILENT, and that is the whole point

The gate read the installed package rather than the docs.
`AsyncioListenerRunner` catches `Exception` at `asyncio_runner.py:70` and
`:119`, routing to `AsyncDefaultListenerErrorHandler` whose entire body is
`logger.exception()` (`async_listener_error_handler.py:66-67`). **Ack fires
BEFORE the listener** in the Socket-Mode branch (`:104-106`), so Slack always
sees 200.

So the user-visible symptom is a **blank assistant panel** -- no welcome
message, no suggested prompts -- plus one stderr line. Never an error anyone
would notice. This is why it survived since April.

**Bolt cannot catch this at registration.** It never validates listener kwargs
at registration time, and at invocation it is *lenient*: an unknown listener
kwarg becomes `None` plus a warning
(`kwargs_injection/async_utils.py:109-111`). The `TypeError` comes from the
plain Python call one frame deeper, which Bolt has no visibility into.

## 3. Which side drifted -- determined from history, not assumed

The step told me not to assume the call site is wrong. Git blame settles it:

- Handler signatures: `0da5a907` (2026-04-05), **never edited since**.
- Registration block: `cb77f065` (2026-04-06).

One kwarg set -- `(body, client, say, set_status, logger)`, which is **exactly
`handle_user_message`'s correct signature** -- was copy-pasted onto all three
inner calls. **Fix the call site; leave the handlers alone.**

## 4. A SECOND defect the step does not mention

`on_thread_started` never declares `body`, and the inner call hardcodes
`body={}`. So `channel_id` and `thread_ts` stay `None` even after the kwarg fix
-- the handler logs `channel=None, thread_ts=None` and `say()` posts into
whatever thread Bolt's context implies rather than the one in the event.

**A raise-free assertion would pass while the handler still sees nothing.** So
criterion 2's guard must assert the handler received REAL values, not merely
that it completed.

## 5. Immutable success criteria (verbatim)

1. "both call sites are bound against the live handler signatures via
   inspect.signature().bind(), asserted by a test that FAILS against the current
   code and names each offending parameter"
2. "a fixture drives each registered Bolt handler with a realistic payload and
   asserts it completes without raising, rather than asserting that the handler
   was merely invoked"
3. "whether Slack Bolt swallows the TypeError is determined and recorded, so the
   operator-visible impact is stated rather than assumed"
4. "every handler registered by register_assistant_lifecycle is enumerated
   structurally and bound against its target, the derived set asserted
   non-empty, and any further mismatch fixed or queued"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_59_assistant_lifecycle_binding.py -q`

## 6. Guard traps named in advance, all measured by the gate

- **Criterion 1 must NOT assert on the exception message.** The two instruments
  disagree for the same site: `inspect.bind()` says *"missing a required
  argument: 'set_suggested_prompts'"* while the real call says *"got an
  unexpected keyword argument 'client'"*. Assert on **derived missing/extra
  sets**, not on text.
- **`assert response.status == 200` is VACUOUS** -- it is green against today's
  broken code, because ack precedes the listener (§2).
- **One shared payload across all three handlers is vacuous for
  `user_message`** -- the gate measured it COMPLETED without raising on a wrong
  body, because every field resolved to `None`. Each listener gets its own
  vendor payload.
- **Criterion 4's sweep must NOT use a name-based AST resolver.** The gate's
  round-4 name-based pass produced 10 hits, **all false positives**
  (`subprocess.run` colliding with local `jobs/*.run` defs). Import-resolved
  only.
- **Completion is not enough** (§4): assert the handler saw real values.

## 7. Plan

- **D1** -- fix both call sites: `on_thread_started` declares `body` and passes
  `body, say, set_suggested_prompts, logger`; `on_context_changed` passes
  `body, logger`. Drop the invented `client` / `set_status`. No handler edits.
- **D2** -- criterion 2: a stub app (`.client` + `.use()`) registers with **no
  Slack token and no network**; listeners reached via
  `assistant._thread_started_listeners[0].ack_function`; payloads taken verbatim
  from Bolt's own `tests/scenario_tests_async/test_events_assistant.py`. The
  gate built and RAN this shape, measuring 2 TypeErrors + 1 pass, so it is
  proven rather than proposed.
- **D3** -- criterion 1: derive missing/extra parameter sets per site from
  `inspect.signature`, and fail naming them.
- **D4** -- criterion 4: enumerate all three listeners structurally
  (import-resolved), assert the set non-empty and of size 3.
- **D5** -- criterion 3: record the swallow finding with its file:line.

## 8. Non-scope, and one thing deliberately left alone

- **No handler-body edits.** §3 shows the handlers are correct.
- Registering `thread_context_changed` **suppresses Bolt's built-in
  `default_thread_context_changed`** (`async_assistant.py:255-256`). Our handler
  does real work (it tracks the new channel context for the message handler), so
  the registration stays. Recorded because it is a real consequence, not
  because it changes.
- `backend/requirements.txt:55` pins `slack-bolt[async]>=1.18.0` -- a **floor,
  not a pin** (installed 1.27.0, published 1.30.0). The fix is version-robust,
  but the unpinned floor is a separate hardening item -> queue, don't fix here.
- The welcome message contains emoji. Pre-existing, outside this step's diff,
  and Slack copy is not the frontend the no-emoji rule governs. Not touched.
- No live Slack message sent by any test. No live positions.

## 9. References

- `handoff/current/research_brief_82.59.md` (audit-class, dry after 3 rounds)
- slack_bolt installed source: `middleware/assistant/async_assistant.py`,
  `listener/asyncio_runner.py:70,104-106,119`,
  `listener/async_listener_error_handler.py:66-67`,
  `kwargs_injection/async_utils.py:109-111`
- Bolt's own `tests/scenario_tests_async/test_events_assistant.py` (payloads)
- Internal: `backend/slack_bot/assistant_lifecycle.py:174-199`,
  `backend/slack_bot/app.py:18,33`
