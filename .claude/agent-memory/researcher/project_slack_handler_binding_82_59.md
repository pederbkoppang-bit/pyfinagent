---
name: slack-handler-binding-82-59
description: Bolt SWALLOWS listener exceptions so every Slack handler defect is silent; status==200 is a vacuous guard; bind-error text differs between inspect.bind() and a real call; name-based AST call resolution false-positives on subprocess.run
metadata:
  type: project
---

Step 82.59 (2026-08-06): two production-wired call sites in
`backend/slack_bot/assistant_lifecycle.py` pass kwargs the handlers do not
accept. Re-derive line numbers; the durable parts are below.

**Bolt swallows listener exceptions -- every Slack handler defect is SILENT.**
`AsyncioListenerRunner.run` catches `Exception` in BOTH branches
(`asyncio_runner.py:70` process-before-response, `:119` async) and routes to
`AsyncDefaultListenerErrorHandler`, whose entire body is
`self.logger.exception(...)` (`async_listener_error_handler.py:66-67`). In the
async branch (Socket-Mode default) `ack()` fires BEFORE the listener runs
(`:104-106`), so Slack always sees 200.
**Why:** decides blast radius on every Slack-handler step -- the symptom is a
blank/missing UI affordance plus one stderr line, never a user-visible error.
**How to apply:** never assert `response.status == 200` as evidence a Bolt
handler worked -- it is green against broken code. Bolt's own scenario tests
add an `asyncio.Event` flag for exactly this reason. Prefer invoking the
listener directly (`assistant._thread_started_listeners[0].ack_function`) so
the exception propagates into the test. A stub app with just `.client` and
`.use()` is enough to register -- no token, no network.

**The two instruments disagree on the error TEXT.**
`inspect.signature().bind()` reported `missing a required argument:
'set_suggested_prompts'`; the real runtime call reported `got an unexpected
keyword argument 'client'` -- same site, both true, Python reports whichever
it hits first.
**Why:** a test asserting the message string is brittle and instrument-coupled.
**How to apply:** assert on the DERIVED `missing` / `extra` parameter sets, not
on the exception message.

**Name-based AST call resolution false-positives.** A sweep resolving
`f(...)` / `x.f(...)` by bare function name flagged 10 sites; all 10 were
`subprocess.run(capture_output=..., text=..., timeout=...)` colliding with
local `jobs/*.run()` defs. 0/10 real.
**Why:** an audit that reports these as findings is worse than no audit.
**How to apply:** resolve through the known local binding (e.g. `handler` ->
`AssistantLifecycleHandler`), and when a name is ambiguous, only report a
failure if it fails against EVERY candidate -- then still eyeball the source.

**Copy-paste is the usual drift mechanism, and blame proves direction.** Here
the handler signatures were untouched since day one; the registration block
was rewritten one day later and pasted ONE kwarg set (the correct signature of
the third handler) onto all three call sites.
**How to apply:** on any "signature mismatch" step, `git blame` both sides
before deciding which to change -- and look for a sibling call site that binds
cleanly, because it is usually the template that was copied.

Related: [[vacuous-type-guards-on-bq-string-columns]],
[[guards-stop-one-seam-short]], [[gcloud-docs-webfetch-nav-only]].
