"""phase-82.59 -- the assistant lifecycle registration bound arguments that do
not exist, and Bolt hid it.

`register_assistant_lifecycle` registers three Bolt listeners. Two of the three
inner calls were copy-pasted from `on_user_message`'s kwarg set --
`(body, client, say, set_status, logger)` -- which is correct for
`handle_user_message` and wrong for the other two. Every invocation raised
TypeError since 2026-04-06.

WHY NOBODY NOTICED, and why "it completes" is not enough:
`AsyncioListenerRunner` catches `Exception` (asyncio_runner.py:70, :119) and
routes it to a handler whose entire body is `logger.exception()`
(async_listener_error_handler.py:66-67) -- and ack fires BEFORE the listener
(:104-106), so Slack always sees 200. The symptom was a blank assistant panel
and one stderr line.

Bolt cannot catch this itself: it never validates listener kwargs at
registration, and at invocation it is lenient -- an unknown listener kwarg
becomes None plus a warning (kwargs_injection/async_utils.py:109-111). The
TypeError comes from the plain Python call one frame deeper.

TRAPS THESE TESTS DELIBERATELY AVOID (each measured during the research gate):
  * asserting on the exception MESSAGE -- the two instruments disagree for the
    same site (`inspect.bind()` reports the missing arg, the live call reports
    the unexpected one). These tests assert on derived missing/extra SETS.
  * `assert response.status == 200` -- vacuous, green against the broken code.
  * one shared payload for all three listeners -- vacuous for `user_message`,
    which COMPLETES on a wrong body because every field resolves to None.
  * a name-based AST resolver for the sweep -- the gate's name-based pass gave
    10 hits, ALL false positives (`subprocess.run` vs local `jobs/*.run`).
  * "completed without raising" alone -- the call used to hardcode `body={}`,
    so the handler saw `channel_id=None` even when it bound.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import pathlib
from unittest import mock

import pytest

from backend.slack_bot import assistant_lifecycle as al
from backend.slack_bot.assistant_lifecycle import AssistantLifecycleHandler

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE = REPO / "backend/slack_bot/assistant_lifecycle.py"

# Realistic event bodies, one per listener. Shapes follow Slack's documented
# assistant events; each carries DISTINCT identifiers so a handler reading the
# wrong one cannot pass by coincidence.
THREAD_STARTED_BODY = {
    "event": {"type": "assistant_thread_started"},
    "assistant_thread": {
        "user_id": "U82590001",
        "context": {"channel_id": "C82590CTX", "team_id": "T8259TEAM"},
        "channel_id": "D82590CHAN",
        "thread_ts": "1786000000.000100",
    },
}
CONTEXT_CHANGED_BODY = {
    "event": {"type": "assistant_thread_context_changed"},
    "assistant_thread": {
        "user_id": "U82590001",
        "context": {"channel_id": "C8259NEWCTX", "team_id": "T8259TEAM"},
        "channel_id": "D82590CHAN",
        "thread_ts": "1786000000.000100",
    },
}
USER_MESSAGE_BODY = {
    "event": {
        "type": "message", "user": "U82590001", "text": "hello",
        "channel": "D82590CHAN", "thread_ts": "1786000000.000100",
        "channel_type": "im",
    },
}


# ---------------------------------------------------------------------------
# criterion 1 -- bind every call site against the LIVE signature
# ---------------------------------------------------------------------------


def _call_sites() -> list[dict]:
    """Every `handler.handle_*(...)` call in the registration module, with the
    keyword names it passes. Resolved by attribute name on the handler object,
    NOT by a bare-name match."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
            continue
        if not n.func.attr.startswith("handle_"):
            continue
        if not hasattr(AssistantLifecycleHandler, n.func.attr):
            continue  # not a call on our handler
        out.append({
            "method": n.func.attr,
            "line": n.lineno,
            "kwargs": {k.arg for k in n.keywords if k.arg is not None},
            "starstar": any(k.arg is None for k in n.keywords),
            "nargs": len(n.args),
        })
    return out


def _binding_report(site: dict) -> tuple[set, set]:
    """(missing_required, unexpected) for one call site, derived from the live
    signature. Sets, not message text -- the two instruments word it
    differently for the very same defect."""
    sig = inspect.signature(getattr(AssistantLifecycleHandler, site["method"]))
    params = {n: p for n, p in sig.parameters.items() if n != "self"}
    required = {
        n for n, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    passed = site["kwargs"]
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    unexpected = set() if accepts_kwargs else passed - set(params)
    return required - passed, unexpected


def test_the_call_site_sweep_actually_sees_the_call_sites():
    """Recall test: an empty derivation and a clean module look identical."""
    sites = _call_sites()
    assert sites, "DERIVED SET IS EMPTY -- the sweep is broken, not the code clean"
    methods = {s["method"] for s in sites}
    assert methods == {
        "handle_thread_started", "handle_context_changed", "handle_user_message"
    }, f"unexpected call-site set {methods!r}"


def test_every_registration_call_site_binds_against_the_live_signature():
    """CRITERION 1. Fails against the shipped code, naming each offending
    parameter -- `handle_thread_started` was missing `set_suggested_prompts`
    while passing `client` and `set_status`; `handle_context_changed` was
    passing `client`, `say` and `set_status` against a `(body, logger)`
    signature."""
    offenders = []
    for site in _call_sites():
        if site["starstar"]:
            continue
        missing, unexpected = _binding_report(site)
        if missing or unexpected:
            offenders.append(
                f"{SOURCE.name}:{site['line']} {site['method']}(): "
                f"missing={sorted(missing) or '-'} unexpected={sorted(unexpected) or '-'}"
            )
    assert not offenders, (
        "registration call sites that cannot bind against their handler's live "
        "signature. Each raises TypeError at invocation, and Bolt swallows it "
        "into logger.exception -- so the only symptom is a blank assistant "
        "panel:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# criterion 2 -- drive each REGISTERED listener with its own realistic payload
# ---------------------------------------------------------------------------


class _StubApp:
    """Enough app for `register_assistant_lifecycle`: no Slack token, no
    network. `.use()` records the middleware so the listeners can be reached."""

    def __init__(self):
        self.client = mock.MagicMock()
        self.middleware = []

    def use(self, mw):
        self.middleware.append(mw)


def _register():
    app = _StubApp()
    al.register_assistant_lifecycle(app)
    assert app.middleware, "register_assistant_lifecycle registered no middleware"
    return app.middleware[0]


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def recorder():
    """Captures what the HANDLER actually received, so a call that binds but is
    fed an empty body is still caught."""
    seen: dict = {}

    async def fake_say(*a, **kw):
        seen.setdefault("said", []).append(a[0] if a else kw)

    async def fake_prompts(payload):
        seen["prompts"] = payload

    async def fake_status(*a, **kw):
        seen.setdefault("status", []).append(a)

    seen["say"], seen["set_suggested_prompts"], seen["set_status"] = (
        fake_say, fake_prompts, fake_status)
    return seen


def test_thread_started_listener_completes_and_receives_the_real_body(recorder):
    """CRITERION 2 for listener 1, plus the second defect: the call used to
    hardcode `body={}`, so completing without raising was NOT sufficient --
    `channel_id` resolved to None. This asserts the handler saw the payload."""
    assistant = _register()
    listener = assistant._thread_started_listeners[0].ack_function

    logger = mock.MagicMock()
    _run(listener(
        body=THREAD_STARTED_BODY, say=recorder["say"],
        set_suggested_prompts=recorder["set_suggested_prompts"],
        get_thread_context=mock.MagicMock(), logger=logger,
    ))

    assert "prompts" in recorder, (
        "set_suggested_prompts was never called -- the suggested-prompt feature "
        "is the whole point of this listener"
    )
    assert recorder["prompts"]["prompts"], "an empty prompt list was sent"
    assert recorder.get("said"), "no welcome message was sent"

    # The handler logs channel/thread from the body. If body were {} again,
    # these would be None -- which is exactly the pre-fix behaviour.
    logged = " ".join(str(c) for c in logger.info.call_args_list)
    assert "D82590CHAN" in logged and "1786000000.000100" in logged, (
        "the handler did not receive the real event body -- it saw "
        f"{logged!r}. A hardcoded body={{}} produces channel=None."
    )


def test_context_changed_listener_completes_and_receives_the_real_body():
    """CRITERION 2 for listener 2, with its OWN payload -- a shared payload
    would let a handler pass on someone else's data."""
    assistant = _register()
    listener = assistant._thread_context_changed_listeners[0].ack_function

    logger = mock.MagicMock()
    _run(listener(body=CONTEXT_CHANGED_BODY, logger=logger))

    logged = " ".join(str(c) for c in logger.info.call_args_list)
    assert "C8259NEWCTX" in logged, (
        f"the handler did not read the changed context from the body: {logged!r}"
    )


def test_user_message_listener_completes_through_the_real_handler(recorder):
    """CRITERION 2 for listener 3 -- the one site that was always correct.

    This drives the REAL `handle_user_message` and patches one seam deeper, at
    `streaming_integration.handle_user_message_with_streaming` (imported
    function-locally, so a module-attribute patch lands at call time).

    THE EARLIER VERSION OF THIS TEST WAS INERT AND ITS DOCSTRING WAS FALSE.
    It replaced the handler itself with `mock.AsyncMock()`, which accepts ANY
    keyword argument -- so no binding defect at this call site could ever turn
    it red, while the docstring claimed "a regression in the fix cannot break
    it silently". The 82.59 Q/A proved it by injecting `bogus_kwarg=1` into the
    call site and watching this test stay green. An AsyncMock subject is a
    fixture that cannot represent the failure it is supposed to catch.

    Driving the real handler makes a wrong kwarg a real TypeError. Note the
    mechanism precisely: for a KWARG defect the TypeError is raised at the call
    site in `on_user_message`, before the handler body runs -- the handler's own
    re-raise is what surfaces failures raised INSIDE the body (a gutted handler,
    a missing status clear). Both reach this test; they just arrive by different
    routes.
    """
    assistant = _register()
    listener = assistant._user_message_listeners[0].ack_function

    from backend.slack_bot import streaming_integration as si

    streamed = mock.AsyncMock()
    logger = mock.MagicMock()
    with mock.patch.object(si, "handle_user_message_with_streaming", new=streamed):
        _run(listener(
            body=USER_MESSAGE_BODY, client=mock.MagicMock(), say=recorder["say"],
            set_status=recorder["set_status"], logger=logger,
        ))

    # Completion, not invocation: the real handler ran end to end, set status
    # on both sides of the streaming call, and forwarded the real body.
    assert streamed.await_count == 1, "the real handler did not reach the stream seam"
    assert streamed.await_args.kwargs["body"] is USER_MESSAGE_BODY
    assert len(recorder.get("status", [])) == 2, (
        "expected status set then cleared -- the handler did not run to completion"
    )
    logged = " ".join(str(c) for c in logger.info.call_args_list)
    assert "U82590001" in logged, (
        f"the handler did not receive the real event body: {logged!r}"
    )


def test_no_listener_swallows_its_own_failure():
    """The listeners must not grow their own try/except -- Bolt already
    swallows, and a second layer would make a future defect invisible even in
    the stderr line that is currently the only signal."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    reg = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "register_assistant_lifecycle"
    )
    inner = [n for n in ast.walk(reg) if isinstance(n, ast.AsyncFunctionDef)]
    assert len(inner) == 3, f"expected 3 registered listeners, found {len(inner)}"
    for fn in inner:
        assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)], (
            f"listener {fn.name} wraps its body in try/except; Bolt already "
            "swallows listener exceptions, so this would hide the next defect "
            "completely"
        )


# ---------------------------------------------------------------------------
# criterion 3 -- the swallow finding, verified against the installed package
# ---------------------------------------------------------------------------


def test_bolt_swallows_listener_exceptions_verified_not_assumed():
    """CRITERION 3. The operator-visible impact rests on this, so it is checked
    against the INSTALLED package rather than documentation or memory: the
    default listener error handler only logs, therefore a TypeError in our
    call cannot reach Slack or the operator as an error."""
    from slack_bolt.listener.async_listener_error_handler import (
        AsyncDefaultListenerErrorHandler,
    )

    src = inspect.getsource(AsyncDefaultListenerErrorHandler.handle)
    assert "logger.exception" in src, (
        "Bolt's default listener error handler no longer just logs -- the "
        "'silent failure' rationale recorded for this step must be re-derived"
    )
    assert "raise" not in src, (
        "the default handler now re-raises; the blast-radius finding is stale"
    )


def test_the_handler_signatures_were_not_edited_to_fit_the_call_sites():
    """Git blame showed the CALL SITE drifted while the handlers stayed put
    since 0da5a907. Fixing this by loosening the handlers (e.g. adding
    **kwargs) would make the defect unfalsifiable forever, so pin the real
    signatures."""
    sigs = {
        "handle_thread_started": ["body", "say", "set_suggested_prompts", "logger"],
        "handle_context_changed": ["body", "logger"],
        "handle_user_message": ["body", "client", "say", "set_status", "logger"],
    }
    for method, expected in sigs.items():
        params = inspect.signature(getattr(AssistantLifecycleHandler, method)).parameters
        assert [p for p in params if p != "self"] == expected, (
            f"{method}'s signature changed; the fix was supposed to be call-site only"
        )
        assert not any(p.kind is p.VAR_KEYWORD for p in params.values()), (
            f"{method} grew **kwargs -- that would absorb ANY wrong call and "
            "make this whole class of defect permanently invisible"
        )
