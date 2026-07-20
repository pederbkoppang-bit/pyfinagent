"""phase-75.2.1: close the two escalated items from 75.2.

(a) The immutable-criteria collision is RECORDED, never amended: this suite
    asserts the three affected steps' verification blocks are byte-identical
    to commit 256867d3 and that each carries a superseded_record.

(b) The push-approval request path is wired and fail-closed, and the approval
    is bound to WHAT WAS SHOWN (HEAD sha + TTL), not merely to a message ts.

Every behavioral guard here is mutation-tested; the evidence lives in
handoff/current/live_check_75.2.1.md. A guard that cannot fail when its
subject is broken does not count -- phase-75.3 shipped three such guards.

All offline: git commands are stubbed, no network, no Slack.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.slack_bot import commands as cmd  # noqa: E402

BASELINE_COMMIT = "256867d3"
COLLIDING_STEPS = ("4.14.4", "4.14.24", "4.17.9")


# ── (a) the collision is recorded, not amended ───────────────────────

def _load_masterplan(ref: str | None = None) -> dict:
    if ref is None:
        return json.loads((REPO_ROOT / ".claude/masterplan.json").read_text(encoding="utf-8"))
    out = subprocess.check_output(
        ["git", "show", f"{ref}:.claude/masterplan.json"],
        cwd=str(REPO_ROOT), text=True,
    )
    return json.loads(out)


def _steps_by_id(mp: dict) -> dict:
    return {s["id"]: s for ph in mp["phases"] for s in ph.get("steps", [])}


@pytest.mark.parametrize("step_id", COLLIDING_STEPS)
def test_verification_is_byte_identical_to_baseline(step_id):
    """The whole point of part (a): annotate without amending."""
    before = _steps_by_id(_load_masterplan(BASELINE_COMMIT))[step_id]
    after = _steps_by_id(_load_masterplan())[step_id]
    assert json.dumps(before["verification"], sort_keys=True) == \
           json.dumps(after["verification"], sort_keys=True), \
           f"{step_id}: verification block was AMENDED, not annotated"


@pytest.mark.parametrize("step_id", COLLIDING_STEPS)
def test_step_stays_done_and_carries_a_superseded_record(step_id):
    step = _steps_by_id(_load_masterplan())[step_id]
    assert step["status"] == "done"          # the work WAS done; the artifact is gone
    rec = step.get("superseded_record")
    assert rec, f"{step_id}: no superseded_record"
    assert rec["criteria_amended"] is False
    assert rec["retired_by_commit"] == "f55e6973"
    assert rec["still_runnable"] is False


def test_4_17_9_record_names_both_causes():
    """It had a name mismatch from day one AND 75.2 removed its real target.
    Recording only one of those misattributes the breakage."""
    rec = _steps_by_id(_load_masterplan())["4.17.9"]["superseded_record"]
    assert rec.get("already_broken_before_retirement") is True
    reason = rec["reason"]
    assert "self_update_audit_test.py" in reason      # the never-existing script
    assert "smoke_test_4_17_9.py" in reason           # the real target 75.2 deleted
    assert "scope_disclosure" in rec                  # the 10-member family


def test_no_other_done_step_references_the_deleted_modules():
    """Guards the sweep's conclusion that these three are the complete set."""
    dead = ("slack_bot.self_update", "slack_bot.assistant_handler",
            "slack_bot.governance", "slack_bot.mcp_tools",
            "slack_bot.streaming_handler", "slack_bot.context_management",
            "slack_bot/self_update.py", "slack_bot/assistant_handler.py")
    offenders = []
    for step in _steps_by_id(_load_masterplan()).values():
        if step.get("status") != "done" or step["id"] in COLLIDING_STEPS:
            continue
        v = step.get("verification")
        blob = json.dumps(v) if v is not None else ""
        # 75.2 itself names the modules but ASSERTS THEIR ABSENCE -- correct.
        if step["id"] in ("75.2", "75.2.1"):
            continue
        if any(d in blob for d in dead):
            offenders.append(step["id"])
    assert offenders == [], f"un-annotated collisions: {offenders}"


# ── (b) push-approval request path ───────────────────────────────────

class _FakeSlackResponse:
    """Mirrors slack_sdk AsyncSlackResponse: exposes .get() but is NOT a dict
    subclass. Module-scope so test_say_stub_is_not_a_dict can inspect it -- a
    dict stub here silently hides the production bug where
    `isinstance(resp, dict)` left the whole push path inert.
    """

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


OP = "U_OPERATOR"
INTRUDER = "U_INTRUDER"
CHANNEL = "C0ANTGNNK8D"
HEAD_SHA = "a" * 40


@pytest.fixture
def wired(monkeypatch):
    """Register the real handlers against a stub Bolt app.

    The 75.2 suite's stub captured only `event` handlers; message handlers were
    discarded by `lambda fn: fn`, so this one captures both plus matchers.
    """
    handlers: dict = {}
    messages: list = []
    says: list = []
    pushes: list = []

    class _App:
        def event(self, name):
            def deco(fn):
                handlers[name] = fn
                return fn
            return deco

        def message(self, pattern=None, matchers=None, **kw):
            def deco(fn):
                messages.append({"pattern": pattern, "matchers": matchers or [], "fn": fn})
                return fn
            return deco

        def command(self, *a, **k):
            return lambda fn: fn

        def action(self, *a, **k):
            return lambda fn: fn

    from backend.config.settings import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "slack_operator_user_id", OP, raising=False)
    monkeypatch.setattr(settings, "slack_channel_id", CHANNEL, raising=False)

    def _fake_check_output(argv, **kwargs):
        if "rev-parse" in argv:
            return HEAD_SHA + "\n"
        if "log" in argv:
            return "abc1234 first commit\ndef5678 second commit\n"
        if "push" in argv:
            pushes.append(argv)
            return "pushed\n"
        return ""

    monkeypatch.setattr(cmd.subprocess, "check_output", _fake_check_output)

    async def _say(**kwargs):
        says.append(kwargs)
        return _FakeSlackResponse(
            {"ts": "999.999", "channel": kwargs.get("channel", CHANNEL)}
        )

    cmd.register_commands(_App())
    cmd._pending_push_ts.clear()

    push_handler = None
    for entry in messages:
        pat = getattr(entry["pattern"], "pattern", "")
        if "PUSH" in str(pat):
            push_handler = entry
    assert push_handler is not None, "push-request handler was never registered"

    return {
        "push": push_handler, "reaction": handlers["reaction_added"],
        "say": _say, "says": says, "pushes": pushes, "settings": settings,
        "registered": messages,
    }


def _msg(user=OP, channel=CHANNEL, text="PUSH"):
    return {"user": user, "channel": channel, "text": text, "ts": "111.1"}


def _run(coro):
    return asyncio.run(coro)


def test_push_handler_registers_before_the_catch_all(wired):
    """Bolt dispatch is first-match-wins, so ORDER is the property under test.

    The previous version of this test asserted only `is not None`, which the
    fixture already guarantees -- it would have passed with the handler
    registered after the catch-all. Assert the actual indices.
    """
    order = [e["fn"].__name__ for e in wired["registered"]]
    assert "handle_push_request" in order, order
    assert "handle_any_message" in order, order
    assert order.index("handle_push_request") < order.index("handle_any_message"), order


def test_say_stub_matches_the_production_response_shape(wired):
    """Pin the FIXTURE, not a library fact.

    The non-dict stub is the single thing that makes a revert to
    `isinstance(resp, dict)` detectable. An earlier version of this test
    asserted only that AsyncSlackResponse is not a dict -- true, unfalsifiable
    by anything we control, and it stayed green while a dict-stub regression
    plus the isinstance bug together left production inert (Q/A M9).

    So: assert the object the fixture ACTUALLY returns has production's shape.
    """
    from slack_sdk.web.async_slack_response import AsyncSlackResponse

    resp = asyncio.run(wired["say"](channel=CHANNEL, text="probe"))

    # Production's contract: exposes .get(), is NOT a dict.
    assert hasattr(resp, "get"), "stub must expose .get() like AsyncSlackResponse"
    assert not isinstance(resp, dict), (
        "stub regressed to a dict -- it can no longer detect the isinstance "
        "bug that left the push path inert"
    )
    assert resp.get("ts")

    # And the shape it mirrors is genuinely non-dict upstream.
    assert not issubclass(AsyncSlackResponse, dict)
    assert hasattr(AsyncSlackResponse, "get")


def test_trigger_does_not_collide_with_the_operator_token_grammar():
    """The two message grammars must stay disjoint.

    Uses the PRODUCTION regex objects: re-declaring copies here would pass
    even if the real patterns drifted apart. Note the ordering hazard runs
    push-first (it registers at index 0), so widening the push pattern would
    make it swallow operator TOKENS -- not the reverse.
    """
    assert cmd.TOKEN_KEYWORD_RE.match("PUSH REQUEST: main")   # the trap avoided
    assert not cmd.TOKEN_KEYWORD_RE.match("PUSH")             # our trigger
    assert cmd.PUSH_REQUEST_KEYWORD_RE.match("PUSH")
    assert not cmd.PUSH_REQUEST_KEYWORD_RE.match("PUSH REQUEST: main")
    assert not cmd.PUSH_REQUEST_KEYWORD_RE.match("65.2 EU SCREENER: ON")


def test_non_operator_request_posts_nothing_and_registers_nothing(wired):
    _run(wired["push"]["fn"](
        message=_msg(user=INTRUDER), say=wired["say"], logger=cmd.logger))
    assert wired["says"] == []
    assert cmd._pending_push_ts == {}


def test_unset_operator_id_is_fail_closed(wired, monkeypatch):
    monkeypatch.setattr(wired["settings"], "slack_operator_user_id", "", raising=False)
    _run(wired["push"]["fn"](
        message=_msg(user=OP), say=wired["say"], logger=cmd.logger))
    assert wired["says"] == []
    assert cmd._pending_push_ts == {}


def test_operator_request_posts_to_approval_channel_with_the_commit_list(wired):
    _run(wired["push"]["fn"](
        message=_msg(), say=wired["say"], logger=cmd.logger))

    posted = [s for s in wired["says"] if s.get("channel") == CHANNEL]
    assert posted, "nothing posted to the approval channel"
    body = posted[-1]["text"]
    assert "first commit" in body and "second commit" in body   # what is signed
    assert HEAD_SHA[:12] in body
    assert "last-known origin/main" in body                     # no false freshness
    assert "999.999" in cmd._pending_push_ts                    # bot ts registered


def test_register_requires_an_explicit_head_sha():
    """No default: a default would silently disable the TOCTOU re-validation."""
    with pytest.raises(TypeError):
        cmd.register_push_approval_request("1.1")          # missing head_sha


def test_registered_entry_binds_the_shown_sha_and_an_expiry(wired):
    _run(wired["push"]["fn"](message=_msg(), say=wired["say"], logger=cmd.logger))
    sha, expires_at = cmd._pending_push_ts["999.999"]
    assert sha == HEAD_SHA
    assert expires_at > time.monotonic()


def test_nothing_to_push_registers_nothing(wired, monkeypatch):
    def _no_commits(argv, **kwargs):
        return HEAD_SHA + "\n" if "rev-parse" in argv else ""
    monkeypatch.setattr(cmd.subprocess, "check_output", _no_commits)

    _run(wired["push"]["fn"](message=_msg(), say=wired["say"], logger=cmd.logger))
    assert cmd._pending_push_ts == {}


def test_git_inspection_runs_off_the_event_loop(wired, monkeypatch):
    seen = []
    real = asyncio.to_thread

    async def _spy(fn, *a, **k):
        seen.append(fn)
        return await real(fn, *a, **k)

    monkeypatch.setattr(cmd.asyncio, "to_thread", _spy)
    _run(wired["push"]["fn"](message=_msg(), say=wired["say"], logger=cmd.logger))
    assert seen, "git inspection did not go through asyncio.to_thread"


# ── (b) the approval still gates the push, and now binds the sha ─────

def _react(ts="999.999", user=OP, reaction="white_check_mark"):
    return {"user": user, "reaction": reaction,
            "item": {"channel": CHANNEL, "ts": ts}}


def test_full_flow_request_then_operator_reaction_pushes(wired):
    _run(wired["push"]["fn"](message=_msg(), say=wired["say"], logger=cmd.logger))
    _run(wired["reaction"](event=_react(), say=wired["say"]))
    assert len(wired["pushes"]) == 1


def test_approval_is_single_use(wired):
    _run(wired["push"]["fn"](message=_msg(), say=wired["say"], logger=cmd.logger))
    _run(wired["reaction"](event=_react(), say=wired["say"]))
    _run(wired["reaction"](event=_react(), say=wired["say"]))
    assert len(wired["pushes"]) == 1


def test_unregistered_ts_still_pushes_nothing(wired):
    _run(wired["reaction"](event=_react(ts="000.000"), say=wired["say"]))
    assert wired["pushes"] == []


def test_head_moving_after_approval_refuses_the_push(wired, monkeypatch):
    """The TOCTOU guard: commits the operator never saw must not ride an
    approval granted for a different HEAD."""
    _run(wired["push"]["fn"](message=_msg(), say=wired["say"], logger=cmd.logger))

    moved = "b" * 40

    def _moved(argv, **kwargs):
        if "rev-parse" in argv:
            return moved + "\n"
        if "push" in argv:
            wired["pushes"].append(argv)
            return "pushed\n"
        return ""

    monkeypatch.setattr(cmd.subprocess, "check_output", _moved)
    _run(wired["reaction"](event=_react(), say=wired["say"]))

    assert wired["pushes"] == [], "HEAD moved but the push went through anyway"
    assert any("HEAD moved" in (s.get("text") or "") for s in wired["says"])


def test_expired_approval_refuses_the_push(wired):
    _run(wired["push"]["fn"](message=_msg(), say=wired["say"], logger=cmd.logger))
    sha, _ = cmd._pending_push_ts["999.999"]
    cmd._pending_push_ts["999.999"] = (sha, time.monotonic() - 1.0)   # expired

    _run(wired["reaction"](event=_react(), say=wired["say"]))
    assert wired["pushes"] == []
    assert any("expired" in (s.get("text") or "").lower() for s in wired["says"])


def test_non_operator_reaction_on_a_valid_request_pushes_nothing(wired):
    """75.2's guarantee must survive the request path being wired."""
    _run(wired["push"]["fn"](message=_msg(), say=wired["say"], logger=cmd.logger))
    _run(wired["reaction"](event=_react(user=INTRUDER), say=wired["say"]))
    assert wired["pushes"] == []
