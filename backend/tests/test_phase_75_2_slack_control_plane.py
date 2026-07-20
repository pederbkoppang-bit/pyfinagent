"""phase-75.2: Slack control-plane authorization + dead-plane removal.

The step's immutable verification command is a substring check; it cannot
prove the behaviours criteria 1, 3, 4 and 6 actually describe. These tests
cover that gap:

  criterion 1 -- reaction sink: non-operator / unset-operator / untracked-ts
                 all perform NO push; the push runs via asyncio.to_thread
  criterion 2 -- the six dead modules are gone and unimportable
  criterion 3 -- deploy verbs refuse BEFORE any LLM/orchestrator call
  criterion 4 -- per-user rate limit + one JSONL audit record per interaction
  criterion 6 -- append_operator_token refuses unauthorized records at the sink
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging

import pytest

from backend.slack_bot import assistant_guards as ag
from backend.slack_bot import commands as cmd
from backend.slack_bot import operator_tokens as ot

OP = "U_OPERATOR"
INTRUDER = "U_INTRUDER"
CHANNEL = "C0ANTGNNK8D"


# ── criterion 2: the dead plane is gone ──────────────────────────────

@pytest.mark.parametrize("mod", [
    "self_update", "assistant_handler", "governance",
    "mcp_tools", "streaming_handler", "context_management",
])
def test_dead_modules_are_unimportable(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"backend.slack_bot.{mod}")


# ── criterion 1: the reaction sink ───────────────────────────────────

@pytest.fixture
def reaction_handler(monkeypatch):
    """Register the real handler and hand back (handler, push_calls, says)."""
    push_calls: list = []
    says: list = []

    def _fake_check_output(*args, **kwargs):
        push_calls.append(args)
        return "ok"

    monkeypatch.setattr(cmd.subprocess, "check_output", _fake_check_output)

    async def _say(**kwargs):
        says.append(kwargs)

    handlers: dict = {}

    class _App:
        def event(self, name):
            def deco(fn):
                handlers[name] = fn
                return fn
            return deco

        def message(self, *a, **k):
            return lambda fn: fn

        def command(self, *a, **k):
            return lambda fn: fn

        def action(self, *a, **k):
            return lambda fn: fn

    cmd.register_commands(_App())
    cmd._pending_push_ts.clear()
    return handlers["reaction_added"], push_calls, says, _say


def _event(user=OP, ts="111.1", channel=CHANNEL, reaction="white_check_mark"):
    return {"user": user, "reaction": reaction,
            "item": {"channel": channel, "ts": ts}}


# phase-75.2.1: register_push_approval_request now REQUIRES head_sha (no
# default -- a default would silently disable the TOCTOU re-validation on a
# git-push authorization path). These 75.2 tests target identity / ts-binding /
# single-use / to_thread, so they pass head_sha="" to opt out of the sha
# re-validation deliberately; that binding has its own suite in
# backend/tests/test_phase_75_2_1_push_approval.py.


def _set_operator(monkeypatch, value):
    from backend.config.settings import get_settings
    settings = get_settings()
    monkeypatch.setattr(settings, "slack_operator_user_id", value, raising=False)


def test_non_operator_reaction_performs_no_push(reaction_handler, monkeypatch):
    handler, push_calls, _, say = reaction_handler
    _set_operator(monkeypatch, OP)
    cmd.register_push_approval_request("111.1", head_sha="")
    asyncio.run(handler(event=_event(user=INTRUDER), say=say))
    assert push_calls == []


def test_unset_operator_is_fail_closed(reaction_handler, monkeypatch):
    handler, push_calls, _, say = reaction_handler
    _set_operator(monkeypatch, "")
    cmd.register_push_approval_request("111.1", head_sha="")
    asyncio.run(handler(event=_event(user=OP), say=say))
    assert push_calls == []


def test_untracked_ts_performs_no_push(reaction_handler, monkeypatch):
    handler, push_calls, _, say = reaction_handler
    _set_operator(monkeypatch, OP)
    # No register_push_approval_request call: nothing is approvable.
    asyncio.run(handler(event=_event(user=OP, ts="999.9"), say=say))
    assert push_calls == []


def test_operator_on_tracked_ts_pushes_once_then_is_single_use(
    reaction_handler, monkeypatch
):
    handler, push_calls, _, say = reaction_handler
    _set_operator(monkeypatch, OP)
    cmd.register_push_approval_request("111.1", head_sha="")
    asyncio.run(handler(event=_event(), say=say))
    assert len(push_calls) == 1
    # Replaying the same approval must not push again.
    asyncio.run(handler(event=_event(), say=say))
    assert len(push_calls) == 1


def test_push_runs_off_the_event_loop(reaction_handler, monkeypatch):
    """The subprocess must go through asyncio.to_thread, not run inline."""
    handler, push_calls, _, say = reaction_handler
    _set_operator(monkeypatch, OP)
    cmd.register_push_approval_request("111.1", head_sha="")

    seen = []
    real_to_thread = asyncio.to_thread

    async def _spy(fn, *a, **k):
        seen.append(fn)
        return await real_to_thread(fn, *a, **k)

    monkeypatch.setattr(cmd.asyncio, "to_thread", _spy)
    asyncio.run(handler(event=_event(), say=say))
    assert seen, "git push did not go through asyncio.to_thread"


# ── criterion 3: pre-LLM deploy refusal ──────────────────────────────

# Every surface the DELETED matcher handled, recovered verbatim from
# `git show HEAD:backend/slack_bot/self_update.py::handle_deploy_command`.
# This list is the parity contract: if the refusal ever narrows again, these
# fail. (The first implementation of _DEPLOY_VERBS silently missed 12 of
# these, including bare "deploy" -- caught by Q/A wf_160a3771-7b7.)
_DELETED_MATCHER_SURFACE = [
    "deploy update", "deploy pull", "update bot", "pull and restart",
    "deploy status", "deploy info", "git status",
    "deploy diff", "deploy changes", "what changed",
    "deploy rollback", "deploy revert", "rollback",
    "deploy logs", "deploy history",
    "deploy cleanup", "deploy clean", "cleanup", "clean old",
    "deploy",                 # the old startswith("deploy") catch-all
    "deploy anything else",   # ditto
]


@pytest.mark.parametrize("text", _DELETED_MATCHER_SURFACE)
def test_deleted_matcher_surface_is_fully_covered(text):
    assert ag.is_deploy_request(text) is True


@pytest.mark.parametrize("text", [
    "DEPLOY UPDATE", "  deploy   ", "please deploy the bot", "roll back the deploy",
])
def test_deploy_detection_is_case_and_phrasing_tolerant(text):
    """Criterion 3 says a message CONTAINING a deploy verb -- broader than
    the old whole-message matcher, deliberately."""
    assert ag.is_deploy_request(text) is True


@pytest.mark.parametrize("text", [
    "what is the portfolio nav?",
    "run a backtest",
    "deployment history question",          # "deployment" != "deploy"
    "tell me what changed in the portfolio",  # not the exact alias
    "show me the git log",
])
def test_non_deploy_text_passes(text):
    assert ag.is_deploy_request(text) is False


def test_refusal_text_contains_the_asserted_literal():
    from backend.slack_bot import streaming_integration as si
    assert "deploy commands are disabled" in si.REFUSAL_TEXT


def test_deploy_request_refused_before_any_llm_call(monkeypatch, tmp_path):
    """The orchestrator must never be constructed for a deploy request."""
    from backend.slack_bot import streaming_integration as si

    monkeypatch.setattr(ag, "AUDIT_PATH", tmp_path / "assistant_audit.jsonl")

    def _boom():
        raise AssertionError("get_orchestrator() called on a deploy request")

    monkeypatch.setattr(si, "get_orchestrator", _boom)

    says: list = []

    async def _say(**kwargs):
        says.append(kwargs)

    async def _set_status(*a, **k):
        return None

    body = {"event": {"channel": "C1", "ts": "1.1", "user": OP,
                      "text": "deploy update"}}
    asyncio.run(si.handle_user_message_with_streaming(
        body=body, client=None, say=_say, set_status=_set_status,
        logger=logging.getLogger("test")))

    assert says and "deploy commands are disabled" in says[0]["text"]


# ── criterion 4: rate limit + audit ──────────────────────────────────

def test_rate_limit_blocks_after_the_window_budget():
    ag.reset_rate_limit()
    allowed = sum(1 for _ in range(50) if ag.rate_ok("U_RL", now=1000.0))
    assert allowed == ag._MAX_PER_WINDOW
    assert ag.rate_ok("U_RL", now=1000.0) is False


def test_rate_limit_is_per_user():
    ag.reset_rate_limit()
    for _ in range(ag._MAX_PER_WINDOW):
        ag.rate_ok("U_A", now=1000.0)
    assert ag.rate_ok("U_A", now=1000.0) is False
    assert ag.rate_ok("U_B", now=1000.0) is True


def test_rate_limit_recovers_after_two_quiet_windows():
    ag.reset_rate_limit()
    for _ in range(ag._MAX_PER_WINDOW):
        ag.rate_ok("U_C", now=1000.0)
    assert ag.rate_ok("U_C", now=1000.0) is False
    later = 1000.0 + 2 * ag._WINDOW_S + 1
    assert ag.rate_ok("U_C", now=later) is True


def test_audit_appends_one_jsonl_record_and_hashes_text(monkeypatch, tmp_path):
    path = tmp_path / "logs" / "assistant_audit.jsonl"
    monkeypatch.setattr(ag, "AUDIT_PATH", path)
    asyncio.run(ag.audit(user=OP, channel="C1", text="secret message",
                         outcome="accepted", agent="main"))
    asyncio.run(ag.audit(user=OP, channel="C1", text="another",
                         outcome="refused_deploy"))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["writer"] == "assistant_audit"
    assert rec["outcome"] == "accepted"
    assert len(rec["text_sha256"]) == 64
    # Raw message text must not be persisted.
    assert "secret message" not in path.read_text(encoding="utf-8")


# ── criterion 6: sink-level authorization ────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_token_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "TOKENS_PATH", tmp_path / "operator_tokens.jsonl")
    monkeypatch.setattr(ot, "_seen_events", set())
    yield


def test_sink_refuses_wrong_user_even_when_matcher_bypassed():
    res = asyncio.run(ot.append_operator_token(
        text="TEST TOKEN: PING", user=INTRUDER, channel=CHANNEL, ts="1.1",
        operator_user_id=OP, allowed_channels={CHANNEL}))
    assert res is None
    assert not ot.TOKENS_PATH.exists()


def test_sink_refuses_wrong_channel_even_when_matcher_bypassed():
    res = asyncio.run(ot.append_operator_token(
        text="TEST TOKEN: PING", user=OP, channel="C_RANDOM", ts="1.1",
        operator_user_id=OP, allowed_channels={CHANNEL}))
    assert res is None
    assert not ot.TOKENS_PATH.exists()


def test_sink_is_fail_closed_when_operator_unset():
    res = asyncio.run(ot.append_operator_token(
        text="TEST TOKEN: PING", user=OP, channel=CHANNEL, ts="1.1",
        operator_user_id="", allowed_channels={CHANNEL}))
    assert res is None
    assert not ot.TOKENS_PATH.exists()


def test_sink_accepts_the_authorized_path():
    res = asyncio.run(ot.append_operator_token(
        text="TEST TOKEN: PING", user=OP, channel=CHANNEL, ts="1.1",
        operator_user_id=OP, allowed_channels={CHANNEL}))
    assert res is not None
    line_no, rec = res
    assert line_no == 1 and rec["key"] == "TEST TOKEN"
