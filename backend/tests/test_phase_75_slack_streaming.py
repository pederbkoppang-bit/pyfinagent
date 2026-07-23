"""phase-75.7 -- Slack assistant streaming await-correctness + P0 pager integrity.

Four defects on the Slack bot's single async event loop:
  pysvc-01  chat_stream/append/stop are coroutines on the live AsyncWebClient but were
            called synchronously, so every non-DIRECT assistant message died with
            AttributeError ('coroutine' has no attribute 'append').
  pysvc-02  the agent fan-out blocked the loop with concurrent.futures.
  gap1-06   _get_live_data / _read_status ran sync in async handlers (~41s stalls).
  gap1-02   the P0 iMessage pager discarded the CompletedProcess and logged 'sent'
            unconditionally -- a silent kill-switch pager failure the operator never saw.

TEST TOOLING: pytest-asyncio is NOT installed -- drive coroutines via asyncio.run() inside
sync defs (repo precedent). The un-awaited-coroutine RuntimeWarning fires at GC time
(non-deterministic), so criterion 1 is anchored on the DETERMINISTIC AttributeError that a
sync-called coroutine's .append() raises; the warning filter, where used, is module-scoped
and forces gc.collect() -- never a global error::RuntimeWarning (would break the suite).
"""

from __future__ import annotations

import ast
import asyncio
import gc
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "backend/slack_bot/streaming_integration.py"
SCHED = REPO / "backend/slack_bot/scheduler.py"


# ── async stub streamer + client (chat_stream/append/stop are coroutines) ────

class _StubStreamer:
    def __init__(self, calls: list):
        self._calls = calls

    async def append(self, *a, **k):
        self._calls.append(("append", k))

    async def stop(self, *a, **k):
        self._calls.append(("stop", k))


class _StubAsyncClient:
    """Mirrors the live AsyncWebClient shape: chat_stream is a coroutine returning a
    streamer whose append/stop are coroutines. A SYNC caller (`streamer.append(...)`
    without await) would hit a coroutine object -> AttributeError, exactly like prod."""

    def __init__(self):
        self.calls: list = []

    async def chat_stream(self, *a, **k):
        self.calls.append(("chat_stream", k))
        return _StubStreamer(self.calls)


def _classification(complexity, agents=None, agent_type=None):
    from backend.agents.agent_definitions import AgentType
    return SimpleNamespace(
        agent_type=agent_type or AgentType.MAIN,
        complexity=complexity,
        parallel_agents=agents,
        confidence=0.9,
    )


# ── criterion 1: both _stream_* helpers complete with an ASYNC client, no un-awaited coro ──

def test_stream_simple_response_awaits_streamer_and_completes():
    import backend.slack_bot.streaming_integration as si
    from backend.agents.agent_definitions import QueryComplexity

    client = _StubAsyncClient()

    class _Orch:
        async def _execute_full_flow(self, *a, **k):
            return {"response": "hello world", "token_usage": {"input": 1, "output": 2},
                    "processing_time_ms": 5, "agent_type": "main"}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # module-scoped, in-context only
        asyncio.run(si._stream_simple_response(
            client, _Orch(), _classification(QueryComplexity.SIMPLE),
            "hi", "U1", "C1", "T1", {"team_id": "TM"}, si.logger,
        ))
        gc.collect()  # force any un-awaited-coroutine warning to surface deterministically

    kinds = [c[0] for c in client.calls]
    assert "chat_stream" in kinds and "append" in kinds and "stop" in kinds, (
        f"streamer coroutines were not driven: {kinds}"
    )


def test_stream_complex_task_plan_awaits_streamer_and_fans_out():
    import backend.slack_bot.streaming_integration as si
    from backend.agents.agent_definitions import AgentType, QueryComplexity

    client = _StubAsyncClient()
    agents = [AgentType.QA, AgentType.RESEARCH]

    class _Orch:
        def call_single_agent_sync(self, agent_type, **k):
            return {"response": f"{agent_type.value} done", "token_usage": {"input": 1, "output": 1},
                    "processing_time_ms": 3}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        asyncio.run(si._stream_complex_task_plan(
            client, _Orch(), _classification(QueryComplexity.COMPLEX, agents=agents),
            "analyze", "U1", "C1", "T1", {"team_id": "TM"}, si.logger,
        ))
        gc.collect()

    kinds = [c[0] for c in client.calls]
    assert "chat_stream" in kinds
    # both agents' task cards streamed + the final synthesis + stop
    assert kinds.count("append") >= 4 and "stop" in kinds


def test_fanout_one_agent_error_does_not_abort_the_others():
    """The 3-tuple `_run_agent` gives per-agent error isolation: if one agent raises
    inside to_thread, the OTHER agent's card + the synthesis still render and the fan-out
    completes (a bare re-raising `await done` would lose this). Regression-protects the
    claim experiment_results makes."""
    import backend.slack_bot.streaming_integration as si
    from backend.agents.agent_definitions import AgentType, QueryComplexity

    client = _StubAsyncClient()
    agents = [AgentType.QA, AgentType.RESEARCH]

    class _Orch:
        def call_single_agent_sync(self, agent_type, **k):
            if agent_type == AgentType.QA:
                raise RuntimeError("QA boom")
            return {"response": "research done", "token_usage": {"input": 1, "output": 1},
                    "processing_time_ms": 3}

    asyncio.run(si._stream_complex_task_plan(
        client, _Orch(), _classification(QueryComplexity.COMPLEX, agents=agents),
        "analyze", "U1", "C1", "T1", {"team_id": "TM"}, si.logger,
    ))
    # fan-out completed (stop called) despite one agent raising, and both agents' cards
    # (one error, one complete) were streamed before the synthesis.
    assert any(c[0] == "stop" for c in client.calls), "fan-out did not complete after one agent raised"
    assert sum(1 for c in client.calls if c[0] == "append") >= 4


def test_sync_streamer_call_would_raise_attributeerror():
    """Anchors criterion 1 on the DETERMINISTIC failure mode: calling .append() on the
    coroutine returned by an un-awaited chat_stream raises AttributeError -- which is what
    prod did before the fix. Proves the async-client stub actually reproduces the bug."""
    client = _StubAsyncClient()
    coro = client.chat_stream(channel="C1")  # NOT awaited -> a coroutine object
    try:
        with pytest.raises(AttributeError):
            coro.append(markdown_text="x")  # 'coroutine' object has no attribute 'append'
    finally:
        coro.close()  # avoid a real un-awaited-coroutine warning from this probe


# ── criterion 2: no concurrent.futures.as_completed / future.result() in the complex helper ──

def _func_source(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_complex_helper_has_no_blocking_concurrent_futures():
    """Structural AST check (not a substring scan -- comments legitimately document the
    old blocking code, so a substring scan would false-fail on this file's own comment,
    the phase-75.5 comment-token trap). Walk actual CALL nodes."""
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = _func_source(tree, "_stream_complex_task_plan")

    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]

    def _is_bare_name(call, name):
        return isinstance(call.func, ast.Name) and call.func.id == name

    def _is_attr(call, obj, attr):
        return (isinstance(call.func, ast.Attribute) and call.func.attr == attr
                and isinstance(call.func.value, ast.Name) and call.func.value.id == obj)

    # forbidden: concurrent.futures.as_completed (imported bare) + any .result() call + ThreadPoolExecutor
    assert not any(_is_bare_name(c, "as_completed") for c in calls), "concurrent.futures.as_completed still used"
    assert not any(isinstance(c.func, ast.Attribute) and c.func.attr == "result" for c in calls), "future.result() still called"
    assert not any(_is_bare_name(c, "ThreadPoolExecutor") for c in calls), "ThreadPoolExecutor still used"
    # module no longer imports concurrent.futures
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("concurrent")]
    assert imports == [], "concurrent.futures still imported"

    # required: asyncio.create_task, asyncio.to_thread, asyncio.as_completed, awaited results
    assert any(_is_attr(c, "asyncio", "to_thread") for c in calls), "no asyncio.to_thread fan-out"
    assert any(_is_attr(c, "asyncio", "create_task") for c in calls), "no asyncio.create_task"
    assert any(_is_attr(c, "asyncio", "as_completed") for c in calls), "no asyncio.as_completed"
    assert any(isinstance(n, ast.Await) for n in ast.walk(fn)), "results not awaited"


# ── criterion 3: the 3 blocking calls dispatch via asyncio.to_thread ──

@pytest.mark.parametrize("relpath,callee", [
    ("backend/slack_bot/app_home.py", "_get_live_data"),
    ("backend/slack_bot/commands.py", "_read_status"),
])
def test_blocking_call_dispatched_via_to_thread(relpath, callee):
    src = (REPO / relpath).read_text(encoding="utf-8")
    assert f"asyncio.to_thread({callee})" in src, (
        f"{callee} is not dispatched via asyncio.to_thread in {relpath}"
    )
    # and NOT called bare in an await-less statement
    assert f"= {callee}()" not in src and f" {callee}()\n" not in src


def test_reaction_handler_push_still_to_thread_wrapped():
    """The reaction push was ALREADY to_thread-wrapped (phase-75.2.1). Criterion 3 lists
    it; assert it remains wrapped (this step must not have un-wrapped or double-wrapped it)."""
    src = (REPO / "backend/slack_bot/commands.py").read_text(encoding="utf-8")
    assert "await asyncio.to_thread(" in src
    # no accidental double-wrap
    assert "asyncio.to_thread(asyncio.to_thread" not in src


# ── criterion 4 + 6: P0 pager exit-code integrity ──

def _run_escalation(returncode, stderr="boom"):
    """Drive send_trading_escalation with subprocess.run stubbed. Returns (slack_posts,
    log_records) so the test can assert the failure was recorded on L1 and logged ERROR."""
    from backend.slack_bot import scheduler

    posts: list = []

    class _Client:
        async def chat_postMessage(self, **k):
            posts.append(k)

    app = SimpleNamespace(client=_Client())

    fake_proc = SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)
    with patch.object(scheduler, "get_settings",
                      return_value=SimpleNamespace(slack_channel_id="C1",
                                                   escalation_phone_e164="+100")), \
         patch("subprocess.run", return_value=fake_proc), \
         patch.object(scheduler, "format_escalation_alert", return_value=[]), \
         patch.object(scheduler.logger, "error") as mock_err, \
         patch.object(scheduler.logger, "warning") as mock_warn:
        asyncio.run(scheduler.send_trading_escalation(
            app, "P0", "Kill Switch Activated", {"nav": "9000"},
        ))
    return posts, mock_err, mock_warn


def test_pager_failure_logs_error_and_posts_slack_fallback():
    posts, mock_err, mock_warn = _run_escalation(returncode=1, stderr="delivery failed")
    # ERROR logged for the pager failure
    assert any("P0 iMessage pager FAILED" in str(c.args) for c in mock_err.call_args_list), (
        "returncode=1 did not log the pager-FAILED ERROR"
    )
    # 'iMessage escalation sent' success line NOT logged
    assert not any("iMessage escalation sent" in str(c.args) for c in mock_warn.call_args_list)
    # a Slack fallback line was posted so L1 records the L2 miss
    assert any("P0 iMessage pager FAILED" in (p.get("text") or "") for p in posts), (
        "no Slack fallback posted on pager failure"
    )


def test_pager_success_path_unchanged():
    posts, mock_err, mock_warn = _run_escalation(returncode=0)
    assert any("iMessage escalation sent" in str(c.args) for c in mock_warn.call_args_list)
    assert not any("P0 iMessage pager FAILED" in str(c.args) for c in mock_err.call_args_list)
    # only the L1 alert post, no pager-failure fallback
    assert not any("P0 iMessage pager FAILED" in (p.get("text") or "") for p in posts)


def test_pager_missing_binary_also_posts_fallback():
    """The exception path (imsg missing / timeout) is ALSO a silent-pager failure and must
    post the fallback (the research flagged the criterion only tests returncode=1)."""
    from backend.slack_bot import scheduler
    posts: list = []

    class _Client:
        async def chat_postMessage(self, **k):
            posts.append(k)

    app = SimpleNamespace(client=_Client())
    with patch.object(scheduler, "get_settings",
                      return_value=SimpleNamespace(slack_channel_id="C1",
                                                   escalation_phone_e164="+100")), \
         patch("subprocess.run", side_effect=FileNotFoundError("imsg not found")), \
         patch.object(scheduler, "format_escalation_alert", return_value=[]):
        asyncio.run(scheduler.send_trading_escalation(
            app, "P0", "Kill Switch Activated", {"nav": "9000"},
        ))
    assert any("P0 iMessage pager FAILED" in (p.get("text") or "") for p in posts)


def test_no_phone_literal_in_scheduler():
    src = SCHED.read_text(encoding="utf-8")
    assert "+4794810537" not in src, "phone literal still hardcoded in scheduler.py"
    assert "escalation_phone_e164" in src, "escalation recipient not resolved from settings"
