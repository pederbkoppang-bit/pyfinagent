"""phase-75.10 event-loop hygiene sweep tests.

Verifies the fixes from the audit75 S10 contract (handoff/current/contract_75.10.md):
  1. MASEventBus() constructs with NO running event loop (Python-3.14
     get_event_loop() crash repro) and _forward_remote spawns a single
     worker, never one thread per event.
  2. Zero `asyncio.get_event_loop` occurrences remain in the 3 named files.
  3. The named async routes contain no direct subprocess.run/sync-httpx/
     .result( calls outside an asyncio.to_thread wrapper; get_optimizer_status
     is a plain def; the p95 query still carries timeout=30.
  4. autonomous_loop.py's screen_universe call is asyncio.to_thread-wrapped
     (source + behavioral thread-of-execution proof) and the sibling
     universe/sector/peer-leadlag fetches are threaded/gathered.
  5. Fire-and-forget create_task sites propagate exceptions into their state
     dicts via add_done_callback (source assert per site + a real driven-to-
     exception behavioral test of the shared track_task() helper).
  6. main.py's lifespan finally shuts down both schedulers + cancels the
     prewarm task, and run_data_ingestion returns 202-immediately semantics
     with a pollable status.

No real BQ/yfinance/subprocess/network calls anywhere in this file --
everything here is source inspection, AST inspection, or offline mocks.
"""
from __future__ import annotations

import ast
import asyncio
import re
import threading
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[2]

MULTI_AGENT_ORCH = REPO / "backend/agents/multi_agent_orchestrator.py"
TASK_BUS = REPO / "backend/agents/task_bus.py"
MAS_EVENTS = REPO / "backend/agents/mas_events.py"
AUTONOMOUS_LOOP = REPO / "backend/services/autonomous_loop.py"
MAIN_PY = REPO / "backend/main.py"
API_MAS_EVENTS = REPO / "backend/api/mas_events.py"
API_PERFORMANCE = REPO / "backend/api/performance_api.py"
API_BACKTEST = REPO / "backend/api/backtest.py"
API_PORTFOLIO = REPO / "backend/api/portfolio.py"
API_CRON_DASHBOARD = REPO / "backend/api/cron_dashboard_api.py"
API_ANALYSIS = REPO / "backend/api/analysis.py"
API_PAPER_TRADING = REPO / "backend/api/paper_trading.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# Criterion 1: MASEventBus no-running-loop construction + single worker
# ══════════════════════════════════════════════════════════════════

def test_mas_event_bus_constructs_with_no_running_event_loop():
    """Plain def body -- NOT wrapped in asyncio.run(). This IS the Python-3.14
    get_event_loop() crash repro context (no current event loop)."""
    from backend.agents.mas_events import MASEventBus

    with suppress(RuntimeError):
        # Guard: if some OTHER test in this process left a loop set as
        # current on this thread, clear it so the repro context is genuine.
        asyncio.set_event_loop(None)

    bus = MASEventBus()  # must not raise
    assert bus is not None
    assert bus._remote_worker is None  # lazy -- zero threads before any emit


def test_forward_remote_spawns_single_worker_not_one_thread_per_event():
    import time as _time
    from backend.agents.mas_events import MASEventBus, MASEvent

    bus = MASEventBus()
    bus.remote_url = "http://127.0.0.1:1"  # unroutable; posts fail fast+quiet

    before = threading.active_count()
    for _ in range(25):
        bus.emit(MASEvent(event_type="test", agent="X"))

    # Let the worker thread actually start (it's spawned lazily on first emit).
    for _ in range(50):
        if bus._remote_worker is not None:
            break
        _time.sleep(0.01)

    assert bus._remote_worker is not None, "worker never started"
    assert bus._remote_worker.is_alive()
    assert bus._remote_worker.daemon
    after = threading.active_count()
    # Exactly one NEW thread for 25 emitted events (not 25).
    assert after - before == 1, f"expected +1 thread for 25 events, got +{after - before}"


def test_forward_remote_source_uses_queue_not_thread_per_event():
    src = _read(MAS_EVENTS)
    assert "threading.Thread(target=_send" not in src, "old thread-per-event pattern still present"
    assert "self._remote_queue.put_nowait" in src
    assert "self._remote_worker" in src


# ══════════════════════════════════════════════════════════════════
# Criterion 2: zero asyncio.get_event_loop occurrences (+ empty-list-trap guard)
# ══════════════════════════════════════════════════════════════════

_CRIT2_FILES = {
    "multi_agent_orchestrator.py": (MULTI_AGENT_ORCH, "class MultiAgentOrchestrator"),
    "task_bus.py": (TASK_BUS, "async def delegate"),
    "mas_events.py": (MAS_EVENTS, "class MASEventBus"),
}


def test_zero_get_event_loop_occurrences_in_named_files():
    for label, (path, marker) in _CRIT2_FILES.items():
        assert path.is_file(), f"{label} missing entirely"
        src = _read(path)
        # Empty-list-trap guard: prove this is the REAL, substantial file,
        # not an empty/truncated one that would vacuously pass the count==0
        # check below.
        assert marker in src, f"{label} sanity marker {marker!r} missing -- wrong/empty file?"
        assert len(src) > 1000, f"{label} suspiciously small ({len(src)} bytes)"
        count = src.count("asyncio.get_event_loop")
        assert count == 0, f"{label} still has {count} asyncio.get_event_loop occurrence(s)"


def test_get_running_loop_present_where_get_event_loop_used_to_be():
    # Positive control: confirm the replacement idiom actually landed, not
    # just that the banned string vanished (e.g. via deletion of the whole
    # call site, which criterion 2 alone wouldn't distinguish).
    orch_src = _read(MULTI_AGENT_ORCH)
    assert orch_src.count("asyncio.get_running_loop()") >= 7
    task_bus_src = _read(TASK_BUS)
    assert "asyncio.get_running_loop().create_future()" in task_bus_src


def test_orchestrator_dead_loop_var_deleted_not_converted():
    """orchestrator:430's `loop` was unused in _execute_full_flow -- contract
    says DELETE, not convert. Verify no dead `loop = asyncio.get_running_loop()`
    sits right before `bus = get_event_bus()` (its old neighbor)."""
    src = _read(MULTI_AGENT_ORCH)
    assert 'bus = get_event_bus()\n        run_id = make_run_id()' in src
    # The line immediately preceding must NOT be a get_running_loop assignment.
    idx = src.index('bus = get_event_bus()\n        run_id = make_run_id()')
    preceding = src[max(0, idx - 120):idx]
    assert "get_running_loop" not in preceding


# ══════════════════════════════════════════════════════════════════
# Criterion 3: AST scan of named async routes
# ══════════════════════════════════════════════════════════════════

def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _find_func(tree: ast.Module, name: str):
    """Hard-fail (return None, caller asserts) if the named function is
    missing or renamed -- never skip-green."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _build_parent_map(root: ast.AST) -> dict:
    parents = {}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            parents[child] = node
            stack.append(child)
    return parents


def _is_to_thread_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to_thread"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "asyncio"
    )


def _base_name(value) -> str | None:
    return value.id if isinstance(value, ast.Name) else None


def _find_banned_calls(func_node) -> list[str]:
    """Return a list of human-readable descriptions of banned direct sync
    calls (subprocess.run / module-level sync httpx / .result() ) found in
    `func_node`'s body OUTSIDE any asyncio.to_thread(...) argument and not
    directly awaited."""
    parents = _build_parent_map(func_node)

    exempt: set = set()
    for node in ast.walk(func_node):
        if _is_to_thread_call(node):
            for descendant in ast.walk(node):
                exempt.add(descendant)

    violations = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call) or node in exempt:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Await):
            continue  # genuinely awaited (e.g. httpx.AsyncClient().get) -- allowed

        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        if func.attr == "run" and _base_name(func.value) == "subprocess":
            violations.append(f"subprocess.run(...) at line {node.lineno}")
        elif func.attr in ("get", "post", "put", "delete", "request") and _base_name(func.value) == "httpx":
            violations.append(f"sync httpx.{func.attr}(...) at line {node.lineno}")
        elif func.attr == "result":
            violations.append(f".result(...) at line {node.lineno}")

    return violations


_CRIT3_ROUTES = [
    (API_MAS_EVENTS, "get_dashboard"),
    (API_PERFORMANCE, "get_llm_p95_latency"),
    (API_BACKTEST, "run_data_ingestion"),
    (API_PORTFOLIO, "list_positions"),
    (API_PORTFOLIO, "get_portfolio_performance"),
    (API_CRON_DASHBOARD, "get_log_tail"),
    (API_CRON_DASHBOARD, "get_all_jobs"),
]


def test_named_routes_exist_and_have_no_unwrapped_blocking_calls():
    for path, name in _CRIT3_ROUTES:
        tree = _parse(path)
        node = _find_func(tree, name)
        assert node is not None, f"{name} not found in {path.name} -- renamed/removed route (hard-fail)"
        violations = _find_banned_calls(node)
        assert not violations, f"{name} in {path.name} has unwrapped blocking calls: {violations}"


def test_mutation_stub_renamed_route_hard_fails_not_skip_green():
    """Guard-can-fail proof for the route-lookup itself: a nonexistent name
    must come back None (and the caller must assert on it), never silently
    report success."""
    tree = _parse(API_MAS_EVENTS)
    node = _find_func(tree, "get_dashboard_DOES_NOT_EXIST")
    assert node is None  # this IS the hard-fail signal the caller must assert on


def test_get_optimizer_status_is_plain_def():
    tree = _parse(API_BACKTEST)
    node = _find_func(tree, "get_optimizer_status")
    assert node is not None, "get_optimizer_status not found -- renamed/removed"
    assert isinstance(node, ast.FunctionDef), "get_optimizer_status must be a plain def (or fully to_thread-wrapped)"
    assert not isinstance(node, ast.AsyncFunctionDef)


def test_p95_query_still_carries_timeout_30():
    tree = _parse(API_PERFORMANCE)
    node = _find_func(tree, "get_llm_p95_latency")
    assert node is not None
    segment = ast.get_source_segment(_read(API_PERFORMANCE), node)
    assert segment is not None
    assert "timeout=30" in segment
    assert ".result(" in segment
    # And it must be inside a to_thread wrap now (not a bare unwrapped .result).
    assert "asyncio.to_thread" in segment


# ══════════════════════════════════════════════════════════════════
# Criterion 4: autonomous_loop.py to_thread wraps
# ══════════════════════════════════════════════════════════════════

def test_screen_universe_to_thread_substring_present():
    src = _read(AUTONOMOUS_LOOP)
    assert "await asyncio.to_thread(screen_universe" in src


def test_sibling_fetches_are_threaded_or_gathered():
    src = _read(AUTONOMOUS_LOOP)
    assert "await asyncio.to_thread(get_sp500_tickers)" in src
    assert "await asyncio.to_thread(build_sector_map, universe)" in src
    # peer_leadlag block: bounded-concurrency gather, not a serial for-loop.
    start = src.index("peer_leadlag_signals = {}")
    end = src.index("peer_leadlag_signals = compute_peer_leadlag_signals", start)
    block = src[start:end]
    assert "asyncio.Semaphore(8)" in block
    assert "asyncio.gather(" in block
    assert re.search(r"for t in target_tickers:\s*\n\s*try:", block) is None, (
        "serial per-ticker for/try loop still present -- perf-10 fix not applied"
    )


def test_screen_universe_wrap_runs_off_main_thread_and_preserves_kwargs_and_return():
    """Extracts the REAL current `screen_data = ...` Assign node from
    autonomous_loop.py via AST (not a hand-copied replica -- a mutation that
    un-threads this call changes what gets extracted), executes it with a
    mocked screen_universe, and proves (a) the mock ran OFF the main thread
    (genuine to_thread offload, not just a same-thread direct call that
    happens to return the same value) and (b) kwargs + return value survive
    the wrap unchanged."""
    tree = ast.parse(_read(AUTONOMOUS_LOOP), filename=str(AUTONOMOUS_LOOP))
    assign_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "screen_data"
        ):
            assign_node = node
            break
    assert assign_node is not None, "screen_data assignment not found in autonomous_loop.py"

    unparsed = ast.unparse(assign_node)
    assert "await asyncio.to_thread(screen_universe" in unparsed

    main_thread = threading.main_thread()
    seen = {}
    sentinel = [{"ticker": "AAPL", "sentinel": True}]

    def mock_screen_universe(**kwargs):
        seen["thread"] = threading.current_thread()
        seen["kwargs"] = kwargs
        return sentinel

    ns = {
        "asyncio": asyncio,
        "screen_universe": mock_screen_universe,
        "universe": ["AAPL", "MSFT"],
        "_sector_lookup": None,
        "short_interest_lookup": None,
        "settings": SimpleNamespace(short_interest_threshold=0.1),
    }
    stmt_src = unparsed  # single-line "screen_data = await asyncio.to_thread(...)"
    wrapper_src = "async def _inner():\n    " + stmt_src + "\n    return screen_data\n"
    exec(compile(wrapper_src, "<extracted-screen-universe-assign>", "exec"), ns, ns)

    result = asyncio.run(ns["_inner"]())

    assert result == sentinel
    assert seen["thread"] is not main_thread, (
        "screen_universe mock ran on the MAIN thread -- to_thread wrap is not real"
    )
    assert seen["kwargs"].get("tickers") == ["AAPL", "MSFT"]
    assert seen["kwargs"].get("period") == "6mo"
    assert seen["kwargs"].get("short_interest_threshold") == 0.1


def test_decision_lines_untouched_boundary_markers_present():
    """Boundary sanity: the gate/threshold getattr(settings, ...) calls this
    step must NOT edit are still present verbatim (byte-identical values)."""
    src = _read(AUTONOMOUS_LOOP)
    for marker in [
        'getattr(settings, "short_interest_threshold", 0.10)',
        'getattr(settings, "peer_leadlag_leader_threshold", 10.0)',
        'getattr(settings, "peer_leadlag_laggard_threshold", 2.0)',
        'getattr(settings, "peer_leadlag_min_analyst_filter", 5)',
        'getattr(settings, "peer_leadlag_min_market_cap_usd", 2_000_000_000.0)',
        'getattr(settings, "peer_leadlag_boost", 0.08)',
    ]:
        assert marker in src, f"decision-line marker changed/missing: {marker}"


# ══════════════════════════════════════════════════════════════════
# Criterion 5: add_done_callback error propagation
# ══════════════════════════════════════════════════════════════════

def test_track_task_flips_state_to_error_on_real_exception():
    from backend.utils.asyncio_tasks import track_task

    async def _raiser():
        raise ValueError("boom-crit5-marker")

    async def _driver():
        state = {"status": "running", "error": None}
        tasks: set = set()

        def _on_error(exc):
            state["status"] = "error"
            state["error"] = str(exc)

        task = track_task(asyncio.create_task(_raiser()), tasks, _on_error, "TestSite")
        with suppress(ValueError):
            await task
        # add_done_callback fires via call_soon -- give the loop a couple of
        # ticks so it has definitely run before we inspect state.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return state, tasks

    state, tasks = asyncio.run(_driver())
    assert state["status"] == "error"
    assert "boom-crit5-marker" in state["error"]
    assert len(tasks) == 0, "task not discarded from keep-set after completion"


def test_track_task_does_not_flip_state_on_success_stub_mutation_guard():
    """STUB mutation guard (feedback_mutation_test_guards_and_fixtures): if the
    fixture above is neutered so the task SUCCEEDS instead of raising, THIS
    test proves the assertion actually distinguishes success from failure --
    it must stay green only because on_error is genuinely never invoked."""
    from backend.utils.asyncio_tasks import track_task

    async def _succeeder():
        return "ok"

    async def _driver():
        state = {"status": "running", "error": None}
        tasks: set = set()

        def _on_error(exc):
            state["status"] = "error"
            state["error"] = str(exc)

        task = track_task(asyncio.create_task(_succeeder()), tasks, _on_error, "TestSite")
        result = await task
        await asyncio.sleep(0)
        return state, tasks, result

    state, tasks, result = asyncio.run(_driver())
    assert result == "ok"
    assert state["status"] == "running"  # never flipped -- on_error was not called
    assert len(tasks) == 0


def test_source_asserts_per_site_use_track_task_or_error_callback():
    analysis_src = _read(API_ANALYSIS)
    assert "track_task(" in analysis_src
    assert "AnalysisStatus.FAILED" in analysis_src

    backtest_src = _read(API_BACKTEST)
    assert backtest_src.count("track_task(") >= 2, "expected track_task at both run_backtest and ingestion sites"
    assert "_on_optimizer_task_done" in backtest_src
    assert "_optimizer_task.add_done_callback(_on_optimizer_task_done)" in backtest_src

    paper_trading_src = _read(API_PAPER_TRADING)
    assert "track_task(" in paper_trading_src
    assert "_last_cycle_error" in paper_trading_src


# ══════════════════════════════════════════════════════════════════
# Criterion 6: lifespan shutdown + run_data_ingestion 202 semantics
# ══════════════════════════════════════════════════════════════════

def test_lifespan_shuts_down_both_schedulers_and_cancels_prewarm():
    src = _read(MAIN_PY)
    assert src.count("scheduler.shutdown(wait=False)") == 2, (
        "expected exactly 2 scheduler shutdowns (queue_scheduler + paper scheduler)"
    )
    assert "if 'scheduler' in locals():" in src
    assert "if 'queue_scheduler' in locals():" in src
    assert "prewarm_task.cancel()" in src
    assert "await prewarm_task" in src
    assert "asyncio.CancelledError" in src


def test_run_data_ingestion_returns_202_immediately_with_pollable_status():
    """Mocked ingestion (offline): assert the route returns 'started' + run_id
    immediately (not 'completed' + result synchronously), then let the
    background task finish and assert the progress endpoint reflects it."""
    import backend.api.backtest as bt

    fake_result = {"prices_ingested": 42}

    async def _drive():
        with patch.object(bt, "BigQueryClient", return_value=MagicMock()), \
             patch("backend.tools.screener.screen_universe", return_value=[{"ticker": "AAPL"}]), \
             patch(
                 "backend.backtest.data_ingestion.DataIngestionService.run_full_ingestion",
                 return_value=fake_result,
             ):
            response = await bt.run_data_ingestion(bt.IngestRequest(start_date=None, end_date=None))
            assert response["status"] == "started"
            assert "run_id" in response
            assert response.get("result") is None  # not synchronous "completed"

            # Let the background task run to completion.
            pending = [t for t in bt._background_tasks]
            for t in pending:
                await t
            await asyncio.sleep(0)

            progress = await bt.get_ingestion_progress()
            return progress

    progress = asyncio.run(_drive())
    assert progress["status"] == "completed"
    assert progress["result"] == fake_result


def test_run_data_ingestion_flips_to_error_on_ingestion_failure():
    import backend.api.backtest as bt

    async def _drive():
        with patch.object(bt, "BigQueryClient", return_value=MagicMock()), \
             patch("backend.tools.screener.screen_universe", return_value=[{"ticker": "AAPL"}]), \
             patch(
                 "backend.backtest.data_ingestion.DataIngestionService.run_full_ingestion",
                 side_effect=RuntimeError("ingest-fail-marker"),
             ):
            response = await bt.run_data_ingestion(bt.IngestRequest(start_date=None, end_date=None))
            pending = [t for t in bt._background_tasks]
            for t in pending:
                await t
            await asyncio.sleep(0)
            return await bt.get_ingestion_progress()

    progress = asyncio.run(_drive())
    assert progress["status"] == "error"
    assert "ingest-fail-marker" in (progress["error"] or "")


def test_ingestion_progress_route_distinct_from_row_count_status_route():
    """Guard against overloading the existing GET /ingest/status (row counts)
    with task progress -- they must be two different routes."""
    tree = _parse(API_BACKTEST)
    progress_node = _find_func(tree, "get_ingestion_progress")
    status_node = _find_func(tree, "get_ingestion_status")
    assert progress_node is not None
    assert status_node is not None
    assert progress_node is not status_node


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
