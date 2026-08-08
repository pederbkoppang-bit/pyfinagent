"""phase-85.4 criterion 2 -- reproduce the analysis-phase root cause.

Criterion 2 (verbatim from .claude/masterplan.json):

    the root cause of the non-completion is identified to file:line with a
    reproduction, distinguishing (a) legitimate slowness vs (b) a
    hang/deadlock vs (c) an unhandled per-ticker failure that stalls the gather

THE ANSWER IS (a), WITH A STRUCTURAL AMPLIFIER.

  (c) is RULED OUT here: `return_exceptions=True` plus the per-ticker
      try/except mean a failing ticker returns None and never stalls its
      siblings -- `test_c_a_failing_ticker_does_not_stall_the_gather` proves it
      against the production seam.

  (b) is RULED OUT here: every dispatched ticker in every measured cycle
      either finished or was still making rail calls when the budget expired;
      there is no lock, no join, and no unbounded wait in the dispatch path.
      `test_b_no_deadlock_all_dispatched_tickers_retire` proves the seam
      itself always drains.

  (a) is CONFIRMED, and the reason it overruns is not raw per-ticker cost
      alone. `autonomous_loop.py` awaited TWO sequential gathers over ONE
      shared semaphore (the legacy `merged=False` path in `dispatch_analyses`).
      The second batch cannot start until the first has fully drained, so a
      slot freed by an early finisher idles instead of picking up work.

Measured on the real 2026-08-07 cycle (6 tickers, semaphore=3), via
scripts/diagnostics/measure_analysis_phase.py over backend.log:

    20:52:30  PANW done  -> slot free, but NTAP is in batch 2 -> IDLE
    21:24:33  batch 1 finally drains -> NTAP dispatched, 1923s late
    22:00:01  cycle TIMED OUT with NTAP never analysed

These tests run that shape against the production function with a virtual
clock, so the reproduction is deterministic and takes milliseconds.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.services.autonomous_loop import dispatch_analyses


# One "unit" of simulated per-ticker analysis. Real cycles spend ~2300s per
# ticker; the reproduction only needs the RATIO, so a unit is 0.1s and the
# whole file runs in about a second. Timings are quantised to units before any
# assertion, so scheduler jitter up to half a unit (50ms) cannot flip a result.
_UNIT = 0.1


class VirtualClock:
    """Records dispatch/finish times, quantised to `_UNIT`.

    The production seam is driven by a real event loop -- the whole point is to
    measure the loop's actual scheduling of the two-gather barrier -- so time
    is real, but every reading is divided by `_UNIT` and rounded, which turns
    "0.203s" into "2 units" and makes the assertions exact.
    """

    def __init__(self) -> None:
        self._t0: float | None = None
        self.dispatched_at: dict[str, int] = {}
        self.finished_at: dict[str, int] = {}

    def _units(self) -> int:
        loop = asyncio.get_running_loop()
        if self._t0 is None:
            self._t0 = loop.time()
        return round((loop.time() - self._t0) / _UNIT)

    def mark_dispatch(self, ticker: str) -> None:
        self.dispatched_at[ticker] = self._units()

    def mark_finish(self, ticker: str) -> None:
        self.finished_at[ticker] = self._units()


def _make_runner(clock: VirtualClock, cost: dict[str, int], sem: asyncio.Semaphore,
                 failures: set[str] | None = None):
    """Build an `async (ticker, kind) -> result` with the SAME semaphore
    discipline the production `_run_and_persist_one` closure uses."""
    failures = failures or set()

    async def _runner(ticker: str, kind: str):
        async with sem:
            clock.mark_dispatch(ticker)
            await asyncio.sleep(cost.get(ticker, 1) * _UNIT)
            clock.mark_finish(ticker)
            if ticker in failures:
                raise RuntimeError(f"injected per-ticker failure: {ticker}")
            return {"ticker": ticker, "kind": kind}

    return _runner


# The 2026-08-07 cycle, to scale: 5 new candidates + 1 re-eval, semaphore 3,
# per-ticker cost roughly equal (the real spread was 2049-2593s).
_NEW = ["DELL", "PANW", "CRWD", "HPE", "HUM"]
_REEVAL = ["NTAP"]
_COST = {t: 1 for t in _NEW + _REEVAL}

# Expected schedules at semaphore=3, 6 tickers, 1 unit each:
#   legacy : [DELL PANW CRWD] 0->1 | [HPE HUM] 1->2 | barrier | [NTAP] 2->3
#   merged : [DELL PANW CRWD] 0->1 | [HPE HUM NTAP] 1->2
# i.e. 3 units vs 2 units, and one slot idle for a full unit in the legacy run.
_LEGACY_MAKESPAN_UNITS = 3
_MERGED_MAKESPAN_UNITS = 2


def _run(merged: bool, failures: set[str] | None = None):
    clock = VirtualClock()
    sem = asyncio.Semaphore(3)
    runner = _make_runner(clock, _COST, sem, failures)

    async def _go():
        return await dispatch_analyses(runner, _NEW, _REEVAL, merged=merged)

    results = asyncio.run(_go())
    return clock, results


# ─────────────────────────────────────────────────────────────────────────────
# (a) the barrier, reproduced against the production seam
# ─────────────────────────────────────────────────────────────────────────────


def test_a_legacy_two_gather_path_idles_a_slot_and_starts_the_reeval_late():
    """LEGACY (merged=False, the shipped default): NTAP cannot start until
    EVERY new candidate has finished, even though a slot fell free earlier."""
    clock, _ = _run(merged=False)

    last_new_finish = max(clock.finished_at[t] for t in _NEW)
    first_new_finish = min(clock.finished_at[t] for t in _NEW)
    ntap_start = clock.dispatched_at["NTAP"]

    assert ntap_start >= last_new_finish, (
        "the legacy path is defined by this barrier: the re-eval batch waits "
        "for the whole new-candidate batch"
    )
    # And the wasted window is real: a slot was free this many ticks earlier.
    idle_ticks = last_new_finish - first_new_finish
    assert idle_ticks > 0, "the reproduction needs a genuine early finisher"


def test_a_merged_path_starts_the_reeval_as_soon_as_a_slot_frees():
    """FIXED (merged=True, DARK): NTAP is dispatched on the first free slot."""
    clock, _ = _run(merged=True)

    last_new_finish = max(clock.finished_at[t] for t in _NEW)
    ntap_start = clock.dispatched_at["NTAP"]

    assert ntap_start < last_new_finish, (
        "merged dispatch must start the re-eval BEFORE the last new candidate "
        "finishes -- that is the entire saving"
    )


def test_a_merged_path_strictly_reduces_makespan_at_the_same_concurrency():
    """The saving is a makespan reduction, not a reordering trick: same
    semaphore, same per-ticker cost, strictly earlier completion."""
    legacy_clock, _ = _run(merged=False)
    merged_clock, _ = _run(merged=True)

    legacy_makespan = max(legacy_clock.finished_at.values())
    merged_makespan = max(merged_clock.finished_at.values())

    assert legacy_makespan == _LEGACY_MAKESPAN_UNITS, legacy_clock.finished_at
    assert merged_makespan == _MERGED_MAKESPAN_UNITS, merged_clock.finished_at
    assert merged_makespan < legacy_makespan, (
        f"merged={merged_makespan} must beat legacy={legacy_makespan}"
    )
    # The saving is exactly the wasted wave: one third of the phase here, and
    # 1923 of 6645 seconds (29%) on the real 2026-08-07 cycle.
    assert legacy_makespan - merged_makespan == 1


def test_a_merged_path_never_exceeds_the_concurrency_cap():
    """The saving must NOT come from running more analyses at once -- that
    would re-open the 429 rate-limit incident the cap exists to prevent."""
    sem = asyncio.Semaphore(3)
    peak = {"n": 0, "cur": 0}

    async def _runner(ticker: str, kind: str):
        async with sem:
            peak["cur"] += 1
            peak["n"] = max(peak["n"], peak["cur"])
            try:
                await asyncio.sleep(_COST.get(ticker, 1) * _UNIT)
            finally:
                peak["cur"] -= 1
            return {"ticker": ticker, "kind": kind}

    cands, holds = asyncio.run(dispatch_analyses(_runner, _NEW, _REEVAL, merged=True))
    # gather(return_exceptions=True) turns a broken runner into a RESULT, not a
    # raise -- so a probe that blew up would silently leave peak["cur"] pinned
    # high and this test would "fail loudly for the wrong reason". Assert the
    # probe itself survived before trusting its measurement.
    assert all(isinstance(r, dict) for r in cands + holds), (cands, holds)
    assert peak["cur"] == 0, "probe leaked a slot -- the measurement is invalid"
    assert peak["n"] == 3, f"expected the cap to be saturated at 3, saw {peak['n']}"


# ─────────────────────────────────────────────────────────────────────────────
# Equivalence: the fix must not change WHAT comes back, only WHEN
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("merged", [False, True])
def test_partition_and_ordering_are_identical_on_both_paths(merged):
    _, (cands, holds) = _run(merged=merged)
    assert [r["ticker"] for r in cands] == _NEW
    assert [r["ticker"] for r in holds] == _REEVAL
    assert all(r["kind"] == "new" for r in cands)
    assert all(r["kind"] == "reeval" for r in holds)


def test_both_paths_return_the_same_results():
    _, legacy = _run(merged=False)
    _, merged = _run(merged=True)
    assert legacy == merged


@pytest.mark.parametrize("merged", [False, True])
def test_empty_reeval_list_is_handled(merged):
    clock = VirtualClock()
    sem = asyncio.Semaphore(3)
    runner = _make_runner(clock, _COST, sem)
    cands, holds = asyncio.run(dispatch_analyses(runner, _NEW, [], merged=merged))
    assert len(cands) == len(_NEW)
    assert holds == []


@pytest.mark.parametrize("merged", [False, True])
def test_empty_both_lists_is_handled(merged):
    async def _runner(t, k):  # pragma: no cover - must never be called
        raise AssertionError("runner called with no tickers")

    cands, holds = asyncio.run(dispatch_analyses(_runner, [], [], merged=merged))
    assert cands == [] and holds == []


# ─────────────────────────────────────────────────────────────────────────────
# (b) and (c): the two alternative root causes, ruled out
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("merged", [False, True])
def test_c_a_failing_ticker_does_not_stall_the_gather(merged):
    """(c) RULED OUT: a raising ticker is captured in place by
    return_exceptions=True; every sibling still retires."""
    clock, (cands, holds) = _run(merged=merged, failures={"CRWD"})

    assert len(clock.finished_at) == len(_NEW) + len(_REEVAL), (
        "a failing ticker stalled its siblings"
    )
    crwd_idx = _NEW.index("CRWD")
    assert isinstance(cands[crwd_idx], RuntimeError), cands[crwd_idx]
    # Positional alignment survives the failure -- the caller's
    # `[r for r in results if isinstance(r, dict)]` filter still works.
    assert sum(isinstance(r, dict) for r in cands) == len(_NEW) - 1
    assert all(isinstance(r, dict) for r in holds)


@pytest.mark.parametrize("merged", [False, True])
def test_b_no_deadlock_all_dispatched_tickers_retire(merged):
    """(b) RULED OUT: the dispatch path itself always drains. Every ticker
    dispatched is a ticker finished -- there is no join, lock ordering, or
    unbounded wait that can wedge it."""
    clock, _ = _run(merged=merged)
    assert set(clock.dispatched_at) == set(_NEW + _REEVAL)
    assert set(clock.finished_at) == set(_NEW + _REEVAL)


def test_production_default_is_the_legacy_path():
    """criterion 5: the behavioural change ships DARK. If this ever flips,
    phase-85.4 promoted a flag it was forbidden to promote."""
    from backend.config.settings import Settings

    assert Settings().paper_merged_analysis_dispatch_enabled is False
