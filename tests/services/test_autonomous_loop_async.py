"""phase-23.1.23: regression guard against blocking trader.* calls in async loop.

The bug: autonomous_loop.run_daily_cycle is `async def` but called
trader.mark_to_market(), trader.save_daily_snapshot(),
trader.get_positions(), trader.execute_sell/buy(), and
trader.check_and_enforce_kill_switch() SYNCHRONOUSLY. Each of these does
~14 positions x (yfinance HTTP + BQ DML insert + BQ DML delete) = 42 blocking
network ops. While running, the asyncio event loop is fully blocked, so
/api/health doesn't respond and the watchdog kicks the backend (3 fails
@ 60s probe = 180s threshold) -- the daily 20:00 CEST cycle never
completes and snapshots haven't updated for 5 days.

The fix: every trader.* call inside run_daily_cycle is now wrapped in
asyncio.to_thread(...).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def _load_run_daily_cycle() -> str:
    src = Path(__file__).resolve().parents[2] / "backend/services/autonomous_loop.py"
    text = src.read_text(encoding="utf-8")
    # Extract the run_daily_cycle body — it's an async def that runs until
    # the next top-level def/async def.
    m = re.search(
        r"async def run_daily_cycle.*?(?=\nasync def |\ndef [a-z])",
        text, re.DOTALL,
    )
    assert m, "could not extract run_daily_cycle body"
    return m.group(0)


def test_no_bare_trader_calls_in_run_daily_cycle():
    """Every `trader.<method>(...)` call inside run_daily_cycle must be
    wrapped with `asyncio.to_thread(...)`. Bare calls block the event loop."""
    body = _load_run_daily_cycle()
    # Find every line that calls trader.<something>( but NOT prefixed by
    # `asyncio.to_thread(`.
    bare_pattern = re.compile(r"^\s*(?!.*to_thread)(?!.*#).*\btrader\.\w+\(", re.MULTILINE)
    bare_hits = [
        m.group(0).strip()
        for m in bare_pattern.finditer(body)
        if "to_thread" not in m.group(0)
    ]
    # Allow comment lines that mention trader.* (just docstrings).
    bare_hits = [h for h in bare_hits if not h.lstrip().startswith("#")]
    assert not bare_hits, \
        f"Bare trader.* calls in run_daily_cycle (must wrap in asyncio.to_thread):\n  " + \
        "\n  ".join(bare_hits)


def test_mark_to_market_is_wrapped():
    """The Step 5 + Step 8 mark_to_market calls must be asyncio.to_thread."""
    body = _load_run_daily_cycle()
    # Find every mark_to_market reference.
    mtm_count = body.count("trader.mark_to_market")
    assert mtm_count >= 3, f"expected >=3 trader.mark_to_market calls, got {mtm_count}"
    # All must be inside asyncio.to_thread(...)
    wrapped_count = len(re.findall(r"asyncio\.to_thread\(\s*trader\.mark_to_market", body))
    assert wrapped_count == mtm_count, \
        f"{mtm_count - wrapped_count} mark_to_market call(s) not wrapped in asyncio.to_thread"


def test_save_daily_snapshot_is_wrapped():
    body = _load_run_daily_cycle()
    snap_count = body.count("trader.save_daily_snapshot")
    assert snap_count >= 2
    # save_daily_snapshot may take kwargs across multiple lines; match the
    # function reference pattern.
    wrapped = len(re.findall(
        r"asyncio\.to_thread\(\s*\n?\s*trader\.save_daily_snapshot",
        body, re.DOTALL,
    ))
    assert wrapped == snap_count, \
        f"{snap_count - wrapped} save_daily_snapshot call(s) not wrapped"


def test_phase_marker_present():
    """phase-23.1.23 marker comment must be present so future grepers find
    the rationale."""
    src = Path(__file__).resolve().parents[2] / "backend/services/autonomous_loop.py"
    text = src.read_text(encoding="utf-8")
    assert "phase-23.1.23" in text, "autonomous_loop.py must carry phase-23.1.23 marker"
