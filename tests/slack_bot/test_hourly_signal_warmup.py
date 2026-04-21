"""phase-9.5 tests."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.slack_bot.jobs.hourly_signal_warmup import run
from backend.slack_bot.job_runtime import IdempotencyStore


def test_warmup_populates_injected_cache():
    cache = {}
    store = IdempotencyStore()
    out = run(
        watchlist=["AAPL", "MSFT"],
        compute_signal_fn=lambda t: {"score": 1.0, "ticker": t},
        cache_backend=cache,
        store=store,
        iso_hour="2026-04-20T02",
    )
    assert out["warmed"] == 2
    assert set(cache.keys()) == {"AAPL", "MSFT"}


def test_watchlist_from_settings_when_not_injected():
    """_load_watchlist falls back to settings; never raises."""
    cache = {}
    store = IdempotencyStore()
    out = run(
        watchlist=None,  # triggers _load_watchlist
        compute_signal_fn=lambda t: 1,
        cache_backend=cache,
        store=store,
        iso_hour="2026-04-20T03",
    )
    assert "warmed" in out


def test_cache_backend_is_injectable():
    """Custom cache is honored (not the default in-memory)."""
    my_cache = {"PRE_EXISTING": "value"}
    store = IdempotencyStore()
    run(
        watchlist=["X"],
        compute_signal_fn=lambda t: "sig_" + t,
        cache_backend=my_cache,
        store=store,
        iso_hour="2026-04-20T04",
    )
    assert my_cache["PRE_EXISTING"] == "value"
    assert my_cache["X"] == "sig_X"
