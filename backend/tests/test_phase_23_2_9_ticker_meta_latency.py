"""phase-23.2.9 (P1) verification: ticker-meta latency stays low.

Per researcher (handoff/current/research_brief_phase_23_2_9.md, 6 sources):
  - Live cache-hit latency: 2-3ms steady-state (>30x inside the 100ms SLO).
  - Boot log: "Prewarming ticker-meta cache" appears 54 times in backend.log
    (one per boot since phase-23.1.16 deploy).
  - Endpoint: backend/api/paper_trading.py:1091-1129
  - TTL: backend/services/api_cache.py:134 (86400s)
  - Prewarm hook: backend/main.py:304-335 (fire-and-forget asyncio.create_task)

This test enforces BOTH invariants:
  1. Source-grep: all 3 code anchors present (endpoint route, TTL config, prewarm log).
  2. Live probe: cache-hit latency <100ms (skips when backend offline).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_URL = "http://localhost:8000"
KNOWN_TICKERS = "AAPL,MSFT,GOOGL,NVDA,META,AMZN,TSLA,AMD,AVGO,INTC,JPM,V,UNH,XOM"
LATENCY_BUDGET_MS = 100.0


def _backend_is_up() -> bool:
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/api/health", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def test_phase_23_2_9_ticker_meta_endpoint_route_present_in_source():
    """The /ticker-meta route must be defined in backend/api/paper_trading.py."""
    api_file = REPO_ROOT / "backend" / "api" / "paper_trading.py"
    text = api_file.read_text(encoding="utf-8")
    assert '"/ticker-meta"' in text or "'/ticker-meta'" in text, (
        "ticker-meta route must be defined in backend/api/paper_trading.py"
    )


def test_phase_23_2_9_ticker_meta_cache_ttl_configured():
    """The ticker_meta cache TTL must be configured in api_cache.py.
    Researcher cited: 86400s (1 day)."""
    cache_file = REPO_ROOT / "backend" / "services" / "api_cache.py"
    text = cache_file.read_text(encoding="utf-8")
    assert "paper:ticker_meta" in text, (
        "api_cache.py must define a 'paper:ticker_meta' TTL bucket"
    )


def test_phase_23_2_9_prewarm_hook_present_in_main():
    """The boot-time prewarm hook must be wired in backend/main.py."""
    main_file = REPO_ROOT / "backend" / "main.py"
    text = main_file.read_text(encoding="utf-8")
    assert "Prewarming ticker-meta cache" in text, (
        "backend/main.py must contain the 'Prewarming ticker-meta cache' log emit"
    )


def test_phase_23_2_9_backend_log_has_prewarm_evidence():
    """backend.log must show the prewarm line has fired at least once
    (researcher counted 54 occurrences). Defensive bound: >= 1."""
    backend_log = REPO_ROOT / "backend.log"
    if not backend_log.exists() or backend_log.stat().st_size < 100:
        pytest.skip(f"backend.log not present or too small: {backend_log}")
    text = backend_log.read_text(encoding="utf-8", errors="replace")
    count = text.count("Prewarming ticker-meta cache")
    assert count >= 1, (
        f"backend.log must contain at least 1 'Prewarming ticker-meta cache' line; "
        f"got {count}. If 0, the prewarm hook is silently broken OR log rotated."
    )


@pytest.mark.requires_live
@pytest.mark.skipif(
    os.getenv("PYFINAGENT_LIVE_TESTS") != "1",
    reason="live HTTP probe to the running backend on :8000; flaky during the "
    "18:00Z trading-cycle window when the event loop is busy (the 55.2 F-C "
    "starvation pattern). Asserts live-process state, not code (phase-56.2 "
    "quarantine; set PYFINAGENT_LIVE_TESTS=1 to run)",
)
@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_9_ticker_meta_endpoint_reachable():
    """Live: the /api/paper-trading/ticker-meta endpoint must respond."""
    import urllib.request
    url = f"{BACKEND_URL}/api/paper-trading/ticker-meta?tickers={KNOWN_TICKERS}"
    with urllib.request.urlopen(url, timeout=5) as r:
        assert r.status == 200, f"endpoint must return 200; got {r.status}"
        body = json.loads(r.read())
        # Endpoint returns dict-like keyed by ticker
        assert isinstance(body, (dict, list)), (
            f"ticker-meta response must be dict or list; got {type(body).__name__}"
        )


@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_9_ticker_meta_cache_hit_latency_under_100ms():
    """Live latency probe per masterplan: "time curl ... should be <100ms
    cache-hit". Per researcher: 14 known tickers; prime cache then run
    5 timed calls; assert max < 100ms."""
    import urllib.request
    url = f"{BACKEND_URL}/api/paper-trading/ticker-meta?tickers={KNOWN_TICKERS}"

    # Prime the cache (untimed; absorbs first-call cold-fetch cost)
    with urllib.request.urlopen(url, timeout=10) as r:
        _ = r.read()

    # 5 back-to-back timed cache-hit calls
    samples_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        with urllib.request.urlopen(url, timeout=5) as r:
            _ = r.read()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    max_ms = max(samples_ms)
    assert max_ms < LATENCY_BUDGET_MS, (
        f"ticker-meta cache-hit max latency {max_ms:.1f}ms exceeds "
        f"{LATENCY_BUDGET_MS}ms budget; samples={[f'{s:.1f}' for s in samples_ms]}"
    )
