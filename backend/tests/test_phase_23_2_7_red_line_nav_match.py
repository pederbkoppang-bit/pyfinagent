"""phase-23.2.7 (P1) verification: Red Line Monitor terminal NAV matches live.

Per masterplan: "curl /api/sovereign/red-line?window=7d; assert last point's
nav equals current paper_portfolio.total_nav within fee tolerance".

Live probe today (2026-05-23):
  - /api/sovereign/red-line?window=7d -> last point nav = 23184.7
  - /api/paper-trading/portfolio -> portfolio.total_nav = 23184.7
  - /api/paper-trading/kill-switch -> current_nav = 23184.7
  All three match exactly. The Red Line Monitor is in sync.

This test covers:
  1. Structural invariant: the 3 NAV sources match within fee tolerance
     (live test; skips if backend offline).
  2. Red-line response has the expected shape (series + last-point fields).
  3. Portfolio response has total_nav field present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_URL = "http://localhost:8000"

# Per masterplan: "within fee tolerance".
# Researcher recommendation (handoff/current/research_brief_phase_23_2_7.md
# Section C #1): for SAME-SOURCE comparisons (e.g. kill-switch vs portfolio
# which both read from the same BQ row), use 1 basis point (0.01%) to
# catch real drift bugs that 1% would silently mask. For CROSS-SOURCE
# (e.g. red-line snapshot vs live portfolio), 1% remains correct because
# legitimate mid-mark-to-market timing drift can be larger than bp-level
# precision (the "two-clock problem" per Fidelity ETF NAV docs).
NAV_MATCH_TOLERANCE_PCT_CROSS_SOURCE = 1.0
NAV_MATCH_TOLERANCE_PCT_SAME_SOURCE = 0.01  # 1 bp


def _backend_is_up() -> bool:
    """Probe /api/health; True if 200 OK."""
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/api/health", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _fetch_json(path: str) -> dict:
    import urllib.request
    with urllib.request.urlopen(f"{BACKEND_URL}{path}", timeout=5) as r:
        return json.loads(r.read())


@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_7_red_line_last_point_matches_portfolio_total_nav():
    """Core invariant: red-line last point nav matches portfolio.total_nav
    within fee tolerance. Catches the failure mode where snapshotting drifts
    away from live state."""
    rl = _fetch_json("/api/sovereign/red-line?window=7d")
    pf = _fetch_json("/api/paper-trading/portfolio")

    series = rl.get("series") or rl.get("points") or rl.get("data")
    assert isinstance(series, list) and series, (
        f"red-line response must have non-empty series; got {rl}"
    )
    last_nav = series[-1].get("nav")
    assert last_nav is not None, (
        f"red-line last point must have 'nav' field; got {series[-1]}"
    )

    portfolio = pf.get("portfolio") or pf
    portfolio_nav = portfolio.get("total_nav")
    assert portfolio_nav is not None, (
        f"portfolio response must have 'total_nav' field; got {pf}"
    )

    delta_pct = abs(last_nav - portfolio_nav) / portfolio_nav * 100.0
    assert delta_pct <= NAV_MATCH_TOLERANCE_PCT_CROSS_SOURCE, (
        f"red-line last nav={last_nav} drifts {delta_pct:.4f}% from "
        f"portfolio.total_nav={portfolio_nav} (cross-source tolerance "
        f"{NAV_MATCH_TOLERANCE_PCT_CROSS_SOURCE}%)"
    )


@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_7_red_line_response_shape():
    """The red-line response must contain window + series + events fields."""
    rl = _fetch_json("/api/sovereign/red-line?window=7d")
    assert "window" in rl, f"red-line response must have 'window' field; got {list(rl.keys())}"
    series = rl.get("series") or rl.get("points") or rl.get("data")
    assert isinstance(series, list), f"series must be a list; got {type(series).__name__}"
    if series:
        last = series[-1]
        assert "nav" in last, f"each series point must have 'nav' field; got {last}"
        assert "date" in last, f"each series point must have 'date' field; got {last}"


@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_7_portfolio_total_nav_field_present():
    """Portfolio endpoint must expose total_nav field (the cross-reference
    target for the red-line invariant)."""
    pf = _fetch_json("/api/paper-trading/portfolio")
    portfolio = pf.get("portfolio") or pf
    assert "total_nav" in portfolio, (
        f"portfolio response must have 'total_nav' field; got keys: {list(portfolio.keys())}"
    )
    assert isinstance(portfolio["total_nav"], (int, float)), (
        f"total_nav must be numeric; got {type(portfolio['total_nav']).__name__}"
    )


@pytest.mark.skipif(not _backend_is_up(), reason="backend not listening on :8000")
def test_phase_23_2_7_kill_switch_current_nav_matches_portfolio_total_nav():
    """Cross-check: a third endpoint (kill-switch) exposes current_nav.
    All three NAV sources must agree within fee tolerance. This catches the
    failure mode where one endpoint reads from a stale cache while another
    reads fresh."""
    ks = _fetch_json("/api/paper-trading/kill-switch")
    pf = _fetch_json("/api/paper-trading/portfolio")

    portfolio = pf.get("portfolio") or pf
    pf_nav = portfolio.get("total_nav")
    ks_nav = ks.get("current_nav")

    if ks_nav is None:
        pytest.skip("kill-switch response has no current_nav field (not exposed)")

    delta_pct = abs(ks_nav - pf_nav) / pf_nav * 100.0
    # Same-source: both endpoints read the same BQ paper_portfolio row.
    # Use 1bp tolerance (researcher recommendation; catches the $230 drift
    # bug that a 1% tolerance would silently mask).
    assert delta_pct <= NAV_MATCH_TOLERANCE_PCT_SAME_SOURCE, (
        f"kill-switch current_nav={ks_nav} drifts {delta_pct:.4f}% from "
        f"portfolio.total_nav={pf_nav} (same-source tolerance "
        f"{NAV_MATCH_TOLERANCE_PCT_SAME_SOURCE}% = 1bp). Both endpoints "
        f"read the same BQ row; any drift > 1bp indicates a real bug "
        f"(stale cache, race condition, or accidental copy of the wrong field)."
    )


def test_phase_23_2_7_red_line_endpoint_exists_in_source():
    """The /api/sovereign/red-line endpoint must remain wired."""
    sovereign_api = REPO_ROOT / "backend" / "api" / "sovereign.py"
    if not sovereign_api.exists():
        pytest.skip(f"backend/api/sovereign.py not present: {sovereign_api}")
    text = sovereign_api.read_text(encoding="utf-8")
    assert "red-line" in text or "red_line" in text, (
        "backend/api/sovereign.py must define a red-line route"
    )
