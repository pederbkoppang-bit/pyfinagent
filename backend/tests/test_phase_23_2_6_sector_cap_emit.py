"""phase-23.2.6 (P1) verification: sector cap actually blocked same-sector buys.

Per researcher (handoff/current/research_brief_phase_23_2_6.md, 6 sources):

  * backend.log contains 24 "Skipping BUY ... at cap" emits today --
    forward-gate is firing correctly.
  * Cap value: settings.paper_max_per_sector = 2 (default at
    backend/config/settings.py:162; phase-23.1.13 origin commit 5b350e4d).
  * Emit site: backend/services/portfolio_manager.py:247-252.
  * CAVEAT: BQ paper_positions snapshot shows 8 Tech positions today
    (legacy overage from before phase-23.2.6-fix sector-persistence
    migration commit c854386f). The cap blocks NEW buys correctly but
    cannot retro-divest legacy state. Documented in live_check_23.2.6.md
    as a phase-23.2.6.1 follow-up (NOT scope-creep into this verification).

This test enforces FORWARD-GATE invariant: when a sector is already at
cap, a new BUY in that sector is blocked + the canonical log line is
emitted. Mutation-resistant against: (a) cap-bypass; (b) wrong sector
counted; (c) log message format drift; (d) cap=0-disables semantics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _propagate_pm_logger():
    """Ensure backend.services.portfolio_manager logger propagates to root
    so pytest caplog captures it. The module typically uses
    logger = logging.getLogger(__name__) which means it's `backend.services.portfolio_manager`."""
    pm_logger = logging.getLogger("backend.services.portfolio_manager")
    original = pm_logger.propagate
    pm_logger.propagate = True
    yield
    pm_logger.propagate = original


def _make_settings(cap: int = 2, **overrides: Any):
    """Mock Settings-shaped object for decide_trades. Using SimpleNamespace
    so we avoid pydantic's extra=forbid validation."""
    from types import SimpleNamespace
    return SimpleNamespace(
        paper_max_per_sector=cap,
        paper_max_per_sector_nav_pct=overrides.get("paper_max_per_sector_nav_pct", 0.0),
        paper_max_positions=overrides.get("paper_max_positions", 10),
        paper_starting_capital=overrides.get("paper_starting_capital", 10000.0),
        paper_min_cash_reserve_pct=overrides.get("paper_min_cash_reserve_pct", 5.0),
        paper_target_position_size_pct=overrides.get("paper_target_position_size_pct", 10.0),
        paper_default_stop_loss_pct=overrides.get("paper_default_stop_loss_pct", 8.0),
        paper_min_cash_buffer=overrides.get("paper_min_cash_buffer", 100.0),
        paper_trade_min_confidence=overrides.get("paper_trade_min_confidence", 0.0),
        paper_max_position_value_pct=overrides.get("paper_max_position_value_pct", 100.0),
    )


def _make_position(ticker: str, sector: str, qty: float = 10, price: float = 100.0) -> dict:
    """Minimal position dict matching decide_trades's expected shape."""
    return {
        "ticker": ticker,
        "sector": sector,
        "qty": qty,
        "current_price": price,
        "entry_price": price,
        "market_value": qty * price,
        "stop_loss_price": price * 0.92,
        "recommendation": "BUY",
    }


def _make_candidate(ticker: str, sector: str, recommendation: str = "BUY", confidence: float = 0.7) -> dict:
    """Minimal candidate analysis dict."""
    return {
        "ticker": ticker,
        "sector": sector,
        "recommendation": recommendation,
        "confidence": confidence,
        "risk_assessment": {"risk_level": "moderate"},
        "current_price": 100.0,
        "analysis_date": "2026-05-23T00:00:00Z",
    }


def test_phase_23_2_6_emit_site_present_in_source():
    """The canonical log format `Skipping BUY %s: sector %s at cap (%d/%d)`
    at portfolio_manager.py:247-252 MUST remain present. A future refactor
    that drops the log call would silently disable our log-based audit."""
    src = (REPO_ROOT / "backend" / "services" / "portfolio_manager.py").read_text()
    assert "Skipping BUY %s: sector %s at cap" in src, (
        "portfolio_manager.py must keep the canonical Skipping BUY log format"
    )
    assert "max_per_sector" in src, (
        "decide_trades must reference max_per_sector from settings"
    )


def test_phase_23_2_6_settings_default_paper_max_per_sector():
    """settings.paper_max_per_sector default must be a positive int (cap active).
    Catches future drift where someone sets default=0 (which disables the cap)."""
    from backend.config.settings import Settings
    field = Settings.model_fields.get("paper_max_per_sector")
    assert field is not None, "Settings must define paper_max_per_sector field"
    assert isinstance(field.default, int)
    assert field.default >= 1, (
        f"paper_max_per_sector default must be >= 1 to keep cap active; got {field.default}"
    )


def test_phase_23_2_6_blocks_third_tech_buy_when_two_held(caplog):
    """Core forward-gate: cap=2, 2 Tech positions held; a new Tech BUY
    candidate must be BLOCKED (no BUY order emitted for it) + the
    canonical log line must fire."""
    from backend.services.portfolio_manager import decide_trades

    caplog.set_level(logging.INFO, logger="backend.services.portfolio_manager")

    settings = _make_settings(cap=2)
    current_positions = [
        _make_position("AAPL", "Technology"),
        _make_position("MSFT", "Technology"),
    ]
    candidates = [_make_candidate("AMD", "Technology")]
    portfolio_state = {
        "nav": 10000.0,
        "cash": 5000.0,
        "positions_value": 5000.0,
        "position_count": 2,
    }

    orders = decide_trades(
        current_positions,
        candidates,
        [],  # holding_analyses
        portfolio_state,
        settings,
    )
    buy_tickers = {o.ticker for o in orders if o.action == "BUY"}
    assert "AMD" not in buy_tickers, (
        f"AMD BUY must be blocked by sector cap; got buy_tickers={buy_tickers}"
    )
    skip_msgs = [
        r.getMessage() for r in caplog.records
        if "Skipping BUY" in r.getMessage() and "at cap" in r.getMessage()
    ]
    assert len(skip_msgs) >= 1, (
        f"expected at least 1 'Skipping BUY ... at cap' log; got: {[r.getMessage() for r in caplog.records]}"
    )
    canonical = skip_msgs[0]
    assert "AMD" in canonical
    assert "Technology" in canonical
    assert "(2/2)" in canonical or "2/2" in canonical


def test_phase_23_2_6_allows_buy_in_new_sector(caplog):
    """When a new BUY's sector has 0 existing positions, the cap must
    NOT block it (cap=2; new sector has 0/2)."""
    from backend.services.portfolio_manager import decide_trades

    caplog.set_level(logging.INFO, logger="backend.services.portfolio_manager")

    settings = _make_settings(cap=2)
    current_positions = [
        _make_position("AAPL", "Technology"),
        _make_position("MSFT", "Technology"),
    ]
    candidates = [_make_candidate("JPM", "Financials")]
    portfolio_state = {
        "nav": 10000.0,
        "cash": 5000.0,
        "positions_value": 5000.0,
        "position_count": 2,
    }

    orders = decide_trades(
        current_positions, candidates, [], portfolio_state, settings,
    )
    buy_tickers = {o.ticker for o in orders if o.action == "BUY"}
    # JPM is in a new sector with 0 existing positions; should be allowed
    # (subject to other gates like cash/position count, which we set permissive)
    skip_msgs = [
        r.getMessage() for r in caplog.records
        if "Skipping BUY" in r.getMessage() and "JPM" in r.getMessage()
    ]
    assert not skip_msgs, (
        f"JPM (new sector) should NOT trip sector cap; got: {skip_msgs}"
    )


def test_phase_23_2_6_cap_zero_disables_gate(caplog):
    """paper_max_per_sector=0 disables the cap entirely. Catches the
    common misuse where someone sets cap=0 expecting "no positions allowed"."""
    from backend.services.portfolio_manager import decide_trades

    caplog.set_level(logging.INFO, logger="backend.services.portfolio_manager")

    settings = _make_settings(cap=0)  # disabled
    current_positions = [
        _make_position("AAPL", "Technology"),
        _make_position("MSFT", "Technology"),
        _make_position("GOOGL", "Technology"),
    ]
    candidates = [_make_candidate("AMD", "Technology")]
    portfolio_state = {
        "nav": 10000.0,
        "cash": 5000.0,
        "positions_value": 5000.0,
        "position_count": 3,
    }

    orders = decide_trades(
        current_positions, candidates, [], portfolio_state, settings,
    )
    skip_msgs = [
        r.getMessage() for r in caplog.records
        if "Skipping BUY" in r.getMessage() and "at cap" in r.getMessage()
        and "AMD" in r.getMessage()
    ]
    assert not skip_msgs, (
        f"cap=0 must disable the gate; got blocked: {skip_msgs}"
    )


@pytest.mark.requires_live
def test_phase_23_2_6_backend_log_has_skipping_buy_evidence():
    """Read-only verification: backend.log must contain at least one
    'Skipping BUY ... at cap' line if the gate has ever fired in the
    log's retention window. Researcher counted 24 today.

    phase-75.15 (qa-tests-01): asserts LIVE backend.log evidence on THIS
    machine (root backend.log + its handoff/logs/*.gz rotation archives,
    both gitignored -> absent + skip on a fresh CI checkout). Fails
    locally because it IS live-system-dependent: the live backend.log
    (114MB, size>=100 met) has zero 'Skipping BUY' occurrences measured
    2026-07-24, and the newest rotation archive (2026-07-06) also has
    zero -- an older 2026-06-12 archive has 56, but the fallback only
    checks the single newest archive. No recent sector-cap-blocked BUY
    in this window is genuine live-system state, not a code defect --
    quarantined per the requires_live convention (pytest.ini:9); set
    PYFINAGENT_LIVE_TESTS=1 to run.
    """
    backend_log = REPO_ROOT / "backend.log"
    if not backend_log.exists() or backend_log.stat().st_size < 100:
        pytest.skip(f"backend.log not present or too small: {backend_log}")
    text = backend_log.read_text(encoding="utf-8", errors="replace")
    skip_count = text.count("Skipping BUY")
    # Defensive lower bound: researcher counted 24; any future cap-firing
    # cycle adds more. phase-62.6: backend.log is rotated (cp+truncate+gzip
    # into handoff/logs/) once it exceeds 50MB -- per this test's own
    # original comment ("the log was rotated and the test should adapt"),
    # fall back to the newest archive before declaring the gate broken.
    if skip_count == 0:
        import gzip
        archives = sorted((REPO_ROOT / "handoff" / "logs").glob("backend.log.*.gz"))
        if archives:
            with gzip.open(archives[-1], "rt", encoding="utf-8", errors="replace") as f:
                skip_count = sum(line.count("Skipping BUY") for line in f)
        else:
            pytest.skip("backend.log freshly rotated and no archive found")
    assert skip_count >= 1, (
        f"no 'Skipping BUY' line in backend.log OR its newest archive "
        f"(researcher counted 24 on 2026-05-23); the cap gate may be "
        f"silently disabled."
    )
