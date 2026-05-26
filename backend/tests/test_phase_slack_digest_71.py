"""phase-71 cycle (2026-05-26) -- Slack digest regression fixes.

Three independent regressions in `#ford-approvals` Morning + Evening
digests (operator-flagged 2026-05-26 via Slack scrape):

1. Portfolio NAV always shows `$0.00 (+0.0%)` (since 2026-05-12
   phase-25.G commit 55241e3a) -- formatter reads top-level
   `total_pnl` + `total_pnl_pct` but `/api/paper-trading/portfolio`
   returns a nested envelope `{"portfolio": {...}, ...}` and
   `paper_portfolio` BQ row has no `total_pnl` column (only
   `total_nav` + `total_pnl_pct` + `starting_capital`).

2. All "Recent Analyses" show `0.0/10` (since 2026-05-22 first clean
   autonomous cycle landed) -- `autonomous_loop.py:1293` read the
   score under the wrong key `final_score`, orchestrator stores it
   under `final_weighted_score`. The 0 propagated through
   `_persist_analysis` to BQ.

3. "Today's Trades" identical 9 days running -- `get_paper_trades`
   has no date filter; the scheduler URL did not request today-only.

Tests cover the formatter unwrap (fixes 1) + the key-drift fallback
(fix 2) + the `since_iso` filter param (fix 3 helper) without
hitting BigQuery.
"""

from __future__ import annotations

import pytest


# --- Fix 1: formatters envelope unwrap --------------------------------


def test_format_morning_digest_unwraps_portfolio_envelope():
    """Morning digest must read NAV from the nested portfolio envelope."""
    from backend.slack_bot.formatters import format_morning_digest

    envelope = {
        "portfolio": {
            "total_nav": 23800.0,
            "starting_capital": 20000.0,
            "total_pnl_pct": 19.0,
        },
        "positions": [],
        "sector_breakdown": {},
    }
    blocks = format_morning_digest(envelope, [])
    # Find the Portfolio section block.
    portfolio_sections = [
        b for b in blocks
        if b.get("type") == "section"
        and "Portfolio:" in b.get("text", {}).get("text", "")
    ]
    assert portfolio_sections, "expected a Portfolio: section block"
    text = portfolio_sections[0]["text"]["text"]
    # total_pnl = 23800 - 20000 = 3800; total_pnl_pct = 19.0.
    assert "$3,800.00" in text
    assert "19.0%" in text


def test_format_evening_digest_unwraps_portfolio_envelope():
    """Evening digest must read NAV from the nested portfolio envelope."""
    from backend.slack_bot.formatters import format_evening_digest

    envelope = {
        "portfolio": {
            "total_nav": 23800.0,
            "starting_capital": 20000.0,
            "total_pnl_pct": 19.0,
        },
        "positions": [],
        "sector_breakdown": {},
    }
    blocks = format_evening_digest(envelope, [])
    eod_sections = [
        b for b in blocks
        if b.get("type") == "section"
        and "End-of-Day Portfolio:" in b.get("text", {}).get("text", "")
    ]
    assert eod_sections, "expected an End-of-Day Portfolio: section block"
    text = eod_sections[0]["text"]["text"]
    assert "$3,800.00" in text
    assert "19.0%" in text


def test_format_morning_digest_handles_flat_dict_defensively():
    """Defensive: if a caller passes the inner dict directly (forward-compat
    with future refactors), the formatter still works."""
    from backend.slack_bot.formatters import format_morning_digest

    flat = {
        "total_nav": 21000.0,
        "starting_capital": 20000.0,
        "total_pnl_pct": 5.0,
    }
    blocks = format_morning_digest(flat, [])
    portfolio_sections = [
        b for b in blocks
        if b.get("type") == "section"
        and "Portfolio:" in b.get("text", {}).get("text", "")
    ]
    text = portfolio_sections[0]["text"]["text"]
    assert "$1,000.00" in text
    assert "5.0%" in text


def test_format_morning_digest_zero_when_empty_envelope():
    """Missing fields render $0.00 / 0.0% (graceful degradation)."""
    from backend.slack_bot.formatters import format_morning_digest

    blocks = format_morning_digest({"portfolio": {}}, [])
    portfolio_sections = [
        b for b in blocks
        if b.get("type") == "section"
        and "Portfolio:" in b.get("text", {}).get("text", "")
    ]
    text = portfolio_sections[0]["text"]["text"]
    assert "$0.00" in text
    assert "0.0%" in text


# --- Fix 2: autonomous_loop final_score key drift ---------------------


def test_autonomous_loop_reads_final_weighted_score_from_synthesis():
    """Source-grep: `autonomous_loop._run_single_analysis` full-path return
    must read `final_weighted_score` (orchestrator's actual output key).
    Previously read `final_score` which the orchestrator never sets;
    defaulted to 0 and cascaded into `analysis_results.final_score=0`.
    """
    from pathlib import Path

    src = Path("backend/services/autonomous_loop.py").read_text(encoding="utf-8")
    # The defensive fallback chain must be present.
    assert "synthesis.get(" in src
    assert '"final_weighted_score"' in src
    # And the bare `synthesis.get("final_score", 0)` MUST NOT appear without
    # the weighted-score fallback wrapping it (this would be a regression).
    # Allow it ONLY when wrapped as the inner fallback to the weighted key.
    bare_pattern = 'synthesis.get("final_score", 0),'
    weighted_pattern = (
        'synthesis.get(\n                "final_weighted_score", '
        'synthesis.get("final_score", 0)\n            ),'
    )
    bare_idx = src.find(bare_pattern)
    weighted_idx = src.find(weighted_pattern)
    if bare_idx >= 0:
        # The bare hit must be the one nested inside the weighted_pattern.
        assert weighted_idx >= 0, (
            "found bare synthesis.get('final_score', 0) without the "
            "weighted-score fallback wrapping"
        )
        assert weighted_idx <= bare_idx <= weighted_idx + len(weighted_pattern), (
            "bare synthesis.get('final_score', 0) appears OUTSIDE the "
            "weighted-score fallback chain -- regression"
        )


# --- Fix 3: get_paper_trades since_iso filter -------------------------


def test_get_paper_trades_signature_accepts_since_iso():
    """The new `since_iso` parameter must be optional + default None so
    existing callers (no kwarg) preserve original behavior."""
    import inspect
    from backend.db.bigquery_client import BigQueryClient

    sig = inspect.signature(BigQueryClient.get_paper_trades)
    params = sig.parameters
    assert "since_iso" in params, "get_paper_trades must accept since_iso"
    assert params["since_iso"].default is None, (
        "since_iso must default to None so existing callers are unaffected"
    )


def test_get_paper_trades_query_adds_where_when_since_iso_set():
    """When `since_iso` is supplied, the BQ query must include `WHERE
    created_at >= @since`. We don't hit BigQuery here -- we monkey-patch
    the client to inspect the query string."""
    from backend.db.bigquery_client import BigQueryClient

    captured: dict = {}

    class _FakeJob:
        def result(self):
            return []

    class _FakeClient:
        def query(self, query, job_config=None):
            captured["query"] = query
            captured["params"] = (
                [p.name for p in job_config.query_parameters] if job_config else []
            )
            return _FakeJob()

    bq = BigQueryClient.__new__(BigQueryClient)
    bq.client = _FakeClient()
    # _pt_table is needed by the query; patch it minimally.
    bq._pt_table = lambda name: f"project.financial_reports.{name}"

    # Call WITHOUT since_iso -- query must NOT contain `WHERE created_at`.
    bq.get_paper_trades(limit=5)
    assert "WHERE created_at" not in captured["query"]
    assert "since" not in captured["params"]

    # Call WITH since_iso -- query MUST contain `WHERE created_at >= @since`.
    bq.get_paper_trades(limit=5, since_iso="2026-05-26T00:00:00+00:00")
    assert "WHERE created_at >= @since" in captured["query"]
    assert "since" in captured["params"]


# --- Fix 3 wire-up: scheduler URL has since_today=true ---------------


def test_evening_digest_scheduler_passes_since_today():
    """The evening-digest scheduler must request today-only trades so the
    'Today's Trades' section isn't replaying stale rows."""
    from pathlib import Path

    src = Path("backend/slack_bot/scheduler.py").read_text(encoding="utf-8")
    # The trades URL in the evening-digest path must include since_today=true.
    assert "/api/paper-trading/trades?limit=10&since_today=true" in src


# --- Fix 3 wire-up: /api/paper-trading/trades exposes since_today ---


def test_paper_trading_trades_endpoint_accepts_since_today():
    """The /trades endpoint must accept `since_today: bool = False` so the
    scheduler's `?since_today=true` query param round-trips."""
    import inspect
    from backend.api import paper_trading as pt

    sig = inspect.signature(pt.get_trades)
    assert "since_today" in sig.parameters, (
        "/api/paper-trading/trades must expose since_today"
    )
