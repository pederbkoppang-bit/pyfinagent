"""phase-38.7 SPY benchmark anchor at first-funded snapshot tests.

Closes closure_roadmap.md section 3 OPEN-9: paper_trader._get_benchmark_return
was anchored to portfolio.inception_date (set at row creation, before any
capital injection -- the "Initialization Date" anti-pattern per
PerformanceMeasurementSolutions industry taxonomy + GIPS). Correct anchor:
first-funded snapshot where positions_value > 0 (the "Initial Trading
Date" / "Funding Date").

Tests cover:
  1. _get_benchmark_return signature accepts first_funded_date kwarg.
  2. first_funded_date wins over inception_date when both present.
  3. inception_date wins when first_funded_date is None (backward compat).
  4. Both None -> return None.
  5. yfinance history is queried with the right start date.
  6. BigQuery helper get_first_funded_snapshot_date exists with correct signature.
  7. Regression: the integration glue at paper_trader:474 passes both args.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


def test_phase_38_7_get_benchmark_return_accepts_first_funded_date_kwarg():
    """The function signature must accept the new optional kwarg."""
    import inspect
    from backend.services.paper_trader import _get_benchmark_return
    sig = inspect.signature(_get_benchmark_return)
    params = list(sig.parameters.keys())
    assert "inception_date" in params
    assert "first_funded_date" in params
    # first_funded_date must be optional (default = None)
    assert sig.parameters["first_funded_date"].default is None


def test_phase_38_7_first_funded_wins_over_inception():
    """When first_funded_date is set, it MUST be used as the anchor,
    not inception_date."""
    from backend.services import paper_trader

    mock_hist = MagicMock()
    mock_hist.__len__ = MagicMock(return_value=2)
    mock_hist.__getitem__ = MagicMock(return_value=MagicMock(
        iloc=MagicMock(__getitem__=lambda self, i: 100.0 if i == 0 else 110.0)
    ))

    captured = {}

    class FakeTicker:
        def __init__(self, *args, **kwargs):
            pass

        def history(self, start=None, **kwargs):
            captured["start"] = start
            return mock_hist

    with patch.object(paper_trader.yf, "Ticker", FakeTicker):
        result = paper_trader._get_benchmark_return(
            inception_date="2025-01-01",
            first_funded_date="2025-03-15",
        )
    assert captured["start"] == "2025-03-15", (
        f"Expected SPY query start = '2025-03-15' (first_funded); got {captured['start']!r}"
    )
    assert result == pytest.approx(10.0, abs=0.01), (
        f"Expected ~10% return (100->110); got {result}"
    )


def test_phase_38_7_inception_fallback_when_first_funded_is_none():
    """Backward compat: when first_funded_date is None, fall back to inception_date.
    This preserves behavior for cold-start portfolios that haven't been funded yet."""
    from backend.services import paper_trader

    captured = {}

    class FakeTicker:
        def __init__(self, *args, **kwargs):
            pass

        def history(self, start=None, **kwargs):
            captured["start"] = start
            mock_hist = MagicMock()
            mock_hist.__len__ = MagicMock(return_value=2)
            mock_hist.__getitem__ = MagicMock(return_value=MagicMock(
                iloc=MagicMock(__getitem__=lambda self, i: 100.0 if i == 0 else 105.0)
            ))
            return mock_hist

    with patch.object(paper_trader.yf, "Ticker", FakeTicker):
        result = paper_trader._get_benchmark_return(
            inception_date="2025-01-01",
            first_funded_date=None,
        )
    assert captured["start"] == "2025-01-01", (
        f"Expected fallback to inception when first_funded None; got {captured['start']!r}"
    )


def test_phase_38_7_both_none_returns_none():
    """Empty inception_date AND None first_funded -> None (no anchor)."""
    from backend.services.paper_trader import _get_benchmark_return
    assert _get_benchmark_return("", None) is None
    assert _get_benchmark_return("", first_funded_date=None) is None


def test_phase_38_7_bq_helper_signature():
    """The new BQ helper get_first_funded_snapshot_date must exist on
    BigQueryClient with the right signature."""
    import inspect
    from backend.db.bigquery_client import BigQueryClient
    assert hasattr(BigQueryClient, "get_first_funded_snapshot_date"), (
        "BigQueryClient.get_first_funded_snapshot_date must exist (phase-38.7)"
    )
    method = BigQueryClient.get_first_funded_snapshot_date
    sig = inspect.signature(method)
    # Should take only `self` (no required params beyond it).
    required = [p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty and p.name != "self"]
    assert not required, (
        f"get_first_funded_snapshot_date should take only self; got required: "
        f"{[p.name for p in required]}"
    )


def test_phase_38_7_bq_helper_returns_min_snapshot_date_where_positions_value_gt_zero():
    """The helper must use the correct SQL pattern: MIN(snapshot_date)
    WHERE positions_value > 0 against paper_portfolio_snapshots."""
    src = open("backend/db/bigquery_client.py").read()
    # Locate the helper
    start = src.find("def get_first_funded_snapshot_date")
    end = src.find("def ", start + 1)
    block = src[start:end]
    assert "MIN(snapshot_date)" in block, "must use MIN(snapshot_date)"
    assert "positions_value > 0" in block, "must filter positions_value > 0"
    assert 'paper_portfolio_snapshots' in block, "must query paper_portfolio_snapshots"


def test_phase_38_7_call_site_passes_first_funded_date():
    """The integration glue at paper_trader.py around line 474 must
    call _get_benchmark_return with the new kwarg."""
    src = open("backend/services/paper_trader.py").read()
    # Locate the mark-to-market block
    assert "self.bq.get_first_funded_snapshot_date()" in src, (
        "paper_trader must query first-funded date from BQ"
    )
    # Locate the _get_benchmark_return call site
    call_site_start = src.find("benchmark_ret = _get_benchmark_return(")
    call_site_end = src.find(")", call_site_start) + 1
    # Walk through multi-line calls
    while src[call_site_end - 1] != ")" or src.count("(", call_site_start, call_site_end) > src.count(")", call_site_start, call_site_end):
        call_site_end = src.find(")", call_site_end) + 1
    call_block = src[call_site_start:call_site_end]
    assert "first_funded_date=first_funded" in call_block, (
        f"call site must pass first_funded_date=first_funded; got: {call_block!r}"
    )


def test_phase_38_7_docstring_cites_phase_and_open_9():
    """The new docstring must cite phase-38.7 + OPEN-9 + the anti-pattern
    rationale so future maintainers understand WHY the fix exists."""
    from backend.services.paper_trader import _get_benchmark_return
    doc = _get_benchmark_return.__doc__ or ""
    assert "phase-38.7" in doc
    assert "OPEN-9" in doc
    assert "first" in doc.lower() and "funded" in doc.lower()
