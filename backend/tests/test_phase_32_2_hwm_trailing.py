"""phase-32.2 tests: HWM-trailing stop + Kaminski-Lo adversarial guard.

Audit basis: handoff/archive/phase-31.0/experiment_results.md section 4 P1.2.
Spec source: .claude/masterplan.json::phase-32.2.implementation_plan.test_specs.

Kaminski-Lo Proposition 2 (verbatim from MIT dspace PDF, cited in
handoff/current/research_brief.md):

    "For a mean-reverting portfolio strategy, rho<0; hence, the stop-loss
    policy hurts expected returns to a first-order approximation. This is
    consistent with the intuition that mean-reversion strategies benefit
    from reversals, thus a stop-loss policy that switches out of the
    portfolio after certain cumulative losses will miss the reversal and
    lower the expected return of the portfolio."

The phase-32.2 adversarial guard codifies this: when
`entry_strategy in {'mean_reversion','pairs'}`, the trailing branch is
SKIPPED. Fail-CLOSED-conservative default: when entry_strategy is None
or unknown, treat as momentum (trail IS applied).

Test plan (6 cases):
  1. test_trail_advances_on_new_peak: momentum entry; breakeven already
     fired; mfe rises 20% -> 30%; trail moves up.
  2. test_trail_monotonic_never_moves_down: peak drops after a new high
     (or mfe regresses) -> stop stays at the higher level.
  3. test_kaminski_lo_guard_mean_reversion: entry_strategy='mean_reversion',
     mfe=30% -- stop does NOT trail (stays at breakeven level from 32.1).
  4. test_kaminski_lo_guard_pairs: entry_strategy='pairs', mfe=30% --
     stop does NOT trail.
  5. test_default_momentum_trails: entry_strategy=None (fail-CLOSED
     default), mfe=30% -- stop trails.
  6. test_phase_32_1_breakeven_still_works: regression check that the
     32.1 breakeven branch is unchanged (helper returns the entry_price
     + ISO tuple for the one-shot fire).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.services.paper_trader import PaperTrader


def _mock_settings(
    default_stop_loss_pct: float = 8.0,
    trailing_stop_pct: float = 8.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        paper_price_tolerance_pct=5.0,
        paper_default_stop_loss_pct=default_stop_loss_pct,
        paper_trailing_stop_pct=trailing_stop_pct,
        paper_max_positions=10,
        paper_transaction_cost_pct=0.05,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_trailing_dd_limit_pct=10.0,
        paper_daily_loss_limit_pct=4.0,
    )


def _trader_with_mocks(settings: SimpleNamespace) -> PaperTrader:
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default",
        "current_cash": 5000.0,
        "starting_capital": 10000.0,
        "inception_date": "2026-05-01T00:00:00+00:00",
    }
    bq.get_paper_positions.return_value = []
    return PaperTrader(settings=settings, bq_client=bq)


# ── 1. Trailing advances on new peak ──────────────────────────────


def test_trail_advances_on_new_peak():
    """Momentum entry; breakeven (32.1) already fired (stop_advanced_at_R
    populated, stop = entry); mfe now at +30%. Trail formula:
    new_trail = peak * (1 - trail_pct/100) = entry*1.30 * 0.92 = 1.196 * entry.
    For entry=$100 -> new_trail = $119.60. Must be above the current
    breakeven stop of $100."""
    trader = _trader_with_mocks(_mock_settings(trailing_stop_pct=8.0))
    pos = {
        "ticker": "MU",
        "avg_entry_price": 100.0,
        "stop_loss_price": 100.0,  # breakeven from 32.1
        "stop_advanced_at_R": "2026-05-20T22:00:00+00:00",
        "entry_strategy": "momentum",
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=30.0)
    assert new_stop == pytest.approx(119.60, abs=1e-6)
    # The trail update does NOT overwrite stop_advanced_at_R; advance_iso is None.
    assert advance_iso is None


# ── 2. Monotonic never moves down ─────────────────────────────────


def test_trail_monotonic_never_moves_down():
    """Position previously trailed to $119.60 (from mfe=30%); now mfe
    regressed to 20%. New computed trail = 100*1.20*0.92 = $110.40, BELOW
    the current stop $119.60. Helper must refuse to lower the stop."""
    trader = _trader_with_mocks(_mock_settings(trailing_stop_pct=8.0))
    pos = {
        "ticker": "MU",
        "avg_entry_price": 100.0,
        "stop_loss_price": 119.60,  # already trailed up
        "stop_advanced_at_R": "2026-05-20T22:00:00+00:00",
        "entry_strategy": "momentum",
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=20.0)
    assert new_stop is None
    assert advance_iso is None


# ── 3. Kaminski-Lo guard: mean_reversion ──────────────────────────


def test_kaminski_lo_guard_mean_reversion():
    """entry_strategy='mean_reversion' + mfe=30% -- stop must NOT trail.
    Kaminski-Lo Proposition 2: trailing stops degrade expected return for
    mean-reverting return processes. The position retains its breakeven
    floor (from 32.1) but does not trail above it."""
    trader = _trader_with_mocks(_mock_settings(trailing_stop_pct=8.0))
    pos = {
        "ticker": "XYZ",
        "avg_entry_price": 100.0,
        "stop_loss_price": 100.0,
        "stop_advanced_at_R": "2026-05-20T22:00:00+00:00",
        "entry_strategy": "mean_reversion",
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=30.0)
    assert new_stop is None
    assert advance_iso is None


# ── 4. Kaminski-Lo guard: pairs ───────────────────────────────────


def test_kaminski_lo_guard_pairs():
    """entry_strategy='pairs' + mfe=30% -- stop must NOT trail.
    Cointegrated pairs strategies share the mean-reverting dynamic per
    Kaminski-Lo's framing."""
    trader = _trader_with_mocks(_mock_settings(trailing_stop_pct=8.0))
    pos = {
        "ticker": "ABC",
        "avg_entry_price": 100.0,
        "stop_loss_price": 100.0,
        "stop_advanced_at_R": "2026-05-20T22:00:00+00:00",
        "entry_strategy": "pairs",
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=30.0)
    assert new_stop is None
    assert advance_iso is None


# ── 5. Fail-CLOSED default: None / unknown trails ─────────────────


def test_default_momentum_trails_when_entry_strategy_is_none():
    """entry_strategy=None (legacy / forgot-to-flag) + mfe=30% -- stop
    MUST trail. Fail-CLOSED-conservative default: forgetting to flag a
    mean-reversion entry should err toward MORE protection (trail
    applied), not less (no protection)."""
    trader = _trader_with_mocks(_mock_settings(trailing_stop_pct=8.0))
    pos = {
        "ticker": "MU",
        "avg_entry_price": 100.0,
        "stop_loss_price": 100.0,
        "stop_advanced_at_R": "2026-05-20T22:00:00+00:00",
        "entry_strategy": None,
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=30.0)
    assert new_stop == pytest.approx(119.60, abs=1e-6)
    assert advance_iso is None


# ── 6. Regression: phase-32.1 breakeven branch still fires ────────


def test_phase_32_1_breakeven_branch_unchanged():
    """Position without stop_advanced_at_R (breakeven has NOT yet fired);
    mfe crosses +1R threshold. Helper must still return (entry, ISO_now)
    as the one-shot breakeven fire. Regression check for 32.1."""
    trader = _trader_with_mocks(_mock_settings(default_stop_loss_pct=8.0))
    pos = {
        "ticker": "FOO",
        "avg_entry_price": 100.0,
        "stop_loss_price": 92.0,  # entry-anchored static
        "stop_advanced_at_R": None,
        "entry_strategy": "momentum",
    }
    new_stop, advance_iso = trader._advance_stop(pos, new_mfe=10.0)
    assert new_stop == 100.0
    assert isinstance(advance_iso, str) and advance_iso.endswith("+00:00")
