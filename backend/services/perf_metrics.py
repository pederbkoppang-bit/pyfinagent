"""
PerformanceSkill — Single source of truth for all P&L and portfolio metrics.

Every optimizer loop, API endpoint, and paper-trading module should delegate
to these canonical formulas instead of computing independently.

Scalar metric for optimization:
    get_scalar_metric() = risk_adjusted_return × (1 − min(0.3, turnover_ratio × tx_cost_pct))

Research basis:
    - analytics.py Sharpe (risk-free adjusted, √252 annualized) is THE Sharpe formula
    - López de Prado Ch. 3-8 for backtest-specific metrics (DSR, sample weights)
    - Transaction cost penalty prevents over-trading (not in Karpathy — our hybrid extension)
"""

from __future__ import annotations

import statistics

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from backend.backtest.analytics import compute_sharpe, compute_max_drawdown

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant, used in DSR threshold (Bailey & Lopez de Prado 2014, Eq. 8).
_EULER_GAMMA = 0.5772156649015329


# ── Position-Level Metrics ───────────────────────────────────────


def compute_position_pnl(
    quantity: float, current_price: float, cost_basis: float
) -> tuple[float, float]:
    """
    Canonical position P&L.

    Returns:
        (unrealized_pnl, unrealized_pnl_pct)
    """
    market_value = quantity * current_price
    pnl = market_value - cost_basis
    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
    return round(pnl, 2), round(pnl_pct, 2)


def compute_return_pct(
    current_price: float, entry_price: float
) -> float:
    """Simple return percentage: ((current - entry) / entry) × 100."""
    if entry_price <= 0:
        return 0.0
    return round(((current_price - entry_price) / entry_price) * 100, 2)


# ── Portfolio-Level Metrics ──────────────────────────────────────


def compute_portfolio_pnl(
    nav: float, starting_capital: float
) -> tuple[float, float]:
    """
    Portfolio-level P&L.

    Returns:
        (total_pnl, total_pnl_pct)
    """
    pnl = nav - starting_capital
    pnl_pct = (pnl / starting_capital * 100) if starting_capital > 0 else 0.0
    return round(pnl, 2), round(pnl_pct, 2)


def compute_alpha(
    portfolio_pnl_pct: float, benchmark_pnl_pct: float
) -> float:
    """Alpha = portfolio cumulative return − benchmark cumulative return."""
    return round(portfolio_pnl_pct - benchmark_pnl_pct, 2)


def compute_sharpe_from_snapshots(
    snapshots: list[dict],
    nav_key: str = "total_nav",
    risk_free_rate: float = 0.04,
) -> float:
    """
    Compute Sharpe ratio from daily NAV snapshots.

    Converts NAV series → daily returns, then delegates to the canonical
    analytics.compute_sharpe (risk-free adjusted, √252 annualized).
    """
    if len(snapshots) < 6:
        return 0.0

    # phase-47.4: get_paper_snapshots returns rows newest-first (ORDER BY
    # snapshot_date DESC). Sharpe is order-sensitive -- np.diff on a reversed
    # NAV series negates every daily return and flips the Sharpe SIGN (observed
    # cockpit -5.72 vs the correct positive; Sortino stayed + because its path
    # was already chronological). Sort chronologically so EVERY caller of this
    # canonical helper is correct. compute_paper_sharpe_window already
    # pre-sorts, so this is an idempotent belt-and-suspenders for it.
    snapshots = sorted(snapshots, key=lambda s: s.get("snapshot_date") or "")

    navs = [s.get(nav_key, 0) for s in snapshots if s.get(nav_key)]
    if len(navs) < 6:
        return 0.0

    nav_arr = np.array(navs, dtype=float)
    # Daily returns from NAV series
    daily_returns = np.diff(nav_arr) / nav_arr[:-1]

    sharpe = compute_sharpe(daily_returns, risk_free_rate)
    # Belt-and-suspenders clamp: a real annualized Sharpe outside
    # [-100, 100] is almost always a float-precision artifact. Return
    # 0.0 rather than surface astronomical garbage to the UI.
    if not np.isfinite(sharpe) or abs(sharpe) > 100.0:
        return 0.0
    return round(sharpe, 2)


def compute_max_drawdown_from_snapshots(
    snapshots: Sequence[dict],
    nav_key: str = "total_nav",
    snapshot_date_key: str = "snapshot_date",
) -> Optional[float]:
    """phase-80.40: canonical MAX peak-to-trough drawdown over a NAV-snapshot
    series, as a NEGATIVE percent. Returns None when it cannot be measured.

    Structural twin of compute_sharpe_from_snapshots above: takes the
    ALREADY-FETCHED snapshot list (never a bq client -- see "no extra query"
    below), normalizes it, and delegates the arithmetic to the single canonical
    primitive `analytics.compute_max_drawdown` (imported at module top). This
    module is the documented single source of truth for drawdown
    (`.claude/rules/backend-services.md`: "Never compute Sharpe, drawdown, or
    alpha outside perf_metrics.py -- import from there"), and NO new formula is
    written here.

    SIGN CONVENTION: NEGATIVE PERCENT (-5.31 means a 5.31% peak-to-trough
    decline); 0.0 means the series never fell below its running high-water mark
    TO 2 DECIMAL PLACES -- see "PRECISION FLOOR" below, which qualifies that claim.
    This is deliberate and load-bearing in three ways:
      1. It matches what the backtest side already publishes --
         backtest_engine._max_drawdown -> analytics.compute_max_drawdown ->
         api/backtest.py "max_drawdown_pct" -- so the paper number and the
         backtest number are directly comparable on the reality-gap card.
      2. The cockpit consumes it as `maxDd > -10 ? SAFE : maxDd > -13 ?
         WARNING : DANGER` (cockpit-helpers.tsx). A POSITIVE-magnitude value
         would satisfy `> -10` at EVERY depth, rendering emerald SAFE forever
         -- i.e. it would INVERT a safety verdict, silently.
      3. The split is real but narrower than "genuinely split" implied (corrected
         2026-07-26 per a research-gate finding): R's PerformanceAnalytics::
         maxDrawdown does ship `invert=TRUE` by default (positive magnitude,
         verified), but Python's empyrical, pyfolio, and quantstats are
         UNANIMOUSLY negative -- the divide is across the R/Python ecosystem
         boundary, not evenly split within either. Reason 2 above is decisive
         regardless of that framing; the sign is pinned by an assertion in
         backend/tests/test_phase_80_40_perf_metrics_drawdown.py, not by this
         docstring.
    DO NOT delegate to backend/services/paper_go_live_gate.py::_snapshot_max_dd_pct
    -- it is the same measurement with the OPPOSITE sign (positive magnitude).

    None, NEVER 0.0, ON EVERY DEGRADED PATH. `analytics.compute_max_drawdown`
    returns 0.0 for an empty series, for a single element AND for a monotonic
    rise -- three states that are indistinguishable in a JSON payload. Under
    phase-80.36's deliberate presence-not-value contract a numeric 0 renders a
    real emerald SAFE, so a 0.0 fallback would reintroduce the exact fabrication
    this step exists to kill, one layer deeper. Absence propagates instead.

    NON-FINITE IS AN OUTAGE, NOT A COSMETIC LEAK. compute_max_drawdown returns
    nan for an all-zero series or any series containing a nan, and FastAPI
    REFUSES to serialize nan ("Out of range float values are not JSON
    compliant"), which would 500 the WHOLE /performance payload -- taking nav,
    sharpe_ratio and round_trip_summary down with it (the phase-80.27 NaN class).
    Non-finite NAVs are dropped and a non-finite result returns None.

    ORDER: sorted ASC internally, never trusting the caller.
    `bq.get_paper_snapshots` is `ORDER BY snapshot_date DESC`, and walking a NAV
    series backwards reads growth as a crash. That has already caused TWO
    production incidents on THIS series: a 60.08% phantom drawdown (phase-47.4,
    paper_go_live_gate.py:47-68) and a phantom -61.51% P1 page against a book UP
    20% (phase-66.2, drawdown_alarm.py:58-108).

    WINDOW: whatever the caller passes -- there is no internal slice. The
    /performance caller passes its full limit=365 series (64 rows today), so the
    published figure is ALL-TIME max drawdown. Contrast paper_go_live_gate.py:139
    which measures the same book over a 30-row window (1.19% vs 5.31% -- a 4.5x
    difference from the window choice alone).

    MAX, NOT CURRENT: this is a historical extreme that never recovers.
    backend/services/kill_switch.py:315-319 trips on CURRENT trailing drawdown
    from a persisted peak (positive magnitude, 10% limit). The two legitimately
    differ; see the "two drawdown ladders" note in
    frontend/src/components/paper-trading/cockpit-helpers.tsx.

    FLOW-BLIND (raw NAV), deliberately scoped: an external deposit/withdrawal
    moves NAV without being a return. On the live series (one +$5,000 deposit on
    2026-05-13) raw NAV gives -5.31% vs -5.73% flow-adjusted. Raw matches the
    kill-switch mechanism and paper_go_live_gate's existing realized-DD number.
    Pending step 72.2.4 owns routing NAV->returns through the canonical GIPS
    helper paper_metrics_v2._nav_to_returns for compute_sharpe_from_snapshots;
    when it lands, this helper is the second leg of that same fix. NOTE a future
    WITHDRAWAL would fabricate a drawdown here until then.

    NO EXTRA QUERY / NO EVENT-LOOP RISK: pure numpy over <=365 floats
    (measured 0.0365 ms per call on a 384-row series), so it needs no
    asyncio.to_thread wrapper -- a thread handoff would cost more than the work.
    Callers MUST pass a list they already hold rather than triggering a second
    get_paper_snapshots round trip (measured ~1.2s).
    """
    if not snapshots:
        return None

    # phase-80.43: VALIDATE THE SORT KEY BEFORE TRUSTING THE SORT.
    #
    # This ASC sort exists precisely to stop a DESC-ordered BQ series being read
    # backwards -- the phantom-drawdown class this function's docstring cites
    # (phase-47.4 / phase-66.2). But `str(s.get(key) or "")` mapped missing, None
    # and "" all to the empty string, which collates FIRST regardless of real
    # chronology. If EVERY row lacked the key the sort became a NO-OP and the
    # caller's original DESC order survived untouched -- so a RISING book would
    # publish a phantom NEGATIVE drawdown, silently. `total_nav` was validated
    # hard here while the key that establishes ORDER was not validated at all.
    #
    # A series whose chronology cannot be established is UNMEASURABLE, not zero
    # and not a computed number: returning None matches this module's existing
    # presence-not-value contract (None = could not be measured, a numeric =
    # measured) and makes the cockpit show "unmeasurable" instead of a fabricated
    # decline. Deliberately strict -- ALL rows must carry a usable key, because a
    # partially-dated series mis-orders exactly as badly as an undated one: the
    # undated rows silently collate to the front and become the "peak".
    missing_dates = [
        s for s in snapshots
        if not str(s.get(snapshot_date_key) or "").strip()
    ]
    if missing_dates:
        logger.warning(
            "perf_metrics: %d of %d snapshots lack a usable %r -- chronology "
            "cannot be established, so max-drawdown is UNMEASURABLE (returning "
            "None) rather than computed from an unknown ordering.",
            len(missing_dates), len(snapshots), snapshot_date_key,
        )
        return None

    try:
        ordered = sorted(snapshots, key=lambda s: str(s.get(snapshot_date_key) or ""))
    except Exception:  # pragma: no cover - defensive; snapshots are plain dicts
        ordered = list(snapshots)

    navs: list[float] = []
    for snap in ordered:
        raw = snap.get(nav_key)
        if raw is None:
            continue
        try:
            nav = float(raw)
        except (TypeError, ValueError):
            continue
        # `nav > 0` also drops nan (nan > 0 is False); isfinite additionally
        # drops +inf, which would otherwise yield a spurious -100.0.
        if nav > 0 and math.isfinite(nav):
            navs.append(nav)

    if len(navs) < 2:
        return None

    dd_pct = compute_max_drawdown(np.array(navs, dtype=float))
    if not math.isfinite(dd_pct):
        return None
    # PRECISION FLOOR (phase-80.45, disposition (a): ACCEPT AND DOCUMENT).
    # Rounding to 2dp means any real decline shallower than 0.005% collapses to
    # -0.0 and is then normalized to 0.0 below -- so a published 0.0 asserts
    # "never fell below the high-water mark TO 2DP", NOT "never fell at all".
    # Measured on the real book: a 19-cent dip on a $23,838 NAV is
    # -0.0006711927430634515 and publishes as 0.0.
    #
    # Accepted deliberately rather than fixed. 2dp is the display precision the
    # cockpit renders and the backtest side publishes, so widening it here would
    # split the paper/backtest comparability that this function exists to
    # preserve; and the practical difference between a 0.0007% drawdown and none
    # is nil for every risk decision downstream (the cockpit bands are -10/-13).
    # The defect was the CLAIM over-reaching, not the arithmetic -- so the claim
    # is narrowed instead. Pinned by test_phase_80_45_precision_floor_is_documented
    # so this comment and the code cannot drift apart.
    rounded = round(float(dd_pct), 2)
    # Normalize -0.0 -> 0.0 so the payload never carries a negative zero.
    # NOT a clamp: a positive result would mean the sign convention broke and
    # must surface to the test, not be silently absorbed.
    return 0.0 if rounded == 0.0 else rounded


def compute_paper_sharpe_window(
    bq: Any,
    *,
    window_days: int = 30,
    risk_free_rate: float = 0.04,
    nav_key: str = "total_nav",
    snapshot_date_key: str = "snapshot_date",
) -> Optional[float]:
    """phase-43.0 cycle-16: trailing N-day paper-Sharpe helper.

    Pulls paper-portfolio snapshots, sorts by snapshot_date ascending, slices
    the last `window_days` entries, and delegates to compute_sharpe_from_snapshots.
    Returns None on insufficient data (window_days < 6 or fewer-than-6
    populated NAVs after slice).

    Reuses the canonical compute_sharpe_from_snapshots primitive (line 87);
    does NOT fork the Sharpe computation. Window slice is the only delta.

    Used by compute_sharpe_gap(window_days=N) to compute a window-matched
    live Sharpe per the corrected DoD-2 wording (cycle 15: gap_rel <=
    SR_GAP_THRESHOLD, per Jacquier-Muhle-Karbe arXiv:2501.03938 30% IS-to-OOS
    decay; statistically valid measurement window per Bailey-LdP MinTRL with
    the explicit understanding that n=30 has SE ~= 0.3 -- so gap_rel comparisons
    against the 30% threshold are at the edge of detectability, but at least
    they are well-defined and not infinite-noise like the deprecated 0.01
    absolute criterion was).
    """
    if window_days < 6:
        return None
    try:
        # Pull window_days * 2 to give headroom for missing days / sort robustness.
        snapshots = bq.get_paper_snapshots(limit=max(window_days * 2, 60)) or []
    except Exception:
        return None
    if not snapshots:
        return None

    # Sort ascending by snapshot_date so the last `window_days` is the trailing window.
    try:
        snapshots_sorted = sorted(snapshots, key=lambda s: str(s.get(snapshot_date_key, "")))
    except Exception:
        snapshots_sorted = snapshots
    window = snapshots_sorted[-window_days:]
    if len(window) < 6:
        return None
    sharpe = compute_sharpe_from_snapshots(
        window, nav_key=nav_key, risk_free_rate=risk_free_rate
    )
    if sharpe == 0.0:
        # 0.0 from the helper means could-not-compute; treat as None.
        return None
    return sharpe


# ── Live-vs-Backtest Sharpe reconciliation (phase-25.A6) ─────────


_OPTIMIZER_BEST_PATH = Path(__file__).resolve().parents[1] / "backtest" / "experiments" / "optimizer_best.json"

# phase-25.A6: industry-benchmark Sharpe-gap threshold. Jacquier et al.
# arxiv 2501.03938 (Jan 2025): 30-50% IS-to-OOS decay range; 30% is the
# stricter lower bound for a single-strategy system. Man Group uses 50%.
# This constant mirrors backend/services/paper_go_live_gate.py:38 -- both
# point at the same value so a future change is centralized.
SR_GAP_THRESHOLD = 0.30


def _load_optimizer_best_sharpe() -> Optional[float]:
    """Read `optimizer_best.json["sharpe"]` if available; None otherwise."""
    try:
        if not _OPTIMIZER_BEST_PATH.exists():
            return None
        with _OPTIMIZER_BEST_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sharpe = data.get("sharpe")
        return float(sharpe) if sharpe is not None else None
    except Exception:
        return None


def _shadow_curve_sharpe(
    bq: Any, min_points: int, risk_free_rate: float,
) -> Optional[float]:
    """Compute Sharpe from the reconciliation shadow-backtest NAV curve.

    The shadow curve is the existing backtest-equivalent path that
    `compute_reconciliation` already builds (frictionless yfinance fills).
    Returns None on any failure (caller falls through to the next tier).
    """
    try:
        from backend.services.reconciliation import compute_reconciliation
        recon = compute_reconciliation(bq) or {}
        series = recon.get("series") or []
        if len(series) < min_points:
            return None
        navs = [float(pt.get("backtest_nav") or 0.0) for pt in series]
        navs = [n for n in navs if n > 0]
        if len(navs) < min_points:
            return None
        # Mirror the formula in compute_sharpe_from_snapshots for parity.
        nav_arr = np.array(navs, dtype=float)
        daily_returns = np.diff(nav_arr) / nav_arr[:-1]
        sharpe = compute_sharpe(daily_returns, risk_free_rate)
        if not np.isfinite(sharpe) or abs(sharpe) > 100.0:
            return None
        return round(float(sharpe), 2)
    except Exception:
        return None


def _reconciliation_divergence_pct(bq: Any) -> Optional[float]:
    """Tertiary fallback: latest reconciliation divergence pct (legacy proxy)."""
    try:
        from backend.services.reconciliation import compute_reconciliation
        recon = compute_reconciliation(bq) or {}
        summary = recon.get("summary") or {}
        v = summary.get("latest_divergence_pct")
        return float(v) if v is not None else None
    except Exception:
        return None


def compute_sharpe_gap(
    bq: Any,
    *,
    backtest_sharpe_source: str = "optimizer_best",
    risk_free_rate: float = 0.04,
    min_snapshots: int = 6,
    window_days: Optional[int] = None,
) -> dict:
    """phase-25.A6: explicit live-vs-backtest Sharpe gap. Industry benchmark
    threshold 30% (Jacquier et al. arxiv 2501.03938; 30-50% IS-to-OOS decay
    range, 30% the stricter lower bound).

    Closes phase-24.6 F-3 -- `paper_go_live_gate.compute_gate` previously used
    NAV-divergence as a proxy for the Sharpe gap. NAV divergence is a dollar
    quantity; Sharpe gap is a risk-adjusted-return quantity. They are not
    the same measurement.

    Fallback chain (returned in `source`):
      1. "optimizer_best": read `optimizer_best.json["sharpe"]`.
      2. "shadow_curve": derive backtest Sharpe from `reconciliation.series.backtest_nav`.
      3. "proxy_fallback": use `reconciliation.summary.latest_divergence_pct / 100`
         as `gap_rel`; sets `proxy_fallback=True` so operators see the legacy proxy.
      4. "no_data": all of the above unavailable; `gap_within_threshold=None`
         so the calling gate stays red.

    Returns dict with keys:
      live_sharpe, backtest_sharpe, gap_abs, gap_rel, threshold,
      gap_within_threshold, source, note, proxy_fallback, computed_at.
    """
    threshold = SR_GAP_THRESHOLD
    note_parts: list[str] = []

    # Live Sharpe from paper snapshots. phase-43.0 cycle-16: when window_days
    # is set, use the trailing-N-day helper for a window-matched gap measurement
    # consistent with the corrected DoD-2 threshold (gap_rel <= SR_GAP_THRESHOLD
    # per Jacquier 2025; the window_days kwarg makes the live Sharpe arm
    # explicit). When window_days is None the prior all-time-snapshot behavior
    # is preserved byte-for-byte for backward compatibility.
    live_sharpe: Optional[float] = None
    if window_days is not None:
        live_sharpe = compute_paper_sharpe_window(
            bq, window_days=window_days, risk_free_rate=risk_free_rate
        )
        if live_sharpe is None:
            note_parts.append(f"windowed_paper_sharpe_unavailable:window_days={window_days}")
    else:
        snapshots: list[dict] = []
        try:
            snapshots = bq.get_paper_snapshots(limit=365) or []
        except Exception as exc:
            note_parts.append(f"paper_snapshots_fail:{type(exc).__name__}")
            snapshots = []
        if len(snapshots) >= min_snapshots:
            live_sharpe = compute_sharpe_from_snapshots(snapshots, risk_free_rate=risk_free_rate)
            if live_sharpe == 0.0:
                # 0.0 from the helper means "could not compute" (clamped or insufficient).
                # Treat as None to make the no-data path unambiguous.
                live_sharpe = None

    # Backtest Sharpe with fallback chain.
    backtest_sharpe: Optional[float] = None
    proxy_fallback = False
    source = "no_data"

    if backtest_sharpe_source != "optimizer_best":
        note_parts.append(f"requested_source={backtest_sharpe_source}")

    primary = _load_optimizer_best_sharpe()
    if primary is not None:
        backtest_sharpe = primary
        source = "optimizer_best"
    else:
        secondary = _shadow_curve_sharpe(bq, min_snapshots, risk_free_rate)
        if secondary is not None:
            backtest_sharpe = secondary
            source = "shadow_curve"

    gap_abs: Optional[float] = None
    gap_rel: Optional[float] = None
    gap_within_threshold: Optional[bool] = None

    if live_sharpe is not None and backtest_sharpe is not None and abs(backtest_sharpe) > 0:
        gap_abs = abs(live_sharpe - backtest_sharpe)
        gap_rel = gap_abs / abs(backtest_sharpe)
        gap_within_threshold = gap_rel <= threshold
    else:
        # Final fallback: legacy NAV-divergence proxy if backtest Sharpe is unavailable.
        divergence_pct = _reconciliation_divergence_pct(bq)
        if divergence_pct is not None:
            proxy_fallback = True
            source = "proxy_fallback"
            gap_rel = divergence_pct / 100.0
            gap_within_threshold = gap_rel <= threshold
            note_parts.append("using NAV-divergence proxy; backtest Sharpe unavailable")
        else:
            note_parts.append("no backtest Sharpe + no divergence proxy; gate stays red")

    note = "; ".join(note_parts) if note_parts else None
    return {
        "live_sharpe": live_sharpe,
        "backtest_sharpe": backtest_sharpe,
        "gap_abs": round(gap_abs, 4) if gap_abs is not None else None,
        "gap_rel": round(gap_rel, 4) if gap_rel is not None else None,
        "threshold": threshold,
        "gap_within_threshold": gap_within_threshold,
        "source": source,
        "note": note,
        "proxy_fallback": proxy_fallback,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Benchmark Comparison ─────────────────────────────────────────


def compute_benchmark_return(holding_days: int, annual_rate: float = 0.10) -> float:
    """
    Expected benchmark return for a holding period.

    Uses geometric compounding (not simple arithmetic) for accuracy
    over long holding periods.

    Args:
        holding_days: Number of calendar days held.
        annual_rate: Assumed annual benchmark return (default 10% for SPY).

    Returns:
        Expected return percentage.
    """
    if holding_days <= 0:
        return 0.0
    daily_rate = (1 + annual_rate) ** (1 / 365) - 1
    return round(((1 + daily_rate) ** holding_days - 1) * 100, 2)


def beat_benchmark(
    return_pct: float, holding_days: int, annual_rate: float = 0.10
) -> bool:
    """Did the position beat the geometric benchmark return?"""
    return return_pct > compute_benchmark_return(holding_days, annual_rate)


# ── Turnover & Transaction Costs ─────────────────────────────────


def compute_turnover_ratio(
    trades: list[dict],
    avg_nav: float,
    period_days: int = 365,
) -> float:
    """
    Annual portfolio turnover = total sell value / average NAV.

    Higher turnover → higher transaction cost drag.
    """
    if avg_nav <= 0 or period_days <= 0:
        return 0.0
    sell_value = sum(
        abs(t.get("total_value", 0))
        for t in trades
        if t.get("action") == "SELL"
    )
    # Annualize
    annualized = sell_value * (365 / period_days)
    return round(annualized / avg_nav, 4)


def compute_tx_cost_drag(
    turnover_ratio: float, tx_cost_pct: float = 0.001
) -> float:
    """
    Transaction cost drag as a fraction of returns.

    Capped at 0.3 (30%) to prevent degenerate values.
    """
    return min(0.3, turnover_ratio * tx_cost_pct)


# ── Scalar Optimization Metric ───────────────────────────────────


@dataclass
class ScalarMetricInputs:
    """Inputs for the unified scalar metric."""
    avg_return_pct: float
    benchmark_beat_rate: float
    turnover_ratio: float = 0.0
    tx_cost_pct: float = 0.001  # 0.1% default


def get_scalar_metric(inputs: ScalarMetricInputs) -> float:
    """
    THE single metric all optimization loops converge on.

    scalar = risk_adjusted_return × (1 − tx_cost_drag)

    where:
        risk_adjusted_return = avg(return_pct) × beat_benchmark_rate
        tx_cost_drag = min(0.3, turnover_ratio × tx_cost_pct)

    This extends Karpathy's single-metric approach with a transaction cost
    penalty that prevents the optimizer from discovering "churn alpha" —
    strategies that look good on paper but lose to friction in practice.
    """
    risk_adjusted = inputs.avg_return_pct * inputs.benchmark_beat_rate
    drag = compute_tx_cost_drag(inputs.turnover_ratio, inputs.tx_cost_pct)
    return round(risk_adjusted * (1 - drag), 4)


def get_scalar_metric_from_bq(bq_client, trades: Optional[list] = None) -> float:
    """
    Convenience: compute scalar metric from BigQuery performance stats
    and optional trade history.
    """
    stats = bq_client.get_performance_stats()
    avg_return = stats.get("avg_return") or 0.0
    benchmark_rate = stats.get("benchmark_beat_rate") or 0.0

    turnover = 0.0
    if trades:
        # Rough NAV estimate from trade values
        total_value = sum(abs(t.get("total_value", 0)) for t in trades)
        avg_nav = total_value / max(len(trades), 1)
        if avg_nav > 0:
            turnover = compute_turnover_ratio(trades, avg_nav)

    return get_scalar_metric(ScalarMetricInputs(
        avg_return_pct=avg_return,
        benchmark_beat_rate=benchmark_rate,
        turnover_ratio=turnover,
    ))


# ── Advanced Risk-Adjusted Metrics (PSR, DSR, Sortino, Calmar) ───


def _sr_per_period(returns: np.ndarray, ddof: int = 1) -> float:
    """Non-annualized per-period Sharpe. PSR/DSR require this form, not annualized."""
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=ddof))
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std)


def compute_psr(returns: Sequence[float], sr_star: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012, Eq. 9).

    PSR(SR*) = CDF((SR_hat - SR*) * sqrt(n-1) / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2))

    Where g4 is RAW kurtosis (normal = 3). scipy's default is excess kurtosis --
    we add 3. Denominator guarded to avoid division-by-zero at extreme SR.
    """
    arr = np.asarray(list(returns), dtype=float)
    n = len(arr)
    if n < 5:
        return 0.0
    from scipy.stats import norm, skew, kurtosis
    sr_hat = _sr_per_period(arr)
    g3 = float(skew(arr, bias=False))
    g4 = float(kurtosis(arr, fisher=True, bias=False)) + 3.0  # excess -> raw
    var_term = 1.0 - g3 * sr_hat + (g4 - 1.0) / 4.0 * sr_hat * sr_hat
    var_term = max(var_term, 1e-8)
    z = (sr_hat - sr_star) * math.sqrt(n - 1) / math.sqrt(var_term)
    return float(norm.cdf(z))


def compute_dsr(
    returns: Sequence[float],
    all_trial_sharpes: Sequence[float],
    n_trials: Optional[int] = None,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, Eq. 8 + Eq. 10).

    Corrects PSR for selection bias when N strategies were tested. The threshold
    SR* is derived from the cross-sectional variance of trial Sharpes and the
    expected maximum of N draws from that distribution.

    SR* = sqrt(Var[SR]) * ((1-gamma)*PPF(1-1/N) + gamma*PPF(1-1/(N*e)))
    DSR = PSR(SR*)
    """
    arr = np.asarray(list(returns), dtype=float)
    trials = np.asarray(list(all_trial_sharpes), dtype=float)
    n = n_trials if n_trials is not None else max(len(trials), 2)
    if len(arr) < 5 or n < 2 or len(trials) < 2:
        return 0.0
    from scipy.stats import norm
    var_sr = float(trials.var(ddof=1))
    if var_sr <= 0.0:
        return compute_psr(arr, sr_star=0.0)
    sr_star = math.sqrt(var_sr) * (
        (1.0 - _EULER_GAMMA) * float(norm.ppf(1.0 - 1.0 / n))
        + _EULER_GAMMA * float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    )
    return compute_psr(arr, sr_star=sr_star)


def compute_sortino(
    returns: Sequence[float], mar: float = 0.0, periods_per_year: int = 252
) -> float:
    """Sortino ratio, annualized. MAR default 0 (downside = negative returns)."""
    arr = np.asarray(list(returns), dtype=float)
    if len(arr) < 5:
        return 0.0
    excess = arr - mar
    downside = excess[excess < 0.0]
    if len(downside) < 2:
        return 0.0
    dstd = float(downside.std(ddof=1))
    if dstd == 0.0:
        return 0.0
    return float(excess.mean() / dstd * math.sqrt(periods_per_year))


def compute_calmar(
    returns: Sequence[float], periods_per_year: int = 252
) -> float:
    """Calmar = annualized return / |max drawdown|. Returns 0 if no drawdown."""
    arr = np.asarray(list(returns), dtype=float)
    if len(arr) < 5:
        return 0.0
    cum = np.cumprod(1.0 + arr)
    dd_pct = compute_max_drawdown(cum)
    if dd_pct >= 0.0:
        return 0.0
    annualized = float(arr.mean() * periods_per_year)
    return float(annualized / (abs(dd_pct) / 100.0))


def compute_rolling_sharpe_bootstrap_ci(
    returns: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.04,
    seed: Optional[int] = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI on annualized Sharpe. Percentile method by default; falls
    back to stationary block bootstrap when |lag-1 autocorr| > 0.2 to preserve
    serial dependence (Politis & Romano 1994).

    Returns (sharpe_point, ci_low, ci_high).
    """
    arr = np.asarray(list(returns), dtype=float)
    n = len(arr)
    if n < 10:
        return 0.0, 0.0, 0.0

    point = compute_sharpe(arr, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

    # Lag-1 autocorr check
    mean = arr.mean()
    centered = arr - mean
    denom = float((centered * centered).sum())
    if denom == 0.0:
        return point, point, point
    acf1 = float((centered[:-1] * centered[1:]).sum() / denom)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)

    if abs(acf1) <= 0.2:
        # Standard IID bootstrap
        idx = rng.integers(0, n, size=(n_resamples, n))
        for i in range(n_resamples):
            samples[i] = compute_sharpe(arr[idx[i]], risk_free_rate, periods_per_year)
    else:
        # Stationary block bootstrap: geometric block lengths with mean ~ sqrt(n)
        p = 1.0 / max(1.0, math.sqrt(n))
        for i in range(n_resamples):
            resample = np.empty(n, dtype=float)
            t = 0
            start = int(rng.integers(0, n))
            while t < n:
                resample[t] = arr[start % n]
                if rng.random() < p:
                    start = int(rng.integers(0, n))
                else:
                    start += 1
                t += 1
            samples[i] = compute_sharpe(resample, risk_free_rate, periods_per_year)

    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(samples, alpha))
    high = float(np.quantile(samples, 1.0 - alpha))
    return point, low, high


# ─────────────────────────────────────────────────────────────────────
# phase-82.5: exit quality (capture ratio / edge ratio)
#
# THE MEAN OF THESE RATIOS DOES NOT EXIST. Both are ratios whose
# denominator can reach zero, which makes the per-trade distribution
# Cauchy-like; Franz (arXiv:0710.2024) proves "neither the expected
# value nor the variance exist", and that the mean of N such variables
# follows the SAME distribution as one of them. So the shipped
# `avg_capture_ratio` of -42.08 was not a bad estimate of a true value
# -- there is no true value, and more trades would never make it
# converge. That also rules out winsorizing or clipping: those produce
# a finite number that estimates no population parameter. The ESTIMATOR
# has to change, not the tails.
#
# The two degeneracies need OPPOSITE treatment, which is why there is
# no single uniform rule here:
#   * mae == 0 means the trade never traded against us -- a real,
#     desirable property. Dropping those rows (as the endpoint did)
#     deletes the BEST trades: survivorship bias with the sign pointing
#     the wrong way. Keep them, ranked at +inf.
#   * mfe == 0 means there was no exit decision to grade -- the ENTRY
#     failed. Scoring it 0.0 blames the exit for an entry failure.
#     Exclude it.
# A median is defined over the extended reals, so +inf can be RANKED
# rather than deleted; a mean cannot, which is the only reason the
# `if mae_abs > 0` filter existed.
# ─────────────────────────────────────────────────────────────────────

# Below this MFE there was no economically meaningful move to capture:
# the round-trip cost is charged twice, so the "available move" sits
# inside the cost+noise band and the ratio's denominator is noise. Every
# published interpretive threshold (0.40 / 0.60 / 0.75) is a fraction of
# a meaningful move, so grading against a sub-1pp denominator compares
# against a different scale. This one floor subsumes both pathologies:
# it removes the exact mfe==0 rows AND the near-zero ones (the real
# 000660.KS row had mfe=0.0001 -> capture -1269.57).
MIN_MFE_PCT = 1.0


def compute_capture_ratio(
    realized_pnl_pct: float,
    mfe_pct: float,
    min_mfe_pct: float = MIN_MFE_PCT,
) -> Optional[float]:
    """Fraction of the favourable excursion actually realised, or None.

    Returns None -- NOT 0.0 -- when the trade never offered a gradeable
    move. 0.0 is a real, distinguishable outcome ("captured nothing of a
    real move"); conflating it with "undefined" is what let 8 fabricated
    zeros into a 32-row average.
    """
    try:
        mfe = float(mfe_pct)
        # `mfe > 0` is UNCONDITIONAL; the floor is an ADDITIONAL constraint on
        # top of it, never a way to admit zero. Written as one combined
        # comparison this reads fine and is wrong: at min_mfe_pct=0.0 a zero MFE
        # satisfies `mfe < 0.0 == False` and falls straight through to a
        # ZeroDivisionError. That is the same shape as the bug being fixed here
        # -- a guard whose threshold does not actually cover its domain -- so it
        # is spelled out rather than folded together.
        if not math.isfinite(mfe) or mfe <= 0.0:
            return None
        if mfe < float(min_mfe_pct):
            return None
        pnl = float(realized_pnl_pct)
        if not math.isfinite(pnl):
            return None
        return pnl / mfe
    except (TypeError, ValueError):
        return None


def compute_edge_ratio(mfe_pct: float, mae_pct: float) -> Optional[float]:
    """MFE / |MAE| over the EXTENDED reals: +inf when there was no adverse
    excursion at all, None only when neither excursion happened.

    +inf is the mathematically honest value for a trade that never went
    against us, and it RANKS correctly in a median. Returning None (or
    dropping the row) would delete the best trades.
    """
    try:
        mfe = float(mfe_pct)
        mae_abs = abs(float(mae_pct))
        if not (math.isfinite(mfe) and math.isfinite(mae_abs)):
            return None
        if mae_abs == 0.0:
            return math.inf if mfe > 0 else None
        return mfe / mae_abs
    except (TypeError, ValueError):
        return None


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    """Median that refuses to report a non-finite headline.

    With >=50% degenerate rows the median can itself land on +inf. That is
    a real answer, but it is not renderable and must not reach a tile as
    `Infinity` -- report None and let the caller disclose.
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    med = float(statistics.median(vals))
    return med if math.isfinite(med) else None


def aggregate_exit_quality(
    rows: Sequence[dict],
    min_mfe_pct: float = MIN_MFE_PCT,
) -> dict:
    """Canonical exit-quality aggregate. THE single definition.

    Before phase-82.5 this arithmetic was duplicated at three sites
    (paper_round_trips, paper_trading endpoint, paper_trader), so the
    /performance and /mfe-mae-scatter surfaces could disagree about the
    same trades. All of them now call this.

    Each `row` needs `mfe_pct`, `mae_pct`, `realized_pnl_pct`.
    """
    captures: list[float] = []
    n_undefined = 0
    pnl_sum = 0.0
    mfe_sum_defined = 0.0
    edges: list[float] = []
    edge_undefined = 0
    mfe_sum_all = 0.0
    mae_abs_sum_all = 0.0

    for r in rows or []:
        mfe = float(r.get("mfe_pct") or 0.0)
        mae_abs = abs(float(r.get("mae_pct") or 0.0))
        pnl = float(r.get("realized_pnl_pct") or 0.0)

        cap = compute_capture_ratio(pnl, mfe, min_mfe_pct)
        if cap is None:
            n_undefined += 1
        else:
            captures.append(cap)
            pnl_sum += pnl
            mfe_sum_defined += mfe

        edge = compute_edge_ratio(mfe, mae_abs)
        if edge is None:
            edge_undefined += 1
        else:
            edges.append(edge)
        mfe_sum_all += mfe
        mae_abs_sum_all += mae_abs

    return {
        # headline: a median over the DEFINED subset
        "capture_median": _median_or_none(captures),
        # secondary: avoids the per-trade division entirely
        "capture_ratio_of_sums": (pnl_sum / mfe_sum_defined) if mfe_sum_defined > 0 else None,
        "capture_n_defined": len(captures),
        "capture_n_undefined": n_undefined,
        "edge_median": _median_or_none(edges),
        "edge_ratio_of_sums": (mfe_sum_all / mae_abs_sum_all) if mae_abs_sum_all > 0 else None,
        "edge_n_defined": len(edges),
        "edge_n_undefined": edge_undefined,
        "edge_n_infinite": sum(1 for e in edges if math.isinf(e)),
        "n_points": len(rows or []),
        "min_mfe_pct": float(min_mfe_pct),
    }
