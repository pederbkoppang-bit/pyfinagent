"""
paper_go_live_gate -- Deterministic booleans for the live-capital promotion gate.

Blocks phase-4 step 4.4 (Go-Live). The rule is: "Promote to live" is disabled
unless ALL five booleans are green. No vibes, no partial credit.

Boolean definitions (carried from handoff/current/phase-4.5-contract.md):
  1. trades_ge_100              -- at least 100 closed round trips exist
  2. psr_ge_95_sustained_30d    -- PSR >= 0.95 over the most recent 30 days
  3. dsr_ge_95                  -- DSR >= 0.95 right now
  4. sr_gap_le_30pct            -- |paper_sharpe - backtest_sharpe| / |backtest|
                                   <= 0.30 (reality-gap sanity)
  5. max_dd_within_tolerance    -- realized max drawdown is not worse than the
                                   backtest max drawdown + 5pp buffer; default
                                   tolerance = 20% absolute cap

References:
  - Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", for thresholds.
  - AFML Ch. 13 (Lopez de Prado) for sustained-period PSR interpretation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from backend.services.paper_metrics_v2 import compute_metrics_v2
from backend.services.paper_round_trips import pair_round_trips
from backend.services.perf_metrics import compute_sharpe_gap
from backend.services.reconciliation import compute_reconciliation

logger = logging.getLogger(__name__)

TRADES_THRESHOLD = 100
PSR_SUSTAINED_DAYS = 30
PSR_THRESHOLD = 0.95
DSR_THRESHOLD = 0.95
SR_GAP_THRESHOLD = 0.30  # 30 percent relative gap vs backtest Sharpe
MAX_DD_ABS_TOLERANCE = 20.0  # percent absolute cap on max drawdown


def _snapshot_max_dd_pct(snapshots: list[dict]) -> float:
    """Max peak-to-trough drawdown across NAV snapshots (positive magnitude %).

    phase-47.4: get_paper_snapshots returns rows newest-first (ORDER BY
    snapshot_date DESC). Drawdown is order-dependent -- walking a NAV series
    backwards reads portfolio GROWTH as a crash (observed 60.08% phantom DD vs
    the correct 5.31%). Sort chronologically before computing peak-to-trough.
    """
    if not snapshots:
        return 0.0
    snapshots = sorted(snapshots, key=lambda s: s.get("snapshot_date") or "")
    navs = [float(s.get("total_nav") or 0.0) for s in snapshots]
    navs = [n for n in navs if n > 0]
    if len(navs) < 2:
        return 0.0
    peak = navs[0]
    worst = 0.0
    for n in navs:
        peak = max(peak, n)
        dd = (peak - n) / peak * 100.0
        worst = max(worst, dd)
    return worst


def compute_gate(bq: Any) -> dict:
    """
    Pull the minimum set of data needed and return the gate response.
    Structure:
      {
        "booleans": { trades_ge_100, psr_ge_95_sustained_30d, dsr_ge_95,
                      sr_gap_le_30pct, max_dd_within_tolerance },
        "promote_eligible": <AND of all booleans>,
        "details": { supporting numbers for each boolean },
        "thresholds": { ... static thresholds for UI display ... },
        "computed_at": ISO8601 timestamp
      }
    """
    trades = bq.get_paper_trades(limit=5000) or []
    round_trips = pair_round_trips(trades)
    n_round_trips = len(round_trips)

    metrics = compute_metrics_v2(bq)
    psr = metrics.get("psr")
    dsr = metrics.get("dsr")
    rolling_sharpe = metrics.get("rolling_sharpe")

    snapshots = bq.get_paper_snapshots(limit=PSR_SUSTAINED_DAYS) or []
    realized_max_dd = _snapshot_max_dd_pct(snapshots)

    # phase-25.A6: explicit Sharpe-gap reconciliation. The legacy proxy
    # used `latest_divergence_pct / 100.0` as a stand-in -- closes phase-24.6
    # F-3 (NAV divergence is a dollar measure; Sharpe gap is risk-adjusted-
    # return measure; they are different quantities). The new helper has a
    # 3-tier fallback (optimizer_best -> shadow curve -> divergence proxy)
    # so existing operator behavior is preserved when explicit backtest
    # Sharpe is unavailable.
    sharpe_gap = compute_sharpe_gap(bq)
    sr_gap_le = sharpe_gap.get("gap_within_threshold")

    # Keep the legacy divergence number in `details` as a sibling signal --
    # operators still find it informative even when the explicit Sharpe gap
    # is the authoritative measurement.
    recon = compute_reconciliation(bq)
    recon_summary = recon.get("summary", {})
    latest_divergence_pct = float(recon_summary.get("latest_divergence_pct") or 0.0)

    booleans = {
        "trades_ge_100": bool(n_round_trips >= TRADES_THRESHOLD),
        "psr_ge_95_sustained_30d": bool(
            psr is not None and psr >= PSR_THRESHOLD and metrics.get("n_obs", 0) >= PSR_SUSTAINED_DAYS
        ),
        "dsr_ge_95": bool(dsr is not None and dsr >= DSR_THRESHOLD),
        # phase-25.A6: None coerces to False, keeping the gate red on no-data.
        "sr_gap_le_30pct": bool(sr_gap_le) if sr_gap_le is not None else False,
        "max_dd_within_tolerance": bool(realized_max_dd <= MAX_DD_ABS_TOLERANCE),
    }
    promote_eligible = all(booleans.values())

    return {
        "booleans": booleans,
        "promote_eligible": promote_eligible,
        "details": {
            "n_round_trips": n_round_trips,
            "psr": psr,
            "dsr": dsr,
            "rolling_sharpe": rolling_sharpe,
            "n_obs": metrics.get("n_obs", 0),
            "latest_reconciliation_divergence_pct": round(latest_divergence_pct, 4),
            "realized_max_dd_pct": round(realized_max_dd, 4),
            # phase-25.A6: explicit Sharpe-gap diagnostics for the UI.
            "live_sharpe": sharpe_gap.get("live_sharpe"),
            "backtest_sharpe": sharpe_gap.get("backtest_sharpe"),
            "sharpe_gap_rel": sharpe_gap.get("gap_rel"),
            "sharpe_gap_source": sharpe_gap.get("source"),
            "sharpe_gap_proxy_fallback": sharpe_gap.get("proxy_fallback"),
            "sharpe_gap_note": sharpe_gap.get("note"),
        },
        "thresholds": {
            "trades": TRADES_THRESHOLD,
            "psr_sustained_days": PSR_SUSTAINED_DAYS,
            "psr": PSR_THRESHOLD,
            "dsr": DSR_THRESHOLD,
            "sr_gap": SR_GAP_THRESHOLD,
            "max_dd_pct": MAX_DD_ABS_TOLERANCE,
        },
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
