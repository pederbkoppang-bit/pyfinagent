"""Risk Agent MCP server (phase-3.7 step 3.7.3).

Wraps existing in-process risk primitives behind a FastMCP surface so
the Data / Strategy / Risk MAS can veto trading candidates without
touching shared state directly.

Tools exposed:
- ping              : liveness probe
- kill_switch       : wrapper around backend.services.kill_switch
- portfolio_cvar    : 95th-percentile portfolio expected shortfall (stub)
- factor_exposure   : Fama-French 3-factor snapshot (stub)
- pbo_check         : CSCV PBO score for a trial PnL matrix; veto if > 0.5
- evaluate_candidate: gate chain (kill_switch -> pbo -> projected_dd),
                       returns {vetoed: bool, reason: str, ...}
"""
from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

logger = logging.getLogger(__name__)


# Default hard caps (can be overridden via settings in future step).
DEFAULT_PBO_VETO_THRESHOLD = 0.5
DEFAULT_MAX_DD_CAP_PCT = 10.0
DEFAULT_DAILY_LOSS_LIMIT_PCT = 4.0


def _projected_max_dd_pct(sigma_ann_pct: float, sharpe: float) -> float:
    """E[MaxDD] ~= sigma / (2 * Sharpe) under the standard diffusion
    approximation (Grossman & Zhou 1993; Magdon-Ismail et al. 2004).
    Returns annualized expected max drawdown as a percent.
    """
    if sharpe <= 0:
        return 100.0
    return float(sigma_ann_pct / (2.0 * sharpe))


def create_risk_server():
    """Factory for the Risk Agent MCP server. Matches the pattern of
    create_data_server / create_backtest_server / create_signals_server.
    """
    try:
        from fastmcp import FastMCP
    except ImportError:
        logger.error("FastMCP not installed. Install with: pip install fastmcp")
        raise

    mcp = FastMCP(name="pyfinagent-risk")

    # ---- ping -------------------------------------------------------

    @mcp.tool
    def ping() -> dict:
        """Liveness probe."""
        return {"ok": True, "server": "pyfinagent-risk", "ts": time.time()}

    # ---- kill_switch -------------------------------------------------

    @mcp.tool
    def kill_switch(current_nav: float | None = None,
                    daily_loss_limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT_PCT,
                    trailing_dd_limit_pct: float = DEFAULT_MAX_DD_CAP_PCT) -> dict:
        """Read the in-process kill-switch state + evaluate an NAV
        against the daily/trailing-DD caps. Does NOT flip state; that
        is the operator's call via the /api/paper-trading routes.
        """
        try:
            from backend.services.kill_switch import get_state, evaluate_breach
        except Exception as e:
            return {"ok": False, "reason": f"kill_switch_unavailable:{type(e).__name__}:{e}"}
        snap = get_state().snapshot()
        breach: dict[str, Any] = {}
        if current_nav is not None:
            try:
                breach = evaluate_breach(
                    current_nav=current_nav,
                    daily_loss_limit_pct=daily_loss_limit_pct,
                    trailing_dd_limit_pct=trailing_dd_limit_pct,
                )
            except Exception as e:
                breach = {"error": f"{type(e).__name__}:{e}"}
        return {
            "ok": True,
            "state": snap,
            "breach": breach,
            "is_paused": bool(snap.get("paused", False)),
        }

    # ---- portfolio_cvar (stub; real impl lands in phase-4.8) --------

    @mcp.tool
    def portfolio_cvar(confidence: float = 0.95,
                        window_days: int = 60) -> dict:
        """95th-percentile portfolio Conditional Value-at-Risk.

        Stub placeholder. Real implementation lives in phase-4.8 step
        4.8.2 (portfolio CVaR + factor-exposure gate). Returning a
        placeholder here keeps the MCP surface stable for the MAS
        layer.
        """
        return {
            "ok": True,
            "confidence": confidence,
            "window_days": window_days,
            "cvar_pct_nav": None,
            "status": "stub_placeholder",
            "todo": "phase-4.8.2 real CVaR implementation",
        }

    # ---- factor_exposure (stub; real impl lands in phase-4.8) -------

    @mcp.tool
    def factor_exposure(factor_model: str = "FF3") -> dict:
        """Fama-French 3 factor exposure snapshot.

        Stub placeholder for phase-4.8 step 4.8.2.
        """
        return {
            "ok": True,
            "factor_model": factor_model,
            "loadings": None,
            "status": "stub_placeholder",
            "todo": "phase-4.8.2 real FF regression",
        }

    # ---- pbo_check --------------------------------------------------

    @mcp.tool
    def pbo_check(pnl_matrix: list[list[float]],
                   threshold: float = DEFAULT_PBO_VETO_THRESHOLD,
                   S: int = 16) -> dict:
        """Compute CSCV PBO on a T x N PnL matrix and issue a veto when
        PBO > threshold. Canonical reference: Bailey, Borwein, Lopez de
        Prado, Zhu (2016) -- SSRN 2326253.
        """
        try:
            from backend.backtest.analytics import compute_pbo
            pbo = float(compute_pbo(pnl_matrix, S=S))
        except Exception as e:
            return {
                "ok": False,
                "vetoed": False,
                "reason": f"pbo_compute_error:{type(e).__name__}:{e}",
            }
        vetoed = pbo > threshold
        return {
            "ok": True,
            "pbo": pbo,
            "threshold": threshold,
            "vetoed": vetoed,
            "reason": "pbo_exceeds_threshold" if vetoed else "pbo_within_bounds",
            "isError": vetoed,  # MCP-native veto signal
        }

    # ---- evaluate_candidate (composite gate) ------------------------

    @mcp.tool
    def evaluate_candidate(candidate: dict,
                            pbo_threshold: float = DEFAULT_PBO_VETO_THRESHOLD,
                            max_dd_cap_pct: float = DEFAULT_MAX_DD_CAP_PCT) -> dict:
        """Composite veto chain: kill_switch -> pbo_check -> projected-
        DD cap. Returns {vetoed: bool, reason: str, gates: {...}}.

        `candidate` should carry at least:
        - pbo (float, optional) OR pnl_matrix (list of lists)
        - sigma_ann_pct (float)
        - sharpe (float)
        Any missing field is skipped (partial gates allowed).
        """
        gates: dict[str, Any] = {}
        reason = None

        # Gate 1: kill-switch hot?
        ks = kill_switch()
        gates["kill_switch"] = ks
        if ks.get("is_paused"):
            return {"vetoed": True, "reason": "kill_switch_paused",
                    "gates": gates, "isError": True}

        # Gate 2: PBO.
        pbo_val = candidate.get("pbo")
        if pbo_val is None and candidate.get("pnl_matrix"):
            pbo_result = pbo_check(pnl_matrix=candidate["pnl_matrix"],
                                    threshold=pbo_threshold)
            gates["pbo"] = pbo_result
            if pbo_result.get("vetoed"):
                reason = "pbo_exceeds_threshold"
        elif pbo_val is not None:
            vetoed = bool(pbo_val > pbo_threshold)
            gates["pbo"] = {"pbo": float(pbo_val), "threshold": pbo_threshold,
                             "vetoed": vetoed}
            if vetoed:
                reason = "pbo_exceeds_threshold"

        # Gate 3: projected DD cap.
        sigma = candidate.get("sigma_ann_pct")
        sharpe = candidate.get("sharpe")
        projected_dd: float | None = None
        if sigma is not None and sharpe is not None:
            projected_dd = _projected_max_dd_pct(float(sigma), float(sharpe))
            vetoed_dd = projected_dd > max_dd_cap_pct
            gates["projected_dd"] = {
                "projected_dd_pct": round(projected_dd, 3),
                "cap_pct": max_dd_cap_pct,
                "vetoed": vetoed_dd,
            }
            if vetoed_dd and reason is None:
                reason = "projected_dd_over_cap"

        vetoed = reason is not None
        return {
            "vetoed": vetoed,
            "reason": reason or "passed_all_gates",
            "gates": gates,
            "projected_dd_pct": projected_dd,
            "isError": vetoed,
        }

    logger.info("Risk server created with 6 tools (ping, kill_switch, "
                 "portfolio_cvar, factor_exposure, pbo_check, evaluate_candidate)")
    return mcp


if __name__ == "__main__":
    create_risk_server().run()
