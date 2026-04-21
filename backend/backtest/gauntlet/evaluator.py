"""phase-4.9.6 pass-criteria evaluator for Gauntlet reports.

Takes a Gauntlet report (shape produced by
`scripts/risk/gauntlet.py`) and returns a PASS/FAIL verdict with
per-check details. Four hard gates:

1. `drawdown_ratio_enforced` -- for every non-skipped regime,
   `drawdown / bt_drawdown <= DRAWDOWN_RATIO_CAP` (2.0x, practitioner
   consensus live/backtest ratio; see Bailey & Lopez de Prado PBO
   paper + AFML ch. 15).
2. `forced_exits_zero_required` -- zero forced exits per regime
   (kill-switch or margin-call breaches are strict zeros).
3. `mc_p99_checked` -- Monte Carlo p99 drawdown / backtest drawdown
   <= `DRAWDOWN_RATIO_CAP`. Using p99 (not mean) guards against
   tail-scenario overfitting.
4. `immutable_limit_breach_blocks` -- `monte_carlo.breaches == 0`;
   any non-zero breach count blocks promotion regardless of the
   drawdown ratios.

`evaluate(report)` returns a dict with `overall_pass`, `regime_checks`,
`mc_check`, `reasons`, `drawdown_ratio_cap`. Pure function, no I/O,
no mutation of input.
"""
from __future__ import annotations

from typing import Any


DRAWDOWN_RATIO_CAP: float = 2.0


def _regime_check(regime: dict) -> dict:
    dd = float(regime.get("drawdown", 0.0) or 0.0)
    bt = float(regime.get("bt_drawdown", 0.0) or 0.0)
    forced = int(regime.get("forced_exits", 0) or 0)
    if bt <= 0.0:
        ratio = 0.0 if dd <= 0.0 else float("inf")
    else:
        ratio = dd / bt
    forced_ok = forced == 0
    ratio_ok = ratio <= DRAWDOWN_RATIO_CAP
    reason = ""
    if not forced_ok:
        reason = f"forced_exits={forced} (must be 0)"
    elif not ratio_ok:
        reason = f"drawdown_ratio={ratio:.3f} exceeds cap {DRAWDOWN_RATIO_CAP}"
    return {
        "id": regime.get("id", "?"),
        "drawdown": dd,
        "bt_drawdown": bt,
        "drawdown_ratio": ratio,
        "forced_exits": forced,
        "forced_exits_ok": forced_ok,
        "pass": forced_ok and ratio_ok,
        "reason": reason,
    }


def _mc_check(mc: dict) -> dict:
    p99 = float(mc.get("p99_drawdown", 0.0) or 0.0)
    bt = float(mc.get("bt_drawdown", 0.0) or 0.0)
    breaches = int(mc.get("breaches", 0) or 0)
    if bt <= 0.0:
        ratio = 0.0 if p99 <= 0.0 else float("inf")
    else:
        ratio = p99 / bt
    breaches_ok = breaches == 0
    ratio_ok = ratio <= DRAWDOWN_RATIO_CAP
    reason = ""
    if not breaches_ok:
        reason = f"immutable_limit_breaches={breaches} (must be 0)"
    elif not ratio_ok:
        reason = f"mc_p99_ratio={ratio:.3f} exceeds cap {DRAWDOWN_RATIO_CAP}"
    return {
        "p99_drawdown": p99,
        "bt_drawdown": bt,
        "ratio": ratio,
        "breaches": breaches,
        "breaches_ok": breaches_ok,
        "pass": breaches_ok and ratio_ok,
        "reason": reason,
    }


def evaluate(report: dict[str, Any]) -> dict[str, Any]:
    per_regime = report.get("per_regime", []) or []
    mc = report.get("monte_carlo", {}) or {}
    regime_checks = [_regime_check(r) for r in per_regime if not r.get("skipped")]
    mc_res = _mc_check(mc)
    reasons: list[str] = []
    for rc in regime_checks:
        if not rc["pass"]:
            reasons.append(f"regime[{rc['id']}]: {rc['reason']}")
    if not mc_res["pass"]:
        reasons.append(f"monte_carlo: {mc_res['reason']}")
    overall_pass = all(rc["pass"] for rc in regime_checks) and mc_res["pass"]
    return {
        "overall_pass": bool(overall_pass),
        "regime_checks": regime_checks,
        "mc_check": mc_res,
        "reasons": reasons,
        "drawdown_ratio_cap": DRAWDOWN_RATIO_CAP,
    }


__all__ = ["evaluate", "DRAWDOWN_RATIO_CAP"]
