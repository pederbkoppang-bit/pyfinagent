"""phase-4.8 step 4.8.2 portfolio-risk audit.

Exercises `daily_check()` with THREE fixtures:
- benign: seeded default returns, both gates pass.
- cvar-trip: a 252-day return series with a fat negative tail that
  pushes CVaR_97.5 above the 2% limit.
- beta-trip: returns designed to regress to market-beta > 1.5.

Asserts:
- Benign run: `new_positions_allowed=True`, no blocking reasons.
- CVaR trip: `new_positions_allowed=False` AND "cvar_exceeded" in
  blocking_reasons.
- Beta trip: `new_positions_allowed=False` AND "beta_cap_exceeded"
  in blocking_reasons.

Emits handoff/portfolio_risk_audit.json. `--check` exits 1 on any
FAIL so the gate can't regress to a constant "true".
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.services.portfolio_risk import (  # noqa: E402
    BETA_CAP, CVAR_LIMIT_PCT, daily_check,
)

OUT = REPO / "handoff" / "portfolio_risk_audit.json"


def _fat_tail_returns(n: int = 252) -> np.ndarray:
    """Mostly-normal returns with 5% of days at -5% (fat left tail).
    Forces empirical CVaR_97.5 above the 2% limit."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0005, 0.010, n)
    # Worst 5% of days: large losses
    k = int(0.05 * n)
    crash_idxs = rng.choice(n, size=k, replace=False)
    r[crash_idxs] = -0.05 + rng.normal(0.0, 0.002, k)
    return r


def _high_beta_returns(n: int = 252) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return port + factor series with deliberately large mkt beta."""
    rng = np.random.default_rng(11)
    mkt = rng.normal(0.0005, 0.010, n)
    smb = rng.normal(0.0, 0.006, n)
    hml = rng.normal(0.0, 0.005, n)
    eps = rng.normal(0.0, 0.003, n)
    port = 2.2 * mkt + 0.1 * smb - 0.05 * hml + eps + 0.0003
    factors = {"Mkt-Rf": mkt, "SMB": smb, "HML": hml}
    return port, factors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    results: dict[str, object] = {}
    ok_all = True

    # 1. benign run (uses seeded defaults inside daily_check)
    benign = daily_check(seed="benign-run")
    benign_ok = (benign["gate"]["new_positions_allowed"] is True
                 and not benign["gate"]["blocking_reasons"])
    results["benign"] = {
        "cvar_97_5": benign["cvar_97_5"],
        "market_beta": benign["ff3"]["market_beta"],
        "gate": benign["gate"],
        "ok": benign_ok,
    }
    ok_all &= benign_ok

    # 2. cvar trip
    port = _fat_tail_returns()
    cvar_trip = daily_check(portfolio_returns=port)
    cvar_triggered = (cvar_trip["cvar_97_5"]["value"] > CVAR_LIMIT_PCT
                      and not cvar_trip["gate"]["new_positions_allowed"]
                      and any("cvar_exceeded" in r
                               for r in cvar_trip["gate"]["blocking_reasons"]))
    results["cvar_trip"] = {
        "cvar_value": cvar_trip["cvar_97_5"]["value"],
        "gate": cvar_trip["gate"],
        "ok": cvar_triggered,
    }
    ok_all &= cvar_triggered

    # 3. beta trip (supply factors matching the constructed port)
    port_b, factors_b = _high_beta_returns()
    beta_trip = daily_check(portfolio_returns=port_b, factor_returns=factors_b)
    beta_triggered = (abs(beta_trip["ff3"]["market_beta"]) > BETA_CAP
                      and not beta_trip["gate"]["new_positions_allowed"]
                      and any("beta_cap_exceeded" in r
                               for r in beta_trip["gate"]["blocking_reasons"]))
    results["beta_trip"] = {
        "market_beta": beta_trip["ff3"]["market_beta"],
        "gate": beta_trip["gate"],
        "ok": beta_triggered,
    }
    ok_all &= beta_triggered

    verdict = "PASS" if ok_all else "FAIL"
    summary = {
        "step": "4.8.2",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "cvar_limit_pct": CVAR_LIMIT_PCT,
        "beta_cap": BETA_CAP,
        "fixtures": results,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=float) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "benign_ok": benign_ok,
        "cvar_trip_ok": cvar_triggered,
        "beta_trip_ok": beta_triggered,
    }))
    if args.check and verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
