"""phase-3.7 step 3.7.8: Virtual-fund reality-gap calibration.

1-week shadow of BQ sim vs Alpaca paper. 5 days x 20 symbols x 10
orders/day (1000 order pairs). Half "small" (qty below 5% ADV) ->
single fill expected. Half "large" (qty >= 5% ADV) -> partial fills
expected in BQ sim (2+ child fills; notional conserved).

Asserts:
1. shadow_week_complete           (1000 pairs, no swallowed exceptions)
2. fill_price_drift_le_1pct       (p95 abs drift <= 0.01)
3. fill_latency_drift_le_200ms    (p95 abs latency drift <= 200)
4. partial_fill_modeled_in_sim    (large orders: >=2 children, sum==qty)

Emits handoff/virtual_fund_parity.json; exits 0 on PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.services.execution_router import (  # noqa: E402
    ADV_PARTIAL_FILL_THRESHOLD, ExecutionRouter,
)


SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
           "AVGO", "ORCL", "AMD", "INTC", "IBM", "CRM", "ADBE", "QCOM",
           "CSCO", "NFLX", "PYPL", "SHOP", "UBER"]


def _stable_adv(symbol: str) -> float:
    """Deterministic ADV per symbol so tests are reproducible.

    Real ADV for liquid names is tens of millions. We use 1M-10M range
    so the 5% threshold lands at 50k-500k shares -- large but plausible
    institutional orders.
    """
    h = int(hashlib.sha1(f"adv:{symbol}".encode()).hexdigest()[:8], 16)
    return 1_000_000.0 + (h % 9_000_000)


def _order_qty(day: int, symbol: str, i: int, large: bool) -> float:
    """Deterministic qty. Large orders = 8% of ADV; small = 0.5%."""
    adv = _stable_adv(symbol)
    return round(adv * (0.08 if large else 0.005), 2)


def _notional_conserved(parent_qty: float, children: list) -> bool:
    s = sum(c["qty"] for c in children)
    return abs(s - parent_qty) < 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--orders-per-day", type=int, default=10,
                     help="orders per symbol per day")
    args = ap.parse_args()

    router = ExecutionRouter(mode="shadow")

    pairs = []
    large_pairs = []
    exceptions = 0

    for day in range(args.days):
        for symbol in SYMBOLS:
            adv = _stable_adv(symbol)
            for i in range(args.orders_per_day):
                large = (i % 2 == 0)
                qty = _order_qty(day, symbol, i, large)
                coid = f"vf-{day:02d}-{symbol}-{i:02d}"
                try:
                    bq, alp = router.shadow_submit(
                        symbol, qty, "buy", coid, adv=adv)
                except Exception as e:
                    exceptions += 1
                    print(f"ERROR on {coid}: {e}")
                    continue
                pairs.append((bq, alp, large, adv))
                if large:
                    large_pairs.append((bq, alp, adv))

    total = len(pairs)
    shadow_week_complete = (total == args.days * len(SYMBOLS) * args.orders_per_day
                              and exceptions == 0)

    # fill-price drift
    drifts = [abs(alp.fill_price - bq.fill_price) / bq.fill_price
              for (bq, alp, _, _) in pairs if bq.fill_price > 0]
    p95_drift = (statistics.quantiles(drifts, n=100)[94]
                   if len(drifts) >= 20 else max(drifts, default=0.0))
    fill_price_drift_le_1pct = p95_drift <= 0.01

    # latency drift (ms)
    lats = [abs(alp.latency_ms - bq.latency_ms) for (bq, alp, _, _) in pairs]
    p95_lat = (statistics.quantiles(lats, n=100)[94]
                 if len(lats) >= 20 else max(lats, default=0.0))
    fill_latency_drift_le_200ms = p95_lat <= 200.0

    # partial fill modeling (only large orders; must have >=2 children,
    # notional conserved, child prices == parent adj_price)
    partial_checks = []
    for (bq, _alp, adv) in large_pairs:
        has_multi = len(bq.child_fills) >= 2
        sum_ok = _notional_conserved(bq.qty, bq.child_fills)
        price_ok = all(abs(c["fill_price"] - bq.fill_price) < 1e-6
                        for c in bq.child_fills)
        partial_checks.append(has_multi and sum_ok and price_ok)
    partial_fill_modeled_in_sim = (len(partial_checks) > 0
                                      and all(partial_checks))

    all_pass = (shadow_week_complete
                  and fill_price_drift_le_1pct
                  and fill_latency_drift_le_200ms
                  and partial_fill_modeled_in_sim)

    result = {
        "step": "3.7.8",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "symbols": len(SYMBOLS),
        "orders_per_day": args.orders_per_day,
        "orders_placed": total,
        "exceptions": exceptions,
        "adv_threshold": ADV_PARTIAL_FILL_THRESHOLD,
        "fill_price_drift_pct": round(p95_drift, 6),
        "fill_latency_drift_ms": round(p95_lat, 3),
        "large_orders": len(large_pairs),
        "partial_fill_checks_passed": sum(partial_checks),
        "shadow_week_complete": shadow_week_complete,
        "fill_price_drift_le_1pct": fill_price_drift_le_1pct,
        "fill_latency_drift_le_200ms": fill_latency_drift_le_200ms,
        "partial_fill_modeled_in_sim": partial_fill_modeled_in_sim,
        "verdict": "PASS" if all_pass else "FAIL",
        "sample_large_order": ({
            "symbol": large_pairs[0][0].symbol,
            "qty": large_pairs[0][0].qty,
            "adv": large_pairs[0][2],
            "child_fills": large_pairs[0][0].child_fills,
            "parent_fill_price": large_pairs[0][0].fill_price,
        } if large_pairs else None),
    }
    out = REPO / "handoff" / "virtual_fund_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "verdict": result["verdict"],
        "orders_placed": total,
        "fill_price_drift_pct": result["fill_price_drift_pct"],
        "fill_latency_drift_ms": result["fill_latency_drift_ms"],
        "partial_fill_modeled_in_sim": result["partial_fill_modeled_in_sim"],
    }))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
