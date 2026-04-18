"""phase-4.8 step 4.8.10 compliance CLI: wash-sale + funding + margin.

`--test` runs an end-to-end sanity check with KNOWN fixtures
covering:
- positive wash pair (loss @t=0, rebuy @t+15 -> blocked)
- negative non-wash pair (rebuy @t+61 -> allowed)
- boundary case (rebuy @t+30 -> blocked; @t+31 -> allowed)
- unsettled-cash same-day BUY blocked
- one-day-later BUY (settled) allowed
- margin-deficit oversized BUY blocked
- under-margin BUY allowed

Exits 0 when every assertion holds; emits
`handoff/wash_sale_filter_test.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.services.funding_guard import (  # noqa: E402
    realtime_margin_guard, t1_funding_guard,
)
from backend.services.wash_sale_filter import (  # noqa: E402
    WashSaleLedger, filter_candidates,
)

OUT = REPO / "handoff" / "wash_sale_filter_test.json"


def _run_tests() -> dict:
    results: dict = {}
    reasons: list[str] = []

    # --- Wash-sale ledger tests ---
    loss_day = date(2026, 4, 1)
    ledger = WashSaleLedger()
    ledger.record_loss(
        symbol="AAPL",
        sell_date=loss_day,
        disallowed_loss_usd=250.00,
        trade_id="t-0001",
    )

    # Positive: rebuy 15 days later -> blocked.
    is_ws_pos, _ = ledger.is_wash_sale("AAPL", loss_day + timedelta(days=15))
    results["positive_15d"] = is_ws_pos
    if not is_ws_pos:
        reasons.append("rebuy at t+15 not flagged as wash")

    # Boundary +30: still inside the window (<=30).
    is_ws_30, _ = ledger.is_wash_sale("AAPL", loss_day + timedelta(days=30))
    results["boundary_+30d"] = is_ws_30
    if not is_ws_30:
        reasons.append("rebuy at t+30 not flagged (boundary inclusive)")

    # Boundary +31: outside window -> allowed.
    is_ws_31, _ = ledger.is_wash_sale("AAPL", loss_day + timedelta(days=31))
    results["boundary_+31d"] = not is_ws_31
    if is_ws_31:
        reasons.append("rebuy at t+31 wrongly flagged (outside 30)")

    # Negative: rebuy at t+61 -> allowed + ledger auto-prunes.
    is_ws_61, _ = ledger.is_wash_sale("AAPL", loss_day + timedelta(days=61))
    results["negative_61d"] = not is_ws_61
    if is_ws_61:
        reasons.append("rebuy at t+61 wrongly flagged")

    # Different ticker -> never a wash.
    is_ws_other, _ = ledger.is_wash_sale("MSFT", loss_day + timedelta(days=5))
    results["different_ticker"] = not is_ws_other
    if is_ws_other:
        reasons.append("different-ticker buy flagged as wash")

    # filter_candidates partitioning.
    # Fresh ledger to avoid pruning from previous calls.
    ledger2 = WashSaleLedger()
    ledger2.record_loss(
        symbol="NVDA", sell_date=loss_day, disallowed_loss_usd=400.0,
    )
    candidates = [
        {"symbol": "NVDA", "trade_date": loss_day + timedelta(days=10),
         "notional_usd": 5000.0},    # blocked
        {"symbol": "NVDA", "trade_date": loss_day + timedelta(days=45),
         "notional_usd": 5000.0},    # allowed (outside window)
        {"symbol": "GOOGL", "trade_date": loss_day + timedelta(days=5),
         "notional_usd": 3000.0},    # allowed (different ticker)
    ]
    allowed, blocked = filter_candidates(candidates, ledger2)
    results["partition_allowed_count"] = len(allowed)
    results["partition_blocked_count"] = len(blocked)
    if len(blocked) != 1 or blocked[0]["symbol"] != "NVDA":
        reasons.append(
            f"filter partitioned wrong: allowed={allowed}, blocked={blocked}"
        )

    # --- T+1 funding-guard tests ---
    # Same-day buy funded only by pending_proceeds -> blocked.
    ok, reason = t1_funding_guard(
        settled_cash=1000.0,
        pending_proceeds=5000.0,
        buy_notional=3000.0,
    )
    results["t1_same_day_unsettled_blocked"] = (not ok) and reason == "UNSETTLED_CASH_INSUFFICIENT"
    if ok:
        reasons.append("t1 guard allowed unsettled-funded buy")

    # Same-day buy fully covered by settled_cash -> allowed.
    ok2, reason2 = t1_funding_guard(
        settled_cash=5000.0,
        pending_proceeds=0.0,
        buy_notional=3000.0,
    )
    results["t1_settled_allowed"] = ok2 and reason2 == "OK"
    if not ok2:
        reasons.append("t1 guard blocked fully-settled buy")

    # --- Margin guard tests ---
    # gross_long + buy > available -> blocked.
    ok3, reason3 = realtime_margin_guard(
        gross_long=80_000.0,
        available_margin=100_000.0,
        buy_notional=25_000.0,
    )
    results["margin_deficit_blocked"] = (not ok3) and reason3 == "MARGIN_DEFICIT"
    if ok3:
        reasons.append("margin guard allowed deficit buy")

    # gross_long + buy under available -> allowed.
    ok4, reason4 = realtime_margin_guard(
        gross_long=50_000.0,
        available_margin=100_000.0,
        buy_notional=25_000.0,
    )
    results["margin_under_allowed"] = ok4 and reason4 == "OK"
    if not ok4:
        reasons.append("margin guard blocked under-margin buy")

    # Deficit threshold leaves headroom: 5% buffer means 95k cap.
    ok5, reason5 = realtime_margin_guard(
        gross_long=70_000.0,
        available_margin=100_000.0,
        buy_notional=27_000.0,
        deficit_threshold_pct=0.05,   # effective cap 95k
    )
    results["margin_threshold_blocks"] = (not ok5)
    if ok5:
        reasons.append("margin threshold_pct did not tighten as expected")

    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "checks": results,
        "reasons": reasons,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    if not args.test:
        print("usage: wash_sale_filter.py --test")
        return 2

    summary = _run_tests()
    summary["step"] = "4.8.10"
    summary["ran_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": summary["verdict"],
        "failed_count": len(summary["reasons"]),
    }))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
