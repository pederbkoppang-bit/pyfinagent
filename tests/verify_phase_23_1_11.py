"""phase-23.1.11 immutable verification — referenced by handoff/current/contract.md.

Asserts:
1. _persist_lite_analysis is importable + async-callable
2. Helper signature: (analysis: dict, bq) -> None
3. When called with a synthetic lite-shape analysis dict and a stub BQ
   client, it invokes bq.save_report exactly once with the expected fields
4. BQ exceptions do NOT propagate (cycle survives BQ outage)
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from unittest.mock import MagicMock


def main() -> int:
    from backend.services.autonomous_loop import _persist_lite_analysis

    assert callable(_persist_lite_analysis), "_persist_lite_analysis not callable"
    assert asyncio.iscoroutinefunction(_persist_lite_analysis), \
        "_persist_lite_analysis must be async"

    sig = inspect.signature(_persist_lite_analysis)
    params = list(sig.parameters)
    assert params == ["analysis", "bq"], f"unexpected signature: {params}"

    lite = {
        "ticker": "COHR",
        "recommendation": "BUY",
        "final_score": 7,
        "risk_assessment": {"reason": "lite reason"},
        "price_at_analysis": 320.91,
        "analysis_date": "2026-04-27T18:00:00+00:00",
        "total_cost_usd": 0.01,
        "full_report": {
            "source": "claude-sonnet-4-6",
            "analysis": {"reason": "x", "confidence": 72},
            "market_data": {"name": "Coherent Corp.", "sector": "IT"},
        },
    }
    bq = MagicMock()
    asyncio.run(_persist_lite_analysis(lite, bq))

    bq.save_report.assert_called_once()
    kwargs = bq.save_report.call_args.kwargs
    assert kwargs["ticker"] == "COHR"
    assert kwargs["recommendation"] == "BUY"
    assert kwargs["final_score"] == 7.0
    assert kwargs["summary"] == "lite reason"
    assert kwargs["company_name"] == "Coherent Corp."

    # BQ outage — must not propagate
    bq2 = MagicMock()
    bq2.save_report.side_effect = RuntimeError("BQ down")
    try:
        asyncio.run(_persist_lite_analysis(lite, bq2))
    except Exception as e:
        print(f"FAIL: exception propagated: {e}")
        return 1

    print("ok _persist_lite_analysis async + signature + save_report invocation + graceful BQ failure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
