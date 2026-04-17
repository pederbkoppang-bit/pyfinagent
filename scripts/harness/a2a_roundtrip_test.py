"""phase-3.7 step 3.7.4: Data -> Strategy -> Risk task-delegation round-trip.

Exercises backend.agents.task_bus.AsyncTaskBus end-to-end:

1. 3 agents (data / strategy / risk) registered on the bus.
2. 20 round-trips: orchestrator -> Data -> Strategy -> Risk -> back.
3. ONE round-trip deliberately raises a TransientFailure on first
   attempt in Strategy so the retry path is exercised.
4. Latency recorded per round-trip (monotonic). p95 / p50 / max
   computed and written to handoff/a2a_roundtrip.json.
5. Immutable test asserts p95_ms <= 2000.

Usage:
    python scripts/harness/a2a_roundtrip_test.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.agents.task_bus import AsyncTaskBus, TaskEnvelope, TransientFailure  # noqa: E402


_strategy_transient_budget: dict[str, int] = {}


async def data_agent(payload: dict) -> dict:
    await asyncio.sleep(0.001)
    ticker = payload.get("ticker", "AAPL")
    return {
        "ticker": ticker,
        "signal_seeds": [
            {"kind": "momentum_3m", "value": 0.08},
            {"kind": "rsi_14",       "value": 55.2},
            {"kind": "earnings_surprise", "value": 0.12},
            {"kind": "insider_buy",  "value": 1.0},
            {"kind": "patent_breakout", "value": 0.0},
        ],
    }


async def strategy_agent(payload: dict) -> dict:
    await asyncio.sleep(0.002)
    ticker = payload.get("ticker", "AAPL")
    inject = payload.get("inject_transient", False)
    if inject:
        counter_key = f"{ticker}:{payload.get('rid')}"
        seen = _strategy_transient_budget.get(counter_key, 0)
        _strategy_transient_budget[counter_key] = seen + 1
        if seen == 0:
            raise TransientFailure("strategy: simulated transient failure (once)")
    return {
        "ticker": ticker,
        "candidates": [
            {"variant_id": f"{ticker}-v{i+1}", "signal": "BUY",
             "confidence": 0.6 + 0.05 * i, "dsr": 0.92 + 0.01 * i}
            for i in range(5)
        ],
    }


async def risk_agent(payload: dict) -> dict:
    await asyncio.sleep(0.001)
    ticker = payload.get("ticker", "AAPL")
    candidates = payload.get("candidates", [])
    approved = [c for c in candidates
                if c.get("confidence", 0) >= 0.6 and c.get("dsr", 0) >= 0.92]
    return {
        "ticker": ticker,
        "approved": approved,
        "approved_count": len(approved),
        "rejected_count": len(candidates) - len(approved),
    }


async def orchestrate_one(bus: AsyncTaskBus, ticker: str, rid: int,
                           inject: bool) -> dict:
    context_id = f"ctx-{rid}"
    t0 = time.monotonic()

    env_data = TaskEnvelope(context_id=context_id,
                             payload={"ticker": ticker, "rid": rid})
    data_out = await bus.delegate("data", env_data, timeout=0.5)

    env_strat = TaskEnvelope(context_id=context_id,
                              payload={**data_out, "rid": rid,
                                       "inject_transient": inject})
    strat_out = await bus.delegate("strategy", env_strat,
                                     timeout=0.5, max_retries=2)

    env_risk = TaskEnvelope(context_id=context_id,
                             payload={**strat_out, "rid": rid})
    risk_out = await bus.delegate("risk", env_risk, timeout=0.5)

    elapsed_ms = (time.monotonic() - t0) * 1000
    retry_observed = any(
        rec.get("kind") == "transient_retry"
        for rec in env_strat.history
    )
    return {
        "rid": rid,
        "ticker": ticker,
        "elapsed_ms": round(elapsed_ms, 3),
        "approved_count": risk_out.get("approved_count", 0),
        "injected_transient": inject,
        "retry_observed": retry_observed,
    }


async def run(samples: int = 20) -> dict:
    bus = AsyncTaskBus()
    bus.register("data", data_agent)
    bus.register("strategy", strategy_agent)
    bus.register("risk", risk_agent)
    await bus.start()

    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                "META", "TSLA", "AVGO", "ORCL", "AMD"]

    rows: list[dict] = []
    try:
        for i in range(samples):
            ticker = tickers[i % len(tickers)]
            inject = (i == samples // 2)
            row = await orchestrate_one(bus, ticker, rid=i, inject=inject)
            rows.append(row)
    finally:
        await bus.stop()

    latencies = sorted(r["elapsed_ms"] for r in rows)
    n = len(latencies)
    p50 = latencies[n // 2] if n else 0.0
    p95 = latencies[int(0.95 * n)] if n else 0.0
    max_ms = latencies[-1] if n else 0.0
    retry_rows = [r for r in rows if r.get("retry_observed")]
    transient_rows = [r for r in rows if r.get("injected_transient")]
    retry_observed_any = bool(retry_rows)
    transient_retried = all(r.get("retry_observed") for r in transient_rows) \
                         if transient_rows else False

    verdict = "PASS" if (p95 <= 2000 and retry_observed_any and transient_retried
                          and all(r["approved_count"] >= 1 for r in rows)) else "FAIL"

    return {
        "step": "3.7.4",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "max_ms": round(max_ms, 3),
        "retry_observed": retry_observed_any,
        "transient_failure_retried": transient_retried,
        "approved_on_every_hop": all(r["approved_count"] >= 1 for r in rows),
        "sample_rows": rows[:5],
        "verdict": verdict,
    }


def main() -> int:
    result = asyncio.run(run(20))
    out = REPO / "handoff" / "a2a_roundtrip.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "verdict": result["verdict"],
        "p50_ms": result["p50_ms"],
        "p95_ms": result["p95_ms"],
        "retry_observed": result["retry_observed"],
        "transient_failure_retried": result["transient_failure_retried"],
    }))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
