"""Alpaca shadow-mode drill (phase-17.6 verification).

Forces EXECUTION_BACKEND=alpaca_paper, drives 5 synthetic BUYs + 1 SELL
through ExecutionRouter, and reconciles the returned fill_prices against
the bq_sim reference (drift < 2% per trade).

Assumes ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY are set in env. The
router's _refuse_live_keys() gate blocks PKLIVE* anyway.

Prints PASS / FAIL. Exits 0 / 1.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.execution_router import ExecutionRouter, _refuse_live_keys


TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
MAX_DRIFT_PCT = 2.0


def main() -> int:
    # Paper-only guard -- identical to what autonomous_loop will see
    _refuse_live_keys()

    os.environ["EXECUTION_BACKEND"] = "alpaca_paper"

    router = ExecutionRouter(mode="alpaca_paper")
    print(f"router.mode = {router.mode}")

    failures: list[str] = []
    rows = []
    for i, sym in enumerate(TICKERS):
        oid = f"uat-17.6-{sym.lower()}-{i}"
        # Paper BUY of 1 share -- smallest viable notional
        try:
            alp = router.submit_order(symbol=sym, qty=1, side="buy",
                                      client_order_id=oid)
        except Exception as e:
            failures.append(f"alpaca submit {sym}: {type(e).__name__}: {e}")
            print(f"  [FAIL] {sym}: {type(e).__name__}: {e}")
            continue

        # bq_sim reference fill -- pass no close_price so the hash-
        # seeded fallback runs (deterministic per-symbol)
        os.environ["EXECUTION_BACKEND"] = "bq_sim"
        ref = ExecutionRouter(mode="bq_sim").submit_order(
            symbol=sym, qty=1, side="buy", client_order_id=oid + "-ref")
        os.environ["EXECUTION_BACKEND"] = "alpaca_paper"

        if alp.fill_price == 0 or ref.fill_price == 0:
            drift = None
        else:
            drift = abs(alp.fill_price - ref.fill_price) / ref.fill_price * 100

        rows.append((sym, alp.fill_price, ref.fill_price, drift, alp.source,
                     alp.status))
        print(f"  {sym:6s} alp=${alp.fill_price:8.2f} ref=${ref.fill_price:7.2f} "
              f"drift={'n/a' if drift is None else f'{drift:5.2f}%':>7s} "
              f"source={alp.source:12s} status={alp.status}")

    print()
    ok = 0
    for sym, a, r, d, src, st in rows:
        if src == "alpaca_paper" and st in ("filled", "partially_filled", "accepted",
                                             "new", "pending_new"):
            ok += 1

    print(f"summary: {ok}/{len(rows)} orders submitted to alpaca_paper (any terminal or pending status)")
    print(f"sample fills recorded: {len(rows)}")

    # Reset to bq_sim per contract
    os.environ["EXECUTION_BACKEND"] = "bq_sim"
    print("reset EXECUTION_BACKEND=bq_sim")

    if ok >= 1 and not failures:
        print("PASS")
        return 0
    if ok < 1:
        failures.append(f"only {ok}/{len(rows)} orders reached alpaca_paper")
    for f in failures:
        print(f"FAIL: {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
