"""
Go-Live Drill: 4.4.2.4 No missed trading days (signal generation reliable)

Verifies:
  1. BQ evidence snapshot exists with signal generation data
  2. signals_log table status (preferred source)
  3. Signal generation days vs NYSE trading calendar
  4. Zero-gap gate: every US market open day has a signal entry
  5. Coverage percentage

Re-run: python3 scripts/go_live_drills/signal_reliability_test.py
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "backend" / "backtest" / "experiments" / "results"

US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # MLK Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 7, 3),   # Independence Day (observed)
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
}


def load_evidence():
    candidates = sorted(
        EVIDENCE_DIR.glob("signal_generation_evidence_*.json"), reverse=True
    )
    if not candidates:
        return None
    with open(candidates[0], encoding="utf-8") as f:
        return json.load(f)


def nyse_trading_days(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in US_MARKET_HOLIDAYS_2026:
            days.append(current)
        current += timedelta(days=1)
    return days


def run_checks():
    results = []
    evidence = load_evidence()

    if evidence is None:
        results.append(("S0", "FAIL", "No signal_generation_evidence_*.json found"))
        print("DRILL FAIL: 0/7 (no evidence file)")
        return False
    results.append(
        ("S0", "PASS", f"Evidence loaded, query_date={evidence['query_date']}")
    )

    signals_log_exists = evidence.get("signals_log_table_exists", False)
    if signals_log_exists:
        results.append(("S1", "PASS", "signals_log table exists in BQ"))
    else:
        note = evidence.get("signals_log_table_note", "not found")
        results.append(("S1", "FAIL", f"signals_log table missing: {note}"))

    inception_str = evidence.get("paper_trading_inception", "")
    if not inception_str:
        results.append(("S2", "FAIL", "No paper_trading_inception in evidence"))
        print_results(results)
        return False

    inception = date.fromisoformat(inception_str)
    query_date = date.fromisoformat(evidence["query_date"])
    trading_days = nyse_trading_days(inception, query_date)
    results.append(
        (
            "S2",
            "PASS",
            f"NYSE trading days in window [{inception} to {query_date}]: {len(trading_days)}",
        )
    )

    signal_days_raw = evidence.get("signal_generation_days", [])
    signal_dates = set()
    for entry in signal_days_raw:
        signal_dates.add(date.fromisoformat(entry["date"]))
    results.append(
        ("S3", "PASS", f"Signal generation days in BQ: {len(signal_dates)}")
    )

    trading_set = set(trading_days)
    covered = signal_dates & trading_set
    missed = sorted(trading_set - signal_dates)

    coverage_pct = (len(covered) / len(trading_days) * 100) if trading_days else 0
    results.append(
        (
            "S4",
            "PASS" if coverage_pct >= 100.0 else "FAIL",
            f"Coverage: {len(covered)}/{len(trading_days)} = {coverage_pct:.1f}% "
            f"(gate: 100%)",
        )
    )

    if missed:
        first_5 = ", ".join(str(d) for d in missed[:5])
        suffix = f" ... +{len(missed) - 5} more" if len(missed) > 5 else ""
        results.append(
            (
                "S5",
                "FAIL",
                f"{len(missed)} missed trading days: {first_5}{suffix}",
            )
        )
    else:
        results.append(("S5", "PASS", "Zero missed trading days"))

    extra = sorted(signal_dates - trading_set)
    if extra:
        results.append(
            (
                "S6",
                "INFO",
                f"{len(extra)} signal days outside trading calendar: "
                + ", ".join(str(d) for d in extra),
            )
        )
    else:
        results.append(("S6", "PASS", "No signals on non-trading days"))

    print_results(results)

    passed = sum(1 for _, v, _ in results if v == "PASS")
    total = len(results)
    all_pass = all(v in ("PASS", "INFO") for _, v, _ in results)

    if all_pass:
        print(f"\nDRILL PASS: {passed}/{total}")
        return True
    else:
        print(f"\nDRILL FAIL: {passed}/{total}")
        return False


def print_results(results):
    print(f"\n{'=' * 60}")
    print("  4.4.2.4 No Missed Trading Days Drill")
    print(f"{'=' * 60}")
    for sid, verdict, detail in results:
        marker = "+" if verdict == "PASS" else ("i" if verdict == "INFO" else "X")
        print(f"  [{marker}] {sid}: {detail}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
