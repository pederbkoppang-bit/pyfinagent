"""
Go-Live Drill: 4.4.2.4 No missed trading days (signal generation reliable)

Verifies:
  1. signals_log table exists in BQ
  2. Paper trading inception date retrieved
  3. NYSE trading calendar loaded for the paper trading window
  4. Every US market open day has at least one signal-generation log entry
  5. Zero-gap gate: 100% coverage required

Queries BQ directly (requires .venv with google-cloud-bigquery and
exchange_calendars). Follows the paper_runtime_test.py pattern.

Re-run: source .venv/bin/activate && python3 scripts/go_live_drills/signal_reliability_test.py
"""

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "backend" / "backtest" / "experiments" / "results"
BQ_PROJECT = "sunny-might-477607-p8"
DATASET = "financial_reports"


def query_bq():
    try:
        from google.cloud import bigquery
    except ImportError:
        print("DRILL SKIP: google-cloud-bigquery not installed (need .venv)")
        sys.exit(2)

    client = bigquery.Client(project=BQ_PROJECT)

    table_exists = True
    try:
        client.get_table(f"{BQ_PROJECT}.{DATASET}.signals_log")
    except Exception:
        table_exists = False

    signal_days = []
    total_publish = 0
    if table_exists:
        q = f"""
            SELECT DATE(signal_date) as sig_date, COUNT(*) as cnt
            FROM `{BQ_PROJECT}.{DATASET}.signals_log`
            WHERE event_kind = 'publish'
            GROUP BY sig_date
            ORDER BY sig_date
        """
        for row in client.query(q).result():
            signal_days.append({"date": str(row["sig_date"]), "count": row["cnt"]})
            total_publish += row["cnt"]

    inception = None
    q2 = f"SELECT inception_date FROM {DATASET}.paper_portfolio LIMIT 1"
    for row in client.query(q2).result():
        inception = row["inception_date"]

    return table_exists, signal_days, total_publish, inception


def get_nyse_trading_days(start_date, end_date):
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar("XNYS")
        sessions = cal.sessions_in_range(
            start_date.isoformat(), end_date.isoformat()
        )
        return [s.date() for s in sessions]
    except ImportError:
        pass

    # stdlib fallback: weekdays minus known 2026 NYSE holidays
    US_HOLIDAYS_2026 = {
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
        date(2026, 4, 3), date(2026, 5, 25), date(2026, 7, 3),
        date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    }
    from datetime import timedelta
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5 and current not in US_HOLIDAYS_2026:
            days.append(current)
        current += timedelta(days=1)
    return days


def run_checks():
    results = []
    now = datetime.now(timezone.utc)
    today = now.date()

    table_exists, signal_days, total_publish, inception_raw = query_bq()

    if not table_exists:
        results.append(("S0", "FAIL", "signals_log table not found in BQ"))
        _print_results(results)
        return False
    results.append(("S0", "PASS", "signals_log table exists in BQ"))

    if inception_raw is None:
        results.append(("S1", "FAIL", "No paper_portfolio row found in BQ"))
        _print_results(results)
        return False
    inception_str = str(inception_raw)
    try:
        inception_dt = datetime.fromisoformat(inception_str)
        if inception_dt.tzinfo is None:
            inception_dt = inception_dt.replace(tzinfo=timezone.utc)
        inception_date = inception_dt.date()
    except Exception as e:
        results.append(("S1", "FAIL", f"Invalid inception_date: {inception_str} ({e})"))
        _print_results(results)
        return False
    results.append(("S1", "PASS",
        f"Paper trading inception: {inception_date}"))

    results.append(("S2", "PASS",
        f"Total publish events in signals_log: {total_publish}"))

    trading_days = get_nyse_trading_days(inception_date, today)
    results.append(("S3", "PASS",
        f"NYSE trading days in window [{inception_date} to {today}]: "
        f"{len(trading_days)}"))

    signal_dates = set()
    for entry in signal_days:
        try:
            signal_dates.add(date.fromisoformat(entry["date"]))
        except (ValueError, TypeError):
            pass

    trading_set = set(trading_days)
    covered = signal_dates & trading_set
    missed = sorted(trading_set - signal_dates)

    coverage_pct = (len(covered) / len(trading_days) * 100) if trading_days else 0
    results.append(("S4", "PASS" if coverage_pct >= 100.0 else "FAIL",
        f"Coverage: {len(covered)}/{len(trading_days)} = {coverage_pct:.1f}% "
        f"(gate: 100%)"))

    if missed:
        first_5 = ", ".join(str(d) for d in missed[:5])
        suffix = f" ... +{len(missed) - 5} more" if len(missed) > 5 else ""
        results.append(("S5", "FAIL",
            f"{len(missed)} missed trading days: {first_5}{suffix}"))
    else:
        results.append(("S5", "PASS", "Zero missed trading days"))

    extra = sorted(signal_dates - trading_set)
    if extra:
        results.append(("S6", "INFO",
            f"{len(extra)} signal days outside trading calendar: "
            + ", ".join(str(d) for d in extra)))
    else:
        results.append(("S6", "PASS", "No signals on non-trading days"))

    evidence = {
        "query_date": today.isoformat(),
        "paper_trading_inception": inception_date.isoformat(),
        "total_publish_events": total_publish,
        "nyse_trading_days_in_window": len(trading_days),
        "signal_generation_days": len(signal_dates),
        "covered_days": len(covered),
        "missed_days": len(missed),
        "missed_dates": [str(d) for d in missed[:20]],
        "coverage_pct": round(coverage_pct, 2),
        "gate_passed": coverage_pct >= 100.0,
        "signals_log_table_exists": table_exists,
    }
    evidence_path = EVIDENCE_DIR / f"signal_generation_evidence_{today.strftime('%Y%m%d')}.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)

    _print_results(results, str(evidence_path))

    passed = sum(1 for _, v, _ in results if v == "PASS")
    total = len(results)
    all_pass = all(v in ("PASS", "INFO") for _, v, _ in results)

    if all_pass:
        print(f"\nDRILL PASS: {passed}/{total}")
    else:
        print(f"\nDRILL FAIL: {passed}/{total}")
    return all_pass


def _print_results(results, evidence_path=None):
    print(f"\n{'=' * 60}")
    print("  4.4.2.4 No Missed Trading Days Drill")
    print(f"{'=' * 60}")
    for sid, verdict, detail in results:
        marker = "+" if verdict == "PASS" else ("i" if verdict == "INFO" else "X")
        print(f"  [{marker}] {sid}: {detail}")
    if evidence_path:
        print(f"  Evidence: {evidence_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
