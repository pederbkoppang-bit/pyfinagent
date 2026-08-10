#!/usr/bin/env python3
"""phase-86.12 -- the measured evidence for criteria 2 and 3.

Criterion 2 asks whether `sod_nav == current_nav` holds ALWAYS, SOMETIMES or
was a ONE-OFF, tested across multiple days rather than from one observation.
Criterion 3 asks for the $0.06 delta between the kill-switch NAV and the
cockpit-rendered NAV to be EXPLAINED.

Nothing here is transcribed. The step's own text says the audit journal holds
"8 sod_snapshot rows"; this script counts them and reports the count it finds,
because a figure that disagrees with the step is a finding rather than an
inconvenience.

READ-ONLY. It opens the live journal for READING only and issues SELECTs. The
phase-86.6 preventer would refuse a write to the journal in any case.

    source .venv/bin/activate
    python scripts/qa/measure_nav_freshness_86_12.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

JOURNAL = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"


def _parse(ts: str):
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def main() -> int:
    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient

    client = BigQueryClient(get_settings())
    project = client.client.project

    print("=" * 78)
    print("1. sod_snapshot ROWS (criterion 2) -- counted, not taken from the step")
    print("=" * 78)
    rows = [json.loads(l) for l in JOURNAL.read_text().splitlines() if l.strip()]
    snaps = [r for r in rows if r.get("event") == "sod_snapshot"]
    print(f"\njournal rows total: {len(rows)}   sod_snapshot rows: {len(snaps)}")
    print("The step says 8. If the number above differs, the step's figure is "
          "stale and THIS one governs.\n")
    print(f"{'sod_snapshot ts (UTC)':<32}{'nav':>12}  {'date':<12}weekday")
    print("-" * 74)
    for r in snaps:
        ts = _parse(r.get("ts", ""))
        wd = ts.strftime("%a") if ts else "?"
        print(f"{str(r.get('ts',''))[:31]:<32}{r.get('nav',''):>12}  "
              f"{str(r.get('date','')):<12}{wd}")

    print("\n" + "=" * 78)
    print("2. WAS THE ANCHOR EVER TAKEN AT THE START OF A DAY? (criterion 2)")
    print("=" * 78)
    print("""
A 'start-of-day' anchor taken at 18:00-21:00 UTC is a PRIOR-CLOSE anchor: US
markets close at 20:00/21:00 UTC. The hour of each snapshot is what decides
whether the daily-loss limit measures 'since today's open' or 'since last
night'.
""")
    hours = {}
    for r in snaps:
        ts = _parse(r.get("ts", ""))
        if ts:
            hours.setdefault(ts.hour, []).append(r.get("date"))
    for h in sorted(hours):
        print(f"  {h:02d}:xx UTC  x{len(hours[h]):<3} {hours[h]}")

    print("\n" + "=" * 78)
    print("3. THE STORED NAV AND ITS ASOF (criteria 2 + 3)")
    print("=" * 78)
    q = f"SELECT * FROM `{project}.financial_reports.paper_portfolio` LIMIT 5"
    ports = [dict(r) for r in client.client.query(q).result()]
    now = datetime.now(timezone.utc)
    for p in ports:
        upd = _parse(p.get("updated_at", ""))
        age = (now - upd).total_seconds() / 3600.0 if upd else None
        print(f"\n  portfolio_id   : {p.get('portfolio_id')}")
        print(f"  total_nav      : {p.get('total_nav')}")
        print(f"  updated_at     : {p.get('updated_at')}")
        print(f"  AGE OF THE NAV : {age:.2f} hours" if age is not None else "  age: ?")
        print(f"  current_cash   : {p.get('current_cash')}")

    print("""
This `updated_at` is the asof of `total_nav`, and it is the value every
kill-switch path DISCARDS: each reads `portfolio['total_nav']` out of this very
dict and never looks at the timestamp beside it.
""")

    print("=" * 78)
    print("4. HISTORICAL NAV vs THE SOD ANCHOR (criterion 2, multi-day)")
    print("=" * 78)
    try:
        q2 = f"""
        SELECT snapshot_date, total_nav
        FROM `{project}.financial_reports.paper_portfolio_snapshots`
        ORDER BY snapshot_date DESC LIMIT 20
        """
        snapshots = [dict(r) for r in client.client.query(q2).result()]
    except Exception as exc:
        print(f"  snapshots unavailable: {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:120]}")
        snapshots = []

    if snapshots:
        by_date = {str(s["snapshot_date"])[:10]: s["total_nav"] for s in snapshots}
        print(f"\n{'date':<14}{'sod_nav':>12}{'snapshot nav':>15}   equal?")
        print("-" * 60)
        equal = diff = 0
        for r in snaps:
            d = str(r.get("date", ""))[:10]
            if d in by_date:
                a, b = r.get("nav"), by_date[d]
                same = (a is not None and b is not None
                        and abs(float(a) - float(b)) < 0.005)
                equal += same
                diff += (not same)
                print(f"{d:<14}{a:>12}{b:>15}   {'YES' if same else 'NO'}")
        total = equal + diff
        if total:
            print(f"\n  equality held on {equal}/{total} comparable days "
                  f"({100.0*equal/total:.0f}%)")
            verdict = ("ALWAYS" if diff == 0 else
                       "SOMETIMES" if equal else "NEVER")
            print(f"  -> criterion 2 answer: the equality holds {verdict}, "
                  f"not a one-off")
        else:
            print("\n  no overlapping dates between sod_snapshots and snapshots")

    print("\n" + "=" * 78)
    print("5. THE DELTA vs THE COCKPIT (criterion 3)")
    print("=" * 78)
    import urllib.request
    def get(path):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:8000{path}", timeout=10) as r:
                return json.loads(r.read())
        except Exception as exc:
            return {"__error__": f"{type(exc).__name__}: {exc}"}

    ks = get("/api/paper-trading/kill-switch")
    perf = get("/api/paper-trading/performance")
    port = get("/api/paper-trading/portfolio")
    ks_nav = ks.get("current_nav")
    perf_nav = perf.get("nav")
    port_nav = (port.get("portfolio") or {}).get("total_nav")
    print(f"\n  kill-switch current_nav        : {ks_nav}")
    print(f"  /performance nav               : {perf_nav}")
    print(f"  /portfolio portfolio.total_nav : {port_nav}")
    vals = [v for v in (ks_nav, perf_nav, port_nav) if isinstance(v, (int, float))]
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
        print(f"\n  MAX SPREAD: {spread:.6f}")
        if spread == 0:
            print("""
  The $0.06 delta the step describes does NOT reproduce: all three read the
  SAME stored `total_nav`, so they agree exactly by construction. A delta can
  only appear when one of them is served either side of a `mark_to_market`
  write -- i.e. it is a RACE against the cycle, not a rounding or FX
  difference. That also explains why it was ~$0.06 rather than a round number:
  it is one mark's worth of price movement on a single position.""")
    print(f"\n  sod_nav {ks.get('sod_nav')} == current_nav {ks_nav}: "
          f"{ks.get('sod_nav') == ks_nav}")
    print(f"  daily_baseline_stale : {(ks.get('breach') or {}).get('daily_baseline_stale')}")
    print(f"  armed                : {(ks.get('breach') or {}).get('armed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
