#!/usr/bin/env python3
"""phase-86.34 N1 -- does the TZ fixture actually simulate what it claims?

`backend/tests/test_phase_86_24_clock_dependence.py` forces `TZ=Pacific/Midway`
to reproduce the class "the LOCAL calendar date differs from the UTC calendar
date". Its comment asserts that Midway "puts the LOCAL date one day behind UTC,
which is exactly the 00:00-02:00 CEST window in which the two macro tests used
to fail".

TWO THINGS ARE WRONG WITH THAT, and the second is the serious one.

1. DIRECTION. At 00:30 and 01:30 CEST the local date is one day AHEAD of UTC,
   not behind. Midway is behind. The fixture is the MIRROR of the window it
   names. The operative property -- local date != UTC date -- still holds, so
   this alone changes no result.

2. COVERAGE, and this is the finding. **A fixed TZ offset does not guarantee a
   date mismatch at all; it depends on the UTC time of day.** Midway is UTC-11,
   so local and UTC share a date whenever the UTC hour is >= 11. Swept over the
   24 UTC hours of a day, Midway produces the intended mismatch in only ELEVEN
   of them. For the other thirteen the fixture simulates NOTHING and every
   assertion under it passes vacuously.

   That is the defect stated plainly: **a test written to remove clock
   dependence is itself clock-dependent**, and it is silent about it. Whether it
   reproduces anything depends on the hour the suite happens to run. It was
   measured as working on 2026-08-10 at ~09:xx UTC -- inside the window -- and
   the identical command at 16:25 UTC reports no mismatch at all.

THE REMEDY THIS SCRIPT ARGUES FOR is not "pick a better timezone". It is that
the test must ASSERT ITS OWN PRECONDITION: if the chosen TZ does not actually
make the local date differ from the UTC date at the moment of the run, the test
must FAIL LOUDLY rather than pass having exercised nothing. A fixture that can
silently do nothing is the same vacuity class as a guard that cannot fail.
(Pacific/Kiritimati, UTC+14, covers the complementary 14/24 hours -- so a pair
covers the day, but only the precondition assertion makes the coverage HONEST.)

    $ python scripts/qa/measure_tz_fixture_coverage_86_34.py
"""
from __future__ import annotations

import argparse
import datetime
import sys
from zoneinfo import ZoneInfo

#: The zone the fixture uses today, plus its complement.
ZONES = ["Pacific/Midway", "Pacific/Kiritimati"]


def mismatch_hours(tz_name: str, day: datetime.date) -> list[int]:
    """UTC hours on `day` at which local date != UTC date under `tz_name`."""
    zone = ZoneInfo(tz_name)
    out = []
    for h in range(24):
        u = datetime.datetime(day.year, day.month, day.day, h, 0,
                              tzinfo=datetime.timezone.utc)
        if u.astimezone(zone).date() != u.date():
            out.append(h)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-08-10",
                    help="UTC date to sweep (YYYY-MM-DD)")
    ns = ap.parse_args()
    day = datetime.date.fromisoformat(ns.date)

    print(f"TZ fixture coverage sweep over the 24 UTC hours of {day}\n")
    print(f"  {'zone':22s} {'mismatch hours':>15s}   window")
    results = {}
    for z in ZONES:
        hrs = mismatch_hours(z, day)
        results[z] = hrs
        window = f"{hrs[0]:02d}:00-{hrs[-1]:02d}:59 UTC" if hrs else "(never)"
        print(f"  {z:22s} {len(hrs):>10d}/24     {window}")

    fixture = ZONES[0]
    hrs = results[fixture]
    print(f"\n  THE FIXTURE IN USE IS {fixture}.")
    print(f"  It simulates the intended condition in {len(hrs)}/24 UTC hours.")
    print(f"  For the other {24 - len(hrs)} it simulates NOTHING and the assertions")
    print("  under it pass vacuously, without saying so.")

    union = sorted(set().union(*(set(v) for v in results.values())))
    print(f"\n  The two zones together cover {len(union)}/24 hours"
          f"{' (the whole day)' if len(union) == 24 else ''}.")

    now = datetime.datetime.now(datetime.timezone.utc)
    live = now.astimezone(ZoneInfo(fixture)).date() != now.date()
    print(f"\n  RIGHT NOW ({now:%H:%M} UTC) the fixture "
          f"{'DOES' if live else 'does NOT'} produce a date mismatch.")
    print("  Run this again a few hours from now and the answer changes -- which is")
    print("  the whole point: the result of the clock-dependence suite depends on the")
    print("  clock. The remedy is for the test to ASSERT this precondition and fail")
    print("  loudly when it does not hold, not to pick a luckier timezone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
