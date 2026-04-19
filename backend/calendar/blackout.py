"""phase-6.6 FOMC blackout window computation.

Rule (ITC Markets 2026-04-19 + Fed official policy):

    Blackout begins at midnight ET on the second Saturday before the first
    day of the meeting and ends at midnight ET the day after the meeting
    ends.

"Second Saturday before" = the Saturday in the 8-14 day range prior to
meeting day-1. The computation is:
    - subtract 1 day from meeting start to get "meeting day-1"
    - walk backwards until we hit a Saturday; that gives us the FIRST
      Saturday before the meeting
    - subtract another 7 days to get the SECOND Saturday before
    - this lands 8-14 days before meeting day-1

Edge cases:
 - Monday meeting (meeting day-1 = Sunday): first Saturday is day-1 minus
   1 day; second Saturday is minus 8 days total.
 - Tue-Wed meeting (meeting day-1 = Monday): first Saturday is day-1 minus
   2 days; second Saturday is minus 9 days total.
 - Wed-Thu meeting: first Saturday is day-1 minus 3 days; second is
   minus 10 days total. <-- this is the canonical "10 days before" phrasing
   that gives the rule its common name.

Times are returned as timezone-aware datetimes in the input's timezone.
The caller is expected to pass meeting-start / meeting-end in ET or UTC --
the rule says "midnight ET" but for BQ storage we keep the caller's tz.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Tuple


def _midnight(dt: datetime) -> datetime:
    """Truncate to midnight in the dt's own tz."""
    return datetime.combine(dt.date(), time(0, 0), tzinfo=dt.tzinfo)


def _second_saturday_before(reference: datetime) -> datetime:
    """Return midnight of the second Saturday strictly before `reference`.

    `reference` is treated as meeting day-1; this returns the Saturday in
    the 8-14 day range before reference.date().
    """
    d = reference.date()
    # Walk back one day at a time until Saturday. weekday(): Monday=0, Sunday=6,
    # Saturday=5.
    first_sat = d - timedelta(days=1)
    while first_sat.weekday() != 5:
        first_sat -= timedelta(days=1)
    second_sat = first_sat - timedelta(days=7)
    return datetime.combine(second_sat, time(0, 0), tzinfo=reference.tzinfo)


def compute_fomc_blackout(
    meeting_start: datetime, meeting_end: datetime
) -> Tuple[datetime, datetime]:
    """Return (blackout_start, blackout_end) for a single FOMC meeting pair.

    `meeting_start` = first-day start (use the publicly-scheduled time; only
    the calendar date matters for this computation).
    `meeting_end` = last-day start (same-day for single-day meetings).

    Returns timezone-aware datetimes. The caller is responsible for
    tz-converting if they need ET specifically.
    """
    # meeting day-1: the calendar day BEFORE meeting_start's date
    day_before_meeting = meeting_start - timedelta(days=1)
    blackout_start = _second_saturday_before(day_before_meeting)
    # blackout ends midnight the day AFTER meeting end
    day_after_end = meeting_end + timedelta(days=1)
    blackout_end = _midnight(day_after_end)
    return blackout_start, blackout_end


__all__ = ["compute_fomc_blackout"]
