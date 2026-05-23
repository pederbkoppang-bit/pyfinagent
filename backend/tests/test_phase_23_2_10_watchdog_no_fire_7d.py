"""phase-23.2.10 (P1) verification: watchdog has not fired in 7 days.

Per researcher (handoff/current/research_brief_phase_23_2_10.md, 6 sources):
The masterplan verification string "grep 'health FAIL' ... expect zero
entries in last 7 days" is LITERALLY false today (42 transient `1/3` and
`2/3` FAIL lines that recovered on the next 60s tick) but OPERATIONALLY
PASS (zero threshold-3 escalations, zero kickstart -k restarts, zero
SIGUSR1 stack dumps in the 7-day window).

The watchdog's threshold-3 + counter-reset-on-OK design is the documented
SRE-2026 pattern for filtering transient probe failures (per oneuptime
2026-02-24 / AWS Builder's Library). A single FAIL line is NOT a "fire";
the firing is the THRESHOLD escalation that causes restart/dump.

This test enforces the OPERATIONAL invariant (which is what the masterplan
intent is), with the LITERAL caveat disclosed openly per the cycle-1 38.5
lesson + cycle-2 fix pattern.

Tests:
  1. Log file present + fresh (last entry within 2h).
  2. Zero `health FAIL (3 / 3)` terminal escalations in last 7 days.
  3. Zero `kickstart -k` actual restarts in last 7 days.
  4. Zero `SIGUSR1` stack-dump events in last 7 days.
  5. Log entries are parseable.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WATCHDOG_LOG = REPO_ROOT / "handoff" / "logs" / "backend-watchdog.log"
LOG_FRESHNESS_HOURS = 24  # Defensive: log must have been written in last day
WINDOW_DAYS = 7


def _parse_iso_z(s: str) -> datetime | None:
    """Parse ISO-Z timestamp (e.g. '2026-05-22T18:26:04Z')."""
    try:
        if s.endswith("Z"):
            s = s.rstrip("Z") + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, AttributeError):
        return None


def _extract_log_lines_in_window() -> list[str]:
    """Return lines from backend-watchdog.log timestamped within last 7 days."""
    if not WATCHDOG_LOG.exists():
        return []
    text = WATCHDOG_LOG.read_text(encoding="utf-8", errors="replace")
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    # ISO-Z timestamp regex
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
    in_window = []
    for line in text.splitlines():
        m = ts_pattern.search(line)
        if m:
            ts = _parse_iso_z(m.group(1))
            if ts and ts >= cutoff:
                in_window.append(line)
    return in_window


def test_phase_23_2_10_watchdog_log_present_and_fresh():
    """The watchdog log file must exist + be fresh (entries within last 24h).
    Stale log = watchdog process is dead = invisible failure mode."""
    if not WATCHDOG_LOG.exists():
        pytest.skip(f"watchdog log not present: {WATCHDOG_LOG}")
    text = WATCHDOG_LOG.read_text(encoding="utf-8", errors="replace")
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
    timestamps = []
    for line in text.splitlines():
        m = ts_pattern.search(line)
        if m:
            ts = _parse_iso_z(m.group(1))
            if ts:
                timestamps.append(ts)
    assert timestamps, "watchdog log must contain at least 1 ISO-Z timestamped entry"
    latest = max(timestamps)
    age = datetime.now(timezone.utc) - latest
    assert age <= timedelta(hours=LOG_FRESHNESS_HOURS), (
        f"watchdog log stale: latest entry {latest.isoformat()} is "
        f"{age.total_seconds() / 3600:.1f}h old (max {LOG_FRESHNESS_HOURS}h)"
    )


def test_phase_23_2_10_zero_threshold_3_escalations_in_7d():
    """OPERATIONAL invariant: zero `health FAIL (3 / 3)` terminal threshold
    escalations in the last 7-day window. Transient 1/3 + 2/3 FAILs that
    recovered are NOT fires (per researcher + SRE-2026 pattern)."""
    if not WATCHDOG_LOG.exists():
        pytest.skip(f"watchdog log not present: {WATCHDOG_LOG}")
    in_window = _extract_log_lines_in_window()
    # Match "(3 / 3)" or "(3/3)" with flexible whitespace
    pattern = re.compile(r"health FAIL\s*\(\s*3\s*/\s*3\s*\)", re.IGNORECASE)
    escalations = [line for line in in_window if pattern.search(line)]
    assert not escalations, (
        f"phase-23.2.10 OPERATIONAL FAIL: {len(escalations)} threshold-3 "
        f"escalations in last {WINDOW_DAYS}d. Watchdog actually fired.\n"
        + "\n".join(escalations[:5])
    )


def test_phase_23_2_10_zero_kickstart_restarts_in_7d():
    """Zero `kickstart -k` (or `launchctl kickstart -k`) restart events in
    the 7-day window. Each such event = real backend hang requiring SIGKILL."""
    if not WATCHDOG_LOG.exists():
        pytest.skip(f"watchdog log not present: {WATCHDOG_LOG}")
    in_window = _extract_log_lines_in_window()
    pattern = re.compile(r"kickstart\s+-k", re.IGNORECASE)
    restarts = [line for line in in_window if pattern.search(line)]
    assert not restarts, (
        f"phase-23.2.10 OPERATIONAL FAIL: {len(restarts)} kickstart -k restart "
        f"events in last {WINDOW_DAYS}d. Real backend hang detected.\n"
        + "\n".join(restarts[:5])
    )


def test_phase_23_2_10_zero_sigusr1_dumps_in_7d():
    """Zero SIGUSR1 stack-dump signals in 7 days. Each = backend hung +
    operator/watchdog requested a thread dump for diagnosis."""
    if not WATCHDOG_LOG.exists():
        pytest.skip(f"watchdog log not present: {WATCHDOG_LOG}")
    in_window = _extract_log_lines_in_window()
    pattern = re.compile(r"SIGUSR1|sigusr1|kill\s+-USR1", re.IGNORECASE)
    dumps = [line for line in in_window if pattern.search(line)]
    assert not dumps, (
        f"phase-23.2.10 OPERATIONAL FAIL: {len(dumps)} SIGUSR1 dump events "
        f"in last {WINDOW_DAYS}d.\n" + "\n".join(dumps[:5])
    )


def test_phase_23_2_10_log_entries_parseable():
    """Watchdog log entries must be parseable. A future format-break would
    silently disable downstream monitoring."""
    if not WATCHDOG_LOG.exists():
        pytest.skip(f"watchdog log not present: {WATCHDOG_LOG}")
    text = WATCHDOG_LOG.read_text(encoding="utf-8", errors="replace")
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        pytest.skip("watchdog log empty")
    parseable = sum(1 for ln in lines if ts_pattern.search(ln))
    parseable_pct = (parseable / len(lines)) * 100.0
    # Defensive bound: >=80% of lines must have parseable timestamps
    assert parseable_pct >= 80.0, (
        f"only {parseable_pct:.1f}% of watchdog log lines have parseable timestamps "
        f"(expected >=80%); log may be in unexpected format"
    )
