"""phase-4.8 step 4.8.7 Secrets-rotation schedule check.

Inventories every secret the project uses (read from the schedule
sidecar + spot-checked against plist/env files where accessible),
flags any secret whose `days_since_rotation` exceeds its
`rotation_days`, and emits `handoff/secrets_rotation_check.json`.

This script does NOT read secret VALUES (that requires elevated
permissions + is itself a leak risk). It validates NAMES + the
rotation metadata sidecar.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


REPO = Path(__file__).resolve().parents[2]
SCHEDULE = REPO / "scripts" / "ops" / "secrets_rotation_schedule.json"
OUT = REPO / "handoff" / "secrets_rotation_check.json"


# Expected-present secrets: the audit fails if the schedule doesn't
# cover each of these names (e.g. someone added a new env var
# without a rotation entry).
EXPECTED_SECRETS = {
    "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY",
    "AUTH_SECRET", "AUTH_GOOGLE_ID", "AUTH_GOOGLE_SECRET",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "FRED_API_KEY", "ALPHAVANTAGE_API_KEY",
    "API_NINJAS_KEY", "GITHUB_TOKEN",
}


def _plist_vars() -> set[str]:
    """Return the set of EnvironmentVariables names from the pyfinagent
    launchd plists. Absence of the plists is NOT an error -- in many
    installations secrets live in shell .env instead."""
    found: set[str] = set()
    for plist in (
        Path.home() / "Library" / "LaunchAgents" / "com.pyfinagent.backend.plist",
        Path.home() / "Library" / "LaunchAgents" / "com.pyfinagent.frontend.plist",
    ):
        if not plist.exists():
            continue
        try:
            text = plist.read_text(encoding="utf-8")
        except Exception:
            continue
        import re
        for m in re.finditer(r"<key>([A-Z][A-Z0-9_]+)</key>", text):
            name = m.group(1)
            if name in EXPECTED_SECRETS:
                found.add(name)
    return found


def main() -> int:
    if not SCHEDULE.exists():
        logger.error("rotation schedule missing at %s", SCHEDULE)
        return 1
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))["secrets"]
    today = date.today()

    entries: list[dict] = []
    overdue: list[str] = []
    for name, cfg in schedule.items():
        try:
            last = date.fromisoformat(cfg["last_rotated_at"])
            days = (today - last).days
        except Exception:
            days = None
        is_overdue = (days is not None
                       and days > int(cfg.get("rotation_days", 0)))
        entries.append({
            "name": name,
            "source": cfg.get("source"),
            "sensitivity": cfg.get("sensitivity"),
            "rotation_days": cfg.get("rotation_days"),
            "last_rotated_at": cfg.get("last_rotated_at"),
            "days_since_rotation": days,
            "overdue": is_overdue,
        })
        if is_overdue:
            overdue.append(name)

    # Coverage check: every expected secret must have a schedule
    # entry (names only; no values read).
    scheduled_names = {e["name"] for e in entries}
    missing = EXPECTED_SECRETS - scheduled_names
    if missing:
        logger.warning(
            "expected secrets missing from schedule: %s",
            sorted(missing),
        )

    # Cross-check with actual plist variable names, if readable.
    plist_names = _plist_vars()

    verdict = "PASS" if (not missing and not overdue) else "FAIL"

    summary = {
        "step": "4.8.7",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "schedule_path": str(SCHEDULE.relative_to(REPO)),
        "expected_count": len(EXPECTED_SECRETS),
        "scheduled_count": len(scheduled_names),
        "missing_from_schedule": sorted(missing),
        "overdue_secrets": overdue,
        "plist_found_names": sorted(plist_names),
        "entries": entries,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "scheduled": len(scheduled_names),
        "overdue": overdue,
    }))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
