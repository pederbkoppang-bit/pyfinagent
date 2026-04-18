"""phase-4.8 step 4.8.7 secrets-rotation audit.

Four teeth:
1. Every expected secret has a schedule entry.
2. No secret is overdue (days_since_rotation <= rotation_days).
3. Drill log exists with literal `RTO_MINUTES=` line.
4. RTO value parsed from that line is < 15.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.ops.secrets_rotation_check import (  # noqa: E402
    EXPECTED_SECRETS, SCHEDULE,
)

DRILL_LOG = REPO / "handoff" / "secrets_drill_log.md"
OUT = REPO / "handoff" / "secrets_rotation_audit.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []

    # 1. Schedule coverage.
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))["secrets"]
    missing = EXPECTED_SECRETS - set(schedule.keys())
    if missing:
        reasons.append(f"missing schedule entries: {sorted(missing)}")

    # 2. No overdue.
    today = date.today()
    overdue: list[dict] = []
    for name, cfg in schedule.items():
        try:
            last = date.fromisoformat(cfg["last_rotated_at"])
        except Exception:
            overdue.append({"name": name, "reason": "bad last_rotated_at"})
            continue
        days_since = (today - last).days
        if days_since > int(cfg.get("rotation_days", 0)):
            overdue.append({
                "name": name,
                "days_since": days_since,
                "rotation_days": cfg.get("rotation_days"),
            })
    if overdue:
        reasons.append(f"overdue: {overdue}")

    # 3. Drill log exists with RTO line.
    drill_ok = DRILL_LOG.exists()
    rto_value: int | None = None
    if drill_ok:
        text = DRILL_LOG.read_text(encoding="utf-8")
        m = re.search(r"RTO_MINUTES\s*=\s*(\d+)", text)
        if not m:
            drill_ok = False
            reasons.append("drill log missing 'RTO_MINUTES=' line")
        else:
            rto_value = int(m.group(1))
    else:
        reasons.append("drill log missing at handoff/secrets_drill_log.md")

    # 4. RTO < 15.
    rto_under_15 = rto_value is not None and rto_value < 15
    if not rto_under_15:
        reasons.append(f"RTO_MINUTES not under 15 (got {rto_value})")

    all_ok = not reasons
    verdict = "PASS" if all_ok else "FAIL"
    result = {
        "step": "4.8.7",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "rotation_schedule_configured": not missing,
        "drill_completed": drill_ok and rto_value is not None,
        "rto_under_15min": rto_under_15,
        "missing_from_schedule": sorted(missing),
        "overdue": overdue,
        "rto_minutes": rto_value,
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "rto_minutes": rto_value,
    }))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
