"""phase-4.9 step 4.9.4 gauntlet-regimes audit.

Seven teeth:
1. import_ok: REGIMES is a tuple of exactly 7 RegimeWindow entries.
2. ids_unique_and_snake_case: no duplicate ids; each matches
   `^[a-z][a-z0-9_]*$`.
3. dates_valid: every start/end parses as ISO date; start <= end.
4. dates_chronologically_sorted: REGIMES ordered by start date
   ascending.
5. immutability: attempting to set a field on a REGIMES entry
   raises `FrozenInstanceError` (proves dataclass is frozen, not
   just trust the decorator).
6. universe_fields_populated: every entry has non-empty
   asset_classes, region, note, primary_source_url; the URL
   starts with `https://`.
7. intraday_flag_consistent: exactly one entry has
   `intraday_only=True` (flash_crash_2010), and that entry has
   `start == end`. All others are `intraday_only=False`.

Plus tooth 8 (immutable masterplan verification mirror):
   `'start' in r and 'end' in r` for every entry.

Exit 1 on failure when `--check` is passed; always writes
`handoff/gauntlet_regimes_audit.json`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.backtest.gauntlet.regimes import REGIMES, RegimeWindow  # noqa: E402

OUT = REPO / "handoff" / "gauntlet_regimes_audit.json"

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    # 1. import_ok
    checks["import_ok"] = (
        isinstance(REGIMES, tuple)
        and len(REGIMES) == 7
        and all(isinstance(r, RegimeWindow) for r in REGIMES)
    )
    if not checks["import_ok"]:
        reasons.append(
            f"REGIMES must be tuple of 7 RegimeWindow; got "
            f"{type(REGIMES).__name__} len={len(REGIMES)}"
        )

    # 2. ids_unique_and_snake_case
    ids = [r.id for r in REGIMES]
    unique = len(set(ids)) == len(ids)
    snake_ok = all(ID_PATTERN.match(i) for i in ids)
    checks["ids_unique_and_snake_case"] = unique and snake_ok
    if not unique:
        reasons.append(f"duplicate ids: {ids}")
    if not snake_ok:
        bad = [i for i in ids if not ID_PATTERN.match(i)]
        reasons.append(f"non-snake-case ids: {bad}")

    # 3. dates_valid
    dates_valid = True
    date_pairs: list[tuple[date, date]] = []
    for r in REGIMES:
        try:
            s = date.fromisoformat(r.start)
            e = date.fromisoformat(r.end)
            if s > e:
                dates_valid = False
                reasons.append(f"{r.id}: start {r.start} > end {r.end}")
            date_pairs.append((s, e))
        except Exception as exc:
            dates_valid = False
            reasons.append(f"{r.id}: bad date parse: {exc}")
    checks["dates_valid"] = dates_valid

    # 4. dates_chronologically_sorted
    starts = [p[0] for p in date_pairs]
    checks["dates_chronologically_sorted"] = starts == sorted(starts)
    if not checks["dates_chronologically_sorted"]:
        reasons.append(f"REGIMES not sorted by start: {[str(s) for s in starts]}")

    # 5. immutability (mutation test -- must raise)
    mutation_raised = False
    try:
        REGIMES[0].end = "2030-01-01"  # type: ignore[misc]
    except FrozenInstanceError:
        mutation_raised = True
    except Exception as exc:
        reasons.append(f"mutation raised wrong exception: {exc!r}")
    checks["immutability"] = mutation_raised
    if not mutation_raised:
        reasons.append("dataclass not frozen -- mutation did not raise")

    # 6. universe_fields_populated
    universe_ok = True
    for r in REGIMES:
        if not r.asset_classes:
            universe_ok = False
            reasons.append(f"{r.id}: empty asset_classes")
        if not r.region:
            universe_ok = False
            reasons.append(f"{r.id}: empty region")
        if not r.note or len(r.note) < 40:
            universe_ok = False
            reasons.append(f"{r.id}: note too short (<40 chars)")
        if not r.primary_source_url.startswith("https://"):
            universe_ok = False
            reasons.append(f"{r.id}: source URL not https")
    checks["universe_fields_populated"] = universe_ok

    # 7. intraday_flag_consistent
    intraday = [r for r in REGIMES if r.intraday_only]
    intraday_ok = (
        len(intraday) == 1
        and intraday[0].id == "flash_crash_2010"
        and intraday[0].start == intraday[0].end
    )
    for r in REGIMES:
        if not r.intraday_only and r.id == "flash_crash_2010":
            intraday_ok = False
    checks["intraday_flag_consistent"] = intraday_ok
    if not intraday_ok:
        reasons.append(
            f"intraday_only flag wrong: {[r.id for r in intraday]}"
        )

    # 8. masterplan verification mirror
    mp_ok = all(("start" in r and "end" in r) for r in REGIMES)
    checks["masterplan_verification_passes"] = mp_ok
    if not mp_ok:
        reasons.append("masterplan 'in' check failed on at least one regime")

    all_ok = all(checks.values()) and not reasons
    verdict = "PASS" if all_ok else "FAIL"

    result = {
        "step": "4.9.4",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        **checks,
        "regime_count": len(REGIMES),
        "regime_ids": ids,
        "date_ranges": [
            {"id": r.id, "start": r.start, "end": r.end} for r in REGIMES
        ],
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        **{k: v for k, v in checks.items()},
    }, indent=2))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
