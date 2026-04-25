"""phase-10.7.4 cron-budget YAML validator (CLI).

Used by the masterplan immutable verification command:

    python scripts/meta/validate_cron_budget.py .claude/cron_budget.yaml

8 checks (per phase-10.7.4 contract):
  1. YAML loads without error
  2. Top-level required keys: version, total_slots, slots
  3. Per-slot required keys: slot_id, job_name, priority, cadence, surface
  4. priority in {reserved, high, medium, low}
  5. No duplicate job_name
  6. total_daily_token_budget (if present) is positive int
  7. min_tokens_per_fire <= max_tokens_per_fire per slot (where both set)
  8. total_slots matches len(slots)

Exit codes:
  0 = all checks pass
  1 = any check fails (details on stderr)
  2 = YAML or filesystem error (file missing, parse error)

CLI:
  --quiet  exit-code-only mode (no stdout)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ALLOWED_PRIORITIES = frozenset({"reserved", "high", "medium", "low"})

REQUIRED_TOP_LEVEL = ("version", "total_slots", "slots")
REQUIRED_PER_SLOT = ("slot_id", "job_name", "priority", "cadence", "surface")


def _check(label: str, ok: bool, detail: str = "", *, quiet: bool) -> bool:
    if not quiet:
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {label}"
        if detail:
            line += f" -- {detail}"
        print(line)
    if not ok and detail:
        print(f"validate_cron_budget: {label}: {detail}", file=sys.stderr)
    return ok


def validate(yaml_path: str | Path, *, quiet: bool = False) -> int:
    p = Path(yaml_path)
    if not p.exists():
        print(f"validate_cron_budget: file not found: {p}", file=sys.stderr)
        return 2

    try:
        cfg: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        print(f"validate_cron_budget: YAML parse error: {e}", file=sys.stderr)
        return 2

    if not isinstance(cfg, dict):
        print("validate_cron_budget: YAML root is not a mapping", file=sys.stderr)
        return 1

    all_ok = True

    # Check 1: YAML loaded (we got here)
    all_ok &= _check("YAML loads", True, quiet=quiet)

    # Check 2: top-level keys
    missing_top = [k for k in REQUIRED_TOP_LEVEL if k not in cfg]
    all_ok &= _check(
        "top-level required keys present",
        not missing_top,
        f"missing: {missing_top}" if missing_top else "",
        quiet=quiet,
    )

    slots = cfg.get("slots") or []
    if not isinstance(slots, list):
        all_ok &= _check(
            "slots is a list",
            False,
            f"got type={type(slots).__name__}",
            quiet=quiet,
        )
        return 0 if all_ok else 1

    # Check 3: per-slot required keys
    per_slot_missing: list[str] = []
    for i, s in enumerate(slots):
        if not isinstance(s, dict):
            per_slot_missing.append(f"slot[{i}] is not a mapping")
            continue
        miss = [k for k in REQUIRED_PER_SLOT if k not in s]
        if miss:
            per_slot_missing.append(
                f"slot[{i}] (id={s.get('slot_id')}): missing {miss}"
            )
    all_ok &= _check(
        "per-slot required keys present",
        not per_slot_missing,
        "; ".join(per_slot_missing),
        quiet=quiet,
    )

    # Check 4: priority validity
    bad_pri: list[str] = []
    for s in slots:
        if not isinstance(s, dict):
            continue
        pri = s.get("priority")
        if pri not in ALLOWED_PRIORITIES:
            bad_pri.append(f"slot {s.get('job_name', '<?>')}: priority={pri!r}")
    all_ok &= _check(
        f"priorities in {sorted(ALLOWED_PRIORITIES)}",
        not bad_pri,
        "; ".join(bad_pri),
        quiet=quiet,
    )

    # Check 5: no duplicate job_name
    names = [s.get("job_name") for s in slots if isinstance(s, dict)]
    seen: set[str] = set()
    dups: list[str] = []
    for n in names:
        if n in seen:
            dups.append(str(n))
        seen.add(n)
    all_ok &= _check(
        "no duplicate job_name",
        not dups,
        f"duplicates: {dups}" if dups else "",
        quiet=quiet,
    )

    # Check 6: total_daily_token_budget (optional but if present must be positive int)
    tdtb = cfg.get("total_daily_token_budget")
    if tdtb is None:
        all_ok &= _check(
            "total_daily_token_budget (optional)",
            True,
            "absent (allocator will use DEFAULT_TOTAL_DAILY_TOKEN_BUDGET)",
            quiet=quiet,
        )
    else:
        ok = isinstance(tdtb, int) and tdtb > 0
        all_ok &= _check(
            "total_daily_token_budget is positive int",
            ok,
            f"got {tdtb!r}" if not ok else f"= {tdtb}",
            quiet=quiet,
        )

    # Check 7: per-slot min <= max
    bad_clamp: list[str] = []
    for s in slots:
        if not isinstance(s, dict):
            continue
        lo = s.get("min_tokens_per_fire")
        hi = s.get("max_tokens_per_fire")
        if lo is not None and hi is not None and lo > hi:
            bad_clamp.append(
                f"slot {s.get('job_name', '<?>')}: min={lo} > max={hi}"
            )
    all_ok &= _check(
        "min_tokens_per_fire <= max_tokens_per_fire",
        not bad_clamp,
        "; ".join(bad_clamp),
        quiet=quiet,
    )

    # Check 8: total_slots matches len(slots)
    expected = cfg.get("total_slots")
    actual = len(slots)
    ok = isinstance(expected, int) and expected == actual
    all_ok &= _check(
        "total_slots matches len(slots)",
        ok,
        f"declared={expected}, actual={actual}" if not ok else f"= {actual}",
        quiet=quiet,
    )

    return 0 if all_ok else 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0] if __doc__ else "validate cron budget"
    )
    parser.add_argument("path", help="path to cron_budget.yaml")
    parser.add_argument("--quiet", action="store_true", help="no stdout (exit code only)")
    args = parser.parse_args(argv)

    rc = validate(args.path, quiet=args.quiet)
    if not args.quiet:
        if rc == 0:
            print(f"validate_cron_budget: PASS ({args.path})")
        elif rc == 1:
            print(f"validate_cron_budget: FAIL ({args.path}) -- see errors above", file=sys.stderr)
        else:
            print(f"validate_cron_budget: ERROR rc={rc} ({args.path})", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
