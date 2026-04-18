"""phase-4.9 step 4.9.0 immutable-limits audit.

Six teeth:
1. limits_file_exists + parses clean.
2. schema_validates: load() returns a RiskLimits instance.
3. six_limits_present: all 6 required fields, exact match to the
   contract set.
4. frozen: setting any field on the instance raises.
5. extra='forbid': a YAML with an extra rogue key fails to load.
6. load() is cached: two calls return the same object (id).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.governance import limits_schema as ls  # noqa: E402

OUT = REPO / "handoff" / "immutable_limits_audit.json"

EXPECTED_FIELDS = {
    "max_position_notional_pct",
    "max_portfolio_leverage",
    "max_daily_loss_pct",
    "max_trailing_dd_pct",
    "max_gross_exposure_pct",
    "max_sector_weight_pct",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []
    checks: dict[str, bool] = {}

    # 1. file exists
    checks["limits_file_exists"] = ls.LIMITS_FILE.exists()
    if not checks["limits_file_exists"]:
        reasons.append(f"limits file missing at {ls.LIMITS_FILE}")

    # 2. schema validates
    limits = None
    try:
        limits = ls.load()
        checks["schema_validates"] = True
    except Exception as e:
        checks["schema_validates"] = False
        reasons.append(f"load() raised: {e}")

    # 3. six limits present with exact field set
    if limits is not None:
        fields = set(type(limits).model_fields.keys())
        checks["six_limits_present"] = fields == EXPECTED_FIELDS
        if fields != EXPECTED_FIELDS:
            extra = fields - EXPECTED_FIELDS
            missing = EXPECTED_FIELDS - fields
            reasons.append(
                f"limits field mismatch; extra={extra}, missing={missing}"
            )
    else:
        checks["six_limits_present"] = False

    # 4. frozen: attempting mutation raises
    frozen_ok = False
    if limits is not None:
        try:
            limits.max_position_notional_pct = 0.99  # type: ignore[misc]
            reasons.append("mutation did NOT raise -- model not frozen")
        except Exception:
            frozen_ok = True
    checks["frozen_enforced"] = frozen_ok

    # 5. extra='forbid': construct with rogue field raises
    extra_forbid_ok = False
    try:
        ls.RiskLimits(
            max_position_notional_pct=0.05,
            max_portfolio_leverage=1.5,
            max_daily_loss_pct=0.02,
            max_trailing_dd_pct=0.10,
            max_gross_exposure_pct=1.00,
            max_sector_weight_pct=0.30,
            rogue_field=3.14,  # should blow up
        )   # type: ignore[call-arg]
        reasons.append("rogue extra field did NOT raise")
    except Exception:
        extra_forbid_ok = True
    checks["extra_forbid"] = extra_forbid_ok

    # 6. load() caches -- two calls return the same object
    cached_ok = False
    try:
        l1 = ls.load()
        l2 = ls.load()
        cached_ok = l1 is l2
        if not cached_ok:
            reasons.append(
                f"load() not cached: id(l1)={id(l1)} id(l2)={id(l2)}"
            )
    except Exception as e:
        reasons.append(f"cache check raised: {e}")
    checks["load_cached"] = cached_ok

    # 7. digest function exists + returns hex
    digest_ok = False
    digest_val = None
    try:
        digest_val = ls.get_limits_digest()
        digest_ok = isinstance(digest_val, str) and len(digest_val) == 64 and all(c in "0123456789abcdef" for c in digest_val)
    except Exception as e:
        reasons.append(f"get_limits_digest raised: {e}")
    checks["digest_ok"] = digest_ok

    all_ok = all(checks.values()) and not reasons
    verdict = "PASS" if all_ok else "FAIL"

    result = {
        "step": "4.9.0",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        **checks,
        "digest_sha256": digest_val,
        "limits": (limits.model_dump() if limits is not None else None),
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        **{k: v for k, v in checks.items()},
    }))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
