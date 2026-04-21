"""phase-8.5.6 promotion verification."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.autoresearch.promoter import Promoter, SHADOW_MIN_DAYS, DD_TRIGGER


def case_shadow_min() -> tuple[bool, str]:
    p = Promoter()
    v = p.promote({"trial_id": "t1", "dsr": 0.99, "shadow_trading_days": 3})
    if v["promoted"]:
        return False, "promoted despite shadow_days=3 < 5"
    if "shadow_days_below_min" not in (v.get("reason") or ""):
        return False, f"reason miss: {v['reason']!r}"
    v2 = p.promote({"trial_id": "t2", "dsr": 0.99, "shadow_trading_days": 5})
    if not v2["promoted"]:
        return False, f"expected promotion at exactly 5 days, got {v2}"
    return True, f"shadow_trading_days >= {SHADOW_MIN_DAYS} enforced"


def case_position_size_dsr_tied() -> tuple[bool, str]:
    p = Promoter()
    assert p.position_size({"dsr": 0.50}, 10000) == 0.0, "dsr=0.5 should yield zero"
    assert p.position_size({"dsr": 1.00}, 10000) == 10000.0, "dsr=1.0 should yield full capital"
    assert p.position_size({"dsr": 0.75}, 10000) == 5000.0, "dsr=0.75 -> half"
    # Below DSR=0.5 floors at 0
    assert p.position_size({"dsr": 0.10}, 10000) == 0.0
    return True, "position_size scales with realized DSR"


def case_kill_switch_on_dd_breach() -> tuple[bool, str]:
    p = Promoter()
    fired: list[str] = []
    ok1 = p.on_dd_breach(-0.05, lambda r: fired.append(r))
    if ok1 or fired:
        return False, "kill fired too early (|dd|=0.05 < 0.10)"
    ok2 = p.on_dd_breach(-0.15, lambda r: fired.append(r))
    if not ok2 or len(fired) != 1:
        return False, f"kill did not fire on |dd|=0.15 (fired={fired})"
    if "dd_breach" not in fired[0]:
        return False, f"kill reason missing dd_breach: {fired[0]!r}"
    return True, f"kill_switch auto-triggers on |dd| > {DD_TRIGGER}"


def main() -> int:
    cases = [
        ("shadow_5_trading_days_minimum", case_shadow_min),
        ("position_size_tied_to_realized_dsr", case_position_size_dsr_tied),
        ("kill_switch_auto_triggers_on_dd_breach", case_kill_switch_on_dd_breach),
    ]
    ok_all = True
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'}: {name} -- {msg}")
        if not ok:
            ok_all = False
    print("---")
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
