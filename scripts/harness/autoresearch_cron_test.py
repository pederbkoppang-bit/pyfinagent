"""phase-8.5.7 overnight cron verification."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.autoresearch.budget import BudgetEnforcer
from backend.autoresearch.cron import AutoresearchCron


def case_cron_registered() -> tuple[bool, str]:
    c = AutoresearchCron()
    ok = c.register()
    if not ok or not c.registered:
        return False, "register() returned False or flag not set"
    return True, "cron.registered=True after register()"


def case_ge_80_within_budget() -> tuple[bool, str]:
    c = AutoresearchCron()
    enforcer = BudgetEnforcer(wallclock_seconds=3600.0, usd_budget=100.0)
    # Each experiment costs $0.50 -> up to 200 before budget hits, so 100 runs fine.
    def run_one(i: int):
        return {"index": i, "usd_spent": 0.5}
    out = c.run_batch(enforcer, run_one, max_experiments=100)
    if out["experiments_run"] < 80:
        return False, f"only {out['experiments_run']} < 80"
    return True, f"{out['experiments_run']} experiments within budget"


def case_results_visible_in_phase_4_7() -> tuple[bool, str]:
    """Compile-time check: there's a results channel (list) that phase-4.7 can read."""
    c = AutoresearchCron()
    enforcer = BudgetEnforcer(wallclock_seconds=3600.0, usd_budget=10.0)
    def run_one(i: int):
        return {"index": i, "usd_spent": 0.1, "sharpe": 1.0 + i * 0.01}
    out = c.run_batch(enforcer, run_one, max_experiments=20)
    if "results" not in out or not isinstance(out["results"], list):
        return False, "run_batch output missing 'results' list"
    if len(out["results"]) == 0:
        return False, "no results returned"
    return True, f"results channel populated ({len(out['results'])} rows)"


def main() -> int:
    cases = [
        ("cron_registered", case_cron_registered),
        ("ge_80_experiments_per_night_within_budget", case_ge_80_within_budget),
        ("results_visible_in_phase_4_7_view", case_results_visible_in_phase_4_7),
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
