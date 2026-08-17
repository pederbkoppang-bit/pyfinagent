#!/usr/bin/env python3
"""phase-86.109 criterion 5: mutation matrix for the freshness calendar gate.

Same strict scoring rule as the phase-86.108 matrix, and for the same reasons:
a cell is a KILL only if the CONTROL was green first, pytest exits **1** (exit
5 -- no tests collected -- is NOT a kill), the mutant collects the SAME number
of tests as the control (a mutant that cannot build is not a killed mutant),
and the **specifically named** test is the one that fails. A cell that goes red
only via collateral damage does not demonstrate that *this* guard catches
*this* defect.

The cells attack the two directions this change could break in. Removing or
inverting the gate is the obvious one. The subtler one is **N3**: committing
the baseline before the gate looks harmless and is not -- it absorbs a weekend
red into "already known", so the source never pages at all and the weekend mute
silently becomes permanent. That was a real bug in this step's first draft,
caught by writing the deferral test before the matrix rather than after.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
# BOTH files: cell N11 mutates the phase-82.10 guard this step inverted, and a
# cell whose named test is not even collected scores UNSCORABLE, not KILLED.
SUITE = ["backend/tests/test_phase_86_109_freshness_calendar.py",
         "backend/tests/test_phase_82_10_freshness_paging.py"]
PY = str(REPO / ".venv" / "bin" / "python")

CRON = "backend/services/freshness_cron.py"
MARKETS = "backend/backtest/markets.py"
OBS = "backend/api/observability_api.py"
PT = "backend/api/paper_trading.py"
SCHED = "backend/slack_bot/scheduler.py"
HEALTH = "backend/services/cycle_health.py"
T8210 = "backend/tests/test_phase_82_10_freshness_paging.py"

_FAILOPEN_LOG = (
    '            logger.warning(\n'
    '                "freshness_evaluator: trading-day check fail-open (%r) -- "\n'
    '                "notifying as if today were a session.", exc,\n'
    '            )'
)

# id -> (file, old_substring, new_substring, test that MUST fail, what it proves)
CELLS = [
    ("N1", CRON,
     '            trading_day = is_us_trading_day_now("US")',
     '            trading_day = True  # MUTANT: gate removed',
     "test_weekend_newly_red_does_not_page",
     "removing the trading-day gate is caught"),

    ("N2", CRON,
     '            trading_day = is_us_trading_day_now("US")',
     '            trading_day = not is_us_trading_day_now("US")  # MUTANT: inverted',
     "test_weekday_newly_red_STILL_pages",
     "an INVERTED gate -- muting weekdays, paging weekends -- is caught"),

    ("N3", CRON,
     "        if trading_day:\n            _last_red_sources = red_now",
     "        if True:  # MUTANT: commit the baseline before the gate\n            _last_red_sources = red_now",
     "test_weekend_suppression_is_DEFERRED_not_dropped",
     "absorbing a withheld source into the baseline -- a permanent silent mute"),

    ("N4", PT,
     "        compute_freshness, bq, cycle_interval_sec, emit_alarm=False",
     "        compute_freshness, bq, cycle_interval_sec, emit_alarm=True  # MUTANT",
     "test_paper_trading_freshness_route_does_not_page",
     "a read path that pages again is caught"),

    ("N5", OBS,
     '    return await _asyncio.to_thread(_cf, bq, cycle_interval_sec, emit_alarm=False)\n\n\n@router.get("/data-freshness")',
     '    return await _asyncio.to_thread(_cf, bq, cycle_interval_sec)  # MUTANT\n\n\n@router.get("/data-freshness")',
     "test_observability_freshness_aliases_do_not_page",
     "an observability alias that pages again is caught"),

    ("N6", CRON,
     _FAILOPEN_LOG,
     "            trading_day = False  # MUTANT: fail CLOSED\n" + _FAILOPEN_LOG,
     "test_calendar_failure_fails_open_and_still_pages",
     "a calendar failure that SUPPRESSES a page (inverted polarity) is caught"),

    ("N7", MARKETS,
     '    et_today = datetime.now(ZoneInfo("America/New_York")).date()\n    return is_trading_day(et_today, market)',
     '    et_today = datetime.now(ZoneInfo("America/New_York")).date()\n    _ = et_today\n    return False  # MUTANT: always says non-trading',
     "test_trading_day_helper_fails_OPEN",
     "a helper that ignores the calendar and refuses is caught"),

    ("N8", SCHED,
     '    from backend.backtest.markets import is_us_trading_day_now\n    return is_us_trading_day_now("US")',
     '    from backend.backtest.markets import is_trading_day  # MUTANT: parallel definition\n    return is_trading_day(datetime.now(ZoneInfo("America/New_York")).date(), "US")',
     "test_trading_day_helper_is_shared_not_duplicated",
     "re-growing a parallel trading-day definition is caught"),
# N9-N11 exist because a Q/A executed three mutants this matrix did not
    # contain and ALL THREE SURVIVED. Every one attacked a GUARD, not the
    # product -- the shipped behaviour was correct in each case, and the
    # assertions that were supposed to protect it could not fail.
    ("N9", HEALTH,
     "    ratio = age_sec / interval_sec",
     '    import datetime as _d\n    from zoneinfo import ZoneInfo as _Z\n'
     '    if _d.datetime.now(_Z("America/New_York")).weekday() >= 5:\n'
     '        return "green"  # MUTANT: calendar reaches DETECTION\n'
     "    ratio = age_sec / interval_sec",
     "test_band_has_no_day_of_week_term_after_the_fix",
     "the calendar reaching _band -- invisible Mon-Fri to the old tautology"),

    ("N10", HEALTH,
     "    if emit_alarm and overall_band == \"red\":",
     "    if False and emit_alarm and overall_band == \"red\":  # MUTANT: alarm inert",
     "test_compute_freshness_still_pages_when_asked",
     "an alarm path made completely inert -- the old 'control' could not see it"),

    # N11 mutates the CODE site, not the guard. Reverting the comment-strip
    # alone only WEAKENS the guard -- the suite stays green because
    # `emit_alarm=False` is genuinely present in code, so that mutant is
    # non-discriminating and scored SURVIVED on its first run here. The
    # discriminating form removes the real suppression while LEAVING the
    # explanatory comment in place: a byte-pin that counts comment tokens
    # passes, a comment-stripped one fails. That is exactly the mutant a Q/A
    # used to defeat the first draft of this guard.
    ("N11", PT,
     "        compute_freshness, bq, cycle_interval_sec, emit_alarm=False\n    )",
     "        compute_freshness, bq, cycle_interval_sec\n    )  # MUTANT: code site reverted, COMMENT left standing",
     "test_http_call_sites_now_suppress_the_alarm_phase_86_109",
     "a byte-pin satisfiable by the comment the same diff added"),
]


def run_suite() -> tuple[int, str]:
    p = subprocess.run(
        [PY, "-m", "pytest", *SUITE, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
    )
    return p.returncode, p.stdout + p.stderr


def collected(out: str) -> int:
    """Total tests that actually RAN. Rejects a mutant that broke the import
    and therefore collected nothing."""
    return sum(int(m.group(1)) for m in re.finditer(r"(\d+) (passed|failed|error)", out))


def main() -> int:
    print("=" * 70)
    print("CONTROL (must be GREEN before any cell is scored)")
    print("=" * 70)
    rc, out = run_suite()
    control_n = collected(out)
    print(out.strip().splitlines()[-1] if out.strip() else "(no output)")
    print(f"CONTROL rc={rc}  collected={control_n}")
    if rc != 0 or control_n == 0:
        print("\nCONTROL IS NOT GREEN -- every cell below would be UNSCORABLE. Aborting.")
        return 2

    results = []
    for cid, relpath, old, new, must_fail, proves in CELLS:
        path = REPO / relpath
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        text = original.decode("utf-8")

        if text.count(old) != 1:
            results.append((cid, "UNSCORABLE", f"anchor matched {text.count(old)}x (need exactly 1)", proves))
            print(f"{cid}: UNSCORABLE -- anchor matched {text.count(old)} times")
            continue

        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        try:
            m_rc, m_out = run_suite()
            m_n = collected(m_out)
            named_failed = must_fail in m_out
            if m_rc == 5:
                verdict, why = "UNSCORABLE", "pytest exit 5 -- no tests collected"
            elif m_rc == 0:
                verdict, why = "SURVIVED", "suite stayed green under the mutation"
            elif m_rc != 1:
                verdict, why = "UNSCORABLE", f"pytest exit {m_rc} (not a test failure)"
            elif m_n != control_n:
                verdict, why = "UNSCORABLE", f"collected {m_n} != control {control_n} (mutant did not build)"
            elif not named_failed:
                verdict, why = "UNSCORABLE", f"red, but {must_fail} was not the failing test"
            else:
                verdict, why = "KILLED", f"{must_fail} failed; collected {m_n} == control"
        finally:
            # No `return` in this block: it would discard an in-flight exception.
            path.write_bytes(original)
            restore_ok = hashlib.sha256(path.read_bytes()).hexdigest() == digest

        if not restore_ok:
            print(f"\nFATAL: restore of {relpath} did NOT verify. Tree is dirty. Aborting.")
            return 3

        results.append((cid, verdict, why, proves))
        print(f"{cid}: {verdict:11s} {why}")

    print("\n" + "=" * 70)
    print("MATRIX")
    print("=" * 70)
    for cid, verdict, why, proves in results:
        print(f"  {cid:4s} {verdict:11s} {proves}")
        if verdict != "KILLED":
            print(f"        -> {why}")
    survivors = [c for c, v, *_ in results if v == "SURVIVED"]
    unscorable = [c for c, v, *_ in results if v == "UNSCORABLE"]
    print(f"\nKILLED={len([1 for _, v, *_ in results if v == 'KILLED'])}/{len(results)}"
          f"  SURVIVORS={survivors or 'none'}  UNSCORABLE={unscorable or 'none'}")
    print("\nRESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256.")
    return 0 if not survivors and not unscorable else 1


if __name__ == "__main__":
    raise SystemExit(main())
