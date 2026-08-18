#!/usr/bin/env python3
"""phase-86.47: is the trade drought anomalous, and if so what causes it?

## Read the ordering before the numbers

Two prior steps (86.38, 86.41) were filed on drought theories their own
research gates refuted, and both failed the same way: they reasoned from a
degradation they had just measured *to* the drought, without first establishing
that the drought was anomalous at all. So the questions are asked in the only
order that avoids that:

  1. Is the zero-run anomalous?
  2. Base-rate check against a HEALTHY-funnel null, with sensitivity.
  3. Is the column a funnel would be keyed on even populated?
  4. Did candidates reach the risk gate, and what did it say?
  5. What is the recommendation supply, split by synthesis outcome and path?

## This script ASSERTS. It is not a printer.

Cycle 1 shipped this file with zero assertions and a Q/A killed it: mutating
the failed-cell BUY count from 0 to 99 left it at exit 0, still printing its
conclusions. A "re-runnable check" that cannot fail is not a check.

Cycle 2 then claimed "every recorded figure is now guarded" and a cycle-3 Q/A
refuted it with a known-member recall test over the constants: 9 of 15 mutants
survived at exit 0 while `--verify` printed "OK". The gap was not the guards
but the CONCLUSIONS -- they were prose, printed regardless of the numbers above
them. Every constant is now guarded, the conclusions that depend on a computed
value are CONDITIONAL on it, and the invariant count is derived from the checks
that actually ran rather than being a literal.

## Two denominator rules, both of which cycle 1 broke

**The opportunity unit is an ANALYSIS**, not a calendar day and not a weekday:
`paper_analyze_top_n` caps analyses per cycle, weekends produce none, and a
cycle that has not yet run contributes none.

**A rate's numerator must live in its denominator's population.** Cycle 1
divided 26 trade-days -- three of them weekend days -- by 79 weekdays and
called the result a weekday rate.

## Emitting SQL, not executing it

This script prints the queries beside the recorded results rather than holding
BigQuery credentials. Every figure was produced by running the printed query
through the BigQuery MCP against `sunny-might-477607-p8` on the stated as-of
date; `analysis_results` grows daily, so a figure without its as-of date is not
reproducible.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys

PROJECT = "sunny-might-477607-p8"
AS_OF = "2026-08-18"
BREAK_DATE = "2026-06-15"        # complete by this date; first contaminated day 2026-06-11
PATH_FIELD_EPOCH = "2026-06-11"  # $._path does not exist before this
SILENCE_DAYS = ("2026-08-14", "2026-08-17")
SECTOR_CAP_PCT = 60.0

_FAILURES: list[str] = []
_RAN: list[str] = []


def _check(label: str, cond: bool, detail: str = "") -> None:
    """Assert an invariant and RECORD the outcome, so the script can exit
    non-zero when a figure it prints as fact has stopped being one.

    The count of checks is DERIVED from what actually ran, never a literal: a
    cycle-3 Q/A found `N_INVARIANTS = 13` hardcoded against 14 real calls, so
    `--verify` reported a number it had not measured and could not notice a
    guard being deleted.
    """
    _RAN.append(label)
    if not cond:
        _FAILURES.append(f"{label}: {detail}")


# ── Recorded measurements, each with the query that produces it ──

Q_TRADES = f"""
SELECT SUBSTR(created_at,1,10) AS d, ticker, action, risk_judge_decision
FROM `{PROJECT}.financial_reports.paper_trades` ORDER BY d DESC
-- created_at is a STRING column, not a TIMESTAMP; a timestamp predicate errors.
"""
TRADE_DAYS = [
    "2026-08-13", "2026-07-31", "2026-07-27", "2026-07-20", "2026-07-13",
    "2026-07-09", "2026-07-03", "2026-06-23", "2026-06-10", "2026-06-09",
    "2026-06-08", "2026-06-05", "2026-06-04", "2026-06-03", "2026-06-02",
    "2026-06-01", "2026-05-29", "2026-05-27", "2026-05-22", "2026-05-17",
    "2026-05-16", "2026-05-14", "2026-05-01", "2026-04-28", "2026-04-27",
    "2026-04-26",
]

Q_RJD = f"""
SELECT action, COUNTIF(risk_judge_decision IS NOT NULL
                       AND TRIM(risk_judge_decision) != '') AS populated, COUNT(*) AS n
FROM `{PROJECT}.financial_reports.paper_trades` GROUP BY action
"""
RJD_POPULATION = {                       # (populated, total) -- name the TABLE, they disagree
    "financial_reports.paper_trades BUY":  (19, 34),
    "financial_reports.paper_trades SELL": (0, 32),
    "financial_reports.analysis_results":  (18, 580),
}

Q_JUDGE = f"""
-- JSON_QUERY, not JSON_VALUE: the judge is an OBJECT, and JSON_VALUE returns
-- NULL for an object, so a JSON_VALUE probe reads 0/526 and looks like absence.
-- Cycle 1 hit exactly that and reported the refusal signal "underivable".
SELECT COUNT(*) AS rows_,
       COUNTIF(JSON_QUERY(full_report_json,
               '$.final_synthesis.risk_assessment.judge') IS NOT NULL) AS judge_present
FROM `{PROJECT}.financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP('2026-05-01')
"""
JUDGE_COVERAGE = {                       # scope -> (present, total)
    "since 2026-05-01":     (382, 526),
    "post-break (>=06-15)": (256, 275),
    "the silence window":   (13, 13),
}

Q_WINDOW = f"""
SELECT DATE(analysis_date) AS d, ticker, JSON_VALUE(full_report_json,'$._path') AS path,
       UPPER(TRIM(recommendation)) AS rec,
       JSON_VALUE(JSON_QUERY(full_report_json,
         '$.final_synthesis.risk_assessment.judge'),'$.decision') AS decision,
       JSON_VALUE(JSON_QUERY(full_report_json,
         '$.final_synthesis.risk_assessment.judge'),'$.recommended_position_pct') AS pct
FROM `{PROJECT}.financial_reports.analysis_results`
WHERE DATE(analysis_date) IN {SILENCE_DAYS}
ORDER BY d, ticker
"""
WINDOW = [   # (date, ticker, recommendation, judge decision, recommended pct)
    ("2026-08-14", "HPE",       "HOLD", "REJECT",          0),
    ("2026-08-14", "MRVL",      "HOLD", "REJECT",          0),
    ("2026-08-14", "NTAP",      "HOLD", "APPROVE_REDUCED", 2),
    ("2026-08-14", "PANW",      "HOLD", "REJECT",          0),
    ("2026-08-14", "STX",       "HOLD", "APPROVE_REDUCED", 2),
    ("2026-08-14", "WDAY",      "HOLD", "REJECT",          0),
    ("2026-08-17", "009150.KS", "HOLD", "REJECT",          0),
    ("2026-08-17", "DELL",      "HOLD", "REJECT",          0),
    ("2026-08-17", "HPE",       "HOLD", "REJECT",          0),
    ("2026-08-17", "MRVL",      "HOLD", "APPROVE_HEDGED",  5),
    ("2026-08-17", "MU",        "HOLD", "APPROVE_REDUCED", 3),
    ("2026-08-17", "NTAP",      "HOLD", "REJECT",          0),
    ("2026-08-17", "SNDK",      "HOLD", "APPROVE_REDUCED", 2),
]
Q_POSITIONS = f"""
SELECT ticker, sector, quantity FROM `{PROJECT}.financial_reports.paper_positions`
WHERE quantity > 0 ORDER BY ticker
"""
POSITIONS = [("DELL", "Technology"), ("NTAP", "Technology")]

Q_BUY_CROSSING = f"""
-- CRITERION 2's REAL POPULATION. Cycles 2-3 measured the BUY->gate stage over
-- the 2-day silence window, which contains ZERO BUY-class recommendations --
-- so the stage the criterion names had no members. This is the crossing over
-- the whole post-break era, where the population is non-empty.
SELECT DATE(analysis_date) AS d, ticker, JSON_VALUE(full_report_json,'$._path') AS path,
       CASE WHEN JSON_QUERY(full_report_json,'$.final_synthesis.risk_assessment') IS NULL
            THEN 'NO_RISK_ASSESSMENT' ELSE 'has_ra' END AS ra_state,
       JSON_VALUE(JSON_QUERY(full_report_json,
         '$.final_synthesis.risk_assessment.judge'),'$.decision') AS judge,
       JSON_VALUE(JSON_QUERY(full_report_json,
         '$.final_synthesis.risk_assessment.judge'),'$.recommended_position_pct') AS pct,
       -- The REASONING, because criterion 2 asks for the stated reason and a
       -- claim about it needs a predicate that produces it. Cycle 4 asserted a
       -- ground this query did not select, and got it wrong.
       SUBSTR(JSON_VALUE(JSON_QUERY(full_report_json,
         '$.final_synthesis.risk_assessment.judge'),'$.reasoning'), 1, 720) AS reasoning_head
FROM `{PROJECT}.financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP('{BREAK_DATE}')
  AND UPPER(TRIM(recommendation)) LIKE '%BUY%'
ORDER BY d, ticker
"""

# The DRIVER the judge actually records for the one BUY that reached the gate.
# Cycle 4 asserted "the same sector-concentration ground as section 4"; the
# source text says otherwise and files concentration as CORROBORATING.
DELL_REJECT_DRIVER = {
    "reason_code": "projected_dd_over_cap",
    "projected_dd_pct": 22.5,
    "cap_pct": 10.0,
    "verbatim": ("DECISION DRIVER - LIVE GATE VETO (verified, not narrative). ... "
                 "Result: vetoed=true, reason=projected_dd_over_cap, projected_dd "
                 "22.5% vs a 10% cap [INTERNAL risk-gate]. The projected-DD formula "
                 "is ~0.5x annualized vol, so the veto trips for ANY realized vol "
                 "above ~20%."),
    "concentration_role": "CORROBORATING DOWNSIDE (independent of the gate)",
}
BUY_CROSSING = [   # (date, ticker, path, reached_gate, judge, pct)
    ("2026-07-09", "AMD",  "lite", False, None,     None),
    ("2026-07-09", "MU",   "lite", False, None,     None),
    ("2026-07-20", "PANW", "lite", False, None,     None),
    ("2026-07-31", "NTAP", "lite", False, None,     None),
    ("2026-08-10", "CRWD", "lite", False, None,     None),
    ("2026-08-10", "HPE",  "lite", False, None,     None),
    ("2026-08-11", "NTAP", "lite", False, None,     None),
    ("2026-08-13", "DELL", "full", True,  "REJECT", 0),
]

Q_REJECT_THEN_TRADE = f"""
-- HANDED TO STEP 86.74, NOT DIAGNOSED HERE. Measured facts only.
SELECT created_at, ticker, action, quantity, reason, risk_judge_decision
FROM `{PROJECT}.financial_reports.paper_trades`
WHERE UPPER(TRIM(COALESCE(risk_judge_decision,''))) = 'REJECT'
   OR SUBSTR(created_at,1,10) = '2026-08-13'
ORDER BY created_at
"""
REJECT_THEN_TRADE = [   # (created_at, ticker, qty, reason, risk_judge_decision)
    ("2026-06-02T19:18:58Z", "HPE",       4.402443, "swap_buy",       "REJECT"),
    ("2026-06-03T19:05:19Z", "DELL",      0.581563, "swap_buy",       "REJECT"),
    ("2026-06-09T18:12:39Z", "066570.KS", 1.468448, "swap_buy",       "REJECT"),
    ("2026-08-13T19:31:19Z", "DELL",      4.806437, "new_buy_signal", "(empty)"),
]

Q_PATH = f"""
SELECT COUNT(*) AS n, COUNTIF(JSON_VALUE(full_report_json,'$._path') IS NOT NULL) AS with_path
FROM `{PROJECT}.financial_reports.analysis_results`
"""
PATH_COVERAGE = {"all time": (288, 580), f"from {PATH_FIELD_EPOCH}": (288, 288)}

Q_FUNNEL = f"""
SELECT CASE WHEN analysis_date < TIMESTAMP('{BREAK_DATE}') THEN 'A_pre' ELSE 'B_post' END AS era,
       CASE WHEN JSON_VALUE(full_report_json,'$.final_synthesis.error') IS NOT NULL
            THEN 'FAILED' ELSE 'ok' END AS synth,
       COALESCE(JSON_VALUE(full_report_json,'$._path'),'(unmarked)') AS path,
       COUNT(*) AS analyses, COUNTIF(UPPER(TRIM(recommendation)) LIKE '%BUY%') AS buys
FROM `{PROJECT}.financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP('2026-05-01') GROUP BY era, synth, path
"""
FUNNEL = [
    ("A_pre",  "FAILED", "(unmarked)",  17,   0),
    ("A_pre",  "FAILED", "full",         8,   0),
    ("A_pre",  "ok",     "(unmarked)", 221, 100),
    ("A_pre",  "ok",     "lite",         3,   3),
    ("A_pre",  "ok",     "full",         2,   0),
    ("B_post", "FAILED", "full",       211,   0),
    ("B_post", "ok",     "full",        45,   1),
    ("B_post", "ok",     "lite",        19,   7),
]

Q_SYNTH_ERROR = f"""
SELECT SUBSTR(JSON_VALUE(full_report_json,'$.final_synthesis.error'),1,160) AS synth_error,
       JSON_VALUE(full_report_json,'$._path') AS path, COUNT(*) AS n,
       MIN(DATE(analysis_date)) AS first_d, MAX(DATE(analysis_date)) AS last_d
FROM `{PROJECT}.financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP('2026-06-01') GROUP BY synth_error, path ORDER BY n DESC
"""
SYNTH_ERROR = ("Failed to parse final report.", "full", 219, "2026-06-11", "2026-08-13")

Q_DAILY = f"""
SELECT DATE(analysis_date) AS d, COUNT(*) AS analyses,
       COUNTIF(JSON_VALUE(full_report_json,'$.final_synthesis.error') IS NOT NULL) AS synth_failed,
       COUNTIF(UPPER(TRIM(recommendation)) LIKE '%BUY%') AS buys
FROM `{PROJECT}.financial_reports.analysis_results`
WHERE analysis_date >= TIMESTAMP('2026-08-05') GROUP BY d ORDER BY d
"""
DAILY_TAIL = [   # ALL 11 rows the query returns -- cycle 1's artifact dropped 08-11
    ("2026-08-05", 6, 3, 0), ("2026-08-06", 5, 2, 0), ("2026-08-07", 5, 4, 0),
    ("2026-08-08", 6, 6, 0), ("2026-08-09", 12, 7, 0), ("2026-08-10", 6, 0, 2),
    ("2026-08-11", 6, 1, 1), ("2026-08-12", 6, 6, 0), ("2026-08-13", 6, 1, 1),
    ("2026-08-14", 6, 0, 0), ("2026-08-17", 7, 0, 0),
]

QUERIES = [("TRADES", Q_TRADES), ("RJD COVERAGE", Q_RJD), ("JUDGE COVERAGE", Q_JUDGE),
           ("SILENCE WINDOW", Q_WINDOW), ("POSITIONS", Q_POSITIONS),
           ("BUY x GATE CROSSING", Q_BUY_CROSSING),
           ("REJECT-THEN-TRADE (handed to 86.74)", Q_REJECT_THEN_TRADE),
           ("PATH COVERAGE", Q_PATH), ("FUNNEL", Q_FUNNEL),
           ("SYNTH ERROR", Q_SYNTH_ERROR), ("DAILY", Q_DAILY)]



def _weekdays(a: dt.date, b: dt.date) -> int:
    n, cur = 0, a
    while cur <= b:
        if cur.weekday() < 5:
            n += 1
        cur += dt.timedelta(days=1)
    return n


def verify() -> None:
    """Every invariant behind a printed conclusion. Each fails loudly."""
    days = sorted(dt.date.fromisoformat(x) for x in TRADE_DAYS)
    _check("last_trade_is_2026_08_13", days[-1] == dt.date(2026, 8, 13),
           f"expected 2026-08-13 (DELL BUY), got {days[-1]}")
    _check("trade_day_count", len(days) == 26, f"got {len(days)}")

    wk = [d for d in days if d.weekday() < 5]
    _check("weekday_numerator_is_a_subset", len(wk) == 23 and len(days) - len(wk) == 3,
           f"{len(wk)} weekday and {len(days)-len(wk)} weekend trade-days")

    failed = [(n, b) for _, s, _, n, b in FUNNEL if s == "FAILED"]
    fb, fn = sum(b for _, b in failed), sum(n for n, _ in failed)
    _check("failed_synthesis_yields_zero_buys", fb == 0,
           f"{fb} BUYs in {fn} failed analyses -- the central claim is FALSE")
    _check("failed_cell_size", fn == 236, f"got {fn}")

    pop, tot = RJD_POPULATION["financial_reports.analysis_results"]
    _check("rjd_is_sparse_in_analysis_results", pop / tot < 0.10,
           f"{pop}/{tot} = {100*pop/tot:.1f}% -- above this bound the blindness "
           "conclusion no longer follows and must be re-derived")

    jp, jt = JUDGE_COVERAGE["the silence window"]
    _check("judge_IS_derivable_in_window", jp == jt == 13, f"{jp}/{jt}")
    rej = [r for r in WINDOW if r[3] == "REJECT"]
    _check("window_reject_count", len(rej) == 8, f"got {len(rej)}")
    _check("rejects_are_zero_pct", all(r[4] == 0 for r in rej), "a REJECT carried a non-zero pct")
    _check("window_has_no_BUY_recommendation", all(r[2] == "HOLD" for r in WINDOW),
           "a BUY-class recommendation exists in the window; the "
           "'no BUY arrived to block' reading must be re-derived")
    _check("window_size", len(WINDOW) == 13, f"got {len(WINDOW)}")

    pc, pt = PATH_COVERAGE["all time"]
    _check("path_coverage_is_roughly_half", 0.40 < pc / pt < 0.60,
           f"{pc}/{pt} = {pc/pt:.3f} -- the 'pre-break baseline is path-UNKNOWN' "
           "conclusion needs a substantial unknown share, not merely pc<pt")
    _check("path_coverage_is_NOT_100pct", pc < pt,
           f"{pc}/{pt} -- cycle 1 asserted 100% in three artifacts; it is "
           f"{100*pc/pt:.1f}% all-time and 100% only from {PATH_FIELD_EPOCH}")

    _check("positions_are_one_sector", len({s for _, s in POSITIONS}) == 1,
           "the judge's stated concentration ground no longer holds")
    _check("positions_membership", {t for t, _ in POSITIONS} == {"DELL", "NTAP"},
           f"got {sorted(t for t, _ in POSITIONS)}")
    _check("sector_cap_is_60", SECTOR_CAP_PCT == 60.0, f"got {SECTOR_CAP_PCT}")

    # ── guards a cycle-3 Q/A proved were missing: it falsified each of these
    # constants individually and the suite stayed green at exit 0, because the
    # conclusions that depend on them are printed as PROSE rather than derived.
    _check("daily_rows_complete", len(DAILY_TAIL) == 11, f"got {len(DAILY_TAIL)}")
    # The WHOLE tuple, not just the dates: the artifact prints this table as
    # fact, so every cell in it is a recorded figure. A date-only guard let a
    # mutant rewrite 2026-08-11's counts and survive.
    _check("daily_rows_exact", DAILY_TAIL == [
               ("2026-08-05", 6, 3, 0), ("2026-08-06", 5, 2, 0), ("2026-08-07", 5, 4, 0),
               ("2026-08-08", 6, 6, 0), ("2026-08-09", 12, 7, 0), ("2026-08-10", 6, 0, 2),
               ("2026-08-11", 6, 1, 1), ("2026-08-12", 6, 6, 0), ("2026-08-13", 6, 1, 1),
               ("2026-08-14", 6, 0, 0), ("2026-08-17", 7, 0, 0)],
           "a recorded daily row changed; re-derive it from Q_DAILY before editing")
    _check("daily_counts_are_coherent",
           all(f <= a and b <= a for _, a, f, b in DAILY_TAIL),
           "a daily row reports more failures or buys than analyses")
    _check("silence_days_have_zero_synth_failures",
           all(f == 0 for d, _, f, _ in DAILY_TAIL if d in SILENCE_DAYS),
           "the PHASE-2 'synthesis healthy' premise no longer holds")

    # n_an is the denominator of every criterion-6 probability. Mutating a
    # silence-day row corrupted all four and the header still read '13 rows'.
    n_an = sum(a for d, a, _, _ in DAILY_TAIL if d in SILENCE_DAYS)
    _check("n_an_matches_window", n_an == len(WINDOW),
           f"{n_an} analyses in the silence days vs {len(WINDOW)} window rows -- "
           "the criterion-6 denominator and the funnel disagree")

    msg, spath, sn, sfirst, slast = SYNTH_ERROR
    _check("synth_error_message", msg == "Failed to parse final report.", f"got {msg!r}")
    _check("synth_error_window", (sfirst, slast) == ("2026-06-11", "2026-08-13"),
           f"got {sfirst}..{slast}")
    _check("synth_error_is_full_path", spath == "full", f"got {spath!r}")
    _check("synth_error_size", sn == 219, f"got {sn}")

    healthy = [r for r in FUNNEL if r[0] == "A_pre" and r[1] == "ok"]
    p_h = sum(r[4] for r in healthy) / sum(r[3] for r in healthy)
    _check("healthy_null_rate_is_high", p_h > 0.30,
           f"p={p_h:.4f} -- the HEALTHY-null comparison is the load-bearing one "
           "in criterion 6 and a low rate silently inverts its conclusion")
    # A bounds-only jp>0 accepted 382->5 and 256->5, which would have printed
    # "1.0% ... It IS derivable" -- the conclusion inverted while the guard held.
    for k, (jp, jt) in JUDGE_COVERAGE.items():
        _check(f"judge_coverage_is_MAJORITY[{k}]", jt > 0 and jp / jt > 0.50,
               f"{jp}/{jt} -- the 'the judge IS derivable' conclusion needs a "
               "majority, not merely a non-zero count")

    # The two FUNNEL cells that carry criterion 4's load-bearing contrast.
    _lite = [r for r in FUNNEL if r[0] == "B_post" and r[1] == "ok" and r[2] == "lite"]
    _full = [r for r in FUNNEL if r[0] == "B_post" and r[1] == "ok" and r[2] == "full"]
    _lr = sum(r[4] for r in _lite) / sum(r[3] for r in _lite)
    _fr = sum(r[4] for r in _full) / sum(r[3] for r in _full)
    _check("lite_ok_rate_is_high", _lr > 0.20,
           f"{_lr:.3f} -- criterion 4's contrast (a WORKING path beside a broken "
           "one) is what makes the path split load-bearing")
    _check("full_ok_rate_is_low", _fr < 0.20, f"{_fr:.3f}")
    _check("path_contrast_survives", _lr - _fr > 0.15,
           f"lite {_lr:.3f} vs full {_fr:.3f} -- the contrast no longer holds")

    # Criterion 2's real population.
    _reached = [r for r in BUY_CROSSING if r[3]]
    _post_buys = sum(r[4] for r in FUNNEL if r[0] == "B_post")
    _check("funnel_and_crossing_agree", _post_buys == len(BUY_CROSSING),
           f"FUNNEL post-break BUYs {_post_buys} vs BUY_CROSSING rows "
           f"{len(BUY_CROSSING)} -- two populations of the same thing disagree")
    _check("buy_crossing_population_nonempty", len(BUY_CROSSING) == 8,
           f"got {len(BUY_CROSSING)} post-break BUY-class recommendations")
    _check("buy_crossing_reached_gate", len(_reached) == 1, f"got {len(_reached)}")
    _check("the_one_that_reached_was_REJECTED",
           all(r[4] == "REJECT" and r[5] == 0 for r in _reached),
           "the single BUY that reached the gate was not a 0% REJECT")
    _check("lite_buys_never_reached_the_gate",
           all(not r[3] for r in BUY_CROSSING if r[2] == "lite"),
           "a lite-path BUY carries a risk_assessment; the criterion-4 "
           "separation no longer holds")
    # The COUNT of each label, not only a quantifier over rows already carrying
    # it: relabelling one lite row to full slipped past the quantifier.
    _check("crossing_path_split_is_7_lite_1_full",
           sum(1 for r in BUY_CROSSING if r[2] == "lite") == 7
           and sum(1 for r in BUY_CROSSING if r[2] == "full") == 1,
           "the 7-lite/1-full split behind 'the other 7 are LITE' changed")
    _check("reject_then_trade_rows", len(REJECT_THEN_TRADE) == 4,
           f"got {len(REJECT_THEN_TRADE)}")
    _check("reject_then_trade_rejects",
           sum(1 for r in REJECT_THEN_TRADE if r[4] == "REJECT") == 3,
           "the 'THREE carry REJECT in the trade row itself' claim no longer holds")
    _check("dell_driver_is_not_the_sector_cap",
           DELL_REJECT_DRIVER["reason_code"] == "projected_dd_over_cap"
           and DELL_REJECT_DRIVER["projected_dd_pct"] > DELL_REJECT_DRIVER["cap_pct"],
           "the recorded driver for the one BUY that reached the gate changed; "
           "re-read the judge reasoning before restating it")
    _check("two_gates_are_distinct",
           DELL_REJECT_DRIVER["cap_pct"] != SECTOR_CAP_PCT,
           "the drawdown cap and the sector cap are being treated as one gate")
    for k, (rp, rt) in RJD_POPULATION.items():
        _check(f"rjd_cell_bounds[{k}]", 0 <= rp <= rt and rt > 0, f"{rp}/{rt}")
    # The FINDING is that the two tables disagree and that BUY coverage is
    # PARTIAL. A bounds-only guard accepted 34/34, which would silently turn
    # "the column is unusable" into "the column is complete".
    _bp, _bt = RJD_POPULATION["financial_reports.paper_trades BUY"]
    _sp, _st = RJD_POPULATION["financial_reports.paper_trades SELL"]
    _ap, _at = RJD_POPULATION["financial_reports.analysis_results"]
    _check("rjd_buy_coverage_is_partial", 0 < _bp < _bt,
           f"{_bp}/{_bt} -- neither empty nor complete is the recorded finding")
    _check("rjd_sell_coverage_is_zero", _sp == 0, f"{_sp}/{_st}")
    _check("rjd_tables_disagree", abs(_bp/_bt - _ap/_at) > 0.20,
           f"paper_trades BUY {_bp/_bt:.3f} vs analysis_results {_ap/_at:.3f} -- "
           "the 'two tables disagree and both are true' claim no longer holds")


def main(show_sql: bool, verify_only: bool) -> int:
    if show_sql:
        for name, q in QUERIES:
            print(f"--- {name} ---{q}")
        return 0

    verify()
    if verify_only:
        for f in _FAILURES:
            print("INVARIANT FAILED:", f)
        print(f"OK: all {len(_RAN)} invariants hold" if not _FAILURES
              else f"{len(_FAILURES)} of {len(_RAN)} FAILED")
        return 1 if _FAILURES else 0

    print("=" * 74)
    print(f"phase-86.47 drought census -- all figures AS OF {AS_OF}")
    print("=" * 74)

    days = sorted(dt.date.fromisoformat(x) for x in TRADE_DAYS)
    wk = [d for d in days if d.weekday() < 5]
    wdn = _weekdays(days[0], days[-1])
    n_an = sum(a for d, a, _, _ in DAILY_TAIL if d in SILENCE_DAYS)

    print("\n## 1. CRITERION 1 -- is the zero-run anomalous?")
    print(f"   NORMALISATION: WEEKDAY trade-days / weekdays over "
          f"[{days[0]} .. {days[-1]}] inclusive.")
    print(f"   {len(wk)} weekday trade-days / {wdn} weekdays = {len(wk)/wdn:.4f} per weekday")
    print(f"   {len(days)-len(wk)} of the {len(days)} trade-days are WEEKEND days")
    print( "   (2026-04-26 Sun, 2026-05-16 Sat, 2026-05-17 Sun) and are EXCLUDED.")
    print(f"   Cycle 1 divided all {len(days)} by {wdn} and called it a weekday rate;")
    print( "   a numerator must live in its denominator's population.")
    print(f"   LAST TRADE = {days[-1]} (DELL BUY). The step says 2026-07-31 (NTAP): STALE.")
    print(f"   THE SILENCE: {len(SILENCE_DAYS)} analysis DAYS / {n_an} analyses "
          f"({', '.join(SILENCE_DAYS)}; the days between are a weekend).")

    print("\n## 2. CRITERION 6 -- base rate under a HEALTHY-FUNNEL null, with sensitivity")
    post = [r for r in FUNNEL if r[0] == "B_post"]
    nulls = [
        ("all post-break  (CONTAMINATED: 211/275 = 77% is the failed cell)", post),
        ("post-break, synthesis ok", [r for r in post if r[1] == "ok"]),
        ("post-break, ok + full  (matches all 13 window rows)",
         [r for r in post if r[1] == "ok" and r[2] == "full"]),
        ("pre-break, synthesis ok  (the HEALTHY-funnel null)",
         [r for r in FUNNEL if r[0] == "A_pre" and r[1] == "ok"]),
    ]
    print(f"   {'null population':60s} {'p':>7s} {'P(0 in '+str(n_an)+')':>12s}")
    print("   " + "-" * 82)
    p_healthy = 0.0
    for label, rows in nulls:
        p = sum(r[4] for r in rows) / sum(r[3] for r in rows)
        print(f"   {label:60s} {p:7.4f} {(1-p)**n_an:12.4f}")
        if label.startswith("pre-break"):
            p_healthy = p
    # Two DIFFERENT questions, and conflating them is how a "we need more data"
    # non-answer gets manufactured:
    #   need_healthy -- analyses for a zero-run to be surprising under the
    #                   HEALTHY null. We already have more than this.
    #   need_post    -- analyses for a zero-run to be surprising under the
    #                   CURRENT post-break rate. We are far short of this.
    need_healthy = math.ceil(math.log(0.05) / math.log(1 - p_healthy))
    p_post_all = sum(r[4] for r in post) / sum(r[3] for r in post)
    need_post = math.ceil(math.log(0.05) / math.log(1 - p_post_all))
    print(f"\n   THE CONCLUSION DEPENDS ENTIRELY ON THE NULL. Under the HEALTHY null "
          f"(p={p_healthy:.4f})")
    print(f"   the silence IS surprising -- P(0 in {n_an}) = {(1-p_healthy)**n_an:.4f}, and only "
          f"{need_healthy} analyses")
    print(f"   are needed to reach that bar, so {n_an} is already MORE than enough:")
    print( "   BUY supply has NOT returned to the pre-break rate.")
    # CONDITIONAL, not prose. A Q/A moved a FUNNEL cell so a post-break null
    # crossed 0.05 while this line still printed the unconditional claim.
    _post_ps = [sum(r[4] for r in rows) / sum(r[3] for r in rows)
                for label, rows in nulls if not label.startswith("pre-break")]
    if all((1 - p) ** n_an >= 0.05 for p in _post_ps):
        print( "   Under every post-break null it is not.")
    else:
        _bad = [f"{(1-p)**n_an:.4f}" for p in _post_ps if (1 - p) ** n_an < 0.05]
        print(f"   NOT under every post-break null: {', '.join(_bad)} now clears 0.05,")
        print( "   so the 'unremarkable under the current rate' reading no longer holds.")
    print( "   Cycle 1 reported ONE null whose denominator was 77% the population it")
    print( "   had just called broken, with no sensitivity and no disclosure.")

    print("\n## 3. CRITERION 3 -- proved BEFORE any funnel keyed on that column")
    for k, (pop, tot) in RJD_POPULATION.items():
        print(f"   {k:42s} {pop:4d}/{tot:4d} = {100*pop/tot:5.1f}%")
    _pop, _tot = RJD_POPULATION["financial_reports.analysis_results"]
    print(f"   => {'too sparse to key a funnel on' if _pop/_tot < 0.10 else 'NOT sparse -- the blindness conclusion does NOT follow and must be re-derived'}. The two tables DISAGREE and both are")
    print( "      true (executed trades vs analyses), so every figure names its table.")
    print( "      `risk_intervention_log` (dataset pyfinagent_data) has 0 rows.")

    print("\n## 4. CRITERION 2 -- the refusal funnel, which cycle 1 wrongly called UNDERIVABLE")
    for k, (p, t) in JUDGE_COVERAGE.items():
        print(f"   judge present, {k:24s} {p:4d}/{t:4d} = {100*p/t:5.1f}%")
    print( "\n   It IS derivable, via JSON_QUERY. JSON_VALUE returns NULL for an")
    print( "   OBJECT, so a JSON_VALUE probe reads 0 and looks like absence -- cycle")
    print( "   1 hit that trap and shipped a negative it had never verified.")
    print(f"\n   THE SILENCE WINDOW, all {len(WINDOW)} rows, all path=full:")
    print(f"   {'date':12s} {'ticker':10s} {'rec':6s} {'judge':17s} {'pct':>4s}")
    print("   " + "-" * 54)
    for d, t, rec, dec, pct in WINDOW:
        print(f"   {d:12s} {t:10s} {rec:6s} {dec:17s} {pct:4d}")
    rej = [r for r in WINDOW if r[3] == "REJECT"]
    app = [r for r in WINDOW if r[3] != "REJECT"]
    print(f"\n   REACHED THE GATE {len(WINDOW)}/{len(WINDOW)}   REFUSED {len(rej)} at 0%   "
          f"APPROVED {len(app)} at {min(r[4] for r in app)}-{max(r[4] for r in app)}%")
    print( "   STATED REASON: a portfolio-construction veto on sector concentration.")
    print(f"   The book is {len(POSITIONS)} positions -- {', '.join(t for t, _ in POSITIONS)} --")
    print(f"   both {POSITIONS[0][1]}, so 100% against a {SECTOR_CAP_PCT}% cap.")
    print( "   Confirmed independently against paper_positions.")

    print("\n## 5. CRITERIA 2+4 -- the SUPPLY funnel, split by synthesis AND path")
    print(f"   {'era':8s} {'synthesis':8s} {'path':11s} {'n':>5s} {'BUYs':>5s} {'BUY%':>7s}")
    print("   " + "-" * 52)
    for era, s, p, n, b in FUNNEL:
        print(f"   {era:8s} {s:8s} {p:11s} {n:5d} {b:5d} {100*b/n:6.1f}%")
    fb = sum(b for _, s, _, _, b in FUNNEL if s == "FAILED")
    fn = sum(n for _, s, _, n, _ in FUNNEL if s == "FAILED")
    print(f"\n   SYNTHESIS FAILED: {fb} BUYs in {fn} analyses, every era and path.")
    pc, pt = PATH_COVERAGE["all time"]
    print(f"   PATH COVERAGE {pc}/{pt} = {100*pc/pt:.1f}% all-time, 100% only from "
          f"{PATH_FIELD_EPOCH}.")
    print( "   The pre-break baseline cell is path-UNKNOWN, so the healthy BUY rate")
    print( "   cannot be attributed to a path. Cycle 1 claimed 100% in three artifacts.")

    print("\n## 6. THE ANSWER, in two phases")
    msg, path, n, first_d, last_d = SYNTH_ERROR
    print(f"   PHASE 1  {first_d} .. {last_d}  --  {msg!r}")
    print(f"            {n} rows, path={path}. Supply suppressed: {fb} BUYs in {fn}.")
    print( "            No gate involved: a gate cannot refuse what never arrives.")
    print( "            This is step 86.108's population. 86.47 hands off and does")
    print( "            NOT claim 86.108's fix.")
    print( "   PHASE 2  from 2026-08-14  --  synthesis HEALTHY (0 failures both days).")
    print(f"            Still 0 BUY-class recommendations in {n_an} analyses, AND the")
    print(f"            risk judge refused {len(rej)} of {len(WINDOW)} at 0% on sector")
    print( "            concentration.")
    print( "\n   THE GATE IS NOT EXONERATED. It is ACTIVE AND REFUSING. But the")
    print( "   counterfactual cycles 2-3 attached to that -- 'a ground that would BIND")
    print( "   any BUY that did arrive' -- is WITHDRAWN: it is falsified by the single")
    print( "   observed instance below. The no-BUY-arrived reading is true ONLY inside")
    print( "   the 2-day window, and is contradicted one day earlier by this step's own")
    print( "   corrected last-trade endpoint.")

    print("\n## 6b. CRITERION 2's REAL POPULATION -- the BUY x GATE crossing")
    print( "   Cycles 2-3 measured this stage over the 2-day silence window, which")
    print( "   contains ZERO BUY-class recommendations: the stage the criterion names")
    print(f"   had no members. Over the post-break era ({BREAK_DATE} onward) it does.")
    print(f"\n   {'date':12s} {'ticker':10s} {'path':6s} {'reached gate':13s} {'judge':8s} {'pct':>4s}")
    print("   " + "-" * 58)
    for d, tk, pa, reached, ju, pc in BUY_CROSSING:
        print(f"   {d:12s} {tk:10s} {pa:6s} {'YES' if reached else 'no':13s} "
              f"{ju or '-':8s} {('-' if pc is None else pc):>4}")
    _reached = [r for r in BUY_CROSSING if r[3]]
    print(f"\n   BUY-class recommendations: {len(BUY_CROSSING)}")
    print(f"   REACHED the recorded gate: {len(_reached)}  "
          f"(the other {len(BUY_CROSSING)-len(_reached)} are LITE with NO risk_assessment at all --")
    print( "   including the 2026-08-10 CRWD+HPE pair criteria 2 and 4 both cite)")
    _d = DELL_REJECT_DRIVER
    print(f"   REFUSED: {sum(1 for r in _reached if r[4] == 'REJECT')} at 0%.")
    print(f"\n   ITS STATED REASON IS A DIFFERENT GATE FROM SECTION 4's, and that")
    print( "   distinction is the finding, not a caveat:")
    print(f"     driver     : {_d['reason_code']}  "
          f"(projected_dd {_d['projected_dd_pct']}% vs a {_d['cap_pct']}% cap)")
    print(f"     verbatim   : {_d['verbatim'][:96]}...")
    print(f"     the judge files concentration as: {_d['concentration_role']}")
    print( "\n   So TWO INDEPENDENT gates bind in the post-break era -- a portfolio")
    print( "   SECTOR cap (section 4) and a projected-DRAWDOWN cap (here) -- and the")
    print( "   drawdown one is the more general: the judge records that it is ~0.5x")
    print( "   annualized vol, so it trips for ANY name above ~20% realized vol.")
    print( "   Cycle 4 collapsed the two into one, which is the single-cause")
    print( "   manufacturing this step exists to prevent, committed by this step.")

    print("\n## 6c. HANDED TO STEP 86.74 -- measured facts, NO mechanism asserted")
    print( "   BUY trades that executed while a REJECT verdict was on record:")
    print(f"\n   {'created_at':22s} {'ticker':10s} {'qty':>9s} {'reason':16s} {'rjd':8s}")
    print("   " + "-" * 70)
    for ca, tk, q, rsn, rjd in REJECT_THEN_TRADE:
        print(f"   {ca:22s} {tk:10s} {q:9.6f} {rsn:16s} {rjd:8s}")
    print( "\n   Three carry 'REJECT' in the TRADE ROW ITSELF -- not an inference, a")
    print( "   recorded field. The fourth is 2026-08-13 DELL: the analysis that day")
    print( "   was a full-path BUY the judge REJECTed at 0%, and a DELL BUY executed")
    print( "   53 minutes later with reason='new_buy_signal'.")
    print( "   THIS STEP ASSERTS NO MECHANISM AND CHANGES NOTHING. It is the subject")
    print( "   of step 86.74 (a falsy-zero check inversion), which is already filed")
    print( "   and parked on an operator decision. Recorded here so 86.74 has the")
    print( "   measurement; criterion 5 forbids this step touching a gate.")

    print("\n## 7. WHAT THIS CENSUS MUST NOT CONCLUDE")
    print( "   Synthesis failures ceased 2026-08-14, but FOUR changes landed that day")
    print( "   (sonnet-4-6 -> sonnet-5, risk columns populating, zero-scores ceasing,")
    print( "   recommendation default -> Hold): NOT IDENTIFIABLE.")
    print( "   And the two nulls answer DIFFERENT questions.")
    if n_an >= need_healthy:
        print(f"   Supply has NOT returned to the pre-break rate ({n_an} analyses is "
              f"already past the {need_healthy}-analysis bar).")
    else:
        print(f"   Whether supply returned is UNSETTLED: {n_an} analyses is short of "
              f"the {need_healthy} the healthy null needs.")
    print(f"   But at the CURRENT post-break rate p={p_post_all:.4f}, {need_post} analyses "
          f"would be")
    print(f"   needed before a zero-run said anything at all -- so {n_an} cannot")
    print( "   distinguish 'the sector cap now binds' from 'the rate is simply low'.")
    print( "   NO miscalibration claim: that needs OUTCOME evidence (what the refused")
    print( "   trades would have returned), which this step does not have.")

    if _FAILURES:
        print("\n" + "!" * 74)
        for f in _FAILURES:
            print("INVARIANT FAILED:", f)
        return 1
    print(f"\nAll {len(_RAN)} invariants hold (run --verify to see them alone).")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sql", action="store_true", help="print every BigQuery query and exit")
    ap.add_argument("--verify", action="store_true",
                    help="run ONLY the invariants; exit 1 if a recorded figure stopped holding")
    a = ap.parse_args()
    sys.exit(main(show_sql=a.sql, verify_only=a.verify))
