#!/usr/bin/env python3
"""phase-86.22 -- RE-DERIVE the distribution and measure the before/after delta.

The step's own text carries a measured distribution and then says, in capitals,
RE-DERIVE; do not trust these. So this script queries the column itself and
recomputes every number from that query. Nothing here is transcribed from the
step, the contract, or a previous run.

It produces the three artefacts the live_check asks for:

  1. the per-value distribution of `recommendation`, with a genuine-row count;
  2. the per-consumer, per-value MATCHED / MISSED table under the BEFORE
     expressions (verbatim from git) and the AFTER expression;
  3. the `directionally_correct` before/after delta -- the number of rows whose
     LABEL changes, split by whether the label was wrong-negative or newly
     correct.

It also answers the step's open question -- whether a wrong reflection has
already been persisted -- by counting `agent_memories` rather than assuming.

    source .venv/bin/activate
    python scripts/qa/measure_vocabulary_impact_86_22.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.recommendation_vocab import (  # noqa: E402
    is_buy_intent,
    is_sell_intent,
)

# ── the consumer expressions, VERBATIM as they were before this step ────────
# Copied from `git show HEAD:<file>` at the lines the step names, not retyped
# from memory. Each returns (is_buy, is_sell) so all consumers are comparable.

def before_title_case(rec):
    """outcome_tracker.py:57-58 and memory.py:229-230 -- no case folding."""
    return (rec in ("Strong Buy", "Buy"), rec in ("Strong Sell", "Sell"))


def before_upper_snake(rec):
    """bias_detector.py:119/:128 and api/portfolio.py:142 -- .upper() only."""
    u = (rec or "").upper()
    return (u in ("STRONG_BUY", "BUY"), u in ("STRONG_SELL", "SELL"))


def before_substring(rec):
    """conflict_detector.py:121/:131/:140 -- substring, first clause wins."""
    u = (rec or "").upper()
    if "STRONG_BUY" in u:
        return (True, False)
    if "BUY" in u:
        return (True, False)
    if "SELL" in u:
        return (False, True)
    return (False, False)


def after_canonical(rec):
    """The single shared vocabulary every consumer now uses."""
    return (is_buy_intent(rec), is_sell_intent(rec))


CONSUMERS = [
    ("outcome_tracker:57-58  (title-case)", before_title_case),
    ("memory:229-230         (title-case)", before_title_case),
    ("bias_detector:119,128  (upper-snake)", before_upper_snake),
    ("api/portfolio:140-142  (upper-snake)", before_upper_snake),
    ("conflict_detector:121+ (substring)", before_substring),
]

DIST_SQL = """
SELECT recommendation AS value,
       COUNT(*) AS n,
       COUNTIF(final_score > 0) AS genuine
FROM `{project}.financial_reports.analysis_results`
GROUP BY value
ORDER BY n DESC
"""

# A row is buy-intent for the delta iff the CANONICAL vocabulary says so --
# that is the ground truth the before-expressions are measured against.


def main() -> int:
    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient

    client = BigQueryClient(get_settings())
    project = client.client.project

    print("=" * 78)
    print("1. DISTRIBUTION of financial_reports.analysis_results.recommendation")
    print("   (re-derived; the step's own numbers are deliberately not reused)")
    print("=" * 78)
    rows = list(client.client.query(DIST_SQL.format(project=project)).result())
    dist = [(r["value"], r["n"], r["genuine"]) for r in rows]
    total = sum(n for _, n, _ in dist)
    print(f"\n{'value':<18}{'n':>7}{'genuine':>10}   canonical")
    print("-" * 60)
    for value, n, genuine in dist:
        b, s = after_canonical(value)
        canon = "BUY" if b else ("SELL" if s else "-")
        print(f"{str(value):<18}{n:>7}{genuine:>10}   {canon}")
    print(f"{'TOTAL':<18}{total:>7}")

    buy_rows = [(v, n, g) for v, n, g in dist if after_canonical(v)[0]]
    buy_total = sum(n for _, n, _ in buy_rows)
    print(f"\nbuy-intent rows (canonical): {buy_total} across "
          f"{len(buy_rows)} spelling(s): {[v for v, _, _ in buy_rows]}")

    print("\n" + "=" * 78)
    print("2. PER-CONSUMER, PER-VALUE  MATCHED / MISSED")
    print("=" * 78)
    for label, fn in CONSUMERS:
        missed_n = 0
        print(f"\n{label}")
        print(f"  {'value':<16}{'n':>7}  before(buy,sell)   after(buy,sell)   verdict")
        print("  " + "-" * 70)
        for value, n, _g in dist:
            bb = fn(value)
            aa = after_canonical(value)
            if bb == aa:
                verdict = "same"
            else:
                verdict = "MISSED" if aa != (False, False) else "over-matched"
                missed_n += n
            print(f"  {str(value):<16}{n:>7}  {str(bb):<18} {str(aa):<17} {verdict}")
        pct = (100.0 * missed_n / total) if total else 0.0
        print(f"  --> rows this consumer classified DIFFERENTLY from the shared "
              f"vocabulary: {missed_n} / {total} ({pct:.1f}%)")

    print("\n" + "=" * 78)
    print("3. directionally_correct  BEFORE -> AFTER  (outcome_tracker)")
    print("=" * 78)
    print("""
`directionally_correct = (is_buy and return>0) or (is_sell and return<0)`.
The label flips ONLY where the before/after intent differs, so the delta is
bounded by the MISSED count above and its sign depends on the realised return.
Both outcomes are reported rather than assumed.""")
    tc_missed = sum(n for v, n, _ in dist
                    if before_title_case(v) != after_canonical(v))
    print(f"\nrows where outcome_tracker's intent changes: {tc_missed} / {total}")
    print("  of these, the label was previously FALSE regardless of return,")
    print("  because neither leg matched. After the fix the label is decided by")
    print("  the return: a winning call reads correct, a losing one reads wrong.")
    for value, n, genuine in dist:
        if before_title_case(value) != after_canonical(value):
            b, s = after_canonical(value)
            print(f"    {str(value):<14} n={n:<5} genuine={genuine:<5} "
                  f"now scored as {'BUY' if b else 'SELL'}")

    print("\n" + "=" * 78)
    print("4. HAS A WRONG REFLECTION ALREADY BEEN PERSISTED?  (measure, not assume)")
    print("=" * 78)
    for table in ("agent_memories", "outcome_tracking"):
        try:
            q = f"SELECT COUNT(*) AS n FROM `{project}.financial_reports.{table}`"
            n = list(client.client.query(q).result())[0]["n"]
            print(f"  {table:<20} rows = {n}")
        except Exception as exc:                     # report, never swallow
            print(f"  {table:<20} UNAVAILABLE: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
