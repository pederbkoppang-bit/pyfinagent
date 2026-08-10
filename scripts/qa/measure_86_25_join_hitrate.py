#!/usr/bin/env python3
"""phase-86.25 P1 -- measure BEFORE building.

The research brief chose "option (A) where the lookup resolves, option (C) where
it does not" and then said, explicitly, that a contract choosing (A) must FIRST
measure how many rows the lookup actually resolves for -- because the join anchor
at `autonomous_loop.py` is `analysis_id or created_at`, i.e. already ambiguous.
Building the (A) path without that number would be designing for a population
whose size nobody had checked.

So this script answers three questions and asserts nothing:

  Q1  the FULL `risk_judge_decision` distribution           (criterion 2:
      re-derived from the table, never copied from the step text)
  Q2  the same, split by `action` -- the step's central claim is that SELL rows,
      the ONLY rows the learn loop reads, are 100% empty
  Q3  the join hit-rate: of the SELL rows, how many have an `analysis_id` that
      resolves to a row in `analysis_results` carrying a recommendation?
      That fraction is option (A)'s reachable population; the remainder is
      option (C)'s.

Read-only. Every query is bounded and none writes. Uses the Python client per
CLAUDE.md "BigQuery Access (MCP)" rule 6 (the MCP tools are not in every
session's surface); ADC on this machine covers auth.

    $ python scripts/qa/measure_86_25_join_hitrate.py
"""
from __future__ import annotations

import collections
import sys

PROJECT = "sunny-might-477607-p8"
DATASET = "financial_reports"          # paper_trades lives HERE, not pyfinagent_pms
TIMEOUT = 30                            # CLAUDE.md: 30s on every fallback query


def main() -> int:
    try:
        from google.cloud import bigquery
    except Exception as exc:                                   # noqa: BLE001
        print(f"google-cloud-bigquery unavailable: {exc}")
        return 1

    client = bigquery.Client(project=PROJECT)

    def q(sql: str):
        return list(client.query(sql).result(timeout=TIMEOUT))

    print("=" * 78)
    print("Q1 -- risk_judge_decision, FULL distribution (criterion 2, re-derived)")
    print("=" * 78)
    rows = q(f"""
        SELECT COALESCE(risk_judge_decision, '<NULL>') AS v, COUNT(*) AS n
        FROM `{PROJECT}.{DATASET}.paper_trades`
        GROUP BY v ORDER BY n DESC
    """)
    total = sum(r["n"] for r in rows)
    for r in rows:
        label = "'' (empty string)" if r["v"] == "" else repr(r["v"])
        print(f"  {label:24s} {r['n']:4d}")
    print(f"  {'TOTAL':24s} {total:4d}")
    vocab = {r["v"] for r in rows if r["v"] not in ("", "<NULL>")}
    directional = {v for v in vocab if v.upper() in ("BUY", "SELL", "HOLD")}
    print(f"\n  non-empty spellings: {sorted(vocab)}")
    print(f"  of those, on the BUY/HOLD/SELL scale: {sorted(directional) or 'NONE'}")

    print("\n" + "=" * 78)
    print("Q2 -- split by action (the learn loop reads SELL rows ONLY)")
    print("=" * 78)
    rows = q(f"""
        SELECT action, COALESCE(risk_judge_decision, '<NULL>') AS v, COUNT(*) AS n
        FROM `{PROJECT}.{DATASET}.paper_trades`
        GROUP BY action, v ORDER BY action, n DESC
    """)
    by_action = collections.defaultdict(list)
    for r in rows:
        by_action[r["action"]].append((r["v"], r["n"]))
    for action, pairs in sorted(by_action.items()):
        tot = sum(n for _v, n in pairs)
        empty = sum(n for v, n in pairs if v in ("", "<NULL>"))
        print(f"  action={action!r:8s} n={tot:3d}   empty risk_judge_decision: {empty}/{tot}"
              f"  ({100.0*empty/tot:.0f}%)")
        for v, n in pairs:
            print(f"      {('<empty>' if v in ('', '<NULL>') else v):20s} {n}")

    print("\n" + "=" * 78)
    print("Q3 -- JOIN HIT-RATE: can option (A) look up a real recommendation?")
    print("=" * 78)
    rows = q(f"""
        SELECT
          COUNT(*) AS sells,
          COUNTIF(t.analysis_id IS NOT NULL AND t.analysis_id != '') AS has_anchor,
          COUNTIF(a.recommendation IS NOT NULL AND a.recommendation != '') AS resolves
        FROM `{PROJECT}.{DATASET}.paper_trades` t
        LEFT JOIN `{PROJECT}.{DATASET}.analysis_results` a
          ON a.analysis_id = t.analysis_id
        WHERE t.action = 'SELL'
    """)
    r = rows[0]
    sells, anchor, resolves = r["sells"], r["has_anchor"], r["resolves"]
    print(f"  SELL rows                                : {sells}")
    print(f"  ... carrying a non-empty analysis_id     : {anchor}"
          f"  ({100.0*anchor/sells:.0f}%)" if sells else "")
    print(f"  ... resolving to an analysis_results rec : {resolves}"
          f"  ({100.0*resolves/sells:.0f}%)" if sells else "")
    print()
    print(f"  OPTION (A) reachable population : {resolves}/{sells}")
    print(f"  OPTION (C) population           : {sells - resolves}/{sells}")
    print("\n  Read this before choosing: if (A) resolves for few or no rows, the")
    print("  design is mostly (C) and must be judged as such -- an (A)-shaped fix")
    print("  guarded only by tests that stub the lookup would be a fix for a")
    print("  population that does not exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
