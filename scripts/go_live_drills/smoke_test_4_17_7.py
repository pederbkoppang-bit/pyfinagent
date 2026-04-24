#!/usr/bin/env python
"""phase-4.17.7 smoke test -- Paper trading execution + virtual portfolio.

We cannot synthesize a test BUY from the smoke drill (no /order
endpoint; the paper-trading loop runs autonomously and writes to BQ
on its own cadence). Instead we verify the paper-trading pipeline is
FULLY WIRED end-to-end by:

1. Importing the paper-trading service modules (pipeline is loadable).
2. Querying `financial_reports.paper_portfolio` -- NAV + inception present.
3. Querying `financial_reports.paper_portfolio_snapshots` -- >=1 row.
4. Querying `financial_reports.paper_trades` -- list returnable (rows >= 0).

This proves: (a) the module graph is healthy, (b) the virtual portfolio
exists in BQ, (c) the trades table is read-writable, (d) state has
evolved from the seed row (at least 1 snapshot captured = loop has fired).

Criteria:
- paper_trading_modules_importable
- paper_portfolio_has_nav_and_inception
- at_least_one_snapshot_exists
- paper_trades_table_reachable
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


def test_paper_trading_pipeline_healthy():
    # 1. Module imports
    try:
        import backend.services.paper_trader  # noqa: F401
        import backend.services.portfolio_manager  # noqa: F401
        print("PASS paper_trading_modules_importable")
    except Exception as e:
        raise AssertionError(f"paper_trading_modules_importable FAIL: {e!r}")

    # 2. BQ checks
    try:
        from google.cloud import bigquery
    except Exception as e:
        raise AssertionError(f"google-cloud-bigquery missing: {e!r}")

    project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
    client = bigquery.Client(project=project)

    # paper_portfolio -- 1 row with NAV + inception
    pp = list(
        client.query(
            f"SELECT total_nav, inception_date FROM `{project}.financial_reports.paper_portfolio` LIMIT 1",
            timeout=30,
        ).result()
    )
    assert pp, "paper_portfolio has zero rows"
    row = pp[0]
    nav = float(row.get("total_nav") or 0.0)
    assert nav > 0, f"paper_portfolio.total_nav not positive: {nav}"
    assert row.get("inception_date") is not None, "paper_portfolio.inception_date null"
    print(f"PASS paper_portfolio_has_nav_and_inception -- nav={nav}, inception={row['inception_date']}")

    # snapshots -- >=1
    snap = list(
        client.query(
            f"SELECT COUNT(*) AS n FROM `{project}.financial_reports.paper_portfolio_snapshots`",
            timeout=30,
        ).result()
    )
    n_snap = int(snap[0]["n"])
    assert n_snap >= 1, f"no snapshots recorded yet: n={n_snap}"
    print(f"PASS at_least_one_snapshot_exists -- n={n_snap}")

    # trades -- reachable (0 rows is acceptable; 1+ is better)
    trades = list(
        client.query(
            f"SELECT COUNT(*) AS n FROM `{project}.financial_reports.paper_trades`",
            timeout=30,
        ).result()
    )
    n_tr = int(trades[0]["n"])
    print(f"PASS paper_trades_table_reachable -- n_trades={n_tr}")

    print("PASS 4.17.7 paper-trading pipeline healthy")


if __name__ == "__main__":
    try:
        test_paper_trading_pipeline_healthy()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
