"""phase-4.8.1 migration: add `delisted_at DATE` to historical_prices.

Idempotent. Safe to re-run. Logs a no-op if the column already exists.
Does NOT populate the column -- a later delistings-feed ingestion
step (queued as phase-4.8.x) will backfill. The SCHEMA presence is
what unblocks PIT-aware queries: `WHERE delisted_at IS NULL OR
delisted_at > :as_of`.
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("add_delisted_at_column")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


PROJECT = "sunny-might-477607-p8"
DATASET = "pyfinagent_data"
TABLE = "historical_prices"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sql = f"""
    ALTER TABLE `{PROJECT}.{DATASET}.{TABLE}`
    ADD COLUMN IF NOT EXISTS delisted_at DATE
    OPTIONS(description='Last trading date before delisting; NULL if still listed. Populated by phase-4.8.x delistings-feed ingestion. Added by phase-4.8.1 survivorship-bias audit.')
    """
    if args.dry_run:
        logger.info("dry-run -- would execute:\n%s", sql.strip())
        return 0

    try:
        from google.cloud import bigquery
    except Exception as e:
        logger.error("google-cloud-bigquery not importable: %s", e)
        return 1
    client = bigquery.Client(project=PROJECT)
    logger.info("executing: %s", sql.strip().replace("\n", " "))
    job = client.query(sql)
    job.result(timeout=30)
    logger.info("delisted_at column added (or already present) on %s.%s.%s",
                PROJECT, DATASET, TABLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
