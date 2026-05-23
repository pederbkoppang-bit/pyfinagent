"""phase-27.4: add 5 missing FLOAT64 columns to financial_reports.analysis_results.

Closes B-2 from the 2026-05-16 pre-prod smoke audit
(`docs/audits/smoke_test_preprod_2026-05-16.md`).

`BigQueryClient.save_report()` (backend/db/bigquery_client.py:113-117) writes
these 5 fields on every lite-path persist (Phase-11 Autoresearch
FEATURE_TO_AGENT bridge), but the BQ analysis_results table is missing
them — the Phase-11 schema migration was partial. Live cycle 756a19c7 /
74c322b2 showed 14/15 analyses failing to persist with
`no such field: <column>` errors rotating across the five missing columns.

BigQuery ADD COLUMN constraints:
  - ONE column per ALTER TABLE statement (multi-column not supported)
  - IF NOT EXISTS is the idempotent idiom
  - New columns must be NULLABLE (REQUIRED columns require backfill DDL)

Existing rows stay NULL; future writes from `save_report()` populate them
naturally. No backfill needed — these are forward-looking feature columns.

Modes:
  --dry-run (default): print DDLs + report current column presence
  --apply              execute the 5 ALTER TABLE statements
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("add_phase27_columns")


# Columns to add. (name, description) — description goes into BigQuery OPTIONS.
COLUMNS = [
    (
        "consumer_sentiment",
        "phase-27.4: consumer-sentiment composite (-1..+1) from Reddit + news + alt-data sources.",
    ),
    (
        "revenue_growth_yoy",
        "phase-27.4: trailing-twelve-month revenue growth year-over-year (decimal fraction; 0.15 = 15%).",
    ),
    (
        "quality_score",
        "phase-27.4: composite quality score (1-10) from gross margin stability + ROIC + leverage.",
    ),
    (
        "momentum_6m",
        "phase-27.4: 6-month price momentum (decimal return; 0.20 = +20%).",
    ),
    (
        "rsi_14",
        "phase-27.4: 14-day Relative Strength Index (0-100).",
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Execute the ALTER TABLE statements (default: dry-run)",
    )
    args = ap.parse_args()

    sys.path.insert(0, "/Users/ford/.openclaw/workspace/pyfinagent")
    from backend.config.settings import get_settings  # type: ignore
    from google.cloud import bigquery  # type: ignore

    settings = get_settings()
    project = settings.gcp_project_id
    dataset = settings.bq_dataset_reports
    table_fqn = f"`{project}.{dataset}.analysis_results`"

    # financial_reports is in us-central1 (per CLAUDE.md note).
    client = bigquery.Client(project=project, location="us-central1")

    # Pre-flight: confirm none of the 5 already exist with a wrong type.
    try:
        table = client.get_table(f"{project}.{dataset}.analysis_results")
        existing = {f.name: f.field_type for f in table.schema}
    except Exception as exc:
        logger.error("get_table failed (%s); aborting", exc)
        return 2

    target_names = {c[0] for c in COLUMNS}
    type_mismatches = []
    for name in target_names & existing.keys():
        if existing[name] != "FLOAT64" and existing[name] != "FLOAT":
            type_mismatches.append((name, existing[name]))
    if type_mismatches:
        logger.error(
            "TYPE MISMATCH on already-present columns; aborting before any DDL: %s",
            type_mismatches,
        )
        return 3

    already_present = sorted(target_names & existing.keys())
    needed = [c for c in COLUMNS if c[0] not in existing]
    logger.info("=== phase-27.4 migration ===")
    logger.info("Target: %s", table_fqn)
    logger.info("Already present (no-op): %s", already_present or "none")
    logger.info("To add: %s", [c[0] for c in needed] or "none -- fully idempotent re-run")

    if not needed:
        logger.info("ok phase-27.4 migration is a no-op; all 5 columns already exist")
        return 0

    for name, description in needed:
        ddl = (
            f"ALTER TABLE {table_fqn}\n"
            f"ADD COLUMN IF NOT EXISTS {name} FLOAT64\n"
            f"OPTIONS(description={description!r})"
        )
        if not args.apply:
            logger.info("DRY RUN -- would execute:\n%s", ddl)
            continue
        try:
            client.query(ddl, location="us-central1").result(timeout=30)
            logger.info("Added column: %s", name)
        except Exception as exc:
            logger.error("ALTER TABLE failed for %s: %s", name, exc)
            return 4

    if not args.apply:
        logger.info("DRY RUN complete. Re-run with --apply to execute.")
        return 0

    # Post-flight verification.
    table_post = client.get_table(f"{project}.{dataset}.analysis_results")
    post_names = {f.name for f in table_post.schema}
    still_missing = target_names - post_names
    if still_missing:
        logger.error("POST-FLIGHT FAILED -- still missing: %s", still_missing)
        return 5
    logger.info("ok phase-27.4 migration: all 5 columns now present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
