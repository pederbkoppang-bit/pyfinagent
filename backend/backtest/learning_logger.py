"""Learning iteration logging for autonomous harness.

Logs each iteration's decisions, test results, and findings to BigQuery.
Falls back to JSON file logging if BigQuery unavailable.

BigQuery table: harness_learning_log
Partitioned by: timestamp (daily)
Clustered by: evaluator_verdict, iteration_id
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IterationLog:
    """Schema for BigQuery harness_learning_log table."""

    # Identifiers
    timestamp: str  # ISO8601 UTC, e.g., "2026-04-06T21:30:00Z"
    iteration_id: str  # UUID of the iteration
    cycle_number: int  # Iteration cycle (1, 2, 3...)

    # Proposal details
    proposal_id: str  # Identifier of the tested proposal
    proposal_ranking: int  # Rank of proposal (1=top, 2=second, etc.)
    proposal_title: str  # Short title of the proposal

    # Evaluation results
    evaluator_verdict: str  # "PASS" | "FAIL" | "CONDITIONAL"
    sharpe_baseline: float  # Baseline Sharpe before experiment
    sharpe_tested: float  # Sharpe from backtest
    sharpe_delta: float  # sharpe_tested - sharpe_baseline
    dsr_baseline: float  # Baseline DSR
    dsr_tested: float  # DSR from backtest
    dsr_delta: float  # dsr_tested - dsr_baseline

    # Learning
    key_findings: str  # JSON or text summary of findings
    evaluator_notes: str  # Evaluator's reasoning for verdict
    next_action: str  # "proceed_to_production" | "revert" | "fix_and_retry" | "defer"

    # Status
    status: str  # "logged" | "error"
    error_msg: Optional[str] = None  # If status="error", reason why


def log_iteration_to_bq(
    project_id: str,
    log: IterationLog,
) -> bool:
    """Log iteration to BigQuery. Falls back to JSON file if unavailable.

    Args:
        project_id: GCP project ID
        log: IterationLog dataclass

    Returns:
        True if logged successfully (BQ or file), False if both failed
    """
    try:
        from google.cloud import bigquery

        bq = bigquery.Client(project=project_id)
        table_id = f"{project_id}.trading.harness_learning_log"

        rows = [asdict(log)]
        errors = bq.insert_rows_json(table_id, rows, timeout=10)

        if errors:
            logger.error("BigQuery insert errors: %s", errors)
            return False

        logger.info("Logged iteration %s to BigQuery (cycle %d)", log.iteration_id, log.cycle_number)
        return True

    except Exception as e:
        logger.warning("BigQuery logging failed: %s. Falling back to file.", e)

        # Fallback: JSON file
        try:
            filename = f"handoff/iteration_{log.iteration_id}.json"
            with open(filename, "w") as f:
                json.dump(asdict(log), f, indent=2, default=str)
            logger.info("Logged iteration %s to file: %s", log.iteration_id, filename)
            return True
        except Exception as e2:
            logger.error("File logging also failed: %s", e2)
            return False


def create_bq_table(project_id: str, dataset_id: str = "trading") -> None:
    """Create harness_learning_log table if not exists (idempotent).

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset (default: "trading")
    """
    try:
        from google.cloud import bigquery
        from google.cloud.bigquery import SchemaField

        bq = bigquery.Client(project=project_id)
        table_id = f"{project_id}.{dataset_id}.harness_learning_log"

        schema = [
            SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            SchemaField("iteration_id", "STRING", mode="REQUIRED"),
            SchemaField("cycle_number", "INTEGER", mode="REQUIRED"),
            SchemaField("proposal_id", "STRING", mode="REQUIRED"),
            SchemaField("proposal_ranking", "INTEGER", mode="REQUIRED"),
            SchemaField("proposal_title", "STRING", mode="REQUIRED"),
            SchemaField("evaluator_verdict", "STRING", mode="REQUIRED"),
            SchemaField("sharpe_baseline", "FLOAT64", mode="REQUIRED"),
            SchemaField("sharpe_tested", "FLOAT64", mode="REQUIRED"),
            SchemaField("sharpe_delta", "FLOAT64", mode="REQUIRED"),
            SchemaField("dsr_baseline", "FLOAT64", mode="REQUIRED"),
            SchemaField("dsr_tested", "FLOAT64", mode="REQUIRED"),
            SchemaField("dsr_delta", "FLOAT64", mode="REQUIRED"),
            SchemaField("key_findings", "STRING", mode="NULLABLE"),
            SchemaField("evaluator_notes", "STRING", mode="NULLABLE"),
            SchemaField("next_action", "STRING", mode="REQUIRED"),
            SchemaField("status", "STRING", mode="REQUIRED"),
            SchemaField("error_msg", "STRING", mode="NULLABLE"),
        ]

        table = bigquery.Table(table_id, schema=schema)

        # Partition by timestamp (daily), cluster by verdict + iteration_id
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp",
        )
        table.clustering_fields = ["evaluator_verdict", "iteration_id"]

        bq.create_table(table)
        logger.info("Created BigQuery table: %s", table_id)

    except Exception as e:
        logger.info("Table creation skipped (may already exist): %s", e)
