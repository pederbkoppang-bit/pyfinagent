"""
BigQuery schema for harness_learning_log table.

Stores every iteration of the autonomous loop for analysis and debugging.
"""

from google.cloud import bigquery


LEARNING_LOG_SCHEMA = [
    bigquery.SchemaField("iteration_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("start_time", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("end_time", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("duration_seconds", "INTEGER", mode="NULLABLE"),
    
    # Planner phase
    bigquery.SchemaField("planner_proposals", "STRING", mode="NULLABLE"),  # JSON array
    
    # Generator phase
    bigquery.SchemaField("selected_proposal", "STRING", mode="NULLABLE"),  # JSON object
    bigquery.SchemaField("backtest_results", "STRING", mode="NULLABLE"),  # JSON array
    
    # Evaluator phase
    bigquery.SchemaField("evaluator_verdict", "STRING", mode="NULLABLE"),  # PASS/FAIL/CONDITIONAL
    bigquery.SchemaField("sharpe_delta", "FLOAT64", mode="NULLABLE"),
    
    # Learning phase
    bigquery.SchemaField("learnings", "STRING", mode="NULLABLE"),  # JSON array of strings
    bigquery.SchemaField("iteration_metadata", "STRING", mode="NULLABLE"),  # JSON metadata
]


def create_learning_log_table(
    project_id: str,
    dataset_id: str = "trading",
    table_id: str = "harness_learning_log",
) -> bigquery.Table:
    """
    Create the harness_learning_log table if it doesn't exist.
    
    Args:
        project_id: GCP project ID
        dataset_id: Dataset to create table in
        table_id: Table name
    
    Returns:
        BigQuery table object
    """
    
    client = bigquery.Client(project=project_id)
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    
    try:
        # Try to get existing table
        table = client.get_table(full_table_id)
        print(f"✅ Table {full_table_id} already exists")
        return table
        
    except Exception:
        # Create new table
        table = bigquery.Table(full_table_id, schema=LEARNING_LOG_SCHEMA)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="start_time",
        )
        table.clustering_fields = ["evaluator_verdict", "iteration_id"]
        
        table = client.create_table(table)
        print(f"✅ Created table {full_table_id}")
        return table


if __name__ == "__main__":
    import os
    project_id = os.getenv("GCP_PROJECT_ID", "pyfinagent-prod")
    create_learning_log_table(project_id)
