# ingestion-agent/utils/bigquery_utils.py (Refactored)
from google.cloud import bigquery
import logging

logging.basicConfig(level=logging.INFO)

# Define the schema explicitly for the Raw_Market_Data table
# We use FLOAT64 for Volume to allow NaNs in the staging area, preserving raw data integrity.
MARKET_STAGING_SCHEMA = [
    bigquery.SchemaField("API_Source", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("Ticker", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("Timestamp_UTC", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("Open", "FLOAT64"),
    bigquery.SchemaField("High", "FLOAT64"),
    bigquery.SchemaField("Low", "FLOAT64"),
    bigquery.SchemaField("Close", "FLOAT64"),
    bigquery.SchemaField("Volume", "FLOAT64"), # Changed to FLOAT64 to accept NaNs
    bigquery.SchemaField("Bid_Close", "FLOAT64"), # Include even if often NULL
    bigquery.SchemaField("Ask_Close", "FLOAT64"), # Include even if often NULL
    bigquery.SchemaField("Dividends", "FLOAT64"),
    bigquery.SchemaField("Stock_Splits", "FLOAT64"),
    bigquery.SchemaField("Ingestion_Timestamp", "TIMESTAMP"),
]

def load_data_to_bigquery(dataframe, project_id, dataset_id, table_id, schema_type="MARKET"):
    """
    LOAD (L): Loads a Pandas DataFrame into the specified BigQuery staging table.
    """
    client = bigquery.Client(project=project_id)
    table_ref = client.dataset(dataset_id).table(table_id)
    
    load_df = dataframe.copy()

    # Data preparation specific to the schema
    if schema_type == "MARKET":
        schema = MARKET_STAGING_SCHEMA
        # Ensure L1 columns exist if they might be missing from the fetcher (e.g., yfinance)
        if 'Bid_Close' not in load_df.columns:
            load_df['Bid_Close'] = None
        if 'Ask_Close' not in load_df.columns:
            load_df['Ask_Close'] = None
    # Add other schema types (MACRO, FUNDAMENTAL) as needed
    else:
        raise ValueError(f"Unknown schema type: {schema_type}")

    # Configure the load job
    job_config = bigquery.LoadJobConfig()
    # CRITICAL: Use WRITE_APPEND for the staging layer
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.schema = schema

    load_job = None
    try:
        load_job = client.load_table_from_dataframe(
            load_df, table_ref, job_config=job_config
        )
        logging.info(f"Starting BigQuery load job {load_job.job_id}...")
        load_job.result()  # Wait for the job to complete
        logging.info(f"Successfully loaded {len(load_df)} rows to {dataset_id}.{table_id}.")
        
    except Exception as e:
        logging.error(f"Error loading data to BigQuery: {e}")
        # If the load fails, inspect the errors
        if load_job and hasattr(load_job, 'errors') and load_job.errors:
            logging.error("Load job errors:")
            for error in load_job.errors:
                logging.error(error)
        raise