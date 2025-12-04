# ingestion-agent/config.py (Refactored)
import os

# BigQuery Configuration
# Set this in your Cloud Function environment variables
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "sunny-might-477607-p8") 
# Changed to the staging dataset
DATASET_ID = "pyfinagent_staging" 

# Table IDs
MARKET_TABLE_ID = "Raw_Market_Data"
MACRO_TABLE_ID = "Raw_Macro_Data"
FUNDAMENTAL_TABLE_ID = "Raw_Fundamental_Data"

# Data Fetching Configuration
START_DATE = "2010-01-01" # Define the historical period for backfill

# API Source Metadata
API_SOURCE_MARKET = "yfinance" # Adjust as needed