"""
Application settings loaded from environment variables.
Uses pydantic-settings for validation and .env file support.
"""
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """All configuration is loaded from environment variables or a .env file."""

    # --- Application ---
    app_name: str = "PyFinAgent"
    debug: bool = False

    # --- GCP ---
    gcp_project_id: str = Field(..., description="Google Cloud Project ID")
    gcp_location: str = Field("us-central1", description="Vertex AI region")
    gcp_credentials_json: str = Field("", description="Service account JSON string (optional, falls back to ADC)")

    # --- Vertex AI ---
    gemini_model: str = Field("gemini-2.0-flash", description="Gemini model name for standard agents")
    deep_think_model: str = Field("", description="Model for deep-think agents (Moderator, Risk Judge, Synthesis, Critic). Defaults to gemini_model if empty.")
    rag_data_store_id: str = Field(..., description="Vertex AI Search datastore ID")

    # --- BigQuery ---
    bq_dataset_reports: str = "financial_reports"
    bq_table_reports: str = "analysis_results"
    bq_dataset_portfolio: str = "pyfinagent_pms"
    bq_dataset_outcomes: str = "financial_reports"
    bq_table_outcomes: str = "outcome_tracking"

    # --- Cloud Function Agent URLs ---
    ingestion_agent_url: str = Field(..., description="Ingestion agent Cloud Function URL")
    quant_agent_url: str = Field(..., description="Quant agent Cloud Function URL")

    # --- External APIs ---
    alphavantage_api_key: str = Field("", description="Alpha Vantage API key")
    slack_webhook_url: str = Field("", description="Slack webhook URL for notifications")
    fred_api_key: str = Field("", description="FRED (Federal Reserve) API key")
    api_ninjas_key: str = Field("", description="API Ninjas key for earnings transcripts")
    patentsview_api_key: str = Field("", description="PatentsView API key (free at patentsview.org)")

    # --- GCS ---
    gcs_bucket_name: str = Field("10k-filling-data", description="GCS bucket for filings and transcripts")

    # --- Execution Mode ---
    use_celery: bool = Field(False, description="Use Celery+Redis for async tasks. When False, runs analysis synchronously.")

    # --- Redis / Celery ---
    redis_url: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    celery_broker_url: str = Field("redis://localhost:6379/0", description="Celery broker URL")
    celery_result_backend: str = Field("redis://localhost:6379/1", description="Celery result backend URL")

    # --- Score Weights (configurable) ---
    weight_corporate: float = 0.35
    weight_industry: float = 0.20
    weight_valuation: float = 0.20
    weight_sentiment: float = 0.15
    weight_governance: float = 0.10

    # --- Debate Depth (configurable) ---
    max_debate_rounds: int = Field(2, description="Number of Bull↔Bear exchange rounds (1-5)")
    max_risk_debate_rounds: int = Field(1, description="Number of Agg/Con/Neu risk debate rounds (1-3)")

    # --- Quality Gate Thresholds (configurable) ---
    data_quality_min: float = Field(0.5, description="Minimum data quality score (0-1) before skipping debate/risk steps")
    conflict_escalation_threshold: int = Field(5, description="Conflict count that triggers +1 debate round")
    critic_major_issues_threshold: int = Field(3, description="Major Critic issues that trigger deep-think re-synthesis")

    # --- Cost Controls ---
    lite_mode: bool = Field(False, description="Cost-saving mode: skips deep dive, devil's advocate, reflection loop, and risk assessment (~50% fewer LLM calls)")
    max_analysis_cost_usd: float = Field(0.50, description="Soft budget per analysis in USD. Logs warnings when exceeded; does not abort.")
    max_synthesis_iterations: int = Field(2, ge=1, le=3, description="Maximum Synthesis↔Critic reflection loop iterations (1=no reflection)")
    # --- Authentication ---
    auth_secret: str = Field("", description="NextAuth.js AUTH_SECRET for JWE decryption. Empty = auth disabled (dev mode).")
    allowed_emails: str = Field("", description="Comma-separated email whitelist. Empty = allow all authenticated users.")

    # --- Slack Bot ---
    slack_bot_token: str = Field("", description="Slack Bot User OAuth Token (xoxb-...)")
    slack_app_token: str = Field("", description="Slack App-Level Token for Socket Mode (xapp-...)")
    slack_channel_id: str = Field("", description="Slack channel ID for proactive alerts and digests")
    morning_digest_hour: int = Field(8, description="Hour (0-23) for daily morning digest in local timezone")
    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
