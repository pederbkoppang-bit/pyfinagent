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
    log_level: str = Field("INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR. Use WARNING for quiet terminals.")

    # --- GCP ---
    gcp_project_id: str = Field(..., description="Google Cloud Project ID")
    gcp_location: str = Field("us-central1", description="Vertex AI region")
    gcp_credentials_json: str = Field("", description="Service account JSON string (optional, falls back to ADC)")

    # --- Vertex AI ---
    gemini_model: str = Field("gemini-2.0-flash", description="Gemini model name for standard agents")
    deep_think_model: str = Field("gemini-2.5-flash", description="Model for deep-think agents (Moderator, Risk Judge, Synthesis, Critic). Recommend gemini-2.5-flash for extended thinking.")
    enable_thinking: bool = Field(False, description="Enable extended thinking on judge agents (requires gemini-2.5-flash or later)")
    thinking_budget_critic: int = Field(8192, description="Thinking budget for Critic agent (tokens)")
    thinking_budget_moderator: int = Field(8192, description="Thinking budget for Moderator agent (tokens)")
    thinking_budget_risk_judge: int = Field(4096, description="Thinking budget for Risk Judge agent (tokens)")
    thinking_budget_synthesis: int = Field(4096, description="Thinking budget for Synthesis agent (tokens)")
    rag_data_store_id: str = Field(..., description="Vertex AI Search datastore ID")

    # --- BigQuery ---
    bq_dataset_reports: str = "financial_reports"
    bq_table_reports: str = "analysis_results"
    bq_dataset_portfolio: str = "pyfinagent_pms"
    bq_dataset_outcomes: str = "financial_reports"
    bq_table_outcomes: str = "outcome_tracking"

    # --- Multi-Market (Phase 2.9) ---
    default_market: str = Field("US", description="Default market code (US, NO, CA, DE, KR)")
    base_currency: str = Field("USD", description="Base currency for portfolio returns")

    # --- Cloud Function Agent URLs ---
    ingestion_agent_url: str = Field(..., description="Ingestion agent Cloud Function URL")
    quant_agent_url: str = Field(..., description="Quant agent Cloud Function URL")

    # --- External APIs ---
    alphavantage_api_key: str = Field("", description="Alpha Vantage API key")
    slack_webhook_url: str = Field("", description="Slack webhook URL for notifications")
    fred_api_key: str = Field("", description="FRED (Federal Reserve) API key")
    api_ninjas_key: str = Field("", description="API Ninjas key for earnings transcripts")
    patentsview_api_key: str = Field("", description="PatentsView API key (free at patentsview.org)")

    # --- Multi-Provider LLM Keys (v3.4) ---
    anthropic_api_key: str = Field("", description="Anthropic API key for direct Claude access (sk-ant-...)")
    openai_api_key: str = Field("", description="OpenAI API key for direct GPT/o-series access (sk-...)")
    github_token: str = Field("", description="GitHub PAT for GitHub Models (Copilot Pro). Routes GITHUB_MODELS_CATALOG models via models.inference.ai.azure.com")

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
    # --- Backtest ---
    backtest_start_date: str = Field("2018-01-01", description="Walk-forward backtest start date")
    backtest_end_date: str = Field("2025-12-31", description="Walk-forward backtest end date")
    backtest_train_window_months: int = Field(12, description="Initial training window in months (expanding)")
    backtest_test_window_months: int = Field(3, description="Test window in months")
    backtest_embargo_days: int = Field(5, description="Embargo gap between train/test (trading days)")
    backtest_holding_days: int = Field(90, description="Triple Barrier time barrier (days)")
    backtest_tp_pct: float = Field(10.0, description="Triple Barrier take-profit %")
    backtest_sl_pct: float = Field(10.0, description="Triple Barrier stop-loss %")
    backtest_frac_diff_d: float = Field(0.4, description="Fractional differentiation order (0.1-0.8)")
    backtest_target_vol: float = Field(0.15, description="Inverse-vol sizing: target annualized volatility")
    backtest_top_n_candidates: int = Field(50, description="Top N candidates per screening pass")
    backtest_starting_capital: float = Field(100_000.0, description="Backtest starting capital ($)")
    backtest_max_positions: int = Field(20, description="Maximum simultaneous positions")
    backtest_transaction_cost_pct: float = Field(0.1, description="Simulated transaction cost per trade (%)")
    backtest_commission_model: str = Field("flat_pct", description="Commission model: flat_pct or per_share")
    backtest_commission_per_share: float = Field(0.005, description="Per-share commission when model=per_share ($)")

    # --- Paper Trading ---
    paper_trading_enabled: bool = Field(False, description="Enable autonomous paper trading scheduler")
    paper_starting_capital: float = Field(10000.0, description="Initial virtual cash for paper portfolio")
    paper_max_positions: int = Field(10, description="Maximum simultaneous open positions")
    paper_min_cash_reserve_pct: float = Field(5.0, description="Minimum cash reserve as % of NAV")
    paper_screen_top_n: int = Field(10, description="Number of candidates from quant screening")
    paper_analyze_top_n: int = Field(5, description="Number of candidates to deep-analyze per day")
    paper_trading_hour: int = Field(10, description="Hour (ET) to run daily trading cycle (0-23)")
    paper_reeval_frequency_days: int = Field(3, description="Re-evaluate existing holdings every N days")
    paper_transaction_cost_pct: float = Field(0.1, description="Simulated transaction cost per trade (%)")
    paper_max_daily_cost_usd: float = Field(2.0, description="Maximum LLM cost per daily trading cycle (USD)")

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
    return Settings()  # type: ignore[call-arg]  # pydantic-settings loads from env/.env
