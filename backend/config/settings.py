"""
Application settings loaded from environment variables.
Uses pydantic-settings for validation and .env file support.
"""
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from functools import lru_cache

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """All configuration is loaded from environment variables or a .env file."""

    # --- Application ---
    app_name: str = "PyFinAgent"
    debug: bool = False
    log_level: str = Field("INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR. Use WARNING for quiet terminals.")
    cost_tier: str = Field("build", description="Model cost tier: 'build' uses the development Opus/Sonnet mix, 'live' uses the launch-time cheap mapping from backend/config/model_tiers.py. Flip via COST_TIER env var at launch.")

    # --- GCP ---
    gcp_project_id: str = Field(..., description="Google Cloud Project ID")
    gcp_location: str = Field("us-central1", description="Vertex AI region")
    gcp_credentials_json: str = Field("", description="Service account JSON string (optional, falls back to ADC)")

    # --- Vertex AI ---
    gemini_model: str = Field("claude-sonnet-4-6", description="Standard-tier model for enrichment + debate. Claude is the default; Gemini still selectable via the Settings UI. Field name preserved for backward compat -- applies to any provider via backend/agents/llm_client.py::make_client routing.")
    deep_think_model: str = Field("claude-opus-4-7", description="Deep-think-tier model for Moderator/Critic/Synthesis/RiskJudge. Claude default. Gemini 2.5 Flash (gemini-2.5-flash) still selectable via the Settings UI.")
    apply_model_to_all_agents: bool = Field(False, description="phase-21.1: when true, override per-role models in model_tiers.resolve_model() with `gemini_model` (the Standard model selector) for ALL non-Gemini-locked roles. Gemini-only roles (RAG / Search Grounding / Vertex structured output) still use their hardcoded gemini-2.0-flash. Per-tier mas_main / mas_qa overrides are bypassed when this flag is true.")
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

    # --- News streaming adapters (phase-6.3) ---
    finnhub_api_key: str = Field("", description="Finnhub API token for market + company news (empty => adapter returns [])")
    benzinga_api_key: str = Field("", description="Benzinga API token for /api/v2/news (empty => adapter returns [])")
    # phase-25.B10: SecretStr-typed so repr() masks the value in logs / stack traces.
    alpaca_api_key_id: SecretStr = Field(SecretStr(""), description="Alpaca API Key ID for data.alpaca.markets/v1beta1/news (empty => adapter returns [])")
    alpaca_api_secret_key: SecretStr = Field(SecretStr(""), description="Alpaca API Secret Key for news endpoint (empty => adapter returns [])")

    # --- Sentiment scorer ladder (phase-6.5) ---
    sentiment_min_confidence: float = Field(0.7, description="Escalation threshold in [0,1]. VADER+FinBERT results below this floor escalate to the next rung (WASSA 2024 cascade operating point).")
    sentiment_use_gemini_flash: bool = Field(False, description="Enable opt-in tier-4 Gemini 2.5 Flash cross-check (default OFF; see phase-6.9 calibration plan).")
    sentiment_haiku_batch_mode: bool = Field(False, description="Route tier-3 Haiku 4.5 calls through Anthropic Batch API (50%% discount) when OK to wait; default OFF for real-time cron.")
    # phase-25.C9.1: backtest hot-path window-batching toggle. When True AND the orchestrator
    # is instantiated with backtest_mode=True AND n_tickers > 3, enrichment agents flow through
    # BatchClient (50%% flat discount + 1h-cache compounds to ~95%% effective). Default OFF so
    # the live single-ticker API path is never accidentally async.
    backtest_batch_mode: bool = Field(False, description="Route backtest-mode enrichment agents through Anthropic Batch API (50%% discount) when n_tickers > 3; default OFF.")

    # --- Observability / rate limits / alerting (phase-6.7) ---
    finnhub_rate_limit_rps: int = Field(25, description="Client-side RPS cap for Finnhub (server limit 30; keep 5 headroom).")
    benzinga_rate_limit_rps: int = Field(2, description="Client-side RPS cap for Benzinga news endpoint.")
    alpaca_rate_limit_rps: int = Field(30, description="Client-side RPS cap for Alpaca news endpoint.")
    fred_rate_limit_rps: int = Field(5, description="Client-side RPS cap for FRED API.")
    alphavantage_rate_limit_rps: int = Field(1, description="Client-side RPS cap for Alpha Vantage (free-tier is 5 req/min -> 1/sec worst case).")
    alert_consecutive_failure_threshold: int = Field(3, description="Consecutive failures per (source, error_type) required to fire an alert.")
    alert_debounce_minutes: int = Field(5, description="Sliding window (min) over which consecutive_failure_threshold is evaluated.")
    alert_repeat_hours: int = Field(1, description="Minimum hours between repeat alerts for the same (source, error_type).")
    bq_dataset_observability: str = Field("pyfinagent_data", description="BQ dataset for observability tables (llm_call_log, api_call_log, news_*, calendar_events).")

    # --- Regime detection (phase-3.3) ---
    regime_detection_enabled: bool = Field(False, description="Opt-in: use VIXRollingQuantileRegimeDetector in spot_checks_harness instead of the static pre/post-COVID fallback.")

    # --- Multi-Provider LLM Keys (v3.4) ---
    # phase-25.B10: SecretStr-typed so repr() masks values in logs / stack traces.
    anthropic_api_key: SecretStr = Field(SecretStr(""), description="Anthropic API key for direct Claude access (sk-ant-...)")
    openai_api_key: SecretStr = Field(SecretStr(""), description="OpenAI API key for direct GPT/o-series access (sk-...)")
    github_token: SecretStr = Field(SecretStr(""), description="GitHub PAT for GitHub Models (Copilot Pro). Routes GITHUB_MODELS_CATALOG models via models.inference.ai.azure.com")
    gemini_api_key: SecretStr = Field(SecretStr(""), description="Google AI Studio API key for direct Gemini access (genai.Client(api_key=...)). When set, gemini-* models route through the direct API instead of Vertex AI ADC. Leave empty to keep using Vertex AI / GCP service-account credentials.")

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
    # phase-25.C3: SR 11-7 paper-only gate. MUST remain False until a compliance
    # review wires the real-capital deployment path. Consumed by
    # backend/autoresearch/monthly_champion_challenger.py::run_monthly_sortino_gate.
    real_capital_enabled: bool = Field(False, description="SR 11-7 paper-only gate; toggling to True deploys approved strategies against real capital")
    paper_starting_capital: float = Field(10000.0, description="Initial virtual cash for paper portfolio")
    paper_max_positions: int = Field(10, description="Maximum simultaneous open positions")
    # phase-23.1.13: hard cap per GICS sector (default 2 = at least 5 sectors for a 10-position portfolio).
    # Mirrors SEC 1940 Act "concentrated" threshold (25% NAV per sector) when paired with 10% per-position cap.
    # 0 disables the cap entirely (legacy behavior); the existing 11/11-Technology bug is the documented failure mode.
    paper_max_per_sector: int = Field(2, ge=0, le=20, description="Maximum BUY positions in any single GICS sector. 0 = no limit (legacy).")
    paper_min_cash_reserve_pct: float = Field(5.0, description="Minimum cash reserve as % of NAV")
    paper_screen_top_n: int = Field(10, description="Number of candidates from quant screening")
    paper_analyze_top_n: int = Field(5, description="Number of candidates to deep-analyze per day")
    paper_trading_hour: int = Field(10, description="Hour (ET) to run daily trading cycle (0-23)")
    paper_reeval_frequency_days: int = Field(3, description="Re-evaluate existing holdings every N days")
    paper_transaction_cost_pct: float = Field(0.1, description="Simulated transaction cost per trade (%)")
    paper_max_daily_cost_usd: float = Field(2.0, description="Maximum LLM cost per daily trading cycle (USD)")
    # phase-27.5.2: daily/monthly cost-budget hard-block caps consumed by
    # backend.agents.llm_client._check_cost_budget. Default $5/day was hit on
    # 2026-05-16 cycle 3e90d15e at the 7-ticker mark of a 15-ticker concurrent-8
    # batch ($5.15 actual). Raised to $25 / $300 — still conservative for an
    # autonomous LLM trading system (Gemini Flash + AI Studio direct keys keep
    # full-cycle costs in the $1-3 range). Env overrides honored.
    cost_budget_daily_usd: float = Field(25.0, description="Daily LLM-spend cap across all cycles (USD). When tripped, _check_cost_budget raises BudgetBreachError on every new generate_content call until midnight UTC rolls the bucket.")
    cost_budget_monthly_usd: float = Field(300.0, description="Monthly LLM-spend cap across all cycles (USD).")
    # phase-26.2: opt-in flag for the Anthropic Advisor Tool (Sonnet executor + Opus advisor)
    # on the synthesis chain. Default False (this step adds capability; flip to True
    # via operator-driven rollout after A/B regression check). When True AND the
    # configured synthesis model is claude-opus-4-*, run_synthesis_pipeline routes
    # through advisor_call() instead of generate_content(). See research_brief.md.
    enable_advisor_tool: bool = Field(False, description="Enable Anthropic Advisor Tool (Sonnet 4.6 executor + Opus 4.7 advisor) on Opus-based synthesis chain")
    # phase-23.1.1: macro regime filter (LLM-as-judge over FRED snapshot)
    macro_regime_filter_enabled: bool = Field(False, description="Apply daily macro regime as a conviction multiplier in screener rank_candidates")
    macro_regime_model: str = Field("claude-haiku-4-5", description="LLM used for daily macro regime classification")
    # phase-23.1.2: earnings PEAD overlay (free SEC EDGAR + Claude sentiment-surprise)
    pead_signal_enabled: bool = Field(False, description="Fetch SEC 8-K + Claude PEAD signals for tickers that reported in the last 7 days; apply boost/filter in screener")
    pead_signal_model: str = Field("claude-haiku-4-5", description="LLM used for PEAD sentiment scoring on press-release text")
    pead_signal_lookback_quarters: int = Field(12, description="phase-28.2: Trailing quarters of PEAD sentiment used to compute surprise (bumped 8->12 per ScienceDirect 2025 SUE-stacking paper, +85% Sharpe lift; equal-weighted mean)")
    # phase-23.1.3: worldwide news idea generator (no-API-key RSS + Claude batch event extractor)
    news_screen_enabled: bool = Field(False, description="Pull worldwide RSS news + Claude classifier; surface positive-polarity tickers as parallel candidates in screener")
    news_screen_model: str = Field("claude-haiku-4-5", description="LLM used for batch news event extraction")
    news_screen_max_headlines: int = Field(100, description="Max deduped headlines per cycle sent to the LLM (caps cost)")
    # phase-23.1.4: sector event calendars (FDA PDUFA + upcoming earnings; pure data-pull)
    sector_calendars_enabled: bool = Field(False, description="Pull FDA PDUFA + earnings calendars; boost catalyst tickers and filter ticker on day-of binary FDA event")
    sector_calendars_lookahead_days: int = Field(7, description="Lookahead window for upcoming earnings in BQ calendar query")
    # phase-23.1.5: LLM-as-judge meta-scorer (single batched Claude call for conviction 1-10 per candidate)
    meta_scorer_enabled: bool = Field(False, description="After multiplicative overlays, call Claude once over top-30 candidates with all sub-signals; conviction_score replaces composite_score for ranking")
    meta_scorer_model: str = Field("claude-haiku-4-5", description="LLM used for the meta-scorer batch call")
    meta_scorer_max_batch: int = Field(30, description="Max candidates sent to the meta-scorer in one batch (cap to bound LLM cost + cross-contamination)")
    # phase-28.5: short-interest exclusion filter (FINRA bimonthly CSV primary, yfinance per-ticker fallback)
    short_interest_filter_enabled: bool = Field(False, description="phase-28.5: Exclude tickers with shortPercentOfFloat > short_interest_threshold from screener. Boehmer-Jones-Zhang 2008: high-short stocks underperform 1.16%/mo. Default OFF.")
    short_interest_threshold: float = Field(0.10, description="phase-28.5: shortPercentOfFloat cutoff above which a ticker is excluded (default 10% = approximate top-decile for S&P 500 large-caps).")
    short_interest_cache_days: int = Field(14, description="phase-28.5: Days to cache the FINRA bimonthly CSV before re-downloading (FINRA publishes bimonthly, so 14 days matches their cadence).")
    # phase-28.1: analyst EPS revision-breadth overlay (yfinance Ticker.upgrades_downgrades)
    analyst_revisions_enabled: bool = Field(False, description="phase-28.1: Apply analyst revision-breadth multiplier in screener rank_candidates. Mill Street Research 19yr: t=2.93, Sharpe~1.60 combined with momentum. Default OFF.")
    analyst_revisions_lookback_days: int = Field(100, description="phase-28.1: Lookback window (days) for revision-breadth count. Mill Street canonical: 100. Shorter (30-60) is faster but lower IC.")
    analyst_revisions_min_analysts: int = Field(3, description="phase-28.1: Minimum up+down grade actions in the lookback window for the signal to fire (statistical-noise guard).")
    analyst_revisions_threshold: float = Field(0.10, description="phase-28.1: |breadth| above which the multiplier fires (deadband when |breadth| <= threshold). breadth = (n_up - n_down) / (n_up + n_down).")
    analyst_revisions_weight: float = Field(0.15, description="phase-28.1: Multiplier intensity. score *= (1 + breadth * weight). 0.15 means a full +1.0 breadth yields a +15% score boost.")
    # phase-28.3: GPR (Geopolitical Risk Acts) sector-tilt trigger (Caldara-Iacoviello)
    gpr_signal_enabled: bool = Field(False, description="phase-28.3: When latest GPR-Acts > quantile threshold, inject configured energy ETFs into macro_regime.sector_hints.overweight. Caldara-Iacoviello AER 2022 + IMF GFSR 2025: US-as-net-exporter asymmetry favors XOM/CVX/COP/OXY on Middle-East GPR spikes. Default OFF.")
    gpr_signal_quantile: float = Field(0.90, description="phase-28.3: Quantile of rolling 5-year GPRA history above which the tilt triggers (0.90 = 90th percentile = ~120-145 absolute in historical baseline).")
    gpr_signal_cache_hours: int = Field(24, description="phase-28.3: Hours to cache the downloaded GPR Excel before refetching. Matches matteoiacoviello.com ~monthly publication cadence (24h is generous).")
    gpr_signal_sector_etfs: str = Field("XLE", description="phase-28.3: Comma-separated list of sector ETF tickers to add to sector_hints.overweight when GPR-Acts crosses threshold. Default XLE (energy SPDR).")
    # phase-28.6: WTI crude (CL=F) momentum secondary trigger (orthogonal to GPR-Acts)
    crude_momentum_enabled: bool = Field(False, description="phase-28.6: When WTI crude (CL=F) 1m momentum z-score exceeds threshold, inject configured energy ETFs into macro_regime.sector_hints.overweight. Orthogonal to phase-28.3 GPR trigger (high-GPR/flat-oil and rising-oil/low-GPR both occur). Default OFF.")
    crude_momentum_window_days: int = Field(21, description="phase-28.6: Trading-day window for 1-month momentum (21 = approximately 1 calendar month).")
    crude_momentum_lookback_days: int = Field(252, description="phase-28.6: Trading-day lookback for z-score normalization of the 1m momentum (252 = 1 trading year).")
    crude_momentum_zscore_threshold: float = Field(1.0, description="phase-28.6: Z-score threshold above which the trigger fires. 1.0 = ~84th percentile under a normal assumption (calibrated for ~78% crude implied vol).")
    crude_momentum_cache_hours: int = Field(24, description="phase-28.6: Hours to cache the yfinance CL=F fetch before re-downloading.")
    crude_momentum_sector_etfs: str = Field("XLE", description="phase-28.6: Comma-separated list of sector ETF tickers to add to sector_hints.overweight when crude momentum crosses threshold. Default XLE (XOM+CVX = 39% of XLE).")
    # phase-28.4: sector-neutral momentum scoring (within-sector percentile rank)
    sector_neutral_momentum_enabled: bool = Field(False, description="phase-28.4: When True, rank_candidates replaces composite_score with within-sector percentile rank (pandas Series.rank pct). Improves Sharpe + reduces regime sensitivity per CFA Institute Dec 2025. Default OFF.")
    sector_neutral_min_group_size: int = Field(3, description="phase-28.4: Minimum candidates per sector to apply within-sector percentile rank. Smaller groups + missing-sector stocks fall back to a global cross-sector percentile pool.")
    # phase-28.12: sector-ETF momentum overlay (Quantpedia top-3 rotation)
    sector_momentum_enabled: bool = Field(False, description="phase-28.12: When True, boost composite_score for candidates in top-N momentum sectors (12m return on 11 SPDR ETFs). Quantpedia: top-3 monthly rotation 13.94%/yr Sharpe 0.54 +4%/yr vs passive. Default OFF.")
    sector_momentum_lookback_months: int = Field(12, description="phase-28.12: Trailing months used to compute sector ETF total return for ranking. Canonical: 12 months.")
    sector_momentum_top_n: int = Field(3, description="phase-28.12: Number of top sectors that receive the boost. Canonical: 3 (top-quartile of 11 GICS sectors).")
    sector_momentum_boost_top: float = Field(1.10, description="phase-28.12: Multiplier for candidates in top-N sectors (default: +10%).")
    sector_momentum_boost_leader: float = Field(1.15, description="phase-28.12: Multiplier for candidates in the #1 momentum sector (default: +15%).")
    sector_momentum_cache_hours: int = Field(24, description="phase-28.12: Hours to cache the yfinance sector-ETF batch fetch.")
    # 4.5.7 Kill-switch v2. Defaults from prop-trading practitioner consensus
    # (see RESEARCH.md Phase 4.5 step 4.5.7): 4% daily loss modal across FTMO /
    # FXIFY / Alpha Capital / FundedNext; 10% EOD trailing drawdown is the
    # upper-end standard for long-only unlevered equity.
    paper_daily_loss_limit_pct: float = Field(4.0, description="Halt trading if intraday loss exceeds this %% of start-of-day NAV")
    paper_trailing_dd_limit_pct: float = Field(10.0, description="Halt trading if trailing drawdown from peak equity exceeds this %% (EOD)")
    # phase-23.1.8: per-ticker stop-loss safety net for the lite Claude analyzer
    # path which does not propose its own stop_loss. O'Neil canonical 7-8%; quant
    # backtest evidence (quant-investing.com 85-year study) shows a 10% momentum
    # stop reduces max monthly loss from -49.79% to -11.34% with average return
    # 1.01% -> 1.73%. 8% is the documented sweet spot.
    paper_default_stop_loss_pct: float = Field(
        8.0,
        ge=1.0,
        le=50.0,
        description="Default stop-loss as %% below entry price when analysis does not provide one (lite-path BUY). O'Neil canonical: 7-8%.",
    )

    # --- Authentication ---
    # phase-25.B10: SecretStr-typed so repr() masks the secret in logs / stack traces.
    auth_secret: SecretStr = Field(SecretStr(""), description="NextAuth.js AUTH_SECRET for JWE decryption. Empty = auth disabled (dev mode).")
    allowed_emails: str = Field("", description="Comma-separated email whitelist. Empty = allow all authenticated users.")

    # --- Slack Bot ---
    # phase-25.B10: SecretStr-typed so repr() masks tokens in logs / stack traces.
    slack_bot_token: SecretStr = Field(SecretStr(""), description="Slack Bot User OAuth Token (xoxb-...)")
    slack_app_token: SecretStr = Field(SecretStr(""), description="Slack App-Level Token for Socket Mode (xapp-...)")
    slack_channel_id: str = Field("", description="Slack channel ID for proactive alerts and digests")
    morning_digest_hour: int = Field(8, description="Hour (0-23) for daily morning digest in local timezone")
    evening_digest_hour: int = Field(17, description="Hour (0-23) for daily evening digest in local timezone")
    watchdog_interval_minutes: int = Field(15, description="Interval (minutes) for watchdog health check")

    # --- First-week monitoring (Phase 4.4.6.3) ---
    first_week_mode: bool = Field(False, description="Tighten alert thresholds for 7 days post-launch. Drawdown de-risk: -10% -> -5%, SLA P3 response: 4h -> 1h. Set FIRST_WEEK_MODE=true at go-live, revert after day 7.")
    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # pydantic-settings loads from env/.env
