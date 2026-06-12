"""
Application settings loaded from environment variables.
Uses pydantic-settings for validation and .env file support.
"""
from pathlib import Path
from typing import Annotated

from pydantic_settings import BaseSettings, NoDecode
from pydantic import Field, SecretStr, field_validator
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
    deep_think_model: str = Field("gemini-2.5-pro", description="Deep-think-tier model for Moderator/Critic/Synthesis/RiskJudge. phase-37.2: default aligned to production (gemini-2.5-pro on Vertex AI via existing operator ADC). Previously claude-opus-4-7 -- caused silent regression to Anthropic credit-exhaustion on fresh checkout / restart without DEEP_THINK_MODEL env override. Operator .env DEEP_THINK_MODEL=gemini-2.5-pro is now redundant but harmless. Other Gemini models (gemini-2.5-flash, gemini-3.0-flash, gemini-3.0-pro) still selectable via Settings UI.")
    paper_cycle_max_seconds: float = Field(7200.0, description="phase-34.2 corrective + cycle-7 (38.12) bump: hard wall-clock budget for one autonomous paper-trading cycle. Read by backend/services/autonomous_loop.py:219 via asyncio.timeout. Default raised from 1800 -> 7200 (2h) because cycle 6 (2026-05-26) found the Claude Code CLI rail (paper_use_claude_code_route=True; ~30s per claude_code_invoke) + serial enrichment-debate-risk-synthesis dependencies push a 13-ticker full-orchestrator cycle past 3600s. Cycle 6 timed out with 7 of 13 tickers analyzed; 7200s gives headroom for the full 13 + Step 6-9 (trade decide / execute / snapshot / outcome). When `paper_use_claude_code_route=False` AND Anthropic-direct rail is available, the lower 1800s remains adequate -- operator can lower via Settings UI.")
    paper_learn_loop_enabled: bool = Field(False, description="phase-35.1: gate the learn-loop writer fan-out in autonomous_loop._learn_from_closed_trades. When True, after a closed_ticker SELL is observed, the dispatcher (a) calls evaluate_recommendation (existing path -- writes outcome_tracking row on success), and (b) calls _generate_and_persist_reflections to write agent_memories lesson rows. Also activates fallback writer when evaluate_recommendation early-returns (yfinance flake on current_price). Default OFF per /goal integration gate 3 -- operator flips to true to enable the learn loop. BQ tables already exist (outcome_tracking + agent_memories); no migration needed.")
    paper_scale_out_enabled: bool = Field(False, description="phase-36.1: gate the scale-out helper in paper_trader.check_scale_out_fires (called from autonomous_loop after Step 5 mark_to_market). When True and a position's MFE >= 2*R (R = paper_default_stop_loss_pct, e.g. 8%), the system fires execute_sell(qty=quantity*0.5, reason='take_profit_2R'). At MFE >= 3*R the remainder is closed (reason='take_profit_3R'). Idempotent via scale_out_levels_hit JSON column on paper_positions (NULL/empty -> not yet fired; populated post-fire). Default OFF per /goal integration gate 3. Migration: scripts/migrations/add_scale_out_levels_hit_column.py (--verify exits 0 on already-applied).")
    apply_model_to_all_agents: bool = Field(False, description="phase-21.1: when true, override per-role models in model_tiers.resolve_model() with `gemini_model` (the Standard model selector) for ALL non-Gemini-locked roles. Gemini-only roles (RAG / Search Grounding / Vertex structured output) still use the hardcoded Gemini workhorse pin (model_tiers.GEMINI_WORKHORSE; phase-60.1). Per-tier mas_main / mas_qa overrides are bypassed when this flag is true.")
    ticket_ingestion_silence_days: int = Field(
        7,
        ge=1,
        le=60,
        description="phase-60.4 (criterion 2): dead-man's-switch threshold for the inbound-ticket ingestion pipeline. If tickets.db has ingested ZERO tickets for this many days, the slack-bot watchdog fires an Ingestion Silence Alarm (state-transition gated, never spams). The 2026-04-24 -> 06-10 six-week outage (#5100 -> #5101 gap) went unnoticed until the 59.3 audit; default 7 days catches that class within a week.",
    )
    paper_data_integrity_enabled: bool = Field(
        False,
        description="phase-60.3 (AW-9): decision-input integrity flag, DEFAULT OFF (do-no-harm). When ON, for non-US tickers in the lite analyzers: (1) prompts present USD-converted values via fx_rates.get_fx_rate (or block when FX is unavailable) instead of '$'-labeling raw KRW magnitudes (the away week rendered 'Market Cap: $1630000.0B' -- $1.63 quadrillion -- and the risk judge's correct 'KRW/USD unit error' prose flag was ignored while the BUY executed, 066570.KS 06-09 stopped out -9.7%); (2) blocking integrity flags from the deterministic pre-check (market cap > $10T post-normalization; currency unverified/mismatch) EXCLUDE the candidate IN CODE pre-LLM (GuardAgent chokepoint pattern -- prose-only flagging is the documented anti-pattern); (3) prompts carry the quote's as-of timestamp (KRX closes 06:30 UTC; away-week analyses ran on ~11.6h-stale quotes presented as live). US prompts byte-identical in BOTH flag states. Additive market_data provenance fields (currency/price_usd/market_cap_usd/fx_rate/as_of/integrity_flags) are UNGATED observability. Promotion to ON is an OPERATOR decision recorded in live_check_60.3.md.",
    )
    fallback_alarm_threshold: float = Field(0.5, description="phase-60.1 (AW-4): cycle-level full->lite fallback-rate alarm threshold. When the fraction of analyses that INTENDED the full orchestrator but landed on the lite fallback strictly exceeds this value, a P1 'fallback_rate' cron alert fires naming per-ticker failure reasons (wired beside the 56.2 degraded-scoring guard in backend/services/autonomous_loop.py). The away week ran 9 days at 100% fallback (retired gemini-2.0-flash pin + KR SEC-CIK aborts) with zero alerts. Deliberate lite_mode analyses are never counted.")
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
    # phase-50.3: markets the LIVE paper loop screens/trades. Default ['US'] is
    # byte-identical to today. Add 'EU'/'KR' to go live international -- only
    # AFTER the 50.5 data-quality gate (never trade unguarded intl data).
    # phase-54.1: Annotated[..., NoDecode] disables pydantic-settings' built-in
    # JSON decode of this complex field so the validator below handles EVERY load
    # path. The cron wrappers `set -a; . backend/.env` bash-source the .env, which
    # strips the JSON quotes (["US","EU","KR"] -> [US,EU,KR]); the default decoder
    # raised SettingsError on that, crashing autoresearch + ablation nightly. The
    # validator accepts JSON, bracket-mangled, and plain-comma forms identically.
    paper_markets: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["US"],
        description="phase-50.3: live-loop markets (subset of US/EU/KR). Default ['US'] = byte-identical.",
    )

    @field_validator("paper_markets", mode="before")
    @classmethod
    def _parse_paper_markets(cls, v):
        """phase-54.1: accept JSON (["US","EU","KR"]), bracket-mangled ([US,EU,KR]
        from a bash-sourced .env), plain comma (US,EU,KR), or a real list -- so
        get_settings() succeeds on the native-dotenv, OS-env, AND shell-sourced
        paths identically. Purely additive: the live JSON path yields the same list
        it always did. Empty/None -> the ['US'] default."""
        if v is None:
            return ["US"]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return ["US"]
            if s.startswith("["):
                try:
                    import json
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except ValueError:
                    pass  # bash-mangled [US,EU,KR] -> fall through to comma split
            return [tok.strip().strip('"').strip("'") for tok in s.strip("[]").split(",") if tok.strip()]
        return v

    # phase-53.1: transaction-cost-aware no-trade / rebalance buffer band (quant
    # elevation lever). Default OFF -> byte-identical (full reconstitution). The
    # band helper lives in backend/backtest/rebalance_band.py; phase-53.1 is
    # measure-first (NO live decide_trades wiring / NO live flag flip here -- the
    # live enable is a SEPARATE operator-gated step after the $0 replay verdict).
    rebalance_band_enabled: bool = Field(False, description="phase-53.1: enable the no-trade rebalance band on the monthly reconstitution. Default OFF = byte-identical full reconstitution. Live wiring deferred (measure-first).")
    rebalance_band_pct: float = Field(0.2, ge=0.0, le=1.0, description="phase-53.1: hysteresis band width; a held name is retained while its rank < top_n*(1+band_pct). Only consulted when rebalance_band_enabled.")

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
    # phase-cycle-3 (2026-05-26): operator-approved testing-phase rail. When
    # True, the autonomous-loop's Claude analysis calls route through the
    # `claude` CLI subprocess (`claude --print --output-format json ...`)
    # which uses the Max-subscription flat-fee auth at ~/.claude/ instead
    # of api.anthropic.com direct billing. Bypasses credit-exhaustion
    # failures during testing. Default False -- existing Anthropic-direct
    # path preserved. Operator opt-in via /api/settings/ PUT.
    # Citations (research_brief_phase_claude_code_routing.md):
    # - TradingAgents arXiv:2412.20138 (LLM-rail abstraction in production
    #   multi-agent trading systems).
    # - Portkey AI Gateway (10B+ req/mo failover-routing canonical).
    # - Bailey/Borwein/Lopez de Prado/Zhu PBO SSRN:2326253 (engine-change
    #   logging required for A/B integrity).
    # - Yin et al. arXiv:2603.20319 (per-row implementation-risk logging).
    paper_use_claude_code_route: bool = Field(
        False,
        description="Route Claude analysis calls through the `claude` CLI subprocess (Max-subscription rail) instead of api.anthropic.com direct billing. Testing-phase only; flip to False before flipping real_capital_enabled to True.",
    )
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
    max_analysis_cost_usd: float = Field(5.00, description="Soft budget per analysis in USD (cost-tracker NOMINAL, not metered). Logs warnings when exceeded; does not abort. phase-60.4 (AW-10): RE-SPECCED 0.50 -> 5.00 per operator decision 2026-06-11 (verbatim: 'RE-SPEC to $5.00 (Recommended)') -- the 0.50 limit predated the restored full pipeline (measured nominal $1.08-4.06 per full analysis; the away week logged '$4.3262 > $0.50' daily) and the nominal number prices flat-fee CC-rail tokens at API rates, so enforcement here would abort full analyses on phantom costs. The REAL circuit breaker is the $25/day hard cap at llm_client.py (cost_budget_daily_usd).")
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
    # phase-30.5: NAV-percentage sector cap (P2-2 from phase-30.0 audit).
    # Complements the COUNT cap above: count blocks "many small positions",
    # NAV-pct blocks "one fat position dominating the sector". Default 30%
    # per arXiv 2512.02227 Dec 2025 Orchestration Framework explicit
    # `"sectorLimit": 0.30` for stocks; brackets between SEC 1940 Act
    # 25% "concentrated" threshold and UCITS 5/10/40 40% aggregate ceiling.
    # 0 = no limit (legacy / disabled). The two caps fire independently.
    paper_max_per_sector_nav_pct: float = Field(
        30.0,
        ge=0.0,
        le=100.0,
        description="Maximum NAV percentage per single GICS sector. 0 = no limit (legacy). Default 30 per arXiv 2512.02227 Dec 2025 + LSEG/CFA/SEC bracket. Fires alongside paper_max_per_sector count cap.",
    )
    # phase-40.8 (OPEN-5): Fama-French 3-factor correlation cap. Catches
    # cross-sector factor crowding that GICS sector cap misses (e.g., two
    # stocks in different GICS sectors but both high-momentum + small-value
    # are still correlated by FF3 loadings). Cosine similarity over
    # (market_beta, smb_beta, hml_beta) > threshold blocks the BUY.
    # 0.0 = disabled (legacy default). Recommended live value 0.85.
    # Quiet-log for 1-2 weeks before enabling, per AQR/Two Sigma 2025
    # factor-crowding research (research_brief_phase_40_8.md).
    paper_max_factor_corr: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Maximum FF3 factor cosine similarity vs portfolio avg before blocking BUY. 0.0 = disabled. Recommended live 0.85.",
    )
    # phase-57.1 (55.3 finding F-3): binding RiskJudge gate. The away week
    # executed 3 BUYs at risk_judge_decision='REJECT' (all via the swap path;
    # net realized -$23.45) because the verdict was advisory-only. When True,
    # a REJECT verdict BLOCKS the BUY at the candidate-build chokepoint
    # (covers BOTH the main BUY path and the swap path) AND the lite RiskJudge
    # prompts gain the configured sector cap + a live sector-breakdown line
    # (F-8) -- one flag, coherent semantics (never bind on a blind judge).
    # Default OFF = byte-identical behavior INCLUDING prompts. Operator flips
    # live only after OOS validation (no flip inside phase-57).
    paper_risk_judge_reject_binding: bool = Field(
        False,
        description="phase-57.1 (F-3/F-8): when True, a lite-path RiskJudge REJECT verdict BLOCKS the BUY (binding pre-execution gate per SEC 15c3-5 rejection doctrine) and the judge prompts receive the real configured sector cap + live sector breakdown. Default OFF preserves byte-identical behavior including prompts.",
    )
    # phase-cycle-1 (2026-05-26): position-swap framework. Sell-to-buy-better
    # path so the autonomous loop doesn't idle on cash when a higher-conviction
    # candidate is sector-blocked by an existing low-conviction holding. North
    # star: maximize profit. Testing-phase mandate: default to firing, not
    # gating, when risk caps permit.
    #
    # Citations (research_brief_phase_zero_buy_triage.md):
    # - Resonanz Capital "upgrade-vs-exit" position-sizing framework.
    # - Kelly-Optimal Rebalancing Frequency (arXiv:1807.05265).
    # - Grinold-Kahn Fundamental Law of Active Management (breadth via
    #   independent bets; concentrated cap stays at 30% per arXiv:2512.02227).
    # - ADVERSARIAL: arXiv:2505.07078v5 KDD 2026 (LLM agents overtrade in
    #   bulls; FinMem commission 5-9x FinAgent, Sharpe 0.12 vs B&H 0.61).
    #   Conservative defaults below.
    paper_swap_enabled: bool = Field(
        True,
        description="Enable sector-blocked swap path: SELL lowest-conviction sector holding to free a slot for a higher-conviction candidate. Default True per goal mandate; the swap STILL passes a delta threshold so churn is bounded.",
    )
    paper_swap_min_delta_pct: float = Field(
        25.0,
        ge=0.0,
        le=200.0,
        description="Minimum (cand_score - holding_score) / max(abs(holding_score), 1.0) * 100 to fire a swap. Conservative initial value per Resonanz Capital upgrade-vs-exit + KDD 2026 adversarial overtrade evidence. Future A/B vs 15% / 35% under backtest evidence only.",
    )
    paper_swap_max_per_cycle: int = Field(
        2,
        ge=0,
        le=10,
        description="Hard cap on swap pairs per autonomous run. 0 disables. Per KDD 2026 adversarial: LLMs overtrade; keep tight until backtest evidence supports loosening.",
    )
    paper_swap_churn_fix_enabled: bool = Field(
        False,
        description="phase-60.2 (AW-5): churn-engine fix flag, DEFAULT OFF (do-no-harm). When ON: (1) a holding absent from the same-cycle holding_lookup is EXCLUDED from swap displacement entirely -- the pre-fix conviction-0.0 sentinel made every fresh BUY swap-out bait the next day (MU/SNDK/DELL round trips, 81.4% weekly turnover). Exclusion chosen over LOCF valuation for displacement decisions: day-over-day score noise (mean |delta| 1.10 on the 1-10 scale, 59.3 stability table) can cross a 25% relative bar, re-admitting churn; the holding keeps its slot (alpha decays over months -- Alpha Decay, Di Mascio/Lines/Naik) and the re-eval cadence restores displaceability on true evidence within 3-4 days. (2) The swap-delta denominator uses the documented max(abs(holding_score), 1.0) clamp (paper_swap_min_delta_pct's own spec) instead of the 0.01 epsilon that turned sentinel comparisons into ~70,000% deltas. (3) The re-eval gate compares hours-precise age instead of truncated .days (behavior-identical for integer day thresholds; exact for fractional). OFF path is byte-identical to pre-60.2 behavior. Promotion to ON is an OPERATOR decision recorded in live_check_60.2.md -- never auto-applied. NOT a churn lever: this repairs fabricated-evidence comparisons; the 25.0 threshold is untouched (53.1/55.3 anti-band rulings respected).",
    )
    # phase-38.1 (OPEN-10): kill-switch auto-resume hysteresis feature flag.
    # Default-OFF preserves the manual-resume behavior. When ON, the
    # autonomous loop's check_and_enforce_kill_switch call also invokes
    # check_auto_resume; if paused + no breach + >=2h elapsed, the system
    # auto-resumes (with a T+1h Slack pager alert as a heads-up).
    # Operator-approval criterion: this flag IS the opt-in surface.
    kill_switch_auto_resume_enabled: bool = Field(
        False,
        description="Enable kill-switch auto-resume after 2h of no-breach. Default OFF. Operator opt-in: phase-38.1 (OPEN-10).",
    )
    # phase-40.8.1 (P3): feature flag for the FF3 factor-loadings producer.
    # Default-OFF preserves byte-identical behavior. Required for the
    # phase-40.8 cap to fire even when paper_max_factor_corr > 0.
    enable_factor_loadings: bool = Field(
        False,
        description="Enable FF3 factor-loadings producer in screener step. Required for paper_max_factor_corr to fire. Default OFF.",
    )
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
    # phase-28.7: multidimensional momentum composite (price + 52w-high + SUE + sector blend)
    multidim_momentum_enabled: bool = Field(False, description="phase-28.7: Replace screener composite_score with a z-blended multidim momentum (CFA Institute Dec 2025: superior Sharpe + lower crash risk vs price-only). Default OFF.")
    multidim_momentum_weight_price: float = Field(0.35, description="phase-28.7: Z-blend weight for the existing price momentum composite (mom_1m/3m/6m + RSI/vol penalties).")
    multidim_momentum_weight_52w_high: float = Field(0.25, description="phase-28.7: Z-blend weight for 52-week-high proximity (George-Hwang 2004 anchoring effect).")
    multidim_momentum_weight_sue: float = Field(0.20, description="phase-28.7: Z-blend weight for SUE momentum (pead_signal.surprise_score; 0 if missing).")
    multidim_momentum_weight_sector: float = Field(0.20, description="phase-28.7: Z-blend weight for sector/factor momentum (sector_momentum_ranks boost_multiplier - 1.0; 0 if missing).")
    momentum_52wh_tilt_enabled: bool = Field(False, description="phase-52.2: When True, rank_candidates applies a CENTERED 52-week-high multiplicative tilt to composite_score (George-Hwang 2004; measured +0.05 ann Sharpe at k=0.5, turnover-neutral, in phase-52.1). Default OFF -> byte-identical. Enable is operator-gated (post-Monday-baseline, DSR-deflated).")
    momentum_52wh_tilt_k: float = Field(0.5, description="phase-52.2: 52wh tilt strength k in composite*(1+k*(pct_to_52w-universe_mean)). 0.5 = the milder/plateau choice (k=1.0 was borderline in the 52.1 replay).")
    # phase-28.8: Russell-1000 universe expansion (addresses Sandisk/SNDK reference-case spinoff miss)
    russell1000_universe_enabled: bool = Field(False, description="phase-28.8: Use Russell-1000 (~1000 tickers) instead of S&P 500 (~503) for screen_universe. Default OFF. Existing two-pass design (cheap screen_universe -> top-N cap) keeps downstream cost bounded.")
    russell1000_cache_days: int = Field(180, description="phase-28.8: Days to cache the IWB ticker list (FTSE Russell semi-annual reconstitution -> 180 day TTL).")
    # phase-28.9: options-flow OI-surge filter (near-expiry OTM call volume spike)
    options_flow_screen_enabled: bool = Field(False, description="phase-28.9: Boost candidates with near-expiry OTM call OI/volume surge. Wayne State / J. Portfolio Mgmt: surge predictive of forward returns. Default OFF.")
    options_otm_threshold: float = Field(1.01, description="phase-28.9: Minimum strike/spot ratio to count as OTM (1.01 = strike at least 1% above spot).")
    options_dte_min: int = Field(2, description="phase-28.9: Minimum days-to-expiration for the signal window.")
    options_dte_max: int = Field(45, description="phase-28.9: Maximum days-to-expiration; the signal is specific to near-expiry options.")
    options_vol_avg_multiplier: float = Field(5.0, description="phase-28.9: A surge requires today's volume to exceed this multiple of the chain's average per-strike volume.")
    options_vol_oi_multiplier: float = Field(3.0, description="phase-28.9: AND the surge requires volume to exceed this multiple of open interest (cross-check vs informed-flow definition).")
    options_strong_boost: float = Field(0.06, description="phase-28.9: Score multiplier added (e.g. 0.06 = +6%) when 2+ surge strikes detected.")
    options_moderate_boost: float = Field(0.03, description="phase-28.9: Score multiplier added when exactly 1 surge strike detected.")
    options_cache_hours: int = Field(4, description="phase-28.9: Hours to cache per-ticker options-surge signals (chain updates intraday).")
    # phase-28.10: opportunistic insider-buying signal (Cohen-Malloy-Pomorski classifier)
    insider_signal_screen_enabled: bool = Field(False, description="phase-28.10: Boost candidates with material opportunistic insider buying. CMP classifier: routine = same-month-of-year for 3 prior years; opportunistic = all others. CMP earns 82bps/mo abnormal. Default OFF.")
    insider_lookback_history_months: int = Field(48, description="phase-28.10: Months of insider history fetched for CMP classification (need >=36 to identify 3-year repeats).")
    insider_signal_window_days: int = Field(30, description="phase-28.10: Aggregation window for opportunistic-buy dollar value.")
    insider_signal_min_aggregate_usd: float = Field(500_000.0, description="phase-28.10: Minimum aggregate opportunistic-buy $ in window to fire moderate boost.")
    insider_signal_strong_aggregate_usd: float = Field(2_000_000.0, description="phase-28.10: Threshold $ for strong boost (typically 4x moderate).")
    insider_strong_boost: float = Field(0.07, description="phase-28.10: Multiplier (additive) for strong opportunistic buying (default +7%).")
    insider_moderate_boost: float = Field(0.04, description="phase-28.10: Multiplier (additive) for moderate opportunistic buying (default +4%).")
    # phase-28.11: LLM analyst-narrative signal (MVP: management-outlook proxy from 8-K Exhibit 99)
    # HONEST DISCLOSURE: the canonical 68bps/mo signal (arXiv 2502.20489v1) needs paid Investext
    # ($10K-100K/yr) — not viable for local-only deployment. This MVP uses management forward-looking
    # tone from 8-K press releases as a free proxy. Different lens from PEAD (sentiment-vs-trend).
    analyst_narrative_enabled: bool = Field(False, description="phase-28.11: LLM-scored management outlook tone from 8-K Exhibit 99 (MVP proxy for canonical analyst Strategic Outlook signal). Default OFF.")
    analyst_narrative_model: str = Field("claude-haiku-4-5", description="phase-28.11: LLM used to score management outlook tone (~$0.001 per call).")
    analyst_narrative_cost_cap_usd: float = Field(0.10, description="phase-28.11: Soft cap on per-cycle cost for this signal. Operator monitoring; not enforced as hard kill.")
    analyst_narrative_strong_threshold: float = Field(0.70, description="phase-28.11: outlook_score above this triggers strong boost (default 0.70 = strongly bullish forward language).")
    analyst_narrative_weak_threshold: float = Field(0.30, description="phase-28.11: outlook_score below this triggers strong penalty (strongly bearish).")
    analyst_narrative_strong_boost: float = Field(0.05, description="phase-28.11: Multiplier (additive) for strong bullish outlook (default +5%; conservatively half PEAD scale pending A/B).")
    analyst_narrative_moderate_boost: float = Field(0.025, description="phase-28.11: Multiplier (additive) for moderate outlook (default +2.5%).")
    # phase-28.13: earnings-call NLP for firm-level GPR exposure (Fed 2025 methodology)
    # HONESTY: Fed showed R²=0.23 CONTEMPORANEOUS only — NO forward predictability. Use as
    # defensive risk filter on candidates, NOT alpha source. Complements phase-28.3
    # sector-tilt by adding firm-level dimension.
    call_transcript_gpr_enabled: bool = Field(False, description="phase-28.13: LLM-classify per-firm GPR exposure tier from earnings call transcripts. DEFENSIVE FILTER (Fed 2025: contemporaneous only, no forward alpha). Default OFF.")
    call_transcript_gpr_model: str = Field("claude-haiku-4-5", description="phase-28.13: LLM used to classify GPR exposure (~$0.001/call).")
    call_transcript_gpr_high_penalty: float = Field(0.97, description="phase-28.13: Multiplier for HIGH-exposure firms not in exempt sectors (default 0.97 = -3% defensive haircut).")
    call_transcript_gpr_exempt_sectors: str = Field("Industrials,Energy", description="phase-28.13: Comma-separated sectors that BENEFIT from elevated GPR — no penalty applied for these (defense contractors live in Industrials, oil majors in Energy).")
    call_transcript_gpr_cost_cap_usd: float = Field(0.10, description="phase-28.13: Per-cycle soft cap. Operator monitoring; ~$0.001 per LLM call.")
    # phase-28.15: social media velocity in screener (lifts existing social_sentiment.py)
    social_velocity_enabled: bool = Field(False, description="phase-28.15: Boost candidates with social-sentiment velocity spikes via Alpha Vantage NEWS_SENTIMENT (bundles Reddit/Twitter/StockTwits/blogs). DNUT July 2025: 500% StockTwits spike preceded 90% pre-market. Default OFF.")
    social_velocity_min_threshold: float = Field(0.10, description="phase-28.15: Minimum velocity (recent_avg - older_avg) to fire moderate boost.")
    social_velocity_min_mentions: int = Field(3, description="phase-28.15: Minimum ticker mention_count to qualify (noise guard).")
    social_velocity_strong_threshold: float = Field(0.20, description="phase-28.15: Velocity above which strong boost fires (2x moderate threshold).")
    social_velocity_strong_boost: float = Field(0.06, description="phase-28.15: Multiplier for strong velocity spike (default +6%).")
    social_velocity_moderate_boost: float = Field(0.03, description="phase-28.15: Multiplier for moderate velocity (default +3%).")
    # phase-28.14: defense/war-stocks reference case (GPR + XAR momentum AND-gate; cycle-level)
    defense_signal_enabled: bool = Field(False, description="phase-28.14: Boost defense-sector tickers when GPR-Acts above threshold AND XAR 5d momentum > 0. Supplement Gap 1: Emerald SEF 2023 +1.00% (-1,-1) + 11.65% CAAR (0,3). Default OFF.")
    defense_xar_window_days: int = Field(5, description="phase-28.14: Trading-day window for XAR (S&P Aerospace & Defense ETF) momentum check.")
    defense_xar_min_momentum: float = Field(0.0, description="phase-28.14: Minimum XAR cumulative return over window to confirm institutional flow into defense.")
    defense_tickers: str = Field("LMT,NOC,RTX,GD,LHX,BA,LDOS,HII,KTOS,BAE.L,RHM.DE,SAAB-B.ST", description="phase-28.14: Comma-separated tickers boosted when defense_signal triggers. US primes + EU primes (Researcher: BAE/RHM most GPR-sensitive).")
    defense_boost: float = Field(0.05, description="phase-28.14: Multiplier for defense tickers when signal triggers (default +5%).")
    defense_budget_pledge_keywords: str = Field("NATO budget,defense spending,Zeitenwende,defense pledge,military spending,5% GDP", description="phase-28.14: Keywords scanned in news_screen headlines as additional signal confirmation (optional).")
    # phase-28.17: peer-correlation laggard catch-up (intra-sector lead-lag)
    peer_leadlag_enabled: bool = Field(False, description="phase-28.17: Boost laggards in sectors with strong-momentum leaders. Hou 2007 + DeltaLag 2025: ~10 bpts/day gross alpha; analyst-coverage-driven information diffusion lag. Default OFF.")
    peer_leadlag_leader_threshold: float = Field(10.0, description="phase-28.17: momentum_1m percent above which a stock counts as a sector leader.")
    peer_leadlag_laggard_threshold: float = Field(2.0, description="phase-28.17: momentum_1m percent below which a stock counts as a laggard candidate.")
    peer_leadlag_min_analyst_filter: int = Field(5, description="phase-28.17: Maximum analyst_count to qualify as laggard (Hou 2007: lag is strongest where coverage is low).")
    peer_leadlag_min_market_cap_usd: float = Field(2_000_000_000.0, description="phase-28.17: Minimum market cap for laggard qualification (DeltaLag $2B liquidity gate).")
    peer_leadlag_boost: float = Field(0.08, description="phase-28.17: Multiplier (additive) for qualifying laggards (default +8%, conservative vs DeltaLag gross).")
    # phase-28.16: M&A pre-announcement aggregator (Legs 1+2 from 28.9+28.10; Leg 3 stubbed)
    ma_preannounce_enabled: bool = Field(False, description="phase-28.16: Aggregate options-surge (28.9) + insider-buying (28.10) + 13D-stub legs into a single M&A pre-announcement signal. Augustin-Brenner-Subrahmanyam + Duong-Pi-Sapp 2025. LEGALITY: uses only public market/EDGAR data. Default OFF.")
    ma_preannounce_strong_boost: float = Field(0.10, description="phase-28.16: Multiplier when 2+ legs fire (default +10%; high-confidence convergence).")
    ma_preannounce_moderate_boost: float = Field(0.05, description="phase-28.16: Multiplier when exactly 1 leg fires (default +5%).")
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
    # phase-32.2: HWM-trailing stop distance after the +1R breakeven ratchet
    # (phase-32.1) has fired. Used in paper_trader._advance_stop's trailing
    # branch. Default 8 matches the breakeven 1R = 8%. Skipped on
    # entry_strategy in {'mean_reversion','pairs'} per Kaminski-Lo
    # Proposition 2 adversarial guard.
    paper_trailing_stop_pct: float = Field(
        8.0,
        ge=0.5,
        le=50.0,
        description="HWM-trailing stop distance (%% below running peak) used after +1R breakeven ratchet fires. Default 8 -- matches phase-32.1's breakeven threshold.",
    )
    # phase-30.6: pre-trade price-tolerance gate (P2-4 from phase-30.0 audit).
    # Reject BUY when live fill price diverges from analysis-time price by
    # more than this percent. Default 5 per SEC LULD Tier 1 5% band -- the
    # canonical regulator-anchored threshold for the pyfinagent universe
    # (S&P 500 + Russell 1000 > $3). 0 = no limit (legacy / disabled).
    # FIA WP July 2024 Sec 1.3 is the canonical pre-trade-gate reference;
    # this implementation places the check BEFORE ExecutionRouter so the
    # non-bypassable-invariants pattern from arXiv 2603.10092 holds.
    paper_price_tolerance_pct: float = Field(
        5.0,
        ge=0.0,
        le=50.0,
        description="Reject BUY when live fill price diverges from analysis-time price by more than this percent. 0 = no limit (legacy). Default 5 per SEC LULD Tier 1 band for S&P 500 + Russell 1000 > $3.",
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
    slack_operator_user_id: str = Field(
        "U0A078KP4FQ",
        description=(
            "phase-62.2 (goal-away-ops): the ONLY Slack user whose messages are "
            "recorded as operator decision tokens (operator_tokens.jsonl). "
            "Identity constant, not a secret (same class as the approval-channel "
            "id in commands.py). Empty string = fail-closed (no tokens accepted)."
        ),
    )
    away_mode_enabled: bool = Field(
        False,
        description=(
            "phase-62.8 (goal-away-ops): OPS toggle (Fowler classification) -- appends "
            "the away-mode sections to the daily digests. NOT a trading-behavior flag. "
            "Operator keystroke into backend/.env at the 62.7 dress rehearsal "
            "(AWAY_MODE_ENABLED=true) + bot restart; remove after the away window."
        ),
    )
    morning_digest_hour: int = Field(8, description="Hour (0-23) for daily morning digest in local timezone")
    evening_digest_hour: int = Field(17, description="Hour (0-23) for daily evening digest in local timezone")
    watchdog_interval_minutes: int = Field(15, description="Interval (minutes) for watchdog health check")

    # --- First-week monitoring (Phase 4.4.6.3) ---
    first_week_mode: bool = Field(False, description="Tighten alert thresholds for 7 days post-launch. Drawdown de-risk: -10% -> -5%, SLA P3 response: 4h -> 1h. Set FIRST_WEEK_MODE=true at go-live, revert after day 7.")
    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # pydantic-settings loads from env/.env
