"""
Settings API routes — read and update model configuration and analysis settings.
"""

import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.config.settings import Settings, get_settings
from backend.services.api_cache import ENDPOINT_TTLS, get_api_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings", tags=["settings"])

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Valid model names — whitelist to prevent arbitrary writes to .env
# Includes Gemini (direct), GitHub Models catalog, Anthropic direct, OpenAI direct
_VALID_MODELS = {
    # Gemini (Vertex AI). phase-60.1: gemini-2.0-flash removed -- discontinued
    # server-side 2026-06-01 (selecting it guarantees 404s).
    "gemini-2.5-flash", "gemini-2.5-pro",
    # GitHub Models — OpenAI
    "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-5", "gpt-5-chat", "gpt-5-mini", "gpt-5-nano",
    "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini",
    # Anthropic — current GA
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5", "claude-opus-4-1",
    "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5",
    # Legacy — retire 2026-06-15
    "claude-sonnet-4", "claude-opus-4",
    # GitHub Models — Meta
    "meta-llama-3.1-405b-instruct", "meta-llama-3.1-8b-instruct",
    "llama-3.3-70b-instruct", "llama-4-maverick", "llama-4-scout",
    # GitHub Models — DeepSeek
    "deepseek-r1", "deepseek-r1-0528", "deepseek-v3-0324",
    # GitHub Models — xAI
    "grok-3", "grok-3-mini",
    # GitHub Models — Microsoft
    "phi-4", "mai-ds-r1", "phi-4-mini-instruct", "phi-4-mini-reasoning", "phi-4-reasoning",
    # GitHub Models — Mistral
    "ministral-3b", "codestral-2501", "mistral-medium-2505", "mistral-small-2503",
}


class ModelConfig(BaseModel):
    """Readable model configuration."""
    gemini_model: str
    deep_think_model: str
    max_debate_rounds: int
    max_risk_debate_rounds: int
    # phase-21.1 -- when true, gemini_model overrides ALL agent role-mappings
    # in model_tiers.resolve_model() except the Gemini-locked roles.
    apply_model_to_all_agents: bool = False


class FullSettings(BaseModel):
    """Complete readable settings for the Settings UI."""
    # Models
    gemini_model: str
    deep_think_model: str
    apply_model_to_all_agents: bool = False
    # Debate depth
    max_debate_rounds: int
    max_risk_debate_rounds: int
    # Pillar weights
    weight_corporate: float
    weight_industry: float
    weight_valuation: float
    weight_sentiment: float
    weight_governance: float
    # Quality gates
    data_quality_min: float
    # Cost controls
    lite_mode: bool
    max_analysis_cost_usd: float
    max_synthesis_iterations: int
    # Multi-provider key status (read-only; shows whether keys are configured)
    anthropic_key_configured: bool = False
    openai_key_configured: bool = False
    github_token_configured: bool = False
    # phase-23.1 — Signal Stack
    macro_regime_filter_enabled: bool = False
    macro_regime_model: str = "claude-haiku-4-5"
    pead_signal_enabled: bool = False
    pead_signal_model: str = "claude-haiku-4-5"
    pead_signal_lookback_quarters: int = 8
    news_screen_enabled: bool = False
    news_screen_model: str = "claude-haiku-4-5"
    news_screen_max_headlines: int = 100
    sector_calendars_enabled: bool = False
    sector_calendars_lookahead_days: int = 7
    meta_scorer_enabled: bool = False
    meta_scorer_model: str = "claude-haiku-4-5"
    meta_scorer_max_batch: int = 30
    # phase-23.1.9 — Paper trading settings
    paper_starting_capital: float = 10000.0  # informational read-only (.env base; live value is paper_portfolio.starting_capital)
    paper_trading_hour: int = 10  # phase-70.5: daily cron hour (ET, 0-23); writable + reschedules the job
    paper_max_positions: int = 10
    paper_max_per_sector: int = 2  # phase-23.1.13
    paper_markets: list[str] = ["US"]  # phase-50.6: live-loop markets (subset of US/EU/KR)
    paper_max_daily_cost_usd: float = 2.0
    paper_default_stop_loss_pct: float = 8.0
    paper_screen_top_n: int = 10
    paper_analyze_top_n: int = 5
    paper_transaction_cost_pct: float = 0.1
    paper_daily_loss_limit_pct: float = 4.0
    paper_trailing_dd_limit_pct: float = 10.0
    paper_min_cash_reserve_pct: float = 5.0
    # phase-cycle-5 (2026-05-26): Claude Code CLI rail flag. Operator opt-in
    # for the testing phase. False = api.anthropic.com direct billing rail
    # (existing). True = `claude --print` subprocess on Max-subscription
    # flat-fee rail. Defaults to False so existing path stays intact.
    paper_use_claude_code_route: bool = False
    # phase-cycle-7 (38.12, 2026-05-27): cycle wall-clock budget exposure.
    # Raised default 1800 -> 7200 because the Claude Code rail's ~30s
    # subprocess overhead doesn't fit 13 tickers in 3600s. Operator can
    # lower via Settings UI when Anthropic-direct rail is restored.
    paper_cycle_max_seconds: float = 7200.0


class SettingsUpdate(BaseModel):
    """Writable fields for the Settings UI. All optional — only provided fields are updated."""
    gemini_model: Optional[str] = None
    deep_think_model: Optional[str] = None
    apply_model_to_all_agents: Optional[bool] = None
    max_debate_rounds: Optional[int] = Field(None, ge=1, le=5)
    max_risk_debate_rounds: Optional[int] = Field(None, ge=1, le=3)
    weight_corporate: Optional[float] = Field(None, ge=0, le=1)
    weight_industry: Optional[float] = Field(None, ge=0, le=1)
    weight_valuation: Optional[float] = Field(None, ge=0, le=1)
    weight_sentiment: Optional[float] = Field(None, ge=0, le=1)
    weight_governance: Optional[float] = Field(None, ge=0, le=1)
    data_quality_min: Optional[float] = Field(None, ge=0, le=1)
    lite_mode: Optional[bool] = None
    max_analysis_cost_usd: Optional[float] = Field(None, ge=0.01, le=10.0)
    max_synthesis_iterations: Optional[int] = Field(None, ge=1, le=3)
    # phase-23.1 — Signal Stack
    macro_regime_filter_enabled: Optional[bool] = None
    macro_regime_model: Optional[str] = None
    pead_signal_enabled: Optional[bool] = None
    pead_signal_model: Optional[str] = None
    pead_signal_lookback_quarters: Optional[int] = Field(None, ge=1, le=12)
    news_screen_enabled: Optional[bool] = None
    news_screen_model: Optional[str] = None
    news_screen_max_headlines: Optional[int] = Field(None, ge=10, le=500)
    sector_calendars_enabled: Optional[bool] = None
    sector_calendars_lookahead_days: Optional[int] = Field(None, ge=1, le=30)
    meta_scorer_enabled: Optional[bool] = None
    meta_scorer_model: Optional[str] = None
    meta_scorer_max_batch: Optional[int] = Field(None, ge=5, le=100)
    # phase-23.1.9 — Paper trading settings (paper_starting_capital is NOT writable post-init)
    paper_max_positions: Optional[int] = Field(None, ge=1, le=50)
    paper_max_per_sector: Optional[int] = Field(None, ge=0, le=20)  # phase-23.1.13
    paper_markets: Optional[list[str]] = None  # phase-50.6: subset of US/EU/KR
    paper_max_daily_cost_usd: Optional[float] = Field(None, ge=0.10, le=50.0)
    paper_default_stop_loss_pct: Optional[float] = Field(None, ge=1.0, le=50.0)
    paper_screen_top_n: Optional[int] = Field(None, ge=1, le=100)
    paper_analyze_top_n: Optional[int] = Field(None, ge=1, le=50)
    paper_transaction_cost_pct: Optional[float] = Field(None, ge=0.0, le=5.0)
    paper_daily_loss_limit_pct: Optional[float] = Field(None, ge=0.5, le=25.0)
    paper_trailing_dd_limit_pct: Optional[float] = Field(None, ge=1.0, le=50.0)
    paper_min_cash_reserve_pct: Optional[float] = Field(None, ge=0.0, le=50.0)
    # phase-cycle-5 (2026-05-26): Claude Code CLI rail flag.
    paper_use_claude_code_route: Optional[bool] = None
    # phase-cycle-7 (38.12, 2026-05-27): cycle wall-clock budget.
    paper_cycle_max_seconds: Optional[float] = Field(None, ge=300.0, le=21600.0)
    # phase-70.5: daily-run hour (ET, 0-23). A change reschedules the APScheduler job
    # in-place (no restart). Does NOT affect the fresh-per-cycle cap reads.
    paper_trading_hour: Optional[int] = Field(None, ge=0, le=23)


class ModelConfigUpdate(BaseModel):
    """Legacy writable model configuration (kept for backward compatibility)."""
    gemini_model: Optional[str] = None
    deep_think_model: Optional[str] = None
    apply_model_to_all_agents: Optional[bool] = None


class ModelPricing(BaseModel):
    """Per-model pricing info."""
    model: str
    provider: str = "Gemini"
    input_per_1m: float
    output_per_1m: float
    copilot_multiplier: Optional[float] = None  # Premium quota multiplier for GitHub Copilot (0.33x / 1x / 3x)
    context_limited: bool = False  # True = GitHub Models enforces a small request body limit; debate prompts will be compacted


AVAILABLE_MODELS = [
    # Gemini (Vertex AI) — always available via Application Default Credentials
    # phase-60.1: gemini-2.0-flash removed (discontinued 2026-06-01); 2.5-flash
    # pricing re-verified 2026-06-11 ($0.30/$2.50).
    {"model": "gemini-2.5-flash",  "provider": "Gemini", "input_per_1m": 0.30, "output_per_1m": 2.50},
    {"model": "gemini-2.5-pro",    "provider": "Gemini", "input_per_1m": 1.25, "output_per_1m": 10.00},
    # GitHub Models — requires GITHUB_TOKEN + Copilot Pro subscription
    # copilot_multiplier = premium quota consumed per request (0.33x light / 1x standard / 3x premium)
    #   based on GitHub Models rate-limit tiers: low→0.33x, high→1x, custom(8-10/day)→3x, custom(12+/day)→1x
    # context_limited = True means GitHub Models enforces a small request body limit (~4K-8K tokens);
    #   the orchestrator compacts enrichment_for_debate to fit — debate quality is somewhat reduced
    # ── OpenAI ──
    {"model": "gpt-4.1",       "provider": "GitHub Models", "input_per_1m": 2.00,  "output_per_1m": 8.00,   "copilot_multiplier": 1.0},
    {"model": "gpt-4.1-mini",  "provider": "GitHub Models", "input_per_1m": 0.40,  "output_per_1m": 1.60,   "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "gpt-4.1-nano",  "provider": "GitHub Models", "input_per_1m": 0.10,  "output_per_1m": 0.40,   "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "gpt-4o",        "provider": "GitHub Models", "input_per_1m": 2.50,  "output_per_1m": 10.00,  "copilot_multiplier": 1.0},
    {"model": "gpt-4o-mini",   "provider": "GitHub Models", "input_per_1m": 0.15,  "output_per_1m": 0.60,   "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "gpt-5",         "provider": "GitHub Models", "input_per_1m": 10.00, "output_per_1m": 40.00,  "copilot_multiplier": 3.0,  "context_limited": True},
    {"model": "gpt-5-chat",    "provider": "GitHub Models", "input_per_1m": 5.00,  "output_per_1m": 20.00,  "copilot_multiplier": 1.0,  "context_limited": True},
    {"model": "gpt-5-mini",    "provider": "GitHub Models", "input_per_1m": 2.00,  "output_per_1m": 8.00,   "copilot_multiplier": 1.0,  "context_limited": True},
    {"model": "gpt-5-nano",    "provider": "GitHub Models", "input_per_1m": 0.50,  "output_per_1m": 2.00,   "copilot_multiplier": 1.0,  "context_limited": True},
    {"model": "o1",            "provider": "GitHub Models", "input_per_1m": 15.00, "output_per_1m": 60.00,  "copilot_multiplier": 3.0},
    {"model": "o1-mini",       "provider": "GitHub Models", "input_per_1m": 3.00,  "output_per_1m": 12.00,  "copilot_multiplier": 1.0,  "context_limited": True},
    {"model": "o1-preview",    "provider": "GitHub Models", "input_per_1m": 15.00, "output_per_1m": 60.00,  "copilot_multiplier": 3.0,  "context_limited": True},
    {"model": "o3",            "provider": "GitHub Models", "input_per_1m": 2.00,  "output_per_1m": 8.00,   "copilot_multiplier": 3.0},
    {"model": "o3-mini",       "provider": "GitHub Models", "input_per_1m": 1.10,  "output_per_1m": 4.40,   "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "o4-mini",       "provider": "GitHub Models", "input_per_1m": 1.10,  "output_per_1m": 4.40,   "copilot_multiplier": 0.33},
    # ── Anthropic direct — current GA ──
    {"model": "claude-opus-4-8",              "provider": "Anthropic",     "input_per_1m": 5.00,  "output_per_1m": 25.00},
    {"model": "claude-opus-4-7",              "provider": "Anthropic",     "input_per_1m": 5.00,  "output_per_1m": 25.00},
    {"model": "claude-opus-4-6",              "provider": "Anthropic",     "input_per_1m": 5.00,  "output_per_1m": 25.00},
    {"model": "claude-opus-4-5",              "provider": "Anthropic",     "input_per_1m": 5.00,  "output_per_1m": 25.00},
    {"model": "claude-opus-4-1",              "provider": "Anthropic",     "input_per_1m": 15.00, "output_per_1m": 75.00},
    {"model": "claude-sonnet-4-6",            "provider": "Anthropic",     "input_per_1m": 3.00,  "output_per_1m": 15.00},
    {"model": "claude-sonnet-4-5",            "provider": "Anthropic",     "input_per_1m": 3.00,  "output_per_1m": 15.00},
    {"model": "claude-haiku-4-5",             "provider": "Anthropic",     "input_per_1m": 1.00,  "output_per_1m": 5.00},
    # ── Anthropic — legacy (deprecated 2026-04-14, retire 2026-06-15) ──
    {"model": "claude-sonnet-4",              "provider": "Anthropic",     "input_per_1m": 3.00,  "output_per_1m": 15.00},
    {"model": "claude-opus-4",                "provider": "Anthropic",     "input_per_1m": 15.00, "output_per_1m": 75.00},
    # ── Meta ──
    {"model": "meta-llama-3.1-405b-instruct", "provider": "GitHub Models", "input_per_1m": 5.00,  "output_per_1m": 15.00, "copilot_multiplier": 1.0},
    {"model": "meta-llama-3.1-8b-instruct",   "provider": "GitHub Models", "input_per_1m": 0.18,  "output_per_1m": 0.18,  "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "llama-3.3-70b-instruct",       "provider": "GitHub Models", "input_per_1m": 0.23,  "output_per_1m": 0.70,  "copilot_multiplier": 1.0},
    {"model": "llama-4-maverick",             "provider": "GitHub Models", "input_per_1m": 0.19,  "output_per_1m": 0.85,  "copilot_multiplier": 1.0},
    {"model": "llama-4-scout",                "provider": "GitHub Models", "input_per_1m": 0.11,  "output_per_1m": 0.40,  "copilot_multiplier": 1.0},
    # ── DeepSeek ──
    {"model": "deepseek-r1",      "provider": "GitHub Models", "input_per_1m": 0.55,  "output_per_1m": 2.19,  "copilot_multiplier": 3.0,  "context_limited": True},
    {"model": "deepseek-r1-0528", "provider": "GitHub Models", "input_per_1m": 0.55,  "output_per_1m": 2.19,  "copilot_multiplier": 3.0,  "context_limited": True},
    {"model": "deepseek-v3-0324", "provider": "GitHub Models", "input_per_1m": 0.27,  "output_per_1m": 1.10,  "copilot_multiplier": 1.0},
    # ── xAI ──
    {"model": "grok-3",      "provider": "GitHub Models", "input_per_1m": 3.00,  "output_per_1m": 15.00, "copilot_multiplier": 1.0,  "context_limited": True},
    {"model": "grok-3-mini", "provider": "GitHub Models", "input_per_1m": 0.30,  "output_per_1m": 0.50,  "copilot_multiplier": 0.33, "context_limited": True},
    # ── Microsoft ──
    {"model": "phi-4",                "provider": "GitHub Models", "input_per_1m": 0.07,  "output_per_1m": 0.14,  "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "mai-ds-r1",            "provider": "GitHub Models", "input_per_1m": 0.55,  "output_per_1m": 2.19,  "copilot_multiplier": 3.0,  "context_limited": True},
    {"model": "phi-4-mini-instruct",  "provider": "GitHub Models", "input_per_1m": 0.07,  "output_per_1m": 0.14,  "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "phi-4-mini-reasoning", "provider": "GitHub Models", "input_per_1m": 0.10,  "output_per_1m": 0.20,  "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "phi-4-reasoning",      "provider": "GitHub Models", "input_per_1m": 0.10,  "output_per_1m": 0.40,  "copilot_multiplier": 0.33, "context_limited": True},
    # ── Mistral ──
    {"model": "ministral-3b",       "provider": "GitHub Models", "input_per_1m": 0.10,  "output_per_1m": 0.10,  "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "codestral-2501",     "provider": "GitHub Models", "input_per_1m": 0.30,  "output_per_1m": 0.90,  "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "mistral-medium-2505","provider": "GitHub Models", "input_per_1m": 2.00,  "output_per_1m": 6.00,  "copilot_multiplier": 0.33, "context_limited": True},
    {"model": "mistral-small-2503", "provider": "GitHub Models", "input_per_1m": 0.10,  "output_per_1m": 0.30,  "copilot_multiplier": 0.33, "context_limited": True},
    # Note: GitHub Models claude-* models also fall back to ANTHROPIC_API_KEY when GITHUB_TOKEN is not set.
]

# Mapping from SettingsUpdate field names to .env variable names
_FIELD_TO_ENV = {
    # phase-61.2: decision-input integrity flags (operator-visible in the
    # Settings UI rather than manual-.env-only -- the 61.1 lesson).
    "paper_synthesis_integrity_enabled": "PAPER_SYNTHESIS_INTEGRITY_ENABLED",
    "paper_position_recommendation_fix_enabled": "PAPER_POSITION_RECOMMENDATION_FIX_ENABLED",
    "paper_risk_judge_shape_fix_enabled": "PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED",
    "claude_code_timeout_s": "CLAUDE_CODE_TIMEOUT_S",
    "claude_code_empty_retry_max": "CLAUDE_CODE_EMPTY_RETRY_MAX",
    "gemini_model": "GEMINI_MODEL",
    "deep_think_model": "DEEP_THINK_MODEL",
    "max_debate_rounds": "MAX_DEBATE_ROUNDS",
    "max_risk_debate_rounds": "MAX_RISK_DEBATE_ROUNDS",
    "weight_corporate": "WEIGHT_CORPORATE",
    "weight_industry": "WEIGHT_INDUSTRY",
    "weight_valuation": "WEIGHT_VALUATION",
    "weight_sentiment": "WEIGHT_SENTIMENT",
    "weight_governance": "WEIGHT_GOVERNANCE",
    "data_quality_min": "DATA_QUALITY_MIN",
    "lite_mode": "LITE_MODE",
    "max_analysis_cost_usd": "MAX_ANALYSIS_COST_USD",
    "max_synthesis_iterations": "MAX_SYNTHESIS_ITERATIONS",
    # phase-21.1
    "apply_model_to_all_agents": "APPLY_MODEL_TO_ALL_AGENTS",
    # phase-23.1 Signal Stack
    "macro_regime_filter_enabled": "MACRO_REGIME_FILTER_ENABLED",
    "macro_regime_model": "MACRO_REGIME_MODEL",
    "pead_signal_enabled": "PEAD_SIGNAL_ENABLED",
    "pead_signal_model": "PEAD_SIGNAL_MODEL",
    "pead_signal_lookback_quarters": "PEAD_SIGNAL_LOOKBACK_QUARTERS",
    "news_screen_enabled": "NEWS_SCREEN_ENABLED",
    "news_screen_model": "NEWS_SCREEN_MODEL",
    "news_screen_max_headlines": "NEWS_SCREEN_MAX_HEADLINES",
    "sector_calendars_enabled": "SECTOR_CALENDARS_ENABLED",
    "sector_calendars_lookahead_days": "SECTOR_CALENDARS_LOOKAHEAD_DAYS",
    "meta_scorer_enabled": "META_SCORER_ENABLED",
    "meta_scorer_model": "META_SCORER_MODEL",
    "meta_scorer_max_batch": "META_SCORER_MAX_BATCH",
    # phase-23.1.9 — Paper trading settings
    "paper_trading_hour": "PAPER_TRADING_HOUR",  # phase-70.5
    "paper_max_positions": "PAPER_MAX_POSITIONS",
    "paper_max_per_sector": "PAPER_MAX_PER_SECTOR",  # phase-23.1.13
    "paper_markets": "PAPER_MARKETS",  # phase-50.6 (serialized as CSV; settings.py validator parses)
    "paper_max_daily_cost_usd": "PAPER_MAX_DAILY_COST_USD",
    "paper_default_stop_loss_pct": "PAPER_DEFAULT_STOP_LOSS_PCT",
    "paper_screen_top_n": "PAPER_SCREEN_TOP_N",
    "paper_analyze_top_n": "PAPER_ANALYZE_TOP_N",
    "paper_use_claude_code_route": "PAPER_USE_CLAUDE_CODE_ROUTE",  # phase-cycle-5
    "paper_cycle_max_seconds": "PAPER_CYCLE_MAX_SECONDS",  # phase-cycle-7 (38.12)
    "paper_transaction_cost_pct": "PAPER_TRANSACTION_COST_PCT",
    "paper_daily_loss_limit_pct": "PAPER_DAILY_LOSS_LIMIT_PCT",
    "paper_trailing_dd_limit_pct": "PAPER_TRAILING_DD_LIMIT_PCT",
    "paper_min_cash_reserve_pct": "PAPER_MIN_CASH_RESERVE_PCT",
}


def _update_env_var(key: str, value: str) -> None:
    """Update or add a variable in the .env file."""
    if not _ENV_FILE.exists():
        _ENV_FILE.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    content = _ENV_FILE.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)

    if pattern.search(content):
        content = pattern.sub(f"{key}={value}", content)
    else:
        content = content.rstrip("\n") + f"\n{key}={value}\n"

    _ENV_FILE.write_text(content, encoding="utf-8")


def _settings_to_full(s: Settings) -> FullSettings:
    return FullSettings(
        gemini_model=s.gemini_model,
        deep_think_model=s.deep_think_model or s.gemini_model,
        apply_model_to_all_agents=bool(getattr(s, "apply_model_to_all_agents", False)),
        max_debate_rounds=s.max_debate_rounds,
        max_risk_debate_rounds=s.max_risk_debate_rounds,
        weight_corporate=s.weight_corporate,
        weight_industry=s.weight_industry,
        weight_valuation=s.weight_valuation,
        weight_sentiment=s.weight_sentiment,
        weight_governance=s.weight_governance,
        data_quality_min=s.data_quality_min,
        lite_mode=s.lite_mode,
        max_analysis_cost_usd=s.max_analysis_cost_usd,
        max_synthesis_iterations=s.max_synthesis_iterations,
        anthropic_key_configured=bool(getattr(s, "anthropic_api_key", "")),
        openai_key_configured=bool(getattr(s, "openai_api_key", "")),
        github_token_configured=bool(getattr(s, "github_token", "")),
        # phase-23.1 Signal Stack
        macro_regime_filter_enabled=bool(getattr(s, "macro_regime_filter_enabled", False)),
        macro_regime_model=getattr(s, "macro_regime_model", "claude-haiku-4-5"),
        pead_signal_enabled=bool(getattr(s, "pead_signal_enabled", False)),
        pead_signal_model=getattr(s, "pead_signal_model", "claude-haiku-4-5"),
        pead_signal_lookback_quarters=int(getattr(s, "pead_signal_lookback_quarters", 8)),
        news_screen_enabled=bool(getattr(s, "news_screen_enabled", False)),
        news_screen_model=getattr(s, "news_screen_model", "claude-haiku-4-5"),
        news_screen_max_headlines=int(getattr(s, "news_screen_max_headlines", 100)),
        sector_calendars_enabled=bool(getattr(s, "sector_calendars_enabled", False)),
        sector_calendars_lookahead_days=int(getattr(s, "sector_calendars_lookahead_days", 7)),
        meta_scorer_enabled=bool(getattr(s, "meta_scorer_enabled", False)),
        meta_scorer_model=getattr(s, "meta_scorer_model", "claude-haiku-4-5"),
        meta_scorer_max_batch=int(getattr(s, "meta_scorer_max_batch", 30)),
        # phase-23.1.9 Paper trading settings
        paper_starting_capital=float(getattr(s, "paper_starting_capital", 10000.0)),
        paper_trading_hour=int(getattr(s, "paper_trading_hour", 10)),  # phase-70.5
        paper_max_positions=int(getattr(s, "paper_max_positions", 10)),
        paper_max_per_sector=int(getattr(s, "paper_max_per_sector", 2)),  # phase-23.1.13
        paper_markets=list(getattr(s, "paper_markets", ["US"]) or ["US"]),  # phase-50.6
        paper_max_daily_cost_usd=float(getattr(s, "paper_max_daily_cost_usd", 2.0)),
        paper_default_stop_loss_pct=float(getattr(s, "paper_default_stop_loss_pct", 8.0)),
        paper_screen_top_n=int(getattr(s, "paper_screen_top_n", 10)),
        paper_analyze_top_n=int(getattr(s, "paper_analyze_top_n", 5)),
        paper_transaction_cost_pct=float(getattr(s, "paper_transaction_cost_pct", 0.1)),
        paper_daily_loss_limit_pct=float(getattr(s, "paper_daily_loss_limit_pct", 4.0)),
        paper_trailing_dd_limit_pct=float(getattr(s, "paper_trailing_dd_limit_pct", 10.0)),
        paper_min_cash_reserve_pct=float(getattr(s, "paper_min_cash_reserve_pct", 5.0)),
        # phase-cycle-5 (2026-05-26): Claude Code CLI rail flag exposure.
        paper_use_claude_code_route=bool(getattr(s, "paper_use_claude_code_route", False)),
        # phase-cycle-7 (38.12, 2026-05-27): cycle wall-clock budget exposure.
        paper_cycle_max_seconds=float(getattr(s, "paper_cycle_max_seconds", 7200.0)),
    )


# ── Full Settings endpoints ──────────────────────────────────────

@router.get("/", response_model=FullSettings)
async def get_all_settings(settings: Settings = Depends(get_settings)):
    """Get all configurable settings."""
    cache = get_api_cache()
    cache_key = "settings:full"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    result = _settings_to_full(settings)
    cache.set(cache_key, result, ENDPOINT_TTLS["settings:full"])
    return result


@router.put("/", response_model=FullSettings)
async def update_settings(body: SettingsUpdate):
    """Update any configurable settings by writing to .env and clearing cache."""
    # Validate model names against whitelist
    if body.gemini_model is not None and body.gemini_model not in _VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {body.gemini_model}")
    if body.deep_think_model is not None and body.deep_think_model not in _VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {body.deep_think_model}")

    # phase-23.1: Validate signal-stack model fields against the same whitelist
    for _field in ("macro_regime_model", "pead_signal_model", "news_screen_model", "meta_scorer_model"):
        _val = getattr(body, _field, None)
        if _val is not None and _val not in _VALID_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {_val}")

    # Validate pillar weights sum if any are being changed
    updates = body.model_dump(exclude_none=True)
    weight_fields = ["weight_corporate", "weight_industry", "weight_valuation", "weight_sentiment", "weight_governance"]
    weight_updates = {k: v for k, v in updates.items() if k in weight_fields}
    if weight_updates:
        # Load current weights, override with updates, check sum
        current = get_settings()
        merged = {
            "weight_corporate": current.weight_corporate,
            "weight_industry": current.weight_industry,
            "weight_valuation": current.weight_valuation,
            "weight_sentiment": current.weight_sentiment,
            "weight_governance": current.weight_governance,
        }
        merged.update(weight_updates)
        total = sum(merged.values())
        if abs(total - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail=f"Pillar weights must sum to 1.0, got {total:.2f}")

    # Write each updated field to .env
    for field_name, value in updates.items():
        env_key = _FIELD_TO_ENV.get(field_name)
        if env_key:
            if isinstance(value, bool):
                env_value = str(value).lower()
            elif isinstance(value, list):
                # phase-50.6: serialize list settings (e.g. paper_markets) as CSV;
                # the settings.py field_validator parses CSV/JSON/bracket forms.
                env_value = ",".join(str(x) for x in value)
            else:
                env_value = str(value)
            _update_env_var(env_key, env_value)

    # Clear cache so next request picks up new values
    get_settings.cache_clear()
    get_api_cache().invalidate("settings:*")
    settings = get_settings()

    # phase-70.5: reschedule the daily cron in-place when paper_trading_hour changed, so
    # the new hour takes effect WITHOUT a backend restart. Fail-open (never 500 the save).
    if "paper_trading_hour" in updates:
        try:
            from backend.api.paper_trading import reschedule_paper_job
            reschedule_paper_job(settings)
        except Exception as e:
            logger.error("phase-70.5: paper cron reschedule failed (fail-open): %r", e)

    logger.info("Settings updated: %s", list(updates.keys()))
    return _settings_to_full(settings)


# ── Legacy model-only endpoints (backward compatible) ─────────────

@router.get("/models", response_model=ModelConfig)
async def get_model_config(settings: Settings = Depends(get_settings)):
    """Get current model configuration."""
    return ModelConfig(
        gemini_model=settings.gemini_model,
        deep_think_model=settings.deep_think_model or settings.gemini_model,
        max_debate_rounds=settings.max_debate_rounds,
        max_risk_debate_rounds=settings.max_risk_debate_rounds,
        apply_model_to_all_agents=bool(getattr(settings, "apply_model_to_all_agents", False)),
    )


@router.put("/models", response_model=ModelConfig)
async def update_model_config(body: ModelConfigUpdate):
    """Update model configuration by writing to .env and clearing cache."""
    if body.gemini_model is not None and body.gemini_model not in _VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {body.gemini_model}")
    if body.deep_think_model is not None and body.deep_think_model not in _VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {body.deep_think_model}")

    if body.gemini_model is not None:
        _update_env_var("GEMINI_MODEL", body.gemini_model)
    if body.deep_think_model is not None:
        _update_env_var("DEEP_THINK_MODEL", body.deep_think_model)
    if body.apply_model_to_all_agents is not None:
        _update_env_var("APPLY_MODEL_TO_ALL_AGENTS", str(body.apply_model_to_all_agents).lower())

    get_settings.cache_clear()
    settings = get_settings()

    logger.info(
        "Model config updated: gemini_model=%s, deep_think_model=%s, apply_to_all=%s",
        settings.gemini_model,
        settings.deep_think_model,
        bool(getattr(settings, "apply_model_to_all_agents", False)),
    )

    return ModelConfig(
        gemini_model=settings.gemini_model,
        deep_think_model=settings.deep_think_model or settings.gemini_model,
        max_debate_rounds=settings.max_debate_rounds,
        max_risk_debate_rounds=settings.max_risk_debate_rounds,
        apply_model_to_all_agents=bool(getattr(settings, "apply_model_to_all_agents", False)),
    )


@router.get("/models/available")
async def get_available_models():
    """Get list of available models with pricing."""
    cache = get_api_cache()
    cache_key = "settings:models"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    cache.set(cache_key, AVAILABLE_MODELS, ENDPOINT_TTLS["settings:models"])
    return AVAILABLE_MODELS
