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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings", tags=["settings"])

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Valid model names — whitelist to prevent arbitrary writes to .env
# Includes Gemini (direct), GitHub Models catalog, Anthropic direct, OpenAI direct
_VALID_MODELS = {
    # Gemini (Vertex AI)
    "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro",
    # GitHub Models (OpenAI-compatible endpoint, served via Copilot Pro)
    "gpt-4o", "gpt-4o-mini",
    "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219", "claude-sonnet-4-6",
    "meta-llama-3.1-405b-instruct", "meta-llama-3.1-70b-instruct", "meta-llama-3.1-8b-instruct",
    "phi-4", "phi-3.5-moe-instruct", "phi-3.5-mini-instruct",
    "mistral-large-2407", "mistral-nemo",
    # Anthropic direct
    # (claude-* already covered above via GitHub Models; same names work for both)
    # OpenAI direct
    "o1", "o1-mini", "o3-mini",
}


class ModelConfig(BaseModel):
    """Readable model configuration."""
    gemini_model: str
    deep_think_model: str
    max_debate_rounds: int
    max_risk_debate_rounds: int


class FullSettings(BaseModel):
    """Complete readable settings for the Settings UI."""
    # Models
    gemini_model: str
    deep_think_model: str
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


class SettingsUpdate(BaseModel):
    """Writable fields for the Settings UI. All optional — only provided fields are updated."""
    gemini_model: Optional[str] = None
    deep_think_model: Optional[str] = None
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


class ModelConfigUpdate(BaseModel):
    """Legacy writable model configuration (kept for backward compatibility)."""
    gemini_model: Optional[str] = None
    deep_think_model: Optional[str] = None


class ModelPricing(BaseModel):
    """Per-model pricing info."""
    model: str
    provider: str = "Gemini"
    input_per_1m: float
    output_per_1m: float


AVAILABLE_MODELS = [
    # Gemini (Vertex AI) — always available via Application Default Credentials
    {"model": "gemini-2.0-flash",  "provider": "Gemini", "input_per_1m": 0.10, "output_per_1m": 0.40},
    {"model": "gemini-2.5-flash",  "provider": "Gemini", "input_per_1m": 0.15, "output_per_1m": 0.60},
    {"model": "gemini-2.5-pro",    "provider": "Gemini", "input_per_1m": 1.25, "output_per_1m": 10.00},
    # GitHub Models — requires GITHUB_TOKEN + Copilot Pro subscription
    {"model": "gpt-4o",                       "provider": "GitHub Models", "input_per_1m": 2.50,  "output_per_1m": 10.00},
    {"model": "gpt-4o-mini",                  "provider": "GitHub Models", "input_per_1m": 0.15,  "output_per_1m": 0.60},
    {"model": "claude-3-5-sonnet-20241022",   "provider": "GitHub Models", "input_per_1m": 3.00,  "output_per_1m": 15.00},
    {"model": "claude-3-5-haiku-20241022",    "provider": "GitHub Models", "input_per_1m": 0.80,  "output_per_1m": 4.00},
    {"model": "claude-3-7-sonnet-20250219",   "provider": "GitHub Models", "input_per_1m": 3.00,  "output_per_1m": 15.00},
    {"model": "claude-sonnet-4-6",            "provider": "GitHub Models", "input_per_1m": 3.00,  "output_per_1m": 15.00},
    {"model": "meta-llama-3.1-405b-instruct", "provider": "GitHub Models", "input_per_1m": 5.00,  "output_per_1m": 15.00},
    {"model": "meta-llama-3.1-70b-instruct",  "provider": "GitHub Models", "input_per_1m": 0.90,  "output_per_1m": 0.90},
    {"model": "phi-4",                        "provider": "GitHub Models", "input_per_1m": 0.07,  "output_per_1m": 0.14},
    {"model": "mistral-large-2407",           "provider": "GitHub Models", "input_per_1m": 3.00,  "output_per_1m": 9.00},
    # Anthropic direct — requires ANTHROPIC_API_KEY
    # (claude-* models above also work via ANTHROPIC_API_KEY when GITHUB_TOKEN is not set)
    # OpenAI direct — requires OPENAI_API_KEY
    {"model": "o1",      "provider": "OpenAI", "input_per_1m": 15.00, "output_per_1m": 60.00},
    {"model": "o1-mini", "provider": "OpenAI", "input_per_1m": 3.00,  "output_per_1m": 12.00},
    {"model": "o3-mini", "provider": "OpenAI", "input_per_1m": 1.10,  "output_per_1m": 4.40},
]

# Mapping from SettingsUpdate field names to .env variable names
_FIELD_TO_ENV = {
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
    )


# ── Full Settings endpoints ──────────────────────────────────────

@router.get("/", response_model=FullSettings)
async def get_all_settings(settings: Settings = Depends(get_settings)):
    """Get all configurable settings."""
    return _settings_to_full(settings)


@router.put("/", response_model=FullSettings)
async def update_settings(body: SettingsUpdate):
    """Update any configurable settings by writing to .env and clearing cache."""
    # Validate model names against whitelist
    if body.gemini_model is not None and body.gemini_model not in _VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {body.gemini_model}")
    if body.deep_think_model is not None and body.deep_think_model not in _VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {body.deep_think_model}")

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
            env_value = str(value).lower() if isinstance(value, bool) else str(value)
            _update_env_var(env_key, env_value)

    # Clear cache so next request picks up new values
    get_settings.cache_clear()
    settings = get_settings()

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

    get_settings.cache_clear()
    settings = get_settings()

    logger.info("Model config updated: gemini_model=%s, deep_think_model=%s",
                settings.gemini_model, settings.deep_think_model)

    return ModelConfig(
        gemini_model=settings.gemini_model,
        deep_think_model=settings.deep_think_model or settings.gemini_model,
        max_debate_rounds=settings.max_debate_rounds,
        max_risk_debate_rounds=settings.max_risk_debate_rounds,
    )


@router.get("/models/available")
async def get_available_models():
    """Get list of available models with pricing."""
    return AVAILABLE_MODELS
