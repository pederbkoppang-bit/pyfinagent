"""
Daily macro regime filter — LLM-as-judge over a FRED snapshot.

Pulls 7-9 daily/monthly FRED series, hands them to Claude Haiku 4.5 with a
structured-output schema, returns a regime tag + conviction multiplier that
`screener.rank_candidates` applies to its composite score.

Cost target: <$0.05/day. 24-hour file cache prevents re-billing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.config.settings import get_settings
from backend.tools.fred_data import get_macro_indicators

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent / "_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_PATH = _CACHE_DIR / "macro_regime.json"
_CACHE_TTL_HOURS = 24

_REGIME_TAGS = ("risk_on", "risk_off", "mixed", "unknown")
_DEFAULT_MULTIPLIERS = {
    "risk_on": 1.15,
    "risk_off": 0.70,
    "mixed": 1.00,
    "unknown": 0.85,
}

_REGIME_SERIES = (
    "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2",
    "FEDFUNDS", "CPIAUCSL", "UNRATE", "INDPRO",
)

# phase-28.3: Caldara-Iacoviello GPR-Acts (geopolitical events) energy sector tilt
# Source: matteoiacoviello.com — monthly Excel, CC-BY 4.0. The GPRA column counts
# REALIZED geopolitical events (vs GPRT = threats). Since the US became a net oil
# exporter (late 2010s), GPRA spikes ASYMMETRICALLY benefit US energy majors (per
# Caldara-Iacoviello AER 2022 + IMF GFSR 2025). When triggered, we post-process the
# LLM's MacroRegimeOutput.sector_hints.overweight to inject the configured energy
# ETFs (default: XLE), deduped, preserving order. Threshold is a QUANTILE of the
# rolling 5-year history (default 0.90 = 90th pct) — a calibrated practitioner
# heuristic (no peer-reviewed paper validates the exact cutoff; the underlying
# directional mechanism is well-documented).
_GPR_URL_PRIMARY = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
_GPR_URL_FALLBACK = "https://www.matteoiacoviello.com/gpr_files/data_gpr.xls"
_GPR_CACHE_DIR = _CACHE_DIR / "gpr"
_GPR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_GPR_CACHE_PATH = _GPR_CACHE_DIR / "data_gpr_export.xls"
_GPR_ROLLING_MONTHS = 60

# phase-28.6: WTI crude (CL=F) 1-month momentum secondary trigger. Orthogonal to
# phase-28.3 GPR-Acts: high-GPR/flat-oil and rising-oil/low-GPR both occur, so
# this is a genuinely additive (not duplicate) trigger. Computed as z-score of
# the trailing 21d cumulative percent change over the rolling 252d distribution.
# When the z-score exceeds threshold (default 1.0), the configured energy ETFs
# are injected into sector_hints.overweight via the SAME _apply_gpr_tilt helper
# (it's generic over `above_threshold`). yfinance >=0.2.40 is already a dep.
_CRUDE_CACHE_DIR = _CACHE_DIR / "crude"
_CRUDE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CRUDE_CACHE_PATH = _CRUDE_CACHE_DIR / "crude_momentum.json"


class SectorWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overweight: list[str] = Field(
        default_factory=list,
        description="Sector ETF tickers to overweight in this regime (e.g. ['XLU','XLP','XLV']). Max 5.",
    )
    underweight: list[str] = Field(
        default_factory=list,
        description="Sector ETF tickers to underweight in this regime. Max 5.",
    )


class MacroRegimeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(
        description="Free-text explanation max 300 chars. Name the signals that drove the call and any conflicts.",
        max_length=300,
    )
    regime: Literal["risk_on", "risk_off", "mixed", "unknown"]
    conviction: float = Field(
        ge=0.0, le=1.0,
        description="0.0-1.0 confidence. 1.0 = all signals agree; 0.5 = mixed; 0.0 = insufficient data.",
    )
    conviction_multiplier: float = Field(
        ge=0.5, le=1.5,
        description="Applied as scalar in screener.rank_candidates. Defaults: risk_on=1.15, mixed=1.0, risk_off=0.7, unknown=0.85.",
    )
    sector_hints: SectorWeights
    series_used: list[str] = Field(
        description="FRED series IDs that were available (e.g. ['T10Y2Y','VIXCLS','BAMLH0A0HYM2']).",
    )
    computed_at: str = Field(description="ISO-8601 UTC timestamp when this regime was computed.")


async def _fetch_gpr_acts(cache_hours: int = 24, quantile: float = 0.90) -> Optional[dict]:
    """phase-28.3: Fetch latest GPR-Acts value from matteoiacoviello.com.

    Returns a dict with keys:
        current: float -- latest GPRA value
        threshold: float -- quantile-th value over the trailing _GPR_ROLLING_MONTHS
        last_date: str -- ISO date of the latest observation
        above_threshold: bool -- current > threshold

    Returns None on any unrecoverable error (network, parse, missing column).
    Uses a local file cache that auto-refreshes after `cache_hours`.
    """
    cache_fresh = False
    if _GPR_CACHE_PATH.exists():
        try:
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                _GPR_CACHE_PATH.stat().st_mtime, tz=timezone.utc
            )
            cache_fresh = age < timedelta(hours=cache_hours)
        except Exception:
            cache_fresh = False

    if not cache_fresh:
        import httpx
        ok = False
        for url in (_GPR_URL_PRIMARY, _GPR_URL_FALLBACK):
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                    resp = await client.get(url, headers={"User-Agent": "PyFinAgent/2.0 (phase-28.3)"})
                if resp.status_code == 200 and len(resp.content) > 1024:
                    _GPR_CACHE_PATH.write_bytes(resp.content)
                    ok = True
                    logger.info("GPR Excel downloaded (%d bytes) from %s", len(resp.content), url)
                    break
            except Exception as e:
                logger.debug("GPR fetch failed from %s: %s", url, e)
                continue
        if not ok:
            logger.warning("GPR fetch: all sources failed; falling back to cached file if present")
            if not _GPR_CACHE_PATH.exists():
                return None

    try:
        import pandas as pd
        df = pd.read_excel(_GPR_CACHE_PATH)
    except Exception as e:
        logger.warning("GPR Excel parse failed: %s", e)
        return None

    if "GPRA" not in df.columns:
        # Some versions of the file expose GPR_ACTS or different casing -- fallback search
        alt = next((c for c in df.columns if "act" in str(c).lower() or "gpra" in str(c).lower()), None)
        if alt is None:
            logger.warning("GPR Excel missing GPRA column; cols=%s", list(df.columns)[:10])
            return None
        df = df.rename(columns={alt: "GPRA"})

    df = df.dropna(subset=["GPRA"])
    if df.empty:
        return None
    series = df["GPRA"].astype(float)
    current = float(series.iloc[-1])
    trailing = series.tail(_GPR_ROLLING_MONTHS)
    threshold = float(trailing.quantile(quantile)) if len(trailing) >= 12 else float(series.quantile(quantile))

    last_date = ""
    for date_col in ("month", "Month", "date", "Date"):
        if date_col in df.columns:
            try:
                last_date = str(df[date_col].iloc[-1])
                break
            except Exception:
                pass

    return {
        "current": current,
        "threshold": threshold,
        "last_date": last_date,
        "above_threshold": current > threshold,
        "rolling_n": int(len(trailing)),
        "quantile": quantile,
    }


async def _fetch_crude_momentum(
    cache_hours: int = 24,
    window_days: int = 21,
    lookback_days: int = 252,
    zscore_threshold: float = 1.0,
) -> Optional[dict]:
    """phase-28.6: Fetch WTI crude (CL=F) 1-month momentum z-score.

    Returns a dict with keys:
        current_momentum: float -- trailing `window_days` percent change of CL=F close
        zscore: float -- z-score of current momentum vs rolling `lookback_days` distribution
        threshold: float -- the configured zscore threshold
        last_date: str -- date of the latest close used
        above_threshold: bool -- zscore > threshold

    Returns None on any unrecoverable error (network, parse).
    """
    # Cache fresh? -> return parsed cached dict
    if _CRUDE_CACHE_PATH.exists():
        try:
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                _CRUDE_CACHE_PATH.stat().st_mtime, tz=timezone.utc
            )
            if age < timedelta(hours=cache_hours):
                cached = json.loads(_CRUDE_CACHE_PATH.read_text(encoding="utf-8"))
                logger.info("Crude momentum cache hit: zscore=%.2f above=%s",
                            cached.get("zscore", float("nan")), cached.get("above_threshold"))
                return cached
        except Exception as e:
            logger.debug("Crude cache unreadable: %s", e)

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; crude_momentum unavailable")
        return None

    try:
        period_days = max(lookback_days + window_days + 30, 365)
        period = f"{period_days}d" if period_days <= 365 else "2y"
        df = await asyncio.to_thread(
            lambda: yf.download("CL=F", period=period, interval="1d",
                                auto_adjust=True, progress=False)
        )
    except Exception as e:
        logger.warning("yfinance CL=F download failed: %s", e)
        return None

    if df is None or len(df) == 0:
        logger.warning("yfinance CL=F returned empty")
        return None

    try:
        close = df["Close"].dropna()
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if len(close) < (window_days + 10):
            logger.warning("CL=F insufficient history: %d closes", len(close))
            return None

        rolling_mom = close.pct_change(periods=window_days).dropna()
        current = float(rolling_mom.iloc[-1])

        recent = rolling_mom.tail(lookback_days)
        mean = float(recent.mean())
        std = float(recent.std())
        zscore = (current - mean) / std if std and std > 1e-9 else 0.0

        result = {
            "current_momentum": round(current, 6),
            "zscore": round(zscore, 4),
            "mean": round(mean, 6),
            "std": round(std, 6),
            "threshold": float(zscore_threshold),
            "last_date": str(close.index[-1]),
            "above_threshold": zscore > zscore_threshold,
            "n_observations": int(len(recent)),
            "window_days": window_days,
            "lookback_days": lookback_days,
        }
    except Exception as e:
        logger.warning("Crude momentum compute failed: %s", e)
        return None

    try:
        _CRUDE_CACHE_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Crude cache write failed: %s", e)
    logger.info(
        "Crude momentum: current=%.3f%% zscore=%+.2f above=%s",
        result["current_momentum"] * 100, result["zscore"], result["above_threshold"],
    )
    return result


def _apply_gpr_tilt(parsed: "MacroRegimeOutput", gpr_info: dict, sector_etfs_csv: str) -> "MacroRegimeOutput":
    """phase-28.3 (reused by phase-28.6): When `gpr_info["above_threshold"]` is True,
    inject configured energy ETFs into sector_hints.overweight (deduped, preserving order).
    Function is generic over the trigger info dict — only `above_threshold` is read,
    so it works for GPR-Acts AND for crude momentum z-score.

    Identity if `above_threshold` is False.
    """
    if not gpr_info.get("above_threshold"):
        return parsed
    etfs = [e.strip().upper() for e in sector_etfs_csv.split(",") if e.strip()]
    if not etfs:
        return parsed
    existing = list(parsed.sector_hints.overweight)
    for e in etfs:
        if e not in existing:
            existing.append(e)
    new_hints = SectorWeights(overweight=existing, underweight=parsed.sector_hints.underweight)
    return parsed.model_copy(update={"sector_hints": new_hints})


def _load_cache() -> Optional[MacroRegimeOutput]:
    if not _CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        computed_at = datetime.fromisoformat(raw["computed_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) - computed_at > timedelta(hours=_CACHE_TTL_HOURS):
            return None
        return MacroRegimeOutput.model_validate(raw)
    except Exception as e:
        logger.warning("Macro regime cache unreadable: %s", e)
        return None


_UNSUPPORTED_SCHEMA_KEYS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "maxLength", "minLength",
)


def _strip_unsupported_schema_keys(node):
    """Recursively remove JSON-schema keys Anthropic structured outputs rejects."""
    if isinstance(node, dict):
        for k in _UNSUPPORTED_SCHEMA_KEYS:
            node.pop(k, None)
        for v in node.values():
            _strip_unsupported_schema_keys(v)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported_schema_keys(item)
    return node


def _save_cache(regime: MacroRegimeOutput) -> None:
    try:
        _CACHE_PATH.write_text(regime.model_dump_json(indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Macro regime cache write failed: %s", e)


_NETLIQ_CACHE_PATH = _CACHE_DIR / "net_liquidity.json"


async def _fetch_net_liquidity(fred_key: str, cache_hours: int = 24) -> Optional[dict]:
    """phase-69.3 (audit item 6): Fed NET LIQUIDITY = WALCL - WTREGEN - RRPONTSYD*1000,
    all in MILLIONS USD (RRPONTSYD is reported in BILLIONS on FRED -> x1000). A usable-
    dollar-liquidity proxy that historically correlates strongly with risk-asset prices
    (netliquidity.org / macrolighthouse / eco3min). 24h file cache; uses the existing
    free FRED key; writes NO BQ table (historical_macro untouched). Mirrors the
    _fetch_gpr_acts / _fetch_crude_momentum cache idiom. None on any failure."""
    if _NETLIQ_CACHE_PATH.exists():
        try:
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                _NETLIQ_CACHE_PATH.stat().st_mtime, tz=timezone.utc)
            if age < timedelta(hours=cache_hours):
                return json.loads(_NETLIQ_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        from backend.tools.fred_data import _fetch_series

        def _cur_prev(obs: list):
            if not obs:
                return None, None
            cur = obs[0]["value"]
            prev = obs[min(3, len(obs) - 1)]["value"] if len(obs) > 1 else cur
            return cur, prev

        walcl = (await _fetch_series("WALCL", fred_key)).get("observations", [])
        wtregen = (await _fetch_series("WTREGEN", fred_key)).get("observations", [])
        rrp = (await _fetch_series("RRPONTSYD", fred_key)).get("observations", [])
        (w, pw), (t, pt), (r, pr) = _cur_prev(walcl), _cur_prev(wtregen), _cur_prev(rrp)
        if w is None or t is None or r is None:
            return None
        # UNIT GOTCHA: WALCL/WTREGEN are millions; RRPONTSYD is billions -> x1000.
        net_liq = float(w) - float(t) - float(r) * 1000.0
        prev_nl = None
        if pw is not None and pt is not None and pr is not None:
            prev_nl = float(pw) - float(pt) - float(pr) * 1000.0
        trend = "flat"
        if prev_nl is not None:
            trend = "rising" if net_liq > prev_nl else ("falling" if net_liq < prev_nl else "flat")
        out = {
            "net_liquidity_musd": round(net_liq, 1),
            "previous_musd": (round(prev_nl, 1) if prev_nl is not None else None),
            "trend": trend,
            "as_of": (walcl[0]["date"] if walcl else None),
        }
        try:
            _NETLIQ_CACHE_PATH.write_text(json.dumps(out), encoding="utf-8")
        except Exception:
            pass
        return out
    except Exception as e:
        logger.warning("net_liquidity fetch failed: %s", e)
        return None


def _build_prompt(indicators: dict, *, net_liquidity: Optional[dict] = None,
                  include_indpro: bool = False) -> str:
    lines = [
        "Classify the current US macro regime for equity portfolio sizing.",
        "",
        "Available FRED indicators (current value, prior, trend):",
    ]
    for sid in _REGIME_SERIES:
        # phase-69.3: INDPRO's regime-prompt inclusion is gated behind the
        # regime_net_liquidity flag so the LIVE prompt is byte-identical when OFF
        # (INDPRO was dead pre-69.3). include_indpro=True only when the flag is on.
        if sid == "INDPRO" and not include_indpro:
            continue
        info = indicators.get(sid)
        if not info or "current" not in info:
            continue
        lines.append(
            f"- {sid} ({info.get('name', sid)}): "
            f"current={info['current']:.3f} previous={info.get('previous', 'n/a')} "
            f"trend={info.get('trend', 'n/a')} as_of={info.get('date', 'n/a')}"
        )
    # phase-69.3: net-liquidity indicator (flag-gated; net_liquidity is None when
    # regime_net_liquidity is OFF -> prompt byte-identical).
    if net_liquidity and net_liquidity.get("net_liquidity_musd") is not None:
        lines.append(
            f"- NET_LIQUIDITY (Fed WALCL-TGA-RRP, $M): "
            f"current={net_liquidity['net_liquidity_musd']:.0f} "
            f"trend={net_liquidity.get('trend', 'n/a')} as_of={net_liquidity.get('as_of', 'n/a')} "
            f"[rising -> more usable dollars chasing risk assets (risk_on lean); falling -> tightening (risk_off lean)]"
        )
    lines += [
        "",
        "Decision thresholds (canonical, from research-brief phase-23.1.1):",
        "- T10Y2Y < 0 -> recession risk (risk_off lean)",
        "- VIXCLS > 25 -> elevated fear (risk_off); VIXCLS < 18 -> calm (risk_on lean)",
        "- BAMLH0A0HYM2 > 5.0 -> credit stress (risk_off); < 3.5 -> credit easy (risk_on)",
        "- FEDFUNDS rising -> tightening (mild risk_off lean); falling -> easing (risk_on lean)",
        "- UNRATE rising trend -> growth slowing (risk_off lean)",
        "- CPIAUCSL YoY > 3.5% -> inflationary pressure (risk_off lean)",
        "",
        "Sector tilts: risk_off favors XLU/XLP/XLV/XLE (defensives + energy); "
        "risk_on favors XLK/XLY/XLF/XLI (cyclicals); mixed = neutral.",
        "",
        "Return JSON ONLY matching the MacroRegimeOutput schema. "
        "Generate the rationale FIRST, then commit to the regime tag. "
        "Set conviction high (>=0.75) only when 4+ signals align; "
        "use 'mixed' when 2-3 signals conflict; use 'unknown' when fewer than 3 series are available.",
    ]
    return "\n".join(lines)


def _fallback_regime(indicators: dict, reason: str) -> MacroRegimeOutput:
    series_used = [sid for sid in _REGIME_SERIES if indicators.get(sid, {}).get("current") is not None]
    return MacroRegimeOutput(
        rationale=f"Fallback: {reason}"[:300],
        regime="unknown",
        conviction=0.0,
        conviction_multiplier=_DEFAULT_MULTIPLIERS["unknown"],
        sector_hints=SectorWeights(),
        series_used=series_used,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


async def compute_macro_regime(use_cache: bool = True) -> MacroRegimeOutput:
    """Compute today's macro regime tag. Caches for 24 hours.

    Returns a `MacroRegimeOutput`. On any failure (no FRED key, no Anthropic key,
    LLM error, schema parse error) returns a `regime='unknown'` fallback so the
    screener stays operational.
    """
    if use_cache:
        cached = _load_cache()
        if cached is not None:
            logger.info("Macro regime cache hit: %s (mult=%s)", cached.regime, cached.conviction_multiplier)
            return cached

    settings = get_settings()
    fred_key = getattr(settings, "fred_api_key", "") or ""
    if not fred_key:
        return _fallback_regime({}, "FRED_API_KEY not configured")

    indicators_payload = await get_macro_indicators(fred_key)
    if not indicators_payload.get("available"):
        return _fallback_regime({}, indicators_payload.get("summary", "FRED unavailable"))
    indicators = indicators_payload.get("indicators", {})

    available = [sid for sid in _REGIME_SERIES if indicators.get(sid, {}).get("current") is not None]
    if len(available) < 3:
        return _fallback_regime(indicators, f"only {len(available)} regime series available")

    # phase-51.1: unwrap SecretStr (truthy wrapper bypassed `or ""` -> SDK header error).
    from backend.agents.llm_client import unwrap_secret
    anthropic_key = unwrap_secret(getattr(settings, "anthropic_api_key", ""))
    if not anthropic_key:
        return _fallback_regime(indicators, "ANTHROPIC_API_KEY not configured")

    from backend.agents.llm_client import ClaudeClient
    client = ClaudeClient(
        model_name=getattr(settings, "macro_regime_model", "claude-haiku-4-5"),
        api_key=anthropic_key,
        enable_prompt_caching=False,
    )

    # phase-69.3 (audit item 6): net-liquidity + INDPRO regime lift, flag-gated
    # (regime_net_liquidity default-OFF -> the regime prompt is byte-identical:
    # net_liquidity=None and INDPRO excluded). Uses the existing free FRED key; the
    # new _fetch_net_liquidity path writes NO BQ table (historical_macro untouched).
    _nl_on = bool(getattr(settings, "regime_net_liquidity", False))
    _net_liq = await _fetch_net_liquidity(fred_key) if _nl_on else None
    prompt = _build_prompt(indicators, net_liquidity=_net_liq, include_indpro=_nl_on)
    cleaned_schema = _strip_unsupported_schema_keys(MacroRegimeOutput.model_json_schema())
    try:
        response = await asyncio.to_thread(
            client.generate_content,
            prompt,
            {
                "response_schema": cleaned_schema,
                "response_mime_type": "application/json",
                "max_output_tokens": 512,
                "temperature": 0.0,
            },
        )
    except Exception as e:
        logger.warning("Macro regime LLM call failed: %s", e)
        return _fallback_regime(indicators, f"LLM error: {type(e).__name__}")

    # Clamp / coerce raw fields BEFORE Pydantic validation, since Anthropic
    # structured outputs cannot enforce numerical or string-length constraints.
    try:
        raw = json.loads(response.text)
        if isinstance(raw.get("rationale"), str):
            raw["rationale"] = raw["rationale"][:300]
        if isinstance(raw.get("conviction"), (int, float)):
            raw["conviction"] = max(0.0, min(1.0, float(raw["conviction"])))
        if isinstance(raw.get("conviction_multiplier"), (int, float)):
            raw["conviction_multiplier"] = max(0.5, min(1.5, float(raw["conviction_multiplier"])))
        parsed = MacroRegimeOutput.model_validate(raw)
    except Exception as e:
        logger.warning("Macro regime LLM output unparsable: %s | raw=%s", e, response.text[:200])
        return _fallback_regime(indicators, "LLM output parse error")

    if not parsed.series_used:
        parsed = parsed.model_copy(update={"series_used": available})

    # phase-28.3: Optional GPR-Acts post-process. When enabled AND latest GPRA exceeds
    # the configured quantile threshold, inject XLE (or configured ETFs) into
    # sector_hints.overweight. Identity when disabled or fetch fails.
    if getattr(settings, "gpr_signal_enabled", False):
        try:
            gpr_info = await _fetch_gpr_acts(
                cache_hours=getattr(settings, "gpr_signal_cache_hours", 24),
                quantile=getattr(settings, "gpr_signal_quantile", 0.90),
            )
            if gpr_info:
                pre_overweight = list(parsed.sector_hints.overweight)
                parsed = _apply_gpr_tilt(parsed, gpr_info, getattr(settings, "gpr_signal_sector_etfs", "XLE"))
                logger.info(
                    "GPR tilt: current=%.2f threshold=%.2f above=%s; overweight %s -> %s",
                    gpr_info["current"], gpr_info["threshold"], gpr_info["above_threshold"],
                    pre_overweight, list(parsed.sector_hints.overweight),
                )
        except Exception as e:
            logger.warning("GPR tilt application failed (non-fatal): %s", e)

    # phase-28.6: Optional WTI crude (CL=F) 1m-momentum post-process. Orthogonal to GPR.
    # When enabled AND z-score exceeds threshold, inject configured energy ETFs into
    # sector_hints.overweight via _apply_gpr_tilt (generic over above_threshold).
    if getattr(settings, "crude_momentum_enabled", False):
        try:
            crude_info = await _fetch_crude_momentum(
                cache_hours=getattr(settings, "crude_momentum_cache_hours", 24),
                window_days=getattr(settings, "crude_momentum_window_days", 21),
                lookback_days=getattr(settings, "crude_momentum_lookback_days", 252),
                zscore_threshold=getattr(settings, "crude_momentum_zscore_threshold", 1.0),
            )
            if crude_info:
                pre_overweight = list(parsed.sector_hints.overweight)
                parsed = _apply_gpr_tilt(parsed, crude_info,
                                         getattr(settings, "crude_momentum_sector_etfs", "XLE"))
                logger.info(
                    "Crude momentum tilt: zscore=%+.2f threshold=%.2f above=%s; overweight %s -> %s",
                    crude_info["zscore"], crude_info["threshold"], crude_info["above_threshold"],
                    pre_overweight, list(parsed.sector_hints.overweight),
                )
        except Exception as e:
            logger.warning("Crude momentum tilt application failed (non-fatal): %s", e)

    _save_cache(parsed)
    logger.info(
        "Macro regime computed: %s conviction=%.2f mult=%.2f series=%d",
        parsed.regime, parsed.conviction, parsed.conviction_multiplier, len(parsed.series_used),
    )
    return parsed


def apply_regime_to_score(
    base_score: float,
    sector: Optional[str],
    sector_etf_for: dict[str, str],
    regime: Optional[MacroRegimeOutput],
) -> float:
    """Apply regime multiplier and sector tilt to a screener composite score.

    Args:
        base_score: Original screener score.
        sector: GICS-style sector name for the candidate ticker (may be None).
        sector_etf_for: Mapping of sector name -> SPDR ETF ticker (e.g. screener.SECTOR_ETFS).
        regime: MacroRegimeOutput or None (no regime applied if None).
    """
    if regime is None:
        return base_score
    # phase-69.3: sign-safe (flag-gated default-OFF = byte-identical base*mult).
    from backend.services.overlay_math import sign_safe_mult
    score = sign_safe_mult(base_score, regime.conviction_multiplier)
    if sector and sector_etf_for:
        etf = sector_etf_for.get(sector)
        if etf:
            if etf in regime.sector_hints.overweight:
                score = sign_safe_mult(score, 1.05)
            elif etf in regime.sector_hints.underweight:
                score = sign_safe_mult(score, 0.95)
    return score
