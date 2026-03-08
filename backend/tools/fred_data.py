"""
FRED economic data tool — Federal Reserve Economic Data API.
Fetches key macro indicators for economic cycle positioning.
"""

import logging
from datetime import datetime, timedelta

import httpx

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Key economic series
SERIES = {
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "CPI (All Urban Consumers)",
    "UNRATE": "Unemployment Rate",
    "GDP": "Real GDP",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "UMCSENT": "Consumer Sentiment",
    "DGS10": "10-Year Treasury Yield",
}


async def _fetch_series(series_id: str, api_key: str, periods: int = 12) -> dict:
    """Fetch the last N observations for a FRED series."""
    start = (datetime.utcnow() - timedelta(days=periods * 35)).strftime("%Y-%m-%d")
    url = (
        f"{FRED_BASE}?series_id={series_id}"
        f"&api_key={api_key}&file_type=json"
        f"&observation_start={start}&sort_order=desc&limit={periods}"
    )
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    observations = data.get("observations", [])
    values = []
    for obs in observations:
        val = obs.get("value", ".")
        if val != ".":
            values.append({"date": obs["date"], "value": float(val)})

    return {"series_id": series_id, "observations": values}


async def get_macro_indicators(api_key: str) -> dict:
    """
    Fetch all key macro indicators from FRED.
    Returns structured data with trend analysis.
    """
    if not api_key:
        return {
            "available": False,
            "signal": "NO_API_KEY",
            "summary": "FRED API key not configured. Set FRED_API_KEY in .env.",
        }

    results = {}
    for series_id, name in SERIES.items():
        try:
            data = await _fetch_series(series_id, api_key)
            obs = data["observations"]
            if obs:
                current = obs[0]["value"]
                prev = obs[min(3, len(obs) - 1)]["value"] if len(obs) > 1 else current
                trend = "rising" if current > prev else "falling" if current < prev else "stable"
                results[series_id] = {
                    "name": name,
                    "current": current,
                    "previous": prev,
                    "trend": trend,
                    "date": obs[0]["date"],
                }
        except Exception as e:
            logger.warning("Failed to fetch FRED series %s: %s", series_id, e)
            results[series_id] = {"name": name, "error": str(e)}

    # Compute macro signal
    fed_rate = results.get("FEDFUNDS", {})
    spread = results.get("T10Y2Y", {})
    unemployment = results.get("UNRATE", {})
    sentiment = results.get("UMCSENT", {})

    signal = "NEUTRAL"
    warnings = []

    # Yield curve inversion check
    spread_val = spread.get("current")
    if spread_val is not None and spread_val < 0:
        signal = "DEFENSIVE"
        warnings.append("Yield curve inverted — recession risk elevated")

    # Rising unemployment
    if unemployment.get("trend") == "rising":
        warnings.append("Unemployment rising — labor market weakening")

    # Falling consumer sentiment
    if sentiment.get("trend") == "falling":
        warnings.append("Consumer sentiment declining")

    # Rate environment
    if fed_rate.get("trend") == "falling":
        if signal != "DEFENSIVE":
            signal = "EASING"

    summary_parts = []
    for sid, info in results.items():
        if "current" in info:
            summary_parts.append(f"{info['name']}: {info['current']:.2f} ({info['trend']})")

    return {
        "available": True,
        "indicators": results,
        "signal": signal,
        "warnings": warnings,
        "summary": (
            f"Macro signal: {signal}. {'; '.join(summary_parts)}. "
            f"Warnings: {'; '.join(warnings) if warnings else 'None'}."
        ),
    }
