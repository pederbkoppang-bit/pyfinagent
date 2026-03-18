"""
Info-Gap Detection — AlphaQuanter-style ReAct iterative loop.

Scans enrichment results for missing/failed data sources, assesses criticality,
and retries critical failures before proceeding to debate + synthesis.

Research basis: AlphaQuanter (NBER) — single-agent ReAct-style info-gap detection
with iterative retry loop for data completeness.
"""

import asyncio
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Enrichment sources and their criticality for different sectors
_SOURCE_CRITICALITY = {
    "insider": "HIGH",
    "options": "HIGH",
    "social_sentiment": "MEDIUM",
    "patent": "MEDIUM",
    "earnings_tone": "HIGH",
    "fred_macro": "MEDIUM",
    "alt_data": "LOW",
    "sector": "HIGH",
    "nlp_sentiment": "MEDIUM",
    "anomaly": "HIGH",
    "monte_carlo": "HIGH",
}

# Sectors where certain signals are more critical
_SECTOR_OVERRIDES = {
    "Technology": {"patent": "HIGH"},
    "Healthcare": {"patent": "HIGH"},
    "Financial Services": {"fred_macro": "HIGH", "patent": "LOW"},
    "Energy": {"fred_macro": "HIGH", "alt_data": "MEDIUM"},
    "Consumer Cyclical": {"alt_data": "MEDIUM", "social_sentiment": "HIGH"},
}


def _assess_source_status(key: str, data: dict) -> str:
    """Classify a data source as SUFFICIENT, PARTIAL, MISSING, or SKIPPED."""
    if not data or not isinstance(data, dict):
        return "MISSING"
    if data.get("signal") == "ERROR":
        return "MISSING"
    if data.get("signal") == "SKIPPED":
        return "SKIPPED"
    summary = data.get("summary", "")
    if "error" in summary.lower() or "failed" in summary.lower():
        return "MISSING"
    if data.get("signal") == "N/A" and not summary:
        return "PARTIAL"
    return "SUFFICIENT"


def detect_info_gaps(
    enrichment_data: dict,
    sector: str = "",
) -> dict:
    """
    Scan enrichment results for information gaps.

    Args:
        enrichment_data: Dict of source_key -> raw data dict from Step 6
        sector: Company sector for criticality overrides

    Returns:
        Info gap report dict with gaps, quality score, critical gaps list
    """
    gaps = []
    critical_gaps = []
    sufficient_count = 0
    skipped_count = 0
    total = len(_SOURCE_CRITICALITY)

    sector_overrides = _SECTOR_OVERRIDES.get(sector, {})

    for key, default_crit in _SOURCE_CRITICALITY.items():
        data = enrichment_data.get(key, {})
        status = _assess_source_status(key, data)
        criticality = sector_overrides.get(key, default_crit)

        gap_entry = {
            "source": key,
            "status": status,
            "criticality": criticality,
            "impact": _describe_impact(key, status, criticality),
        }
        gaps.append(gap_entry)

        if status == "SKIPPED":
            skipped_count += 1
        elif status == "SUFFICIENT":
            sufficient_count += 1
        elif status == "MISSING" and criticality == "HIGH":
            critical_gaps.append(key)

    # Exclude skipped tools from the quality denominator
    effective_total = total - skipped_count
    data_quality_score = round(sufficient_count / effective_total, 2) if effective_total > 0 else 0.0

    return {
        "gaps": gaps,
        "data_quality_score": data_quality_score,
        "critical_gaps": critical_gaps,
        "recommendation_at_risk": len(critical_gaps) >= 3,
        "summary": _build_summary(sufficient_count, effective_total, critical_gaps),
    }


async def retry_critical_gaps(
    critical_gaps: list[str],
    retry_funcs: dict[str, Callable],
    max_retries: int = 2,
    on_progress: Optional[Callable] = None,
) -> dict:
    """
    Retry failed critical data sources.

    Args:
        critical_gaps: List of source keys that failed and are critical
        retry_funcs: Dict of source_key -> async/sync callable to retry
        max_retries: Max retry attempts per source
        on_progress: Optional callback(message: str) for progress

    Returns:
        Dict of source_key -> new data (or original error if still failed)
    """
    if not critical_gaps:
        return {}

    def _progress(msg: str):
        if on_progress:
            on_progress(msg)

    results = {}

    for key in critical_gaps:
        func = retry_funcs.get(key)
        if not func:
            logger.warning(f"Info-Gap: no retry function for {key}")
            continue

        for attempt in range(1, max_retries + 1):
            _progress(f"Retrying {key} (attempt {attempt}/{max_retries})...")
            logger.info(f"Info-Gap: retrying {key}, attempt {attempt}")
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = await asyncio.to_thread(func)

                status = _assess_source_status(key, result)
                if status == "SUFFICIENT":
                    results[key] = result
                    _progress(f"{key} recovered successfully")
                    logger.info(f"Info-Gap: {key} recovered on attempt {attempt}")
                    break
                elif attempt == max_retries:
                    results[key] = result
                    logger.warning(f"Info-Gap: {key} still {status} after {max_retries} attempts")
            except Exception as e:
                logger.warning(f"Info-Gap: {key} retry {attempt} failed: {e}")
                if attempt == max_retries:
                    results[key] = {"signal": "ERROR", "summary": f"Failed after {max_retries} retries: {e}"}

    return results


def _describe_impact(key: str, status: str, criticality: str) -> str:
    """Generate human-readable impact description."""
    if status == "SUFFICIENT":
        return "Data available and complete"
    verb = "Missing" if status == "MISSING" else "Incomplete"
    if criticality == "HIGH":
        return f"{verb} — could significantly affect recommendation accuracy"
    elif criticality == "MEDIUM":
        return f"{verb} — may reduce analysis confidence"
    return f"{verb} — minor impact on overall assessment"


def _build_summary(sufficient: int, total: int, critical_gaps: list[str]) -> str:
    """Build overall gap summary."""
    pct = round(sufficient / total * 100) if total > 0 else 0
    parts = [f"{sufficient}/{total} data sources available ({pct}% coverage)"]
    if critical_gaps:
        parts.append(f"Critical gaps: {', '.join(critical_gaps)}")
    else:
        parts.append("No critical data gaps detected")
    return ". ".join(parts)
