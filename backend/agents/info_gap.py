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
import math
import re
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
    # phase-80.27: quant_model was ABSENT from this table, so the only
    # validity test the Layer-1 pipeline applies never assessed it at all --
    # a NaN-poisoned quant payload could not even be counted as a gap.
    "quant_model": "HIGH",
}

# Sectors where certain signals are more critical
_SECTOR_OVERRIDES = {
    "Technology": {"patent": "HIGH"},
    "Healthcare": {"patent": "HIGH"},
    "Financial Services": {"fred_macro": "HIGH", "patent": "LOW"},
    "Energy": {"fred_macro": "HIGH", "alt_data": "MEDIUM"},
    "Consumer Cyclical": {"alt_data": "MEDIUM", "social_sentiment": "HIGH"},
}


# phase-80.27: `nan` as a WORD, not as a substring. A naive
# `"nan" in summary.lower()` matches "fi-NAN-cial", "governance", "tenant" and
# every other ordinary word containing those three letters -- it would mark
# almost every real summary MISSING and cause a self-inflicted analysis
# outage. Boundaries are non-letters so "+nan%", "-nan", "nan," all match
# while "financial" does not.
_NAN_TOKEN_RE = re.compile(r"(?<![a-z])[-+]?(nan|inf(inity)?)(?![a-z])", re.IGNORECASE)


def _has_non_finite(obj: object, _depth: int = 0) -> bool:
    """True if any float anywhere in `obj` is NaN or +/-Inf.

    Recurses dicts and sequences. `bool` subclasses `int`, not `float`, so
    flags are not scanned. Depth-capped so a pathological payload cannot
    blow the stack inside the analysis pipeline.
    """
    if _depth > 12:
        return False
    if isinstance(obj, float):
        return not math.isfinite(obj)
    if isinstance(obj, dict):
        return any(_has_non_finite(v, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(_has_non_finite(v, _depth + 1) for v in obj)
    return False


def _assess_source_status(key: str, data: dict) -> str:
    """Classify a data source as SUFFICIENT, PARTIAL, MISSING, or SKIPPED.

    phase-80.27 -- this is the ONLY validity test the Layer-1 pipeline
    applies to an enrichment payload, and it used to inspect exactly four
    things: not-a-dict, signal == 'ERROR', signal == 'SKIPPED', and the
    substrings 'error'/'failed' in the summary. **It never looked at a
    single number.**

    So a NaN-poisoned payload -- every value non-finite, but carrying
    signal='NEUTRAL' and a summary reading '3M return: +nan% vs sector
    +nan%' -- fell through to SUFFICIENT. The pipeline then reported
    "Data quality: 100%" and "Data available and complete" for a
    HIGH-criticality source whose every number was garbage, and proceeded
    to full debate, risk assessment and synthesis on it. The retry loop and
    the debate-skip gate -- the designed safety nets for exactly this --
    never fired.

    Two numeric checks are added below. Both can only move a source
    TOWARD MISSING, i.e. toward more gating and fewer trades, so this
    tightening is fail-safe by construction and ships un-flagged.
    """
    if not data or not isinstance(data, dict):
        return "MISSING"
    signal = data.get("signal")
    # phase-80.27: NO_DATA was an unhandled verdict -- backend/tools/alt_data.py
    # already returns it, and with no case here it was classified SUFFICIENT.
    # That is the same defect as the NaN one: an explicit failure read as
    # healthy data.
    if signal in ("ERROR", "NO_DATA"):
        return "MISSING"
    if signal == "SKIPPED":
        return "SKIPPED"
    summary = data.get("summary", "") or ""
    if "error" in summary.lower() or "failed" in summary.lower():
        return "MISSING"
    # A rendered 'nan'/'inf' in the prose is a data outage that reached the
    # text layer -- and that text is what gets handed to the LLM agents.
    if _NAN_TOKEN_RE.search(summary):
        return "MISSING"
    # The payload itself. This is the check whose absence let a 31-non-finite
    # sector block be reported as "complete".
    if _has_non_finite(data):
        return "MISSING"
    if signal == "N/A" and not summary:
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
                # phase-27.6.7: detect lambda-wrapped-coroutine. The call
                # sites in orchestrator.py:1660+ use `lambda: self.X(t)`
                # which is itself sync, but X may be async — calling the
                # lambda returns a coroutine. Check both the function and
                # the result to cover all three shapes:
                #   (a) async def func (direct)
                #   (b) lambda wrapping async call (returns coroutine)
                #   (c) sync def func (returns the value directly)
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func() if callable(func) else func
                    if asyncio.iscoroutine(result):
                        result = await result
                    elif not asyncio.iscoroutinefunction(func) and result is None:
                        # Truly sync function returned None or already-computed value;
                        # honor original semantics by re-invoking via to_thread for
                        # blocking-call parity. Should not be reached in practice.
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
