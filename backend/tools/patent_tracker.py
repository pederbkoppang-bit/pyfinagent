"""
Patent tracking tool — PatentsView REST API (USPTO public dataset).
No API key or credentials required. Free government data.
https://api.patentsview.org/patents/query
"""

import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

_PATENTSVIEW_URL = "https://api.patentsview.org/patents/query"
_TIMEOUT = 20.0

def _clean_company_name(company_name: str) -> str:
    """Extract the core company name for searching, stripping legal suffixes."""
    name = company_name.split(",")[0]
    for suffix in [" Inc", " Corp", " Ltd", " LLC", " Co.", " Holdings", " Technologies", " Technology"]:
        name = name.split(suffix)[0]
    return name.strip()


async def get_patent_data(
    company_name: str, ticker: str, years: int = 3, api_key: str = "",
) -> dict:
    """
    Query PatentsView (USPTO public dataset) for US patent filing trends.
    No API key or GCP credentials required — free public government API.
    api_key parameter kept for backward compatibility but unused.
    """
    clean_name = _clean_company_name(company_name)
    start_year = datetime.now(tz=timezone.utc).year - years
    start_date = f"{start_year}-01-01"

    query_payload = {
        "q": {"_and": [
            {"_contains": {"assignee_organization": clean_name}},
            {"_gte": {"patent_date": start_date}},
        ]},
        "f": ["patent_id", "patent_date", "patent_year", "patent_title",
              "cited_patent_count", "assignee_organization"],
        "o": {"per_page": 500, "page": 1},
        "s": [{"patent_date": "desc"}],
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                _PATENTSVIEW_URL,
                json=query_payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                raise ValueError(f"PatentsView HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
    except Exception as e:
        logger.error("PatentsView query failed for %s: %s", company_name, e)
        return {
            "ticker": ticker,
            "company": company_name,
            "total_patents": 0,
            "signal": "ERROR",
            "summary": f"Patent data unavailable: {e}",
        }

    patents = data.get("patents") or []
    total_count = int(data.get("total_patent_count") or len(patents))

    if not patents:
        return {
            "ticker": ticker,
            "company": company_name,
            "total_patents": 0,
            "signal": "NO_DATA",
            "summary": f"No US patents found for {company_name} in the last {years} years.",
        }

    # Aggregate by year
    by_year: dict[int, int] = {}
    total_citations = 0
    recent_patents = []

    for p in patents:
        year_str = (p.get("patent_year") or p.get("patent_date", "")[:4])
        try:
            year = int(year_str)
        except (ValueError, TypeError):
            continue
        by_year[year] = by_year.get(year, 0) + 1
        citations = int(p.get("cited_patent_count") or 0)
        total_citations += citations
        if len(recent_patents) < 5:
            recent_patents.append({
                "number": p.get("patent_id", ""),
                "title": (p.get("patent_title") or "")[:100],
                "date": p.get("patent_date", ""),
                "citations": citations,
            })

    total = sum(by_year.values())
    avg_citations = total_citations / total if total else 0

    # Year-over-year velocity using the two most recent full years
    sorted_years = sorted(by_year.keys())
    velocity_pct = 0.0
    if len(sorted_years) >= 2:
        prev = by_year[sorted_years[-2]]
        curr = by_year[sorted_years[-1]]
        if prev > 0:
            velocity_pct = ((curr - prev) / prev) * 100

    signal = "NEUTRAL"
    if velocity_pct >= 20:
        signal = "INNOVATION_BREAKOUT"
    elif velocity_pct >= 10:
        signal = "GROWING"
    elif velocity_pct <= -10:
        signal = "DECLINING"

    return {
        "ticker": ticker,
        "company": company_name,
        "total_patents": total_count,
        "patents_by_year": by_year,
        "velocity_pct": round(velocity_pct, 1),
        "avg_citations": round(avg_citations, 1),
        "recent_patents": recent_patents,
        "signal": signal,
        "summary": (
            f"{total_count} US patents in {years}yr. "
            f"YoY velocity: {velocity_pct:+.1f}%. "
            f"Avg citations: {avg_citations:.1f}. "
            f"Signal: {signal}."
        ),
    }
