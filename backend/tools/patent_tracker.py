"""
Patent tracking tool -- Google Patents Public Datasets via BigQuery.

Replaces the PatentsView REST API, which USPTO shut down (HTTP 410) on
2025-05-01. The BigQuery public dataset `patents-public-data.patents.publications`
contains the same USPTO corpus plus worldwide filings, free to query within the
existing `sunny-might-477607-p8` project's BQ budget (1 TB/month free tier).

Contract matches backend/tools/*.py:
    {"ticker": "...", "signal": "BULLISH|...|ERROR|NO_DATA|NEUTRAL",
     "summary": "...", "data"/fields...}
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_QUERY = """
SELECT
  publication_number AS patent_id,
  grant_date AS grant_date_int,
  CAST(grant_date / 10000 AS INT64) AS patent_year,
  (SELECT t.text FROM UNNEST(title_localized) t WHERE t.language = 'en' LIMIT 1) AS patent_title,
  ARRAY_LENGTH(citation) AS cited_patent_count
FROM `patents-public-data.patents.publications`
WHERE country_code = 'US'
  AND grant_date > 0
  AND grant_date >= @start_date_int
  AND EXISTS (
    SELECT 1 FROM UNNEST(assignee_harmonized) a
    WHERE UPPER(a.name) LIKE UPPER(@assignee_pattern)
  )
ORDER BY grant_date DESC
LIMIT 500
"""


def _clean_company_name(company_name: str) -> str:
    name = company_name.split(",")[0]
    for suffix in (" Inc", " Corp", " Ltd", " LLC", " Co.", " Holdings",
                   " Technologies", " Technology"):
        name = name.split(suffix)[0]
    return name.strip()


def _run_sync_query(assignee: str, start_year: int) -> list[dict]:
    from google.cloud import bigquery

    client = bigquery.Client()  # picks up GCP_PROJECT_ID from env
    job = client.query(
        _QUERY,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date_int", "INT64",
                                              start_year * 10000 + 101),
                bigquery.ScalarQueryParameter("assignee_pattern", "STRING",
                                              f"%{assignee}%"),
            ]
        ),
    )
    return [dict(row) for row in job.result(timeout=20)]


async def get_patent_data(
    company_name: str, ticker: str, years: int = 3, api_key: str = "",
) -> dict:
    """Query Google Patents BQ public dataset for US grant trends.

    api_key param retained for backward compatibility (unused; BQ auth is
    via ADC/GCP_PROJECT_ID).
    """
    clean_name = _clean_company_name(company_name)
    start_year = datetime.now(tz=timezone.utc).year - years

    try:
        rows = await asyncio.to_thread(_run_sync_query, clean_name, start_year)
    except Exception as e:
        logger.error("BQ patent query failed for %s: %s", company_name, e)
        return {
            "ticker": ticker,
            "company": company_name,
            "total_patents": 0,
            "signal": "ERROR",
            "summary": f"Patent data unavailable: {type(e).__name__}: {e}",
        }

    if not rows:
        return {
            "ticker": ticker,
            "company": company_name,
            "total_patents": 0,
            "signal": "NO_DATA",
            "summary": f"No US grants found for {company_name} in the last {years} years.",
        }

    by_year: dict[int, int] = {}
    total_citations = 0
    recent_patents: list[dict] = []

    for p in rows:
        year = p.get("patent_year")
        if year is None:
            continue
        by_year[int(year)] = by_year.get(int(year), 0) + 1
        citations = int(p.get("cited_patent_count") or 0)
        total_citations += citations
        if len(recent_patents) < 5:
            grant_int = p.get("grant_date_int") or 0
            date_str = (
                f"{grant_int // 10000:04d}-{(grant_int // 100) % 100:02d}-{grant_int % 100:02d}"
                if grant_int else ""
            )
            recent_patents.append({
                "number": p.get("patent_id", ""),
                "title": (p.get("patent_title") or "")[:100],
                "date": date_str,
                "citations": citations,
            })

    total = sum(by_year.values())
    avg_citations = total_citations / total if total else 0

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
        "total_patents": total,
        "patents_by_year": by_year,
        "velocity_pct": round(velocity_pct, 1),
        "avg_citations": round(avg_citations, 1),
        "recent_patents": recent_patents,
        "signal": signal,
        "summary": (
            f"{total} US grants in {years}yr. "
            f"YoY velocity: {velocity_pct:+.1f}%. "
            f"Avg citations: {avg_citations:.1f}. "
            f"Signal: {signal}."
        ),
    }
