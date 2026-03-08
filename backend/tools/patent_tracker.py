"""
Patent tracking tool — Google Patents Public Data on BigQuery.
The PatentsView API is unavailable (API key grants suspended).
Uses the free `patents-public-data` BigQuery public dataset instead,
which the project already has access to via GCP credentials.
"""

import asyncio
import logging
from datetime import datetime

from google.cloud import bigquery

logger = logging.getLogger(__name__)

# Count patents per year and total citations per year for the assignee
_COUNT_QUERY = """
SELECT
  CAST(FLOOR(publication_date / 10000) AS INT64) AS pub_year,
  COUNT(*) AS patent_count,
  SUM(ARRAY_LENGTH(citation)) AS total_citations
FROM
  `patents-public-data.patents.publications`
WHERE
  EXISTS (
    SELECT 1 FROM UNNEST(assignee_harmonized) a
    WHERE LOWER(a.name) LIKE LOWER(@pattern)
  )
  AND country_code = 'US'
  AND grant_date > 0
  AND publication_date >= @start_date
GROUP BY pub_year
ORDER BY pub_year
"""

# Fetch a few recent patents for display
_RECENT_QUERY = """
SELECT
  publication_number,
  publication_date,
  title.text AS title,
  ARRAY_LENGTH(citation) AS citation_count
FROM
  `patents-public-data.patents.publications`,
  UNNEST(title_localized) AS title
WHERE
  EXISTS (
    SELECT 1 FROM UNNEST(assignee_harmonized) a
    WHERE LOWER(a.name) LIKE LOWER(@pattern)
  )
  AND country_code = 'US'
  AND grant_date > 0
  AND publication_date >= @start_date
  AND title.language = 'en'
ORDER BY publication_date DESC
LIMIT 5
"""


async def get_patent_data(
    company_name: str, ticker: str, years: int = 3, api_key: str = "",
) -> dict:
    """
    Query Google Patents Public Data on BigQuery for patent filing trends.
    Uses the `patents-public-data.patents.publications` public dataset.
    The api_key parameter is kept for backward compatibility but unused.
    """
    start_date = (datetime.utcnow().year - years) * 10000 + 101  # e.g. 20230101

    # Build a LIKE pattern from the company name (first meaningful word)
    # e.g. "NVIDIA CORP" -> "%nvidia%"
    clean_name = company_name.split(",")[0].split(" Inc")[0].split(" Corp")[0].strip()
    pattern = f"%{clean_name}%"

    try:
        result = await asyncio.to_thread(_run_bq_query, pattern, start_date)
    except Exception as e:
        logger.error("BigQuery patent query failed for %s: %s", company_name, e)
        return {
            "ticker": ticker,
            "company": company_name,
            "total_patents": 0,
            "signal": "ERROR",
            "summary": f"Error querying patent data: {e}",
        }

    by_year = result.get("by_year", {})
    total = sum(by_year.values())
    total_citations = result.get("total_citations", 0)
    recent_patents = result.get("recent", [])

    if not by_year:
        return {
            "ticker": ticker,
            "company": company_name,
            "total_patents": 0,
            "signal": "NO_DATA",
            "summary": f"No US patents found for {company_name} in the last {years} years.",
        }

    # Compute year-over-year velocity
    sorted_years = sorted(by_year.keys())
    velocity_pct = 0.0
    if len(sorted_years) >= 2:
        prev = by_year[sorted_years[-2]]
        curr = by_year[sorted_years[-1]]
        if prev > 0:
            velocity_pct = ((curr - prev) / prev) * 100

    avg_citations = total_citations / total if total else 0

    # Signal: ≥20% growth = innovation velocity breakout
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
            f"{total} US patents in {years}yr. "
            f"YoY velocity: {velocity_pct:+.1f}%. "
            f"Avg citations: {avg_citations:.1f}. "
            f"Signal: {signal}."
        ),
    }


def _run_bq_query(pattern: str, start_date: int) -> dict:
    """Execute the BigQuery patent queries (runs in a thread)."""
    client = bigquery.Client()
    params = [
        bigquery.ScalarQueryParameter("pattern", "STRING", pattern),
        bigquery.ScalarQueryParameter("start_date", "INT64", start_date),
    ]

    # 1. Get counts per year
    count_config = bigquery.QueryJobConfig(query_parameters=params)
    count_job = client.query(_COUNT_QUERY, job_config=count_config)
    by_year: dict[int, int] = {}
    total_citations = 0
    for row in count_job.result():
        by_year[int(row.pub_year)] = row.patent_count
        total_citations += row.total_citations or 0

    # 2. Get recent patents for display
    recent_config = bigquery.QueryJobConfig(query_parameters=params)
    recent_job = client.query(_RECENT_QUERY, job_config=recent_config)
    recent = []
    for row in recent_job.result():
        pub_int = row.publication_date
        date_str = f"{pub_int // 10000}-{(pub_int % 10000) // 100:02d}-{pub_int % 100:02d}" if pub_int else ""
        recent.append({
            "number": row.publication_number,
            "title": (row.title or "")[:100],
            "date": date_str,
            "citations": row.citation_count or 0,
        })

    return {"by_year": by_year, "total_citations": total_citations, "recent": recent}
