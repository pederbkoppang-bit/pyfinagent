# ingestion-agent/response.py
"""
Pure HTTP-status decision helper for the ingestion Cloud Function
(phase-75.16, leg c).

Deliberately has ZERO third-party imports (no functions_framework,
google-cloud, pandas) so it can be unit-tested (backend/tests/
test_phase_75_deploy_surface.py) without installing any Cloud
Function-only dependency. `main.py` imports this module for its
actual HTTP responses; keep it pure if you extend it.

Cloud Scheduler treats ANY 2xx response as success and does not
inspect the body (google.cloud.scheduler.v1 RPC reference) -- so a
"Failure" string wrapped in a 200 is invisible to the retry policy.
The 4 outcomes below map genuine failures to 500 and genuine
"nothing to do" to 200.
"""

from typing import Optional, Tuple


def decide_response(fetch_ok: bool, rows_fetched: int, load_ok: Optional[bool]) -> Tuple[str, int]:
    """Decide the (body, status) HTTP response for one ingestion run.

    - fetch_ok=False -> the Extract step raised. Always 500, regardless of
      rows_fetched/load_ok (both are meaningless in this case).
    - fetch_ok=True, rows_fetched==0 -> genuine no-data (e.g. weekend/holiday
      range). 200 -- this is a successful run that had nothing to load,
      not a failure. load_ok is meaningless here (the Load step never ran).
    - fetch_ok=True, rows_fetched>0, load_ok=True -> 200 (success).
    - fetch_ok=True, rows_fetched>0, load_ok=False -> the Load step raised.
      500.
    """
    if not fetch_ok:
        return ("Ingestion failed. Status: Failure (fetch exception). Rows: 0.", 500)
    if rows_fetched == 0:
        return ("Ingestion completed. Status: Success (No Data). Rows: 0.", 200)
    if load_ok:
        return (f"Ingestion completed. Status: Success. Rows: {rows_fetched}.", 200)
    return (f"Ingestion failed. Status: Failure (BQ load exception). Rows: {rows_fetched}.", 500)
