"""verify_phase_25_C7 -- Unified /api/observability/data-freshness endpoint + page.

Verifies:
  1. Backend route `GET /api/observability/data-freshness` is registered in
     `backend/api/observability_api.py`.
  2. Handler body delegates to `compute_freshness` from `backend.services.cycle_health`.
  3. Frontend page `frontend/src/app/observability/page.tsx` exists and
     renders a per-table table with band coloring.
  4. Frontend Sidebar has a nav link to `/observability`.

Behavioral round-trip:
  5. Import `observability_api` and assert the FastAPI router exposes the
     `/data-freshness` path. Mock `compute_freshness` and call the route
     to confirm it returns the mocked payload (no real BQ contact).

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import asyncio
import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO = Path(__file__).resolve().parents[1]

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: backend route registered ──────────────────────────────────
obs_api = (REPO / "backend/api/observability_api.py").read_text(encoding="utf-8")
route_present = bool(
    re.search(r'@router\.get\(\s*["\']/data-freshness["\']\s*\)', obs_api)
)
claim(
    "1. new_route_data_freshness_in_observability_api",
    route_present,
    "Found @router.get('/data-freshness') decorator" if route_present else "Missing /data-freshness route",
)


# ── Claim 2: handler delegates to compute_freshness ────────────────────
tree = ast.parse(obs_api)
delegates = False
for node in ast.walk(tree):
    if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_observability_data_freshness":
        body_src = ast.unparse(node)
        if "compute_freshness" in body_src and "to_thread" in body_src:
            delegates = True
        break
claim(
    "2. handler_delegates_to_compute_freshness",
    delegates,
    "Handler routes through asyncio.to_thread(_cf, ...)" if delegates else "Handler does not delegate to compute_freshness",
)


# ── Claim 3: frontend page exists with table + bands ───────────────────
page_path = REPO / "frontend/src/app/observability/page.tsx"
page_exists = page_path.exists()
page_src = page_path.read_text(encoding="utf-8") if page_exists else ""
has_table = "<table" in page_src
has_band = "BandPill" in page_src or "band" in page_src
has_freshness_fetch = "getObservabilityDataFreshness" in page_src
page_ok = page_exists and has_table and has_band and has_freshness_fetch
claim(
    "3. frontend_observability_page_renders_per_table_table_with_bands",
    page_ok,
    f"exists={page_exists} table={has_table} band={has_band} fetch={has_freshness_fetch}",
)


# ── Claim 4: Sidebar nav link to /observability ────────────────────────
sidebar_src = (REPO / "frontend/src/components/Sidebar.tsx").read_text(encoding="utf-8")
sidebar_has_link = bool(
    re.search(r'href:\s*["\']/observability["\']', sidebar_src)
)
claim(
    "4. sidebar_links_to_observability_page",
    sidebar_has_link,
    "Sidebar contains /observability nav entry" if sidebar_has_link else "Missing /observability nav link",
)


# ── Claim 5: behavioral round-trip ─────────────────────────────────────
async def _round_trip() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO))
    try:
        # Avoid touching BQ / settings: patch the imports inside the handler body
        with patch("backend.services.cycle_health.compute_freshness") as mock_cf, \
             patch("backend.db.bigquery_client.BigQueryClient") as mock_bq, \
             patch("backend.config.settings.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(paper_cycle_interval_sec=24 * 3600.0)
            mock_bq.return_value = MagicMock()
            mock_cf.return_value = {
                "sources": {
                    "paper_trades": {
                        "last_tick_age_sec": 120.0,
                        "interval_sec": 86400.0,
                        "ratio": 0.0014,
                        "band": "green",
                    }
                },
                "overall_band": "green",
                "heartbeat": {},
                "bq_ingest_lag_sec": 120.0,
                "thresholds": {},
                "computed_at": "2026-05-13T00:00:00Z",
            }
            from backend.api.observability_api import get_observability_data_freshness  # noqa: WPS433
            out = await get_observability_data_freshness()
        if not isinstance(out, dict):
            return False, f"Expected dict, got {type(out).__name__}"
        if "sources" not in out:
            return False, f"Missing 'sources' key; keys={list(out)}"
        if "overall_band" not in out:
            return False, f"Missing 'overall_band' key; keys={list(out)}"
        if out["overall_band"] != "green":
            return False, f"Expected overall_band='green', got {out['overall_band']!r}"
        if not mock_cf.called:
            return False, "compute_freshness was not invoked"
        return True, "Handler returned mocked payload via compute_freshness"
    except Exception as e:  # pragma: no cover -- debug aid
        return False, f"Exception: {type(e).__name__}: {e}"


round_trip_ok, round_trip_detail = asyncio.run(_round_trip())
claim("5. behavioral_round_trip_returns_compute_freshness_payload", round_trip_ok, round_trip_detail)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.C7 verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
