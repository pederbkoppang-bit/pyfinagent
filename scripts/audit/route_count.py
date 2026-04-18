"""phase-4.7 step 4.7.1: Top-level route-count audit.

Enumerates every user-facing top-level route under `frontend/src/app/`.
Excludes:
- api/* (API route handlers, not user pages)
- login (required auth surface; not counted against the <=8 budget
  per contract rationale)
- nested pages (only folders directly under `app/` are top-level)

Writes `handoff/route_count.json` with:
- top_level_routes (integer; the number asserted by masterplan gate)
- routes (array of path strings)
- removed_in_this_step (documents 4.7.1 consolidations with merge
  target + rationale; satisfies the 'zero_open_pages_removed_or_
  justified' criterion)

Exit 0 always (it's an audit).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP_DIR = REPO / "frontend" / "src" / "app"
OUT = REPO / "handoff" / "route_count.json"


# Routes physically removed in step 4.7.1 with their merge targets.
# Each entry records the rationale so a later reviewer can audit the
# consolidation decision without reading the step contract.
REMOVED_IN_STEP_4_7_1 = [
    {
        "path": "/compare",
        "merge_target": "/reports",
        "reason": "Functionality already exists as a tab inside the "
                  "/reports page (reports/page.tsx lines 328-620). "
                  "Standalone route was pure duplication.",
        "redirect_status": 308,
    },
    {
        "path": "/analyze",
        "merge_target": "/signals",
        "reason": "Both are single-ticker entry flows. /signals "
                  "remains the canonical enrichment surface; /analyze's "
                  "deep-dive invocation can still be reached via its "
                  "backend /api/analysis/* endpoints and consolidated "
                  "into /signals in a follow-up UI merge.",
        "redirect_status": 308,
    },
    {
        "path": "/portfolio",
        "merge_target": "/paper-trading",
        "reason": "Portfolio was a manual position tracker (2 opens/30d) "
                  "while /paper-trading (12 opens/30d) is the primary "
                  "portfolio surface with live simulated positions.",
        "redirect_status": 308,
    },
]


# /login is a user-facing page, but it is NOT counted against the
# <= 8 top-level budget. Justification recorded here for auditors.
EXCLUDED_FROM_BUDGET = {
    "/login": (
        "auth surface required by NextAuth v5 middleware; opens_30d=1 "
        "in the 4.7.0 window is the floor for access events (at least "
        "one login must occur to access the app)."
    ),
}


def _enumerate_top_level() -> list[str]:
    """Return top-level user-facing route paths (strings)."""
    routes: list[str] = []
    if (APP_DIR / "page.tsx").exists():
        routes.append("/")
    for entry in sorted(APP_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("api") or entry.name.startswith("_"):
            continue
        # must contain a page.tsx directly (true top-level route)
        if not (entry / "page.tsx").exists():
            continue
        routes.append("/" + entry.name)
    return routes


def main() -> int:
    all_routes = _enumerate_top_level()
    budget_routes = [r for r in all_routes if r not in EXCLUDED_FROM_BUDGET]
    top_level_count = len(budget_routes)

    result = {
        "step": "4.7.1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "top_level_routes": top_level_count,
        "top_level_route_paths": budget_routes,
        "excluded_from_budget": EXCLUDED_FROM_BUDGET,
        "removed_in_this_step": REMOVED_IN_STEP_4_7_1,
        "budget_ceiling": 8,
        "notes": [
            "top_level_routes counts user-facing top-level page.tsx "
            "files under frontend/src/app/, excluding api/ and /login. "
            "Nested routes (sub-paths like /reports/compare as a tab) "
            "are NOT top-level and are not counted.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "top_level_routes": top_level_count,
        "paths": budget_routes,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
