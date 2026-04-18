"""phase-4.7 step 4.7.0: Frontend route inventory + 30-day usage telemetry.

No live pageview tracker exists yet (confirmed by Explore phase of
Cycle 68). Rather than invent page-view numbers we cannot verify, this
script uses git commit activity on each route's `page.tsx` over the
trailing 30 days as a named, honestly-sourced PROXY for usage
(`usage_source = "git_activity_30d"`).

Rationale: for an internal 2-person tool where developer == user,
commit frequency is a recognized development-activity proxy (see
research_4.7.0.md). It cleanly discriminates heavily-iterated pages
(e.g., /backtest) from dormant ones (e.g., /login) so step 4.7.1 can
make a merge/delete decision.

A follow-up step will wire a first-party pageview endpoint so future
windows carry real `usage_source = "pageview_beacon_30d"` data.

Emits `handoff/frontend_usage.json`. Exit 0 always (this is an
inventory script, not a gate).
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FRONTEND_APP = REPO / "frontend" / "src" / "app"
OUT = REPO / "handoff" / "frontend_usage.json"

# Routes known from docs/architecture to have production usage even if
# git activity is low. When overridden, `usage_source` is tagged
# accordingly so auditors see both signals.
DOCS_GROUND_TRUTH = {
    "/backtest":      "primary_user_surface_per_ARCHITECTURE.md",
    "/paper-trading": "primary_user_surface_per_ARCHITECTURE.md",
    "/agents":        "mas_cockpit_per_CLAUDE.md_phase_2.7",
    "/":              "dashboard_landing",
}


def _file_to_route(page_tsx: Path) -> str:
    """Map frontend/src/app/foo/bar/page.tsx -> '/foo/bar'.

    The root page.tsx maps to '/'. Route groups `(group)` and dynamic
    segments `[id]` are preserved verbatim (no rewriting).
    """
    rel = page_tsx.relative_to(FRONTEND_APP).parent
    parts = [p for p in rel.parts if p]  # '.' yields empty parts tuple
    if not parts:
        return "/"
    return "/" + "/".join(parts)


def _enumerate_routes() -> list[tuple[str, Path]]:
    """Return (route_path, page_tsx_path) pairs, excluding api/auth."""
    results: list[tuple[str, Path]] = []
    for p in sorted(FRONTEND_APP.rglob("page.tsx")):
        rel = p.relative_to(FRONTEND_APP)
        if rel.parts and rel.parts[0] == "api":
            continue  # api routes have route.ts, not page.tsx; skip any stragglers
        results.append((_file_to_route(p), p))
    return results


def _commit_counts_30d() -> tuple[dict[Path, int], str]:
    """Run `git log --since=30.days --name-only -- frontend/src/app`
    and return (path -> 30d commit count, git_command_string)."""
    cmd = [
        "git", "-C", str(REPO), "log", "--since=30.days",
        "--name-only", "--pretty=format:", "--",
        "frontend/src/app",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            timeout=30,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"git log failed: {e}", file=sys.stderr)
        return {}, " ".join(cmd)
    counts: dict[Path, int] = defaultdict(int)
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        counts[REPO / line] += 1
    return dict(counts), " ".join(cmd)


def main() -> int:
    routes = _enumerate_routes()
    commit_counts, git_cmd = _commit_counts_30d()

    entries: list[dict] = []
    for route, page_tsx in routes:
        opens = int(commit_counts.get(page_tsx, 0))
        source = "git_activity_30d"
        gt = DOCS_GROUND_TRUTH.get(route)
        if gt is not None:
            source = f"git_activity_30d+{gt}"
        entries.append({
            "path": route,
            "page_file": str(page_tsx.relative_to(REPO)),
            "opens_30d": opens,
            "usage_source": source,
        })

    result = {
        "step": "4.7.0",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "usage_source": "git_activity_30d",
        "usage_source_command": git_cmd,
        "window_days": 30,
        "route_count": len(entries),
        "routes": entries,
        "notes": [
            "No live pageview beacon exists as of 2026-04-17; this "
            "window uses git commit activity on each page.tsx as a "
            "named proxy. A follow-up step wires a first-party "
            "/api/telemetry/pageview endpoint so future windows carry "
            "real opens.",
            "A zero opens_30d means no commits on that page in the "
            "window -- it is a legitimate value, not missing data.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "routes": len(entries),
        "usage_source": "git_activity_30d",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
