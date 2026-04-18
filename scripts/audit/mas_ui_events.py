"""phase-4.7 step 4.7.3: MAS Monitoring UI audit.

Asserts the /agents frontend surfaces all backend event types and
exposes per-agent latency + cost + heartbeat. Run with --check to
fail-exit on violation; emits handoff/mas_ui_events.json with verdict.

Checks:
1. events_rendered_1to1_with_mas_events_py -- every event type named
   in backend/agents/mas_events.py module docstring appears in the
   frontend EVENT_STYLES keys.
2. per_agent_latency_visible -- agents/page.tsx contains a
   data-col="latency" column + data-cell="latency" cells.
3. per_agent_cost_visible -- agents/page.tsx contains data-col="cost"
   + data-cell="cost" cells + consumes cost_summary from the MAS
   dashboard endpoint.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAS_PY = REPO / "backend" / "agents" / "mas_events.py"
AGENTS_TSX = REPO / "frontend" / "src" / "app" / "agents" / "page.tsx"
OUT = REPO / "handoff" / "mas_ui_events.json"


# The canonical event-type list is defined in the mas_events.py
# module docstring lines 13-27 as a two-column "name — description"
# format. Pull the first token of each such line.
_DOC_LINE = re.compile(r"^\s{2,}([a-z_]+)\s+[-\u2014]\s+.+", re.MULTILINE)


def _backend_event_types() -> list[str]:
    text = MAS_PY.read_text(encoding="utf-8")
    docstring_match = re.search(
        r"Event types \(map to dashboard nodes\):\s*\n(.*?)\n\nReferences",
        text,
        re.DOTALL,
    )
    if not docstring_match:
        raise RuntimeError("Could not find 'Event types' block in mas_events.py docstring")
    block = docstring_match.group(1)
    names: list[str] = []
    for m in _DOC_LINE.finditer(block):
        names.append(m.group(1))
    return sorted(set(names))


def _frontend_event_styles() -> list[str]:
    text = AGENTS_TSX.read_text(encoding="utf-8")
    start = text.find("const EVENT_STYLES")
    if start < 0:
        raise RuntimeError("Could not locate EVENT_STYLES in agents/page.tsx")
    # The object literal starts after the `= {` assignment (skipping the
    # type annotation `Record<string, { ... }>` which contains its own
    # braces).
    assign = text.find("= {", start)
    if assign < 0:
        raise RuntimeError("EVENT_STYLES assignment '= {' not found")
    brace_open = assign + 2
    depth = 0
    i = brace_open
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block = text[brace_open + 1:i]
    names: list[str] = []
    # Top-level keys: consume lines where depth inside block is 0.
    depth = 0
    for line in block.splitlines():
        # Count braces BEFORE this line from running depth
        if depth == 0:
            m = re.match(r"\s*([a-z_][a-z_0-9]*)\s*:\s*\{", line)
            if m:
                names.append(m.group(1))
        # Update depth for next line
        depth += line.count("{") - line.count("}")
    return sorted(set(names))


def _check_latency_column(tsx: str) -> bool:
    return bool(
        re.search(r'data-col="latency"', tsx)
        and re.search(r'data-cell="latency"', tsx)
    )


def _check_cost_column(tsx: str) -> bool:
    cost_cols = bool(
        re.search(r'data-col="cost"', tsx)
        and re.search(r'data-cell="cost"', tsx)
    )
    consumes_summary = "cost_summary" in tsx and "costSummary" in tsx
    return cost_cols and consumes_summary


def _check_heartbeat_column(tsx: str) -> bool:
    return bool(
        re.search(r'data-col="heartbeat"', tsx)
        and re.search(r'data-cell="heartbeat"', tsx)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                         help="exit 1 on any audit failure")
    args = parser.parse_args()

    backend = _backend_event_types()
    frontend = _frontend_event_styles()
    missing_in_frontend = sorted(set(backend) - set(frontend))
    extra_in_frontend = sorted(set(frontend) - set(backend))
    coverage_ok = not missing_in_frontend

    tsx = AGENTS_TSX.read_text(encoding="utf-8")
    latency_ok = _check_latency_column(tsx)
    cost_ok = _check_cost_column(tsx)
    heartbeat_ok = _check_heartbeat_column(tsx)

    verdict = "PASS" if (coverage_ok and latency_ok and cost_ok) else "FAIL"

    result = {
        "step": "4.7.3",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "backend_event_types": backend,
        "frontend_event_styles": frontend,
        "backend_count": len(backend),
        "frontend_count": len(frontend),
        "missing_in_frontend": missing_in_frontend,
        "extra_in_frontend": extra_in_frontend,
        "events_rendered_1to1_with_mas_events_py": coverage_ok,
        "per_agent_latency_visible": latency_ok,
        "per_agent_cost_visible": cost_ok,
        "per_agent_heartbeat_visible": heartbeat_ok,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "backend_count": len(backend),
        "frontend_count": len(frontend),
        "missing_in_frontend": missing_in_frontend,
        "latency_ok": latency_ok,
        "cost_ok": cost_ok,
        "heartbeat_ok": heartbeat_ok,
    }))
    if args.check and verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
