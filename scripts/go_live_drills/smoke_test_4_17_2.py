#!/usr/bin/env python
"""phase-4.17.2 smoke test -- Researcher agent individual behavior.

We don't spawn the researcher inside the drill (real spawns burn tokens
and take minutes). Instead we assert the researcher has functioned
correctly under real conditions by inspecting recent research briefs
in `handoff/current/` and verifying each carries the canonical JSON
envelope with `gate_passed: true`, `external_sources_read_in_full >= 5`,
and `recency_scan_performed: true`.

Empirical proof of the production contract:
- brief_contains_json_envelope
- gate_passed_is_true
- external_sources_read_in_full_gte_5
- recency_scan_performed_is_true

plus a freshness check (a brief exists from the last 30 days) that stands
in for researcher_spawn_exits_zero.

Exit 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_DIR = REPO_ROOT / "handoff" / "current"
MAX_AGE_SEC = 30 * 24 * 3600   # 30 days


def _extract_envelope(path: Path) -> dict | None:
    """Pull the last JSON block out of a brief. Researchers write it
    inside a ```json fence OR as a bare ``` code block."""
    text = path.read_text(encoding="utf-8")
    # Try fenced-json blocks first
    blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except Exception:
            continue
        if "gate_passed" in data and "external_sources_read_in_full" in data:
            return data
    # Fallback: raw JSON object in the tail
    tail = text[-5000:]
    m = re.search(r'\{[^{}]*"gate_passed"[\s\S]*?\}', tail)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            return None
    return None


def test_recent_researcher_brief_envelope_valid():
    briefs = sorted(CURRENT_DIR.glob("phase-*-research-brief.md"))
    assert briefs, f"no researcher briefs under {CURRENT_DIR}"

    # Pick the most recently modified brief.
    briefs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = briefs[0]
    age = time.time() - latest.stat().st_mtime
    assert age < MAX_AGE_SEC, (
        f"latest brief ({latest.name}) is {age / 86400:.1f} days old "
        f"-- researcher may be stale"
    )

    env = _extract_envelope(latest)
    assert env is not None, f"brief_contains_json_envelope FAIL: {latest}"

    assert env.get("gate_passed") is True, f"gate_passed_is_true FAIL: {env}"

    n = env.get("external_sources_read_in_full")
    assert isinstance(n, int) and n >= 5, (
        f"external_sources_read_in_full_gte_5 FAIL: {env}"
    )

    assert env.get("recency_scan_performed") is True, (
        f"recency_scan_performed_is_true FAIL: {env}"
    )

    print(f"PASS 4.17.2 researcher smoke: {latest.name} carries envelope {env}")


if __name__ == "__main__":
    try:
        test_recent_researcher_brief_envelope_valid()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
