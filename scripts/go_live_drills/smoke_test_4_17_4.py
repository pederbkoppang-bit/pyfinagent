#!/usr/bin/env python
"""phase-4.17.4 smoke test -- Inter-agent handoff integrity.

Canonical five-file protocol per Anthropic harness design doc:

  1. handoff/current/contract.md                (PLAN)
  2. handoff/current/experiment_results.md      (GENERATE)
  3. handoff/current/evaluator_critique.md      (EVALUATE)
  4. handoff/harness_log.md                     (LOG append)
  5. .claude/masterplan.json                    (STATUS flip)

Asserts every artifact exists and is non-empty, plus the masterplan
JSON is valid. Empirical-artifact strategy -- the file-based handoff
is the canonical integration surface; if all 5 exist and carry
current content, the handoff is functional.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED = [
    "handoff/current/contract.md",
    "handoff/current/experiment_results.md",
    "handoff/current/evaluator_critique.md",
    "handoff/harness_log.md",
    ".claude/masterplan.json",
]


def test_five_file_protocol_end_to_end():
    missing = [f for f in REQUIRED if not (REPO_ROOT / f).exists()]
    empty = [
        f for f in REQUIRED
        if (REPO_ROOT / f).exists()
        and (REPO_ROOT / f).stat().st_size == 0
    ]
    assert not missing, f"missing artifacts: {missing}"
    assert not empty, f"empty artifacts: {empty}"

    # masterplan JSON valid
    try:
        with (REPO_ROOT / ".claude/masterplan.json").open() as f:
            mp = json.load(f)
    except Exception as e:
        raise AssertionError(f"masterplan_json_updated FAIL: invalid JSON: {e}")
    assert "phases" in mp and isinstance(mp["phases"], list)

    # harness_log contains at least one recent cycle header
    log = (REPO_ROOT / "handoff/harness_log.md").read_text(encoding="utf-8")
    assert "## " in log and ("Cycle " in log or "phase-" in log), (
        "harness_log_appended FAIL: no cycle/phase markers in log"
    )

    print("PASS 4.17.4: five-file handoff protocol intact")


if __name__ == "__main__":
    try:
        test_five_file_protocol_end_to_end()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
