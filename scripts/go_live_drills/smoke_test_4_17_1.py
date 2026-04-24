#!/usr/bin/env python
"""phase-4.17.1 smoke test -- Main/Orchestrator agent individual behavior.

Runs `scripts/harness/run_harness.py --dry-run --cycles 1` and asserts
that the dry-run completes, a new cycle entry lands in
`handoff/harness_log.md`, and the three rolling handoff artifacts
still exist in `handoff/current/`.

Exit 0 on PASS. 1 on FAIL.

Intended to be collected by pytest under `scripts/go_live_drills/` so
the phase-4.17.10 aggregate gate replays it alongside the other drills.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def test_harness_dry_run_exits_zero_and_appends_cycle():
    """All 5 immutable criteria for 4.17.1."""
    log_before = (REPO_ROOT / "handoff/harness_log.md").read_text(encoding="utf-8")

    code, out = _run(
        [
            str(REPO_ROOT / ".venv/bin/python"),
            "scripts/harness/run_harness.py",
            "--dry-run",
            "--cycles",
            "1",
            "--iterations-per-cycle",
            "1",
        ]
    )
    assert code == 0, f"harness_dry_run_exits_zero FAIL: exit={code}\n{out[-1000:]}"

    # Canonical handoff artifacts (rolling top-level files).
    for art in (
        "handoff/current/contract.md",
        "handoff/current/experiment_results.md",
        "handoff/current/evaluator_critique.md",
    ):
        assert (REPO_ROOT / art).exists(), f"missing artifact: {art}"

    # harness_log gained at least one new cycle header.
    log_after = (REPO_ROOT / "handoff/harness_log.md").read_text(encoding="utf-8")
    added = log_after[len(log_before):]
    assert re.search(r"^## Cycle \d+", added, re.MULTILINE), (
        f"harness_log_gains_new_cycle_entry FAIL; last 500 new chars:\n{added[-500:]}"
    )

    print("PASS 4.17.1 all 5 criteria")


if __name__ == "__main__":
    try:
        test_harness_dry_run_exits_zero_and_appends_cycle()
    except AssertionError as e:
        print("FAIL:", e, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
