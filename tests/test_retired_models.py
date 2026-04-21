"""phase-4.14.10 CI assertion: retired Claude snapshot IDs stay out.

Anthropic deprecation dates (as of 2026-04):
- claude-3-haiku-20240307: retired 2026-04-19
- claude-3-5-sonnet-20241022: retired 2025-10-28
- claude-3-5-haiku-20241022: retired 2026-02-19
- claude-3-7-sonnet-20250219: retired 2026-02-19
- claude-sonnet-4-20250514: retired snapshot alias superseded by claude-sonnet-4-6

These IDs MUST NOT appear in:
  - backend/**/*.py
  - scripts/**/*.py
  - backend/agents/llm_client.py :: GITHUB_MODELS_CATALOG

Any caller must use the canonical IDs from backend/config/model_tiers.py.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Canonical "retired" set mirroring masterplan success_criteria
# (haiku_3_ci_assert_added_retired_2026_04_19) + sibling snapshots.
RETIRED_SNAPSHOTS: tuple[str, ...] = (
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
)


def _grep_source(pattern: str) -> list[str]:
    """Return list of grep hits in backend/ + scripts/ Python files only."""
    result = subprocess.run(
        [
            "grep", "-rn", "--include=*.py",
            "-E", pattern,
            str(REPO / "backend"),
            str(REPO / "scripts"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return [ln for ln in result.stdout.splitlines() if ln.strip()]


def test_no_retired_snapshot_ids_in_source() -> None:
    """Assert retired Claude snapshot IDs are absent from live Python source."""
    pattern = "|".join(RETIRED_SNAPSHOTS)
    hits = _grep_source(pattern)
    assert not hits, (
        "Retired Claude snapshot IDs found in source -- replace with canonical "
        "ids from backend.config.model_tiers:\n" + "\n".join(hits)
    )


def test_github_models_catalog_has_no_retired_snapshots() -> None:
    """Direct assertion against the in-memory catalog."""
    from backend.agents.llm_client import GITHUB_MODELS_CATALOG

    leaked = set(RETIRED_SNAPSHOTS) & set(GITHUB_MODELS_CATALOG)
    assert not leaked, (
        f"GITHUB_MODELS_CATALOG still includes retired snapshots: {sorted(leaked)}"
    )


def test_haiku_3_retirement_is_explicitly_asserted() -> None:
    """The specific success criterion: claude-3-haiku-20240307 (retired 2026-04-19).

    This test exists so the CI history has a named assertion the auditor can
    point at (masterplan 4.14.10 success_criteria: haiku_3_ci_assert_added_retired_2026_04_19).
    """
    haiku_3 = "claude-3-haiku-20240307"
    assert haiku_3 in RETIRED_SNAPSHOTS
    hits = _grep_source(haiku_3)
    assert not hits, (
        f"{haiku_3} found in source (retired by Anthropic 2026-04-19):\n"
        + "\n".join(hits)
    )
