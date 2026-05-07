"""phase-23.3.5: regression guard for the log allowlist paths.

Pre-fix: _log_paths() pointed mas-harness/autoresearch/mas-harness.launchd
at handoff/logs/<x>.log, but the live launchd services write to
handoff/<x>.log (repo root). The /cron Logs tab silently showed 18-day
stale duplicates.

Post-fix: 9 keys total (was 6), all pointing at the correct live paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.api import cron_dashboard_api as cda


def _expected_relpath(p: Path) -> str:
    """Render a Path relative to the repo root for stable matching."""
    return str(p.relative_to(cda._REPO_ROOT))


def test_allowlist_has_9_keys():
    paths = cda._log_paths()
    assert len(paths) == 9, f"expected 9 allowlisted keys, got {len(paths)}: {sorted(paths)}"


def test_harness_repointed_to_repo_root():
    paths = cda._log_paths()
    assert _expected_relpath(paths["harness"]) == "handoff/mas-harness.log", (
        f"harness must point at repo-root handoff/mas-harness.log; "
        f"got {paths['harness']}"
    )


def test_autoresearch_repointed_to_repo_root():
    paths = cda._log_paths()
    assert _expected_relpath(paths["autoresearch"]) == "handoff/autoresearch.log"


def test_mas_harness_launchd_repointed_to_repo_root():
    paths = cda._log_paths()
    assert _expected_relpath(paths["mas_harness_launchd"]) == "handoff/mas-harness.launchd.log"


def test_new_keys_present():
    paths = cda._log_paths()
    expected_new = {
        "autoresearch_launchd": "handoff/autoresearch.launchd.log",
        "ablation":             "handoff/ablation.log",
        "ablation_launchd":     "handoff/ablation.launchd.log",
    }
    for key, rel in expected_new.items():
        assert key in paths, f"missing new key: {key}"
        assert _expected_relpath(paths[key]) == rel, (
            f"{key} should map to {rel}; got {paths[key]}"
        )


def test_unchanged_keys_preserved():
    """backend, watchdog, restart must NOT be re-pointed -- they
    write correctly to their current locations."""
    paths = cda._log_paths()
    assert _expected_relpath(paths["backend"]) == "backend.log"
    assert _expected_relpath(paths["watchdog"]) == "handoff/logs/backend-watchdog.log"
    assert _expected_relpath(paths["restart"]) == "handoff/logs/backend-restart.log"
