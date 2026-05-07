"""phase-23.3.4: regression guard for the launchd manifest expansion.

Pre-fix: _LAUNCHD_JOBS had 1 entry (backend-watchdog only). 5 other
pyfinagent launchd services were silently invisible on /cron.

Post-fix: 6 entries covering backend, frontend, mas-harness, ablation,
autoresearch.
"""

from __future__ import annotations

import pytest

from backend.api import cron_dashboard_api as cda


def test_launchd_manifest_has_6_entries():
    assert len(cda._LAUNCHD_JOBS) == 6, \
        f"expected 6 launchd entries (was 1 pre-23.3.4); got {len(cda._LAUNCHD_JOBS)}"


def test_launchd_manifest_includes_5_new_services():
    ids = {j["id"] for j in cda._LAUNCHD_JOBS}
    expected_new = {
        "com.pyfinagent.backend",
        "com.pyfinagent.frontend",
        "com.pyfinagent.mas-harness",
        "com.pyfinagent.ablation",
        "com.pyfinagent.autoresearch",
    }
    missing = expected_new - ids
    assert not missing, f"missing launchd ids: {missing}"


def test_launchd_manifest_excludes_claude_code_proxy():
    """Claude Code's own service must NOT be in pyfinagent's manifest."""
    ids = {j["id"] for j in cda._LAUNCHD_JOBS}
    assert "com.pyfinagent.claude-code-proxy" not in ids, \
        "claude-code-proxy is Claude Code's own service; must not be in pyfinagent's manifest"


def test_autoresearch_failure_is_documented():
    """Documenting the active failure in the manifest description."""
    autoresearch = next(
        (j for j in cda._LAUNCHD_JOBS if j["id"] == "com.pyfinagent.autoresearch"),
        None,
    )
    assert autoresearch is not None
    desc = autoresearch["description"].lower()
    assert "failing" in desc or "fail" in desc, \
        "autoresearch entry must surface its current FAILING state"
