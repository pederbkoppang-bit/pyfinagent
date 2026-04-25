"""phase-10.7.2 unit tests for the Research Directive rewriter.

7 cases:
1. Below MIN_BRIEFS_FOR_PROPOSAL → returns None
2. Empty directive text → returns None
3. LLM returns invalid JSON → returns None
4. LLM returns valid JSON below score floor → returns None
5. LLM returns valid JSON above score floor → returns DirectiveVersion
6. No-op proposal (proposed_text == current) → returns None
7. Migration script --dry-run exits 0 + prints CREATE TABLE SQL

Tests use FakeLLM (via `llm_call_override`) + FakeBQ stubs;
NO live network calls.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.meta_evolution.directive_rewriter import (  # noqa: E402
    MIN_BRIEFS_FOR_PROPOSAL,
    MIN_LLM_JUDGE_SCORE,
    DirectiveVersion,
    persist_version,
    rewrite_directive,
)


CURRENT_DIRECTIVE = """---
name: researcher
description: ...
---

# Researcher Subagent

You are a research-gate subagent. Read at least 5 sources in full.
"""


def _make_briefs(n: int, gate_passed: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "tier": "simple",
            "gate_passed": gate_passed,
            "external_sources_read_in_full": 5,
            "urls_collected": 12,
            "recency_scan_performed": True,
        }
        for _ in range(n)
    ]


class FakeBQ:
    def __init__(self):
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def insert_rows_json(self, table_fqn: str, rows: list[dict[str, Any]]):
        self.calls.append((table_fqn, rows))
        return []


def test_below_min_briefs_returns_none():
    out = rewrite_directive(
        current_directive_text=CURRENT_DIRECTIVE,
        recent_briefs=_make_briefs(MIN_BRIEFS_FOR_PROPOSAL - 1),
        llm_call_override=lambda prompt: pytest.fail("LLM should not have been called"),
    )
    assert out is None


def test_empty_directive_returns_none():
    out = rewrite_directive(
        current_directive_text="",
        recent_briefs=_make_briefs(MIN_BRIEFS_FOR_PROPOSAL + 1),
        llm_call_override=lambda prompt: pytest.fail("LLM should not have been called"),
    )
    assert out is None


def test_llm_returns_invalid_json():
    out = rewrite_directive(
        current_directive_text=CURRENT_DIRECTIVE,
        recent_briefs=_make_briefs(MIN_BRIEFS_FOR_PROPOSAL + 2),
        llm_call_override=lambda prompt: None,
    )
    assert out is None


def test_llm_score_below_floor_returns_none():
    fake_response = {
        "diff_summary": "Minor wording tweak.",
        "proposed_text": CURRENT_DIRECTIVE.replace("at least 5", "at least 6"),
        "judge_score": MIN_LLM_JUDGE_SCORE - 0.1,
    }
    out = rewrite_directive(
        current_directive_text=CURRENT_DIRECTIVE,
        recent_briefs=_make_briefs(MIN_BRIEFS_FOR_PROPOSAL + 2),
        llm_call_override=lambda prompt: fake_response,
    )
    assert out is None


def test_llm_above_floor_returns_version():
    new_text = CURRENT_DIRECTIVE.replace(
        "Read at least 5 sources",
        "Read at least 5 sources and explicitly cite a 2026 source",
    )
    fake_response = {
        "diff_summary": "Add explicit 2026-recency requirement.",
        "proposed_text": new_text,
        "judge_score": 0.78,
    }
    captured = {}

    def _override(prompt: str):
        captured["prompt"] = prompt
        return fake_response

    briefs = _make_briefs(MIN_BRIEFS_FOR_PROPOSAL + 3)
    out = rewrite_directive(
        current_directive_text=CURRENT_DIRECTIVE,
        recent_briefs=briefs,
        outcome_signals={"recent_qa_verdicts": ["PASS", "PASS", "CONDITIONAL"]},
        llm_call_override=_override,
    )
    assert out is not None
    assert isinstance(out, DirectiveVersion)
    assert out.is_acceptable()
    # rewriter strips proposed_text, so compare stripped
    assert out.proposed_text == new_text.strip()
    assert out.judge_score == 0.78
    assert out.diff_size_bytes > 0
    assert "n_briefs" in captured["prompt"]
    assert "recent_qa_verdicts" in captured["prompt"]


def test_noop_proposal_returns_none():
    fake_response = {
        "diff_summary": "No change needed.",
        "proposed_text": CURRENT_DIRECTIVE,
        "judge_score": 0.9,
    }
    out = rewrite_directive(
        current_directive_text=CURRENT_DIRECTIVE,
        recent_briefs=_make_briefs(MIN_BRIEFS_FOR_PROPOSAL + 2),
        llm_call_override=lambda prompt: fake_response,
    )
    assert out is None


def test_persist_via_fake_bq_records_call():
    version = DirectiveVersion(
        version_id="rev-test-001",
        parent_version_id=None,
        proposed_text="---\nname: researcher\n---\nNew text",
        diff_summary="Test diff.",
        diff_size_bytes=42,
        judge_score=0.7,
        components={"foo": "bar"},
    )
    bq = FakeBQ()
    persist_version(bq, version)
    assert len(bq.calls) == 1
    table_fqn, rows = bq.calls[0]
    assert table_fqn.endswith("pyfinagent_pms.directive_versions")
    row = rows[0]
    assert row["version_id"] == "rev-test-001"
    assert row["judge_score"] == 0.7
    assert row["diff_size_bytes"] == 42
    components = json.loads(row["components_json"])
    assert components["foo"] == "bar"


def test_migration_script_dry_run():
    script = REPO_ROOT / "scripts" / "migrations" / "create_directive_versions_table.py"
    assert script.exists(), f"missing migration script: {script}"
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"dry-run exit {result.returncode}: {result.stderr}"
    out = result.stdout
    assert "DRY RUN" in out
    assert "CREATE TABLE IF NOT EXISTS" in out
    assert "directive_versions" in out
    assert "PARTITION BY DATE(proposed_at)" in out
    assert "CLUSTER BY proposer, parent_version_id" in out
