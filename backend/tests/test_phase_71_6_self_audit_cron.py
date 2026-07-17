"""phase-71.6: deterministic weekly report-only harness self-audit scheduler.

Covers register_harness_self_audit_cron (weekly cron shape + fail-open) and
run_harness_self_audit_report (deterministic health report, report-only,
roster re-split guard, deep-audit staleness). No network / no LLM / no agents.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.harness_self_audit_report import (
    JOB_ID,
    DEEP_AUDIT_STALE_DAYS,
    register_harness_self_audit_cron,
    run_harness_self_audit_report,
)

_NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


class StubScheduler:
    """Minimal scheduler-like object capturing add_job kwargs."""

    def __init__(self, raise_on_add: bool = False):
        self.calls: list[dict] = []
        self._raise = raise_on_add

    def add_job(self, func, **kwargs):
        if self._raise:
            raise RuntimeError("simulated scheduler failure")
        self.calls.append({"func": func, **kwargs})


def _make_repo(tmp: Path, *, resplit: bool = False, proposals_age_days: int | None = 1) -> Path:
    """Build a minimal repo skeleton for the report writer."""
    agents = tmp / ".claude" / "agents"
    workflows = tmp / ".claude" / "workflows"
    current = tmp / "handoff" / "current"
    for d in (agents, workflows, current):
        d.mkdir(parents=True, exist_ok=True)
    (agents / "researcher.md").write_text("researcher", encoding="utf-8")
    (agents / "qa.md").write_text("qa", encoding="utf-8")
    if resplit:
        (agents / "explore.md").write_text("explore", encoding="utf-8")
    (workflows / "qa-verdict.js").write_text("// qa", encoding="utf-8")
    (workflows / "harness-self-audit.js").write_text("// audit", encoding="utf-8")
    for name in ("contract.md", "experiment_results.md", "evaluator_critique.md",
                 "research_brief.md"):
        (current / name).write_text(name, encoding="utf-8")
    if proposals_age_days is not None:
        p = current / "harness_proposals.json"
        p.write_text("{}", encoding="utf-8")
        old = _NOW.timestamp() - proposals_age_days * 86400
        os.utime(p, (old, old))
    return tmp


# ---- register_harness_self_audit_cron -------------------------------------

def test_register_adds_weekly_cron_job():
    sched = StubScheduler()
    job_id = register_harness_self_audit_cron(sched)
    assert job_id == JOB_ID
    assert len(sched.calls) == 1
    call = sched.calls[0]
    assert call["trigger"] == "cron"
    assert call["id"] == JOB_ID
    assert call["replace_existing"] is True
    assert call["day_of_week"] == "sun"
    assert call["func"] is run_harness_self_audit_report


def test_register_is_fail_open():
    sched = StubScheduler(raise_on_add=True)
    # Must NOT raise -- returns None so app startup is never broken.
    assert register_harness_self_audit_cron(sched) is None


# ---- run_harness_self_audit_report ----------------------------------------

def test_report_written_and_status_ok(tmp_path):
    root = _make_repo(tmp_path)
    res = run_harness_self_audit_report(repo_root=root, now=_NOW)
    assert res["status"] == "OK"
    assert res["attention"] == []
    report_path = Path(res["report_path"])
    assert report_path.is_file()
    assert report_path.name == "2026-07-17-harness-health.md"
    body = report_path.read_text(encoding="utf-8")
    assert "Layer-3 roster" in body
    assert "Saved harness workflows" in body
    assert "Five-file handoff protocol" in body
    assert "Deep agentic self-audit staleness" in body
    assert "REPORT-ONLY" in body


def test_report_is_deterministic(tmp_path):
    root = _make_repo(tmp_path)
    a = run_harness_self_audit_report(repo_root=root, now=_NOW)
    b = run_harness_self_audit_report(repo_root=root, now=_NOW)
    assert Path(a["report_path"]).read_text() == Path(b["report_path"]).read_text()


def test_resplit_guard_flags_attention(tmp_path):
    root = _make_repo(tmp_path, resplit=True)
    res = run_harness_self_audit_report(repo_root=root, now=_NOW)
    assert res["roster"]["ok"] is False
    assert "explore.md" in res["roster"]["resplit_files_found"]
    assert res["status"] == "ATTENTION"
    assert any("roster drift" in a for a in res["attention"])


def test_stale_deep_audit_flags_attention(tmp_path):
    root = _make_repo(tmp_path, proposals_age_days=DEEP_AUDIT_STALE_DAYS + 5)
    res = run_harness_self_audit_report(repo_root=root, now=_NOW)
    assert res["deep_audit"]["stale"] is True
    assert res["status"] == "ATTENTION"
    assert any("stale" in a for a in res["attention"])


def test_missing_proposals_flags_stale(tmp_path):
    root = _make_repo(tmp_path, proposals_age_days=None)
    res = run_harness_self_audit_report(repo_root=root, now=_NOW)
    assert res["deep_audit"]["exists"] is False
    assert res["deep_audit"]["stale"] is True


def test_report_only_no_writes_outside_self_audit(tmp_path):
    """The writer touches ONLY handoff/self_audit/ (report-only)."""
    root = _make_repo(tmp_path)
    before = {p for p in root.rglob("*") if p.is_file()}
    run_harness_self_audit_report(repo_root=root, now=_NOW)
    after = {p for p in root.rglob("*") if p.is_file()}
    new_files = after - before
    assert new_files, "expected a report file to be created"
    for p in new_files:
        assert (root / "handoff" / "self_audit") in p.parents, (
            "report-only writer created a file outside handoff/self_audit: %s" % p
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
