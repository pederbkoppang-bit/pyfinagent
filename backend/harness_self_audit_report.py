"""phase-71.6 Weekly REPORT-ONLY harness self-audit scheduler (deterministic).

The Layer-3 stress-test doctrine says the harness should be re-audited on a
cadence. The DEEP audit is the agentic workflow `.claude/workflows/
harness-self-audit.js` (read-only Explore fan-out) -- but *scheduling an
agentic run* is the background-agent-resumption risk the operator flagged
(auto-memory `feedback_background_agent_resumption_risk`: a completed
background agent self-resumed and took an unauthorized action; review-only
prompts are NOT enforcement). So this module is the ENFORCEMENT-SAFE half:
a DETERMINISTIC, zero-agency weekly report writer that greps the harness
invariants and writes a health report for the operator to review.

Report-only by construction: NO LLM call, NO agent spawn, NO subprocess, NO
git, NO network, NO BigQuery, NO trade, NO risk-parameter read/write, NO
masterplan flip. It writes ONE markdown file under `handoff/self_audit/` and
returns a summary dict. It is observability, not a live-loop/trading change.

Two top-level functions (mirrors `backend/meta_evolution/cron.py`):

- `register_harness_self_audit_cron(scheduler, *, replace_existing=True, ...)`:
  Adds a weekly cron job to a scheduler-like object (any object with
  `add_job(func, trigger=..., id=..., replace_existing=..., **kwargs)`).
  Defaults to Sunday 03:00 America/New_York. Returns the registered job_id
  on success or None if add_job raised (fail-open).
- `run_harness_self_audit_report(*, repo_root=None, now=None)`: Executes one
  deterministic weekly health check and writes the report. Each sub-check is
  wrapped in its own try/except (fail-open per Google SRE monitoring-tier
  discipline -- a single sub-failure must not blank the whole report).

ASCII-only logger messages (per `.claude/rules/security.md`).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

JOB_ID = "harness_self_audit_weekly"
TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_DAY_OF_WEEK = "sun"
DEFAULT_HOUR = 3
DEFAULT_MINUTE = 0

# A deep agentic audit older than this many days earns a "run the deep audit"
# nudge in the weekly report. The nudge is advisory -- scheduling the agentic
# run stays the operator's call (resumption-risk category).
DEEP_AUDIT_STALE_DAYS = 14

# The Layer-3 harness is EXACTLY 3 agents: Main (the Claude Code session) +
# Researcher + Q/A. These two files are the only subagent definitions; the
# re-split guard flags any reappearance of the merged-away roles.
_EXPECTED_AGENT_FILES = ("researcher.md", "qa.md")
_RESPLIT_GUARD_FILES = (
    "explore.md",
    "harness-verifier.md",
    "harness_verifier.md",
    "qa-evaluator.md",
)
_EXPECTED_WORKFLOWS = ("qa-verdict.js", "harness-self-audit.js")
_FIVE_FILE_ARTIFACTS = (
    "contract.md",
    "experiment_results.md",
    "evaluator_critique.md",
    "research_brief.md",
)


def register_harness_self_audit_cron(
    scheduler: Any,
    *,
    replace_existing: bool = True,
    day_of_week: str = DEFAULT_DAY_OF_WEEK,
    hour: int = DEFAULT_HOUR,
    minute: int = DEFAULT_MINUTE,
) -> Optional[str]:
    """Register the weekly harness self-audit report job on a scheduler.

    `scheduler` must expose `.add_job(func, trigger, id, replace_existing,
    **kwargs)` (AsyncIOScheduler, BackgroundScheduler, StubScheduler).
    Returns the job_id on success or None if add_job raised (fail-open --
    a scheduler hiccup must never break app startup).

    Defaults: Sunday 03:00 America/New_York (offset from the meta_evolution
    02:00 slot). `replace_existing=True` per APScheduler userguide so a
    restart never duplicates the job in a persistent jobstore.
    """
    try:
        scheduler.add_job(
            run_harness_self_audit_report,
            trigger="cron",
            id=JOB_ID,
            replace_existing=replace_existing,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            timezone=TIMEZONE,
        )
    except Exception as exc:
        logger.warning(
            "register_harness_self_audit_cron add_job fail-open: %r", exc
        )
        return None
    logger.info(
        "registered harness_self_audit_weekly cron: %s %02d:%02d %s (report-only)",
        day_of_week,
        hour,
        minute,
        TIMEZONE.key,
    )
    return JOB_ID


def _check_roster(agents_dir: Path) -> dict[str, Any]:
    """Layer-3 roster integrity: exactly Researcher + Q/A, no re-split files."""
    present = [f for f in _EXPECTED_AGENT_FILES if (agents_dir / f).is_file()]
    resplit = [f for f in _RESPLIT_GUARD_FILES if (agents_dir / f).is_file()]
    ok = len(present) == len(_EXPECTED_AGENT_FILES) and not resplit
    return {
        "ok": ok,
        "expected_present": present,
        "missing": [f for f in _EXPECTED_AGENT_FILES if f not in present],
        "resplit_files_found": resplit,
    }


def _check_workflows(workflows_dir: Path) -> dict[str, Any]:
    present = [f for f in _EXPECTED_WORKFLOWS if (workflows_dir / f).is_file()]
    return {
        "ok": len(present) == len(_EXPECTED_WORKFLOWS),
        "present": present,
        "missing": [f for f in _EXPECTED_WORKFLOWS if f not in present],
    }


def _check_five_file(current_dir: Path, now: datetime) -> dict[str, Any]:
    rows = []
    for name in _FIVE_FILE_ARTIFACTS:
        p = current_dir / name
        if p.is_file():
            age_days = (
                now - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            ).days
            rows.append({"file": name, "present": True, "age_days": age_days})
        else:
            rows.append({"file": name, "present": False, "age_days": None})
    return {"ok": all(r["present"] for r in rows), "artifacts": rows}


def _check_deep_audit_staleness(
    proposals_path: Path, now: datetime
) -> dict[str, Any]:
    if not proposals_path.is_file():
        return {"ok": False, "exists": False, "age_days": None, "stale": True}
    age_days = (
        now - datetime.fromtimestamp(proposals_path.stat().st_mtime, tz=timezone.utc)
    ).days
    return {
        "ok": age_days <= DEEP_AUDIT_STALE_DAYS,
        "exists": True,
        "age_days": age_days,
        "stale": age_days > DEEP_AUDIT_STALE_DAYS,
    }


def run_harness_self_audit_report(
    *,
    repo_root: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Execute one deterministic weekly harness-health report.

    Zero-agency: greps checked-in harness invariants and writes a markdown
    report to `handoff/self_audit/<date>-harness-health.md`. Returns an
    aggregate dict (also usable for a log-line/telemetry). Each sub-check is
    fail-open so a single failure degrades that section, not the whole report.
    """
    root = repo_root or _REPO_ROOT_DEFAULT
    stamp = now or datetime.now(timezone.utc)

    agents_dir = root / ".claude" / "agents"
    workflows_dir = root / ".claude" / "workflows"
    current_dir = root / "handoff" / "current"
    proposals_path = current_dir / "harness_proposals.json"
    out_dir = root / "handoff" / "self_audit"

    results: dict[str, Any] = {
        "generated_at": stamp.isoformat(),
        "report_path": None,
        "roster": None,
        "workflows": None,
        "five_file": None,
        "deep_audit": None,
        "attention": [],
        "errors": [],
    }

    for key, fn in (
        ("roster", lambda: _check_roster(agents_dir)),
        ("workflows", lambda: _check_workflows(workflows_dir)),
        ("five_file", lambda: _check_five_file(current_dir, stamp)),
        ("deep_audit", lambda: _check_deep_audit_staleness(proposals_path, stamp)),
    ):
        try:
            results[key] = fn()
        except Exception as exc:  # fail-open per sub-check
            logger.warning("harness_self_audit %s check fail-open: %r", key, exc)
            results["errors"].append({"check": key, "error": repr(exc)})

    # Derive the attention list (deterministic; no LLM judgment).
    if results["roster"] and not results["roster"].get("ok"):
        results["attention"].append(
            "Layer-3 roster drift: expected exactly Researcher + Q/A"
        )
    if results["workflows"] and not results["workflows"].get("ok"):
        results["attention"].append("Saved harness workflow(s) missing")
    if results["deep_audit"] and results["deep_audit"].get("stale"):
        age = results["deep_audit"].get("age_days")
        results["attention"].append(
            "Deep agentic self-audit is stale (%s days) -- run the "
            "harness-self-audit workflow manually" % (age,)
        )

    status = "OK" if not results["attention"] else "ATTENTION"
    report = _render_report(results, stamp, status)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / ("%s-harness-health.md" % stamp.strftime("%Y-%m-%d"))
        report_path.write_text(report, encoding="utf-8")
        results["report_path"] = str(report_path)
        logger.info(
            "harness_self_audit report written: %s status=%s",
            report_path.name,
            status,
        )
    except Exception as exc:  # fail-open on the write itself
        logger.warning("harness_self_audit report write fail-open: %r", exc)
        results["errors"].append({"check": "write", "error": repr(exc)})

    results["status"] = status
    return results


def _render_report(results: dict[str, Any], stamp: datetime, status: str) -> str:
    """Render the deterministic health report as markdown (no LLM)."""
    lines: list[str] = []
    lines.append("# Harness self-audit -- weekly health report (deterministic)")
    lines.append("")
    lines.append("Generated: %s UTC | Status: **%s**" % (stamp.isoformat(), status))
    lines.append("")
    lines.append(
        "REPORT-ONLY / zero-agency: this is the deterministic weekly health "
        "nudge (no LLM, no agents, no git, no trades). For DEEP findings run "
        "the agentic `harness-self-audit` workflow MANUALLY (also report-only). "
        "Scheduling the *agentic* run stays the operator's call -- it is the "
        "background-agent-resumption category."
    )
    lines.append("")

    roster = results.get("roster") or {}
    lines.append("## Layer-3 roster (exactly 3: Main + Researcher + Q/A)")
    lines.append("- present: %s" % (roster.get("expected_present")))
    lines.append("- missing: %s" % (roster.get("missing")))
    lines.append("- re-split files found (should be empty): %s"
                 % (roster.get("resplit_files_found")))
    lines.append("")

    wf = results.get("workflows") or {}
    lines.append("## Saved harness workflows")
    lines.append("- present: %s" % (wf.get("present")))
    lines.append("- missing: %s" % (wf.get("missing")))
    lines.append("")

    ff = results.get("five_file") or {}
    lines.append("## Five-file handoff protocol (handoff/current/)")
    for row in (ff.get("artifacts") or []):
        lines.append(
            "- %s: present=%s age_days=%s"
            % (row.get("file"), row.get("present"), row.get("age_days"))
        )
    lines.append("")

    da = results.get("deep_audit") or {}
    lines.append("## Deep agentic self-audit staleness")
    lines.append(
        "- harness_proposals.json exists=%s age_days=%s stale(>%sd)=%s"
        % (da.get("exists"), da.get("age_days"), DEEP_AUDIT_STALE_DAYS, da.get("stale"))
    )
    lines.append("")

    lines.append("## Attention")
    if results.get("attention"):
        for item in results["attention"]:
            lines.append("- %s" % item)
    else:
        lines.append("- none")
    lines.append("")

    if results.get("errors"):
        lines.append("## Sub-check errors (fail-open)")
        for e in results["errors"]:
            lines.append("- %s: %s" % (e.get("check"), e.get("error")))
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "JOB_ID",
    "register_harness_self_audit_cron",
    "run_harness_self_audit_report",
]
