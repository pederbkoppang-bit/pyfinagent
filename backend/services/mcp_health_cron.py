"""Weekly MCP health check (phase-3.5 step 3.5.7).

Reads handoff/mcp_risk_scores.json (produced by scripts/audit/
mcp_risk_score.py) and re-pings the GitHub API for each server's
latest commit. Flags:
- stale > 180 days  -> advisory: "stale_repo"
- license change    -> advisory: "license_changed"
- high risk_band    -> advisory: "high_risk_band"
- gh unreachable    -> advisory: "github_error"

Critical advisories (stale_repo + license_changed) trigger a Slack
post to MCP_HEALTH_SLACK_CHANNEL (falls back to SLACK_TEST_CHANNEL_ID)
if SLACK_BOT_TOKEN is set.

The cron is wired into the existing APScheduler instance via
register_health_cron() -- no new scheduler process.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
RISK_SCORES_PATH = REPO / "handoff" / "mcp_risk_scores.json"
LAST_SNAPSHOT_PATH = REPO / "handoff" / "mcp_health_last.json"
DEFAULT_STALE_DAYS = 180

_CRITICAL_REASONS = {"stale_repo", "license_changed"}


def _gh_latest_commit(repo: str) -> tuple[str | None, str | None]:
    url = f"https://api.github.com/repos/{repo}/commits?per_page=1"
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "pyfinagent-mcp-health-cron/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if not data:
                return None, "empty_response"
            c = data[0]
            ts = c.get("commit", {}).get("committer", {}).get("date")
            sha = c.get("sha")
            return ts, sha
    except urllib.error.HTTPError as e:
        return None, f"http_{e.code}"
    except Exception as e:
        return None, f"{type(e).__name__}"


def _load_scores() -> dict:
    if not RISK_SCORES_PATH.exists():
        return {}
    try:
        return json.loads(RISK_SCORES_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("failed to read mcp_risk_scores.json: %s", e)
        return {}


def _load_last() -> dict:
    if not LAST_SNAPSHOT_PATH.exists():
        return {}
    try:
        return json.loads(LAST_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_last(snapshot: dict) -> None:
    LAST_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n",
                                   encoding="utf-8")


def _slack_post(text: str) -> dict:
    token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("MCP_HEALTH_SLACK_CHANNEL") or os.getenv("SLACK_TEST_CHANNEL_ID")
    if not token or not channel:
        return {"ok": False, "skipped": "env_missing"}
    try:
        from slack_sdk import WebClient
        resp = WebClient(token=token).chat_postMessage(channel=channel, text=text)
        return {"ok": bool(resp.get("ok")), "ts": resp.get("ts")}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}:{e}"}


def check_once(*, max_age_days: int = DEFAULT_STALE_DAYS,
                gh_sample_limit: int = 3) -> dict:
    """Run one health check pass. Returns a dict with `servers` list."""
    scores = _load_scores()
    last = _load_last()
    now = datetime.now(timezone.utc)

    servers: list[dict] = []
    advisories: list[dict] = []
    gh_calls = 0

    for name, meta in scores.items():
        repo = meta.get("repo")
        prior = last.get(name, {})
        entry: dict = {
            "name": name,
            "repo": repo,
            "license": meta.get("license"),
            "risk_band": meta.get("risk_band"),
        }

        if gh_calls < gh_sample_limit and repo:
            ts, sha_or_err = _gh_latest_commit(repo)
            gh_calls += 1
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    age = (now - dt).days
                except Exception:
                    age = None
                entry.update({
                    "last_commit_iso": ts,
                    "last_commit_age_days": age,
                    "github_ok": True,
                })
                if age is not None and age > max_age_days:
                    advisories.append({
                        "server": name, "reason": "stale_repo",
                        "detail": f"last commit {age}d ago",
                    })
            else:
                entry.update({
                    "github_ok": False,
                    "github_error": sha_or_err,
                })
                advisories.append({
                    "server": name, "reason": "github_error",
                    "detail": sha_or_err,
                })

        if prior.get("license") and prior["license"] != meta.get("license"):
            advisories.append({
                "server": name, "reason": "license_changed",
                "detail": f"{prior['license']} -> {meta.get('license')}",
            })

        if meta.get("risk_band") == "high":
            advisories.append({
                "server": name, "reason": "high_risk_band",
                "detail": f"total={meta.get('total_risk')}",
            })

        servers.append(entry)

    critical = [a for a in advisories if a["reason"] in _CRITICAL_REASONS]
    slack_result = None
    if critical:
        text = (
            f"*MCP health cron -- {len(critical)} critical advisories*\n"
            + "\n".join(f"- {a['server']}: {a['reason']} ({a['detail']})"
                         for a in critical[:10])
        )
        slack_result = _slack_post(text)

    snapshot = {
        name: {"license": meta.get("license"),
               "risk_band": meta.get("risk_band")}
        for name, meta in scores.items()
    }
    _write_last(snapshot)

    return {
        "run_at": now.isoformat(),
        "gh_calls": gh_calls,
        "servers": servers,
        "advisories": advisories,
        "critical_count": len(critical),
        "slack_post": slack_result,
    }


def register_health_cron(scheduler) -> None:
    """Register the weekly health check on an existing APScheduler
    instance. Called from paper_trading.init_scheduler. Returns None;
    side-effect: one job added."""
    from apscheduler.triggers.cron import CronTrigger

    scheduler.add_job(
        check_once,
        trigger=CronTrigger(day_of_week="sun", hour=2, minute=0,
                             timezone="UTC"),
        id="mcp_health_cron",
        replace_existing=True,
        name="mcp_health_cron",
    )
    logger.info("MCP health cron registered (Sun 02:00 UTC)")
