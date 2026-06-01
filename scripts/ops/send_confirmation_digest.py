"""phase-54.2: send ONE confirmation digest to the operator's Slack channel.

Operator-away lifeline proof. This is a STANDALONE one-shot: it builds a bare
`AsyncWebClient` and POSTs once via the Web API. It opens NO Socket Mode connection,
so it CANNOT create a second bot instance (the running bot, PID-managed by the 5-min
crontab monitor, is untouched). $0 / no LLM -- pure template + internal HTTP GETs.

Usage:  python scripts/ops/send_confirmation_digest.py
Prints the chat.postMessage result (ok / ts / channel) as the live_check evidence.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Make `backend` importable when run as a plain script (python scripts/ops/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from slack_sdk.web.async_client import AsyncWebClient

from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_morning_digest

_BACKEND = "http://127.0.0.1:8000"


async def _fetch(client: httpx.AsyncClient, path: str, default):
    try:
        r = await client.get(f"{_BACKEND}{path}")
        return r.json() if r.status_code == 200 else default
    except Exception:
        return default


async def _cron_health(client: httpx.AsyncClient) -> str | None:
    jobs = (await _fetch(client, "/api/jobs/all", {})).get("jobs", [])
    if not jobs:
        return None
    bad = [j.get("id", "?") for j in jobs if j.get("status") == "failed"]
    n_green = sum(1 for j in jobs if j.get("status") in ("ok", "scheduled", "running"))
    if bad:
        return f":warning: *Crons:* {n_green}/{len(jobs)} healthy -- FAILED: {', '.join(bad)}"
    return f":white_check_mark: *Crons:* {n_green}/{len(jobs)} healthy"


async def _system_state(client: httpx.AsyncClient) -> str | None:
    """Kill-switch + go-live-gate state (mirrors scheduler._compute_system_state)."""
    lines: list[str] = []
    k = await _fetch(client, "/api/paper-trading/kill-switch", {})
    if k:
        br = k.get("breach", {}) or {}
        if k.get("paused"):
            r = k.get("pause_reason")
            lines.append(":octagonal_sign: *Kill switch:* PAUSED" + (f" -- {r}" if r else ""))
        elif br.get("any_breached"):
            lines.append(":red_circle: *Kill switch:* BREACH")
        else:
            lines.append(":large_green_circle: *Kill switch:* ACTIVE")
        d, t = br.get("daily_loss_pct"), br.get("trailing_dd_pct")
        dl, tl = br.get("daily_loss_limit_pct"), br.get("trailing_dd_limit_pct")
        if None not in (d, t, dl, tl):
            lines[-1] += f" (daily {d:+.1f}%/{dl:.0f}% | trail {t:+.1f}%/{tl:.0f}%)"
    g = await _fetch(client, "/api/paper-trading/gate", {})
    if g:
        b = g.get("booleans", {}) or {}
        elig = "ELIGIBLE" if g.get("promote_eligible") else "NOT ELIGIBLE"
        lines.append(f"*Go-live gate:* {elig} ({sum(1 for v in b.values() if v)}/{len(b)})")
    return "\n".join(lines) if lines else None


def _away_week_blocks() -> list[dict]:
    """One-time away-week context (NOT part of the shared daily formatter)."""
    return [
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    ":satellite: *Away-week monitoring CONFIRMED* (operator remote "
                    "2026-06-01 -> 2026-06-08).\n"
                    "This is a one-time confirmation that your Slack lifeline delivers "
                    "end-to-end. The *daily morning digest* now carries a cron-health "
                    "line (above)."
                ),
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*Autonomous harness status*\n"
                    ":white_check_mark: repo synced to origin/main\n"
                    ":white_check_mark: phase-54.1 -- cron audit + `paper_markets` parse fix\n"
                    "_(Note: the cron line above may still list `autoresearch` + "
                    "`ablation` as FAILED for ONE more night -- that is their LAST run "
                    "(yesterday); the 54.1 fix clears on tonight's 02:00/03:00 fire. "
                    "`never_run` = a job that simply hasn't fired yet today, e.g. the "
                    "evening digest.)_\n"
                    ":hourglass_flowing_sand: phase-54.2 -- this digest\n"
                    ":soon: 50.6 multi-market UI -> 43.0 DoD audit -> 53.1-53.5 elevation + remote go-live\n"
                    "Operator-gated items (LLM spend / pip / BQ-DROP / NextAuth visual "
                    "confirms) are batched into `cycle_block_summary.md` for your review -- "
                    "I will not spend or do destructive ops without your go."
                ),
            },
        },
    ]


async def main() -> int:
    s = get_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        portfolio = await _fetch(client, "/api/paper-trading/portfolio", {})
        reports = await _fetch(client, "/api/reports/?limit=5", [])
        cron = await _cron_health(client)
        sysstate = await _system_state(client)

    blocks = format_morning_digest(portfolio, reports, cron_health=cron, system_state=sysstate)
    blocks += _away_week_blocks()

    client = AsyncWebClient(token=s.slack_bot_token.get_secret_value())
    resp = await client.chat_postMessage(
        channel=s.slack_channel_id,
        blocks=blocks,
        text=f"PyFinAgent away-week monitoring confirmed -- {datetime.now().strftime('%B %d, %Y')}",
    )
    print(f"ok={resp.get('ok')} channel={resp.get('channel')} ts={resp.get('ts')}")
    return 0 if resp.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
