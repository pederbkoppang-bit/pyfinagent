"""One-shot away-digest sender (phase-62.8, goal-away-ops).

Sends a real morning|evening digest to the configured Slack channel NOW,
without waiting for the scheduler cron. Used for the 62.8 live_check proof
and by PM sessions that need an out-of-band digest. Standalone WebClient
(no Socket Mode) per the scripts/ops/send_confirmation_digest.py pattern.

Usage (inside .venv):
    python scripts/away_ops/send_away_digest.py evening [--force-away]
    python scripts/away_ops/send_away_digest.py morning [--force-away]

--force-away renders the away sections regardless of settings.away_mode_enabled
(the flag stays OFF in .env until the operator's 62.7 keystroke; the live proof
must not depend on flipping it).
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from slack_sdk import WebClient

from backend.config.settings import get_settings
from backend.slack_bot.formatters import (
    format_away_compact_sections,
    format_away_digest_sections,
    format_evening_digest,
    format_morning_digest,
)
from backend.slack_bot.scheduler import _LOCAL_BACKEND_URL, _compute_system_state, _gather_away_data


async def _build(kind: str, force_away: bool) -> list[dict]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        portfolio_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/portfolio")
        portfolio_data = portfolio_res.json() if portfolio_res.status_code == 200 else {}

        away_sections = None
        if force_away or settings.away_mode_enabled:
            away = await _gather_away_data(client)
            away_sections = (format_away_digest_sections(away) if kind == "evening"
                             else format_away_compact_sections(away))

        if kind == "evening":
            trades_res = await client.get(
                f"{_LOCAL_BACKEND_URL}/api/paper-trading/trades?limit=200&since_today=true")
            raw = trades_res.json() if trades_res.status_code == 200 else []
            trades = raw.get("trades", []) if isinstance(raw, dict) else raw
            return format_evening_digest(portfolio_data, trades, away_sections=away_sections)

        reports_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/reports/?limit=5")
        reports = reports_res.json() if reports_res.status_code == 200 else []
        system_state = await _compute_system_state(client)
        return format_morning_digest(portfolio_data, reports, system_state=system_state,
                                     away_sections=away_sections)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["morning", "evening"])
    ap.add_argument("--force-away", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    token = settings.slack_bot_token.get_secret_value()
    if not token or not settings.slack_channel_id:
        print("slack token/channel not configured", file=sys.stderr)
        return 1

    blocks = asyncio.run(_build(args.kind, args.force_away))
    assert len(blocks) <= 50, f"block cap exceeded: {len(blocks)}"

    client = WebClient(token=token)
    resp = client.chat_postMessage(
        channel=settings.slack_channel_id,
        blocks=blocks,
        text=f"PyFinAgent {args.kind} digest (one-shot) -- {datetime.now():%Y-%m-%d %H:%M}",
    )
    link = client.chat_getPermalink(channel=resp["channel"], message_ts=resp["ts"])
    print(f"sent ts={resp['ts']} blocks={len(blocks)}")
    print(f"permalink={link['permalink']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
