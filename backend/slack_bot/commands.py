"""
Slash command handlers: /analyze, /portfolio, /report
Message handlers: status command, push approval reactions
Ticket ingestion: all messages in #ford-approvals are persisted as tickets
"""

import asyncio
import logging
import subprocess
from pathlib import Path

import httpx
from slack_bolt.async_app import AsyncApp

from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_analysis_result, format_portfolio_summary, format_report_card
from backend.services.ticket_ingestion import get_ingestion_service

logger = logging.getLogger(__name__)

# Backend base URL (internal Docker network or localhost)
_BACKEND_URL = "http://localhost:8000"

# Ford approval channel
_APPROVAL_CHANNEL = "C0ANTGNNK8D"

# Project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _read_status() -> str:
    """Read current plan status from memory and plan files."""
    lines = []

    # Read today's memory
    from datetime import date
    today = date.today().isoformat()
    mem_path = Path.home() / ".openclaw" / "workspace" / "memory" / f"{today}.md"
    if mem_path.exists():
        content = mem_path.read_text()
        # Extract the last section (most recent work)
        sections = content.split("## ")
        if len(sections) > 1:
            last = sections[-1][:500]
            lines.append(f"*Today's work:*\n{last}")

    # Read plan status
    plan_path = _PROJECT_ROOT / "PLAN.md"
    if plan_path.exists():
        content = plan_path.read_text()
        # Count checked vs unchecked items
        checked = content.count("- [x]")
        unchecked = content.count("- [ ]")
        total = checked + unchecked
        pct = int(checked / total * 100) if total > 0 else 0
        lines.append(f"*Plan progress:* {checked}/{total} items ({pct}%)")

    # Git status
    try:
        local_commits = subprocess.check_output(
            ["git", "log", "origin/main..HEAD", "--oneline"],
            cwd=str(_PROJECT_ROOT), text=True, timeout=5
        ).strip()
        if local_commits:
            count = len(local_commits.splitlines())
            lines.append(f"*Local commits waiting for push:* {count}\n```{local_commits}```")
        else:
            lines.append("*Git:* All pushed, up to date.")
    except Exception:
        pass

    # Backtest status
    try:
        import urllib.request, json
        req = urllib.request.Request(f"{_BACKEND_URL}/api/backtest/status")
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        if resp.get("status") == "running":
            p = resp.get("progress", {})
            lines.append(f"*Backtest running:* Window {p.get('window', '?')}/{p.get('total_windows', '?')} ({p.get('elapsed_seconds', 0):.0f}s)")
        elif resp.get("status") == "completed" and resp.get("has_result"):
            lines.append("*Backtest:* Completed ✅")
    except Exception:
        pass

    return "\n\n".join(lines) if lines else "No status available."


def register_commands(app: AsyncApp):
    """Register all slash command handlers."""

    @app.command("/analyze")
    async def handle_analyze(ack, respond, command):
        """Start analysis for a ticker and post result when complete."""
        await ack()

        ticker = (command.get("text") or "").strip().upper()
        if not ticker or not ticker.isalpha() or len(ticker) > 5:
            await respond("Usage: `/analyze AAPL` — provide a valid ticker symbol")
            return

        await respond(f":hourglass_flowing_sand: Analysis started for **{ticker}**... This takes 2-5 minutes.")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Start analysis
                start_res = await client.post(f"{_BACKEND_URL}/api/analysis/", json={"ticker": ticker})
                start_res.raise_for_status()
                analysis_id = start_res.json()["analysis_id"]

                # Poll until complete (max 10 minutes)
                for _ in range(120):
                    await asyncio.sleep(5)
                    status_res = await client.get(f"{_BACKEND_URL}/api/analysis/{analysis_id}")
                    status_res.raise_for_status()
                    data = status_res.json()

                    if data["status"] == "completed":
                        blocks = format_analysis_result(data.get("report", {}), ticker)
                        await respond(blocks=blocks)
                        return
                    elif data["status"] == "failed":
                        await respond(f":x: Analysis failed for {ticker}: {data.get('error', 'Unknown error')}")
                        return

                await respond(f":warning: Analysis for {ticker} timed out. Check the dashboard.")

        except Exception as e:
            logger.exception(f"Error in /analyze for {ticker}")
            await respond(f":x: Error analyzing {ticker}: {str(e)[:200]}")

    @app.command("/portfolio")
    async def handle_portfolio(ack, respond, command):
        """Show portfolio P&L summary."""
        await ack()

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
                res.raise_for_status()
                data = res.json()
                blocks = format_portfolio_summary(data)
                await respond(blocks=blocks)
        except Exception as e:
            logger.exception("Error in /portfolio")
            await respond(f":x: Error fetching portfolio: {str(e)[:200]}")

    @app.command("/report")
    async def handle_report(ack, respond, command):
        """Show latest report for a ticker."""
        await ack()

        ticker = (command.get("text") or "").strip().upper()
        if not ticker or not ticker.isalpha() or len(ticker) > 5:
            await respond("Usage: `/report AAPL` — provide a valid ticker symbol")
            return

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(f"{_BACKEND_URL}/api/reports/{ticker}")
                res.raise_for_status()
                data = res.json()
                blocks = format_report_card(data, ticker)
                await respond(blocks=blocks)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                await respond(f":mag: No report found for {ticker}. Run `/analyze {ticker}` first.")
            else:
                await respond(f":x: Error fetching report for {ticker}")
        except Exception as e:
            logger.exception(f"Error in /report for {ticker}")
            await respond(f":x: Error: {str(e)[:200]}")

    # ── Channel message handlers ─────────────────────────────────

    @app.event("message_deleted")
    async def handle_message_deleted(event, logger):
        """Handle deleted messages — DO NOT delete associated tickets."""
        # When a user deletes their original message, we keep the ticket in the system
        # This preserves queue position and processing state
        message_ts = event.get("deleted_ts")
        logger.info(f"Message {message_ts} was deleted, but ticket preserved in system")
        # Tickets are NOT deleted — they continue processing and remain queryable

    @app.message("")  # Catch all messages
    async def handle_any_message(message, say, logger):
        """Respond to any message in #ford-approvals. Persists as ticket."""
        channel = message.get("channel", "")
        if channel != _APPROVAL_CHANNEL:
            return
        # Don't respond to bot messages
        if message.get("bot_id"):
            return
        
        text = message.get("text", "").strip()
        
        # Validate: reject empty messages
        if not text:
            logger.debug("Empty message received, skipping")
            return
        
        logger.info(f"Message received in #ford-approvals: {text[:100]}")
        
        # ── Ticket ingestion ────────────────────────────────────
        ingestion = get_ingestion_service()
        ticket_id = None
        try:
            ticket_id = ingestion.ingest_slack_message(
                event=message,
                sender_id=message.get("user", "unknown"),
                channel_id=channel,
            )
        except Exception as e:
            logger.exception(f"Failed to ingest message as ticket: {e}")
        
        # Send acknowledgment with ticket info
        ack_msg = None
        if ticket_id is not None:
            try:
                ack_info = ingestion.acknowledge_ticket_immediately(ticket_id)
                ack_msg = ack_info["message"]
                logger.info(f"Ticket #{ticket_id} created and acknowledged")
            except Exception as e:
                logger.exception(f"Failed to acknowledge ticket: {e}")
                ack_msg = f"✅ Message received (ticket #{ticket_id} created). Timestamp: {message.get('ts')}"
        else:
            # Ingestion failed — always send acknowledgment
            ack_msg = f"⚠️ Message received but failed to create ticket. Timestamp: {message.get('ts')}"
            logger.warning(f"Failed to create ticket from message: {text[:100]}")
        
        # ── Route based on content (existing behavior) ──────────
        text_lower = text.lower()
        if "status" in text_lower:
            try:
                status_text = _read_status()
                await say(f"📊 *PyFinAgent Status*\n\n{status_text}")
            except Exception as e:
                logger.exception("Error generating status")
                await say(f":x: Error generating status: {str(e)[:200]}")
        
        # Always send acknowledgment (either ticket confirmation or error)
        if ack_msg:
            thread_ts = message.get("thread_ts") or message.get("ts")
            try:
                await say(text=ack_msg, thread_ts=thread_ts)
                logger.debug(f"Acknowledgment sent for message: {text[:50]}")
            except Exception as e:
                logger.exception(f"Failed to send acknowledgment: {e}")

    @app.event("reaction_added")
    async def handle_reaction(event, say):
        """Handle ✅/❌ reactions on push approval messages."""
        reaction = event.get("reaction", "")
        item = event.get("item", {})
        channel = item.get("channel", "")

        if channel != _APPROVAL_CHANNEL:
            return

        if reaction == "white_check_mark":
            # Approved — push to GitHub
            logger.info("Push approved via ✅ reaction")
            try:
                result = subprocess.check_output(
                    ["git", "push", "origin", "main"],
                    cwd=str(_PROJECT_ROOT), text=True, timeout=30,
                    stderr=subprocess.STDOUT
                )
                await say(f"✅ *Pushed to GitHub*\n```{result.strip()}```")
            except subprocess.CalledProcessError as e:
                await say(f"❌ *Push failed:*\n```{e.output[:500]}```")
            except Exception as e:
                await say(f"❌ *Push error:* {str(e)[:200]}")

        elif reaction == "x":
            logger.info("Push rejected via ❌ reaction")
            await say("❌ Push rejected. Commits stay local.")
