"""
Slack Bot Commands — Slash commands and channel message handlers.

The Assistant side-panel is handled by assistant_handler.py.
This module handles:
- /analyze, /portfolio, /report slash commands
- Messages in #ford-approvals channel (routed through multi-agent orchestrator)
- Reaction-based push approval (✅/❌)

Both paths use the same multi-agent orchestrator for consistent behavior.
"""

import logging
import subprocess
import time
from pathlib import Path

import httpx
from slack_bolt.app import App

from backend.config.settings import get_settings
from backend.slack_bot.formatters import (
    format_analysis_result,
    format_portfolio_summary,
    format_report_card,
)

logger = logging.getLogger(__name__)

_BACKEND_URL = "http://localhost:8000"
_APPROVAL_CHANNEL = "C0ANTGNNK8D"
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def register_commands(app: App):
    """Register slash commands and channel message handlers."""

    # ── Slash Commands ──────────────────────────────────────────

    @app.command("/analyze")
    def handle_analyze(ack, respond, command):
        ack()
        ticker = (command.get("text") or "").strip().upper()
        if not ticker or not ticker.isalpha() or len(ticker) > 5:
            respond("Usage: `/analyze AAPL` — provide a valid ticker symbol")
            return

        respond(
            f":hourglass_flowing_sand: Analysis started for **{ticker}**... "
            "This takes 2-5 minutes."
        )

        try:
            with httpx.Client(timeout=30.0) as client:
                start_res = client.post(
                    f"{_BACKEND_URL}/api/analysis/", json={"ticker": ticker}
                )
                start_res.raise_for_status()
                analysis_id = start_res.json()["analysis_id"]

                for _ in range(120):
                    time.sleep(5)
                    status_res = client.get(
                        f"{_BACKEND_URL}/api/analysis/{analysis_id}"
                    )
                    status_res.raise_for_status()
                    data = status_res.json()

                    if data["status"] == "completed":
                        blocks = format_analysis_result(data.get("report", {}), ticker)
                        respond(blocks=blocks)
                        return
                    elif data["status"] == "failed":
                        respond(
                            f":x: Analysis failed for {ticker}: "
                            f"{data.get('error', 'Unknown error')}"
                        )
                        return

                respond(f":warning: Analysis for {ticker} timed out.")
        except Exception as e:
            logger.exception(f"Error in /analyze for {ticker}")
            respond(f":x: Error analyzing {ticker}: {str(e)[:200]}")

    @app.command("/portfolio")
    def handle_portfolio(ack, respond, command):
        ack()
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.get(f"{_BACKEND_URL}/api/portfolio/performance")
                res.raise_for_status()
                blocks = format_portfolio_summary(res.json())
                respond(blocks=blocks)
        except Exception as e:
            logger.exception("Error in /portfolio")
            respond(f":x: Error: {str(e)[:200]}")

    @app.command("/report")
    def handle_report(ack, respond, command):
        ack()
        ticker = (command.get("text") or "").strip().upper()
        if not ticker or not ticker.isalpha() or len(ticker) > 5:
            respond("Usage: `/report AAPL`")
            return
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.get(f"{_BACKEND_URL}/api/reports/{ticker}")
                res.raise_for_status()
                blocks = format_report_card(res.json(), ticker)
                respond(blocks=blocks)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                respond(f":mag: No report for {ticker}. Run `/analyze {ticker}` first.")
            else:
                respond(f":x: Error fetching report for {ticker}")
        except Exception as e:
            logger.exception(f"Error in /report for {ticker}")
            respond(f":x: Error: {str(e)[:200]}")

    # ── Channel Message Handler (#ford-approvals) ───────────────

    @app.message("")
    def handle_channel_message(message, say, client, logger):
        """
        Handle messages in #ford-approvals via the multi-agent orchestrator.

        This is SEPARATE from the Assistant side-panel — it handles
        regular channel messages where users @mention or post in
        the designated channel.
        """
        channel = message.get("channel", "")
        if channel != _APPROVAL_CHANNEL:
            return
        if message.get("bot_id"):
            return

        text = message.get("text", "").strip()
        if not text:
            return

        thread_ts = message.get("thread_ts") or message.get("ts")
        sender = message.get("user", "unknown")
        logger.info(f"📨 Channel message from {sender}: {text[:100]}")

        # ── Deploy commands (self-update via Slack) ─────────────
        try:
            from backend.slack_bot.self_update import handle_deploy_command
            deploy_response = handle_deploy_command(text)
            if deploy_response is not None:
                say(text=deploy_response, thread_ts=thread_ts)
                return
        except Exception as e:
            logger.error(f"Deploy command error: {e}")

        try:
            import asyncio
            from backend.agents.multi_agent_orchestrator import get_orchestrator
            from backend.agents.agent_definitions import AgentType, QueryComplexity

            orchestrator = get_orchestrator()
            classification = orchestrator.classify_message_sync(text)

            # Show routing indicator for non-trivial queries
            if classification.agent_type != AgentType.DIRECT:
                agent_name = {
                    AgentType.MAIN: "Ford (Main)",
                    AgentType.QA: "Analyst (Q&A)",
                    AgentType.RESEARCH: "Researcher",
                }.get(classification.agent_type, "Agent")

                say(text=f"🔄 Routing to *{agent_name}*...", thread_ts=thread_ts)

            # Execute through orchestrator (pre-classified, no re-classification)
            result = orchestrator.execute_classified_sync(text, classification, sender)

            response = result.get("response", "No response generated.")
            processing_ms = result.get("processing_time_ms", 0)
            tokens = result.get("token_usage", {})

            # Add metadata footer for non-trivial responses
            if classification.agent_type != AgentType.DIRECT:
                footer = (
                    f"\n\n_🤖 {result.get('agent_type', '?')} · "
                    f"{processing_ms:.0f}ms · "
                    f"{tokens.get('input', 0)}+{tokens.get('output', 0)} tokens_"
                )
                response += footer

            say(text=response, thread_ts=thread_ts)
            logger.info(
                f"✅ Channel response via {result.get('agent_type', '?')} "
                f"in {processing_ms:.0f}ms"
            )

        except Exception as e:
            logger.exception(f"Orchestrator error: {text[:50]}")
            say(text=f"⚠️ Error: {str(e)[:200]}", thread_ts=thread_ts)

    # ── Reaction Handlers ───────────────────────────────────────

    @app.event("reaction_added")
    def handle_reaction(event, say):
        reaction = event.get("reaction", "")
        item = event.get("item", {})
        channel = item.get("channel", "")

        if channel != _APPROVAL_CHANNEL:
            return

        if reaction == "white_check_mark":
            logger.info("Push approved via ✅")
            try:
                result = subprocess.check_output(
                    ["git", "push", "origin", "main"],
                    cwd=str(_PROJECT_ROOT), text=True, timeout=30,
                    stderr=subprocess.STDOUT,
                )
                say(f"✅ *Pushed to GitHub*\n```{result.strip()}```")
            except subprocess.CalledProcessError as e:
                say(f"❌ *Push failed:*\n```{e.output[:500]}```")
            except Exception as e:
                say(f"❌ *Push error:* {str(e)[:200]}")
        elif reaction == "x":
            say("❌ Push rejected. Commits stay local.")
