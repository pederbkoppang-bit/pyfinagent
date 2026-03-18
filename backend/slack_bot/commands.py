"""
Slash command handlers: /analyze, /portfolio, /report
"""

import asyncio
import logging

import httpx
from slack_bolt.async_app import AsyncApp

from backend.config.settings import get_settings
from backend.slack_bot.formatters import format_analysis_result, format_portfolio_summary, format_report_card

logger = logging.getLogger(__name__)

# Backend base URL (internal Docker network)
_BACKEND_URL = "http://backend:8000"


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
