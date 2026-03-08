"""
Slack notification tool.
Migrated from pyfinagent-app/tools/slack.py — no Streamlit dependency.
"""

import logging
import httpx

logger = logging.getLogger(__name__)


async def send_notification(webhook_url: str, message: str, metadata: dict, alert_type: str = "info"):
    """Sends a formatted notification to a Slack channel."""
    if not webhook_url:
        logger.warning("Slack webhook URL not configured. Skipping notification.")
        return

    colors = {
        "info": "#17a2b8",
        "success": "#28a745",
        "warning": "#ffc107",
        "error": "#dc3545",
    }

    metadata_str = "\n".join([f"*{key}:* {value}" for key, value in metadata.items()])
    payload = {
        "attachments": [{
            "color": colors.get(alert_type, "#6c757d"),
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"*{message}*\n{metadata_str}"}}],
        }]
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")
