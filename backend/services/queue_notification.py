"""
Queue Position Update Notifications

Notifies users when their ticket moves up in the queue.
Sends updates via Slack or iMessage depending on message source.
"""

import logging
import subprocess
from typing import Dict, Any

logger = logging.getLogger(__name__)


class QueueNotificationService:
    """Send queue position updates to users."""

    def __init__(self):
        self.slack_client = None
        # NOTE: Client is lazy-loaded on first use to avoid duplicate instantiation
        self._slack_client_initialized = False

    async def send_failover_notification(self, ticket_number: int, from_agent: str, to_agent: str) -> bool:
        """Notify user that we're failing over due to rate limit/timeout."""
        message = (
            f"⚠️ Model Failover: {from_agent} throttled/hanging. "
            f"Switching to {to_agent} for faster response to Ticket #{ticket_number}..."
        )
        
        try:
            # Send to iMessage if we have the number
            import subprocess
            subprocess.run(
                ["imsg", "send", "--to", "+4794810537", "--text", message],
                timeout=5
            )
            logger.info(f"✅ Failover notification sent for ticket #{ticket_number}")
            return True
        except Exception as e:
            logger.error(f"Failed to send failover notification: {e}")
            return False
    
    async def notify_queue_position_change(self, ticket: Dict[str, Any], new_position: int) -> bool:
        """
        Notify user of queue position change.
        
        Args:
            ticket: Ticket dict with source, channel_id/sender_id, ticket_number
            new_position: New position in queue (1 = next to process)
            
        Returns:
            bool: True if notification sent successfully
        """
        ticket_number = ticket.get('ticket_number')
        source = ticket.get('source', 'slack')
        
        # Format message
        message = f"📍 Queue Update: Ticket #{ticket_number} moved to position #{new_position}"
        
        try:
            if source == 'slack':
                return await self._notify_slack(ticket, message, new_position)
            elif source == 'imessage':
                return await self._notify_imessage(ticket, message, new_position)
            else:
                logger.warning(f"Unknown source: {source}")
                return False
        except Exception as e:
            logger.error(f"Error notifying user of position change: {e}")
            return False

    def _ensure_slack_client(self):
        """Lazy-initialize Slack client to avoid duplicate instantiation."""
        if not self._slack_client_initialized:
            try:
                from slack_sdk import WebClient
                from backend.config.settings import get_settings
                settings = get_settings()
                self.slack_client = WebClient(token=settings.slack_bot_token)
                self._slack_client_initialized = True
                logger.debug("Slack client initialized")
            except Exception as e:
                logger.warning(f"Slack client not available: {e}")
                self._slack_client_initialized = True  # Mark as attempted

    async def _notify_slack(self, ticket: Dict[str, Any], message: str, position: int) -> bool:
        """Send queue position update via Slack."""
        self._ensure_slack_client()
        if not self.slack_client:
            logger.warning("Slack client not available")
            return False
        
        try:
            channel_id = ticket.get('channel_id', '')
            thread_ts = ticket.get('slack_thread_id')
            
            # Send thread reply if we have thread ID, otherwise send channel message
            if thread_ts:
                self.slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=message,
                    blocks=[
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": f"*{message}*\n_Your ticket is being processed..._"}
                        }
                    ]
                )
            else:
                # Fallback: send as channel message
                self.slack_client.chat_postMessage(
                    channel=channel_id,
                    text=message
                )
            
            logger.info(f"✅ Slack notification sent for ticket #{ticket.get('ticket_number')}: position #{position}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    async def _notify_imessage(self, ticket: Dict[str, Any], message: str, position: int) -> bool:
        """Send queue position update via iMessage."""
        try:
            phone = ticket.get('sender_id', '+4794810537')
            
            result = subprocess.run(
                ["imsg", "send", "--to", phone, "--text", message],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logger.info(f"✅ iMessage notification sent for ticket #{ticket.get('ticket_number')}: position #{position}")
                return True
            else:
                logger.error(f"Failed to send iMessage: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Failed to send iMessage notification: {e}")
            return False


# Global instance
_notification_service = None


def get_queue_notification_service() -> QueueNotificationService:
    """Get the global queue notification service."""
    global _notification_service
    if _notification_service is None:
        _notification_service = QueueNotificationService()
    return _notification_service
