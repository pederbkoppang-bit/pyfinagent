"""
Phase 4: Response Delivery Service
Sends ticket responses back to users via Slack and iMessage.
"""

import logging
import subprocess
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from backend.db.tickets_db import get_tickets_db, TicketSource

logger = logging.getLogger(__name__)

class ResponseDeliveryService:
    """Service that delivers ticket responses back to users."""
    
    def __init__(self):
        self.db = get_tickets_db()
        
    def send_imessage_response(
        self, 
        phone_number: str, 
        message: str,
        ticket_number: int = None
    ) -> bool:
        """
        Send response via iMessage.
        
        Args:
            phone_number: Recipient phone number
            message: Response message
            ticket_number: Optional ticket number for logging
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Add ticket footer if ticket number provided
            if ticket_number:
                footer = f"\n\n— Ticket #{ticket_number}"
                # Keep message under 160 chars if possible
                if len(message) + len(footer) <= 160:
                    message += footer
            
            result = subprocess.run(
                ["imsg", "send", "--to", phone_number, "--text", message],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"✅ iMessage sent to {phone_number}: {message[:50]}...")
                return True
            else:
                logger.error(f"❌ iMessage send failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ iMessage send timeout to {phone_number}")
            return False
        except Exception as e:
            logger.error(f"❌ iMessage send error to {phone_number}: {e}")
            return False

    async def send_slack_response(
        self,
        channel_id: str,
        message: str,
        thread_ts: str = None,
        ticket_number: int = None
    ) -> bool:
        """
        Send response via Slack using AsyncWebClient.

        Args:
            channel_id: Slack channel ID
            message: Response message
            thread_ts: Thread timestamp for thread replies
            ticket_number: Optional ticket number for logging

        Returns:
            bool: True if sent successfully
        """
        try:
            from slack_sdk.web.async_client import AsyncWebClient
            from backend.config.settings import get_settings

            settings = get_settings()
            if not settings.slack_bot_token:
                logger.error("SLACK_BOT_TOKEN not configured, cannot send Slack response")
                return False

            # Add ticket footer
            if ticket_number:
                message += f"\n\n_Ticket #{ticket_number}_"

            client = AsyncWebClient(token=settings.slack_bot_token)

            kwargs = {
                "channel": channel_id,
                "text": message,
            }
            if thread_ts:
                kwargs["thread_ts"] = thread_ts

            result = await client.chat_postMessage(**kwargs)

            if result["ok"]:
                thread_info = f" (thread: {thread_ts})" if thread_ts else ""
                logger.info(f"Slack sent to {channel_id}{thread_info}: {message[:50]}...")
                return True
            else:
                logger.error(f"Slack API error: {result.get('error', 'unknown')}")
                return False

        except Exception as e:
            logger.error(f"Slack send error to {channel_id}: {e}")
            return False

    async def deliver_ticket_response(self, ticket_id: int) -> bool:
        """
        Deliver the response for a resolved ticket back to the user.
        
        Args:
            ticket_id: ID of the resolved ticket
            
        Returns:
            bool: True if delivered successfully
        """
        # Get ticket details
        ticket = self.db.get_ticket(ticket_id)
        if not ticket:
            logger.error(f"❌ Ticket {ticket_id} not found for delivery")
            return False
        
        if ticket['status'] != 'RESOLVED':
            logger.error(f"❌ Ticket {ticket_id} not resolved, cannot deliver")
            return False
        
        if not ticket['response_text']:
            logger.error(f"❌ Ticket {ticket_id} has no response text")
            return False
        
        # Determine delivery method
        source = ticket['source']
        response_text = ticket['response_text']
        ticket_number = ticket['ticket_number']
        
        try:
            if source == TicketSource.IMESSAGE.value:
                # Deliver via iMessage
                success = self.send_imessage_response(
                    phone_number=ticket['sender_id'],
                    message=response_text,
                    ticket_number=ticket_number
                )
                
            elif source == TicketSource.SLACK.value:
                # Deliver via Slack
                success = await self.send_slack_response(
                    channel_id=ticket['channel_id'],
                    message=response_text,
                    thread_ts=ticket['slack_thread_id'],
                    ticket_number=ticket_number
                )
                
            else:
                logger.error(f"❌ Unknown ticket source: {source}")
                return False
            
            if success:
                # Update ticket to mark response as delivered
                logger.info(f"✅ Response delivered for ticket #{ticket_number}")
                return True
            else:
                logger.error(f"❌ Failed to deliver response for ticket #{ticket_number}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error delivering ticket {ticket_id} response: {e}")
            return False

    async def deliver_pending_responses(self, limit: int = 50) -> int:
        """
        Deliver all pending responses (resolved tickets without delivery).
        
        Args:
            limit: Maximum number of tickets to process
            
        Returns:
            int: Number of responses delivered
        """
        # Get resolved tickets (these need response delivery)
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id FROM tickets 
                WHERE status = 'RESOLVED' 
                  AND response_text IS NOT NULL
                ORDER BY resolved_at ASC
                LIMIT ?
            """, (limit,))
            
            ticket_ids = [row[0] for row in cursor.fetchall()]
        
        if not ticket_ids:
            return 0
        
        logger.info(f"📤 Delivering responses for {len(ticket_ids)} tickets")
        
        delivered_count = 0
        for ticket_id in ticket_ids:
            try:
                success = await self.deliver_ticket_response(ticket_id)
                if success:
                    delivered_count += 1
                    
                # Small delay between deliveries
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Failed to deliver ticket {ticket_id}: {e}")
        
        logger.info(f"📤 Delivered {delivered_count}/{len(ticket_ids)} responses")
        return delivered_count

    def get_delivery_stats(self) -> Dict[str, int]:
        """Get delivery statistics."""
        import sqlite3
        with sqlite3.connect(self.db.db_path) as conn:
            # Count resolved tickets (these have responses to deliver)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM tickets 
                WHERE status = 'RESOLVED' AND response_text IS NOT NULL
            """)
            pending_delivery = cursor.fetchone()[0]
            
            # Count by source
            cursor = conn.execute("""
                SELECT source, COUNT(*) FROM tickets 
                WHERE status = 'RESOLVED' 
                GROUP BY source
            """)
            by_source = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                "pending_delivery": pending_delivery,
                "by_source": by_source
            }

# Global service instance
_delivery_service: Optional[ResponseDeliveryService] = None

def get_delivery_service() -> ResponseDeliveryService:
    """Get the global response delivery service."""
    global _delivery_service
    if _delivery_service is None:
        _delivery_service = ResponseDeliveryService()
    return _delivery_service