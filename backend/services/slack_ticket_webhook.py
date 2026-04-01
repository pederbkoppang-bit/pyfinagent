"""
Phase 2: Slack Webhook Handler with Ticket System
Processes Slack events and creates tickets for reliable message handling.
"""

import logging
import time
from typing import Dict, Any, Optional
from fastapi import HTTPException

from backend.services.ticket_ingestion import get_ingestion_service
from backend.db.tickets_db import get_tickets_db

logger = logging.getLogger(__name__)

class SlackTicketWebhookHandler:
    """Handles Slack webhook events and creates tickets."""
    
    def __init__(self):
        self.ingestion = get_ingestion_service()
        self.db = get_tickets_db()
        
    def handle_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming Slack webhook event.
        
        Must respond within 3 seconds to avoid Slack retries.
        
        Args:
            event_data: Full Slack webhook payload
            
        Returns:
            dict: Response data including ticket info
        """
        start_time = time.time()
        
        try:
            # Extract event details
            event = event_data.get("event", {})
            event_type = event.get("type")
            
            # Only handle message events for now
            if event_type != "message":
                logger.debug(f"Ignoring non-message event: {event_type}")
                return {"status": "ignored", "reason": "not_a_message"}
            
            # Skip bot messages and message changes/deletes
            if event.get("subtype") or event.get("bot_id"):
                logger.debug("Ignoring bot message or subtype")
                return {"status": "ignored", "reason": "bot_or_subtype"}
            
            # Extract message details
            text = event.get("text", "").strip()
            user_id = event.get("user")
            channel_id = event.get("channel")
            ts = event.get("ts")
            thread_ts = event.get("thread_ts")
            
            # Skip empty messages
            if not text:
                logger.debug("Ignoring empty message")
                return {"status": "ignored", "reason": "empty_text"}
            
            # Create envelope_id for deduplication
            envelope_id = f"{channel_id}_{ts}"
            
            # Check for deduplication first
            if self.db.is_duplicate_envelope(envelope_id):
                logger.info(f"Duplicate Slack event {envelope_id}, marking as duplicate")
                duplicate_id = self.db.mark_duplicate(envelope_id)
                return {
                    "status": "duplicate", 
                    "envelope_id": envelope_id,
                    "duplicate_ticket_id": duplicate_id
                }
            
            # Create enhanced event data for ticket
            enhanced_event = {
                **event,
                "envelope_id": envelope_id,
                "webhook_received_at": time.time(),
                "channel_id": channel_id
            }
            
            # Create ticket
            ticket_id = self.ingestion.ingest_slack_message(
                event=enhanced_event,
                sender_id=user_id,
                sender_name=self._get_user_name(user_id),
                channel_id=channel_id
            )
            
            if ticket_id is None:
                # Was a duplicate
                return {"status": "duplicate", "envelope_id": envelope_id}
            
            # Generate acknowledgment
            ack_info = self.ingestion.acknowledge_ticket_immediately(ticket_id)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Slack → Ticket #{ack_info['ticket_number']} "
                       f"in {processing_time_ms:.1f}ms (target: <100ms)")
            
            # Return success response
            return {
                "status": "success",
                "ticket_id": ticket_id,
                "ticket_number": ack_info["ticket_number"],
                "envelope_id": envelope_id,
                "processing_time_ms": processing_time_ms,
                "acknowledgment": ack_info["message"],
                "priority": ack_info["priority"],
                "classification": ack_info["agent_type"]
            }
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Failed to process Slack event in {processing_time_ms:.1f}ms: {e}")
            
            # Return error but don't fail the webhook (Slack will retry)
            return {
                "status": "error",
                "error": str(e),
                "processing_time_ms": processing_time_ms
            }

    def _get_user_name(self, user_id: str) -> str:
        """
        Get user display name from Slack user ID.
        
        For now, return a placeholder. In production, this would
        call the Slack Users API to resolve the name.
        """
        # Known user mapping (could be expanded with Slack API call)
        known_users = {
            "U01234567": "Peder B. Koppang",
            # Add more as needed
        }
        return known_users.get(user_id, f"User_{user_id}")

    def handle_url_verification(self, challenge: str) -> str:
        """Handle Slack URL verification challenge."""
        logger.info("Handling Slack URL verification challenge")
        return challenge

# Convenience functions for FastAPI integration

def create_slack_webhook_handler() -> SlackTicketWebhookHandler:
    """Create a new Slack webhook handler instance."""
    return SlackTicketWebhookHandler()

async def process_slack_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a Slack webhook payload.
    
    Args:
        payload: Full Slack webhook request body
        
    Returns:
        dict: Processing result
        
    Raises:
        HTTPException: If processing fails critically
    """
    handler = create_slack_webhook_handler()
    
    # Handle URL verification
    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge")
        if not challenge:
            raise HTTPException(status_code=400, detail="Missing challenge parameter")
        return {"challenge": challenge}
    
    # Handle events
    if payload.get("type") == "event_callback":
        result = handler.handle_event(payload)
        
        # If processing took too long, log a warning
        if result.get("processing_time_ms", 0) > 100:
            logger.warning(f"Slow ticket processing: {result['processing_time_ms']:.1f}ms (target: <100ms)")
        
        return result
    
    # Unknown event type
    logger.warning(f"Unknown Slack webhook type: {payload.get('type')}")
    return {"status": "ignored", "reason": "unknown_type"}