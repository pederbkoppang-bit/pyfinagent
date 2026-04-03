"""
Phase 2: Message Ingestion Service
Hooks into Slack/iMessage to create tickets for reliable message processing.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from backend.db.tickets_db import (
    get_tickets_db, TicketSource, TicketPriority, TicketClassification,
    TicketStatus, TicketsDB
)

logger = logging.getLogger(__name__)

class MessageIngestionService:
    """Service that converts incoming messages into tickets."""
    
    def __init__(self):
        self.db = get_tickets_db()
        
    def classify_message(self, message_text: str) -> TicketClassification:
        """
        Classify message to route to appropriate agent.
        
        Returns: TicketClassification (operational, analytical, research)
        """
        analytical_keywords = [
            "why", "should", "explain", "analyze", "trade-off", "decision",
            "regression", "sharpe", "recommendation", "recommend", "suggest", "improve",
            "compare", "better", "worse", "review", "feedback", "thoughts"
        ]
        
        research_keywords = [
            "research", "paper", "literature", "evidence",
            "experiment", "hypothesis", "theory", "mechanism", "solution",
            "study", "baseline", "benchmark", "findings",
            "investigate", "explore", "discover"
        ]
        
        msg_lower = message_text.lower()
        
        # Check analytical first (more general)
        if any(kw in msg_lower for kw in analytical_keywords):
            return TicketClassification.ANALYTICAL
        
        # Check research (more specific)
        if any(kw in msg_lower for kw in research_keywords):
            return TicketClassification.RESEARCH
        
        # Default to operational
        return TicketClassification.OPERATIONAL

    def determine_priority(self, message_text: str, sender_id: str) -> TicketPriority:
        """
        Determine ticket priority based on message content and sender.
        
        Returns: TicketPriority
        """
        # High priority keywords
        urgent_keywords = [
            "urgent", "critical", "error", "broken", "down", "failed",
            "emergency", "immediately", "asap"
        ]
        
        # Low priority indicators
        low_keywords = [
            "when you have time", "no rush", "fyi", "eventually"
        ]
        
        msg_lower = message_text.lower()
        
        # Check for urgent indicators
        if any(kw in msg_lower for kw in urgent_keywords):
            return TicketPriority.P0  # Critical
        
        # Check for low priority indicators
        if any(kw in msg_lower for kw in low_keywords):
            return TicketPriority.P3  # Low
        
        # Default priorities based on classification
        classification = self.classify_message(message_text)
        if classification == TicketClassification.OPERATIONAL:
            return TicketPriority.P1  # Urgent for operations
        else:
            return TicketPriority.P2  # Standard for analysis/research

    def ingest_slack_message(
        self,
        event: Dict[str, Any],
        sender_id: str,
        sender_name: str = None,
        channel_id: str = None
    ) -> Optional[int]:
        """
        Ingest a Slack message and create a ticket.
        
        Args:
            event: Slack event data
            sender_id: Slack user ID
            sender_name: Display name of sender
            channel_id: Slack channel ID
            
        Returns:
            int: Ticket ID if created, None if duplicate
            
        Raises:
            Exception: If ticket creation fails
        """
        message_text = event.get("text", "")
        envelope_id = event.get("envelope_id")
        event_ts = event.get("ts")
        thread_ts = event.get("thread_ts")
        
        # Check for deduplication first
        if envelope_id and self.db.is_duplicate_envelope(envelope_id):
            logger.info(f"Duplicate Slack event {envelope_id}, skipping")
            # Mark as duplicate
            self.db.mark_duplicate(envelope_id)
            return None
        
        # Classify and prioritize
        classification = self.classify_message(message_text)
        priority = self.determine_priority(message_text, sender_id)
        
        # Create ticket
        try:
            ticket_id = self.db.create_ticket(
                source=TicketSource.SLACK,
                sender_id=sender_id,
                sender_name=sender_name,
                channel_id=channel_id,
                message_text=message_text,
                priority=priority,
                classification=classification,
                slack_envelope_id=envelope_id,
                slack_event_ts=event_ts,
                slack_thread_id=thread_ts,
                metadata={
                    "original_event": event
                }
            )
            
            logger.info(f"✅ Slack → Ticket #{self.db.get_ticket(ticket_id)['ticket_number']} "
                       f"[{priority.value}/{classification.value}]: {message_text[:50]}")
            
            return ticket_id
            
        except Exception as e:
            logger.error(f"Failed to create ticket for Slack message: {e}")
            raise

    def ingest_imessage(
        self,
        sender_id: str,
        message_text: str,
        sender_name: str = None,
        message_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        Ingest an iMessage and create a ticket.
        
        Args:
            sender_id: Phone number or contact ID
            message_text: Message content
            sender_name: Display name of sender
            message_id: iMessage ID for deduplication
            metadata: Additional message metadata
            
        Returns:
            int: Ticket ID
            
        Raises:
            Exception: If ticket creation fails
        """
        # Classify and prioritize
        classification = self.classify_message(message_text)
        priority = self.determine_priority(message_text, sender_id)
        
        # Create ticket
        try:
            ticket_id = self.db.create_ticket(
                source=TicketSource.IMESSAGE,
                sender_id=sender_id,
                sender_name=sender_name,
                message_text=message_text,
                priority=priority,
                classification=classification,
                metadata={
                    "message_id": message_id,
                    **(metadata or {})
                }
            )
            
            logger.info(f"✅ iMessage → Ticket #{self.db.get_ticket(ticket_id)['ticket_number']} "
                       f"[{priority.value}/{classification.value}]: {message_text[:50]}")
            
            return ticket_id
            
        except Exception as e:
            logger.error(f"Failed to create ticket for iMessage: {e}")
            raise

    def acknowledge_ticket_immediately(self, ticket_id: int) -> Dict[str, str]:
        """
        Generate immediate acknowledgment message for a new ticket.
        
        Returns:
            dict: Contains 'message' and 'agent_type' for the response
        """
        ticket = self.db.get_ticket(ticket_id)
        if not ticket:
            return {"message": "❌ Ticket not found", "agent_type": "unknown"}
        
        # Mark as acknowledged
        self.db.acknowledge_ticket(ticket_id)
        
        # Generate agent assignment
        agent_map = {
            TicketClassification.OPERATIONAL.value: "MAIN (Ford)",
            TicketClassification.ANALYTICAL.value: "Analyst",
            TicketClassification.RESEARCH.value: "Researcher"
        }
        
        agent = agent_map.get(ticket['classification'], "Agent")
        priority_emoji = {
            "P0": "🚨",
            "P1": "⚡",
            "P2": "📋",
            "P3": "🕐"
        }.get(ticket['priority'], "📋")
        
        # Get queue position
        queue_position = self.db.get_ticket_queue_position(ticket_id)
        # Always show queue position so users can see their position
        queue_str = f"\n📍 Queue position: #{queue_position}"
        
        # Add priority reasoning
        priority_reasoning = self._get_priority_reasoning(ticket['message_text'], ticket['priority'])
        
        ack_message = (
            f"{priority_emoji} Got it! Ticket #{ticket['ticket_number']} created, "
            f"assigning to {agent}... (ETA: {self._get_sla_eta(ticket['priority'])}){queue_str}\n"
            f"_Priority: {ticket['priority']} - {priority_reasoning}_"
        )
        
        return {
            "message": ack_message,
            "agent_type": ticket['classification'],
            "ticket_number": ticket['ticket_number'],
            "priority": ticket['priority'],
            "queue_position": queue_position
        }

    def _get_priority_reasoning(self, message_text: str, priority: str) -> str:
        """Get reasoning for why this priority was assigned."""
        msg_lower = message_text.lower()
        
        urgent_keywords = ["urgent", "critical", "error", "broken", "down", "failed", "emergency", "immediately", "asap"]
        low_keywords = ["when you have time", "no rush", "fyi", "eventually"]
        
        if any(kw in msg_lower for kw in urgent_keywords):
            return "Urgent keywords detected"
        elif any(kw in msg_lower for kw in low_keywords):
            return "Low priority indicators present"
        elif priority == "P0":
            return "Critical/operational issue"
        elif priority == "P1":
            return "Urgent - requires quick response"
        elif priority == "P2":
            return "Standard priority"
        else:
            return "Low priority - can wait"

    def _get_sla_eta(self, priority: str) -> str:
        """Get human-readable SLA ETA."""
        eta_map = {
            "P0": "5 minutes",
            "P1": "15 minutes", 
            "P2": "1 hour",
            "P3": "4 hours"
        }
        return eta_map.get(priority, "soon")

# Global service instance
_ingestion_service: Optional[MessageIngestionService] = None

def get_ingestion_service() -> MessageIngestionService:
    """Get the global message ingestion service."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = MessageIngestionService()
    return _ingestion_service