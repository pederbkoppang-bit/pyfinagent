"""
Phase 3: Ticket Queue Processor
Processes tickets from the queue and routes them to appropriate agents.

This service:
1. Pulls open tickets in FIFO order by priority
2. Routes by classification (analytical → Q&A, research → Research, operational → MAIN)
3. Spawns agent session per ticket
4. Updates ticket status throughout lifecycle
"""

import asyncio
import logging
import time
import subprocess
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path

from backend.db.tickets_db import (
    get_tickets_db, TicketStatus, TicketClassification, TicketsDB
)

logger = logging.getLogger(__name__)

class TicketQueueProcessor:
    """Processes tickets from the queue and routes to agents."""

    def __init__(self, max_concurrent: int = 1):  # Reduced to 1 to avoid rate limiting spikes
        self.db = get_tickets_db()
        self.max_concurrent = max_concurrent
        self.running = False
        self.active_processors = {}  # ticket_id -> task

    def _increment_retries(self, ticket_id: int):
        """Increment the retry counter for a ticket."""
        import sqlite3
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute(
                    "UPDATE tickets SET retries = COALESCE(retries, 0) + 1 WHERE id = ?",
                    (ticket_id,)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to increment retries for ticket {ticket_id}: {e}")

    def get_agent_for_classification(self, classification) -> str:
        """Map ticket classification to agent type."""
        # Handle both enum and string values
        if hasattr(classification, 'value'):
            classification = classification.value

        agent_map = {
            "operational": "MAIN",
            "analytical": "Q&A",
            "research": "Research"
        }
        return agent_map.get(classification, "MAIN")

    def build_agent_prompt(self, ticket: Dict[str, Any]) -> str:
        """Build the prompt to send to the agent."""
        source_emoji = "📱" if ticket['source'] == 'imessage' else "💬"
        priority_emoji = {
            'P0': '🚨',
            'P1': '⚡',
            'P2': '📋',
            'P3': '🕐'
        }.get(ticket['priority'], '📋')

        prompt = f"""{source_emoji} Ticket #{ticket['ticket_number']} {priority_emoji} {ticket['priority']}

From: {ticket['sender_name'] or ticket['sender_id']}
Source: {ticket['source'].title()}
Classification: {ticket['classification'].title()}

Message:
{ticket['message_text']}

---

Please provide a helpful response. This will be sent back to the user via {ticket['source']}."""

        return prompt

    def spawn_agent_session(self, ticket: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        """
        Spawn a real agent session to handle the ticket.

        Args:
            ticket: Ticket data
            agent_type: Agent type (MAIN, Q&A, Research)

        Returns:
            dict: Contains 'success', 'response', 'error', 'session_id'
        """
        try:
            prompt = self.build_agent_prompt(ticket)
            start_time = time.time()

            # Map agent type to actual agent
            agent_map = {
                "MAIN": "main",
                "Q&A": "q-and-a",
                "Research": "research"
            }
            agent_id = agent_map.get(agent_type, "main")

            # Spawn real agent session via openclaw CLI
            response = self._spawn_real_agent(
                agent_id=agent_id,
                task=prompt,
                ticket_id=ticket['id'],
                ticket_number=ticket['ticket_number']
            )

            elapsed = time.time() - start_time

            return {
                "success": True,
                "response": response,
                "agent_type": agent_type,
                "processing_time_seconds": elapsed
            }

        except Exception as e:
            logger.error(f"Failed to spawn agent for ticket {ticket['id']}: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent_type": agent_type
            }

    def _spawn_real_agent(self, agent_id: str, task: str, ticket_id: int, ticket_number: str) -> str:
        """
        Spawn a real agent by calling Claude (Anthropic) directly.

        Uses the Anthropic SDK to invoke Claude agents,
        matching the rest of the agentic architecture.

        Args:
            agent_id: Agent identifier (main, q-and-a, research)
            task: Task prompt for the agent
            ticket_id: Database ticket ID
            ticket_number: Human-readable ticket number

        Returns:
            str: Agent response text

        Raises:
            Exception: If LLM call fails
        """
        import anthropic
        from backend.config.settings import get_settings

        logger.debug(f"Invoking agent {agent_id} for ticket #{ticket_number}: {task[:100]}")

        try:
            settings = get_settings()

            # Select model based on agent type (per Peder's explicit spec)
            agent_model_map = {
                "main": "claude-opus-4-6",       # Opus 4-6 for main agent (complex reasoning)
                "q-and-a": "claude-opus-4-6",   # Opus 4-6 for Q&A agent (accuracy required)
                "research": "claude-sonnet-4-6" # Sonnet 4-6 for research (cost efficient)
            }
            
            model_name = agent_model_map.get(agent_id, "claude-opus-4-6")

            # Create Anthropic client
            # Check for API key from environment or settings
            import os
            api_key = settings.anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in settings or environment. Please configure ANTHROPIC_API_KEY.")

            client = anthropic.Anthropic(api_key=api_key)

            # Build system prompt based on agent type
            system_prompts = {
                "main": "You are Ford, the primary agent for operational tasks. "
                        "Respond helpfully and directly to user requests. Be concise.",
                "q-and-a": "You are Q&A agent. Answer questions about the pyfinAgent system "
                          "clearly and concisely. Provide factual, helpful responses.",
                "research": "You are Research agent. Conduct thorough research and provide "
                           "evidence-backed insights. Include sources and methodology."
            }

            system = system_prompts.get(agent_id, "You are a helpful assistant.")

            # Call Claude directly via Anthropic SDK
            response = client.messages.create(
                model=model_name,
                max_tokens=1000,
                system=system,
                messages=[
                    {"role": "user", "content": task}
                ]
            )

            # Extract text from response
            response_text = response.content[0].text if response.content else "No response"

            logger.info(f"✅ Agent {agent_id} completed for ticket #{ticket_number}: "
                       f"{response_text[:80]}...")
            return response_text

        except Exception as e:
            logger.error(f"Error invoking agent {agent_id}: {e}")
            raise Exception(f"Agent {agent_id} failed: {str(e)[:500]}")

    def _simulate_agent_response(self, ticket: Dict[str, Any], agent_type: str, prompt: str) -> str:
        """
        Simulate agent response for testing.
        In production, this would be replaced with actual agent spawning.
        """
        # Simulate processing time
        time.sleep(0.1)

        message_lower = ticket['message_text'].lower()
        classification = ticket['classification']

        if classification == 'operational':
            if 'status' in message_lower or 'update' in message_lower:
                return "✅ Services Status:\n• Backend (port 8000): Running\n• Frontend (port 3000): Running\n• Queue processor: Active\n• All systems operational"
            elif 'portfolio' in message_lower:
                return "📊 Portfolio Status:\n• Paper trading active\n• 3 positions open\n• Total value: $98,450 (+2.1%)\n• Daily P&L: +$450"
            else:
                return f"🔧 Operational response from {agent_type} agent for ticket #{ticket['ticket_number']}. Request processed successfully."

        elif classification == 'analytical':
            if 'sharpe' in message_lower:
                return "📈 Sharpe Ratio Analysis:\nRecent drop due to increased volatility in tech positions. Portfolio Sharpe fell from 1.42 to 1.28 over the past week. Key factors: AAPL earnings miss (-3.2%), NVDA sector rotation (-5.1%). Recommend rebalancing toward defensive sectors."
            elif 'why' in message_lower:
                return f"🧠 Analysis from {agent_type}: Based on recent market data and our models, the primary drivers appear to be sector rotation and earnings sentiment. Detailed analysis shows correlation with broader market volatility patterns."
            else:
                return f"📊 Analytical response from {agent_type} agent. Performed quantitative analysis and identified key factors for your question."

        elif classification == 'research':
            if 'paper' in message_lower or 'research' in message_lower:
                return "🔬 Research Update:\nFound 3 relevant papers on portfolio optimization published in the last 6 months:\n1. 'Deep Learning for Risk-Adjusted Returns' (Nature Finance, 2025)\n2. 'Multi-Asset Portfolio Theory Extensions' (QJE, 2025)\n3. 'Behavioral Factors in Systematic Trading' (RFS, 2025)\n\nSummary and implementation recommendations attached."
            else:
                return f"📚 Research response from {agent_type} agent. Conducted literature review and experimental analysis for your inquiry."

        return f"🤖 Response from {agent_type} agent for ticket #{ticket['ticket_number']}: {ticket['message_text'][:50]}..."

    async def process_single_ticket(self, ticket: Dict[str, Any]) -> bool:
        """
        Process a single ticket through the full lifecycle.

        Returns:
            bool: True if successfully processed, False otherwise
        """
        ticket_id = ticket['id']
        ticket_number = ticket['ticket_number']

        # BUG 2 FIX: Skip tickets that have exceeded max retries
        max_retries = 3
        retries = ticket.get('retries', 0) or 0
        if retries >= max_retries:
            logger.warning(f"Ticket #{ticket_number} exceeded max retries ({retries}/{max_retries}), closing")
            self.db.update_ticket_status(
                ticket_id,
                TicketStatus.CLOSED,
                error_message=f"Max retries ({max_retries}) exceeded"
            )
            
            # CLOSURE PROTOCOL: Mandatory follow-up for closed tickets
            logger.info(f"🔴 CLOSURE TRIGGER: Ticket #{ticket_number} closed after {retries} retries. Follow-up action pending.")
            # TODO: Implement follow-up trigger (notify user, escalate, archive, etc.)
            
            return False

        try:
            logger.info(f"Processing ticket #{ticket_number} [{ticket['classification']}/{ticket['priority']}] (retry {retries}/{max_retries})")

            # Step 1: Mark as assigned + log dispatch
            agent_type = self.get_agent_for_classification(ticket['classification'])

            self.db.update_ticket_status(
                ticket_id,
                TicketStatus.ASSIGNED,
                assigned_agent=agent_type
            )
            
            # DISPATCH STATE TRACKING: Log the moment ticket is assigned to agent
            dispatch_timestamp = datetime.now(timezone.utc).isoformat()
            logger.info(f"📤 DISPATCH: Ticket #{ticket_number} assigned to {agent_type} agent at {dispatch_timestamp}")

            # Step 2: Mark as in progress
            self.db.update_ticket_status(ticket_id, TicketStatus.IN_PROGRESS)

            # Step 3: Process with agent (run in thread to avoid blocking)
            # Add exponential backoff to prevent rate limiting
            retry_count = ticket.get('retries', 0)
            if retry_count > 0:
                # Increased base to 10s (10s, 20s, 40s, max 120s) — Anthropic needs more spacing
                wait_time = min(10 * (2 ** (retry_count - 1)), 120)
                logger.info(f"Waiting {wait_time}s before retry (attempt {retry_count + 1})")
                await asyncio.sleep(wait_time)

            loop = asyncio.get_event_loop()
            agent_result = await loop.run_in_executor(
                None,
                self.spawn_agent_session,
                ticket,
                agent_type
            )

            # Step 4: Handle response
            if agent_result['success']:
                # Mark as resolved with response
                response_text = agent_result['response']
                logger.debug(f"Updating ticket {ticket_id} with response: {response_text[:50]}...")

                success = self.db.update_ticket_status(
                    ticket_id,
                    TicketStatus.RESOLVED,
                    response_text=response_text
                )

                if not success:
                    logger.error(f"Failed to update ticket {ticket_id} status")
                    return False

                logger.info(f"✅ Ticket #{ticket_number} resolved by {agent_type}: "
                           f"{response_text[:80]}...")

                # Step 5: Deliver response to user via Slack or iMessage
                try:
                    from backend.services.response_delivery import ResponseDeliveryService
                    delivery = ResponseDeliveryService()

                    delivery_success = await delivery.deliver_ticket_response(
                        ticket_id=ticket_id
                    )

                    if delivery_success:
                        logger.info(f"Response delivered for ticket #{ticket_number} to {ticket.get('source', 'slack')}")
                    else:
                        logger.warning(f"Failed to deliver response for ticket #{ticket_number}")

                except Exception as e:
                    logger.error(f"Failed to deliver response for ticket #{ticket_number}: {e}")
                    # Don't fail the ticket -- response is stored, just not delivered yet

                return True
            else:
                # BUG 2 FIX: Increment retries and set back to OPEN for retry
                self._increment_retries(ticket_id)
                self.db.update_ticket_status(
                    ticket_id,
                    TicketStatus.OPEN,
                    error_message=agent_result['error']
                )

                logger.error(f"Ticket #{ticket_number} failed (retry {retries + 1}/{max_retries}): {agent_result['error']}")
                return False

        except Exception as e:
            logger.error(f"Critical error processing ticket #{ticket_number}: {e}")

            # BUG 2 FIX: Increment retries and set back to OPEN for retry
            self._increment_retries(ticket_id)
            self.db.update_ticket_status(
                ticket_id,
                TicketStatus.OPEN,
                error_message=str(e)
            )
            return False
        finally:
            # Remove from active processors
            self.active_processors.pop(ticket_id, None)

    async def process_queue_batch(self, batch_size: int = 10) -> int:
        """
        Process a batch of tickets from the queue.

        Returns:
            int: Number of tickets processed successfully
        """
        # Get open tickets in priority order
        open_tickets = self.db.get_open_tickets(limit=batch_size)

        if not open_tickets:
            return 0

        logger.info(f"📥 Processing batch of {len(open_tickets)} tickets")

        # Process tickets concurrently (respecting max_concurrent)
        tasks = []
        processed_count = 0

        for ticket in open_tickets[:self.max_concurrent]:
            ticket_id = ticket['id']

            # Skip if already being processed
            if ticket_id in self.active_processors:
                continue

            # Create processing task
            task = asyncio.create_task(self.process_single_ticket(ticket))
            self.active_processors[ticket_id] = task
            tasks.append(task)

        # Wait for all tasks to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_count = sum(1 for r in results if r is True)

            logger.info(f"✅ Batch complete: {processed_count}/{len(tasks)} succeeded")

        return processed_count

    async def start_processing_loop(self, batch_interval: float = 5.0):
        """
        Start the main processing loop.

        Args:
            batch_interval: Seconds between batch processing cycles
        """
        self.running = True
        logger.info("🚀 Ticket queue processor started")

        while self.running:
            try:
                # Process a batch
                processed = await self.process_queue_batch()

                # Log stats periodically
                stats = self.db.get_ticket_stats()
                logger.debug(f"📊 Queue stats: {stats['status_counts']}")

                # Wait before next batch (shorter if we processed tickets)
                wait_time = batch_interval if processed == 0 else batch_interval / 2
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                await asyncio.sleep(batch_interval)

    def stop(self):
        """Stop the processing loop."""
        self.running = False
        logger.info("🛑 Ticket queue processor stopping")

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "running": self.running,
            "active_processors": len(self.active_processors),
            "max_concurrent": self.max_concurrent,
            **self.db.get_ticket_stats()
        }

# Global processor instance
_processor: Optional[TicketQueueProcessor] = None

def get_queue_processor() -> TicketQueueProcessor:
    """Get the global queue processor instance."""
    global _processor
    if _processor is None:
        _processor = TicketQueueProcessor()
    return _processor

async def start_queue_processor(batch_interval: float = 5.0):
    """Start the global queue processor."""
    processor = get_queue_processor()
    await processor.start_processing_loop(batch_interval)

def stop_queue_processor():
    """Stop the global queue processor."""
    processor = get_queue_processor()
    processor.stop()