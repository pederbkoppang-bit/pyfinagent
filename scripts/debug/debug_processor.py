#!/usr/bin/env python3
"""
Debug script for queue processor issue.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from db.tickets_db import (
    TicketsDB, TicketStatus, TicketPriority, TicketSource, 
    TicketClassification, init_tickets_db
)
from services.ticket_queue_processor import TicketQueueProcessor

async def debug_processor():
    """Debug the processor."""
    print("=== Debug Processor ===")
    
    # Use a test database
    test_db_path = "/tmp/debug_processor.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = init_tickets_db(test_db_path)
    print(f"✅ Initialized test database: {test_db_path}")
    
    # Create a test ticket
    ticket_id = db.create_ticket(
        source=TicketSource.IMESSAGE,
        sender_id="+4794810537",
        sender_name="Peder",
        message_text="URGENT: System down!",
        priority=TicketPriority.P0,
        classification=TicketClassification.OPERATIONAL
    )
    
    ticket = db.get_ticket(ticket_id)
    print(f"Created ticket: {ticket}")
    
    # Test processor
    processor = TicketQueueProcessor()
    processor.db = db
    
    # Test agent routing
    agent = processor.get_agent_for_classification("operational")
    print(f"Agent for operational: {agent}")
    
    # Test prompt building
    prompt = processor.build_agent_prompt(ticket)
    print(f"Prompt:\n{prompt}")
    
    # Test agent response
    response_result = processor.spawn_agent_session(ticket, agent)
    print(f"Agent response: {response_result}")
    
    # Test full processing
    success = await processor.process_single_ticket(ticket)
    print(f"Processing success: {success}")
    
    # Check final ticket state
    final_ticket = db.get_ticket(ticket_id)
    print(f"Final ticket: {final_ticket}")

if __name__ == "__main__":
    asyncio.run(debug_processor())