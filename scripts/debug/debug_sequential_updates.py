#!/usr/bin/env python3
"""
Debug script for sequential database updates.
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from db.tickets_db import (
    TicketsDB, TicketStatus, TicketPriority, TicketSource, 
    TicketClassification, init_tickets_db
)

def debug_sequential_updates():
    """Debug sequential database updates like in the processor."""
    print("=== Debug Sequential Updates ===")
    
    # Use a test database
    test_db_path = "/tmp/debug_sequential.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = init_tickets_db(test_db_path)
    print(f"✅ Initialized test database: {test_db_path}")
    
    # Create a test ticket
    ticket_id = db.create_ticket(
        source=TicketSource.IMESSAGE,
        sender_id="+4794810537",
        sender_name="Peder",
        message_text="Test message",
        priority=TicketPriority.P0,
        classification=TicketClassification.OPERATIONAL
    )
    
    print(f"Created ticket ID: {ticket_id}")
    
    # Show initial state
    ticket = db.get_ticket(ticket_id)
    print(f"Initial: status={ticket['status']}, agent={ticket['assigned_agent']}, "
          f"assigned_at={ticket['assigned_at']}, in_progress_at={ticket['in_progress_at']}, "
          f"resolved_at={ticket['resolved_at']}, response_text={ticket['response_text']}")
    
    # Step 1: ASSIGNED (like in processor)
    print("\n--- Step 1: Mark as ASSIGNED ---")
    success = db.update_ticket_status(ticket_id, TicketStatus.ASSIGNED, assigned_agent="MAIN")
    print(f"Success: {success}")
    
    ticket = db.get_ticket(ticket_id)
    print(f"After ASSIGNED: status={ticket['status']}, agent={ticket['assigned_agent']}, "
          f"assigned_at={ticket['assigned_at']}, in_progress_at={ticket['in_progress_at']}, "
          f"resolved_at={ticket['resolved_at']}, response_text={ticket['response_text']}")
    
    # Step 2: IN_PROGRESS (like in processor)
    print("\n--- Step 2: Mark as IN_PROGRESS ---")
    success = db.update_ticket_status(ticket_id, TicketStatus.IN_PROGRESS)
    print(f"Success: {success}")
    
    ticket = db.get_ticket(ticket_id)
    print(f"After IN_PROGRESS: status={ticket['status']}, agent={ticket['assigned_agent']}, "
          f"assigned_at={ticket['assigned_at']}, in_progress_at={ticket['in_progress_at']}, "
          f"resolved_at={ticket['resolved_at']}, response_text={ticket['response_text']}")
    
    # Step 3: RESOLVED (like in processor)
    print("\n--- Step 3: Mark as RESOLVED ---")
    test_response = "This is a test response from the agent"
    success = db.update_ticket_status(ticket_id, TicketStatus.RESOLVED, response_text=test_response)
    print(f"Success: {success}")
    
    ticket = db.get_ticket(ticket_id)
    print(f"After RESOLVED: status={ticket['status']}, agent={ticket['assigned_agent']}, "
          f"assigned_at={ticket['assigned_at']}, in_progress_at={ticket['in_progress_at']}, "
          f"resolved_at={ticket['resolved_at']}, response_text={ticket['response_text'][:50] if ticket['response_text'] else 'None'}...")

if __name__ == "__main__":
    debug_sequential_updates()