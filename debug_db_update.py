#!/usr/bin/env python3
"""
Debug script for database update issue.
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

def debug_db_update():
    """Debug the database update."""
    print("=== Debug DB Update ===")
    
    # Use a test database
    test_db_path = "/tmp/debug_db_update.db"
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
    
    # Test update to ASSIGNED
    print("\n--- Test ASSIGNED ---")
    success = db.update_ticket_status(ticket_id, TicketStatus.ASSIGNED, assigned_agent="MAIN")
    print(f"ASSIGNED update success: {success}")
    
    ticket = db.get_ticket(ticket_id)
    print(f"Status: {ticket['status']}, Agent: {ticket['assigned_agent']}, Assigned at: {ticket['assigned_at']}")
    
    # Test update to IN_PROGRESS
    print("\n--- Test IN_PROGRESS ---")
    success = db.update_ticket_status(ticket_id, TicketStatus.IN_PROGRESS)
    print(f"IN_PROGRESS update success: {success}")
    
    ticket = db.get_ticket(ticket_id)
    print(f"Status: {ticket['status']}, In progress at: {ticket['in_progress_at']}")
    
    # Test update to RESOLVED
    print("\n--- Test RESOLVED ---")
    test_response = "This is a test response from the agent"
    success = db.update_ticket_status(ticket_id, TicketStatus.RESOLVED, response_text=test_response)
    print(f"RESOLVED update success: {success}")
    
    ticket = db.get_ticket(ticket_id)
    print(f"Status: {ticket['status']}")
    print(f"Response text: {ticket['response_text']}")
    print(f"Resolved at: {ticket['resolved_at']}")
    
    # Test direct SQL to see what's in the database
    print("\n--- Direct SQL Check ---")
    import sqlite3
    with sqlite3.connect(test_db_path) as conn:
        cursor = conn.execute("SELECT status, response_text, resolved_at FROM tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        print(f"Direct SQL result: status={row[0]}, response_text={row[1]}, resolved_at={row[2]}")

if __name__ == "__main__":
    debug_db_update()