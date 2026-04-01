#!/usr/bin/env python3
"""
Debug script for ingestion issue.
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from db.tickets_db import TicketsDB, init_tickets_db
from services.ticket_ingestion import MessageIngestionService

def debug_ingestion():
    """Debug the ingestion process."""
    print("=== Debug Ingestion ===")
    
    # Use a test database
    test_db_path = "/tmp/debug_ingestion.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = init_tickets_db(test_db_path)
    print(f"✅ Initialized test database: {test_db_path}")
    
    # Test direct DB creation
    print("\n--- Direct DB Test ---")
    from db.tickets_db import TicketSource, TicketPriority, TicketClassification
    
    ticket_id = db.create_ticket(
        source=TicketSource.IMESSAGE,
        sender_id="+4794810537",
        message_text="Test message",
        priority=TicketPriority.P2,
        classification=TicketClassification.ANALYTICAL
    )
    print(f"Direct DB ticket_id: {ticket_id}")
    
    ticket = db.get_ticket(ticket_id)
    print(f"Retrieved ticket: {ticket}")
    
    # Test ingestion service
    print("\n--- Ingestion Service Test ---")
    ingestion = MessageIngestionService()
    
    try:
        ticket_id2 = ingestion.ingest_imessage(
            sender_id="+4794810537",
            sender_name="Peder",
            message_text="Test message 2",
            message_id="msg_12345"
        )
        print(f"Ingestion service ticket_id: {ticket_id2}")
        
        ticket2 = db.get_ticket(ticket_id2)
        print(f"Retrieved ticket2: {ticket2}")
        
    except Exception as e:
        print(f"Error in ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ingestion()