#!/usr/bin/env python3
"""
Test script for tickets database schema - Phase 1 validation.
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from db.tickets_db import (
    TicketsDB, TicketStatus, TicketPriority, TicketSource, 
    TicketClassification, get_tickets_db
)

def test_phase_1_schema():
    """Phase 1: Test database schema creation and basic operations."""
    print("=== Phase 1: Database Schema Test ===")
    
    # Use a test database
    test_db_path = "/tmp/test_tickets.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = TicketsDB(test_db_path)
    print(f"✅ Created database: {test_db_path}")
    
    # Test 1: Create a ticket
    ticket_id = db.create_ticket(
        source=TicketSource.SLACK,
        sender_id="U01234567",
        sender_name="Peder B. Koppang",
        channel_id="C0ANTGNNK8D",
        message_text="Why did the Sharpe ratio drop this week?",
        priority=TicketPriority.P1,
        classification=TicketClassification.ANALYTICAL,
        slack_envelope_id="env_12345",
        slack_event_ts="1709251234.567890"
    )
    print(f"✅ Created ticket ID: {ticket_id}")
    
    # Test 2: Retrieve the ticket
    ticket = db.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket['source'] == 'slack'
    assert ticket['priority'] == 'P1'
    assert ticket['status'] == 'OPEN'
    assert ticket['ticket_number'] == 5001
    print(f"✅ Retrieved ticket #{ticket['ticket_number']}: {ticket['message_text'][:30]}...")
    
    # Test 3: Test deduplication
    duplicate_detected = db.is_duplicate_envelope("env_12345")
    assert duplicate_detected == True
    print("✅ Deduplication detection works")
    
    # Test 4: Update ticket status
    success = db.update_ticket_status(
        ticket_id, 
        TicketStatus.ASSIGNED, 
        assigned_agent="Q&A"
    )
    assert success == True
    
    ticket = db.get_ticket(ticket_id)
    assert ticket['status'] == 'ASSIGNED'
    assert ticket['assigned_agent'] == 'Q&A'
    assert ticket['assigned_at'] is not None
    print("✅ Status update works")
    
    # Test 5: Create iMessage ticket
    imsg_id = db.create_ticket(
        source=TicketSource.IMESSAGE,
        sender_id="+4794810537",
        sender_name="Peder",
        message_text="Status update needed",
        priority=TicketPriority.P2,
        classification=TicketClassification.OPERATIONAL
    )
    print(f"✅ Created iMessage ticket ID: {imsg_id}")
    
    # Test 6: Get open tickets (FIFO by priority)
    open_tickets = db.get_open_tickets()
    assert len(open_tickets) == 1  # Only iMessage ticket should be open
    assert open_tickets[0]['id'] == imsg_id
    assert open_tickets[0]['ticket_number'] == 5002
    print("✅ FIFO queue retrieval works")
    
    # Test 7: Acknowledge ticket
    ack_success = db.acknowledge_ticket(imsg_id)
    assert ack_success == True
    
    ticket = db.get_ticket(imsg_id)
    assert ticket['acknowledged_at'] is not None
    print("✅ Acknowledgment works")
    
    # Test 8: Get stats
    stats = db.get_ticket_stats()
    assert stats['total_tickets'] == 2
    assert stats['status_counts']['OPEN'] == 1
    assert stats['status_counts']['ASSIGNED'] == 1
    print(f"✅ Stats: {stats}")
    
    # Test 9: Resolve ticket
    resolve_success = db.update_ticket_status(
        imsg_id,
        TicketStatus.RESOLVED,
        response_text="Services are running. Backend: ✅ Frontend: ✅"
    )
    assert resolve_success == True
    
    ticket = db.get_ticket(imsg_id)
    assert ticket['status'] == 'RESOLVED'
    assert ticket['response_text'] is not None
    assert ticket['resolved_at'] is not None
    print("✅ Ticket resolution works")
    
    print("\n🎉 Phase 1: Database Schema - ALL TESTS PASSED")
    print(f"📊 Final stats: {db.get_ticket_stats()}")
    
    # Cleanup
    os.remove(test_db_path)
    print("🧹 Test database cleaned up")

if __name__ == "__main__":
    test_phase_1_schema()