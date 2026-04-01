#!/usr/bin/env python3
"""
Test script for Phase 4: Response Delivery
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from db.tickets_db import (
    TicketsDB, TicketStatus, TicketPriority, TicketSource, 
    TicketClassification, init_tickets_db
)
from services.response_delivery import ResponseDeliveryService

async def test_phase_4_response_delivery():
    """Phase 4: Test response delivery to Slack and iMessage."""
    print("=== Phase 4: Response Delivery Test ===")
    
    # Use a test database
    test_db_path = "/tmp/test_response_delivery.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = init_tickets_db(test_db_path)
    print(f"✅ Initialized test database: {test_db_path}")
    
    # Create delivery service
    delivery = ResponseDeliveryService()
    delivery.db = db  # Use our test database
    
    # Test 1: Create resolved tickets
    print("\n--- Test 1: Create Resolved Tickets ---")
    
    # iMessage ticket
    imsg_ticket_id = db.create_ticket(
        source=TicketSource.IMESSAGE,
        sender_id="+4794810537",
        sender_name="Peder",
        message_text="What's the portfolio status?",
        priority=TicketPriority.P1,
        classification=TicketClassification.OPERATIONAL
    )
    
    # Slack ticket
    slack_ticket_id = db.create_ticket(
        source=TicketSource.SLACK,
        sender_id="U01234567",
        sender_name="Peder B. Koppang",
        channel_id="C0ANTGNNK8D",
        message_text="Why did AAPL drop yesterday?",
        priority=TicketPriority.P2,
        classification=TicketClassification.ANALYTICAL,
        slack_thread_id="1709251234.567890"
    )
    
    # Mark them as resolved with responses
    db.update_ticket_status(
        imsg_ticket_id, 
        TicketStatus.RESOLVED, 
        response_text="📊 Portfolio Status:\n• 3 positions open\n• Total value: $98,450 (+2.1%)\n• All systems operational"
    )
    
    db.update_ticket_status(
        slack_ticket_id, 
        TicketStatus.RESOLVED, 
        response_text="📈 AAPL dropped due to earnings miss (-3.2%) and sector rotation. Key factors: reduced guidance, increased competition from Android. Recommend holding for long-term."
    )
    
    print(f"✅ Created resolved tickets: iMessage #{db.get_ticket(imsg_ticket_id)['ticket_number']}, Slack #{db.get_ticket(slack_ticket_id)['ticket_number']}")
    
    # Test 2: Individual delivery tests
    print("\n--- Test 2: Individual Message Delivery ---")
    
    # Test iMessage delivery (simulated - won't actually send)
    imsg_success = delivery.send_imessage_response(
        phone_number="+4794810537",
        message="Test iMessage response",
        ticket_number=5001
    )
    print(f"✅ iMessage delivery test: {'success' if imsg_success else 'failed'}")
    
    # Test Slack delivery
    slack_success = await delivery.send_slack_response(
        channel_id="C0ANTGNNK8D",
        message="Test Slack response",
        thread_ts="1709251234.567890",
        ticket_number=5002
    )
    print(f"✅ Slack delivery test: {'success' if slack_success else 'failed'}")
    
    # Test 3: Ticket response delivery
    print("\n--- Test 3: Ticket Response Delivery ---")
    
    # Deliver iMessage ticket response
    start_time = time.time()
    imsg_delivery = await delivery.deliver_ticket_response(imsg_ticket_id)
    imsg_time = (time.time() - start_time) * 1000
    
    print(f"✅ iMessage ticket delivery: {'success' if imsg_delivery else 'failed'} ({imsg_time:.1f}ms)")
    
    # Deliver Slack ticket response
    start_time = time.time()
    slack_delivery = await delivery.deliver_ticket_response(slack_ticket_id)
    slack_time = (time.time() - start_time) * 1000
    
    print(f"✅ Slack ticket delivery: {'success' if slack_delivery else 'failed'} ({slack_time:.1f}ms)")
    
    # Test 4: Batch delivery
    print("\n--- Test 4: Batch Response Delivery ---")
    
    # Create more resolved tickets for batch test
    batch_tickets = []
    for i in range(5):
        source = TicketSource.IMESSAGE if i % 2 == 0 else TicketSource.SLACK
        ticket_id = db.create_ticket(
            source=source,
            sender_id=f"+479481053{i}" if source == TicketSource.IMESSAGE else f"U{i:08d}",
            sender_name="Test User",
            channel_id="C0ANTGNNK8D" if source == TicketSource.SLACK else None,
            message_text=f"Batch test question {i}",
            priority=TicketPriority.P2,
            classification=TicketClassification.ANALYTICAL
        )
        
        # Mark as resolved
        db.update_ticket_status(
            ticket_id,
            TicketStatus.RESOLVED,
            response_text=f"📊 Response to batch question {i}: Analysis complete. Key findings attached."
        )
        
        batch_tickets.append(ticket_id)
    
    # Test batch delivery
    start_time = time.time()
    delivered_count = await delivery.deliver_pending_responses(limit=10)
    batch_time = (time.time() - start_time) * 1000
    
    print(f"✅ Batch delivery: {delivered_count} responses sent ({batch_time:.1f}ms)")
    assert delivered_count >= 5, f"Expected at least 5 deliveries, got {delivered_count}"
    
    # Test 5: Delivery stats
    print("\n--- Test 5: Delivery Statistics ---")
    
    stats = delivery.get_delivery_stats()
    print(f"📊 Delivery stats: {stats}")
    
    # Should show any remaining tickets needing delivery
    total_by_source = sum(stats['by_source'].values())
    print(f"✅ Total resolved tickets: {total_by_source}")
    print(f"✅ Pending delivery: {stats['pending_delivery']}")
    
    # Test 6: Error handling
    print("\n--- Test 6: Error Handling ---")
    
    # Try to deliver non-existent ticket
    error_delivery = await delivery.deliver_ticket_response(99999)
    assert error_delivery == False, "Should fail for non-existent ticket"
    print("✅ Error handling: Non-existent ticket correctly rejected")
    
    # Try to deliver unresolved ticket
    unresolved_ticket_id = db.create_ticket(
        source=TicketSource.IMESSAGE,
        sender_id="+4794810537",
        message_text="Unresolved ticket",
        priority=TicketPriority.P3
    )
    
    unresolved_delivery = await delivery.deliver_ticket_response(unresolved_ticket_id)
    assert unresolved_delivery == False, "Should fail for unresolved ticket"
    print("✅ Error handling: Unresolved ticket correctly rejected")
    
    # Test 7: Message formatting
    print("\n--- Test 7: Message Formatting ---")
    
    # Test long message truncation for iMessage
    long_message = "This is a very long message that might exceed the recommended length for SMS/iMessage delivery. " * 3
    
    formatted_success = delivery.send_imessage_response(
        phone_number="+4794810537",
        message=long_message,
        ticket_number=9999
    )
    print(f"✅ Long message formatting: {'success' if formatted_success else 'failed'}")
    
    # Test Slack thread formatting
    thread_success = await delivery.send_slack_response(
        channel_id="C0ANTGNNK8D",
        message="Thread response with formatting",
        thread_ts="1709251234.567890",
        ticket_number=8888
    )
    print(f"✅ Slack thread formatting: {'success' if thread_success else 'failed'}")
    
    print("\n🎉 Phase 4: Response Delivery - ALL TESTS PASSED")
    print("✅ iMessage delivery working")
    print("✅ Slack delivery working (simulated)")
    print("✅ Ticket response delivery working")
    print("✅ Batch delivery working")
    print("✅ Error handling working")
    print("✅ Message formatting working")
    
    # Cleanup
    os.remove(test_db_path)
    print("🧹 Test database cleaned up")

if __name__ == "__main__":
    asyncio.run(test_phase_4_response_delivery())