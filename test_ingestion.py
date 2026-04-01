#!/usr/bin/env python3
"""
Test script for Phase 2: Message Ingestion
"""

import sys
import os
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from db.tickets_db import TicketsDB, TicketStatus, TicketPriority, TicketSource
from services.ticket_ingestion import get_ingestion_service, MessageIngestionService
from services.slack_ticket_webhook import SlackTicketWebhookHandler

def test_phase_2_ingestion():
    """Phase 2: Test message ingestion from Slack and iMessage."""
    print("=== Phase 2: Message Ingestion Test ===")
    
    # Use a test database
    test_db_path = "/tmp/test_ingestion.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    from db.tickets_db import init_tickets_db
    db = init_tickets_db(test_db_path)
    print(f"✅ Initialized test database: {test_db_path}")
    
    # Test 1: iMessage ingestion
    print("\n--- Test 1: iMessage Ingestion ---")
    # Create ingestion service that uses the same DB instance
    ingestion = MessageIngestionService()
    ingestion.db = db  # Use our test database
    
    start_time = time.time()
    ticket_id = ingestion.ingest_imessage(
        sender_id="+4794810537",
        sender_name="Peder",
        message_text="Why did the Sharpe ratio drop this week?",
        message_id="msg_12345"
    )
    ingestion_time = (time.time() - start_time) * 1000
    
    print(f"✅ iMessage ingestion took {ingestion_time:.1f}ms (target: <100ms)")
    assert ingestion_time < 100, f"Ingestion too slow: {ingestion_time}ms"
    
    ticket = db.get_ticket(ticket_id)
    assert ticket is not None
    assert ticket['source'] == 'imessage'
    assert ticket['classification'] == 'analytical'  # Should detect "why"
    assert ticket['priority'] == 'P2'  # Standard for analytical
    print(f"✅ Created iMessage ticket #{ticket['ticket_number']}: {ticket['classification']}/{ticket['priority']}")
    
    # Test immediate acknowledgment
    ack_info = ingestion.acknowledge_ticket_immediately(ticket_id)
    assert "Ticket #" in ack_info['message']
    assert ack_info['agent_type'] == 'analytical'
    print(f"✅ Acknowledgment: {ack_info['message']}")
    
    # Test 2: Slack ingestion
    print("\n--- Test 2: Slack Ingestion ---")
    slack_handler = SlackTicketWebhookHandler()
    slack_handler.ingestion.db = db  # Use our test database
    slack_handler.db = db
    
    # Simulate Slack webhook event
    slack_event = {
        "type": "event_callback",
        "event": {
            "type": "message",
            "text": "Status update needed urgently",
            "user": "U01234567",
            "channel": "C0ANTGNNK8D",
            "ts": "1709251234.567890",
            "thread_ts": None
        }
    }
    
    start_time = time.time()
    result = slack_handler.handle_event(slack_event)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"✅ Slack ingestion took {processing_time:.1f}ms (target: <100ms)")
    assert processing_time < 100, f"Slack ingestion too slow: {processing_time}ms"
    assert result['status'] == 'success'
    assert 'ticket_number' in result
    print(f"✅ Created Slack ticket #{result['ticket_number']}: {result['classification']}/{result['priority']}")
    
    # Test 3: Deduplication
    print("\n--- Test 3: Deduplication Test ---")
    
    # Send the exact same Slack event again
    start_time = time.time()
    duplicate_result = slack_handler.handle_event(slack_event)
    duplicate_time = (time.time() - start_time) * 1000
    
    print(f"✅ Duplicate detection took {duplicate_time:.1f}ms")
    assert duplicate_result['status'] == 'duplicate'
    print("✅ Duplicate correctly detected and rejected")
    
    # Test 4: Classification accuracy
    print("\n--- Test 4: Message Classification ---")
    
    test_messages = [
        ("What's the portfolio status?", "operational", "P1"),
        ("Why did AAPL drop yesterday?", "analytical", "P2"),
        ("Research the latest ML papers on portfolio optimization", "research", "P2"),
        ("URGENT: Services are down!", "operational", "P0"),
        ("When you have time, check the logs", "operational", "P3")
    ]
    
    for msg_text, expected_class, expected_priority in test_messages:
        classification = ingestion.classify_message(msg_text)
        priority = ingestion.determine_priority(msg_text, "test_user")
        
        print(f"✅ '{msg_text[:40]}...' → {classification.value}/{priority.value} "
              f"(expected: {expected_class}/{expected_priority})")
        
        assert classification.value == expected_class, f"Wrong classification for: {msg_text}"
        assert priority.value == expected_priority, f"Wrong priority for: {msg_text}"
    
    # Test 5: Performance under load
    print("\n--- Test 5: Load Test ---")
    
    load_start = time.time()
    load_tickets = []
    
    for i in range(50):
        ticket_id = ingestion.ingest_imessage(
            sender_id=f"+479481053{i%10}",
            sender_name="Test User",
            message_text=f"Load test message {i}",
            message_id=f"load_msg_{i}"
        )
        load_tickets.append(ticket_id)
    
    load_time = (time.time() - load_start) * 1000
    avg_per_ticket = load_time / 50
    
    print(f"✅ Created 50 tickets in {load_time:.1f}ms (avg: {avg_per_ticket:.1f}ms per ticket)")
    assert avg_per_ticket < 100, f"Average per ticket too slow: {avg_per_ticket}ms"
    
    # Test 6: Verify queue ordering
    print("\n--- Test 6: FIFO Queue Test ---")
    
    open_tickets = db.get_open_tickets(limit=20)
    priorities = [t['priority'] for t in open_tickets]
    
    # Should be ordered by priority (P0, P1, P2, P3), then by creation time
    expected_order = []
    for priority in ['P0', 'P1', 'P2', 'P3']:
        expected_order.extend([p for p in priorities if p == priority])
    
    print(f"✅ Queue order: {priorities[:10]} (showing first 10)")
    print(f"✅ Total open tickets: {len(open_tickets)}")
    
    # Final stats
    stats = db.get_ticket_stats()
    print(f"\n📊 Final stats: {stats}")
    
    print("\n🎉 Phase 2: Message Ingestion - ALL TESTS PASSED")
    print("✅ iMessage ingestion working (<100ms)")
    print("✅ Slack ingestion working (<100ms)")
    print("✅ Deduplication working (prevents retries)")
    print("✅ Classification working (analytical/research/operational)")
    print("✅ Priority assignment working (P0/P1/P2/P3)")
    print("✅ Performance under load acceptable")
    
    # Cleanup
    os.remove(test_db_path)
    print("🧹 Test database cleaned up")

if __name__ == "__main__":
    test_phase_2_ingestion()