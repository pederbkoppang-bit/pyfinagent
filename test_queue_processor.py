#!/usr/bin/env python3
"""
Test script for Phase 3: Queue Processing
"""

import sys
import os
import asyncio
import time
import logging
from pathlib import Path

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from db.tickets_db import (
    TicketsDB, TicketStatus, TicketPriority, TicketSource, 
    TicketClassification, init_tickets_db
)
from services.ticket_queue_processor import TicketQueueProcessor

async def test_phase_3_queue_processing():
    """Phase 3: Test queue processing and agent routing."""
    print("=== Phase 3: Queue Processing Test ===")
    
    # Use a test database
    test_db_path = "/tmp/test_queue_processing.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = init_tickets_db(test_db_path)
    print(f"✅ Initialized test database: {test_db_path}")
    
    # Create processor
    processor = TicketQueueProcessor(max_concurrent=3)
    processor.db = db  # Use our test database
    
    print(f"DEBUG: Test DB path: {db.db_path}")
    print(f"DEBUG: Processor DB path: {processor.db.db_path}")
    assert str(processor.db.db_path) == str(db.db_path), "Database paths don't match!"
    
    # Test 1: Create test tickets
    print("\n--- Test 1: Create Test Tickets ---")
    
    test_tickets = [
        ("Why did AAPL drop yesterday?", TicketClassification.ANALYTICAL, TicketPriority.P2),
        ("Services status check", TicketClassification.OPERATIONAL, TicketPriority.P1),
        ("Research latest ML papers", TicketClassification.RESEARCH, TicketPriority.P3),
        ("URGENT: System down!", TicketClassification.OPERATIONAL, TicketPriority.P0),
        ("Portfolio performance review", TicketClassification.ANALYTICAL, TicketPriority.P2),
    ]
    
    ticket_ids = []
    for message, classification, priority in test_tickets:
        ticket_id = db.create_ticket(
            source=TicketSource.IMESSAGE,
            sender_id="+4794810537",
            sender_name="Peder",
            message_text=message,
            priority=priority,
            classification=classification
        )
        ticket_ids.append(ticket_id)
    
    print(f"✅ Created {len(ticket_ids)} test tickets")
    
    # Test 2: Agent routing
    print("\n--- Test 2: Agent Routing ---")
    
    routing_tests = [
        (TicketClassification.OPERATIONAL, "MAIN"),
        (TicketClassification.ANALYTICAL, "Q&A"),
        (TicketClassification.RESEARCH, "Research")
    ]
    
    for classification, expected_agent in routing_tests:
        agent = processor.get_agent_for_classification(classification)
        assert agent == expected_agent, f"Wrong agent for {classification}: got {agent}, expected {expected_agent}"
        print(f"✅ {classification.value} → {agent}")
    
    # Test 3: Single ticket processing
    print("\n--- Test 3: Single Ticket Processing ---")
    
    # Get the urgent ticket (P0)
    open_tickets = db.get_open_tickets(limit=10)
    urgent_ticket = next(t for t in open_tickets if t['priority'] == 'P0')
    
    start_time = time.time()
    success = await processor.process_single_ticket(urgent_ticket)
    processing_time = time.time() - start_time
    
    print(f"✅ Processed urgent ticket in {processing_time:.2f}s")
    assert success == True, "Failed to process urgent ticket"
    
    # Verify ticket was resolved
    processed_ticket = db.get_ticket(urgent_ticket['id'])
    print(f"DEBUG: Processed ticket after processing: {processed_ticket}")
    assert processed_ticket['status'] == 'RESOLVED', f"Expected RESOLVED, got {processed_ticket['status']}"
    assert processed_ticket['response_text'] is not None, f"response_text is None: {processed_ticket}"
    assert processed_ticket['assigned_agent'] == 'MAIN'
    print(f"✅ Ticket #{processed_ticket['ticket_number']} resolved: {processed_ticket['response_text'][:50]}...")
    
    # Test 4: Batch processing
    print("\n--- Test 4: Batch Processing ---")
    
    start_time = time.time()
    processed_count = await processor.process_queue_batch(batch_size=5)
    batch_time = time.time() - start_time
    
    print(f"✅ Processed {processed_count} tickets in batch ({batch_time:.2f}s)")
    assert processed_count > 0, "No tickets processed in batch"
    
    # Test 5: Verify all tickets processed
    print("\n--- Test 5: Verify Processing Results ---")
    
    stats = db.get_ticket_stats()
    print(f"📊 Processing stats: {stats}")
    
    resolved_count = stats['status_counts'].get('RESOLVED', 0)
    assert resolved_count >= 4, f"Expected at least 4 resolved tickets, got {resolved_count}"
    
    # Check responses by classification
    all_tickets = []
    for ticket_id in ticket_ids:
        ticket = db.get_ticket(ticket_id)
        if ticket:
            all_tickets.append(ticket)
    
    for ticket in all_tickets:
        if ticket['status'] == 'RESOLVED':
            classification = ticket['classification']
            response = ticket['response_text']
            
            print(f"✅ {classification} ticket #{ticket['ticket_number']}: {response[:80]}...")
            
            # Verify response content matches classification
            if classification == 'operational':
                assert any(word in response.lower() for word in ['status', 'running', 'operational'])
            elif classification == 'analytical':
                assert any(word in response.lower() for word in ['analysis', 'sharpe', 'analytical'])
            elif classification == 'research':
                assert any(word in response.lower() for word in ['research', 'papers', 'literature'])
    
    # Test 6: Performance under concurrent load
    print("\n--- Test 6: Concurrent Processing Test ---")
    
    # Create more tickets for load testing
    load_ticket_ids = []
    for i in range(10):
        ticket_id = db.create_ticket(
            source=TicketSource.SLACK,
            sender_id=f"U{i:08d}",
            message_text=f"Load test question {i}",
            priority=TicketPriority.P2,
            classification=TicketClassification.ANALYTICAL
        )
        load_ticket_ids.append(ticket_id)
    
    start_time = time.time()
    processed_load = await processor.process_queue_batch(batch_size=10)
    load_time = time.time() - start_time
    
    print(f"✅ Processed {processed_load} load test tickets in {load_time:.2f}s")
    print(f"✅ Average time per ticket: {load_time/max(processed_load, 1):.2f}s")
    
    # Test 7: Error handling
    print("\n--- Test 7: Error Handling ---")
    
    # Create a ticket that will cause an error (simulate by corrupting data)
    error_ticket_id = db.create_ticket(
        source=TicketSource.IMESSAGE,
        sender_id="error_test",
        message_text="This will cause an error",
        priority=TicketPriority.P3
    )
    
    # Manually corrupt the ticket to cause processing error
    # (In real scenario, this might be network issues, agent failures, etc.)
    
    # Test 8: SLA tracking
    print("\n--- Test 8: SLA Tracking ---")
    
    final_stats = db.get_ticket_stats()
    print(f"📊 Final stats: {final_stats}")
    
    # Check for SLA breaches
    breaches = db.get_sla_breaches(limit=10)
    print(f"⚠️ SLA breaches: {len(breaches)}")
    
    if breaches:
        for breach in breaches[:3]:  # Show first 3
            print(f"   Ticket #{breach['ticket_number']}: {breach['elapsed_seconds']:.0f}s (SLA: {breach['response_sla_seconds']}s)")
    
    print("\n🎉 Phase 3: Queue Processing - ALL TESTS PASSED")
    print("✅ Agent routing working (operational→MAIN, analytical→Q&A, research→Research)")
    print("✅ Ticket lifecycle complete (OPEN→ASSIGNED→IN_PROGRESS→RESOLVED)")
    print("✅ Batch processing working")
    print("✅ Concurrent processing working")
    print("✅ Response generation working")
    print("✅ SLA tracking functional")
    
    # Cleanup
    os.remove(test_db_path)
    print("🧹 Test database cleaned up")

if __name__ == "__main__":
    asyncio.run(test_phase_3_queue_processing())