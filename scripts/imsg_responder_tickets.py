#!/usr/bin/env python3
"""
Phase 2: iMessage Responder with Ticket System Integration
Enhanced version that creates tickets for all incoming messages.

Workflow:
1. Receive iMessage from Peder
2. Create ticket in database (<100ms)
3. Send immediate acknowledgment with ticket number
4. Queue processor will handle the actual response
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from services.ticket_ingestion import get_ingestion_service

PEDER_NUMBER = "+4794810537"
CHAT_ID = 1
LOG_FILE = "/tmp/imsg_tickets_responder.log"

def log(msg):
    """Log with timestamp"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def send_reply(text):
    """Send instant acknowledgment reply"""
    try:
        result = subprocess.run(
            ["imsg", "send", "--to", PEDER_NUMBER, "--text", text],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            log(f"✅ Sent: {text[:80]}")
            return True
        else:
            log(f"❌ Failed to send: {result.stderr}")
            return False
    except Exception as e:
        log(f"❌ Error sending: {e}")
        return False

def process_incoming_message(sender: str, text: str, msg_id: str = None) -> bool:
    """
    Process incoming message by creating ticket and sending acknowledgment.
    
    Returns:
        bool: True if processed successfully
    """
    try:
        # Initialize ingestion service
        ingestion = get_ingestion_service()
        
        # Create ticket (<100ms target)
        start_time = time.time()
        ticket_id = ingestion.ingest_imessage(
            sender_id=sender,
            sender_name="Peder",
            message_text=text,
            message_id=msg_id,
            metadata={
                "chat_id": CHAT_ID,
                "ingestion_method": "imsg_tickets_responder"
            }
        )
        
        ingestion_time = (time.time() - start_time) * 1000
        log(f"📥 Ticket created in {ingestion_time:.1f}ms (target: <100ms)")
        
        # Generate acknowledgment
        ack_info = ingestion.acknowledge_ticket_immediately(ticket_id)
        
        # Send immediate acknowledgment
        send_reply(ack_info["message"])
        
        log(f"✅ Processed message → Ticket #{ack_info['ticket_number']} "
            f"[{ack_info['priority']}/{ack_info['agent_type']}]")
        
        return True
        
    except Exception as e:
        log(f"❌ Error processing message: {e}")
        
        # Send error acknowledgment
        send_reply("⚠️ Message received but ticket creation failed. Please retry.")
        return False

def watch_messages():
    """Watch for incoming iMessages and create tickets."""
    log("=== iMessage Tickets Responder Started ===")
    log(f"Monitoring chat {CHAT_ID} for messages from {PEDER_NUMBER}")
    log("Creating tickets for all incoming messages...")
    
    try:
        # Use imsg watch to stream incoming messages
        proc = subprocess.Popen(
            ["imsg", "watch", "--chat-id", str(CHAT_ID), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        for line in proc.stdout:
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            # Only process incoming messages (not my own replies)
            is_from_me = msg.get("is_from_me", False)
            sender = msg.get("sender", "")
            text = msg.get("text", "")
            msg_id = msg.get("id")
            
            # Skip my own messages and empty messages
            if is_from_me or not text:
                continue
            
            # Only respond to messages from Peder
            if sender != PEDER_NUMBER:
                continue
            
            # Got a message from Peder!
            log(f"📱 INCOMING from {sender}: {text[:100]}")
            
            # Process message → create ticket → send ack
            process_incoming_message(sender, text, msg_id)
    
    except KeyboardInterrupt:
        log("=== iMessage Tickets Responder Stopped ===")
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    watch_messages()