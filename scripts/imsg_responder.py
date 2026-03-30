#!/usr/bin/env python3
"""
Phase 2.11: iMessage Responder
Listens for incoming iMessages from Peder and responds instantly.
Only processes incoming messages (is_from_me: false), not own replies.
"""

import json
import subprocess
import sys
import time
from datetime import datetime

PEDER_NUMBER = "+4794810537"
CHAT_ID = 1
LOG_FILE = "/tmp/imsg_responder.log"

def log(msg):
    """Log with timestamp"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def send_reply(text):
    """Send instant reply"""
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

def watch_messages():
    """Watch for incoming iMessages and respond instantly"""
    log("=== iMessage Responder Started ===")
    log(f"Monitoring chat {CHAT_ID} for messages from {PEDER_NUMBER}")
    
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
            
            # Send instant reply
            reply = f"✅ Received at {datetime.now().strftime('%H:%M:%S')}: '{text[:40]}...'"
            send_reply(reply)
    
    except KeyboardInterrupt:
        log("=== Responder Stopped ===")
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    watch_messages()
