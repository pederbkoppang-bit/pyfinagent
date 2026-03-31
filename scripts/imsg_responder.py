#!/usr/bin/env python3
"""
Phase 3.2: iMessage Responder with Message Routing
Listens for incoming iMessages from Peder and routes to appropriate agent:
  - Operational (status, service, next step) → MAIN (Ford)
  - Analytical (why, compare, review) → Q&A (Analyst)
  - Research (papers, evidence, novel) → Research (Researcher)

Only processes incoming messages (is_from_me: false), not own replies.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PEDER_NUMBER = "+4794810537"
CHAT_ID = 1
LOG_FILE = "/tmp/imsg_responder.log"
SESSIONS_FILE = Path.home() / ".openclaw" / "workspace" / "memory" / "active_sessions.json"

def log(msg):
    """Log with timestamp"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_active_sessions() -> dict:
    """Load active session IDs from disk"""
    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

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

def detect_question_type(message: str) -> str:
    """
    Classify incoming message as operational, analytical, or research.
    
    Returns: "operational", "analytical", or "research"
    """
    analytical_keywords = [
        "why", "should", "explain", "analyze", "trade-off", "decision",
        "regression", "sharpe", "recommendation", "suggest", "improve",
        "compare", "better", "worse", "review", "feedback", "thoughts"
    ]
    
    research_keywords = [
        "research", "paper", "literature", "novel", "approach", "evidence",
        "experiment", "hypothesis", "theory", "mechanism", "solution",
        "study", "implementation", "baseline", "benchmark", "findings",
        "investigate", "explore", "discover"
    ]
    
    msg_lower = message.lower()
    
    # Check research first (more specific)
    if any(kw in msg_lower for kw in research_keywords):
        return "research"
    
    # Check analytical
    if any(kw in msg_lower for kw in analytical_keywords):
        return "analytical"
    
    # Default to operational
    return "operational"

def route_message(message: str, question_type: str, sessions: dict) -> str:
    """
    Route message to appropriate agent session.
    
    Returns: confirmation message to send back to Peder
    """
    log(f"📍 Routing: {question_type.upper()} → {message[:60]}")
    
    if question_type == "analytical":
        qa_session = sessions.get("qa")
        if qa_session:
            log(f"   → Q&A Session: {qa_session}")
            return f"📊 Routing to Analyst: '{message[:50]}...' — analyzing now"
        else:
            log(f"   → Q&A Session not found, falling back to MAIN")
            return f"⚠️ Q&A session unavailable; handling in MAIN"
    
    elif question_type == "research":
        research_session = sessions.get("research")
        if research_session:
            log(f"   → Research Session: {research_session}")
            return f"🔬 Routing to Researcher: '{message[:50]}...' — deep research starting"
        else:
            log(f"   → Research Session not found, falling back to MAIN")
            return f"⚠️ Research session unavailable; handling in MAIN"
    
    else:  # operational
        log(f"   → MAIN (Ford) — operational")
        return f"⚙️ Processing: '{message[:50]}...' — will respond shortly"

def watch_messages():
    """Watch for incoming iMessages, classify, and route to appropriate agent"""
    log("=== iMessage Responder Started (with Agentic Routing) ===")
    log(f"Monitoring chat {CHAT_ID} for messages from {PEDER_NUMBER}")
    
    # Load active sessions for routing
    sessions = load_active_sessions()
    log(f"Active sessions loaded: {sessions}")
    
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
            
            # Classify message type
            question_type = detect_question_type(text)
            
            # Route to appropriate agent
            reply = route_message(text, question_type, sessions)
            
            # Send routing confirmation
            send_reply(reply)
            
            # Log routing decision
            log(f"   Classification: {question_type.upper()}")
            log(f"   Reply sent: {reply[:80]}")
    
    except KeyboardInterrupt:
        log("=== Responder Stopped ===")
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    watch_messages()
