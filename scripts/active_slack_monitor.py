#!/usr/bin/env python3
"""
Phase 2.11: Active Slack Monitor
Continuously polls #ford-approvals every 60 seconds for mentions.
Routes to Ford session immediately (not via cron).
"""

import time
import json
import subprocess
import os
from datetime import datetime
from slack_sdk import WebClient

CHANNEL_ID = "C0ANTGNNK8D"
POLL_INTERVAL = 60  # seconds
RESPONSE_SLA = 120  # 2 minutes

def get_slack_token():
    """Load token from OpenClaw config"""
    try:
        with open(os.path.expanduser("~/.openclaw/openclaw.json")) as f:
            config = json.load(f)
            return config["channels"]["slack"]["botToken"]
    except:
        return None

def poll_mentions(client, last_ts):
    """Poll for new messages since last_ts"""
    try:
        result = client.conversations_history(
            channel=CHANNEL_ID,
            oldest=last_ts,
            limit=20
        )
        return result.get("messages", []), result.get("response_metadata", {}).get("latest_ts")
    except Exception as e:
        print(f"[ERROR] Failed to poll: {e}")
        return [], last_ts

def route_to_session(message_text):
    """Route mention to Ford's main Slack session"""
    try:
        # This would integrate with OpenClaw's message routing
        # For now, just log it
        with open("/tmp/active_slack_monitor.log", "a") as f:
            f.write(f"[{datetime.now().isoformat()}] ROUTED: {message_text[:100]}\n")
        return True
    except:
        return False

def main():
    token = get_slack_token()
    if not token:
        print("[ERROR] SLACK_BOT_TOKEN not found")
        exit(1)
    
    client = WebClient(token=token)
    last_ts = "0"
    
    print(f"[{datetime.now().isoformat()}] Active Slack Monitor started")
    print(f"  Channel: {CHANNEL_ID}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print(f"  SLA: {RESPONSE_SLA}s")
    
    while True:
        try:
            messages, last_ts = poll_mentions(client, last_ts)
            
            for msg in reversed(messages):  # Process oldest first
                text = msg.get("text", "")
                ts = msg.get("ts")
                user = msg.get("user")
                
                # Check for mention
                if any(mention in text for mention in ["<@B0A13KXG4TS>", "<@U0A0CTMGF5J>", "<@B0ANUU5TTFY>"]):
                    route_to_session(f"[{ts}] <@{user}>: {text[:100]}")
                    print(f"[{datetime.now().isoformat()}] MENTION DETECTED: {text[:80]}")
            
            time.sleep(POLL_INTERVAL)
        
        except KeyboardInterrupt:
            print(f"\n[{datetime.now().isoformat()}] Monitor stopped")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
