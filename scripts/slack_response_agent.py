#!/usr/bin/env python3
"""
Phase 2.11: Slack Response Agent
Dedicated process that listens for mentions in #ford-approvals via Socket Mode.
Responds instantly (<1 second) without blocking main work.

Run in background: python slack_response_agent.py &
"""

import os
import json
import logging
from datetime import datetime
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/tmp/slack_response_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SlackResponseAgent")

def get_slack_tokens():
    """Load Slack tokens from OpenClaw config"""
    try:
        config_path = os.path.expanduser("~/.openclaw/openclaw.json")
        with open(config_path) as f:
            config = json.load(f)
            return (
                config["channels"]["slack"]["botToken"],
                config["channels"]["slack"]["appToken"]
            )
    except Exception as e:
        logger.error(f"Failed to load tokens: {e}")
        return None, None

def main():
    # Load tokens
    bot_token, app_token = get_slack_tokens()
    if not bot_token or not app_token:
        logger.error("Slack tokens not configured")
        exit(1)
    
    # Initialize Bolt app
    app = App(token=bot_token)
    
    logger.info("=" * 60)
    logger.info("Slack Response Agent Starting")
    logger.info(f"Bot Token: {bot_token[:20]}...")
    logger.info(f"App Token: {app_token[:20]}...")
    logger.info("=" * 60)
    
    # Listen for app mentions
    @app.event("app_mention")
    def handle_app_mention(event, say, logger):
        user = event.get("user")
        text = event.get("text", "")
        ts = event.get("ts")
        
        logger.info(f"[MENTION] User: {user}, Text: {text[:100]}")
        
        # Respond immediately
        response = f"✅ Ford received your message at {datetime.now().strftime('%H:%M:%S')}. Processing..."
        say(text=response, thread_ts=ts)
        
        # Log to state file for main session to detect
        try:
            state_file = os.path.expanduser("~/.openclaw/workspace/pyfinagent/handoff/.slack_mention_state")
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, "a") as f:
                f.write(f"[{ts}] @{user}: {text[:200]}\n")
        except:
            pass
    
    # Listen for direct messages
    @app.event("message")
    def handle_message(event, say, logger):
        # Only respond to messages that mention Ford
        text = event.get("text", "")
        if any(mention in text for mention in ["<@B0A13KXG4TS>", "<@U0A0CTMGF5J>", "<@B0ANUU5TTFY>", "<@B09V7T55K0X>"]):
            user = event.get("user")
            ts = event.get("ts")
            logger.info(f"[MESSAGE] User: {user}, Text: {text[:100]}")
            
            # Acknowledge immediately
            response = f"✅ Acknowledged at {datetime.now().strftime('%H:%M:%S')}"
            say(text=response, thread_ts=ts)
    
    # Start Socket Mode handler with auto-reconnect
    logger.info("Starting Socket Mode listener...")
    handler = SocketModeHandler(app, app_token)
    
    reconnect_count = 0
    max_reconnects = 10
    
    while reconnect_count < max_reconnects:
        try:
            handler.start()
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            handler.close()
            exit(0)
        except ConnectionResetError as e:
            reconnect_count += 1
            logger.error(f"Connection reset. Reconnecting... ({reconnect_count}/{max_reconnects})")
            logger.error(f"Details: {e}")
            import time
            time.sleep(min(2 ** reconnect_count, 60))  # Exponential backoff
            handler = SocketModeHandler(app, app_token)
        except BrokenPipeError as e:
            reconnect_count += 1
            logger.error(f"Broken pipe. Reconnecting... ({reconnect_count}/{max_reconnects})")
            logger.error(f"Details: {e}")
            import time
            time.sleep(min(2 ** reconnect_count, 60))  # Exponential backoff
            handler = SocketModeHandler(app, app_token)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            exit(1)
    
    logger.error(f"Failed to reconnect after {max_reconnects} attempts")

if __name__ == "__main__":
    main()
