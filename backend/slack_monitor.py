"""
Phase 2.11: Slack Responsiveness Monitor
Background task that keeps Ford actively responsive on Slack.
- Monitors #ford-approvals for messages every 2-3 min
- Routes mentions to main Ford session for immediate reply
- Posts realtime status updates
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("slack_monitor")

class SlackMonitor:
    def __init__(self, slack_client, channel_id: str = "C0ANTGNNK8D"):
        self.slack = slack_client
        self.channel_id = channel_id
        self.last_processed_ts = None
        self.running = False
        
    async def start(self):
        """Start background monitoring loop"""
        self.running = True
        logger.info("SlackMonitor: Starting background monitoring")
        
        while self.running:
            try:
                await self.check_for_messages()
                await asyncio.sleep(120)  # Check every 2 minutes
            except Exception as e:
                logger.error(f"SlackMonitor error: {e}")
                await asyncio.sleep(120)
    
    async def check_for_messages(self):
        """Poll for new messages since last check"""
        try:
            # Get recent messages in channel
            result = await asyncio.to_thread(
                self.slack.conversations_history,
                channel=self.channel_id,
                oldest=self.last_processed_ts or "0",
                limit=10
            )
            
            messages = result.get("messages", [])
            if not messages:
                return
            
            # Process in chronological order (oldest first)
            messages.reverse()
            
            for msg in messages:
                ts = msg.get("ts")
                if ts:
                    self.last_processed_ts = ts
                
                # Check if message mentions Ford
                text = msg.get("text", "")
                user = msg.get("user")
                
                if "<@B09V7T55K0X>" in text or "<@B0A13KXG4TS>" in text:
                    logger.info(f"SlackMonitor: Found Ford mention: {text[:50]}")
                    # TODO: Route to main Ford session for response
                    # For now, just acknowledge we saw it
                    await asyncio.to_thread(
                        self.slack.reactions_add,
                        channel=self.channel_id,
                        timestamp=ts,
                        emoji="eyes"
                    )
                    
        except Exception as e:
            logger.error(f"SlackMonitor.check_for_messages error: {e}")
    
    async def post_status(self, status_text: str):
        """Post realtime status update to #ford-approvals"""
        try:
            await asyncio.to_thread(
                self.slack.chat_postMessage,
                channel=self.channel_id,
                text=status_text
            )
        except Exception as e:
            logger.error(f"SlackMonitor.post_status error: {e}")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("SlackMonitor: Stopping background monitoring")

# Global monitor instance
_monitor: Optional[SlackMonitor] = None

async def init_slack_monitor(slack_client):
    """Initialize and start slack monitor"""
    global _monitor
    _monitor = SlackMonitor(slack_client)
    asyncio.create_task(_monitor.start())
    logger.info("SlackMonitor initialized")

def get_slack_monitor() -> Optional[SlackMonitor]:
    """Get the running slack monitor instance"""
    return _monitor
