"""
Stuck-Task Reaper Service

Monitors tickets in "IN_PROGRESS" state for >15 minutes.
Automatically kills hanging tasks and notifies user.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from backend.db.tickets_db import get_tickets_db, TicketStatus

logger = logging.getLogger(__name__)


class StuckTaskReaper:
    """Monitors and kills stuck tickets that exceed 15-minute threshold."""

    def __init__(self, check_interval: int = 60):
        """
        Args:
            check_interval: Seconds between checks (default: 60)
        """
        self.db = get_tickets_db()
        self.check_interval = check_interval
        self.running = False

    async def start(self):
        """Start the reaper loop."""
        self.running = True
        logger.info("🔪 Stuck-Task Reaper started (15-minute timeout)")
        
        while self.running:
            try:
                await self.check_and_reap()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in reaper loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def check_and_reap(self):
        """Check for stuck tickets and kill them."""
        # Get all IN_PROGRESS tickets
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, ticket_number, source, sender_id, assigned_at 
                    FROM tickets 
                    WHERE status = 'IN_PROGRESS'
                """)
                
                stuck_tickets = []
                now = datetime.now(timezone.utc)
                
                for ticket_id, ticket_number, source, sender_id, assigned_at_str in cursor.fetchall():
                    if not assigned_at_str:
                        continue
                    
                    assigned_at = datetime.fromisoformat(assigned_at_str.replace('Z', '+00:00'))
                    elapsed = (now - assigned_at).total_seconds()
                    
                    # 15-minute threshold (900 seconds)
                    if elapsed > 900:
                        stuck_tickets.append({
                            'id': ticket_id,
                            'number': ticket_number,
                            'source': source,
                            'sender_id': sender_id,
                            'elapsed': elapsed
                        })
                
                # Kill stuck tickets
                for ticket in stuck_tickets:
                    await self.kill_ticket(ticket)
                
                if stuck_tickets:
                    logger.warning(f"🔪 Reaped {len(stuck_tickets)} stuck tasks")
        
        except Exception as e:
            logger.error(f"Error checking for stuck tasks: {e}")

    async def kill_ticket(self, ticket: Dict[str, Any]):
        """Kill a stuck ticket and notify user."""
        ticket_id = ticket['id']
        ticket_number = ticket['number']
        source = ticket['source']
        sender_id = ticket['sender_id']
        elapsed = ticket['elapsed']
        
        try:
            # Mark ticket as CLOSED with error
            self.db.update_ticket_status(
                ticket_id,
                TicketStatus.CLOSED,
                error_message=f"Execution timeout: {int(elapsed)}s > 15min threshold"
            )
            
            logger.warning(f"🔪 KILLED: Ticket #{ticket_number} ({int(elapsed)}s elapsed)")
            
            # Notify user
            await self.notify_user_of_kill(ticket_number, source, sender_id, elapsed)
            
            # Update queue positions for remaining tickets
            self.db.update_queue_positions()
            
        except Exception as e:
            logger.error(f"Error killing ticket #{ticket_number}: {e}")

    async def notify_user_of_kill(self, ticket_number: int, source: str, sender_id: str, elapsed: float):
        """Send notification to user about killed task."""
        message = (
            f"⚠️ SYSTEM ALERT: Ticket #{ticket_number} exceeded the 15-minute execution threshold "
            f"(ran for {int(elapsed)}s) and has been terminated to prevent queue bloat.\n"
            f"Please resubmit or check logs for details."
        )
        
        try:
            if source == 'slack':
                # Post to Slack
                from slack_sdk import WebClient
                from backend.config.settings import get_settings
                settings = get_settings()
                client = WebClient(token=settings.slack_bot_token)
                client.chat_postMessage(
                    channel=sender_id,
                    text=message
                )
                logger.info(f"✅ Notification sent to Slack for killed ticket #{ticket_number}")
            
            elif source == 'imessage':
                # Send to iMessage
                import subprocess
                subprocess.run(
                    ["imsg", "send", "--to", sender_id, "--text", message],
                    timeout=5
                )
                logger.info(f"✅ Notification sent to iMessage for killed ticket #{ticket_number}")
        
        except Exception as e:
            logger.error(f"Failed to notify user of killed task: {e}")

    def stop(self):
        """Stop the reaper loop."""
        self.running = False
        logger.info("🔪 Stuck-Task Reaper stopped")


# Global instance
_reaper = None


async def start_stuck_task_reaper(check_interval: int = 60):
    """Start the stuck-task reaper service."""
    global _reaper
    _reaper = StuckTaskReaper(check_interval=check_interval)
    await _reaper.start()


def get_stuck_task_reaper() -> StuckTaskReaper:
    """Get the global reaper instance."""
    global _reaper
    return _reaper
