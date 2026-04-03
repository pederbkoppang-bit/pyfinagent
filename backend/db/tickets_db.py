"""
Ticket System Database - SQLite backend for reliable message handling.

This module implements the ticket system for Slack/iMessage message queue processing.
Uses SQLite for simplicity and to avoid additional infrastructure requirements.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class TicketStatus(Enum):
    OPEN = "OPEN"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    DUPLICATE = "DUPLICATE"

class TicketPriority(Enum):
    P0 = "P0"  # Critical - 5 min SLA
    P1 = "P1"  # Urgent - 15 min SLA
    P2 = "P2"  # Standard - 1 hour SLA
    P3 = "P3"  # Low - 4 hours SLA

class TicketSource(Enum):
    SLACK = "slack"
    IMESSAGE = "imessage"

class TicketClassification(Enum):
    OPERATIONAL = "operational"
    ANALYTICAL = "analytical"
    RESEARCH = "research"

class TicketsDB:
    """SQLite-based ticket system database."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path.home() / ".openclaw" / "workspace" / "pyfinagent" / "tickets.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database and tables
        self._init_database()
        
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create tickets table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_number INTEGER UNIQUE NOT NULL,
                    source TEXT NOT NULL CHECK(source IN ('slack', 'imessage')),
                    sender_id TEXT NOT NULL,
                    sender_name TEXT,
                    channel_id TEXT,  -- Slack channel or iMessage contact
                    message_text TEXT NOT NULL,
                    priority TEXT NOT NULL DEFAULT 'P2' CHECK(priority IN ('P0', 'P1', 'P2', 'P3')),
                    status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'ASSIGNED', 'IN_PROGRESS', 'RESOLVED', 'CLOSED', 'DUPLICATE')),
                    classification TEXT DEFAULT 'operational' CHECK(classification IN ('operational', 'analytical', 'research')),
                    assigned_agent TEXT,  -- Q&A, Research, MAIN, etc.
                    queue_position INTEGER DEFAULT 0,  -- Dynamic position in queue (updates as tickets move)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acknowledged_at TIMESTAMP,
                    assigned_at TIMESTAMP,
                    in_progress_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    closed_at TIMESTAMP,
                    response_text TEXT,
                    slack_thread_id TEXT,
                    slack_envelope_id TEXT,  -- For deduplication
                    slack_event_ts TEXT,     -- Slack event timestamp
                    response_sla_seconds INTEGER,  -- Target response time in seconds
                    resolution_sla_seconds INTEGER,  -- Target resolution time in seconds
                    retries INTEGER DEFAULT 0,
                    error_message TEXT,
                    metadata TEXT  -- JSON for additional data
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_envelope_id ON tickets(slack_envelope_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_source ON tickets(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_number ON tickets(ticket_number)")
            
            # Create ticket_counter table for generating unique ticket numbers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticket_counter (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_number INTEGER DEFAULT 5000
                )
            """)
            
            # Initialize counter if empty
            conn.execute("INSERT OR IGNORE INTO ticket_counter (id, current_number) VALUES (1, 5000)")
            
            conn.commit()
            logger.info(f"Tickets database initialized at {self.db_path}")

    def _get_next_ticket_number(self, conn: sqlite3.Connection) -> int:
        """Generate the next unique ticket number."""
        cursor = conn.execute("UPDATE ticket_counter SET current_number = current_number + 1 WHERE id = 1")
        cursor = conn.execute("SELECT current_number FROM ticket_counter WHERE id = 1")
        row = cursor.fetchone()
        return row[0] if row else 5001

    def _calculate_sla_seconds(self, priority: TicketPriority) -> tuple[int, int]:
        """Calculate response and resolution SLA in seconds based on priority."""
        sla_map = {
            TicketPriority.P0: (5 * 60, 30 * 60),      # 5 min, 30 min
            TicketPriority.P1: (15 * 60, 2 * 3600),    # 15 min, 2 hours
            TicketPriority.P2: (60 * 60, 8 * 3600),    # 1 hour, 8 hours
            TicketPriority.P3: (4 * 3600, 24 * 3600),  # 4 hours, 24 hours
        }
        return sla_map.get(priority, (60 * 60, 8 * 3600))

    def create_ticket(
        self,
        source: TicketSource,
        sender_id: str,
        message_text: str,
        sender_name: str = None,
        channel_id: str = None,
        priority: TicketPriority = TicketPriority.P2,
        classification: TicketClassification = TicketClassification.OPERATIONAL,
        slack_envelope_id: str = None,
        slack_event_ts: str = None,
        slack_thread_id: str = None,
        metadata: dict = None
    ) -> int:
        """
        Create a new ticket.
        
        Returns:
            int: The ticket ID
            
        Raises:
            sqlite3.IntegrityError: If duplicate slack_envelope_id (for deduplication)
        """
        response_sla, resolution_sla = self._calculate_sla_seconds(priority)
        
        with sqlite3.connect(self.db_path) as conn:
            ticket_number = self._get_next_ticket_number(conn)
            
            cursor = conn.execute("""
                INSERT INTO tickets (
                    ticket_number, source, sender_id, sender_name, channel_id,
                    message_text, priority, classification,
                    slack_envelope_id, slack_event_ts, slack_thread_id,
                    response_sla_seconds, resolution_sla_seconds,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticket_number, source.value, sender_id, sender_name, channel_id,
                message_text, priority.value, classification.value,
                slack_envelope_id, slack_event_ts, slack_thread_id,
                response_sla, resolution_sla,
                json.dumps(metadata) if metadata else None
            ))
            
            ticket_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created ticket #{ticket_number} (ID {ticket_id}) from {source.value}: {message_text[:50]}")
            return ticket_id

    def is_duplicate_envelope(self, envelope_id: str) -> bool:
        """Check if a Slack envelope_id has already been processed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id FROM tickets WHERE slack_envelope_id = ? LIMIT 1",
                (envelope_id,)
            )
            return cursor.fetchone() is not None

    def mark_duplicate(self, envelope_id: str) -> Optional[int]:
        """Mark a ticket as duplicate by envelope_id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE tickets SET status = 'DUPLICATE' WHERE slack_envelope_id = ?",
                (envelope_id,)
            )
            if cursor.rowcount > 0:
                conn.commit()
                # Get the ticket ID
                cursor = conn.execute(
                    "SELECT id FROM tickets WHERE slack_envelope_id = ?",
                    (envelope_id,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
        return None

    def get_ticket(self, ticket_id: int) -> Optional[Dict[str, Any]]:
        """Get a ticket by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
            row = cursor.fetchone()
            if row:
                ticket = dict(row)
                if ticket['metadata']:
                    ticket['metadata'] = json.loads(ticket['metadata'])
                return ticket
        return None

    def get_ticket_by_number(self, ticket_number: int) -> Optional[Dict[str, Any]]:
        """Get a ticket by ticket number."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM tickets WHERE ticket_number = ?", (ticket_number,))
            row = cursor.fetchone()
            if row:
                ticket = dict(row)
                if ticket['metadata']:
                    ticket['metadata'] = json.loads(ticket['metadata'])
                return ticket
        return None

    def get_open_tickets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all open tickets, ordered by priority and creation time (FIFO)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM tickets 
                WHERE status = 'OPEN' 
                ORDER BY 
                    CASE priority 
                        WHEN 'P0' THEN 0 
                        WHEN 'P1' THEN 1 
                        WHEN 'P2' THEN 2 
                        WHEN 'P3' THEN 3 
                    END,
                    created_at ASC
                LIMIT ?
            """, (limit,))
            
            tickets = []
            for row in cursor.fetchall():
                ticket = dict(row)
                if ticket['metadata']:
                    ticket['metadata'] = json.loads(ticket['metadata'])
                tickets.append(ticket)
            return tickets

    def update_ticket_status(
        self,
        ticket_id: int,
        status: TicketStatus,
        assigned_agent: str = None,
        response_text: str = None,
        error_message: str = None
    ) -> bool:
        """Update ticket status and related fields."""
        now = datetime.now(timezone.utc).isoformat()
        

        
        updates = ["status = ?"]
        params = [status.value]
        
        # Set timestamp based on status - use string values to avoid enum import issues
        status_value = status.value if hasattr(status, 'value') else str(status)
        
        if status_value == 'ASSIGNED':
            updates.append("assigned_at = ?")
            params.append(now)
            if assigned_agent:
                updates.append("assigned_agent = ?")
                params.append(assigned_agent)
        elif status_value == 'IN_PROGRESS':
            updates.append("in_progress_at = ?")
            params.append(now)
        elif status_value == 'RESOLVED':
            updates.append("resolved_at = ?")
            params.append(now)
            if response_text:
                updates.append("response_text = ?")
                params.append(response_text)
        elif status_value == 'CLOSED':
            updates.append("closed_at = ?")
            params.append(now)
        
        if error_message:
            updates.append("error_message = ?")
            params.append(error_message)
        
        params.append(ticket_id)
        
        sql = f"UPDATE tickets SET {', '.join(updates)} WHERE id = ?"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            success = cursor.rowcount > 0
            if success:
                conn.commit()
                logger.info(f"Updated ticket {ticket_id} status to {status.value}")
            else:
                logger.error(f"Update failed for ticket {ticket_id} - no rows affected")
            return success

    def acknowledge_ticket(self, ticket_id: int) -> bool:
        """Mark ticket as acknowledged."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE tickets SET acknowledged_at = ? WHERE id = ?",
                (now, ticket_id)
            )
            success = cursor.rowcount > 0
            if success:
                conn.commit()
            return success

    def get_ticket_stats(self) -> Dict[str, Any]:
        """Get ticket system statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Basic counts
            cursor = conn.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM tickets 
                GROUP BY status
            """)
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # SLA breaches (tickets where resolution took longer than SLA)
            cursor = conn.execute("""
                SELECT COUNT(*) 
                FROM tickets 
                WHERE resolved_at IS NOT NULL 
                  AND (julianday(resolved_at) - julianday(created_at)) * 86400 > resolution_sla_seconds
            """)
            sla_breaches = cursor.fetchone()[0]
            
            # Average response time for resolved tickets (in minutes)
            cursor = conn.execute("""
                SELECT AVG((julianday(resolved_at) - julianday(created_at)) * 1440) 
                FROM tickets 
                WHERE resolved_at IS NOT NULL
            """)
            avg_response_time = cursor.fetchone()[0]
            
            return {
                "status_counts": status_counts,
                "sla_breaches": sla_breaches,
                "avg_response_time_minutes": avg_response_time,
                "total_tickets": sum(status_counts.values())
            }

    def get_sla_breaches(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get tickets that have breached their SLA."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT *,
                       (julianday(COALESCE(resolved_at, datetime('now'))) - julianday(created_at)) * 86400 as elapsed_seconds
                FROM tickets 
                WHERE elapsed_seconds > response_sla_seconds
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            breaches = []
            for row in cursor.fetchall():
                ticket = dict(row)
                if ticket['metadata']:
                    ticket['metadata'] = json.loads(ticket['metadata'])
                breaches.append(ticket)
            return breaches

    def cleanup_old_tickets(self, days_old: int = 30) -> int:
        """Archive tickets older than specified days. Returns number of archived tickets."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE tickets 
                SET status = 'CLOSED' 
                WHERE status = 'RESOLVED' 
                  AND date(resolved_at) < date('now', '-{} days')
            """.format(days_old))
            archived_count = cursor.rowcount
            if archived_count > 0:
                conn.commit()
                logger.info(f"Archived {archived_count} old tickets")
            return archived_count

    def get_ticket_queue_position(self, ticket_id: int) -> int:
        """Get queue position for a ticket (1-based index in OPEN/IN_PROGRESS queue)."""
        with sqlite3.connect(self.db_path) as conn:
            # Count how many OPEN/IN_PROGRESS/ASSIGNED tickets were created before this one
            cursor = conn.execute("""
                SELECT COUNT(*) 
                FROM tickets t1
                WHERE t1.status IN ('OPEN', 'IN_PROGRESS', 'ASSIGNED')
                  AND t1.created_at <= (SELECT created_at FROM tickets WHERE id = ?)
                  AND t1.id <= ?
            """, (ticket_id, ticket_id))
            position = cursor.fetchone()[0]
            return position if position > 0 else 1

    def update_queue_positions(self) -> None:
        """
        DYNAMIC QUEUE POSITIONING: Update queue_position for all OPEN/IN_PROGRESS/ASSIGNED tickets.
        Called when queue state changes to reflect current position to users.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get all active tickets ordered by creation time
            cursor = conn.execute("""
                SELECT id FROM tickets
                WHERE status IN ('OPEN', 'IN_PROGRESS', 'ASSIGNED')
                ORDER BY created_at ASC, id ASC
            """)
            
            rows = cursor.fetchall()
            for position, (ticket_id,) in enumerate(rows, start=1):
                conn.execute(
                    "UPDATE tickets SET queue_position = ? WHERE id = ?",
                    (position, ticket_id)
                )
            
            if rows:
                conn.commit()
                logger.info(f"Updated queue positions for {len(rows)} active tickets")

# Global instance
_tickets_db: Optional[TicketsDB] = None

def get_tickets_db() -> TicketsDB:
    """Get the global tickets database instance."""
    global _tickets_db
    if _tickets_db is None:
        _tickets_db = TicketsDB()
    return _tickets_db

def init_tickets_db(db_path: str = None) -> TicketsDB:
    """Initialize the tickets database."""
    global _tickets_db
    _tickets_db = TicketsDB(db_path)
    return _tickets_db