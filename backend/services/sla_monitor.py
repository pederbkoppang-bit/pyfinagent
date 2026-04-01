"""
Phase 6: SLA Monitoring Service
Monitors response times and escalates SLA breaches.
"""

import logging
import asyncio
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

from backend.db.tickets_db import get_tickets_db, TicketPriority

logger = logging.getLogger(__name__)

class SLAMonitoringService:
    """Service that monitors SLA compliance and handles escalation."""
    
    def __init__(self, escalation_phone: str = "+4794810537"):
        self.db = get_tickets_db()
        self.escalation_phone = escalation_phone
        
    def get_sla_thresholds(self) -> Dict[str, Dict[str, int]]:
        """Get SLA thresholds in seconds for each priority level."""
        return {
            'P0': {'response': 5 * 60, 'resolution': 30 * 60},      # 5 min, 30 min
            'P1': {'response': 15 * 60, 'resolution': 2 * 3600},    # 15 min, 2 hours
            'P2': {'response': 60 * 60, 'resolution': 8 * 3600},    # 1 hour, 8 hours
            'P3': {'response': 4 * 3600, 'resolution': 24 * 3600},  # 4 hours, 24 hours
        }
    
    def check_active_sla_breaches(self) -> List[Dict[str, Any]]:
        """Check for tickets that are currently breaching their SLA."""
        import sqlite3
        
        breaches = []
        now = datetime.now(timezone.utc)
        
        with sqlite3.connect(self.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get tickets that haven't been resolved and might be breaching SLA
            cursor = conn.execute("""
                SELECT *,
                       (julianday(?) - julianday(created_at)) * 86400 as elapsed_seconds
                FROM tickets 
                WHERE status != 'RESOLVED' 
                  AND status != 'CLOSED'
                  AND status != 'DUPLICATE'
                ORDER BY created_at ASC
            """, (now.isoformat(),))
            
            for row in cursor.fetchall():
                ticket = dict(row)
                elapsed = ticket['elapsed_seconds']
                priority = ticket['priority']
                
                # Check against response SLA
                response_sla = ticket['response_sla_seconds']
                resolution_sla = ticket['resolution_sla_seconds']
                
                breach_info = {
                    'ticket_id': ticket['id'],
                    'ticket_number': ticket['ticket_number'],
                    'priority': priority,
                    'status': ticket['status'],
                    'elapsed_seconds': elapsed,
                    'response_sla_seconds': response_sla,
                    'resolution_sla_seconds': resolution_sla,
                    'message': ticket['message_text'][:100],
                    'breach_type': None,
                    'breach_amount_seconds': 0
                }
                
                # Check for response SLA breach (if not acknowledged yet)
                if not ticket['acknowledged_at'] and elapsed > response_sla:
                    breach_info['breach_type'] = 'response'
                    breach_info['breach_amount_seconds'] = elapsed - response_sla
                    breaches.append(breach_info.copy())
                
                # Check for resolution SLA breach (if not resolved yet)
                elif ticket['status'] != 'RESOLVED' and elapsed > resolution_sla:
                    breach_info['breach_type'] = 'resolution'
                    breach_info['breach_amount_seconds'] = elapsed - resolution_sla
                    breaches.append(breach_info.copy())
        
        return breaches
    
    def get_sla_compliance_stats(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get SLA compliance statistics for the past N hours."""
        import sqlite3
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        with sqlite3.connect(self.db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get resolved tickets from the time period
            cursor = conn.execute("""
                SELECT *,
                       (julianday(resolved_at) - julianday(created_at)) * 86400 as response_time_seconds
                FROM tickets 
                WHERE status = 'RESOLVED'
                  AND created_at >= ?
                ORDER BY created_at DESC
            """, (cutoff_time.isoformat(),))
            
            tickets = [dict(row) for row in cursor.fetchall()]
        
        if not tickets:
            return {
                'period_hours': hours_back,
                'total_tickets': 0,
                'sla_compliance_rate': 1.0,
                'by_priority': {},
                'avg_response_time_minutes': 0
            }
        
        # Calculate compliance by priority
        by_priority = {}
        total_compliant = 0
        total_response_time = 0
        
        for ticket in tickets:
            priority = ticket['priority']
            response_time = ticket['response_time_seconds']
            response_sla = ticket['response_sla_seconds']
            
            if priority not in by_priority:
                by_priority[priority] = {
                    'count': 0,
                    'compliant': 0,
                    'avg_response_time_minutes': 0,
                    'compliance_rate': 0.0
                }
            
            by_priority[priority]['count'] += 1
            total_response_time += response_time
            
            if response_time <= response_sla:
                by_priority[priority]['compliant'] += 1
                total_compliant += 1
        
        # Calculate rates and averages
        total_tickets = len(tickets)
        overall_compliance = total_compliant / total_tickets if total_tickets > 0 else 1.0
        avg_response_time = (total_response_time / total_tickets / 60) if total_tickets > 0 else 0
        
        for priority_stats in by_priority.values():
            count = priority_stats['count']
            priority_stats['compliance_rate'] = priority_stats['compliant'] / count if count > 0 else 1.0
        
        return {
            'period_hours': hours_back,
            'total_tickets': total_tickets,
            'sla_compliance_rate': overall_compliance,
            'by_priority': by_priority,
            'avg_response_time_minutes': avg_response_time
        }
    
    def send_escalation_alert(self, breach: Dict[str, Any]) -> bool:
        """Send escalation alert for SLA breach."""
        try:
            ticket_num = breach['ticket_number']
            priority = breach['priority']
            breach_type = breach['breach_type']
            breach_mins = breach['breach_amount_seconds'] // 60
            message = breach['message']
            
            alert_text = (
                f"🚨 SLA BREACH ALERT\n\n"
                f"Ticket #{ticket_num} ({priority})\n"
                f"Breach: {breach_type.title()} SLA exceeded by {breach_mins} minutes\n\n"
                f"Message: {message[:80]}...\n\n"
                f"Immediate attention required."
            )
            
            result = subprocess.run(
                ["imsg", "send", "--to", self.escalation_phone, "--text", alert_text],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.warning(f"📱 SLA escalation sent for ticket #{ticket_num}")
                return True
            else:
                logger.error(f"❌ Failed to send SLA escalation: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending escalation alert: {e}")
            return False
    
    async def monitor_sla_compliance(self) -> Dict[str, Any]:
        """Monitor SLA compliance and handle escalations."""
        # Check for active breaches
        breaches = self.check_active_sla_breaches()
        
        escalation_results = []
        critical_breaches = []
        
        for breach in breaches:
            # Critical: P0 tickets breaching resolution SLA
            if breach['priority'] == 'P0' and breach['breach_type'] == 'resolution':
                critical_breaches.append(breach)
                
                # Send immediate escalation
                escalation_sent = self.send_escalation_alert(breach)
                escalation_results.append({
                    'ticket_number': breach['ticket_number'],
                    'escalation_sent': escalation_sent
                })
        
        # Get compliance stats
        compliance_stats = self.get_sla_compliance_stats(hours_back=24)
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'active_breaches': len(breaches),
            'critical_breaches': len(critical_breaches),
            'escalations_sent': len([r for r in escalation_results if r['escalation_sent']]),
            'compliance_stats': compliance_stats,
            'breach_details': breaches[:10]  # First 10 breaches for debugging
        }
    
    async def start_monitoring_loop(self, check_interval: float = 300):  # 5 minutes
        """Start continuous SLA monitoring."""
        logger.info("🔍 SLA monitoring started")
        
        while True:
            try:
                monitoring_result = await self.monitor_sla_compliance()
                
                if monitoring_result['active_breaches'] > 0:
                    logger.warning(
                        f"⚠️ SLA status: {monitoring_result['active_breaches']} active breaches, "
                        f"{monitoring_result['critical_breaches']} critical, "
                        f"{monitoring_result['escalations_sent']} escalations sent"
                    )
                else:
                    logger.debug("✅ SLA status: All tickets within SLA")
                
                # Wait for next check
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"❌ SLA monitoring error: {e}")
                await asyncio.sleep(check_interval)

# Global monitor instance
_sla_monitor: Optional[SLAMonitoringService] = None

def get_sla_monitor() -> SLAMonitoringService:
    """Get the global SLA monitor instance."""
    global _sla_monitor
    if _sla_monitor is None:
        _sla_monitor = SLAMonitoringService()
    return _sla_monitor

async def start_sla_monitoring(check_interval: float = 300):
    """Start the global SLA monitor."""
    monitor = get_sla_monitor()
    await monitor.start_monitoring_loop(check_interval)