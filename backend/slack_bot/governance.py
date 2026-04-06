"""
Slack AI Agent — Phase 6: Governance & Human-in-the-Loop

Implements:
1. Audit logging (request tracking, model usage, costs)
2. Human-in-the-loop controls (approval gates, undo/redo)
3. Content disclaimers (AI transparency)
4. Rate limiting + safeguards

Reference: https://docs.slack.dev/ai/agent-governance
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AuditLog:
    """Record of each agent request for auditability"""
    
    request_id: str
    timestamp: str
    user_id: str
    channel_id: str
    thread_ts: str
    query: str
    agent_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency_ms: int
    outcome: str  # success | partial | failure
    error_msg: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "query": self.query[:100],
            "agent_id": self.agent_id,
            "model": self.model,
            "tokens": self.total_tokens,
            "cost_usd": self.cost,
            "latency_ms": self.latency_ms,
            "outcome": self.outcome
        }


class AuditLogger:
    """Log all agent requests for governance + debugging"""
    
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__)
        self.audit_logs = []  # In production: persist to BQ
    
    def log_request(
        self,
        request_id: str,
        user_id: str,
        channel_id: str,
        thread_ts: str,
        query: str,
        agent_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        outcome: str,
        error_msg: Optional[str] = None
    ) -> None:
        """Log a request"""
        
        total_tokens = input_tokens + output_tokens
        # Rough cost estimate (update for actual pricing)
        cost = (input_tokens * 0.00001 + output_tokens * 0.00003) / 1000
        
        log = AuditLog(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            user_id=user_id,
            channel_id=channel_id,
            thread_ts=thread_ts,
            query=query,
            agent_id=agent_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency_ms=latency_ms,
            outcome=outcome,
            error_msg=error_msg
        )
        
        self.audit_logs.append(log)
        self.logger.info(f"📊 Audit: {agent_id} - {outcome} ({total_tokens} tokens)")
    
    def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get aggregate stats"""
        
        recent = [l for l in self.audit_logs]  # TODO: filter by time
        
        if not recent:
            return {}
        
        total_tokens = sum(l.total_tokens for l in recent)
        total_cost = sum(l.cost for l in recent)
        success_count = len([l for l in recent if l.outcome == "success"])
        
        return {
            "requests": len(recent),
            "success_rate": success_count / len(recent),
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "avg_latency_ms": sum(l.latency_ms for l in recent) / len(recent)
        }


class HumanInTheLoopManager:
    """Controls for user approval + undo/redo"""
    
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__)
    
    async def send_approval_gate(
        self,
        channel_id: str,
        thread_ts: str,
        action: str,
        description: str
    ) -> None:
        """
        Send approval request for high-impact action.
        
        Examples:
        - Create canvas
        - Delete message
        - Post to channel (not thread)
        """
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⚠️ *Approval Required*\n\nAction: {action}\n{description}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "value": "approve",
                        "style": "primary",
                        "action_id": "approval_approve"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "value": "deny",
                        "style": "danger",
                        "action_id": "approval_deny"
                    }
                ]
            }
        ]
        
        await self.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            blocks=blocks
        )
        
        self.logger.info(f"⚠️ Approval gate sent: {action}")
    
    async def send_next_steps(
        self,
        channel_id: str,
        thread_ts: str,
        options: list
    ) -> None:
        """
        Send action buttons for next steps.
        
        Options: ["Refine", "Redo", "Share", "Archive"]
        """
        
        elements = []
        for opt in options:
            elements.append({
                "type": "button",
                "text": {"type": "plain_text", "text": opt},
                "value": opt.lower(),
                "action_id": f"next_step_{opt.lower()}"
            })
        
        blocks = [
            {
                "type": "actions",
                "elements": elements
            }
        ]
        
        await self.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            blocks=blocks
        )
        
        self.logger.info(f"➡️ Next steps offered: {options}")


class ContentDisclaimer:
    """Add AI transparency disclaimer to responses"""
    
    @staticmethod
    def get_disclaimer() -> Dict[str, Any]:
        """Return disclaimer block for messages"""
        
        return {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "🤖 *AI Generated*: This content was generated by AI. " +
                            "Check for accuracy before acting on it. " +
                            "It may contain bias or hallucinations."
                }
            ]
        }
    
    @staticmethod
    def add_disclaimer_to_blocks(blocks: list) -> list:
        """Add disclaimer to message blocks"""
        return blocks + [ContentDisclaimer.get_disclaimer()]


class RateLimiter:
    """Prevent abuse + manage API quotas"""
    
    def __init__(self, max_requests_per_hour: int = 60):
        self.max_requests = max_requests_per_hour
        self.user_requests = {}  # user_id -> [timestamp, timestamp, ...]
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is within rate limit"""
        
        import time
        now = time.time()
        
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        # Remove old requests (>1 hour)
        self.user_requests[user_id] = [
            t for t in self.user_requests[user_id]
            if now - t < 3600
        ]
        
        # Check limit
        if len(self.user_requests[user_id]) >= self.max_requests:
            return False
        
        # Record request
        self.user_requests[user_id].append(now)
        return True
    
    def get_remaining(self, user_id: str) -> int:
        """Get remaining requests for user"""
        if user_id not in self.user_requests:
            return self.max_requests
        return max(0, self.max_requests - len(self.user_requests[user_id]))


# Integration functions

async def log_agent_response(
    audit_logger: AuditLogger,
    user_id: str,
    channel_id: str,
    thread_ts: str,
    query: str,
    agent_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    outcome: str,
    error_msg: Optional[str] = None
) -> None:
    """Log response for audit trail"""
    
    import uuid
    request_id = str(uuid.uuid4())
    
    audit_logger.log_request(
        request_id=request_id,
        user_id=user_id,
        channel_id=channel_id,
        thread_ts=thread_ts,
        query=query,
        agent_id=agent_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        outcome=outcome,
        error_msg=error_msg
    )
