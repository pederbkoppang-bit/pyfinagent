"""
OpenClaw Session Monitor — feeds OpenClaw activity to MAS Dashboard.

Polls OpenClaw sessions every 30 seconds and emits events for new activity.
This bridges OpenClaw-native agents with the existing MAS event system.
"""

import asyncio
import logging
import time
from typing import Set
import httpx

from .openclaw_client import list_openclaw_sessions
from .mas_events import get_event_bus, MASEvent

logger = logging.getLogger(__name__)


class OpenClawSessionMonitor:
    """Monitor OpenClaw sessions and emit events for dashboard visibility."""
    
    def __init__(self):
        self.seen_sessions: Set[str] = set()
        self.last_token_counts = {}
        self.running = False

    async def start_monitoring(self, interval_seconds: int = 30):
        """Start monitoring OpenClaw sessions for activity."""
        self.running = True
        logger.info(f"🔍 Starting OpenClaw session monitor (every {interval_seconds}s)")
        
        while self.running:
            try:
                await self._poll_sessions()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Session monitor error: {e}")
                await asyncio.sleep(10)  # Shorter retry on error

    async def stop(self):
        """Stop the session monitor."""
        self.running = False

    async def _poll_sessions(self):
        """Poll OpenClaw sessions and emit events for new activity."""
        try:
            sessions = list_openclaw_sessions()
            bus = get_event_bus()
            
            for session in sessions:
                session_key = session.get("key", "")
                if not session_key:
                    continue
                
                # Extract agent info from session key
                agent_id = self._extract_agent_id(session_key)
                if not agent_id:
                    continue
                
                # Check if this is a new session
                if session_key not in self.seen_sessions:
                    self.seen_sessions.add(session_key)
                    await self._emit_session_start(bus, agent_id, session)
                
                # Check for token usage changes (indicating activity)
                current_tokens = self._extract_token_count(session)
                if current_tokens > 0:
                    last_tokens = self.last_token_counts.get(session_key, 0)
                    if current_tokens > last_tokens:
                        await self._emit_session_activity(bus, agent_id, session, current_tokens - last_tokens)
                        self.last_token_counts[session_key] = current_tokens
                
        except Exception as e:
            logger.error(f"Failed to poll OpenClaw sessions: {e}")

    def _extract_agent_id(self, session_key: str) -> str:
        """Extract agent ID from session key like 'agent:pyfinagent:slack:direct:u...'"""
        parts = session_key.split(":")
        if len(parts) >= 2 and parts[0] == "agent":
            return parts[1]
        return ""

    def _extract_token_count(self, session: dict) -> int:
        """Extract total token count from session data."""
        try:
            # Look for token info in session data
            tokens_str = session.get("tokens", "")
            if isinstance(tokens_str, str) and "(" in tokens_str:
                # Parse "19k/200k (9%)" format
                parts = tokens_str.split("(")[0].strip()
                if "/" in parts:
                    used_part = parts.split("/")[0].strip()
                    if used_part.endswith("k"):
                        return int(float(used_part[:-1]) * 1000)
                    return int(used_part)
            return 0
        except Exception:
            return 0

    async def _emit_session_start(self, bus, agent_id: str, session: dict):
        """Emit event for new session start."""
        event = MASEvent(
            event_type="session_start",
            agent=agent_id,
            run_id=session.get("key", "")[:8],
            data={
                "session_key": session.get("key", ""),
                "model": session.get("model", ""),
                "kind": session.get("kind", ""),
                "channel": session.get("channel", ""),
            },
            duration_ms=0,
            tokens={"total": 0},
            iteration=0,
        )
        bus.emit(event)
        logger.info(f"📍 Session start: {agent_id} ({session.get('model', '')})")

    async def _emit_session_activity(self, bus, agent_id: str, session: dict, new_tokens: int):
        """Emit event for session activity (new tokens)."""
        event = MASEvent(
            event_type="message",
            agent=agent_id,
            run_id=session.get("key", "")[:8],
            data={
                "session_key": session.get("key", ""),
                "activity": "token_usage",
                "new_tokens": new_tokens,
            },
            duration_ms=0,
            tokens={"completion": new_tokens, "total": new_tokens},
            iteration=0,
        )
        bus.emit(event)
        logger.info(f"🔄 Activity: {agent_id} +{new_tokens} tokens")


# Global monitor instance
_monitor = None


async def start_openclaw_monitor():
    """Start the global OpenClaw session monitor."""
    global _monitor
    if _monitor and _monitor.running:
        return
    
    _monitor = OpenClawSessionMonitor()
    await _monitor.start_monitoring()


def get_monitor() -> OpenClawSessionMonitor:
    """Get the global monitor instance."""
    global _monitor
    if not _monitor:
        _monitor = OpenClawSessionMonitor()
    return _monitor


async def inject_test_events():
    """Inject some test events to verify dashboard functionality."""
    bus = get_event_bus()
    
    # Test events for different agents with complete dashboard fields
    test_events = [
        MASEvent(
            event_type="classify",
            agent="Communication",
            run_id="test001",
            data={
                "action": "route_query", 
                "query": "Test Slack message",
                "route_to": "main",
                "complexity": "simple",
                "steps": ["classify", "route"]
            },
            duration_ms=1200,
            tokens={"input": 100, "output": 50, "total": 150},
            iteration=1,
        ),
        MASEvent(
            event_type="delegate",
            agent="Ford (Main)",
            run_id="test001",
            data={
                "action": "spawn_subagent", 
                "target_agent": "QA", 
                "task": "Analyze current harness performance",
                "complexity": "medium",
                "steps": ["plan", "delegate", "spawn"]
            },
            duration_ms=800,
            tokens={"input": 200, "output": 150, "total": 350},
            iteration=1,
        ),
        MASEvent(
            event_type="complete",
            agent="QA Analyst",
            run_id="test001",
            data={
                "action": "analysis_complete", 
                "result": "Performance analysis: Sharpe 1.17, DSR 0.95",
                "complexity": "high",
                "steps": ["analyze", "validate", "report"]
            },
            duration_ms=65000,
            tokens={"input": 1500, "output": 2000, "total": 3500},
            iteration=1,
        ),
    ]
    
    for event in test_events:
        bus.emit(event)
        await asyncio.sleep(0.1)
    
    logger.info("📊 Injected test events for dashboard")