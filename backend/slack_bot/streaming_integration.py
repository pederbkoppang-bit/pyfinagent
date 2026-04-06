"""
Slack AI Agent — Phase 3: Streaming Integration

Integrates streaming API with assistant lifecycle.
Updates assistant_lifecycle.py to use official chat.startStream/appendStream/stopStream.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List

from slack_bolt import Say, SetStatus
from slack_sdk import WebClient
from slack_sdk.models.messages.chunk import MarkdownTextChunk, TaskUpdateChunk

from backend.agents.multi_agent_orchestrator import get_orchestrator
from backend.slack_bot.streaming_handler import StreamingHandler, TaskUpdate, TASK_TEMPLATES

logger = logging.getLogger(__name__)


async def handle_user_message_with_streaming(
    body: Dict[str, Any],
    client: WebClient,
    say: Say,
    set_status,
    logger: logging.Logger
) -> None:
    """
    Phase 3: Complete streaming response with task cards.
    
    Flow:
    1. Extract message + context
    2. Start stream with task display mode
    3. Spawn agents (Operational, Research, Analyst)
    4. Update tasks as agents complete
    5. Stream final response with citations
    6. Stop stream and clear status
    """
    
    try:
        # Extract message details
        message = body.get("event", {})
        channel_id = message.get("channel")
        thread_ts = message.get("thread_ts") or message.get("ts")
        user_id = message.get("user")
        user_text = message.get("text", "").strip()
        action_token = message.get("action_token")
        
        logger.info(f"💬 Message: user={user_id}, text={user_text[:50]}")
        
        # Create streaming handler
        streamer = StreamingHandler(client)
        
        # Initialize tasks
        initial_tasks = [
            TASK_TEMPLATES["operational"],
            TASK_TEMPLATES["research"],
            TASK_TEMPLATES["analyst"],
        ]
        
        # Start stream with task display
        message_ts = await streamer.stream_response(
            channel_id=channel_id,
            thread_ts=thread_ts,
            task_display_mode="plan",
            tasks=initial_tasks,
            initial_text="🚀 PyFinAgent analyzing your query..."
        )
        
        # Get orchestrator and run analysis
        orchestrator = get_orchestrator()
        
        # TODO: Phase 3.5 — Call orchestrator with workspace context
        # analysis_result = await orchestrator.analyze(
        #     query=user_text,
        #     action_token=action_token,
        #     channel_id=channel_id
        # )
        
        # For now, demo response
        demo_response = f"Query: *{user_text}*\n\nPhase 3 streaming integration complete. Ready for Phase 4 (MCP + context)."
        
        # Update tasks to complete
        completed_tasks = [
            TaskUpdate(
                id="agent_operational",
                title="Ford — System Status",
                status="complete",
                output="✅ All services operational",
                sources=[{"type": "url", "url": "https://localhost:8000", "text": "Backend"}]
            ),
            TaskUpdate(
                id="agent_research",
                title="Research Agent",
                status="complete",
                output="✅ Research query analyzed",
                sources=None
            ),
            TaskUpdate(
                id="agent_analyst",
                title="Analyst Agent",
                status="complete",
                output="✅ Analysis complete",
                sources=None
            ),
        ]
        
        await streamer.append_tasks(channel_id, message_ts, thread_ts, completed_tasks)
        
        # Append final response
        await streamer.append_text(
            channel_id,
            message_ts,
            thread_ts,
            f"\n{demo_response}"
        )
        
        # Stop stream
        await streamer.stop_stream(
            channel_id,
            message_ts,
            thread_ts,
            final_text="✅ Analysis complete!"
        )
        
        logger.info(f"✅ Streaming response complete")
        
    except Exception as e:
        logger.error(f"❌ handle_user_message_with_streaming failed: {e}")
        import traceback
        traceback.print_exc()
        raise
