"""
Slack AI Agent — Streaming Response Handler (Phase 3)

Implements official Slack streaming API:
- chat.startStream() — Begin streaming response
- chat.appendStream() — Add chunks (markdown + task updates)
- chat.stopStream() — Finalize response

Task display modes:
- "plan" — Grouped tasks (multi-agent workflow visualization)
- "timeline" — Sequential step-by-step tasks

Reference: https://docs.slack.dev/ai/developing-agents#text-streaming
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from slack_sdk import WebClient
from slack_sdk.models.messages.chunk import (
    MarkdownTextChunk,
    TaskUpdateChunk,
    TaskStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskUpdate:
    """Represents a task update for streaming"""
    id: str
    title: str
    status: str  # pending | in_progress | complete | error
    details: Optional[str] = None  # Markdown, what agent is doing
    output: Optional[str] = None  # Final result text
    sources: Optional[List[Dict[str, str]]] = None  # [{"type": "url", "url": "...", "text": "..."}]


class StreamingHandler:
    """Manages streaming responses with task cards"""
    
    def __init__(self, client: WebClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
    
    async def stream_response(
        self,
        channel_id: str,
        thread_ts: str,
        task_display_mode: str = "plan",
        tasks: Optional[List[TaskUpdate]] = None,
        initial_text: str = "Working on your request..."
    ) -> str:
        """
        Start streaming response with task updates.
        
        Args:
            channel_id: Slack channel ID
            thread_ts: Thread timestamp
            task_display_mode: "plan" (grouped) or "timeline" (sequential)
            tasks: Initial tasks to display
            initial_text: Opening text
            
        Returns:
            Message timestamp (ts) for appending/stopping
        """
        
        try:
            # Start the stream
            response = await self.client.chat_startStream(
                channel=channel_id,
                thread_ts=thread_ts,
                task_display_mode=task_display_mode,
                chunks=[
                    MarkdownTextChunk(markdown_text=initial_text)
                ]
            )
            
            message_ts = response.get("ts")
            self.logger.info(f"✅ Stream started: {message_ts}")
            
            # Send initial tasks if provided
            if tasks:
                await self.append_tasks(channel_id, message_ts, thread_ts, tasks)
            
            return message_ts
            
        except Exception as e:
            self.logger.error(f"❌ stream_response failed: {e}")
            raise
    
    async def append_tasks(
        self,
        channel_id: str,
        message_ts: str,
        thread_ts: str,
        tasks: List[TaskUpdate]
    ) -> None:
        """
        Append task updates to streaming message.
        
        Each task shows: pending → in_progress → complete
        """
        
        try:
            # Convert TaskUpdate objects to Slack chunk format
            chunks = []
            for task in tasks:
                chunk = TaskUpdateChunk(
                    id=task.id,
                    title=task.title,
                    status=task.status,
                    details=task.details,
                    output=task.output,
                    sources=task.sources
                )
                chunks.append(chunk)
            
            await self.client.chat_appendStream(
                channel=channel_id,
                ts=message_ts,
                thread_ts=thread_ts,
                chunks=chunks
            )
            
            self.logger.info(f"✅ Tasks appended: {len(tasks)} tasks")
            
        except Exception as e:
            self.logger.error(f"❌ append_tasks failed: {e}")
            raise
    
    async def append_text(
        self,
        channel_id: str,
        message_ts: str,
        thread_ts: str,
        text: str
    ) -> None:
        """Append markdown text to streaming message"""
        
        try:
            await self.client.chat_appendStream(
                channel=channel_id,
                ts=message_ts,
                thread_ts=thread_ts,
                chunks=[MarkdownTextChunk(markdown_text=text)]
            )
            
        except Exception as e:
            self.logger.error(f"❌ append_text failed: {e}")
            raise
    
    async def stop_stream(
        self,
        channel_id: str,
        message_ts: str,
        thread_ts: str,
        final_text: str = "Done!",
        blocks: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Finalize streaming response.
        
        Args:
            channel_id: Slack channel ID
            message_ts: Stream message timestamp
            thread_ts: Thread timestamp
            final_text: Closing text
            blocks: Optional Block Kit blocks (feedback buttons, etc.)
        """
        
        try:
            chunks = [MarkdownTextChunk(markdown_text=final_text)]
            
            stop_kwargs = {
                "channel": channel_id,
                "ts": message_ts,
                "thread_ts": thread_ts,
                "chunks": chunks
            }
            
            if blocks:
                stop_kwargs["blocks"] = blocks
            
            await self.client.chat_stopStream(**stop_kwargs)
            
            self.logger.info(f"✅ Stream stopped: {message_ts}")
            
        except Exception as e:
            self.logger.error(f"❌ stop_stream failed: {e}")
            raise
    
    async def update_task_status(
        self,
        channel_id: str,
        message_ts: str,
        thread_ts: str,
        task_id: str,
        status: str,
        output: Optional[str] = None
    ) -> None:
        """
        Update single task status in streaming message.
        
        Common pattern:
        1. Create task with status="pending"
        2. Update to status="in_progress" as agent works
        3. Update to status="complete" with final output
        """
        
        task = TaskUpdate(
            id=task_id,
            status=status,
            title="",  # Title should be set initially
            output=output
        )
        
        await self.append_tasks(channel_id, message_ts, thread_ts, [task])


# Predefined task templates

TASK_TEMPLATES = {
    "operational": TaskUpdate(
        id="agent_operational",
        title="Ford — Checking System Status",
        status="pending",
        details="- Scanning service health\n- Checking git status\n- Evaluating configuration"
    ),
    "research": TaskUpdate(
        id="agent_research",
        title="Research Agent — Literature Search",
        status="pending",
        details="- Scanning arXiv, SSRN\n- Cross-referencing papers\n- Extracting methods"
    ),
    "analyst": TaskUpdate(
        id="agent_analyst",
        title="Analyst — Quantitative Review",
        status="pending",
        details="- Computing metrics\n- Evaluating robustness\n- Assessing risk"
    ),
}
