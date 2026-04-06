"""
Slack AI Agent — Assistant Lifecycle Handlers (Phase 2)

Implements three core Slack Assistant events:
1. assistant_thread_started — User opens agent container
2. assistant_thread_context_changed — User switches channels
3. message.im — User sends message (core response loop)

Reference: https://docs.slack.dev/ai/developing-agents
"""

import logging
from typing import Dict, Any, Optional

from slack_bolt import BoltRequest, BoltResponse, Say, SetStatus, Ack
from slack_sdk import WebClient

logger = logging.getLogger(__name__)


class AssistantLifecycleHandler:
    """Manages assistant lifecycle: startup → context → messages"""
    
    def __init__(self, client: WebClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
    
    async def handle_thread_started(
        self,
        body: Dict[str, Any],
        say: Say,
        set_suggested_prompts,
        logger: logging.Logger
    ) -> None:
        """
        Handle assistant_thread_started event.
        
        Fired when user opens the agent container.
        Tasks:
        1. Send welcome message
        2. Set suggested prompts
        3. Initialize thread state
        """
        
        channel_id = body.get("assistant_thread", {}).get("channel_id")
        thread_ts = body.get("assistant_thread", {}).get("thread_ts")
        
        logger.info(f"🔄 Thread started: channel={channel_id}, thread_ts={thread_ts}")
        
        try:
            # Send welcome message
            await say({
                "text": "👋 Welcome to PyFinAgent — Your AI Financial Analyst\n\n"
                        "I can help you:\n"
                        "• Analyze stocks with evidence-based methods\n"
                        "• Backtest trading strategies\n"
                        "• Research financial topics\n"
                        "• Review portfolio risk\n\n"
                        "Pick a suggested prompt below or ask me anything!"
            })
            
            # Set suggested prompts (context-aware)
            await set_suggested_prompts({
                "prompts": [
                    {
                        "title": "Analyze a stock",
                        "message": "Analyze AAPL using our framework. What's the investment thesis?"
                    },
                    {
                        "title": "Backtest a strategy",
                        "message": "Backtest mean reversion in tech. What's the Sharpe ratio over the last 2 years?"
                    },
                    {
                        "title": "Research topic",
                        "message": "What's the latest research on AI-driven factor selection?"
                    },
                    {
                        "title": "Portfolio review",
                        "message": "Review our current portfolio allocation and risk profile."
                    }
                ]
            })
            
            logger.info(f"✅ Thread started: welcome + prompts sent")
            
        except Exception as e:
            logger.error(f"❌ handle_thread_started failed: {e}")
            raise
    
    async def handle_context_changed(
        self,
        body: Dict[str, Any],
        logger: logging.Logger
    ) -> None:
        """
        Handle assistant_thread_context_changed event.
        
        Fired when user switches channels while container is open.
        Tasks:
        1. Track new channel context
        2. Update thread state if needed
        """
        
        context = body.get("assistant_thread", {}).get("context", {})
        channel_id = context.get("channel_id")
        team_id = context.get("team_id")
        
        logger.info(f"🔄 Context changed: channel={channel_id}")
        
        # Store context for use in message handler
        # (In production, store in Redis or thread-local state)
        
        logger.info(f"✅ Context updated: {channel_id}")
    
    async def handle_user_message(
        self,
        body: Dict[str, Any],
        client: WebClient,
        say: Say,
        set_status,
        logger: logging.Logger
    ) -> None:
        """
        Handle message.im event (user sends message in thread).
        
        Core response loop:
        1. Extract user message + thread context
        2. Set status ("Thinking...")
        3. Call LLM with message + context
        4. Stream response with task updates
        5. Clear status
        
        Reference: https://docs.slack.dev/ai/developing-agents#respond-to-the-message-im-event
        """
        
        try:
            # Extract message details
            message = body.get("event", {})
            channel_id = message.get("channel")
            thread_ts = message.get("thread_ts") or message.get("ts")
            user_id = message.get("user")
            user_text = message.get("text", "").strip()
            action_token = message.get("action_token")  # For workspace search
            
            logger.info(f"💬 Message: user={user_id}, text={user_text[:50]}")
            
            # Set visible status
            await set_status({"status": "Thinking..."})
            
            # Phase 3: Use streaming integration with task cards
            from backend.slack_bot.streaming_integration import handle_user_message_with_streaming
            
            await handle_user_message_with_streaming(
                body=body,
                client=client,
                say=say,
                set_status=set_status,
                logger=logger
            )
            
            # Clear status (handled in streaming_integration)
            await set_status({"status": ""})
            
            logger.info(f"✅ Message handled")
            
        except Exception as e:
            logger.error(f"❌ handle_user_message failed: {e}")
            await set_status({"status": ""})  # Clear status on error
            raise


def register_assistant_lifecycle(app):
    """Register assistant lifecycle handlers with Bolt app."""
    from slack_bolt.middleware.assistant.async_assistant import AsyncAssistant

    handler = AssistantLifecycleHandler(app.client)
    assistant = AsyncAssistant()

    @assistant.thread_started
    async def on_thread_started(say, set_suggested_prompts, get_thread_context, logger):
        await handler.handle_thread_started(
            body={}, client=app.client, say=say,
            set_status=lambda *a, **kw: None, logger=logger,
        )

    @assistant.thread_context_changed
    async def on_context_changed(body, logger):
        await handler.handle_context_changed(
            body=body, client=app.client, say=lambda *a, **kw: None,
            set_status=lambda *a, **kw: None, logger=logger,
        )

    @assistant.user_message
    async def on_user_message(body, client, say, set_status, logger):
        await handler.handle_user_message(
            body=body, client=client, say=say,
            set_status=set_status, logger=logger,
        )

    app.use(assistant)
    logger.info("✅ Assistant lifecycle handlers registered")
