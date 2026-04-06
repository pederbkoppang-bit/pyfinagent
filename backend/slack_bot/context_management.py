"""
Slack AI Agent — Phase 5: Context Management

Implements smart workspace search + structured state management.
Uses assistant.search.context() API for semantic search.
Maintains structured state: {goal, constraints, decisions, artifacts, sources}

Reference: https://docs.slack.dev/ai/agent-context-management
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

from slack_sdk import WebClient

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Structured state for multi-turn conversations"""
    goal: str  # User's current objective
    constraints: str  # Date range, channel scope, filters
    decisions: List[str] = field(default_factory=list)  # Key decisions
    artifacts: List[Dict[str, Any]] = field(default_factory=list)  # Outputs created
    sources: List[Dict[str, str]] = field(default_factory=list)  # [{"text": "...", "url": "..."}]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_prompt(self) -> str:
        """Convert state to LLM prompt context"""
        prompt = f"GOAL: {self.goal}\n"
        if self.constraints:
            prompt += f"CONSTRAINTS: {self.constraints}\n"
        if self.decisions:
            prompt += f"DECISIONS:\n" + "\n".join(f"- {d}" for d in self.decisions) + "\n"
        if self.sources:
            prompt += f"SOURCES:\n" + "\n".join(f"- {s['text']}: {s['url']}" for s in self.sources) + "\n"
        return prompt


class ContextManager:
    """Manages workspace search + state for agents"""
    
    def __init__(self, client: WebClient):
        self.client = client
        self.logger = logging.getLogger(__name__)
    
    async def search_workspace(
        self,
        query: str,
        action_token: str,
        content_types: Optional[List[str]] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search workspace using assistant.search.context API.
        
        Args:
            query: Natural language search query
            action_token: From triggering event (required for user context)
            content_types: ["messages", "files", "channels", "canvases"]
            limit: Number of results to return
            
        Returns:
            Search results with messages, files, channels
        """
        
        if not action_token:
            self.logger.warning("⚠️ No action_token, cannot search workspace")
            return {"results": {}}
        
        try:
            # Query workspace for context
            result = await self.client.assistant_search_context(
                query=query,
                action_token=action_token,
                content_types=content_types or ["messages", "files", "channels"],
                channel_types=["public_channel", "private_channel"],
                include_context_messages=True,
                limit=limit
            )
            
            self.logger.info(f"✅ Workspace search complete: {len(result.get('results', {}))} results")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Workspace search failed: {e}")
            return {"results": {}}
    
    async def get_thread_context(
        self,
        channel_id: str,
        thread_ts: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get full thread context using conversations.replies.
        
        Useful for drilling into specific thread from search results.
        """
        
        try:
            result = await self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=limit
            )
            
            messages = result.get("messages", [])
            self.logger.info(f"✅ Thread context: {len(messages)} messages")
            return messages
            
        except Exception as e:
            self.logger.error(f"❌ Get thread context failed: {e}")
            return []
    
    async def get_channel_context(
        self,
        channel_id: str,
        days_back: int = 7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get channel history using conversations.history.
        
        Useful for understanding channel context.
        """
        
        try:
            import time
            oldest = str(int(time.time()) - (days_back * 24 * 60 * 60))
            
            result = await self.client.conversations_history(
                channel=channel_id,
                oldest=oldest,
                limit=limit
            )
            
            messages = result.get("messages", [])
            self.logger.info(f"✅ Channel context: {len(messages)} messages")
            return messages
            
        except Exception as e:
            self.logger.error(f"❌ Get channel context failed: {e}")
            return []
    
    def build_initial_state(
        self,
        user_query: str,
        search_results: Optional[Dict[str, Any]] = None,
        thread_context: Optional[List[Dict[str, Any]]] = None
    ) -> AgentState:
        """
        Build structured state from search results.
        
        Pattern:
        1. Parse user goal from query
        2. Extract sources from search results
        3. Initialize decisions/artifacts as empty
        4. Ready for LLM to update iteratively
        """
        
        state = AgentState(goal=user_query)
        
        # Extract sources from search results
        if search_results and "results" in search_results:
            results = search_results["results"]
            
            # Messages
            for msg in results.get("messages", []):
                state.sources.append({
                    "text": msg.get("content", "")[:100],
                    "url": msg.get("permalink", "")
                })
            
            # Files
            for file in results.get("files", []):
                state.sources.append({
                    "text": file.get("name", ""),
                    "url": file.get("permalink_public", "")
                })
        
        # Add thread context if available
        if thread_context:
            state.constraints = f"From thread with {len(thread_context)} messages"
        
        self.logger.info(f"✅ State initialized: goal={user_query[:50]}, sources={len(state.sources)}")
        return state
    
    def update_state(
        self,
        state: AgentState,
        decisions: Optional[List[str]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[Dict[str, str]]] = None
    ) -> AgentState:
        """
        Update state after LLM processing.
        
        Pattern: Don't refetch, just update iteratively.
        """
        
        if decisions:
            state.decisions.extend(decisions)
        if artifacts:
            state.artifacts.extend(artifacts)
        if sources:
            state.sources.extend(sources)
        
        return state


class ContextAwareResponseBuilder:
    """Builds LLM prompts with proper context"""
    
    @staticmethod
    def build_system_prompt() -> str:
        """System prompt for Slack AI agent"""
        return """You are PyFinAgent, an AI financial analyst for Slack.

You have access to Slack MCP tools:
- search_messages: Search workspace messages
- search_channels: Find channels
- post_message: Reply in Slack
- read_thread: Get thread context

When responding:
1. Ground answers in workspace context (use search tools)
2. Cite sources (include links)
3. Keep responses concise for Slack
4. Use emoji for emphasis
5. Format with markdown for readability"""
    
    @staticmethod
    def build_user_prompt(state: AgentState) -> str:
        """User prompt with structured state"""
        return f"""User Query: {state.goal}

Available Context:
{state.to_prompt()}

Instructions:
1. Use MCP tools to search workspace if needed
2. Synthesize answer from context
3. Cite sources
4. Be concise"""
