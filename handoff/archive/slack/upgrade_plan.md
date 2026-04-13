# Slack AI Agent Framework Upgrade Plan

**Critical Issue:** Research and implement Slack's official AI Agent framework  
**Date:** 2026-04-05 16:18 GMT+2  
**Authority:** Peder B. Koppang  
**Status:** URGENT — Planning phase

---

## Executive Summary

Slack has released official **AI Agent framework** with native support for:
1. **Agent container** (split-pane UI with assistant entry point)
2. **Suggested prompts** (contextual quick-start suggestions)
3. **Streaming responses** (word-by-word LLM output with task visibility)
4. **MCP Server integration** (Model Context Protocol for tool access)
5. **Thread management** (automatic conversation organization)
6. **Context-aware responses** (workspace search + thread history)

**Current state:** We have a Slack bot with Socket Mode, but we're using legacy patterns.  
**Target state:** Full Slack AI Agent framework with native container, task streaming, and MCP tools.

---

## Key Slack AI Agent Features (From Docs)

### 1. **Agent Container Entry Point**
- Top bar "App" launch → split pane UI (native assistant panel)
- Enables: `Agents & AI Apps` feature in app settings
- Requires: `assistant:write` scope (auto-added when feature enabled)

### 2. **Assistant Lifecycle Events**
Three core events to implement:
```
assistant_thread_started
  → User opens container
  → Send welcome + suggested prompts (setSuggestedPrompts API)

assistant_thread_context_changed  
  → User switches channels while container open
  → Update state if context-grounded

message.im
  → User sends message in thread
  → Core response loop: status → LLM → stream → respond
```

### 3. **Response Loop Pattern**
```
1. User sends message
2. Call assistant.threads.setStatus("Thinking...")
3. Call chat.startStream() to begin streaming
4. Call LLM with message + context
5. For each LLM output chunk:
   - chat.appendStream() with task updates
   - Update task status: pending → in_progress → complete
6. Call chat.stopStream() to finalize
7. Call assistant.threads.setStatus("") to clear
```

### 4. **Streaming with Task Cards**
Task display modes:
- **plan:** Grouped tasks (multi-step workflow)
- **timeline:** Step-by-step sequential

Each task has:
- `id`: unique identifier
- `title`: human-readable label
- `status`: pending | in_progress | complete | error
- `details`: markdown (what agent is doing)
- `output`: final result text
- `sources`: [{type: "url", url, text}] citations

### 5. **Slack MCP Server**
Slack hosts MCP server at `https://mcp.slack.com/mcp`
- No separate server to run
- Pass to LLM as tool alongside standard tools
- Requires `mcp_servers` or `tools` field in LLM request

MCP tools available:
- `search_messages` / `search_channels` / `search_files` / `search_users`
- `post_message` / `create_canvas`
- `read_channel_history` / `read_thread`
- Full context gathering API

### 6. **Context Management**
Recommended pattern:
- Use `assistant.search.context()` API (action_token required)
- Pull workspace search results across messages, files, channels
- Drill into thread with `conversations.replies()`
- Build **structured state object**: {goal, constraints, decisions, artifacts, sources}
- Pass state to LLM, update iteratively (don't re-query everything each turn)

### 7. **Governance & Human-in-the-Loop**
Essential for enterprise:
- **Transparency:** Use `assistant.threads.setStatus()` for visible progress
- **Control:** Buttons/actions for user steering (pause, resume, redo, approve)
- **Auditability:** Log which agent handled request, models used, tokens consumed
- **Safeguards:** Require approval for high-impact actions (delete, modify, send)

---

## Current Implementation Gaps

### ✅ What We Have
- Slack bot running (Socket Mode, always-on)
- Command handlers (/analyze, /portfolio, /report)
- Ticket queue processor + SLA monitor
- Multi-agent architecture (Main, Q&A, Research, Direct)
- Custom message streaming (partial)

### ❌ What We're Missing
1. **Agent container entry point** — Not using top-bar agent launch
2. **Suggested prompts** — Not calling `setSuggestedPrompts` on thread start
3. **Official streaming API** — Using custom chunk format, not `chat.startStream/appendStream`
4. **Task card visibility** — Not showing agent work as task updates
5. **Slack MCP Server** — Not integrated with LLM calls
6. **Context management** — Not using `assistant.search.context()` for workspace search
7. **Thread management** — Not calling `assistant.threads.setTitle()` for organization
8. **Governance framework** — Limited audit logging, no human-in-the-loop controls

---

## Implementation Roadmap

### Phase 1: App Config & Enablement (1 hour)
**Goal:** Enable `Agents & AI Apps` feature in Slack app settings
**Steps:**
1. Go to https://api.slack.com/apps → pyfinAgent app
2. Navigate to `Agents & AI Apps` feature in sidebar
3. Toggle **ON** `Agents & AI Apps`
4. Verify `assistant:write` scope added to manifest
5. Add `assistant_view.assistant_description` to manifest
6. Verify these event subscriptions in app settings:
   - `assistant_thread_started`
   - `assistant_thread_context_changed`
   - `message.im`

**Deliverable:** App settings configured, feature enabled

### Phase 2: Assistant Lifecycle Implementation (2-3 hours)
**Goal:** Implement three core event handlers
**Changes to `app.py`:**
```python
# Bolt's Assistant class (handles lifecycle)
from slack_bolt.assistant import Assistant

assistant = Assistant(
    threadStarted=handle_thread_started,  # welcome + prompts
    threadContextChanged=handle_context_changed,  # track context
    userMessage=handle_user_message  # core response loop
)
app.use(assistant)
```

**Implementation details:**
- `handle_thread_started`: Send welcome, call `setSuggestedPrompts()` with 3-4 contextual prompts
- `handle_context_changed`: Update internal state with new channel context
- `handle_user_message`: Core loop (status → LLM → stream → respond)

**Deliverable:** Three event handlers fully implemented

### Phase 3: Streaming & Task Cards (2-3 hours)
**Goal:** Replace custom streaming with official Slack streaming API
**Changes:**
- Swap custom `send_chunk()` with `chat.startStream()` / `appendStream()` / `stopStream()`
- Emit task updates for each agent:
  1. Operational Agent task
  2. Research Agent task
  3. Analyst (Q&A) task
4. Update task status as agents complete
5. Show sources/citations in task output

**Code pattern:**
```python
# Start stream
stream_result = await client.chat.startStream(
    channel=channel_id,
    thread_ts=thread_ts,
    task_display_mode="plan"
)

# Update with task status
await client.chat.appendStream(
    channel=channel_id,
    ts=stream_result.ts,
    chunks=[
        TaskUpdateChunk(
            id="agent_main",
            title="Ford — Checking Status",
            status="in_progress",
            details="- Scanning services\n- Pulling git state"
        )
    ]
)

# Finalize
await client.chat.stopStream(
    channel=channel_id,
    ts=stream_result.ts,
    chunks=[final_response_chunk]
)
```

**Deliverable:** Official streaming API integrated, task cards showing

### Phase 4: Slack MCP Server Integration (2-3 hours)
**Goal:** Connect LLM calls to Slack MCP server for tools
**Changes to `assistant_handler.py`:**

When calling LLM (Gemini, Claude, OpenAI), include MCP server:

**For Claude (Anthropic):**
```python
response = await client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1000,
    messages=messages,
    mcp_servers=[
        {
            "type": "url",
            "url": "https://mcp.slack.com/mcp",
            "name": "slack"
        }
    ]
)
```

**For Gemini:**
Check if Vertex AI supports MCP (may need workaround with tool schema)

Available tools via MCP:
- `search_messages`: Query messages with NL
- `search_channels`: Find channels by topic
- `search_files`: Find files with keywords
- `search_users`: Find users
- `post_message`: Send message on user's behalf
- `create_canvas`: Create collaborative canvas
- `read_channel_history`: Get channel history
- `read_thread`: Get full thread context

**Deliverable:** LLM calls include MCP server, agents can invoke Slack tools

### Phase 5: Context Management (2 hours)
**Goal:** Implement smart workspace search with `assistant.search.context()`
**Pattern:**
```python
# Get workspace context for user query
search_result = await client.assistant.search.context(
    query="What decisions did we make on project X?",
    action_token=event.action_token,
    content_types=["messages", "files", "channels"],
    channel_types=["public_channel", "private_channel"],
    limit=20
)

# Build structured state
state = {
    "goal": user_query,
    "constraints": "",  # date range, channel scope
    "decisions": [],
    "artifacts": [],
    "sources": search_result.results.messages
}

# Pass to LLM with full context
llm_response = await call_llm_with_state(state)

# Update state iteratively (don't refetch)
state["decisions"] = parsed_response.decisions
state["artifacts"].append(...)
```

**Deliverable:** Workspace search integrated, structured state management working

### Phase 6: Thread Management & Governance (1-2 hours)
**Goal:** Professional thread titles, audit logging, human-in-the-loop
**Changes:**
1. Set thread titles with `assistant.threads.setTitle()` after first message
   ```python
   await client.assistant.threads.setTitle(
       channel_id=channel_id,
       thread_ts=thread_ts,
       title=first_user_message[:50]  # Auto-title from first prompt
   )
   ```

2. Add audit logging:
   - Request timestamp, user_id, agent_id, model, tokens, outcome
   - Store in structured log (BQ, local file, or Slack metadata)

3. Add control surfaces:
   - Buttons in response: "Refine", "Redo", "Share", "Approve"
   - Require confirmation for high-impact actions

4. Add content disclaimers:
   ```python
   {
       "type": "context",
       "elements": [{
           "type": "mrkdwn",
           "text": "Generated by AI. Check for accuracy before acting."
       }]
   }
   ```

**Deliverable:** Professional governance framework, audit trail, user controls

---

## Testing & Validation

### Unit Tests
- [ ] Thread lifecycle events fire correctly
- [ ] Suggested prompts render in UI
- [ ] Stream API calls succeed (mock)
- [ ] MCP tools parse correctly

### Integration Tests
- [ ] User opens agent container → receives welcome + prompts
- [ ] User clicks suggested prompt → message sent
- [ ] LLM responds → streamed in real-time with task updates
- [ ] Multi-agent tasks show sequentially
- [ ] Final response includes sources/citations

### Manual Testing (End-to-End)
1. Install app in workspace
2. Open agent from top bar
3. Verify welcome message + 3-4 suggested prompts appear
4. Click a prompt → message sent
5. Watch streaming response + task cards
6. Verify response includes citations + action buttons
7. Click action button → app responds appropriately

---

## Risk & Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Breaking existing Socket Mode flows | HIGH | Keep old handlers alongside new Assistant class, migrate gradually |
| Slack API changes | MEDIUM | Pin API version, test on sandbox first |
| MCP tool failures | MEDIUM | Graceful fallback if MCP tools unavailable, continue with LLM only |
| Token cost explosion | MEDIUM | Monitor MCP tool usage, implement rate limiting |
| Workspace search too slow | MEDIUM | Add query timeout (5s), fallback to simpler search if needed |

---

## Files to Modify

1. **app.py** — Add Assistant lifecycle, new event handlers
2. **assistant_handler.py** — Replace streaming, add task cards, integrate MCP
3. **commands.py** — Keep slash commands, but route through assistant context
4. **app_home.py** — Add App Home for status/controls
5. **governance.py** — Enhance audit logging + human-in-the-loop
6. **manifest.json** — Enable assistant_view, add event subscriptions

---

## Success Criteria

✅ **Feature Parity with Slack Docs Examples**
- Assistant container renders in top bar
- Suggested prompts appear on thread start
- Streaming response shows task progress
- MCP tools available to LLM

✅ **Performance**
- Thread starts in <2s
- First response streams within 5s
- No rate limit errors

✅ **User Experience**
- Suggested prompts are contextual (not generic)
- Task cards show real agent work (not fake)
- Final response includes sources
- User can take action (buttons) on response

✅ **Governance**
- All requests logged (user, agent, model, cost)
- High-impact actions require approval
- Audit trail queryable

---

## Next Steps

1. **IMMEDIATE:** Read Slack docs (provided above)
2. **Then:** Update app.py with Assistant lifecycle
3. **Then:** Implement streaming with chat.startStream
4. **Then:** Add MCP server to LLM calls
5. **Then:** Enhance context management with workspace search
6. **Then:** Add governance/audit framework
7. **Then:** Comprehensive testing + sandbox validation
8. **Then:** Deploy to production workspace

**Estimated effort:** 8-12 hours over 2-3 days  
**Estimated cost:** $5-10 (LLM calls + testing)  
**Ready to proceed:** YES — Peder's authorization received

---

**Prepared by:** Ford  
**Sources:** Slack Developer Docs (official)  
**Status:** RESEARCH COMPLETE — Ready for PLAN & GENERATE phases
