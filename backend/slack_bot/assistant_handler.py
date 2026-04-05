"""
Slack AI Agent — Assistant handler with full streaming and task plan display.

Implements the complete Slack AI Agent response loop:
1. assistant_thread_started → welcome + suggested prompts
2. assistant_thread_context_changed → context tracking
3. message.im → classify → set_status → stream response

Streaming modes:
- SIMPLE: chat_stream() with markdown_text chunks (word-by-word)
- COMPLEX: chat_stream(task_display_mode="plan") with task_update chunks
  showing per-agent progress as they complete in real-time

Task update API (from docs.slack.dev/reference/methods/chat.appendStream):
  task_update chunk fields:
    id:      unique task identifier
    title:   task label
    status:  "pending" | "in_progress" | "complete" | "error"
    details: markdown text (expandable area — what the agent is doing)
    output:  text shown when complete (agent's key finding)
    sources: [{type: "url", text, url}] — citation links

Reference: https://docs.slack.dev/ai/developing-agents
Reference: https://docs.slack.dev/reference/methods/chat.appendStream#task_update-chunks
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import Logger
from typing import Dict, List, Optional

from slack_bolt import BoltContext, Say, SetStatus
from slack_sdk import WebClient
from slack_sdk.models.messages.chunk import (
    MarkdownTextChunk,
    PlanUpdateChunk,
    TaskUpdateChunk,
)

from backend.agents.agent_definitions import (
    AGENT_CONFIGS,
    AgentType,
    ClassificationResult,
    QueryComplexity,
    classify_trivial,
)

logger = logging.getLogger(__name__)


# ── Agent metadata for task display ──────────────────────────────

AGENT_TASK_META = {
    AgentType.MAIN: {
        "task_id": "agent_main",
        "title": "Ford — checking operational status",
        "details_pending": "- Scanning service health\n- Checking git status\n- Reviewing task queue",
        "details_working": "- Querying system state\n- Evaluating current configuration",
    },
    AgentType.QA: {
        "task_id": "agent_qa",
        "title": "Analyst — quantitative reasoning",
        "details_pending": "- Reviewing backtest metrics\n- Analyzing feature importance\n- Comparing walk-forward windows",
        "details_working": "- Computing Sharpe ratio analysis\n- Evaluating sub-period robustness\n- Assessing parameter sensitivity",
    },
    AgentType.RESEARCH: {
        "task_id": "agent_research",
        "title": "Researcher — searching literature",
        "details_pending": "- Scanning arXiv, SSRN, Journal of Finance\n- Cross-referencing practitioner sources\n- Evaluating implementation feasibility",
        "details_working": "- Reading relevant papers\n- Extracting methods and thresholds\n- Identifying pitfalls and trade-offs",
    },
}


# ── Loading messages per agent ───────────────────────────────────

LOADING_MESSAGES = {
    AgentType.MAIN: [
        "Checking the engine room…",
        "Scanning service heartbeats…",
        "Pulling the latest from git…",
    ],
    AgentType.QA: [
        "Crunching the numbers…",
        "Comparing walk-forward windows…",
        "Interrogating the Sharpe ratio…",
    ],
    AgentType.RESEARCH: [
        "Diving into the literature…",
        "Scanning arXiv and SSRN…",
        "Cross-referencing López de Prado…",
    ],
    AgentType.DIRECT: ["On it…"],
}

LOADING_MESSAGES_COMPLEX = [
    "Assembling the research team…",
    "Analyst and Researcher working in parallel…",
    "Synthesizing multi-agent findings…",
    "Almost there — merging results…",
]


# ── Suggested prompts ────────────────────────────────────────────

def _get_suggested_prompts(thread_context=None) -> List[Dict[str, str]]:
    prompts = [
        {
            "title": "System status",
            "message": "What's the current status of all pyfinAgent services?",
        },
        {
            "title": "Portfolio performance",
            "message": "Show me the current paper trading portfolio performance and P&L.",
        },
        {
            "title": "Explain Sharpe ratio",
            "message": "Why did the Sharpe ratio change in the last backtest run? What drove it?",
        },
        {
            "title": "Research momentum",
            "message": "Research the latest academic papers on momentum factor decay in ML-based trading systems.",
        },
    ]
    return prompts[:4]


# ── Feedback block ───────────────────────────────────────────────

def _create_feedback_block() -> list:
    return [
        {
            "type": "actions",
            "block_id": "agent_feedback",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "👍"},
                    "action_id": "agent_feedback_positive",
                    "value": "positive",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "👎"},
                    "action_id": "agent_feedback_negative",
                    "value": "negative",
                },
            ],
        }
    ]


# ═══════════════════════════════════════════════════════════════════
# ASSISTANT EVENT HANDLERS
# ═══════════════════════════════════════════════════════════════════


def handle_thread_started(say, set_suggested_prompts, get_thread_context, logger):
    """User opened the assistant side panel — show welcome + prompts."""
    try:
        thread_context = get_thread_context()
        say(
            "👋 Hey! I'm the pyfinAgent assistant. I can help with system status, "
            "portfolio analysis, strategy questions, and research.\n\n"
            "Try a prompt below or ask me anything."
        )
        set_suggested_prompts(prompts=_get_suggested_prompts(thread_context))
        logger.info("✅ Assistant thread started — prompts set")
    except Exception as e:
        logger.exception(f"Failed to handle thread_started: {e}")
        say(f"⚠️ Something went wrong setting up: {e}")


def handle_context_changed(get_thread_context, logger):
    """User switched channels while panel is open — track context."""
    try:
        ctx = get_thread_context()
        if ctx:
            logger.info(f"📋 Context changed — channel: {getattr(ctx, 'channel_id', None)}")
    except Exception as e:
        logger.exception(f"Failed to handle context change: {e}")


def handle_user_message(client, context, get_thread_context, logger, payload, say, set_status):
    """
    Core response loop with governance:
      receive → budget check → classify → set_status → stream → audit log

    Governance features integrated:
    - Token budget check before processing (rate limiting)
    - Audit record logged for every interaction
    - Deterministic fallback messages on errors (never raw model output)
    - Next-step action buttons in responses
    """
    start_time = time.time()
    audit_record = None

    try:
        from backend.slack_bot.governance import (
            AuditRecord, get_audit_logger, get_token_tracker,
            classify_error, get_fallback_message,
        )

        channel_id = payload["channel"]
        team_id = context.team_id
        thread_ts = payload["thread_ts"]
        user_id = context.user_id
        user_message = payload.get("text", "").strip()

        if not user_message:
            return

        logger.info(f"📨 Assistant message: {user_message[:80]}")

        # ── Initialize audit record ─────────────────────────────
        audit_record = AuditRecord(
            user_id=user_id,
            channel_id=channel_id,
            source="slack_assistant",
            query_preview=user_message[:100],
        )

        # ── Token budget check ──────────────────────────────────
        tracker = get_token_tracker()
        allowed, remaining = tracker.check_budget(user_id)
        if not allowed:
            audit_record.outcome = "failure"
            audit_record.error_type = "rate_limited"
            audit_record.total_latency_ms = (time.time() - start_time) * 1000
            get_audit_logger().log(audit_record)

            say(get_fallback_message("rate_limited"))
            return

        # ── Deploy commands (check BEFORE LLM classification) ─────
        try:
            from backend.slack_bot.self_update import handle_deploy_command
            deploy_response = handle_deploy_command(user_message)
            if deploy_response is not None:
                say(deploy_response)
                audit_record.agent_id = "deploy"
                audit_record.outcome = "success"
                audit_record.total_latency_ms = (time.time() - start_time) * 1000
                get_audit_logger().log(audit_record)
                return
        except Exception as e:
            logger.debug(f"Deploy check error: {e}")

        # ── Classify via Communication Agent (Sonnet 4.6) ───────
        from backend.agents.multi_agent_orchestrator import get_orchestrator
        orchestrator = get_orchestrator()

        # Set initial loading status while classifying
        set_status(status="thinking…", loading_messages=["Routing your request…"])

        classification = orchestrator.classify_message_sync(user_message)
        audit_record.agent_id = classification.agent_type.value
        audit_record.complexity = classification.complexity.value
        audit_record.classification_confidence = classification.confidence
        audit_record.parallel_agents = [a.value for a in (classification.parallel_agents or [])]

        logger.info(
            f"📋 → {classification.agent_type.value} "
            f"({classification.complexity.value}, {classification.confidence:.0%}) "
            f"— {classification.reasoning}"
        )

        # Update loading status with agent-specific messages
        loading = LOADING_MESSAGES.get(classification.agent_type, ["Working on it…"])
        if classification.complexity == QueryComplexity.COMPLEX:
            loading = LOADING_MESSAGES_COMPLEX
        set_status(status="thinking…", loading_messages=loading)

        # ── Thread title ────────────────────────────────────────
        title = user_message[:60] + ("…" if len(user_message) > 60 else "")
        try:
            client.assistant_threads_setTitle(
                channel_id=channel_id, thread_ts=thread_ts, title=title,
            )
        except Exception:
            pass

        # ── Route ───────────────────────────────────────────────
        if classification.agent_type == AgentType.DIRECT:
            result = orchestrator.execute_classified_sync(user_message, classification, user_id)
            say(result["response"])

            audit_record.outcome = "success"
            audit_record.total_latency_ms = (time.time() - start_time) * 1000
            get_audit_logger().log(audit_record)
            return

        if classification.complexity == QueryComplexity.COMPLEX and classification.parallel_agents:
            result = _stream_complex_with_task_plan(
                client=client,
                orchestrator=orchestrator,
                classification=classification,
                user_message=user_message,
                user_id=user_id,
                channel_id=channel_id,
                team_id=team_id,
                thread_ts=thread_ts,
                logger=logger,
            )
            audit_record.task_plan_used = True
        else:
            result = _stream_simple(
                client=client,
                orchestrator=orchestrator,
                classification=classification,
                user_message=user_message,
                user_id=user_id,
                channel_id=channel_id,
                team_id=team_id,
                thread_ts=thread_ts,
                logger=logger,
            )

        # ── Post-response: audit + token tracking ───────────────
        if result:
            tokens = result.get("token_usage", {})
            audit_record.total_latency_ms = (time.time() - start_time) * 1000
            audit_record.model = "claude-sonnet-4-6"
            audit_record.input_tokens = tokens.get("input", 0)
            audit_record.output_tokens = tokens.get("output", 0)
            audit_record.total_tokens = tokens.get("input", 0) + tokens.get("output", 0)
            audit_record.response_length = len(result.get("response", ""))
            audit_record.streamed = True
            audit_record.outcome = "success" if "error" not in result else "partial"

            tracker.record_usage(user_id, audit_record.total_tokens)

        get_audit_logger().log(audit_record)

    except Exception as e:
        logger.exception(f"Failed to handle user message: {e}")

        # ── Deterministic fallback ──────────────────────────────
        from backend.slack_bot.governance import (
            classify_error, get_fallback_message, get_audit_logger, AuditRecord,
        )
        error_type = classify_error(e)
        say(get_fallback_message(error_type))

        # Log the failure
        if audit_record is None:
            audit_record = AuditRecord(
                user_id=payload.get("user", context.user_id if context else "unknown"),
                source="slack_assistant",
            )
        audit_record.outcome = "failure"
        audit_record.error_type = error_type
        audit_record.error_message = str(e)[:200]
        audit_record.total_latency_ms = (time.time() - start_time) * 1000
        get_audit_logger().log(audit_record)


# ═══════════════════════════════════════════════════════════════════
# STREAMING STRATEGIES
# ═══════════════════════════════════════════════════════════════════


def _stream_simple(
    client, orchestrator, classification, user_message,
    user_id, channel_id, team_id, thread_ts, logger,
):
    """
    Stream a single-agent response using markdown_text chunks.
    User sees the response appear word-by-word like ChatGPT.
    """
    start = time.time()

    # Call agent (uses pre-classified routing — no re-classification)
    result = orchestrator.execute_classified_sync(user_message, classification, user_id)
    response_text = result.get("response", "No response generated.")
    tokens = result.get("token_usage", {})
    proc_ms = result.get("processing_time_ms", 0)

    agent_label = result.get("agent_type", "unknown")
    agent_name = {"main": "Ford", "qa": "Analyst", "research": "Researcher"}.get(agent_label, agent_label)
    footer = f"\n\n_🤖 {agent_name} · {proc_ms:.0f}ms · {tokens.get('input', 0)}+{tokens.get('output', 0)} tokens_"
    full = response_text + footer

    # Stream it
    streamer = client.chat_stream(
        channel=channel_id,
        recipient_team_id=team_id,
        recipient_user_id=user_id,
        thread_ts=thread_ts,
    )

    for chunk in _split_chunks(full, 80):
        streamer.append(markdown_text=chunk)
        time.sleep(0.04)

    streamer.stop(blocks=_create_feedback_block())
    logger.info(f"✅ Streamed {agent_name} ({len(full)} chars) in {(time.time()-start)*1000:.0f}ms")
    return result


def _stream_complex_with_task_plan(
    client, orchestrator, classification, user_message,
    user_id, channel_id, team_id, thread_ts, logger,
):
    """
    Stream a multi-agent response with Slack's task plan display.

    Uses task_display_mode="plan" so users see a visual plan with:
    - Each agent as a task card (pending → in_progress → complete)
    - Real-time updates as each agent finishes
    - Agent output shown in the task's "output" field
    - Details showing what the agent investigated

    Per-agent progress is achieved by running agents in a ThreadPoolExecutor
    and updating the Slack stream via task_update chunks as each completes.

    Reference: https://docs.slack.dev/reference/methods/chat.appendStream#task_update-chunks
    """
    start = time.time()
    agents = classification.parallel_agents or [classification.agent_type]

    # ── 1. Start stream with plan mode ──────────────────────────
    streamer = client.chat_stream(
        channel=channel_id,
        recipient_team_id=team_id,
        recipient_user_id=user_id,
        thread_ts=thread_ts,
        task_display_mode="plan",
    )

    # ── 2. Initial plan: all agents as pending ──────────────────
    initial_chunks = [
        PlanUpdateChunk(
            title=f"Research plan — {len(agents)} agents",
        ),
    ]
    for agent_type in agents:
        meta = AGENT_TASK_META.get(agent_type, {})
        initial_chunks.append(
            TaskUpdateChunk(
                id=meta.get("task_id", agent_type.value),
                title=meta.get("title", agent_type.value),
                status="pending",
                details=meta.get("details_pending", ""),
            )
        )
    streamer.append(chunks=initial_chunks)

    # Small delay so user sees the pending state
    time.sleep(0.8)

    # ── 3. Mark all as in_progress ──────────────────────────────
    progress_chunks = []
    for agent_type in agents:
        meta = AGENT_TASK_META.get(agent_type, {})
        progress_chunks.append(
            TaskUpdateChunk(
                id=meta.get("task_id", agent_type.value),
                title=meta.get("title", agent_type.value),
                status="in_progress",
                details=meta.get("details_working", "Working…"),
            )
        )
    streamer.append(chunks=progress_chunks)

    # ── 4. Run agents in parallel, update as each completes ─────
    total_usage = {"input": 0, "output": 0}
    agent_responses = {}

    def _run_agent(agent_type):
        """Run a single agent (called from thread pool)."""
        return agent_type, orchestrator.call_single_agent_sync(
            agent_type=agent_type,
            message=user_message,
            is_subtask=True,
            classification=classification,
        )

    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        futures = {pool.submit(_run_agent, at): at for at in agents}

        for future in as_completed(futures):
            agent_type = futures[future]
            meta = AGENT_TASK_META.get(agent_type, {})
            task_id = meta.get("task_id", agent_type.value)

            try:
                _, result = future.result()
                response_text = result.get("response", "")
                usage = result.get("token_usage", {})
                proc_ms = result.get("processing_time_ms", 0)
                has_error = "error" in result

                total_usage["input"] += usage.get("input", 0)
                total_usage["output"] += usage.get("output", 0)
                agent_responses[agent_type] = response_text

                # Extract first line as output summary
                lines = response_text.strip().split("\n")
                output_summary = lines[0][:200] if lines else "Completed"

                # Update task to complete with output
                streamer.append(chunks=[
                    TaskUpdateChunk(
                        id=task_id,
                        title=meta.get("title", agent_type.value),
                        status="error" if has_error else "complete",
                        details=f"Completed in {proc_ms:.0f}ms · {usage.get('input',0)}+{usage.get('output',0)} tokens",
                        output=output_summary,
                    ),
                ])

                logger.info(f"✅ {agent_type.value} completed in {proc_ms:.0f}ms")

            except Exception as e:
                logger.error(f"❌ {agent_type.value} failed: {e}")
                agent_responses[agent_type] = f"⚠️ Error: {str(e)[:150]}"

                streamer.append(chunks=[
                    TaskUpdateChunk(
                        id=task_id,
                        title=meta.get("title", agent_type.value),
                        status="error",
                        details=f"Failed: {str(e)[:100]}",
                    ),
                ])

    # ── 5. Update plan title to "completed" ─────────────────────
    streamer.append(chunks=[
        PlanUpdateChunk(title="All agents completed"),
    ])

    # ── 6. Stream the full synthesized response ─────────────────
    time.sleep(0.3)

    # Build the synthesis with labeled agent sections
    synthesis_parts = []
    emoji_map = {AgentType.QA: "📊", AgentType.RESEARCH: "🔬", AgentType.MAIN: "⚙️"}

    for agent_type in agents:
        response = agent_responses.get(agent_type, "No response")
        config = AGENT_CONFIGS.get(agent_type)
        label = config.name if config else agent_type.value
        emoji = emoji_map.get(agent_type, "🤖")
        synthesis_parts.append(f"{emoji} *{label}:*\n{response}")

    full_synthesis = "\n\n---\n\n".join(synthesis_parts)

    total_ms = (time.time() - start) * 1000
    footer = (
        f"\n\n_🤖 Multi-agent ({', '.join(a.value for a in agents)}) · "
        f"{total_ms:.0f}ms · {total_usage['input']}+{total_usage['output']} tokens_"
    )
    full_synthesis += footer

    # Stream the synthesis text
    for chunk in _split_chunks(full_synthesis, 100):
        streamer.append(chunks=[MarkdownTextChunk(text=chunk)])
        time.sleep(0.04)

    # ── 7. Stop stream with feedback buttons ────────────────────
    streamer.stop(blocks=_create_feedback_block())

    logger.info(
        f"✅ Complex response — {len(agents)} agents, {total_ms:.0f}ms total, "
        f"{total_usage['input']}+{total_usage['output']} tokens"
    )

    return {
        "response": full_synthesis,
        "agent_type": "multi-agent",
        "token_usage": total_usage,
        "processing_time_ms": total_ms,
    }


# ═══════════════════════════════════════════════════════════════════
# FEEDBACK HANDLERS
# ═══════════════════════════════════════════════════════════════════


def handle_feedback_positive(ack, body, logger):
    ack()
    user = body.get("user", {}).get("id", "unknown")
    logger.info(f"👍 Positive feedback from {user}")
    from backend.slack_bot.governance import get_audit_logger
    get_audit_logger().record_feedback(user, "positive")


def handle_feedback_negative(ack, body, logger):
    ack()
    user = body.get("user", {}).get("id", "unknown")
    logger.info(f"👎 Negative feedback from {user}")
    from backend.slack_bot.governance import get_audit_logger
    get_audit_logger().record_feedback(user, "negative")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _run_orchestrator_sync(orchestrator, message, sender):
    """Run the async orchestrator from sync context."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            orchestrator.handle_message(message=message, sender=sender, source="slack")
        )
    finally:
        loop.close()


def _split_chunks(text: str, size: int = 80) -> list:
    """Split text into word-boundary chunks for smooth streaming."""
    if len(text) <= size:
        return [text]
    chunks, current = [], ""
    for word in text.split(" "):
        if len(current) + len(word) + 1 > size and current:
            chunks.append(current + " ")
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        chunks.append(current)
    return chunks
