"""
App Home & Slash Commands — MAS Dashboard + Governance controls.

Wired to real MAS system:
- Agent configs from agent_definitions.py (live model changes)
- Event bus from mas_events.py (real-time event stream)
- Cost tracker from cost_tracker.py (token/cost usage)
- System health from backend + frontend HTTP checks
- /agent slash commands for quick inspection

Reference: https://docs.slack.dev/ai/agent-governance#control
"""

import logging
from slack_sdk import WebClient

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-sonnet-4-20250514",
    "claude-haiku-35-20241022",
]

AGENT_DISPLAY = [
    ("main", "⚙️ Ford (Main)", "Orchestrator, planner, synthesizer"),
    ("qa", "📊 Q&A Analyst", "Quantitative reasoning + harness tools"),
    ("research", "🔬 Researcher", "Literature & evidence search"),
    ("communication", "💬 Communication", "Router + Quality Gate reviewer"),
]


def _get_live_data():
    """Pull live data from MAS system components."""
    from backend.agents.agent_definitions import AGENT_CONFIGS, AgentType
    from backend.agents.mas_events import get_event_bus

    bus = get_event_bus()
    event_stats = bus.stats
    recent_events = bus.get_buffer()[-5:]

    # Cost tracker
    cost_summary = None
    try:
        from backend.agents.cost_tracker import CostTracker
        tracker = CostTracker()
        cost_summary = tracker.summarize()
    except Exception:
        pass

    # System health
    health = {}
    try:
        import httpx
        r = httpx.get("http://localhost:8000/api/health", timeout=3)
        health["backend"] = f"✅ Backend (8000): `{r.status_code}`"
    except Exception:
        health["backend"] = "❌ Backend (8000): DOWN"

    try:
        import httpx
        r = httpx.get("http://localhost:3000/", timeout=3)
        health["frontend"] = f"✅ Frontend (3000): `{r.status_code}`"
    except Exception:
        health["frontend"] = "❌ Frontend (3000): DOWN"

    health["event_bus"] = (
        f"📡 Event Bus: {event_stats.get('total_events', 0)} events, "
        f"{event_stats.get('subscribers', 0)} subscribers"
    )

    return {
        "configs": AGENT_CONFIGS,
        "AgentType": AgentType,
        "event_stats": event_stats,
        "recent_events": recent_events,
        "cost_summary": cost_summary,
        "health": health,
    }


def _build_home_blocks(data):
    """Build Slack Block Kit blocks for the App Home."""
    configs = data["configs"]
    AgentType = data["AgentType"]
    event_stats = data["event_stats"]
    recent_events = data["recent_events"]
    cost_summary = data["cost_summary"]
    health = data["health"]

    blocks = []

    # ── Header ──────────────────────────────────────────────
    blocks.append({
        "type": "header",
        "text": {"type": "plain_text", "text": "pyfinAgent — Multi-Agent System"},
    })
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": (
            "Evidence-based trading signals · May 2026 go-live · "
            "<https://www.anthropic.com/engineering/multi-agent-research-system|Anthropic MAS pattern>"
        )}],
    })
    blocks.append({"type": "divider"})

    # ── Architecture ────────────────────────────────────────
    # Build dynamic diagram from actual config models
    main_model = configs.get(AgentType.MAIN)
    qa_model = configs.get(AgentType.QA)
    res_model = configs.get(AgentType.RESEARCH)
    comm_model = configs.get(AgentType.COMMUNICATION)

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": (
            "*Architecture*\n```\n"
            f"User → Communication ({comm_model.model if comm_model else '?'})\n"
            f"         ├→ Ford ({main_model.model if main_model else '?'})\n"
            f"         ├→ Q&A ({qa_model.model if qa_model else '?'})\n"
            f"         └→ Researcher ({res_model.model if res_model else '?'})\n"
            "              → Quality Gate → CitationAgent\n```"
        )},
    })
    blocks.append({"type": "divider"})

    # ── Agent Inventory (with model selectors) ──────────────
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Agent Inventory*  _(change models with dropdowns)_"},
    })

    type_map = {t.value: t for t in AgentType}
    for agent_value, label, desc in AGENT_DISPLAY:
        agent_type = type_map.get(agent_value)
        config = configs.get(agent_type) if agent_type else None
        current_model = config.model if config else "claude-sonnet-4-6"

        options = []
        initial = None
        for m in AVAILABLE_MODELS:
            opt = {"text": {"type": "plain_text", "text": m}, "value": m}
            options.append(opt)
            if m == current_model:
                initial = opt

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"{label}\n_{desc}_\nCurrent: `{current_model}`"},
            "accessory": {
                "type": "static_select",
                "placeholder": {"type": "plain_text", "text": "Change model"},
                "action_id": f"agent_model_change_{agent_value}",
                "options": options,
                **({"initial_option": initial} if initial else {}),
            },
        })

    blocks.append({"type": "divider"})

    # ── System Health ───────────────────────────────────────
    health_text = "\n".join(health.values())
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*System Health*\n{health_text}"},
    })
    blocks.append({"type": "divider"})

    # ── Live Stats (Event Bus + Cost Tracker) ───────────────
    stats_text = (
        f"*Event Bus*\n"
        f"• Events: *{event_stats.get('total_events', 0)}*\n"
        f"• Subscribers: *{event_stats.get('subscribers', 0)}*\n"
        f"• Buffer: *{event_stats.get('buffer_size', 0)}*"
    )
    if cost_summary:
        stats_text += (
            f"\n\n*Cost Tracker*\n"
            f"• Calls: *{cost_summary.get('total_calls', 0)}*\n"
            f"• Tokens: *{cost_summary.get('total_tokens', 0):,}*\n"
            f"• Cost: *${cost_summary.get('total_cost_usd', 0):.4f}*"
        )
        mb = cost_summary.get("model_breakdown", {})
        if mb:
            stats_text += "\n\n*Model Usage*"
            for model, d in mb.items():
                stats_text += f"\n• `{model}`: {d['calls']} calls, {d['tokens']:,} tok, ${d['cost_usd']:.4f}"

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": stats_text},
    })
    blocks.append({"type": "divider"})

    # ── Recent MAS Events ───────────────────────────────────
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "*Recent MAS Events*"},
    })

    event_emoji = {
        "classify": "📬", "delegate": "➡️", "tool_call": "🔧",
        "complete": "✅", "error": "❌", "quality_gate": "🛡️",
        "plan": "📋", "think": "🧠",
    }
    if recent_events:
        for evt in reversed(recent_events):
            etype = evt.get("event_type", "unknown")
            emoji = event_emoji.get(etype, "🟠")
            agent = evt.get("agent", "?")
            ts = evt.get("timestamp", "")[:19]
            detail = evt.get("detail", "")[:60]
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": (
                    f"{emoji} `{ts}` *{agent}* · {etype}" +
                    (f" · _{detail}_" if detail else "")
                )}],
            })
    else:
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "_No events yet. Send a message to generate events._"}],
        })

    # ── Key Features ────────────────────────────────────────
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": (
            "*Key Features*\n"
            "• 🧠 *Interleaved thinking* — Subagents reason after each tool call\n"
            "• 🛡️ *Quality Gate* — 4-criterion rubric (Accuracy, Completeness, Groundedness, Conciseness)\n"
            "• 📚 *CitationAgent* — Source markers on research responses\n"
            "• 🔄 *Iterative research* — \"More research needed?\" loop (max 3 rounds)\n"
            "• 🗜️ *Observation masking* — ACON-inspired context compression at 60%\n"
            "• ⚡ *Parallel tools* — Multiple harness reads in parallel\n"
            "• 📡 *Event bus* — Real-time observability at `/agents`"
        )},
    })

    # ── Controls ────────────────────────────────────────────
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "actions",
        "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "🔄 Refresh"}, "action_id": "app_home_refresh"},
            {"type": "button", "text": {"type": "plain_text", "text": "📋 Logs"}, "action_id": "app_home_full_logs"},
            {"type": "button", "text": {"type": "plain_text", "text": "⚙️ Settings"}, "action_id": "app_home_settings"},
        ],
    })

    # ── Footer ──────────────────────────────────────────────
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": (
            "pyfinAgent v5.14 · "
            "<https://www.anthropic.com/engineering/multi-agent-research-system|Research System> · "
            "<https://www.anthropic.com/engineering/harness-design-long-running-apps|Harness Design>"
        )}],
    })

    return blocks


def register_governance(app):
    """Register all governance features: App Home, slash commands, actions."""

    # ═══════════════════════════════════════════════════════════════
    # APP HOME
    # ═══════════════════════════════════════════════════════════════

    @app.event("app_home_opened")
    async def update_app_home(client: WebClient, event, logger):
        """Render the App Home tab with full MAS dashboard."""
        user_id = event["user"]
        logger.info(f"🏠 App Home opened by {user_id}")

        try:
            data = _get_live_data()
            blocks = _build_home_blocks(data)

            await client.views_publish(
                user_id=user_id,
                view={"type": "home", "blocks": blocks},
            )

        except Exception as e:
            logger.exception(f"Failed to render App Home: {e}")
            await client.views_publish(
                user_id=user_id,
                view={
                    "type": "home",
                    "blocks": [{"type": "section", "text": {
                        "type": "mrkdwn", "text": f"⚠️ Error: {str(e)[:200]}",
                    }}],
                },
            )

    # ── Agent Model Change Handlers ─────────────────────────

    async def _handle_model_change(ack, body, client, agent_type_str):
        """Handle model change for any agent."""
        await ack()
        from backend.agents.agent_definitions import AGENT_CONFIGS, AgentType

        selected = body["actions"][0]["selected_option"]["value"]
        user_id = body["user"]["id"]

        type_map = {t.value: t for t in AgentType}
        agent_type = type_map.get(agent_type_str)

        if agent_type and agent_type in AGENT_CONFIGS:
            old_model = AGENT_CONFIGS[agent_type].model
            AGENT_CONFIGS[agent_type].model = selected
            logger.info(f"🔄 Agent model changed: {agent_type.value} {old_model} → {selected} (by {user_id})")

        # Refresh
        await update_app_home(client=client, event={"user": user_id}, logger=logger)

    @app.action("agent_model_change_main")
    async def handle_model_main(ack, body, client):
        await _handle_model_change(ack, body, client, "main")

    @app.action("agent_model_change_qa")
    async def handle_model_qa(ack, body, client):
        await _handle_model_change(ack, body, client, "qa")

    @app.action("agent_model_change_research")
    async def handle_model_research(ack, body, client):
        await _handle_model_change(ack, body, client, "research")

    @app.action("agent_model_change_communication")
    async def handle_model_comm(ack, body, client):
        await _handle_model_change(ack, body, client, "communication")

    # ── App Home Action Buttons ─────────────────────────────

    @app.action("app_home_refresh")
    async def handle_refresh(ack, body, client):
        await ack()
        await update_app_home(client=client, event={"user": body["user"]["id"]}, logger=logger)

    @app.action("app_home_full_logs")
    async def handle_full_logs(ack, body, client):
        await ack()
        from backend.agents.mas_events import get_event_bus
        bus = get_event_bus()
        events = bus.get_buffer()[-25:]

        blocks = [{"type": "header", "text": {"type": "plain_text", "text": "MAS Event Log (last 25)"}}]
        event_emoji = {
            "classify": "📬", "delegate": "➡️", "tool_call": "🔧",
            "complete": "✅", "error": "❌", "quality_gate": "🛡️",
        }
        for evt in reversed(events):
            etype = evt.get("event_type", "unknown")
            emoji = event_emoji.get(etype, "🟠")
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": (
                    f"{emoji} `{evt.get('timestamp', '')[:19]}` "
                    f"*{evt.get('agent', '?')}* · {etype} · "
                    f"_{evt.get('detail', '')[:80]}_"
                )},
            })

        if not events:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "_No events recorded yet._"}})

        await client.views_open(
            trigger_id=body["trigger_id"],
            view={"type": "modal", "title": {"type": "plain_text", "text": "MAS Events"}, "blocks": blocks[:50]},
        )

    @app.action("app_home_settings")
    async def handle_settings(ack, body, client):
        await ack()
        from backend.agents.agent_definitions import AGENT_CONFIGS, AgentType

        agent_lines = []
        for at in [AgentType.COMMUNICATION, AgentType.MAIN, AgentType.QA, AgentType.RESEARCH]:
            c = AGENT_CONFIGS.get(at)
            if c:
                agent_lines.append(f"• *{c.name}*: `{c.model}` (max {c.max_tokens} tok)")

        cost_text = ""
        try:
            from backend.agents.cost_tracker import CostTracker
            tracker = CostTracker()
            s = tracker.summarize()
            cost_text = (
                f"\n\n*Cost Summary*\n"
                f"• Calls: {s['total_calls']}\n"
                f"• Tokens: {s['total_tokens']:,}\n"
                f"• Cost: ${s['total_cost_usd']:.4f}"
            )
        except Exception:
            pass

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "Agent Settings"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": (
                "*Current Models*\n" + "\n".join(agent_lines) +
                "\n\n*Thinking*\n"
                "• Budget: 2048 tokens/turn (interleaved)\n\n"
                "*Quality Gate*\n"
                "• 4-criterion rubric (0.0-1.0)\n"
                "• Threshold: any < 0.6 OR avg < 0.7 = FAIL\n"
                "• 3 few-shot calibration examples\n\n"
                "*Routing Tiers*\n"
                "• TRIVIAL → Direct (0 tokens)\n"
                "• SIMPLE → 1 agent (~500 tok)\n"
                "• MODERATE → 1-2 agents (~1,500 tok)\n"
                "• COMPLEX → 2-3 agents parallel (~4,000 tok)"
                + cost_text
            )}},
        ]

        await client.views_open(
            trigger_id=body["trigger_id"],
            view={"type": "modal", "title": {"type": "plain_text", "text": "Settings"}, "blocks": blocks},
        )

    # ═══════════════════════════════════════════════════════════════
    # SLASH COMMANDS: /agent logs | state | settings
    # ═══════════════════════════════════════════════════════════════

    @app.command("/agent")
    async def handle_agent_command(ack, respond, command):
        await ack()
        subcommand = (command.get("text") or "help").strip().lower()

        if subcommand == "logs":
            from backend.agents.mas_events import get_event_bus
            bus = get_event_bus()
            events = bus.get_buffer()[-10:]
            if not events:
                await respond("📋 No MAS events recorded yet.")
                return
            lines = [f"📋 *Recent MAS Events* ({bus.stats.get('total_events', 0)} total)\n"]
            for evt in reversed(events):
                lines.append(
                    f"• `{evt.get('timestamp', '')[:16]}` *{evt.get('agent', '?')}* · "
                    f"{evt.get('event_type', '?')} · _{evt.get('detail', '')[:50]}_"
                )
            await respond("\n".join(lines))

        elif subcommand == "state":
            from backend.agents.agent_definitions import AGENT_CONFIGS, AgentType
            from backend.agents.mas_events import get_event_bus
            bus = get_event_bus()
            es = bus.stats

            agent_lines = []
            for at in [AgentType.MAIN, AgentType.QA, AgentType.RESEARCH, AgentType.COMMUNICATION]:
                c = AGENT_CONFIGS.get(at)
                if c:
                    agent_lines.append(f"  • {c.name}: `{c.model}`")

            await respond(
                f"🔍 *Agent State*\n\n"
                f"*Models*\n" + "\n".join(agent_lines) + "\n\n"
                f"*Event Bus*\n"
                f"• Events: {es.get('total_events', 0)}\n"
                f"• Subscribers: {es.get('subscribers', 0)}\n"
                f"• Buffer: {es.get('buffer_size', 0)}"
            )

        elif subcommand == "settings":
            from backend.agents.agent_definitions import AGENT_CONFIGS, AgentType
            lines = ["⚙️ *Agent Settings*\n\n*Models*"]
            for at in [AgentType.COMMUNICATION, AgentType.MAIN, AgentType.QA, AgentType.RESEARCH]:
                c = AGENT_CONFIGS.get(at)
                if c:
                    lines.append(f"• {c.name}: `{c.model}` (max {c.max_tokens} tok)")
            lines.append("\n*Features*\n• Interleaved thinking: 2048 tok\n• Quality Gate: 4-criterion rubric\n• Observation masking: 60%\n• Parallel tools: enabled")
            await respond("\n".join(lines))

        else:
            await respond(
                "🤖 *pyfinAgent Commands*\n\n"
                "• `/agent logs` — Recent MAS events\n"
                "• `/agent state` — Agent models + event bus stats\n"
                "• `/agent settings` — Full configuration\n"
            )
