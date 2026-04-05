"""
App Home & Slash Commands — Governance controls and state inspection.

Implements Slack governance features:
- App Home: persistent dashboard with agent stats, recent activity, controls
- /agent logs  — view recent agent activity
- /agent state — inspect current agent state (config, routing, budgets)
- /agent settings — view/configure agent behavior

Reference: https://docs.slack.dev/ai/agent-governance#control
"""

import logging
from slack_bolt.app import App
from slack_sdk import WebClient

logger = logging.getLogger(__name__)


def register_governance(app: App):
    """Register all governance features: App Home, slash commands, actions."""

    # ═══════════════════════════════════════════════════════════════
    # APP HOME
    # ═══════════════════════════════════════════════════════════════

    @app.event("app_home_opened")
    def update_app_home(client: WebClient, event, logger):
        """
        Render the App Home tab with agent dashboard.

        Shows: agent status, aggregate stats, per-agent breakdown,
        recent activity log, and control buttons.
        """
        user_id = event["user"]

        try:
            from backend.slack_bot.governance import get_audit_logger, get_token_tracker

            audit = get_audit_logger()
            tracker = get_token_tracker()

            stats = audit.get_stats()
            breakdown = audit.get_agent_breakdown()
            recent = audit.get_recent(limit=8)
            user_usage = tracker.get_usage(user_id)

            # Build the view
            blocks = []

            # ── Header ──────────────────────────────────────────
            blocks.append({
                "type": "header",
                "text": {"type": "plain_text", "text": "pyfinAgent Dashboard"}
            })
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "Multi-agent system status and governance controls.\n"
                        "Agents: *Ford* (main) · *Analyst* (Q&A) · *Researcher*"
                    ),
                },
            })
            blocks.append({"type": "divider"})

            # ── Aggregate Stats ─────────────────────────────────
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Aggregate Stats*\n"
                        f"• Total requests: *{stats['total_requests']}*\n"
                        f"• Success rate: *{stats['success_rate']}%*\n"
                        f"• Avg latency: *{stats['avg_latency_ms']:.0f}ms*\n"
                        f"• Avg tokens/req: *{stats['avg_tokens_per_request']:.0f}*\n"
                        f"• Total tokens used: *{stats['total_tokens_used']:,}*"
                    ),
                },
            })

            # ── Per-Agent Breakdown ─────────────────────────────
            if breakdown:
                agent_lines = []
                emoji_map = {"main": "⚙️", "qa": "📊", "research": "🔬", "direct": "⚡"}
                for agent, data in breakdown.items():
                    emoji = emoji_map.get(agent, "🤖")
                    fail_str = f" ({data['failures']} failed)" if data['failures'] else ""
                    agent_lines.append(
                        f"{emoji} *{agent}*: {data['count']} calls, "
                        f"avg {data['avg_latency']:.0f}ms, "
                        f"{data['tokens']:,} tokens{fail_str}"
                    )
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "*Per-Agent Breakdown*\n" + "\n".join(agent_lines)},
                })

            # ── Your Usage ──────────────────────────────────────
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Your Usage Today*\n"
                        f"• Tokens used: *{user_usage['tokens']:,}*\n"
                        f"• Remaining budget: *{user_usage['remaining']:,}*"
                    ),
                },
            })

            # ── Recent Activity ─────────────────────────────────
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "header",
                "text": {"type": "plain_text", "text": "Recent Activity"}
            })

            if recent:
                for record in recent[:8]:
                    outcome_emoji = {"success": "✅", "partial": "⚠️", "failure": "❌"}.get(
                        record.get("outcome", ""), "❓"
                    )
                    ts = record.get("timestamp", "")[:19]
                    agent = record.get("agent_id", "?")
                    latency = record.get("total_latency_ms", 0)
                    tokens = record.get("total_tokens", 0)
                    query = record.get("query_preview", "")[:60]
                    feedback_str = ""
                    if record.get("feedback"):
                        feedback_str = f" · {'👍' if record['feedback'] == 'positive' else '👎'}"

                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"{outcome_emoji} `{ts}` *{agent}* "
                                f"({latency:.0f}ms, {tokens} tok{feedback_str})\n"
                                f"_{query}_"
                            ),
                        },
                    })
            else:
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "_No activity recorded yet._"},
                })

            # ── Controls ────────────────────────────────────────
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "🔄 Refresh"},
                        "action_id": "app_home_refresh",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📊 Full Logs"},
                        "action_id": "app_home_full_logs",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "⚙️ Settings"},
                        "action_id": "app_home_settings",
                    },
                ],
            })

            # Publish the view
            client.views_publish(
                user_id=user_id,
                view={"type": "home", "blocks": blocks},
            )

        except Exception as e:
            logger.exception(f"Failed to render App Home: {e}")
            client.views_publish(
                user_id=user_id,
                view={
                    "type": "home",
                    "blocks": [{
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"⚠️ Error loading dashboard: {str(e)[:200]}"},
                    }],
                },
            )

    # ── App Home Actions ────────────────────────────────────────

    @app.action("app_home_refresh")
    def handle_refresh(ack, body, client):
        ack()
        # Re-trigger the app_home_opened event handler
        update_app_home(
            client=client,
            event={"user": body["user"]["id"]},
            logger=logger,
        )

    @app.action("app_home_full_logs")
    def handle_full_logs(ack, body, client):
        ack()
        from backend.slack_bot.governance import get_audit_logger
        audit = get_audit_logger()
        records = audit.get_recent(limit=25)

        blocks = [{"type": "header", "text": {"type": "plain_text", "text": "Agent Audit Log (last 25)"}}]
        for r in records:
            outcome_emoji = {"success": "✅", "partial": "⚠️", "failure": "❌"}.get(r.get("outcome", ""), "❓")
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{outcome_emoji} `{r.get('timestamp', '')[:19]}` "
                        f"*{r.get('agent_id', '?')}* | {r.get('complexity', '?')} | "
                        f"{r.get('total_latency_ms', 0):.0f}ms | "
                        f"{r.get('total_tokens', 0)} tok | "
                        f"err={r.get('error_type', 'none')}\n"
                        f"_{r.get('query_preview', '')[:80]}_"
                    ),
                },
            })

        client.views_open(
            trigger_id=body["trigger_id"],
            view={"type": "modal", "title": {"type": "plain_text", "text": "Audit Log"}, "blocks": blocks},
        )

    @app.action("app_home_settings")
    def handle_settings(ack, body, client):
        ack()
        from backend.slack_bot.governance import get_token_tracker
        tracker = get_token_tracker()
        usage = tracker.get_usage(body["user"]["id"])

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "Agent Settings"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": (
                "*Models*\n"
                "• Lead classifier: local keyword (no API)\n"
                "• Subagents: `claude-sonnet-4-6`\n\n"
                "*Token Budgets*\n"
                f"• Daily budget: *50,000* tokens/user\n"
                f"• Your usage today: *{usage['tokens']:,}*\n"
                f"• Remaining: *{usage['remaining']:,}*\n\n"
                "*Agent Routing*\n"
                "• Trivial → Direct responder (local, 0 tokens)\n"
                "• Simple → Single agent (~500 tokens)\n"
                "• Moderate → Single agent + context (~1,500 tokens)\n"
                "• Complex → 2-3 agents parallel (~4,000 tokens)"
            )}},
        ]

        client.views_open(
            trigger_id=body["trigger_id"],
            view={"type": "modal", "title": {"type": "plain_text", "text": "Settings"}, "blocks": blocks},
        )

    # ═══════════════════════════════════════════════════════════════
    # SLASH COMMANDS: /agent logs | state | settings
    # ═══════════════════════════════════════════════════════════════

    @app.command("/agent")
    def handle_agent_command(ack, respond, command):
        """
        /agent [subcommand]

        Subcommands:
          logs     — View recent agent activity
          state    — Inspect current agent state and configuration
          settings — View agent settings and token budgets
          help     — Show available commands
        """
        ack()
        subcommand = (command.get("text") or "help").strip().lower()
        user_id = command.get("user_id", "unknown")

        if subcommand == "logs":
            _handle_agent_logs(respond, user_id)
        elif subcommand == "state":
            _handle_agent_state(respond, user_id)
        elif subcommand == "settings":
            _handle_agent_settings(respond, user_id)
        else:
            respond(
                "🤖 *pyfinAgent Commands*\n\n"
                "• `/agent logs` — View recent agent activity and audit trail\n"
                "• `/agent state` — Inspect current routing config and budgets\n"
                "• `/agent settings` — View model settings and token limits\n"
            )

    # ── Slash Command Handlers ──────────────────────────────────

    def _handle_agent_logs(respond, user_id):
        from backend.slack_bot.governance import get_audit_logger
        audit = get_audit_logger()
        records = audit.get_recent(limit=10)
        stats = audit.get_stats()

        if not records:
            respond("📋 No activity recorded yet. Send a message to the assistant to get started.")
            return

        lines = [
            f"📋 *Recent Agent Activity* ({stats['total_requests']} total, "
            f"{stats['success_rate']}% success rate)\n"
        ]
        for r in records[:10]:
            emoji = {"success": "✅", "partial": "⚠️", "failure": "❌"}.get(r.get("outcome", ""), "❓")
            lines.append(
                f"{emoji} `{r.get('timestamp', '')[:16]}` *{r.get('agent_id', '?')}* — "
                f"{r.get('total_latency_ms', 0):.0f}ms, {r.get('total_tokens', 0)} tok — "
                f"_{r.get('query_preview', '')[:50]}_"
            )

        respond("\n".join(lines))

    def _handle_agent_state(respond, user_id):
        from backend.slack_bot.governance import get_audit_logger, get_token_tracker

        audit = get_audit_logger()
        tracker = get_token_tracker()
        stats = audit.get_stats()
        breakdown = audit.get_agent_breakdown()
        usage = tracker.get_usage(user_id)

        agent_lines = []
        emoji_map = {"main": "⚙️", "qa": "📊", "research": "🔬", "direct": "⚡"}
        for agent, data in breakdown.items():
            emoji = emoji_map.get(agent, "🤖")
            agent_lines.append(
                f"  {emoji} {agent}: {data['count']} calls, avg {data['avg_latency']:.0f}ms"
            )

        respond(
            f"🔍 *Agent State*\n\n"
            f"*System*\n"
            f"• Total requests: {stats['total_requests']}\n"
            f"• Success rate: {stats['success_rate']}%\n"
            f"• Avg latency: {stats['avg_latency_ms']:.0f}ms\n\n"
            f"*Agent Breakdown*\n" + ("\n".join(agent_lines) or "  No data yet") + "\n\n"
            f"*Your Budget*\n"
            f"• Used today: {usage['tokens']:,} / 50,000 tokens\n"
            f"• Remaining: {usage['remaining']:,}"
        )

    def _handle_agent_settings(respond, user_id):
        from backend.slack_bot.governance import get_token_tracker
        tracker = get_token_tracker()
        usage = tracker.get_usage(user_id)

        respond(
            f"⚙️ *Agent Settings*\n\n"
            f"*Models*\n"
            f"• Classifier: local keyword matching (no API call)\n"
            f"• Subagent model: `claude-sonnet-4-6`\n\n"
            f"*Token Budgets*\n"
            f"• Daily limit: 50,000 tokens/user\n"
            f"• Your usage: {usage['tokens']:,} / 50,000\n\n"
            f"*Routing Rules*\n"
            f"• Trivial (status/ping) → Direct responder, 0 tokens\n"
            f"• Simple (operational) → Ford, ~500 tokens\n"
            f"• Moderate (analytical) → Analyst/Researcher, ~1,500 tokens\n"
            f"• Complex (multi-faceted) → 2-3 agents parallel, ~4,000 tokens\n\n"
            f"*Data Handling*\n"
            f"• Messages sent to Anthropic API for processing\n"
            f"• No message content stored beyond audit metadata\n"
            f"• Audit logs retained for 30 days"
        )
