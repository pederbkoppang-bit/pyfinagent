"""
Direct Responder — handles common queries LOCALLY without Anthropic API calls.

This is the key to instant responses. Instead of routing every message through:
    message → ticket → queue → Anthropic API (30s+) → response

Simple queries are answered directly:
    message → direct_responder (< 1 second) → response

Only complex/analytical queries get routed to the ticket system.
"""

import logging
import subprocess
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def can_handle_directly(text: str) -> bool:
    """Return True if this message can be answered without an LLM call."""
    text_lower = text.lower().strip()

    # Direct-answer triggers (no AI needed)
    direct_triggers = [
        "status", "health", "ping", "alive", "running",
        "services", "uptime", "check",
        "portfolio", "nav", "positions", "pnl",
        "tickets", "queue", "backlog",
        "plan", "progress", "phase",
        "git", "commits", "push",
        "help", "commands",
        "hello", "hi", "hey", "yo",
    ]

    return any(trigger in text_lower for trigger in direct_triggers)


def get_direct_response(text: str) -> Optional[str]:
    """
    Generate a response locally for common queries.
    Returns None if the query can't be handled directly.
    """
    text_lower = text.lower().strip()

    # Greetings
    if text_lower in ("hello", "hi", "hey", "yo", "ping"):
        return "👋 Hey! I'm online and responsive. What do you need?"

    # Help
    if "help" in text_lower or "commands" in text_lower:
        return (
            "📋 *Available Commands:*\n"
            "• `status` — System health + services\n"
            "• `portfolio` / `nav` / `positions` — Paper trading status\n"
            "• `tickets` / `queue` — Ticket system status\n"
            "• `plan` / `progress` — PLAN.md progress\n"
            "• `git` / `commits` — Git status\n"
            "• `/analyze TICKER` — Run full analysis\n"
            "• `/portfolio` — Portfolio performance\n"
            "• `/report TICKER` — Latest report\n\n"
            "_For complex questions (why, analyze, research), I'll route to AI agents._"
        )

    # System status
    if any(kw in text_lower for kw in ("status", "health", "services", "uptime", "alive", "running", "check")):
        return _build_status_response()

    # Portfolio / NAV
    if any(kw in text_lower for kw in ("portfolio", "nav", "positions", "pnl")):
        return _build_portfolio_response()

    # Ticket queue status
    if any(kw in text_lower for kw in ("tickets", "queue", "backlog")):
        return _build_ticket_status()

    # Plan progress
    if any(kw in text_lower for kw in ("plan", "progress", "phase")):
        return _build_plan_progress()

    # Git status
    if any(kw in text_lower for kw in ("git", "commits", "push")):
        return _build_git_status()

    return None


def _build_status_response() -> str:
    """Build system status without any API calls."""
    lines = ["📊 *System Status*\n"]

    # Check backend
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8000/api/health", headers={"User-Agent": "pyfinagent"})
        resp = urllib.request.urlopen(req, timeout=3)
        data = json.loads(resp.read())
        lines.append(f"✅ Backend (8000): {data.get('status', 'ok')} — v{data.get('version', '?')}")
    except Exception:
        lines.append("❌ Backend (8000): Not responding")

    # Check frontend
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:3000", headers={"User-Agent": "pyfinagent"})
        urllib.request.urlopen(req, timeout=3)
        lines.append("✅ Frontend (3000): Running")
    except Exception:
        lines.append("❌ Frontend (3000): Not responding")

    # Check Slack bot process
    try:
        result = subprocess.run(
            ["pgrep", "-f", "backend.slack_bot.app"],
            capture_output=True, text=True, timeout=3
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            lines.append(f"✅ Slack Bot: Running (PID {pids[0]})")
        else:
            lines.append("⚠️ Slack Bot: No process found")
    except Exception:
        lines.append("⚠️ Slack Bot: Check failed")

    # Check iMessage responder
    try:
        result = subprocess.run(
            ["pgrep", "-f", "imsg_responder"],
            capture_output=True, text=True, timeout=3
        )
        if result.stdout.strip():
            lines.append("✅ iMessage Responder: Running")
        else:
            lines.append("⚠️ iMessage Responder: Not running")
    except Exception:
        pass

    # Uptime
    try:
        result = subprocess.run(["uptime"], capture_output=True, text=True, timeout=3)
        uptime_str = result.stdout.strip()
        lines.append(f"🖥️ {uptime_str}")
    except Exception:
        pass

    return "\n".join(lines)


def _build_portfolio_response() -> str:
    """Build portfolio status from local data."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8000/api/paper-trading/status",
            headers={"User-Agent": "pyfinagent"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())

        portfolio = data.get("portfolio", {})
        nav = portfolio.get("nav", 0)
        cash = portfolio.get("cash", 0)
        pnl = portfolio.get("pnl_pct", 0)
        pos_count = data.get("position_count", 0)

        sign = "+" if pnl >= 0 else ""
        emoji = "📈" if pnl >= 0 else "📉"

        return (
            f"{emoji} *Paper Portfolio*\n"
            f"• NAV: ${nav:,.2f}\n"
            f"• Cash: ${cash:,.2f}\n"
            f"• P&L: {sign}{pnl:.2f}%\n"
            f"• Positions: {pos_count}\n"
            f"• Status: {data.get('status', 'unknown')}"
        )
    except Exception as e:
        return f"⚠️ Portfolio data unavailable: {e}"


def _build_ticket_status() -> str:
    """Build ticket queue status from SQLite directly (no API call needed)."""
    try:
        db_path = _PROJECT_ROOT / "tickets.db"
        if not db_path.exists():
            return "📋 No ticket database found."

        with sqlite3.connect(str(db_path)) as conn:
            # Count by status
            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM tickets GROUP BY status"
            )
            counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Recent tickets
            cursor = conn.execute(
                "SELECT ticket_number, status, priority, message_text "
                "FROM tickets ORDER BY created_at DESC LIMIT 5"
            )
            recent = cursor.fetchall()

        total = sum(counts.values())
        open_count = counts.get("OPEN", 0)
        in_progress = counts.get("IN_PROGRESS", 0) + counts.get("ASSIGNED", 0)
        resolved = counts.get("RESOLVED", 0)

        lines = [
            f"🎫 *Ticket System*",
            f"• Total: {total}",
            f"• Open: {open_count}",
            f"• In Progress: {in_progress}",
            f"• Resolved: {resolved}",
        ]

        if recent:
            lines.append("\n*Recent:*")
            for num, status, priority, msg in recent[:3]:
                emoji = {"OPEN": "🔵", "RESOLVED": "✅", "CLOSED": "⚫"}.get(status, "🟡")
                lines.append(f"{emoji} #{num} [{priority}] {msg[:50]}")

        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ Ticket status unavailable: {e}"


def _build_plan_progress() -> str:
    """Build plan progress from PLAN.md (local file read, no API)."""
    plan_path = _PROJECT_ROOT / "PLAN.md"
    if not plan_path.exists():
        return "⚠️ PLAN.md not found."

    try:
        content = plan_path.read_text(encoding="utf-8")
        checked = content.count("- [x]")
        unchecked = content.count("- [ ]")
        total = checked + unchecked
        pct = int(checked / total * 100) if total > 0 else 0

        # Find current phase (last "## Phase" with ✅ or 🔄)
        current_phase = "Unknown"
        for line in content.split("\n"):
            if line.startswith("## Phase") and ("🔄" in line or "IN PROGRESS" in line.upper()):
                current_phase = line.strip("# ").strip()
                break

        return (
            f"📋 *Plan Progress*\n"
            f"• Completed: {checked}/{total} items ({pct}%)\n"
            f"• Current: {current_phase}"
        )
    except Exception as e:
        return f"⚠️ Plan status unavailable: {e}"


def _build_git_status() -> str:
    """Build git status (local command, no API)."""
    try:
        # Unpushed commits
        result = subprocess.run(
            ["git", "log", "origin/main..HEAD", "--oneline"],
            cwd=str(_PROJECT_ROOT), capture_output=True, text=True, timeout=5
        )
        commits = result.stdout.strip()

        if commits:
            count = len(commits.splitlines())
            return (
                f"🔀 *Git Status*\n"
                f"• {count} unpushed commit(s):\n```{commits[:500]}```"
            )
        else:
            return "🔀 *Git Status*\n• All pushed, up to date ✅"
    except Exception as e:
        return f"⚠️ Git status unavailable: {e}"
