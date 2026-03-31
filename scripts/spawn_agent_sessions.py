#!/usr/bin/env python3
"""
Phase 3.2: Agentic Coordination Loop — Session Manager

Manages the 4-session architecture via OpenClaw sessions_spawn API:
  1. MAIN (Coordinator) — Opus 4.6, orchestrates harness
  2. Q&A (Analyst) — Opus 4.6, answers analytical questions
  3. Research (Researcher) — Sonnet, deep research on novel approaches
  4. Slack (Broadcaster) — Sonnet, posts team updates

This script provides utilities to:
  - Spawn sessions via sessions_spawn (OpenClaw API)
  - Track active session IDs
  - Route incoming messages (iMessage, heartbeat) to appropriate agent
  - Enforce cost budgets per session
  - Log all session lifecycle events
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import OpenClaw session spawn API
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configuration
WORKSPACE = Path.home() / ".openclaw" / "workspace"
SESSIONS_LOG = Path("/tmp/agent_sessions.log")
ACTIVE_SESSIONS_FILE = WORKSPACE / "memory" / "active_sessions.json"

# System prompts (trimmed for API use)
QA_SYSTEM = """You are Analyst, Ford's reasoning specialist for strategic questions.

Answer questions about pyfinAgent's design, performance, and decisions. Cite evidence from backtest results, code, and RESEARCH.md. Explain trade-offs clearly. Read-only access."""

RESEARCH_SYSTEM = """You are Researcher, Ford's deep learning specialist.

Execute RESEARCH gates: search 7 source categories, read 3-5 best sources, document findings in RESEARCH.md. Extract methods, thresholds, pitfalls. Append to RESEARCH.md with citations."""

SLACK_SYSTEM = """You are the Slack Bot for pyfinAgent. Post status updates to #ford-approvals.

Morning (7am): Master plan status, commits, blockers. Evening (6pm): Harness summary, Sharpe, results. Read-only, <$1/day budget."""


def log_event(message: str):
    """Log session lifecycle events"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    print(line.rstrip())
    with open(SESSIONS_LOG, "a") as f:
        f.write(line)


def load_active_sessions() -> dict:
    """Load active session IDs from disk"""
    if ACTIVE_SESSIONS_FILE.exists():
        try:
            with open(ACTIVE_SESSIONS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}


def save_active_sessions(sessions: dict):
    """Save active session IDs to disk"""
    ACTIVE_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def detect_question_type(message: str) -> str:
    """
    Classify incoming message as operational, analytical, or research.
    
    Returns: "operational", "analytical", or "research"
    """
    analytical_keywords = [
        "why", "should", "explain", "analyze", "trade-off", "decision",
        "regression", "sharpe", "recommendation", "suggest", "improve",
        "compare", "better", "worse", "review"
    ]
    
    research_keywords = [
        "research", "paper", "literature", "novel", "approach", "evidence",
        "experiment", "hypothesis", "theory", "mechanism", "solution",
        "study", "implementation", "baseline", "benchmark"
    ]
    
    msg_lower = message.lower()
    
    # Check research first (more specific)
    if any(kw in msg_lower for kw in research_keywords):
        return "research"
    
    # Check analytical
    if any(kw in msg_lower for kw in analytical_keywords):
        return "analytical"
    
    # Default to operational
    return "operational"


def format_session_info(name: str, model: str, session_id: str, cost_daily: str) -> str:
    """Format session info for logging"""
    return f"✅ {name} (model={model}, session={session_id}, budget={cost_daily}/day)"


def main():
    """
    Initialize agentic coordination loop.
    
    NOTE: This script documents the session architecture and provides utilities.
    Actual session spawning is done via sessions_spawn in the main agent flow.
    """
    log_event("=== Agentic Coordination Loop Initialization ===")
    
    sessions = load_active_sessions()
    
    log_event("Session Architecture:")
    log_event("  MAIN (Coordinator): Opus 4.6, orchestrates harness, iMessage, Slack")
    log_event("  Q&A (Analyst): Opus 4.6, answers analytical questions ($3/day budget)")
    log_event("  Research (Researcher): Sonnet 4, deep research on novelty ($2/day budget)")
    log_event("  Slack (Broadcaster): Sonnet 4, posts team updates (<$0.30/day budget)")
    log_event("")
    log_event("Spawn via: sessions_spawn(runtime='subagent', model='...', task='...') in main agent")
    log_event("")
    log_event("Message routing:")
    log_event("  Operational (status, service, next step) → MAIN")
    log_event("  Analytical (why, compare, review) → Q&A")
    log_event("  Research (papers, evidence, novel) → Research")
    log_event("")
    log_event(f"Active sessions file: {ACTIVE_SESSIONS_FILE}")
    log_event(f"Session log: {SESSIONS_LOG}")
    
    return sessions


if __name__ == "__main__":
    sessions = main()
    print("\n=== Agentic Coordination Loop Ready ===")
    print("Use sessions_spawn() in main agent to spawn Q&A, Research, Slack sessions.")
    print(f"Active sessions: {sessions}")
