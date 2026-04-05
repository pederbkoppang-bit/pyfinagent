"""
Governance & Trust — Audit logging, observability, and guardrails.

Implements Slack's governance framework:
- Audit trail with all recommended metrics per response
- Deterministic fallback messages (never raw model payload)
- Rate limiting and token budget tracking
- Outcome classification (success/partial/failure)

Metrics tracked per interaction (from docs.slack.dev/ai/agent-governance):
  total_latency_ms    End-to-end clock time including tool calls
  outcome             success | partial | failure
  user_id             User in the interaction
  agent_id            Which agent produced the response
  tools_called        Array of tool names invoked
  model               Model name and version
  retry_attempts      Count of retries
  total_tokens        Input + output token count
  token_efficiency    Output/input ratio (low = over-prompting)
  error_type          llm_error | tool_error | validation_error | timeout | none

Reference: https://docs.slack.dev/ai/agent-governance
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


# ── Audit Record ─────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """Single audit record for one agent interaction."""
    timestamp: str = ""
    user_id: str = ""
    channel_id: str = ""
    source: str = ""                  # slack | imessage | channel
    query_preview: str = ""           # First 100 chars of user message

    # Agent routing
    agent_id: str = ""                # main | qa | research | direct
    complexity: str = ""              # trivial | simple | moderate | complex
    classification_confidence: float = 0.0
    parallel_agents: list = field(default_factory=list)

    # Performance
    total_latency_ms: float = 0.0
    model: str = ""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    token_efficiency: float = 0.0     # output/input ratio
    retry_attempts: int = 0

    # Outcome
    outcome: str = "success"          # success | partial | failure
    error_type: str = "none"          # llm_error | tool_error | validation_error | timeout | none
    error_message: str = ""

    # Response
    response_length: int = 0
    streamed: bool = False
    task_plan_used: bool = False
    feedback: str = ""                # positive | negative | none


# ── Audit Logger ─────────────────────────────────────────────────

class AuditLogger:
    """
    Audit trail for all agent interactions.

    Stores records in memory (ring buffer) and optionally to disk.
    Provides query methods for /agent logs and App Home.
    """

    def __init__(self, max_records: int = 500, log_dir: Optional[str] = None):
        self._records: deque[AuditRecord] = deque(maxlen=max_records)
        self._lock = Lock()
        self._log_dir = log_dir or os.getenv(
            "AGENT_AUDIT_DIR",
            str(Path.home() / "pyfinAgent" / "logs" / "audit")
        )
        self._stats = {
            "total_requests": 0,
            "successes": 0,
            "failures": 0,
            "total_tokens_used": 0,
            "total_latency_ms": 0.0,
        }

        # Ensure log directory exists
        try:
            Path(self._log_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def log(self, record: AuditRecord):
        """Log an audit record."""
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc).isoformat()

        # Compute token efficiency
        if record.input_tokens > 0:
            record.token_efficiency = round(
                record.output_tokens / record.input_tokens, 3
            )

        with self._lock:
            self._records.append(record)
            self._stats["total_requests"] += 1
            self._stats["total_tokens_used"] += record.total_tokens
            self._stats["total_latency_ms"] += record.total_latency_ms
            if record.outcome == "success":
                self._stats["successes"] += 1
            elif record.outcome == "failure":
                self._stats["failures"] += 1

        # Log to file (daily rotation)
        self._write_to_file(record)

        # Structured log for observability
        logger.info(
            f"📊 AUDIT: user={record.user_id} agent={record.agent_id} "
            f"outcome={record.outcome} latency={record.total_latency_ms:.0f}ms "
            f"tokens={record.total_tokens} efficiency={record.token_efficiency:.2f} "
            f"error={record.error_type}"
        )

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Get recent audit records as dicts."""
        with self._lock:
            records = list(self._records)[-limit:]
        return [asdict(r) for r in reversed(records)]

    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        with self._lock:
            stats = dict(self._stats)
            total = stats["total_requests"] or 1
            stats["success_rate"] = round(stats["successes"] / total * 100, 1)
            stats["avg_latency_ms"] = round(stats["total_latency_ms"] / total, 0)
            stats["avg_tokens_per_request"] = round(stats["total_tokens_used"] / total, 0)
        return stats

    def get_agent_breakdown(self) -> dict:
        """Get per-agent statistics."""
        breakdown = {}
        with self._lock:
            for record in self._records:
                agent = record.agent_id
                if agent not in breakdown:
                    breakdown[agent] = {
                        "count": 0, "tokens": 0, "avg_latency": 0.0,
                        "failures": 0, "total_latency": 0.0,
                    }
                breakdown[agent]["count"] += 1
                breakdown[agent]["tokens"] += record.total_tokens
                breakdown[agent]["total_latency"] += record.total_latency_ms
                if record.outcome == "failure":
                    breakdown[agent]["failures"] += 1

        for agent, data in breakdown.items():
            if data["count"] > 0:
                data["avg_latency"] = round(data["total_latency"] / data["count"], 0)
            del data["total_latency"]

        return breakdown

    def record_feedback(self, user_id: str, feedback: str):
        """Update the most recent record for a user with feedback."""
        with self._lock:
            for record in reversed(self._records):
                if record.user_id == user_id:
                    record.feedback = feedback
                    break

    def _write_to_file(self, record: AuditRecord):
        """Append record to daily audit log file."""
        try:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = Path(self._log_dir) / f"audit_{date_str}.jsonl"
            with open(path, "a") as f:
                f.write(json.dumps(asdict(record), default=str) + "\n")
        except Exception as e:
            logger.debug(f"Audit file write failed: {e}")


# ── Rate Limiter ─────────────────────────────────────────────────

class TokenBudgetTracker:
    """
    Track token usage per user and enforce daily budgets.

    Implements Slack governance: "Rate-limit and monitor API usage.
    Token/cost runaway is a real risk."
    """

    def __init__(self, daily_budget_per_user: int = 50_000):
        self._usage: dict[str, dict] = {}  # user_id -> {date, tokens}
        self._daily_budget = daily_budget_per_user
        self._lock = Lock()

    def check_budget(self, user_id: str) -> tuple[bool, int]:
        """
        Check if user has budget remaining.
        Returns (allowed, remaining_tokens).
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            usage = self._usage.get(user_id, {"date": today, "tokens": 0})
            if usage["date"] != today:
                usage = {"date": today, "tokens": 0}
                self._usage[user_id] = usage

            remaining = self._daily_budget - usage["tokens"]
            return remaining > 0, max(0, remaining)

    def record_usage(self, user_id: str, tokens: int):
        """Record token usage for a user."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if user_id not in self._usage or self._usage[user_id]["date"] != today:
                self._usage[user_id] = {"date": today, "tokens": 0}
            self._usage[user_id]["tokens"] += tokens

    def get_usage(self, user_id: str) -> dict:
        """Get usage stats for a user."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            usage = self._usage.get(user_id, {"date": today, "tokens": 0})
            if usage["date"] != today:
                return {"date": today, "tokens": 0, "remaining": self._daily_budget}
            return {
                "date": today,
                "tokens": usage["tokens"],
                "remaining": max(0, self._daily_budget - usage["tokens"]),
            }


# ── Deterministic Fallback Messages ──────────────────────────────

FALLBACK_MESSAGES = {
    "llm_error": (
        "⚠️ I hit a temporary issue with the AI model. "
        "Please try again in a moment."
    ),
    "tool_error": (
        "⚠️ One of my tools failed to respond. "
        "Try rephrasing your question, or check `/agent state` for details."
    ),
    "validation_error": (
        "⚠️ I couldn't process that request safely. "
        "Please try again with a clearer question."
    ),
    "timeout": (
        "⚠️ Your request took too long to process. "
        "Try a simpler question, or break it into smaller parts."
    ),
    "rate_limited": (
        "⚠️ You've reached your daily token budget. "
        "Usage resets at midnight UTC. Check `/agent settings` for details."
    ),
    "unknown": (
        "⚠️ Something unexpected happened. "
        "Please try again. If this persists, check `/agent logs`."
    ),
}


def classify_error(exception: Exception) -> str:
    """Classify an exception into an error_type for audit logging."""
    error_str = str(exception).lower()
    error_type_name = type(exception).__name__.lower()

    if "timeout" in error_str or "timed out" in error_str:
        return "timeout"
    if "rate" in error_str or "429" in error_str:
        return "rate_limited"
    if "api" in error_str or "anthropic" in error_type_name:
        return "llm_error"
    if "tool" in error_str or "mcp" in error_str:
        return "tool_error"
    if "valid" in error_str or "parse" in error_str:
        return "validation_error"
    return "llm_error"


def get_fallback_message(error_type: str) -> str:
    """Get a deterministic, user-friendly fallback message."""
    return FALLBACK_MESSAGES.get(error_type, FALLBACK_MESSAGES["unknown"])


# ── Module Singletons ────────────────────────────────────────────

_audit_logger: Optional[AuditLogger] = None
_token_tracker: Optional[TokenBudgetTracker] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def get_token_tracker() -> TokenBudgetTracker:
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenBudgetTracker()
    return _token_tracker
