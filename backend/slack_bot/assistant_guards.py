"""Live-path guards for the Slack assistant (phase-75.2, gap1-05/gap1-04).

Three controls that used to live in the now-deleted control plane, rebuilt
on the path that is actually registered at runtime:

1. `is_deploy_request` -- deterministic, pre-LLM refusal routing. OWASP
   LLM01:2025 requires privileged operations be gated in code rather than
   exposed to the model; matching before classification means the assistant
   cannot hallucinate a deploy it never performed.
2. `rate_ok` -- per-user sliding-window limiter. Cloudflare's approximation
   (rate = prev * ((period - elapsed) / period) + cur) needs two integers
   per user and no external store, with ~0.003% error at scale.
3. `audit` -- append-only JSONL record per interaction, mirroring the
   `operator_tokens.py` writer idiom (asyncio.Lock, mkdir parents, one
   `json.dumps(..., ensure_ascii=False)` line, append mode).

The audit path is under `handoff/logs/`, which is gitignored -- deliberate,
because records describe user messages. We store `text_sha256` rather than
the message text so the record is privacy-preserving and could later be
promoted to a tracked path without leaking conversation content.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
AUDIT_PATH = _PROJECT_ROOT / "handoff" / "logs" / "assistant_audit.jsonl"

# Deploy-request surface. Derived from the ACTUAL matcher this step deleted
# (recovered with `git show HEAD:backend/slack_bot/self_update.py`,
# handle_deploy_command), not from memory -- an earlier revision of this list
# was materially narrower than the code it claimed parity with, and bare
# "deploy" fell straight through to the LLM.
#
# The old matcher had two arms: exact whole-message matches, and a
# `startswith("deploy")` catch-all. Both are reproduced here.
#
# `_DEPLOY_PATTERN` covers every "deploy ..." phrasing anywhere in the message
# (criterion 3 says "a message CONTAINING a deploy verb"), so it is slightly
# broader than the original catch-all -- "please deploy the bot" now refuses
# too. The `(?:s|ed|ing)?` suffix group deliberately excludes "deployment", so
# asking about deployment history stays answerable.
_DEPLOY_PATTERN = re.compile(
    r"\b(?:deploy(?:s|ed|ing)?|redeploy|rollback|roll\s+back)\b"
)

# Whole-message aliases from the old matcher that contain no "deploy" token.
# Matched EXACTLY (as the original did) rather than as substrings, so
# "tell me what changed in the portfolio" stays a legitimate question.
_DEPLOY_EXACT = frozenset({
    "update bot",
    "pull and restart",
    "git status",
    "what changed",
    "deploy changes",
    "cleanup",
    "clean old",
})

_WINDOW_S = 60.0
_MAX_PER_WINDOW = 20
# user_id -> (window_start, prev_window_count, current_window_count)
_rl: dict[str, tuple[float, int, int]] = {}

_audit_lock = asyncio.Lock()


def is_deploy_request(text: str) -> bool:
    """True when the message asks for a deploy. Checked BEFORE any LLM call."""
    low = (text or "").lower().strip()
    if low in _DEPLOY_EXACT:
        return True
    return _DEPLOY_PATTERN.search(low) is not None


def rate_ok(user_id: str, now: float | None = None) -> bool:
    """Sliding-window check. False means the user is over the limit."""
    now = time.time() if now is None else now
    start, prev, cur = _rl.get(user_id, (now, 0, 0))
    elapsed = now - start
    if elapsed >= _WINDOW_S:
        # One window fully elapsed keeps its count as `prev`; two or more
        # means the user went quiet, so the history is dropped.
        prev = cur if elapsed < 2 * _WINDOW_S else 0
        cur, start, elapsed = 0, now, 0.0
    rate = prev * ((_WINDOW_S - elapsed) / _WINDOW_S) + cur
    if rate >= _MAX_PER_WINDOW:
        _rl[user_id] = (start, prev, cur)
        return False
    _rl[user_id] = (start, prev, cur + 1)
    return True


def reset_rate_limit() -> None:
    """Test helper -- drop all sliding-window state."""
    _rl.clear()


async def audit(
    *, user: str | None, channel: str | None, text: str,
    outcome: str, agent: str | None = None, slack_ts: str | None = None,
) -> dict:
    """Append one interaction record. Never raises into the request path."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "writer": "assistant_audit",
        "user": user,
        "channel": channel,
        "slack_ts": slack_ts,
        "text_sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest(),
        "outcome": outcome,
        "agent": agent,
    }
    try:
        async with _audit_lock:
            AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("assistant_audit: append failed: %r", exc)
    return record
