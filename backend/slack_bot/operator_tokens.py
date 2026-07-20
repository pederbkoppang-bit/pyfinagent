"""Operator-token ingestion + FO-2 semantic cursor (phase-62.2, goal-away-ops).

The Socket-Mode bot records verbatim operator decisions ("reply tokens") to an
append-only, git-tracked JSONL so headless away sessions can act on them.
claude.ai connectors are absent in headless runs, so this bot-side path is the
ONLY token-ingestion mechanism (research_brief_62.2.md).

Grammar (single line, uppercase keys are deliberate friction -- lowercase
falls through to ticket ingestion):
    [<step> ]<KEY>: <value>     e.g. "65.2 EU SCREENER: ON", "KILL SWITCH: RESUME"
    bare reserved words          "HALT-DEV", "RESUME-DEV"

Durability: append shape mirrors kill_switch.py's audit writer (single
json.dumps line, atomic under PIPE_BUF) plus an asyncio.Lock. Dedupe keys:
Slack event_id and (channel, ts) -- Bolt 1.27.0's Socket Mode adapter drops
retry headers (async_internals.py:18), and the envelope is acked only after
dispatch, so crash-mid-handler means redelivery; the dedupe set absorbs it.
The set is process-lifetime: a redelivery that straddles a bot restart could
double-append (accepted; sessions treat identical raw+slack_ts as one token).

FO-2 (62.0 forward obligation): the tokens_cursor is SEMANTIC -- sessions may
only touch backend/.env after validating the SPECIFIC token line via
KNOWN_TOKEN_ENV_MAP, then advance_cursor() (temp+rename refreshes mtime, which
opens the 62.0 PreToolUse gate's 6h window). The bot never writes the cursor.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
TOKENS_PATH = _PROJECT_ROOT / "handoff" / "operator_tokens.jsonl"
CURSOR_PATH = _PROJECT_ROOT / "handoff" / "away_ops" / "tokens_cursor"

TOKEN_RE = re.compile(
    r"^(?:(?P<step>[0-9][0-9.]*)\s+)?(?P<key>[A-Z][A-Z0-9 _-]+):\s*(?P<value>.+)$"
)
RESERVED_BARE = {"HALT-DEV", "RESUME-DEV"}

# Sessions consult this map before ANY backend/.env write (FO-2): a token whose
# key is absent here authorizes NO env change (it is still recorded and may
# gate non-env actions, e.g. KILL SWITCH: RESUME -> paper-trading API call).
# Steps that ship a new dark flag MUST register their key here in the same PR.
KNOWN_TOKEN_ENV_MAP: dict[str, str] = {
    # phase-62.7: drill key -- exercises the FULL semantic-cursor + hook-gate +
    # .env-write chain ONCE, attended, at the Sunday rehearsal (otherwise the
    # chain's first live execution would be 65.2's EU SCREENER, unattended).
    # AWAY_DRILL_NOOP is read by nothing; the rehearsal writes it true, then
    # the operator's ENV-LINE-81 cleanup keystroke removes it.
    "AWAY DRILL": "AWAY_DRILL_NOOP",
    # "FEE TABLE": "PAPER_FEE_TABLE_ENABLED",        # registered by 61.5 when it ships
    # "EU SCREENER": "PAPER_SCREENER_PER_MARKET",    # registered by 65.2 when it ships
}

_append_lock = asyncio.Lock()
_seen_events: set = set()


def parse_operator_token(text: str | None) -> dict | None:
    """Parse a candidate token line. Returns {step, key, value} or None."""
    t = (text or "").strip()
    if t in RESERVED_BARE:
        return {"step": None, "key": t, "value": ""}
    m = TOKEN_RE.match(t)
    if not m:
        return None
    d = m.groupdict()
    return {"step": d["step"], "key": d["key"].strip(), "value": d["value"].strip()}


def _authorized(
    *, user: str | None, channel: str | None, operator_user_id: str,
    allowed_channels: set[str], bot_id: str | None = None,
) -> bool:
    """Shared identity/channel predicate (phase-75.2, gap1-11).

    Used by BOTH the matcher and the append sink so the two can never drift.
    A matcher is a capability gate, not an authorization decision, so the same
    check is repeated at the sink and fails closed on anything unexpected.
    """
    if not operator_user_id:
        return False  # fail-closed: unconfigured operator accepts nothing
    if bot_id:
        return False
    if user != operator_user_id:
        return False
    if channel not in allowed_channels:
        return False
    return True


def is_operator_token_message(
    message: dict, operator_user_id: str, allowed_channels: set[str]
) -> bool:
    """Pure allowlist + parseability matcher (unit-testable; commands.py wraps it).

    Matcher-False makes Bolt fall through to the catch-all, so non-operator
    lookalikes and operator non-tokens still become tickets -- never swallowed.
    """
    if not _authorized(
        user=message.get("user"),
        channel=message.get("channel"),
        operator_user_id=operator_user_id,
        allowed_channels=allowed_channels,
        bot_id=message.get("bot_id"),
    ):
        return False
    return parse_operator_token(message.get("text", "")) is not None


async def append_operator_token(
    *, text: str, user: str, channel: str, ts: str,
    operator_user_id: str, allowed_channels: set[str],
    event_id: str | None = None,
) -> tuple[int, dict] | None:
    """Dedupe-check -> append -> return (1-based line number, record).

    Returns None for duplicates or unparseable text (the matcher should have
    filtered the latter; double-checked here because append is the last line
    of defense for file integrity).

    phase-75.2 (gap1-11): the identity/channel check is repeated HERE, at the
    sink, so a future caller that skips `is_operator_token_message` cannot
    write. `operator_user_id` and `allowed_channels` are REQUIRED -- there is
    deliberately no default, because any default would be a fail-open hazard.
    """
    if not _authorized(
        user=user, channel=channel,
        operator_user_id=operator_user_id, allowed_channels=allowed_channels,
    ):
        logger.warning(
            "operator_token: REFUSED unauthorized append (user=%r channel=%r) "
            "-- sink-level check, caller bypassed the matcher",
            user, channel,
        )
        return None

    parsed = parse_operator_token(text)
    if parsed is None:
        return None
    dedupe_keys = {k for k in (event_id, (channel, ts)) if k}
    async with _append_lock:
        if any(k in _seen_events for k in dedupe_keys):
            logger.info("operator_token: duplicate delivery suppressed (ts=%s)", ts)
            return None
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "channel": channel,
            "slack_ts": ts,
            "event_id": event_id,
            "raw": (text or "").strip(),
            **parsed,
        }
        TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(TOKENS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        _seen_events.update(dedupe_keys)
        with open(TOKENS_PATH, encoding="utf-8") as f:
            line_no = sum(1 for _ in f)
        logger.info("operator_token: recorded line %d key=%r", line_no, parsed["key"])
        return line_no, record


# ── FO-2 cursor helpers (consumed by away SESSIONS, never by the bot) ──────

def read_cursor() -> dict | None:
    """Return the cursor JSON, or None if absent/corrupt (gate stays closed)."""
    try:
        with open(CURSOR_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def unapplied_tokens() -> list[tuple[int, dict]]:
    """Token lines past cursor.applied_line, as (line_no, record) pairs."""
    cur = read_cursor()
    start = (cur or {}).get("applied_line", 0)
    out: list[tuple[int, dict]] = []
    if not TOKENS_PATH.exists():
        return out
    with open(TOKENS_PATH, encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            if i <= start or not raw.strip():
                continue
            try:
                out.append((i, json.loads(raw)))
            except json.JSONDecodeError:
                logger.warning("operator_token: corrupt jsonl line %d skipped", i)
    return out


def advance_cursor(line_no: int, raw_line: str) -> dict:
    """Semantically advance the cursor after APPLYING token at line_no.

    Writes {applied_line, token_sha256, step, key, value, applied_at} via
    temp+rename in the same directory (atomic; rename refreshes mtime, which
    is what opens the 62.0 hook's 6h .env-write window).
    """
    rec = json.loads(raw_line) if isinstance(raw_line, str) else dict(raw_line)
    payload = {
        "applied_line": line_no,
        "token_sha256": hashlib.sha256(
            json.dumps(rec, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "step": rec.get("step"),
        "key": rec.get("key"),
        "value": rec.get("value"),
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(CURSOR_PATH.parent), prefix=".tokens_cursor.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, CURSOR_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return payload
