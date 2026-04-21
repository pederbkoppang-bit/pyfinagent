"""
OpenClaw Gateway Client — routes all MAS agent calls through OpenClaw.

Instead of direct Anthropic API calls, this module sends requests to the
OpenClaw Gateway's OpenAI-compatible /v1/chat/completions endpoint.

Every MAS agent call becomes an OpenClaw session:
  model: "openclaw/qa"       → routes to the QA agent
  model: "openclaw/research"  → routes to the Researcher agent
  x-openclaw-model: "anthropic/claude-opus-4-6"  → overrides backend model
  x-openclaw-session-key: "mas:qa:123"  → persistent session

Benefits:
- All sessions visible in OpenClaw (sessions_list, MAS Dashboard)
- Cost tracking unified
- Model routing through Gateway config
- Session history searchable via memory_search
"""

import os
import json

from backend.utils import json_io
import logging
import time
import httpx
from typing import Optional, Generator

logger = logging.getLogger(__name__)

# Gateway config
GATEWAY_URL = os.getenv("OPENCLAW_GATEWAY_URL", "http://127.0.0.1:18789")
GATEWAY_TOKEN = os.getenv(
    "OPENCLAW_GATEWAY_TOKEN",
    "120288ce760863de36268579db931b706758ee20eceaca86",
)

# Map agent types to OpenClaw agent IDs
AGENT_ID_MAP = {
    "COMMUNICATION": "communication",
    "MAIN": "main",
    "QA": "qa",
    "RESEARCH": "research",
}

# Model overrides (OpenClaw agents have defaults, but we can override)
AGENT_MODEL_OVERRIDES = {
    "communication": "anthropic/claude-sonnet-4-6",
    "main": "anthropic/claude-opus-4-6",
    "qa": "anthropic/claude-opus-4-6",
    "research": "anthropic/claude-sonnet-4-6",
}


def _headers(
    agent_id: str = "main",
    model_override: Optional[str] = None,
    session_key: Optional[str] = None,
) -> dict:
    """Build headers for Gateway API calls."""
    h = {
        "Authorization": f"Bearer {GATEWAY_TOKEN}",
        "Content-Type": "application/json",
    }
    if model_override:
        h["x-openclaw-model"] = model_override
    if session_key:
        h["x-openclaw-session-key"] = session_key
    return h


def openclaw_chat(
    agent_id: str,
    messages: list[dict],
    system_prompt: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    model_override: Optional[str] = None,
    session_key: Optional[str] = None,
) -> dict:
    """
    Send a chat completion through OpenClaw Gateway.

    Uses model: "openclaw/<agent_id>" to route to the right agent.
    Optionally overrides the backend model with x-openclaw-model header.

    Returns dict with:
        - content: str (the response text)
        - model: str (model used)
        - usage: dict (prompt_tokens, completion_tokens, total_tokens)
        - agent_id: str
    """
    # Build messages — system prompt goes into messages (OpenAI format)
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    # Resolve model override
    resolved_override = model_override or AGENT_MODEL_OVERRIDES.get(agent_id)

    payload = {
        "model": f"openclaw/{agent_id}",
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    start = time.time()
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{GATEWAY_URL}/v1/chat/completions",
                headers=_headers(agent_id, resolved_override, session_key),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            content = ""
            if data.get("choices"):
                content = data["choices"][0].get("message", {}).get("content", "")

            usage = data.get("usage", {})
            elapsed = round((time.time() - start) * 1000, 1)

            logger.info(
                f"🔗 OpenClaw [{agent_id}] → {data.get('model', '?')} | "
                f"tokens: {usage.get('total_tokens', '?')} | "
                f"{elapsed}ms"
            )

            return {
                "content": content,
                "model": data.get("model", ""),
                "usage": usage,
                "agent_id": agent_id,
                "elapsed_ms": elapsed,
            }

    except httpx.HTTPStatusError as e:
        logger.error(
            f"OpenClaw Gateway error [{agent_id}]: "
            f"{e.response.status_code} {e.response.text[:200]}"
        )
        raise
    except httpx.ConnectError:
        logger.error(
            f"OpenClaw Gateway unreachable at {GATEWAY_URL} — is the gateway running?"
        )
        raise


def openclaw_chat_stream(
    agent_id: str,
    messages: list[dict],
    system_prompt: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    model_override: Optional[str] = None,
    session_key: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream a chat completion through OpenClaw Gateway.
    Yields text chunks as they arrive (SSE stream).
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    resolved_override = model_override or AGENT_MODEL_OVERRIDES.get(agent_id)

    payload = {
        "model": f"openclaw/{agent_id}",
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    logger.info(f"🔗 OpenClaw stream [{agent_id}] starting...")

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{GATEWAY_URL}/v1/chat/completions",
            headers=_headers(agent_id, resolved_override, session_key),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json_io.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue


def check_gateway_health() -> bool:
    """Check if OpenClaw Gateway is reachable and /v1 endpoint enabled."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                f"{GATEWAY_URL}/v1/models",
                headers={"Authorization": f"Bearer {GATEWAY_TOKEN}"},
            )
            return resp.status_code == 200
    except Exception:
        return False


def list_openclaw_sessions() -> list[dict]:
    """List active OpenClaw sessions via /tools/invoke."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{GATEWAY_URL}/tools/invoke",
                headers={"Authorization": f"Bearer {GATEWAY_TOKEN}",
                         "Content-Type": "application/json"},
                json={
                    "tool": "sessions_list",
                    "args": {"activeMinutes": 60, "messageLimit": 1},
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result", {}).get("sessions", [])
    except Exception as e:
        logger.error(f"Failed to list OpenClaw sessions: {e}")
    return []
