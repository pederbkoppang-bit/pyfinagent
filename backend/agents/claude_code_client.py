"""phase-cycle-3 (2026-05-26): Claude Code CLI subprocess client.

Provides `claude_code_invoke()` for routing LLM analysis calls through the
`claude --print --output-format json` CLI on the Max-subscription flat-fee
rail, bypassing the direct api.anthropic.com billing rail when the
operator sets settings.paper_use_claude_code_route=True.

The CLI invocation is verified live (researcher aff3444de945e98c2,
2026-05-26): Max-subscription auth honored from non-CLI subprocess (uses
~/.claude/ credentials; no ANTHROPIC_API_KEY env var required).

Citations:
- Anthropic Claude Code SDK programmatic-invocation docs
  (https://code.claude.com/docs/en/headless).
- Anthropic structured-outputs docs
  (https://code.claude.com/docs/en/agent-sdk/structured-outputs).
- TradingAgents arXiv:2412.20138 -- LLM-rail abstraction in production
  multi-agent trading systems.
- Portkey AI Gateway -- failover-routing canonical (10B+ req/mo).
- Bailey/Borwein/Lopez de Prado/Zhu PBO SSRN:2326253 -- engine-change
  logging required for A/B integrity.

ASCII-only log messages per backend-services.md::Logging.
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ClaudeCodeError(RuntimeError):
    """Raised when the `claude` CLI returns a non-success envelope."""


def claude_code_invoke(
    prompt: str,
    *,
    max_tokens: Optional[int] = None,
    system: Optional[str] = None,
    timeout_s: int = 120,
    json_schema: Optional[dict] = None,
    cwd: Optional[str] = None,
    disallowed_tools: str = "Bash,Edit,Write,Read,Glob,Grep,Agent",
    binary: str = "claude",
) -> dict[str, Any]:
    """Invoke `claude --print --output-format json` as a subprocess.

    Returns the parsed JSON envelope dict from the CLI. Raises
    ClaudeCodeError if subtype != 'success' or the CLI exits non-zero.

    Args:
        prompt: the user prompt to send.
        max_tokens: optional output cap (passed via --max-tokens if set).
        system: optional system prompt (passed via --append-system-prompt).
        timeout_s: subprocess timeout in seconds. Default 120.
        json_schema: optional JSON schema dict for structured output.
        cwd: optional working directory for the subprocess.
        disallowed_tools: comma-separated list of CC tools to block.
            Default disables all side-effect tools so the invocation is
            text-only (critical for autonomous use).
        binary: the `claude` executable path. Override for tests.

    Returns:
        Parsed JSON envelope. Key fields: type, subtype, is_error, result,
        structured_output, session_id, total_cost_usd, duration_ms,
        duration_api_ms, ttft_ms, num_turns, stop_reason, usage{...},
        modelUsage{...}, uuid.

    Note: check `envelope["subtype"] == "success"` for success detection;
    `is_error` has known mis-flag history (researcher source #18).
    """
    # phase-cycle-4 bugfix (2026-05-26): pass prompt via STDIN, not argv.
    # `--disallowedTools <tools...>` is variadic; a trailing positional
    # prompt gets consumed by the tool list and the CLI fails with
    # "Input must be provided either through stdin or as a prompt
    # argument when using --print" (cycle-3 unit tests mocked
    # subprocess.run and never exercised the parser). Researcher
    # ab1987d4ec80af4dd confirmed the canonical headless pattern is
    # stdin-piping. Do NOT add `--bare` -- per the same researcher's
    # Section 2: --bare rejects OAuth + keychain reads and requires
    # ANTHROPIC_API_KEY, which would break the Max-subscription rail.
    args: list[str] = [
        binary,
        "--print",
        "--output-format", "json",
        "--disallowedTools", disallowed_tools,
    ]
    if system is not None:
        args.extend(["--append-system-prompt", system])
    if json_schema is not None:
        args.extend(["--json-schema", json.dumps(json_schema)])
    if max_tokens is not None:
        args.extend(["--max-tokens", str(max_tokens)])

    logger.info(
        "claude_code_invoke: args=%d prompt_len=%d timeout_s=%d schema=%s",
        len(args), len(prompt), timeout_s, "yes" if json_schema else "no",
    )

    try:
        completed = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "claude_code_invoke: subprocess timeout after %ds prompt_len=%d",
            timeout_s, len(prompt),
        )
        raise ClaudeCodeError(
            f"claude CLI timeout after {timeout_s}s"
        ) from exc
    except FileNotFoundError as exc:
        logger.error(
            "claude_code_invoke: `%s` binary not found on PATH -- ensure Claude Code is installed and ~/.claude/ auth is configured",
            binary,
        )
        raise ClaudeCodeError(
            f"claude CLI not found at '{binary}' -- install Claude Code and ensure it is on PATH"
        ) from exc

    if completed.returncode != 0:
        logger.error(
            "claude_code_invoke: non-zero exit code=%d stderr=%s",
            completed.returncode,
            (completed.stderr or "")[:500],
        )
        raise ClaudeCodeError(
            f"claude CLI exited with code {completed.returncode}: {(completed.stderr or '')[:200]}"
        )

    stdout = completed.stdout or ""
    if not stdout.strip():
        logger.error("claude_code_invoke: empty stdout")
        raise ClaudeCodeError("claude CLI returned empty stdout")

    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.error(
            "claude_code_invoke: invalid JSON envelope (first 200 chars): %s",
            stdout[:200],
        )
        raise ClaudeCodeError(
            f"claude CLI returned invalid JSON: {exc}"
        ) from exc

    subtype = envelope.get("subtype")
    if subtype != "success":
        logger.warning(
            "claude_code_invoke: subtype=%s (expected 'success'); is_error=%s stop_reason=%s",
            subtype, envelope.get("is_error"), envelope.get("stop_reason"),
        )
        raise ClaudeCodeError(
            f"claude CLI envelope subtype='{subtype}' -- expected 'success'. stop_reason={envelope.get('stop_reason')}"
        )

    logger.info(
        "claude_code_invoke: success duration_ms=%s input_tokens=%s output_tokens=%s",
        envelope.get("duration_ms"),
        (envelope.get("usage") or {}).get("input_tokens"),
        (envelope.get("usage") or {}).get("output_tokens"),
    )
    return envelope


def extract_result_text(envelope: dict[str, Any]) -> str:
    """Pull the assistant-text result from a successful envelope.

    Prefer `structured_output` when present (set via --json-schema), else
    fall back to `result`. Returns empty string if neither populated.
    """
    structured = envelope.get("structured_output")
    if isinstance(structured, str) and structured:
        return structured
    if isinstance(structured, (dict, list)):
        return json.dumps(structured)
    result = envelope.get("result")
    if isinstance(result, str):
        return result
    return ""


# ---------------------------------------------------------------------------
# LLMClient adapter
# ---------------------------------------------------------------------------
# phase-cycle-3: make_client() expects an object adhering to the LLMClient
# interface (generate_content(prompt, generation_config) -> LLMResponse).
# Wrapping claude_code_invoke() in a class lets us drop into the existing
# orchestrator code path without touching every call site.
def _make_claude_code_client_class():
    # Imported lazily to avoid a hard cycle with llm_client (which imports
    # this module on demand from inside make_client).
    from backend.agents.llm_client import LLMClient, LLMResponse, UsageMeta

    class ClaudeCodeClient(LLMClient):
        """LLMClient that routes generate_content through the `claude` CLI.

        Used when settings.paper_use_claude_code_route=True so the
        autonomous-loop's Claude calls hit the Max-subscription flat-fee
        rail instead of api.anthropic.com direct billing.

        Capability flags: supports_thinking=False (CC doesn't surface a
        separate thoughts stream the same way the SDK does); supports_
        grounding=False (Claude doesn't do Google Search Grounding).
        """
        supports_thinking = False
        supports_grounding = False

        def __init__(self, model_name: str, timeout_s: int = 120):
            self.model_name = model_name
            self._timeout_s = timeout_s

        def generate_content(
            self,
            prompt: str,
            generation_config: Optional[dict] = None,
        ) -> "LLMResponse":
            config = generation_config or {}
            max_tokens = config.get("max_output_tokens")
            system = config.get("system") or config.get("system_instruction")
            json_schema = None
            if config.get("response_mime_type") == "application/json":
                # The orchestrator may pass a Pydantic model class as
                # response_schema. We surface no schema to the CLI (the
                # caller injects schema text into the prompt itself per
                # existing convention) but flip --json-schema when the
                # caller explicitly provides a dict.
                schema = config.get("response_schema")
                if isinstance(schema, dict):
                    json_schema = schema

            try:
                envelope = claude_code_invoke(
                    prompt,
                    max_tokens=max_tokens,
                    system=system,
                    timeout_s=self._timeout_s,
                    json_schema=json_schema,
                )
            except ClaudeCodeError as exc:
                logger.warning(
                    "ClaudeCodeClient: generate_content failed (%r); returning empty LLMResponse",
                    exc,
                )
                return LLMResponse(
                    text="",
                    thoughts=f"errored: {exc}",
                    usage_metadata=UsageMeta(),
                )

            text = extract_result_text(envelope)
            usage = envelope.get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            cache_read = int(usage.get("cache_read_input_tokens") or 0)
            cache_create = int(usage.get("cache_creation_input_tokens") or 0)

            return LLMResponse(
                text=text,
                usage_metadata=UsageMeta(
                    prompt_token_count=input_tokens,
                    candidates_token_count=output_tokens,
                    total_token_count=input_tokens + output_tokens,
                    cache_read_input_tokens=cache_read,
                    cache_creation_input_tokens=cache_create,
                ),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    return ClaudeCodeClient


# Lazily resolved at first access to avoid the import cycle with llm_client.
# Callers reference `ClaudeCodeClient` (capitalized) -- the module __getattr__
# below resolves it the first time.
def __getattr__(name: str):
    if name == "ClaudeCodeClient":
        cls = _make_claude_code_client_class()
        globals()["ClaudeCodeClient"] = cls
        return cls
    raise AttributeError(name)
