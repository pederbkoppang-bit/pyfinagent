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
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# phase-cycle-5 (2026-05-26): the launchd-supervised backend does NOT
# inherit the operator's interactive-shell PATH (claude typically lives
# at ~/.local/bin/claude, not on /usr/bin or /usr/local/bin). subprocess
# defaults to execvp which only searches the inherited PATH, so a bare
# "claude" arg fails with FileNotFoundError. Resolve at call-time using
# shutil.which() against an explicit search list (env override + common
# install locations) and fall through to "claude" so test mocks still
# see the literal binary name.
_DEFAULT_SEARCH_PATHS = [
    os.environ.get("CLAUDE_CODE_BINARY"),
    os.path.expanduser("~/.local/bin/claude"),
    "/opt/homebrew/bin/claude",
    "/usr/local/bin/claude",
]


def _resolve_claude_binary(binary: str) -> str:
    """Return an absolute path to the claude CLI, or `binary` as a fallback.

    Searches in order:
    1. The exact path passed in (if it already resolves via which or exists).
    2. CLAUDE_CODE_BINARY env override.
    3. Common install locations (~/.local/bin, Homebrew, /usr/local/bin).
    4. Original `binary` string as last-resort fallback (preserves test
       mock behavior so unit tests can patch subprocess.run without
       caring about absolute paths).
    """
    if binary and (os.path.isabs(binary) and os.path.isfile(binary)):
        return binary
    resolved = shutil.which(binary)
    if resolved:
        return resolved
    for candidate in _DEFAULT_SEARCH_PATHS:
        if candidate and os.path.isfile(candidate):
            return candidate
    return binary


class ClaudeCodeError(RuntimeError):
    """Raised when the `claude` CLI returns a non-success envelope."""


# ── phase-66.1: rail guard (probe gate + circuit breaker) ────────────────
# The 2026-06-15..07-06 outage fired ~162 doomed 5s subprocess calls per
# cycle for three weeks with zero pages. The guard makes that structurally
# impossible: a failed pre-cycle probe disables the rail for the cycle, and
# a consecutive-failure breaker trips mid-cycle with exactly ONE P1 page
# (caller-side latch -- P1s bypass the AlertDeduper by design, phase-62.7).
# Healthy-path behavior is byte-identical: the guard engages only on probe
# failure or after the failure threshold.
@dataclass
class _RailGuardState:
    cycle_id: Optional[str] = None
    disabled_reason: Optional[str] = None  # probe gate (set by the loop)
    consecutive_failures: int = 0
    open: bool = False                     # breaker tripped this cycle
    paged: bool = False                    # exactly-once page latch
    last_error: str = ""
    skipped_calls: int = 0


_RAIL_GUARD = _RailGuardState()
_RAIL_GUARD_LOCK = threading.Lock()


def _rail_breaker_threshold() -> int:
    try:
        from backend.config.settings import get_settings

        return int(getattr(get_settings(), "claude_rail_breaker_threshold", 20))
    except Exception:  # settings must never break the rail
        return 20


def rail_guard_reset(cycle_id: Optional[str] = None) -> None:
    """Per-cycle reset (Azure circuit-breaker per-window semantics).

    Called by the autonomous loop at cycle start, BEFORE the health probe.
    """
    global _RAIL_GUARD
    with _RAIL_GUARD_LOCK:
        _RAIL_GUARD = _RailGuardState(cycle_id=cycle_id)


def rail_guard_disable(reason: str) -> None:
    """Probe-gate: skip ALL rail calls for this cycle.

    The loop's own probe-failure branch already raises the P1 (site
    autonomous_loop rail-probe), so the latch is consumed here -- the guard
    must not double-page the same rail-down incident.
    """
    with _RAIL_GUARD_LOCK:
        _RAIL_GUARD.disabled_reason = reason[:400]
        _RAIL_GUARD.paged = True


def rail_guard_status() -> dict[str, Any]:
    with _RAIL_GUARD_LOCK:
        return {
            "cycle_id": _RAIL_GUARD.cycle_id,
            "rail_skipped": _RAIL_GUARD.disabled_reason is not None,
            "breaker_tripped": _RAIL_GUARD.open,
            "consecutive_failures": _RAIL_GUARD.consecutive_failures,
            "skipped_calls": _RAIL_GUARD.skipped_calls,
            "disabled_reason": _RAIL_GUARD.disabled_reason,
            "last_error": _RAIL_GUARD.last_error,
        }


def _rail_guard_blocked() -> Optional[str]:
    """Reason the rail is blocked right now, else None."""
    with _RAIL_GUARD_LOCK:
        if _RAIL_GUARD.disabled_reason is not None:
            _RAIL_GUARD.skipped_calls += 1
            return f"probe gate: {_RAIL_GUARD.disabled_reason}"
        if _RAIL_GUARD.open:
            _RAIL_GUARD.skipped_calls += 1
            return (
                f"breaker open after {_RAIL_GUARD.consecutive_failures} "
                f"consecutive failures: {_RAIL_GUARD.last_error}"
            )
    return None


def _rail_guard_record_success() -> None:
    with _RAIL_GUARD_LOCK:
        _RAIL_GUARD.consecutive_failures = 0


def _rail_guard_record_failure(error: str) -> None:
    """Count a real (subprocess-attempted) failure; trip + page on threshold.

    The page fires on the closed->open TRANSITION only (Fowler/PagerDuty
    alert-on-transition), via the live-proven bot-token path. Fail-open:
    paging problems never break the rail's own error handling.
    """
    threshold = _rail_breaker_threshold()
    should_page = False
    with _RAIL_GUARD_LOCK:
        _RAIL_GUARD.consecutive_failures += 1
        _RAIL_GUARD.last_error = error[:400]
        if _RAIL_GUARD.consecutive_failures >= threshold and not _RAIL_GUARD.open:
            _RAIL_GUARD.open = True
            if not _RAIL_GUARD.paged:
                _RAIL_GUARD.paged = True
                should_page = True
        cycle_id = _RAIL_GUARD.cycle_id
        n = _RAIL_GUARD.consecutive_failures
        last = _RAIL_GUARD.last_error
    if should_page:
        try:
            from backend.services.observability.alerting import raise_cron_alert_sync

            raise_cron_alert_sync(
                source="claude_code_rail",
                error_type="breaker_open",
                severity="P1",
                title=(
                    f"Claude Code rail breaker OPEN -- {n} consecutive "
                    f"failures; remaining rail calls skipped this cycle"
                ),
                details={
                    "cycle_id": cycle_id or "unknown",
                    "consecutive_failures": n,
                    "threshold": threshold,
                    "last_error": last,
                    "consequence": "rail calls skipped for the rest of the cycle; pipeline runs degraded fallbacks (policy: hold)",
                    "operator_action": "check `claude auth status` on the host; see docs/runbooks/claude-rail-degraded-mode.md",
                },
            )
        except Exception as page_exc:  # paging must never break the rail
            logger.warning("rail breaker page failed (non-fatal): %r", page_exc)


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
    resolved_binary = _resolve_claude_binary(binary)
    args: list[str] = [
        resolved_binary,
        "--print",
        "--output-format", "json",
        "--disallowedTools", disallowed_tools,
    ]
    if system is not None:
        args.extend(["--append-system-prompt", system])
    if json_schema is not None:
        args.extend(["--json-schema", json.dumps(json_schema)])
    # phase-cycle-5 follow-up (2026-05-26): --max-tokens is the SDK option
    # name, NOT the CLI flag. The `claude` CLI uses model-default ceilings
    # (32K for Haiku, 64K for Opus, 4K for Sonnet via Max plan) and exposes
    # --max-budget-usd <amount> instead. Q/A cycle-5 caught that ~63% of
    # calls were rejected with "error: unknown option '--max-tokens'".
    # Drop the flag entirely; callers that need a tight output budget can
    # use --json-schema for structured output or rely on the prompt.
    _ = max_tokens  # accepted but no-op at the CLI layer; preserved in signature for API-compat

    logger.info(
        "claude_code_invoke: args=%d prompt_len=%d timeout_s=%d schema=%s",
        len(args), len(prompt), timeout_s, "yes" if json_schema else "no",
    )

    try:
        # phase-38.13.1 (cycle 11, 2026-05-27): scrub ANTHROPIC_API_KEY +
        # ANTHROPIC_AUTH_TOKEN from the subprocess env. Per Anthropic CLI
        # auth precedence (code.claude.com/docs/en/authentication), API-key
        # env vars OUTRANK ~/.claude/ OAuth -- so without the scrub the CLI
        # bills against the (credit-exhausted) direct-API account instead
        # of the Max subscription. Cycle-8 was a routing observability fix
        # only; this is the actual rail wiring.
        scrubbed_env = {
            k: v for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
        }
        completed = subprocess.run(
            args,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
            check=False,
            env=scrubbed_env,
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
        # phase-66.2 (2026-07-08): on failure the CLI's diagnostic usually
        # lives on STDOUT (a JSON error envelope or a plain-text limit
        # message like "You've hit your session limit"), NOT stderr -- the
        # 07-07 quota-exhaustion burst logged 65 failures as "stderr=" (empty)
        # and the cause was only recoverable from the away-session JSON.
        # Log + raise with both streams so limit/auth/API errors are
        # identifiable from backend.log alone.
        _out_snip = (completed.stdout or "").strip()[:300]
        logger.error(
            "claude_code_invoke: non-zero exit code=%d stderr=%s stdout=%s",
            completed.returncode,
            (completed.stderr or "")[:300],
            _out_snip,
        )
        raise ClaudeCodeError(
            f"claude CLI exited with code {completed.returncode}: "
            f"{(completed.stderr or '')[:150]} | stdout: {_out_snip[:150]}"
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


def claude_code_health_probe(binary: str = "claude", timeout_s: int = 15) -> tuple[bool, str]:
    """phase-56.2 (55.3 finding F-4): free, token-less health probe of the
    claude-CLI OAuth rail.

    Runs `claude auth status` in the SAME scrubbed env the real rail uses
    (no ANTHROPIC_API_KEY -- probing the OAuth/keychain path, not the direct
    API). Consumes no tokens. Returns (ok, detail). NEVER raises -- callers
    alert on ok=False; a probe bug must not break a trading cycle.

    Rationale: the 2026-06-01..06-09 away week ran with this rail silently
    down (expired OAuth session in unattended mode); no health check
    distinguished "rail down" from "no work" (55.2 F-A1; OneUptime 2026
    heartbeat pattern: probe the path you actually use).
    """
    try:
        resolved_binary = _resolve_claude_binary(binary)
        scrubbed_env = {
            k: v for k, v in os.environ.items()
            if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
        }
        completed = subprocess.run(
            [resolved_binary, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=scrubbed_env,
        )
    except subprocess.TimeoutExpired:
        return False, f"auth-status probe timeout after {timeout_s}s"
    except FileNotFoundError:
        return False, f"claude CLI not found at '{binary}'"
    except Exception as exc:  # defensive: the probe must never raise
        return False, f"probe error: {exc}"

    if completed.returncode != 0:
        return False, (
            f"auth status exit={completed.returncode}: "
            f"{(completed.stderr or completed.stdout or '')[:200]}"
        )
    out = (completed.stdout or "")
    # Belt-and-braces: when the CLI emits the JSON-ish status, require the
    # loggedIn flag; exit-0 alone is the primary signal.
    if '"loggedIn"' in out and '"loggedIn": true' not in out and '"loggedIn":true' not in out:
        return False, "auth status reports loggedIn != true"
    return True, "ok"


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
        # phase-60.1 (AW-4): the CLI round-trip routinely takes 60-90s
        # (observed 88.9s live 2026-06-11), so the orchestrator's default
        # 90s per-step budget races the rail itself -- the away week's "90s
        # agent timeouts" leg. Declare the rail's latency profile so
        # _generate_with_retry lifts its budget ABOVE this client's own
        # 120s subprocess timeout (the CLI timeout then fails first and is
        # retried, instead of the step giving up mid-flight).
        recommended_step_timeout = 150

        def __init__(self, model_name: str, timeout_s: int = 150):
            self.model_name = model_name
            self._timeout_s = timeout_s
            # phase-61.2 (criterion 2): keep the orchestrator's per-step
            # budget ABOVE the subprocess timeout for any configured value,
            # not just the class-attribute default -- 150/150 would recreate
            # the race the phase-60.1 note above warns about.
            self.recommended_step_timeout = timeout_s + 30

        @staticmethod
        def _log_cc_call(envelope, *, agent, ticker, latency_ms, model, ok):
            """phase-60.4 (AW-7, criterion 1): llm_call_log row for the CC
            rail. The rail that did ALL the away-week deciding logged ZERO
            rows (claude_code_client had no writer) -- burn and firing audits
            were blind to the working rail. Cost is the flat-fee rail:
            session_cost_usd delta 0 (tokens still recorded for volume
            audits). Never raises (mirrors the 56.2 lite-path helper)."""
            try:
                from backend.services.observability.api_call_log import log_llm_call

                usage = (envelope or {}).get("usage") or {}
                log_llm_call(
                    provider="anthropic",
                    model=model,
                    agent=f"cc_rail:{agent}" if agent else "cc_rail",
                    latency_ms=float(latency_ms),
                    input_tok=int(usage.get("input_tokens") or 0),
                    output_tok=int(usage.get("output_tokens") or 0),
                    cache_creation_tok=int(usage.get("cache_creation_input_tokens") or 0),
                    cache_read_tok=int(usage.get("cache_read_input_tokens") or 0),
                    request_id=str((envelope or {}).get("session_id") or "") or None,
                    ok=ok,
                    ticker=ticker,
                )
            except Exception as log_exc:  # observability never breaks the rail
                logger.debug("ClaudeCodeClient: llm_call_log write skipped (%r)", log_exc)

        def generate_content(
            self,
            prompt: str,
            generation_config: Optional[dict] = None,
        ) -> "LLMResponse":
            config = generation_config or {}
            max_tokens = config.get("max_output_tokens")
            system = config.get("system") or config.get("system_instruction")
            # phase-60.4: orchestrator side-channel labels (same convention
            # the SDK clients consume) so the rail's rows carry agent + ticker.
            _agent = config.get("_role") or config.get("_agent")
            _ticker = config.get("_ticker")
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

            # phase-66.1: rail guard -- probe gate / open breaker means NO
            # subprocess spawn and NO llm_call_log row (the skip is cycle-
            # level state, recorded once via rail_guard_status; per-call
            # rows for skips would re-create the 06-17/18 phantom-row spam).
            _blocked = _rail_guard_blocked()
            if _blocked is not None:
                return LLMResponse(
                    text="",
                    thoughts=f"rail_guard_skipped: {_blocked}",
                    usage_metadata=UsageMeta(),
                )

            _t0 = time.monotonic()
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
                self._log_cc_call(
                    None, agent=_agent, ticker=_ticker,
                    latency_ms=(time.monotonic() - _t0) * 1000.0,
                    model=self.model_name, ok=False,
                )
                _rail_guard_record_failure(str(exc))
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

            self._log_cc_call(
                envelope, agent=_agent, ticker=_ticker,
                latency_ms=(time.monotonic() - _t0) * 1000.0,
                model=self.model_name, ok=True,
            )
            _rail_guard_record_success()

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
