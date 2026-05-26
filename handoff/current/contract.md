# Contract -- Cycle 3: Claude Code CLI routing layer (operator-approved)

**Cycle:** 3 (production-readiness mode + testing-phase trading mandate)
**Date:** 2026-05-26
**Step targeted:** unblocks `27.6` "End-to-end smoke verify: full path on Claude" (P0). New capability not previously enumerated in masterplan; future cycle can map to a step if desired.
**Class:** trading-policy-adjacent (changes the LLM rail driving every recommendation). Citation floor APPLIES: >=2 AI-in-trading + >=2 academic.

**File-collision note (FIFTH occurrence today):** `handoff/current/contract.md` was overwritten by the autonomous-loop's parameter-optimization sprint contract at 19:56, 20:36, 20:47, 22:47, and 21:02:13 UTC. Layer-3 harness Main re-wrote the trading-cycle content each time. Permanent deconfliction (separate paths or discriminator field) is on the follow-up backlog. This document supersedes the parameter-optimization stub for cycle 3.

## North star

Per `project_system_goal`: maximize profit at lowest cost live. Operator approved 2026-05-26: route the autonomous-loop's LLM rail through the Claude Code CLI for the testing phase. Cost rationale: Max-subscription flat-fee tier covers `claude --print --output-format json` invocations; bypasses the credit-exhausted `api.anthropic.com` direct rail; unblocks 13 ticker analyses per cycle without any per-token charge during testing.

## Research gate

- Researcher `aff3444de945e98c2`, tier=deep, 24 sources read in full, 34 URLs collected, 3 adversarial sources, recency scan performed, **gate_passed=true**.
- Brief: `handoff/current/research_brief_phase_claude_code_routing.md`.
- AI-in-trading citations (>=2 required): **2 cited** -- TradingAgents Multi-Agents LLM Framework (`arXiv:2412.20138` Tauric Research v0.2.0 Feb 2026); Portkey AI Gateway (10B+ req/mo, 99.9999% uptime, ToS-compliant LLM-rail abstraction in production).
- Academic-method citations (>=2 required): **3 cited** -- Bailey/Borwein/Lopez de Prado/Zhu "Probability of Backtest Overfitting" (SSRN 2326253); Harvey/Liu/Zhu NBER w20592 (RFS 2016); Yin et al. "Implementation Risk in Portfolio Backtesting" (`arXiv:2603.20319`) -- justifies per-row engine provenance logging.

## N* delta

- **B primary:** unblock the autonomous-loop without operator action on credits. Backend's 13-ticker analysis pipeline routes through `claude` CLI behind a feature flag. Max-subscription flat-fee rail honored from non-CLI subprocess (verified live 2026-05-26 by researcher: no `ANTHROPIC_API_KEY` env var needed; uses `~/.claude/` auth).
- **R secondary:** feature flag defaults OFF so existing Anthropic-direct path stays untouched. Operator opt-in via `paper_use_claude_code_route=true` flip. Per-row engine provenance logged so future analysis-quality A/B comparisons (Claude Code rail vs Anthropic-direct rail) are statistically clean per Yin et al. 2026 implementation-risk framework.

## Empirical findings from research (researcher Section 1-4)

### Verified `claude` CLI invocation pattern (live-tested 2026-05-26)

```
claude --bare --print --output-format json \
       --append-system-prompt "<role system>" \
       --json-schema '<schema>' \
       --disallowedTools "Bash,Edit,Write,Read,Glob,Grep,Agent" \
       "<prompt>"
```

- `--bare` flag suppresses interactive shell init.
- `--print` returns single JSON envelope and exits.
- `--output-format json` -- structured envelope.
- `--disallowedTools` -- locks the invocation to text-only (no side effects). Critical for autonomous use.
- Max-subscription auth via `~/.claude/` -- no env-var required.
- `total_cost_usd` field reported but NOT billed under Max flat-fee.

### JSON output envelope (key fields)

```
type, subtype, is_error, result, structured_output, session_id,
total_cost_usd, duration_ms, duration_api_ms, ttft_ms, num_turns,
stop_reason, usage{input_tokens, output_tokens, cache_read_input_tokens,
cache_creation_input_tokens}, modelUsage{<model>:{...}}, uuid
```

**Critical:** check `subtype == "success"` for success detection -- `is_error` has known mis-flag history (researcher source #18, GitHub issue).

### Rate limits (Max plan)

Per researcher Section 2: 13 concurrent invocations per cycle is well within Max-plan ceilings provided we keep request-level concurrency <= 5 (the empirical safe ceiling per truefoundry.com 2026 Claude-Code-limits writeup). Cycle 3 lowers `_concurrency` from current default to 3 as a conservative ramp.

## Scope -- 1 new file + 3 modified backend files + 1 settings field

### NEW

1. `backend/agents/claude_code_client.py` -- new `ClaudeCodeClient(LLMClient)` subclass + standalone `claude_code_invoke()` function. Signature:

   ```python
   def claude_code_invoke(
       prompt: str,
       *,
       max_tokens: int | None = None,
       system: str | None = None,
       timeout_s: int = 120,
       json_schema: dict | None = None,
       cwd: str | None = None,
       disallowed_tools: str = "Bash,Edit,Write,Read,Glob,Grep,Agent",
   ) -> dict[str, Any]:
       """Invoke `claude --print --output-format json` as a subprocess.
       Returns the parsed JSON envelope. Checks subtype=='success'."""
   ```

   Uses `subprocess.run` for sync path; async variant via `asyncio.create_subprocess_exec` for orchestrator concurrent fan-out. ASCII-only log messages per `backend-services.md::Logging`.

### MODIFIED

2. `backend/config/settings.py` -- new field near `anthropic_api_key`:
   ```python
   paper_use_claude_code_route: bool = Field(
       False,
       description="Route the autonomous-loop LLM rail through the Claude Code CLI ..."
   )
   ```
   Default OFF so existing Anthropic-direct path stays untouched. Operator opt-in.

3. `backend/agents/llm_client.py` -- in `make_client()` (around the existing Anthropic-vs-Gemini branching, researcher confirms `:1888-1890`):
   - Gate a new branch BEFORE the existing `ClaudeClient` branch.
   - When `settings.paper_use_claude_code_route` is True AND model is a Claude variant, return `ClaudeCodeClient(...)` instead of `ClaudeClient(...)`.

4. `backend/services/autonomous_loop.py` -- in `_run_claude_analysis` (researcher confirms `:1438-1442`):
   - When `settings.paper_use_claude_code_route` is True, route the analysis through the new client.
   - Preserve the existing direct `anthropic.Anthropic()` path when the flag is False (default).
   - Log the rail used per analysis: `logger.info("Analysis ticker=%s rail=%s", ticker, "claude_code" if route else "anthropic_direct")`.

### DEFERRED to follow-up cycle

- BQ schema column `paper_trades.signals.claude_code_route BOOL` -- researcher recommended for per-row engine provenance per Yin et al. 2026. Out of cycle-3 scope (BQ schema change is operator-gated per CLAUDE.md "BQ schema mutations outside autonomous-loop Step 7 stay operator-gated"). Log to backend.log only this cycle.

## Tests (new + existing)

- NEW `backend/tests/test_claude_code_client.py` -- 3 cases:
  1. `test_claude_code_invoke_returns_envelope` -- mocks subprocess, asserts `result` field extracted from JSON envelope.
  2. `test_claude_code_invoke_raises_on_error_subtype` -- mocks `subtype != "success"`, asserts exception.
  3. `test_claude_code_invoke_handles_timeout` -- mocks subprocess timeout, asserts graceful handling.
- AST parse: new + modified .py files.
- Regression: existing `pytest backend/tests/ -k "llm_client or autonomous_loop"` still passes (the flag defaults OFF, so existing behavior preserved).

## Immutable success criteria

1. `backend/agents/claude_code_client.py` exists with `claude_code_invoke` function.
2. `backend/config/settings.py` has `paper_use_claude_code_route` field with default `False`.
3. `pytest backend/tests/test_claude_code_client.py -v` -- all 3 cases pass.
4. `pytest backend/tests/ -k "llm_client or autonomous_loop"` regression -- no new failures.
5. `python -c "import ast; ast.parse(open('backend/agents/claude_code_client.py').read())"` exit 0.
6. `python -c "import ast; ast.parse(open('backend/config/settings.py').read())"` exit 0.
7. `python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"` exit 0.
8. `python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read())"` exit 0.
9. `grep -c "claude_code_invoke" backend/agents/claude_code_client.py` >= 1.
10. `grep -c "paper_use_claude_code_route" backend/agents/llm_client.py` >= 1 (the gating check).
11. `grep -c "paper_use_claude_code_route" backend/services/autonomous_loop.py` >= 1.
12. `grep -c "rail=" backend/services/autonomous_loop.py` >= 1 (the per-analysis rail log).
13. ZERO frontend changes.
14. ZERO new npm deps.
15. NO `npm run build`, NO `rm -rf .next/*`.
16. ZERO emojis introduced.
17. ASCII-only log messages in `claude_code_client.py`.
18. Feature flag defaults to `False` so existing Anthropic-direct path unaffected.
19. Cycle-3 self contract.md content is on disk at commit (collision-protected via re-write if autonomous-loop clobbers).

## /goal integration gates

1. pytest green. 2. AST parse green. 3. Citation gate (>=2 AI-in-trading + >=2 academic) verified above. 4. Log LAST. 5. No self-evaluation. 6. North-star aligned (flat-fee rail saves test-phase token cost; per-row rail logging enables future A/B integrity).
