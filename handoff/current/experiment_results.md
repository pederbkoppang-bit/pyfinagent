# Experiment Results -- Cycle 3: Claude Code CLI routing layer

**Date:** 2026-05-26
**Phase:** trading-policy-adjacent (changes the LLM rail driving all recommendations). Operator-approved.
**Result:** GENERATE complete; awaiting Q/A.

## What changed (1 new file + 1 new test + 3 modified backend files + 1 settings field)

### NEW

1. `backend/agents/claude_code_client.py` -- `claude_code_invoke()` standalone function + `ClaudeCodeClient(LLMClient)` adapter class.
   - `claude_code_invoke(prompt, max_tokens, system, timeout_s, json_schema, cwd, disallowed_tools, binary)` shells out to `claude --print --output-format json --disallowedTools <list> [...]`. Returns parsed JSON envelope; raises `ClaudeCodeError` on subtype!='success', non-zero exit, timeout, missing binary, or invalid JSON. ASCII-only log messages.
   - `extract_result_text(envelope)` -- pulls assistant text (prefer `structured_output`, fall back to `result`).
   - `ClaudeCodeClient(LLMClient)` -- generate_content(prompt, generation_config) -> LLMResponse adapter so make_client() can drop in. Lazily resolved via module __getattr__ to avoid the import cycle with llm_client. Errors return an empty-text LLMResponse with thoughts='errored: ...' (existing convention).

2. `backend/tests/test_claude_code_client.py` -- 11 pytest cases mocking subprocess.run:
   - happy path returns envelope
   - subtype!=success raises ClaudeCodeError
   - timeout raises ClaudeCodeError
   - non-zero exit raises ClaudeCodeError
   - missing `claude` binary raises ClaudeCodeError with actionable message
   - invalid JSON raises ClaudeCodeError
   - extract_result_text prefers structured_output
   - extract_result_text falls back to result
   - extract_result_text returns empty when missing
   - ClaudeCodeClient adapter returns valid LLMResponse with token counts
   - ClaudeCodeClient adapter returns empty-text LLMResponse on error

### MODIFIED

3. `backend/config/settings.py` -- added `paper_use_claude_code_route: bool = Field(False, ...)` next to `anthropic_api_key`. Default OFF; operator opt-in. Citations in field-doc comment (TradingAgents, Portkey, Bailey/Borwein/LdP/Zhu PBO, Yin et al. implementation-risk).

4. `backend/agents/llm_client.py::make_client()` -- new branch BEFORE the existing direct-Anthropic branch: when `settings.paper_use_claude_code_route` is True AND `model_name.startswith("claude-")`, return `ClaudeCodeClient(model_name=...)` instead of `ClaudeClient(...)`. ImportError caught and logged; falls through to Anthropic-direct as defense in depth.

5. `backend/services/autonomous_loop.py::_run_claude_analysis()` -- entry-point rail-log + dual-rail dispatch:
   - Added `logger.info("Analysis ticker=%s rail=%s", ticker, "claude_code" if use_claude_code_route else "anthropic_direct")` per Yin et al. 2026 implementation-risk framework.
   - Gated `anthropic.Anthropic(api_key=...)` instantiation behind `not use_claude_code_route`.
   - Wrapped both LLM calls (trader analysis at L1465-1473 + risk judge at L1502-1515) in `if use_claude_code_route` branches that route through `claude_code_invoke` instead of `client.messages.create`. Async-safe via `asyncio.to_thread`.

## Verification (verbatim)

```
$ pytest backend/tests/test_claude_code_client.py -v
11 passed in 0.23s

$ pytest backend/tests/ -k "llm_client or autonomous_loop or claude_code"
33 passed, 597 deselected, 1 warning in 2.65s

$ python -c "import ast; ast.parse(open('backend/agents/claude_code_client.py').read())"
(exit 0)
$ python -c "import ast; ast.parse(open('backend/config/settings.py').read())"
(exit 0)
$ python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"
(exit 0)
$ python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read())"
(exit 0)

$ grep -c "claude_code_invoke" backend/agents/claude_code_client.py
6  (function def + multiple references inside the file)

$ grep -c "paper_use_claude_code_route" backend/agents/llm_client.py
1

$ grep -c "paper_use_claude_code_route" backend/services/autonomous_loop.py
1

$ grep -c "rail=" backend/services/autonomous_loop.py
1
```

## Operator opt-in

Once cycle 3 lands, the operator flips the flag via:
- Settings UI -- the new `paper_use_claude_code_route` toggle.
- OR backend `PUT /api/settings/ {"paper_use_claude_code_route": true}` (FastAPI Pydantic validation handles the bool cast).
- Confirm `claude --version` works in the backend's environment first (the launchd-supervised process needs `claude` on PATH; researcher confirms Max-subscription auth resolves from `~/.claude/`).

Cycle 4 candidate: run a smoke cycle with the flag ON, observe rail=claude_code logs, confirm analyses persisted to BQ, then re-run step 27.6 verification.

## Citations (>=2 AI-in-trading + >=2 academic per goal mandate)

AI-in-trading (2):
- TradingAgents Multi-Agents LLM Framework (`arXiv:2412.20138` Tauric Research v0.2.0 Feb 2026).
- Portkey AI Gateway (10B+ req/mo, 99.9999% uptime, failover-routing canonical).

Academic methods (3):
- Bailey/Borwein/Lopez de Prado/Zhu "Probability of Backtest Overfitting" (SSRN 2326253).
- Harvey/Liu/Zhu NBER w20592 (RFS 2016).
- Yin et al. "Implementation Risk in Portfolio Backtesting" (`arXiv:2603.20319`) -- justifies per-row engine provenance logging.

Brief: `handoff/current/research_brief_phase_claude_code_routing.md`.

## Memory-rule compliance

- ZERO frontend changes.
- ZERO new npm deps.
- NO `npm install`, NO `npm run build`, NO `rm -rf .next/*`.
- ZERO emojis introduced.
- ASCII-only log messages in `claude_code_client.py`.
- Feature flag defaults to `False` so existing Anthropic-direct path unaffected.

## Not in scope

- BQ schema column `paper_trades.signals.claude_code_route BOOL` (operator-gated BQ schema change per CLAUDE.md).
- Cycle-4 smoke run with the flag ON (operator opt-in needed; that's the next cycle).
- Shared-credit anti-pattern split-keys remediation (researcher Section 7 follow-up).
- Removing the autonomous-loop sprint-contract overwrite of contract.md (collision deconfliction; follow-up backlog).
