---
name: cc-rail-vs-claudeclient-78-1
description: CC-rail (claude -p) vs direct ClaudeClient API differences measured for phase-78.1 -- schema guarantee, temperature, max_tokens, system prompt, cwd context tax
metadata:
  type: project
---

The Claude Code CLI rail is NOT API-equivalent to `ClaudeClient`. Measured
2026-07-25 against official docs for masterplan step 78.1 (rewire the six
signal overlays onto the rail).

- **`--json-schema` is POST-HOC validated with internal re-prompting**, not
  constrained decoding. CLI reference: "Get validated JSON output matching a
  JSON Schema **after the agent completes its workflow**"; Agent SDK doc: "the
  SDK validates the output against it, re-prompting on mismatch. If validation
  does not succeed within the retry limit, the result is an error"
  (`subtype: error_max_structured_output_retries`). The API's
  `output_config.format` DOES guarantee via constrained decoding. Also: a run
  can end `subtype: success` with NO `structured_output` -- treat as failure.
  Schema dialect on the CLI is **draft-07**; a newer `$schema` declaration is
  rejected outright.
- **No `--temperature` and no output-token flag exist.** Only
  `--max-budget-usd` / `--max-turns`. So `temperature: 0.0` and
  `max_output_tokens` are silently unreachable on the rail (the repo already
  no-ops max_tokens at `claude_code_client.py:280` after ~63% of calls were
  rejected with `unknown option '--max-tokens'`).
- **`--bare` "will become the default for `-p` in a future release"** and bare
  mode skips OAuth/keychain, requiring `ANTHROPIC_API_KEY`. That flip would
  silently move the whole Max rail back to metered billing -- needs a
  `claude --version` tripwire.
- **Context tax:** without `--bare`, `claude -p` loads everything in the cwd.
  `claude_code_invoke` passes `cwd=None` and the backend's launchd
  `WorkingDirectory` is the repo root, so every rail call loads CLAUDE.md
  (34 KB), 8 MCP servers from `.mcp.json`, and the SessionStart /
  InstructionsLoaded hooks. `--disallowedTools` with bare tool names does NOT
  remove MCP tools (needs `mcp__*` or `--strict-mcp-config`). External
  benchmark (e-INFRA CZ, Apr 2026, 600+ headless runs): CC system prompt
  ~21K tokens, median native-Claude latency 32-64s per invocation.
- **Prompt cache TTL is ~5 min of inactivity**, and each `claude -p` is a fresh
  session, so once-per-cycle callers are always cold.
- **Agent tagging:** `claude_code_client.py:504` emits `cc_rail:<role>` only
  when the caller passes `_role`/`_agent` in `generation_config`; otherwise the
  BARE `cc_rail`. No repo caller passes `config["system"]` either, so
  `--append-system-prompt` is dead and `_HOUSE_INSTRUCTIONS` is absent from
  every rail call today.

**Why:** 78.1 was scoped as "no behavior change to the signals", and four of
these differences ARE behavior changes that a constructor-swap diff hides.

**How to apply:** any step that moves a call site onto the rail must enumerate
these, not assume parity. See `handoff/current/research_brief_78.1.md` for the
full per-service table. Related: [[project_gemini_lifecycle_pipeline_restoration]],
[[project_cc_rail_guard_66_1]], [[project_cost_truth_66_3]].
