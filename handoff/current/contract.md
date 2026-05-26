# Contract -- Cycle 4: claude_code_invoke stdin-pipe bugfix

**Cycle:** 4 (production-readiness mode + testing-phase trading mandate)
**Date:** 2026-05-26
**Class:** bugfix on cycle-3 ship (the routing layer's CLI invocation pattern was wrong; live smoke caught it). NOT a trading-policy change; citation floor does NOT apply.
**Status flip:** NONE -- this fix lands the prerequisite for the operator-opt-in smoke test that will close step 27.6.

**File-collision note (SIXTH occurrence today):** `handoff/current/contract.md` was clobbered by the autonomous-loop sprint contract at 19:56, 20:36, 20:47, 22:47, 21:02, and likely one more by the time this writes. Layer-3 harness Main re-wrote each time. Backlog deconfliction stands.

## Research gate

- Researcher `ab1987d4ec80af4dd`, tier=simple, 5 sources read in full, 7 snippet-only, 12 URLs, recency scan performed, internal_files_inspected=2, **gate_passed=true**.
- Brief: `handoff/current/research_brief_phase_claude_code_stdin_fix.md`.

## Empirical bug (verified live)

Cycle 3 ships `claude --print --output-format json --disallowedTools "..." "<prompt>"`. Live smoke 2026-05-26 returned:

```
Error: Input must be provided either through stdin or as a prompt argument when using --print
```

Root cause: `--disallowedTools <tools...>` is variadic per the CLI reference; the trailing positional prompt gets consumed by the tool list. The cycle-3 unit tests passed because `subprocess.run` was mocked -- the tests never exercised the actual CLI parser.

Stdin-piping smoke worked first try (verified):

```
$ echo 'Say "ok" and nothing else.' | claude --print --output-format json --disallowedTools "Bash,Edit,Write,Read,Glob,Grep,Agent"
{"type":"result","subtype":"success","result":"ok",...}
```

## Critical clarification (researcher Section 2)

**`--bare` MUST NOT be added.** Per `code.claude.com/docs/en/headless`: `--bare` skips OAuth + keychain reads and REQUIRES `ANTHROPIC_API_KEY`. Adding it would shift billing back to the credit-exhausted API-key rail and defeat the entire cycle-3 purpose. Current cycle-3 code correctly OMITS `--bare` -- preserve that.

The `total_cost_usd` field in the envelope (e.g., $0.32 for the smoke "say ok") is API-equivalent REPORTING, not actual Max billing. Max plan covers Agent SDK use through the flat-fee rail (and Anthropic's published transition to Agent SDK credit pool starts June 15, 2026 -- not yet active).

## N* delta

- **B primary:** the cycle-3 routing layer becomes operationally invokable. Without this fix, flipping `paper_use_claude_code_route=true` would break every Claude ticker analysis. With it, the autonomous loop can route through the Max-subscription rail end-to-end.
- **R secondary:** ZERO behavioral change beyond what cycle 3 already promised. Pure correctness fix.

## Scope -- 1 modified file + tests + 1 live smoke

### MODIFIED

1. `backend/agents/claude_code_client.py::claude_code_invoke`:
   - REMOVE the `args.append(prompt)` line.
   - PASS `input=prompt` kwarg to `subprocess.run`.
   - Keep `--print --output-format json --disallowedTools "..."` and optional `--append-system-prompt`, `--json-schema`, `--max-tokens` as-is.
   - Do NOT add `--bare` (would break Max-subscription auth per researcher Section 2).

2. `backend/tests/test_claude_code_client.py`:
   - Update mocks to assert `subprocess.run` is called with `input=prompt` kwarg (not prompt-in-args).
   - Add one new test: `test_claude_code_invoke_passes_prompt_via_stdin_not_argv` -- explicit regression guard against the cycle-3 bug.

### LIVE SMOKE

3. Execute a Python smoke script that imports `claude_code_invoke` and calls it from the venv. Capture the returned envelope verbatim. Confirm:
   - `subtype == "success"`
   - `result` is non-empty
   - `usage.output_tokens > 0`
   - No subprocess errors raised.
   - Write the verbatim envelope into `handoff/current/live_check_cycle_4_stdin_fix.md` as an audit-grade artifact.

## Immutable success criteria

1. `python -c "import ast; ast.parse(open('backend/agents/claude_code_client.py').read())"` exit 0.
2. `python -c "import ast; ast.parse(open('backend/tests/test_claude_code_client.py').read())"` exit 0.
3. `pytest backend/tests/test_claude_code_client.py -v` -- all tests pass (existing 11 + the new stdin-guard test).
4. `pytest backend/tests/ -k "llm_client or autonomous_loop or claude_code"` -- regression clean.
5. `grep -c "args.append(prompt)" backend/agents/claude_code_client.py` -- returns 0 (line removed).
6. `grep -c "input=prompt" backend/agents/claude_code_client.py` -- returns 1.
7. `grep -c "\\-\\-bare" backend/agents/claude_code_client.py` -- returns 0 (defensive against accidental re-introduction).
8. `grep -c "input=prompt\\|input=" backend/tests/test_claude_code_client.py` -- the new test passes a stdin-input assertion.
9. `handoff/current/live_check_cycle_4_stdin_fix.md` exists with verbatim envelope.
10. ZERO frontend changes.
11. ZERO new npm deps.
12. NO `npm run build`, NO `rm -rf .next/*`.
13. ZERO emojis introduced.

## /goal integration gates

1. pytest green. 2. AST parse green. 3. Live smoke evidence captured. 4. Log LAST. 5. No self-evaluation.
