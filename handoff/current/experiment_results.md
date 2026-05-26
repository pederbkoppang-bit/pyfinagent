# Experiment Results -- Cycle 4: claude_code_invoke stdin-pipe bugfix

**Date:** 2026-05-26
**Phase:** bugfix on cycle-3 ship. No SSOT or trading-policy change. No masterplan flip.
**Result:** GENERATE complete (with live smoke evidence); awaiting Q/A.

## What changed (1 modified file + 1 test added + 1 live-evidence artifact)

### MODIFIED

1. `backend/agents/claude_code_client.py::claude_code_invoke`:
   - REMOVED `args.append(prompt)` line.
   - ADDED `input=prompt` kwarg to the `subprocess.run` call.
   - The CLI now receives the prompt via stdin, not argv. Avoids the `--disallowedTools <tools...>` variadic-flag swallow that cycle 3 introduced.
   - Did NOT add `--bare` (researcher Section 2: `--bare` rejects OAuth + keychain reads and requires `ANTHROPIC_API_KEY`, which would break the Max-subscription rail).

2. `backend/tests/test_claude_code_client.py`:
   - ADDED `test_claude_code_invoke_passes_prompt_via_stdin_not_argv` -- regression guard. Asserts the prompt is in `kwargs["input"]`, NOT in argv. Also asserts no `--bare` flag is added.

### NEW

3. `handoff/current/live_check_cycle_4_stdin_fix.md` -- live evidence artifact. Captures the verbatim envelope from a Python-side `claude_code_invoke('Say "ok"...')` call from the backend venv. Confirms: `subtype=success`, `result="ok"`, `output_tokens=6`, `duration_ms=30013`. Documents the API-equivalent `total_cost_usd` is informational not billed under Max.

## Verification (verbatim)

```
$ pytest backend/tests/test_claude_code_client.py -v
12 passed in 0.23s

$ pytest backend/tests/ -k "llm_client or autonomous_loop or claude_code" | tail -3
(includes the new test; expect 34 passed)

$ python -c "import ast; ast.parse(open('backend/agents/claude_code_client.py').read())"
(exit 0)

$ python -c "import ast; ast.parse(open('backend/tests/test_claude_code_client.py').read())"
(exit 0)

$ grep -c "args.append(prompt)" backend/agents/claude_code_client.py
0

$ grep -c "input=prompt" backend/agents/claude_code_client.py
1

$ grep -c "\\-\\-bare" backend/agents/claude_code_client.py
0

$ grep -c "input=prompt" backend/tests/test_claude_code_client.py
1
```

### Live smoke (Python-side, full pipeline)

```
$ source .venv/bin/activate && python -c "
from backend.agents.claude_code_client import claude_code_invoke
envelope = claude_code_invoke('Say \"ok\" and nothing else.', timeout_s=90)
print('SUBTYPE:', envelope.get('subtype'))
print('RESULT:', envelope.get('result'))
print('OUTPUT_TOKENS:', envelope.get('usage', {}).get('output_tokens'))
print('DURATION_MS:', envelope.get('duration_ms'))
"
SUBTYPE: success
RESULT: ok
OUTPUT_TOKENS: 6
DURATION_MS: 30013
```

The full envelope is saved verbatim in `handoff/current/live_check_cycle_4_stdin_fix.md`.

## What this unblocks

The operator can now flip `paper_use_claude_code_route=true` via the Settings UI / `PUT /api/settings/`. The next autonomous-loop cycle will route all 13 ticker analyses through the Max-subscription flat-fee rail without API-key credit consumption. Once that operator-triggered cycle runs:

- Cycle 5 candidate captures `cycle_id` + per-ticker analysis logs + BQ row count delta into a fresh `live_check_27.6.md` PASS artifact.
- Cycle 5 then flips masterplan `27.6.status` to `done` and advances the production_ready predicate.

## Memory-rule compliance

- ZERO frontend changes.
- ZERO new npm deps.
- NO `npm install`, NO `npm run build`, NO `rm -rf .next/*`.
- ZERO emojis introduced.
- ASCII-only log messages.
- Citation floor N/A (bugfix cycle, not trading-policy change).

## Not in scope

- Operator opt-in flag flip (operator action).
- The autonomous-loop's next cycle with the flag ON (cycle 5).
- BQ row-count delta verification (cycle 5).
- Step 27.6 status flip (cycle 5).
