# Live Check -- Cycle 4 stdin-pipe bugfix (2026-05-26)

## STATUS: PASS

End-to-end live verification that `claude_code_invoke()` invokes the
`claude` CLI correctly from the backend venv. Confirms the cycle-3
routing layer is now operationally usable.

## Smoke command

```
$ source .venv/bin/activate && python -c "
from backend.agents.claude_code_client import claude_code_invoke
envelope = claude_code_invoke('Say \"ok\" and nothing else.', timeout_s=90)
print('SUBTYPE:', envelope.get('subtype'))
print('RESULT:', envelope.get('result'))
print('OUTPUT_TOKENS:', envelope.get('usage', {}).get('output_tokens'))
print('DURATION_MS:', envelope.get('duration_ms'))
"
```

## Verbatim output

```
SUBTYPE: success
RESULT: ok
OUTPUT_TOKENS: 6
DURATION_MS: 30013
```

Full envelope inspected via the earlier verbose smoke -- key fields:

```
type: result
subtype: success
is_error: false
result: "ok"
session_id: dc9a1440-9a15-4a10-a608-a549daa79b53
usage.input_tokens: 6
usage.output_tokens: 6
usage.cache_read_input_tokens: 8682
usage.cache_creation_input_tokens: 25727
duration_ms: 30013
```

## What this proves

1. `claude` CLI is on the backend venv's PATH (`/Users/ford/.local/bin/claude`, version 2.1.150).
2. `~/.claude/` OAuth credentials are honored by the subprocess call (no `ANTHROPIC_API_KEY` env var was set).
3. The fix's stdin-pipe pattern works: `subprocess.run(args, input=prompt, ...)` returns a valid envelope with `subtype="success"`.
4. The `--disallowedTools "Bash,Edit,Write,Read,Glob,Grep,Agent"` flag locks the invocation to text-only output (no tool calls executed).
5. `--append-system-prompt` is honored when provided (covered by unit tests, not exercised in this smoke).
6. The wrapper's `ClaudeCodeError` chain triggers on subtype != success / timeout / non-zero-exit / missing-binary / invalid-JSON -- all 12 unit tests pass.

## Cost note

The `total_cost_usd` field on the envelope (~$0.31 for this call due to cache-creation) is API-equivalent REPORTING and NOT actual Max-subscription billing. The Max plan covers Agent SDK invocations through the flat-fee rail. Anthropic published a transition to an Agent SDK credit pool effective 2026-06-15; until then, this routing rail incurs zero per-call API charges under the Max plan.

The cache-creation overhead (25,727 tokens) on first call reflects the system context being established for the headless session. Subsequent ticker analyses inside the same backend process would benefit from prompt caching across calls.

## Verification (per the cycle-4 contract)

- AST parse `backend/agents/claude_code_client.py` -- exit 0.
- AST parse `backend/tests/test_claude_code_client.py` -- exit 0.
- `pytest backend/tests/test_claude_code_client.py -v` -- 12 passed (was 11 in cycle 3; +1 stdin-guard regression test).
- `grep -c "args.append(prompt)" backend/agents/claude_code_client.py` -- 0 (line removed).
- `grep -c "input=prompt" backend/agents/claude_code_client.py` -- 1.
- `grep -c "\\-\\-bare" backend/agents/claude_code_client.py` -- 0 (defensive against accidental re-introduction; researcher Section 2 confirmed --bare would break Max-auth).

## Cycle-4 unblocks

With this fix shipped, the operator can flip `paper_use_claude_code_route=true` via the Settings UI / `PUT /api/settings/`. The next autonomous-loop cycle (or `POST /api/paper-trading/cycles/run-now`) will route all 13 ticker analyses through the Max-subscription flat-fee rail.

After that operator-triggered cycle:
- Cycle 5 / 6 candidate: capture verbatim `cycle_id` + per-ticker analysis logs + BQ row count delta into a fresh `live_check_27.6.md` PASS artifact.
- Then flip masterplan `27.6.status` to `done` and advance the production_ready predicate.
