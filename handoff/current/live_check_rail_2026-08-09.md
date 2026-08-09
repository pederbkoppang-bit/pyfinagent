# live_check — the Claude Code analysis rail, 2026-08-09

**RESULT: the rail is ALIVE.** Dead since at least 2026-08-08; fixed by
**removing** `CLAUDE_CODE_OAUTH_TOKEN`, not by supplying a better one.

## The finding

The token was not merely invalid — **its presence overrode a working keychain
credential.** A/B against the same binary in the same shell, only that variable
differing:

```
WITH env token   : is_error=True  status=401 api_ms=0     out_tokens=0
WITHOUT env token: is_error=False status=None api_ms=53566 out_tokens=4
```

Two separately-minted tokens (`15dc02093816`, then `35558d206b8c`) both returned
`401 OAuth access token is invalid` with `duration_api_ms: 0` — the CLI never
reached the API. Reproduced by hand outside the backend both times. So this was
never a paste, a plist, or a reload problem: `claude setup-token` is producing
tokens this account will not authenticate, while the interactive credential in
the keychain (`"Claude Code-credentials"`) works.

## The fix

`CLAUDE_CODE_OAUTH_TOKEN` **removed from all four plists** (`backend`,
`away-session-am`, `away-session-pm`, `away-watchdog`), each backed up as
`*.bak.pre-token-removal.*`. The CLI then falls back to the keychain credential.

## Proof, through the PRODUCTION client

Not a bare CLI call — `backend.agents.claude_code_client.ClaudeCodeClient`, the
class the autonomous loop uses:

```
RAIL CALL SUCCEEDED
response: LLMResponse(text='OK', usage_metadata=UsageMeta(
  prompt_token_count=2, candidates_token_count=4, total_token_count=6,
  cache_creation_input_tokens=57878, cache_read_input_tokens=0), ...)
```

Real tokens in and out. Contrast the failure signature all week:
`duration_api_ms: 0, input_tokens: 0, output_tokens: 0`.

## State after

```
backend  pid=24708  /api/health=200
CLAUDE_CODE_OAUTH_TOKEN in process env: 0   (want 0)
kill switch: paused=False sod_date=2026-08-09 armed=True nav=23833.94
claude CLI 2.1.226  (past the 2.1.225 OAuth fix -- not what we were waiting on)
```

## Not yet established

- **No trading cycle has run since the fix.** The rail answers a direct call;
  that is not the same as six analyses completing inside the cycle budget. The
  authorized verification cycle was already spent (`cycle-1786280622`, 0 trades,
  `$0.60`) BEFORE this fix, so proving end-to-end trading needs a fresh
  authorization.
- **Why the keychain credential works for a launchd agent** is not explained
  here, only measured. If the keychain locks (reboot, logout), the away-ops
  headless sessions may lose auth with no env-var fallback — that is a real
  regression risk introduced by this fix and it is queued, not waved away.
- `claude setup-token` remains broken for this account. Removing the variable
  routes around it; it does not repair it.

## Incident during the fix

`launchctl bootout` succeeded and `launchctl bootstrap` failed with
`Bootstrap failed: 5: Input/output error` — a race, bootout being asynchronous.
**The backend was DOWN for roughly four minutes** (unregistered, port free,
health 000) until re-bootstrapped to `pid 22466`. This is exactly the hazard
away-ops rail 9 reserves the verb for. The remedy is a `sleep 8` between the two
commands, now in the operator instructions.
