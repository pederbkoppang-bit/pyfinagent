# Experiment Results -- Cycle 5: settings exposure + binary-path fix + rail verification

**Date:** 2026-05-26
**Phase:** infrastructure (cycle 3 + 4 follow-through). No SSOT or trading-policy change. No masterplan flip in cycle 5; cycle 6 closes 27.6 after the in-flight live cycle finishes.
**Result:** GENERATE complete; awaiting Q/A.

## What changed (2 modified backend files; ZERO new tests beyond cycle 4's 12)

### MODIFIED

1. `backend/api/settings_api.py` -- exposed `paper_use_claude_code_route` to the HTTP layer in 4 places:
   - `FullSettings.paper_use_claude_code_route: bool = False` (response model, line ~115).
   - `SettingsUpdate.paper_use_claude_code_route: Optional[bool] = None` (write payload, line ~160).
   - `_FIELD_TO_ENV["paper_use_claude_code_route"] = "PAPER_USE_CLAUDE_CODE_ROUTE"` (env-var persistence map, line ~282).
   - `_settings_to_full` body now copies the field into the response (line ~352).

   Without this change, the cycle-3 Pydantic field was UNREACHABLE from the HTTP layer -- PUT returned 200 but silently dropped the field; GET never exposed it.

2. `backend/agents/claude_code_client.py` -- added `_resolve_claude_binary()` resolver and updated `claude_code_invoke` to use the resolved binary path. The launchd-supervised backend doesn't inherit the interactive-shell PATH (`claude` lives at `~/.local/bin/claude`, not on `/usr/bin` or `/usr/local/bin`), so a bare `subprocess.run(["claude", ...])` failed with FileNotFoundError. Fix:
   - `_DEFAULT_SEARCH_PATHS` env-overridable list: `$CLAUDE_CODE_BINARY` > `~/.local/bin/claude` > `/opt/homebrew/bin/claude` > `/usr/local/bin/claude`.
   - `_resolve_claude_binary(binary)` returns the first matching absolute path.
   - Test mocks of `subprocess.run` still see `"claude"` (literal) because `shutil.which("claude")` resolves to a real path during test runs.

### Operational steps (executed live)

3. `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` -- reloaded backend twice. HTTP 200 verified both times.
4. `PUT /api/settings/ {"paper_use_claude_code_route": true}` -- verified via re-GET = `true`.
5. `PUT /api/settings/ {"gemini_model": "claude-sonnet-4-6"}` -- satisfies 27.6 criterion #1.
6. `POST /api/paper-trading/run-now` at 2026-05-26T23:39:44+0200.

## Live evidence captured in `handoff/current/live_check_cycle_5_rail_verification.md`

Snapshot at 23:43:27 (3 min 43 sec into cycle, ~38% through expected ~50-call total):

```
calls started:        33
calls succeeded:      19
errors:                0
"Full orchestrator failed" lines: 0
```

Sample verbatim success log:
```
23:41:07 claude_code_invoke: success duration_ms=28373 input_tokens=6 output_tokens=941
23:41:13 claude_code_invoke: success duration_ms=38823 input_tokens=6 output_tokens=1245
```

## Verification

```
$ python -c "import ast; ast.parse(open('backend/api/settings_api.py').read())"
(exit 0)

$ python -c "import ast; ast.parse(open('backend/agents/claude_code_client.py').read())"
(exit 0)

$ pytest backend/tests/test_claude_code_client.py -v
12 passed in 0.21s

$ curl -s http://localhost:8000/api/settings/ | jq -r '.paper_use_claude_code_route'
true

$ curl -s http://localhost:8000/api/settings/ | jq -r '.gemini_model'
claude-sonnet-4-6

$ git diff --stat HEAD -- frontend/
(empty -- ZERO frontend changes)

$ git diff HEAD -- frontend/package.json
(empty -- ZERO new deps)
```

## Cycle 6 scope (the closure)

Cycle 6 (queued as a wakeup at 00:08) will:
1. Verify the autonomous-loop completed: `grep "cycle complete" backend.log | tail -1` shows the 23:39 cycle's completion line.
2. Query BQ: `SELECT COUNT(*) FROM financial_reports.analysis_results WHERE DATE(analysis_date) = CURRENT_DATE() AND model = 'claude-sonnet-4-6';`. Expect >= 14 to satisfy 27.6 criterion #5.
3. Capture verbatim `cycle_id` from `strategy_decisions`.
4. Rewrite `handoff/current/live_check_27.6.md` with PASS evidence (replacing the cycle-2 BLOCKED-state version).
5. Spawn Q/A for cycle 6.
6. Flip masterplan `27.6.status` from `pending` to `done`.
7. Append harness_log cycle 6 closure entry.
8. Commit + push.

## Memory-rule compliance

- ZERO frontend changes.
- ZERO new npm deps.
- NO `npm install`, NO `npm run build`, NO `rm -rf .next/*`.
- ZERO emojis introduced.
- ASCII-only log messages.

## Not in scope

- 27.6 closure (cycle 6).
- BQ row-count assertion (cycle 6).
- Contract.md collision deconfliction (backlog; SEVENTH occurrence today).
- BQ schema column for engine provenance (operator-gated).
