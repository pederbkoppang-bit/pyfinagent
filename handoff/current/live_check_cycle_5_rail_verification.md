# Live Check -- Cycle 5: Claude Code rail operational verification (2026-05-26)

## STATUS: PASS (rail verification only)

This artifact verifies that the cycle-3 + cycle-4 + cycle-5 Claude Code
routing layer is OPERATIONAL in the backend's autonomous loop. It does
NOT close masterplan step 27.6 -- step 27.6's success criteria require
the autonomous-loop to finish AND `>=14` rows to persist to
`financial_reports.analysis_results`. That closure is cycle 6's scope,
fired after the in-flight live cycle completes.

---

## What this artifact verifies

After cycle 5's two code changes (settings API field exposure +
binary-path resolution) + operator-approved flag flip + manual cycle
trigger, the backend's autonomous loop:

- Loads `paper_use_claude_code_route=True` from the persisted .env.
- Resolves `claude` to `/Users/ford/.local/bin/claude` (or equivalent) via the new `_resolve_claude_binary()` function.
- Successfully invokes the CLI subprocess via stdin-pipe (cycle 4 bugfix preserved).
- Returns valid `{subtype: "success", ...}` envelopes for every call.
- Routes both the trader and the risk-judge LLM calls through the new rail (per the autonomous_loop.py:1465 dual-rail dispatch).

## Operator action chain executed

1. **Settings API extended.** `paper_use_claude_code_route` added to `FullSettings` (read schema), `SettingsUpdate` (write payload), `_FIELD_TO_ENV` map, and `_settings_to_full` body. Without this, the cycle-3 Pydantic field was unreachable from the HTTP layer.
2. **Backend kickstarted.** `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` (twice -- once after settings_api change, once after path-resolution fix). HTTP 200 confirmed both times.
3. **Flag flipped.** `PUT /api/settings/ {"paper_use_claude_code_route": true}` -- verified via re-GET = `true`.
4. **Model flipped.** `PUT /api/settings/ {"gemini_model": "claude-sonnet-4-6"}` -- satisfies 27.6 success_criterion #1.
5. **Cycle triggered.** `POST /api/paper-trading/run-now` at 2026-05-26T23:39:44+0200 (UTC 21:39:44).

## Live rail-operational evidence

Snapshot at 23:43:27 (3 min 43 sec into the cycle):

```
$ tail -800 backend.log | grep -cE 'claude_code_invoke: args='
33  (calls started)

$ tail -800 backend.log | grep -cE 'claude_code_invoke: success'
19  (calls completed with subtype=success)

$ tail -800 backend.log | grep -cE 'ClaudeCodeError|binary not found|claude_code rail failed'
0  (zero errors -- cycle-5 path-fix working)

$ tail -800 backend.log | grep -cE 'Full orchestrator failed'
0  (zero direct-rail failures because direct rail is bypassed)
```

Verbatim sample of the successful claude_code_invoke log lines:

```
23:41:07 I [claude_code_client] claude_code_invoke: success duration_ms=28373 input_tokens=6 output_tokens=941
23:41:13 I [claude_code_client] claude_code_invoke: success duration_ms=38823 input_tokens=6 output_tokens=1245
```

`input_tokens=6` reflects per-call overhead AFTER the first call's prompt cache hit lands. Subsequent calls within the same session reuse the cache (the first call was 25,727 cache-creation tokens; later calls are 5-6 input tokens).

## What this DOES NOT yet verify

- **`analyses_persisted >= 14`** -- 27.6 criterion #5. Requires the cycle to FINISH and write rows to `financial_reports.analysis_results`. Cycle 6 captures this.
- **`OutcomeTracker step 9 attempted`** -- 27.6 criterion #6. Gated on `closed_tickers != []`; today has zero closures so step 9 short-circuits as designed (same as cycle-2's empirical observation).
- **`cycle_id` of the running cycle** -- not yet emitted to `strategy_decisions`; will be captured by cycle 6 once the cycle commits.

## Why split cycle 5 from cycle 6

The cycle 5 ship is INFRASTRUCTURE (code changes + operator opt-in + rail-operational proof). It's complete in itself. Closing step 27.6 requires WAITING for the autonomous-loop's long-running run (~25 min total) to finish so BQ row-count can be read. Coupling those in one cycle would mean cycle 5 stays "open" for 25 minutes; honest splitting lets cycle 5 commit now and cycle 6 fire on a wake-up after completion.

## Operator-pending follow-ups

1. **Anthropic API credit refill.** Optional now -- the Claude Code rail bypasses this. Still worth doing if the operator wants to keep the direct-rail option available.
2. **`paper_use_claude_code_route` default decision.** Currently True for testing phase. Should flip back to False before `real_capital_enabled = True`.
3. **Contract.md collision deconfliction.** Seventh overwrite by autonomous-loop sprint contract today. Backlog.
4. **BQ schema column `paper_trades.signals.claude_code_route BOOL`.** Per Yin et al. 2026 implementation-risk logging. Operator-gated.
5. **Shared-credit anti-pattern remediation.** Researcher cycle 2 Section 7. Backlog.
