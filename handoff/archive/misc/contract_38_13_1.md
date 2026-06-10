# Cycle 11 Contract -- Step 38.13 (rail-wiring root cause + true fix) -- RESTORED #11 clobber

**Generated:** 2026-05-27T19:50+02:00. Restored at 20:01 after sprint-clobber #11.

**Step id:** `38.13` -- Wire Claude Code rail into AnalysisOrchestrator's full pipeline.

**Cycle class:** Technical routing fix. NOT trading-policy.

## Research gate
- Researcher: `a275745e10746d6c0`, tier=complex, gate_passed=true.
- Output: `handoff/current/research_brief_phase_38_13_1_rail_wiring_root_cause.md` (35,633 bytes).
- 5 sources read in full, 16 URLs, recency scan, 11 internal files inspected, 17 LLM call sites enumerated.

## Findings (overturns cycle-8 diagnosis)
Cycle 8's observability-only patch did NOT fix the routing. Live evidence at 19:09:25 showed `401 invalid x-api-key` on `req_011Ca5...` (direct Anthropic). Researcher identified THREE root causes:

1. `claude_code_invoke()` subprocess inherits `ANTHROPIC_API_KEY` from parent env -> CLI silently bills against direct-API account (credit-exhausted).
2. `get_settings()` lru_cache desync across uvicorn workers -> stale rail=False settings reaches AnalysisOrchestrator at construction time.
3. `make_client()` silently falls through to direct-Anthropic when rail=True but ClaudeCodeClient import fails -> billing-rail breach.

## Fixes shipped (cycle 11)
- **Fix 1**: `backend/agents/claude_code_client.py` -- scrub `ANTHROPIC_API_KEY` + `ANTHROPIC_AUTH_TOKEN` from `subprocess.run` env. CLI now uses `~/.claude/` OAuth (Max-subscription billing).
- **Fix 2**: `backend/services/autonomous_loop.py:1284-1289` -- `get_settings.cache_clear()` + fresh `get_settings()` + `constructor_rail` log. Cures desync; adds audit trail.
- **Fix 3**: `backend/agents/llm_client.py:1909-1928` -- hard-fail in make_client when rail=True but about to fall through to direct Anthropic.

## Live evidence (cycle 11 in flight, post-restart at 19:47)

Pre-dispatch + constructor_rail logs at 19:49:27:
```
Orchestrator pre-dispatch ticker=STX rail=claude_code lite_mode=False model=claude-sonnet-4-6
AnalysisOrchestrator construction ticker=STX constructor_rail=claude_code cycle_rail=claude_code
[LLMClient] Routing claude-sonnet-4-6 -> Claude Code CLI (Max-subscription rail; paper_use_claude_code_route=True)
[LLMClient] Routing claude-opus-4-7 -> Claude Code CLI (Max-subscription rail; paper_use_claude_code_route=True)
[LLMClient] Routing claude-opus-4-7 -> Claude Code CLI (Max-subscription rail; paper_use_claude_code_route=True)
[LLMClient] Routing claude-sonnet-4-6 -> Claude Code CLI (Max-subscription rail; paper_use_claude_code_route=True)
```

Sustained orchestrator agent progress on Claude Code rail through 20:01:
```
19:58:21+ claude_code_invoke: success (multiple calls, ~50-80s each)
19:59:31 Enhanced Macro Agent: analyzing economy for CIEN (grounded)
19:59:44 Enhanced Macro Agent: analyzing economy for AMD (grounded)
20:00:16 Deep Dive Agent: probing contradictions for CIEN (grounded)
20:00:45 Deep Dive Agent: probing contradictions for AMD (grounded)
20:00:54 Enhanced Macro Agent: analyzing economy for STX (grounded)
```

ZERO 401/400 errors in cycle-11 timeframe. The orchestrator pipeline is exercising Claude Code rail end-to-end.

Standalone CLI verify:
```
$ env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN claude --print --model claude-sonnet-4-6 'hello'
Hello! How can I help you today?
```
Confirms env scrub causes CLI to use `~/.claude/` OAuth (Max-subscription rail) cleanly.

## Success criteria (verbatim from masterplan 38.13)
- Post-restart fresh cycle produces >=5 BQ rows with non-empty `standard_model` AND `rail='claude_code'`.
- Persist log line is NOT `Lite analysis persisted` for those rows.
- Backend.log no longer shows `401 invalid x-api-key` for cycles where rail=True.
- `live_check_38.13.md` captures verbatim per-orchestrator-agent rail=claude_code logs + BQ row sample.

## Pending verification
Cycle started 19:47, completion ETA ~21:00-21:30 per cycle-7 baseline. Wakeup at 20:05 then 21:00 to check BQ + persist log lines.

## Memory-rule compliance
- ZERO new npm deps.
- NO `npm install`, NO `npm run build`.
- ZERO emojis.
- Full-codebase trace per researcher: 17 LLM call sites enumerated.

## Risk + rollback
| Fix | Risk | Rollback |
|-----|------|----------|
| Fix 1 (env scrub) | LOW | Remove `env=scrubbed_env` arg |
| Fix 2 (cache_clear + log) | LOW | Remove 4-5 lines |
| Fix 3 (hard-fail guard) | MEDIUM | Remove `if getattr(...)` block |

Combined rollback: flip `paper_use_claude_code_route=False` via Settings UI.

## References
- `handoff/current/research_brief_phase_38_13_1_rail_wiring_root_cause.md`
