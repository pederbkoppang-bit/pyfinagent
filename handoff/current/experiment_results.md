# Step 2.12 (partial): Logger ASCII Hardening — Experiment Results

## Context

Per `security.md` rule: `logger.*()` calls must use ASCII only (Windows cp1252
crash in uvicorn handlers). Prior Ford sessions fixed subsets but many commits
were lost to force-pushes. Full-repo scan on `origin/main @ c1a4302` showed
**216 non-ASCII chars in logger calls across 36 files**.

This session targets the **12 harness-critical files** — the hot path that
actually runs during `run_harness.py` execution. Ticketing / Slack-bot files
are deferred to a separate pass.

## Files Fixed (12)

| File | Sites |
|------|------:|
| `backend/agents/planner_agent.py` | 4 |
| `backend/agents/planner_enhanced.py` | 7 |
| `backend/agents/evaluator_agent.py` | 11 |
| `backend/agents/evidence_engine.py` | 2 |
| `backend/agents/multi_agent_orchestrator.py` | 23 |
| `backend/agents/mcp_servers/backtest_server.py` | 1 |
| `backend/agents/mcp_servers/signals_server.py` | 1 |
| `backend/agents/skill_optimizer.py` | 1 |
| `backend/autonomous_harness.py` | 4 |
| `backend/backtest/quant_optimizer.py` | 1 |
| `backend/backtest/spot_checks.py` | 2 |
| `backend/backtest/candidate_selector.py` | 1 |
| **Total** | **58** |

All changes are pure string substitution. No control flow, signatures,
imports, or non-logger code touched.

## Method

AST-based fixer:
1. `ast.walk` collects every `Call(func=Attribute(attr in LOG_METHODS,
   value.id in LOG_NAMES))` node, recording `(lineno, end_lineno)`.
2. For each file, a boolean mask marks lines inside any logger-call range.
3. Char-substitution map is applied **only to masked lines**, leaving all
   other code bytes untouched.
4. Post-edit `ast.parse` confirms syntax validity.

Substitution map:
```
→ -> | ← <- | — -- | – - | × x | ✓ [ok] | ✗ [x]
⚡ [spark]    ⏱ [time]      🔧 [tool]     🗜 [mask]
📋 [Plan]     🔍 [Research]  🔀 [Classify] 📚 [Citation]
⚠️ [warn]     ✅ [OK]        ❌ [FAIL]     🚨 [ALERT]
🎯 [target]   📊 [stats]     🧠 [think]    🚀 [go]
🎉 [done]     ⚙️ [cfg]       💡 [idea]     🤖 [bot]
📍 [pin]      🔄 [cycle]     🔗 [link]     📤 [send]
📥 [recv]     🛑 [stop]      🔴 [red]      🎫 [ticket]
🔪 [kill]     🧹 [clean]     📱 [phone]    👍 [+1] 👎 [-1]
🔓 [unlock]   🏠 [home]      ➡️ [next]     ⬆️ [up]
💬 [chat]     📖 [read]      📞 [call]
```

## Verification

```
AST re-scan of 12 target files -> 0 non-ASCII chars in logger calls
py_compile on 12 target files -> all clean
git diff --stat:
  13 files changed, 58 insertions(+), 57 deletions(-)
```

The 13th file is `CHANGELOG.md` (1 line auto-added by PostToolUse hook for
commit `c1a4302`, present in the local worktree after reset).

## Out of Scope

- 24 non-harness files with ~158 remaining violations (Slack bot, ticket
  queue, response delivery, SLA monitor, tickets_db, feature_generator,
  openclaw client/monitor, autonomous_loop). Fix in a separate
  ticket-queue maintenance pass.
- Non-logger emoji (e.g. `emoji_map` dicts, greeting strings, exception
  list-appends) — data, not logger input, intentionally left untouched.

## Phase 2.12 Progress

This closes the long-running "logger ASCII hardening" thread of Phase 2.12
for the harness hot path. EVALUATE (token-efficiency & 4-tier memory
measurements) still requires LLM API budget approval from Peder.
