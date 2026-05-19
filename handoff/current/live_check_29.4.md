# Live check — phase-29.4 (3 OWASP LLM Top-10 v2.0 heuristics)

**Step ID:** phase-29.4
**Date:** 2026-05-19

## Pre/post evidence

```
Before: 432 lines (qa.md after phase-29.2 edits)
After:  437 lines (+5 net)

$ grep -nE 'rag-memory-poisoning|unbounded-llm-loop|agent_config.system_prompt' .claude/agents/qa.md
288:| system-prompt-leakage | ... `agent_config.system_prompt` ... | WARN |
289:| rag-memory-poisoning | ... | WARN |
290:| unbounded-llm-loop | ... | WARN |

$ grep -nE 'OWASP LLM Top-10 v2\\.0|OWASP LLM08:2025|OWASP LLM10:2025|BM25 corpus is not subject to Vec2Text' .claude/agents/qa.md
(matches in row 289 + 290 detection cues + negation bullet + source line — confirmed)
```

## Cited file:line anchors targeted by the heuristics

These are the REAL pyfinagent code locations the heuristics protect:

| Heuristic | Protects | File:line |
|---|---|---|
| `system-prompt-leakage` (enhanced) | `agent_config.system_prompt` passed to `client.messages.create()` | `backend/agents/multi_agent_orchestrator.py:985, :1076, :1187` |
| `rag-memory-poisoning` (new) | `add_memory()` API + BM25 corpus loader | `backend/agents/memory.py:23-54` (safe seeds), `:86` (add_memory), `:145` (load_from_bq_rows) |
| `unbounded-llm-loop` (new) | Bounded harness/orchestrator loops | `scripts/harness/run_harness.py:1111` (cycles), `:57-58` (MAX_*); `backend/agents/multi_agent_orchestrator.py:523` (research), `:1048` (tool turns) |

## Post-restart sanity check

Body-content edits to `qa.md` (not frontmatter) MAY activate without full session restart on Claude Code v2.1.140+. Conservatively still requires `/clear`. Recipe:

```
1. /clear or restart Claude Code
2. Spawn a fresh qa subagent with the trivial prompt "list your Dimension-1 heuristics"
3. Verify response includes: rag-memory-poisoning, unbounded-llm-loop, enhanced system-prompt-leakage
4. Verify response cites OWASP LLM Top-10 v2.0 (2025)
```

If the response shows ONLY the old heuristic list, the snapshot didn't refresh — `/clear` again.

## Honest disclosure

THIS overnight session's Q/A subagent (spawned for cycles 29.2, 29.1, 29.7, 29.5, 29.4) ran on the snapshotted qa.md from session start (pre-29.4 content). The new heuristics will be visible to future Q/A spawns post-restart. The cycle's PASS verdict from the current Q/A is on the EVIDENCE of the new content being correctly written, not on Q/A having dispatched WITH the new content.
