# Phase 2.12 Sub-Task — Logger ASCII Hardening (multi_agent_orchestrator.py) v5

**Date:** 2026-04-14
**Session:** Ford (Remote Agent, Opus 4.6)
**Scope:** Fix residual Unicode in `logger.*()` calls per `.claude/rules/security.md`

## Context

The previous session (2026-04-14-0030) landed Step 2.13 and Phase 3.0 MCP data_server work on origin/main (commit `c1a4302`). Static inspection on this attached `main` revealed that `backend/agents/multi_agent_orchestrator.py` still had **21 logger-ASCII violations** -- prior Ford sessions (v1..v4) attempted this same fix on detached-HEAD worktrees and the fixes were lost when origin was force-pushed past them. This session re-applies the fix cleanly on attached `main` so it will survive future pushes.

## Changes

Single file modified: `backend/agents/multi_agent_orchestrator.py` (21 logger sites, 20 Edit calls).

Unicode -> ASCII mapping applied:
- `[clipboard]` -> `[Plan]` / `[QualityGate]`
- `[retry]` -> `[Research]` / `[QualityGate]`
- `[branch]` -> `[Delegate]` / `[Classify]`
- `[check]` -> `[OK]` / `[QualityGate] PASS`
- `[warn-sign]` -> `[warn]`
- `[timer]` -> `[Research] Max iterations`
- `[books]` -> `[Citation]`
- `[wrench]` -> `[ToolLoop]`
- `[compress]` -> `[Mask]`
- em-dash -> `--`
- right-arrow -> `->`

Affected lines: 307, 444, 506, 509, 511, 549, 603, 738, 751, 754, 757, 762, 772, 774, 778, 874, 1003, 1018, 1027, 1035, 1122.

Non-logger emoji (`emoji_map` dict at L554, dashboard event data) deliberately left untouched -- those are data, not logger input, transmitted via event bus to UI consumers that handle UTF-8 correctly.

## Verification (deterministic, static)

AST scan walking every `logger.*()` Call node's Constant-str subtrees:
```
TOTAL violations: 0
```

`py_compile` clean on both target files. `harness_memory.py` confirmed 0 violations (already pushed in prior session).

## Rationale

`.claude/rules/security.md`: "ASCII-only logger messages -- Never use Unicode characters in `logger.*()` calls. Windows cp1252 encoding in uvicorn handlers crashes on non-ASCII."

Defense-in-depth: `setup_logging()` in `main.py` clears uvicorn handlers and forces UTF-8 TextIOWrapper, but any log emission bypassing that wrapper (pytest captures, subprocess pipes, alternative handlers) would crash on Windows/some container stdout configs. Pure ASCII in logger calls removes the class of bug entirely.

## Out of scope

- `emoji_map` dict L554 -- event bus data, not logger input
- Other modules with residual violations (feature_generator, response_delivery, sla_monitor, tickets_db, queue_notification, skill_optimizer, quant_optimizer, MCP stubs, candidate_selector) -- different owner surface, high merge-conflict risk
- Full Phase 2.12 EVALUATE (token-efficiency + 4-tier memory measurements) -- requires LLM budget (blocked)

## Diff stat

- `backend/agents/multi_agent_orchestrator.py`: 22 insertions / 22 deletions
- Pure string substitution. No control flow, signatures, imports, or behavior changes.

## Next

Phase 2.12 remains `in-progress` in masterplan.json -- core harness-memory implementation is done (commit `ed8814e`), logger ASCII hardening now complete on main, but the "minimum 2x token reduction" success criterion needs LLM budget to measure. Sub-task complete; step remains open pending Peder's Phase 3 budget approval for the EVALUATE measurement runs.
