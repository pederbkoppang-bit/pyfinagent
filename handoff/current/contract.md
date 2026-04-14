# Step 2.12 (partial): Logger ASCII Hardening — Harness-Critical Files

## Hypothesis

Non-ASCII characters (emoji, arrows, em-dashes) inside `logger.*()` calls cause
uvicorn handler crashes on Windows (cp1252 encoding) and degrade log readability
on every platform. The `security.md` project rule forbids them. A 36-file scan
found 216 violations. Fixing the 12 harness-critical files (planner, evaluator,
orchestrator, MCP servers, backtest engine, optimizer) eliminates every
logger-emoji path that can fire during `run_harness.py` execution.

## Success Criteria (Research-Backed)

1. **Zero non-ASCII in logger calls** — AST scan walks `Call` nodes whose
   `func` is `Attribute(value=Name("logger|log|LOGGER|_logger"), attr in
   {debug,info,warn,warning,error,critical,exception})` and checks every
   `Constant(str)` descendant. Target: **0 violations across 12 files**.
2. **All 12 files compile clean** — `py_compile.compile(path, doraise=True)`
   succeeds on every target.
3. **Edits confined to logger lines** — no control flow, signature, import, or
   non-logger string touched. Diff stat insertions ≈ deletions.
4. **No regressions in unmodified files** — the 24 deferred files still parse.

## Fail Conditions

- Any target file fails `ast.parse` or `py_compile` after edits.
- Any residual non-ASCII char inside a logger call node in target files.
- Edits leak outside logger call line ranges (e.g. touching `emoji_map` dicts,
  greeting strings, or non-logger exception list-appends — these are data, not
  logger input, and must remain unchanged per prior Ford PASS pattern).
- Any target file's public API (function signatures, exports) changes.

## Research Backing

Pattern + scope is a direct continuation of 4 prior Ford sessions that reached
PASS verdict on the same substitution pattern but whose commits were lost to
force-pushes (`b49cb69`, `a6b1700`, `106a5d4`, `7182f43`, `8a86ed2`, `2ac17db`,
`94b7ca1`, `2cb2f7f`). All prior QA verdicts: PASS 10/10/10/10/9 on
correctness, scope, security rule compliance, simplicity, conventions.
No new research gate needed for a re-apply of a proven pattern on an expanded
file set. The 3 files (`planner_agent`, `planner_enhanced`, `evidence_engine`)
added beyond the prior scope follow the identical rule and replacement map.

## Out of Scope (Deferred)

- `backend/slack_bot/*.py` (ticketing + Slack bot wiring)
- `backend/services/*.py` (ticket queue, response delivery, SLA monitor)
- `backend/db/tickets_db.py`
- `backend/api/mas_events.py`
- `backend/agents/feature_generator.py`, `openclaw_client.py`, `openclaw_monitor.py`
- `backend/autonomous_loop.py` (separate from `autonomous_harness.py`)

Rationale: these are outside the harness hot path and carry non-trivial merge
conflict risk with other active work.
