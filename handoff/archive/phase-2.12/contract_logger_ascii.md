# Phase 2.12 — Logger ASCII Hardening (on `main`, not detached HEAD)

## Context
`security.md` (lines 37-38) forbids non-ASCII characters in `logger.*()` call strings —
Windows cp1252 encoding in uvicorn handlers crashes on non-ASCII. `setup_logging()` in
`main.py` mitigates this by clearing handlers and forcing UTF-8 `TextIOWrapper`, but the
rule mandates ASCII for defense-in-depth.

## History
Multiple prior Ford sessions (commits `b49cb69`, `a6b1700`, `106a5d4`, `7182f43`, `8a86ed2`,
`2ac17db`, `94b7ca1`) applied this exact fix on detached HEADs that were later force-pushed
away from `origin/main`. This session is the first to operate cleanly on `main` after the
`b472502 -> 2b94be0` force update; the fix is being re-landed where it will actually persist
upstream.

## Scope (in)
Replace 24 non-ASCII characters in `logger.*()` call f-strings across two files. Pure
string substitution. No control-flow, signature, or import changes.

| File | Sites | Chars |
|------|-------|-------|
| `backend/agents/multi_agent_orchestrator.py` | 23 | emoji + arrow |
| `backend/agents/harness_memory.py` | 1 | arrow `->` |

### Substitution map
- `📋` -> `[Plan]`
- `🔄` -> `[Loop]`
- `✅` -> `[OK]`
- `⏱` / `⏱️` -> `[Timeout]`
- `🔀` -> `[Route]`
- `🔧` -> `[Tool]`
- `🗜` / `🗜️` -> `[Compress]`
- `⚠` / `⚠️` -> `[Warn]`
- `📚` -> `[Cite]`
- `→` -> `->`
- `—` -> `--` (em-dash, where present in logger strings)

## Scope (out)
- Non-logger non-ASCII: `emoji_map` dicts, exception list-appends, agent display strings.
  Those are *data*, not logger format-string input.
- Other files with logger non-ASCII (feature_generator, response_delivery, sla_monitor,
  tickets_db, queue_notification, skill_optimizer, quant_optimizer, MCP stubs, candidate_selector).
  Different owner surface; deferred to a focused sweep.

## Acceptance criteria
1. AST scan walking `logger.*()` call subtrees -> 0 non-ASCII string constants in the two
   target files.
2. `python -c 'import ast; ast.parse(open(F).read())'` clean for both files.
3. Diff is line-for-line substitution; no other lines touched.
4. QA Evaluator (anti-leniency) PASS.

## Why this lands this time
Operating directly on `main`, not detached HEAD. Will be pushed via the GitHub MCP server
(`mcp__github__create_or_update_file`) since the bootstrap PAT lacks push scope on this repo.

## Verification commands
```
python3 -c "
import ast
for f in ['backend/agents/multi_agent_orchestrator.py', 'backend/agents/harness_memory.py']:
    tree = ast.parse(open(f).read())
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('debug','info','warning','error','critical','exception','log'):
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        if any(ord(c) > 127 for c in sub.value):
                            n += 1
                            break
    print(f, n)
"
python -c "import ast; ast.parse(open('backend/agents/multi_agent_orchestrator.py').read())"
python -c "import ast; ast.parse(open('backend/agents/harness_memory.py').read())"
```
