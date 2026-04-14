# Phase 2.12 — Logger ASCII Hardening (Orchestrator) — Sprint Contract

## Hypothesis
`backend/agents/multi_agent_orchestrator.py` contains 21 `logger.*()` calls with
non-ASCII characters (emoji + Unicode arrows/em-dashes). Per `.claude/rules/security.md`:
"ASCII-only logger messages: Never use Unicode characters in logger.*() calls. Windows
cp1252 encoding in uvicorn handlers crashes on non-ASCII." These calls are a latent
crash risk every time one fires under a cp1252-bound handler.

Prior Ford sessions fixed this four times (`b49cb69`, `a6b1700`, `106a5d4`, `7182f43`,
`8a86ed2`, `2ac17db`, `94b7ca1`) but each fix landed on a detached HEAD that then got
reset, or on a local branch that could not be pushed (PAT 403). `origin/main` was
force-pushed to `c1a4302` in this session, and that snapshot still has all 21
violations present — the fix has never actually reached origin.

## Success Criteria (Research-Backed)
1. AST scan of `logger.*()` call sites walks every f-string / str subtree; returns
   **0** non-ASCII code points for both `harness_memory.py` and
   `multi_agent_orchestrator.py`. (security.md rule)
2. `python3 -c "import ast; ast.parse(open('backend/agents/multi_agent_orchestrator.py').read())"`
   returns exit 0 (syntax clean).
3. Diff is pure string substitution inside `logger.*()` call sites. No control flow,
   no signatures, no imports touched. (scope discipline — backend-agents.md)
4. ASCII replacements use bracketed tags already established by prior Ford sessions:
   `[Plan]`, `[Research]`, `[Classify]`, `[Delegate]`, `[QualityGate]`, `[ToolLoop]`,
   `[Citation]`, `[Mask]`, `[warn]`. Unicode arrows `\u2192` -> `->`; em-dash
   `\u2014` -> `--`. (consistency across the file's log style.)
5. Non-logger emoji in data structures (`emoji_map` dict at L554, exception
   list-append at L475) deliberately left untouched — data, not logger input.

## Fail Conditions
- Any remaining non-ASCII char inside a `logger.*()` call argument.
- Any modification outside the 21 target call sites.
- Any syntax error post-edit.
- Any Unicode arrow or em-dash added anywhere new.
