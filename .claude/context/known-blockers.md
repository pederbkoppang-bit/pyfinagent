---
name: Known Blockers (Remote Agent Environment)
description: Read this at startup to avoid repeating past issues. Last updated 2026-04-14 08:00 UTC.
---

# Known Blockers — Remote Agent Environment

## RESOLVED (no longer issues)

### 1. git push 403 — FIXED
**Was:** PAT lacked `Contents: Write` scope.
**Fixed:** New PAT with Read+Write access deployed 2026-04-14. The bootstrap command in your prompt has the correct PAT. Use `git push origin main` directly.

### 2. Disconnected git histories — FIXED
**Was:** Local main and origin/main had no common ancestor due to force-pushes.
**Fixed:** Histories reconnected. `git merge-base` confirms common ancestor.

### 3. Phase 3 budget — FIXED
**Was:** Phase 3 blocked on Peder's budget approval.
**Fixed:** Peder approved all phases including LLM API costs on 2026-04-14. Gate removed from masterplan.json. All phases are APPROVED.

### 4. Step 2.10 dependency on Phase 3 — CLARIFIED
**Was:** 2.10 (Karpathy Autoresearch) semantically needs LLM planner from Phase 3.1.
**Status:** Phase 3 is now approved, so this dependency is unblocked. Work on 2.10 after 3.1 delivers the LLM planner.

## STILL ACTIVE

### 5. Remote runner has NO .venv, NO backend deps
The remote CCR environment is bare Python — no FastAPI, pandas, GCP libs, etc.
- `run_harness.py --dry-run` will fail (needs backend imports)
- Use `python -c "import ast; ast.parse(...)"` for syntax checks (stdlib only)
- For harness-gated verification: check existing evaluator critiques in `handoff/`, don't try to run the harness
- Code-only work (frontend, config, docs, small backend edits) can be verified with AST parse

### 6. ALWAYS work on main branch
CCR creates feature branches by default (e.g., `claude/compassionate-knuth-SKDQy`). At startup, run:
```bash
git checkout main && git pull origin main
```
Then push directly to main. Do NOT create PRs or feature branches. All work goes to `origin/main`.

### 7. NEVER manually update CHANGELOG.md
The PostToolUse hook in `.claude/settings.json` automatically updates CHANGELOG.md on every `git commit`. Do NOT:
- Manually add changelog entries
- Commit "changelog drift" or "changelog backfill" fixes
- Spend time on changelog cleanup at session startup
The hook handles everything. Skip straight to masterplan work.

### 7. Researcher subagent turn limit
The researcher subagent has `maxTurns: 15`. Deep research with many web fetches can exceed this.
- If research times out, commit partial findings and note gaps in session log
- Next session can continue the research from where it left off

## Reading order for new sessions

1. Last 3 session logs in `.claude/context/sessions/`
2. This file (`.claude/context/known-blockers.md`)
3. `.claude/masterplan.json` — authoritative step state
4. `CLAUDE.md` — critical rules
5. `.claude/context/*.md` — project knowledge
