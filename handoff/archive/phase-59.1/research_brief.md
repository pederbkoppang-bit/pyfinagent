# Research Brief — phase-59.2: MCP audit + integration (Playwright full, Figma frontend workflow)

Tier: moderate (caller-stated). Status: IN PROGRESS — incremental write.
Date: 2026-06-11. Researcher session (Layer-3 MAS). 59.1 brief archived.

## Headline findings (running)

1. **Version delta exists**: `.mcp.json:84` pins `@playwright/mcp@0.0.75`
   (published 2026-05-07T23:10Z). npm `latest` = **0.0.76** (published
   2026-06-10T00:16Z, one day old). `next` = 0.0.76-alpha-2026-06-10.
   Source: npm registry query 2026-06-11. Changelog delta below.
2. **Masterplan audit_basis is STALE on one point**: `.mcp.json:91` ALREADY
   has `"alwaysLoad": false` on the playwright server. The verification
   command's python assert passes today. The REAL remaining gap is the
   CLAUDE.md discipline section (lines 64-68): it lists only the four
   pyfinagent-* servers — playwright must be added with rationale, and the
   `.mcp.json:44,55,66,77` line-ref list there is now stale too (playwright
   alwaysLoad sits at :91).
3. **CLAUDE.md:38-42 already carries a session-level Playwright UI rule**
   (goal-post-away-review, 2026-06-10). 59.2 adds the BINDING enforcement
   layer in `.claude/agents/qa.md` — natural slot: new `### 1c` under
   "Verification order", mirroring the phase-23.2.24 `### 1b` frontend-lint
   precedent (same "REQUIRED if diff touches X" shape).
4. **Smoke path verified**: backend :8000 alive (`/api/health` 200, version
   6.43.0), frontend :3000 alive (302 → /login, NextAuth wall intact).
   Smoke = `browser_navigate http://localhost:3000` + `browser_snapshot`
   (login page renders, no auth needed for liveness) satisfies the
   criterion's "live browser_navigate + snapshot against the running app".
5. **Skip-auth pattern canonical facts** (from
   `handoff/archive/misc/live_check_55.1.md` §A method disclosure +
   phase-56.1 contract.md:34): second dev server
   `LIGHTHOUSE_SKIP_AUTH=1 npx next dev --port 3100` from `frontend/`;
   bypass = `frontend/src/middleware.ts:24`; operator :3000 untouched;
   kill the :3100 server after capture and verify :3000 still 302;
   disclosure paragraph required in the live_check.
6. Claude Code local = 2.1.172; `alwaysLoad` requires v2.1.121+ — OK.

## (pending sections — external research in flight)

- Version-delta table 0.0.75 → 0.0.76 + bump-or-justify recommendation
- Playwright MCP best-practice config notes (headless/isolated/snapshot-vs-screenshot)
- Figma MCP capability audit vs Next.js cockpit
- alwaysLoad recommendation + rationale
- qa.md rule wording + placement
- frontend.md section drafts (skip-auth + figma)
- Source tables + recency scan + query log + JSON envelope
