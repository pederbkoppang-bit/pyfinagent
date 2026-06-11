# Contract — Step 59.2

**Step id:** 59.2 — MCP audit + integration (Playwright full, Figma frontend workflow)
**Date:** 2026-06-11
**Phase:** phase-59 (operator answers 2026-06-11: Playwright=full integration incl. binding qa rule; Figma=integrate for frontend work)
**Researcher gate:** PASSED — `handoff/current/research_brief.md` (tier=moderate, 6 external sources read in full, 17 URLs, recency scan; 9 internal files; envelope `gate_passed: true`)

## Research-gate summary

Playwright: pinned 0.0.75 (2026-05-07) vs npm latest **0.0.76** (2026-06-10) — patch-level delta (2 irrelevant video-annotation tools, `--output-max-size`, ~10 bugfixes incl. nav-waitUntil + response-closure fixes), all four pinned flags survive, **zero breaking changes → bump + smoke**. Execution pitfall: editing .mcp.json mid-session does NOT respawn the stdio server — the smoke runs on the session's already-connected server; disclose the capture-time version in the live_check. STALE AUDIT-BASIS FINDING: `.mcp.json:91` ALREADY carries `"alwaysLoad": false` on playwright (the verification command's assert passes today); the REAL gap is CLAUDE.md's discipline list (:64-68) which omits playwright — keep `false` (official doc: alwaysLoad is for every-turn tools and blocks session startup until connect, 5s cap; playwright fires ~1-3x per UI step with ~22 tool defs — mirrors pyfinagent-backtest). qa.md binding rule: new `### 1c. Live UI capture gate` after §1b (~:100), 23.2.24-precedent shape — UI-claims steps CANNOT receive PASS without a live `browser_navigate` + `browser_snapshot`/`browser_take_screenshot` referenced in the live_check (snapshot=structure/text claims, screenshot=visual claims; missing capture → CONDITIONAL at best with Missing_Assumption). frontend.md: both new sections after Auth Flow (~:71) — the :3100 `LIGHTHOUSE_SKIP_AUTH=1` workflow (middleware.ts:24 verified; 55.1 §A disclosure paragraph = canonical template) + the Figma workflow. Figma capability audit: remote server free-during-beta on all seats (usage-based pricing later → the LLM-cost approval rule applies then); `get_design_context` outputs React+Tailwind by default (direct stack fit; MUST be reconciled to the navy/slate token rules); the claude.ai connector is session-only → Figma stays ADVISORY-ONLY, never verification-load-bearing; near-term value = code-to-design (`generate_figma_design` capturing the live cockpit; no repo Figma file exists today). Smoke verified live: :3000 → 302 /login; navigate + snapshot of the login page satisfies the criterion without auth. Restart caveat: qa.md/researcher.md edits snapshot next session (same as 59.1).

## Hypothesis

A patch bump + config-doc alignment + a precisely-worded binding capture gate turns the proven 55.x/56.x Playwright verification pattern into standing harness law, and Figma enters as an advisory design tool with honest availability caveats — all verifiable by greps + a live smoke.

## Immutable success criteria (verbatim from .claude/masterplan.json, step 59.2)

1. "Playwright: the pinned @playwright/mcp version is checked against the latest release (bumped + smoke-tested if newer, or the current pin justified); .mcp.json gains an explicit alwaysLoad key for the playwright server with a rationale consistent with the CLAUDE.md discipline section (which is updated to list it); the smoke evidence (a live browser_navigate + snapshot against the running app) is in live_check_59.2.md"

2. "qa.md gains a BINDING rule: any step whose contract or criteria make UI claims must carry a live Playwright capture in its live_check or the verdict cannot be PASS; researcher.md gains Playwright awareness for UI-related research; the :3100 LIGHTHOUSE_SKIP_AUTH verification pattern (operator :3000 untouched, start/stop, disclosure requirements) is documented in .claude/rules/frontend.md or a dedicated rules file"

3. "Figma: .claude/rules/frontend.md gains a Figma-MCP workflow section (when to use design-to-code for new dashboard views, code-to-design for design review, and the headless-absence caveat for the claude.ai connector); researcher.md/qa.md gain one-line awareness; a capability audit of the connector against the Next.js cockpit (what it can/cannot do for this repo) is recorded in live_check_59.2.md"

4. "the verification command exits 0; all agent-file edits carry the restart caveat; no emojis introduced"

**Verification command (immutable):** `source .venv/bin/activate && python -c "import json; cfg=json.load(open('.mcp.json')); assert 'alwaysLoad' in cfg['mcpServers']['playwright'], 'playwright alwaysLoad missing'" && grep -q 'Playwright' .claude/agents/qa.md && grep -qi 'figma' .claude/rules/frontend.md && test -f handoff/current/live_check_59.2.md
## Plan

1. `.mcp.json`: bump `@playwright/mcp@0.0.75` → `0.0.76`; keep `alwaysLoad: false` (already present — note the stale audit basis honestly); no flag changes (all four survive the delta).
2. CLAUDE.md discipline section: add the playwright line (`alwaysLoad: false` + rationale) and extend the `.mcp.json` line-ref sentence; note figma is a claude.ai session connector NOT in .mcp.json.
3. qa.md: insert `### 1c. Live UI capture gate` after §1b with the brief's binding wording (PASS impossible without a live capture on UI-claims steps; snapshot vs screenshot admissibility; CONDITIONAL + Missing_Assumption on absence) + restart-caveat note.
4. researcher.md: Playwright + Figma awareness note in the internal-audit instructions (after ~:96).
5. frontend.md: two new sections after Auth Flow — "Live-UI verification (Playwright MCP + skip-auth :3100)" (start/stop commands, operator :3000 untouched, disclosure template per 55.1 §A, mid-session version caveat) + "Figma MCP workflow" (design-to-code for NEW views with navy/slate reconciliation, code-to-design review, free-during-beta → cost-approval-later note, headless-absence caveat → advisory-only).
6. Smoke: browser_navigate localhost:3000 + snapshot (login page; capture-time server version disclosed as 0.0.75-running/0.0.76-pinned if no reconnect).
7. live_check_59.2.md (version-delta table, smoke evidence, Figma capability audit, config diffs) + experiment_results.md → fresh Q/A → log → flip.

## Constraints

- Keep alwaysLoad false (doc-grounded); no emojis; agent-file edits carry the restart caveat; Figma never verification-load-bearing (session-only connector); the smoke must not touch the operator's :3000 state (read-only navigate).

## References

- handoff/current/research_brief.md (researcher 59.2, gate_passed: true; github.com/microsoft/playwright-mcp releases + README, code.claude.com MCP docs, developers.figma.com MCP docs x3, help.figma.com guide)
- Operator answers 2026-06-11 (Playwright=full, Figma=integrate)
- Code anchors: .mcp.json:79-92; CLAUDE.md:64-68 (discipline list); qa.md ~:100 (§1b); researcher.md ~:96; frontend.md ~:71 (after Auth Flow); middleware.ts:24
