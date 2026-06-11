# Experiment Results — Step 59.2 (GENERATE)

**Step:** 59.2 — MCP audit + integration (Playwright full, Figma frontend workflow). **Date:** 2026-06-11.

## What was built (5 files)

| File | Change |
|---|---|
| `.mcp.json` | `@playwright/mcp` 0.0.75 → **0.0.76** (patch delta, zero breaking changes; all four pinned flags survive); `alwaysLoad: false` kept (already present — stale audit-basis premise honestly noted) |
| `CLAUDE.md` | MCP discipline list gains the playwright entry (`alwaysLoad: false` + episodic-server rationale + the mid-session no-respawn caveat) and the Figma connector note (session-only, advisory, never verification-load-bearing) |
| `.claude/agents/qa.md` | NEW §1c "Live UI capture gate" — BINDING: UI-claims steps cannot PASS without a live Playwright capture; CONDITIONAL + Missing_Assumption on absence; snapshot-vs-screenshot admissibility; 55.1 precedent cited; Figma excluded; restart caveat inline |
| `.claude/agents/researcher.md` | MCP-awareness block (live snapshot over code inference for UI questions; Figma advisory-only/absent-headless) |
| `.claude/rules/frontend.md` | TWO new sections: "Live-UI verification" (the full :3100 skip-auth workflow, disclosure template, kill-after-capture, version caveat) + "Figma MCP workflow" (code-to-design first, design-to-code with token reconciliation, beta-pricing/cost-approval note, availability check) |

## Verification command output (verbatim)

```
$ python -c "...assert 'alwaysLoad' in cfg['mcpServers']['playwright']..." 
alwaysLoad: False | version: @playwright/mcp@0.0.76
$ grep -q 'Playwright' .claude/agents/qa.md  -> ok
$ grep -qi 'figma' .claude/rules/frontend.md -> ok
$ test -f handoff/current/live_check_59.2.md -> PASS
```

## Smoke evidence

Live `browser_navigate http://localhost:3000/login` + `browser_snapshot`: full login page (PyFinAgent heading, Google + Passkey sign-in buttons, restricted-access note). One transient "Internal Server Error" on the first navigate (dev-server compile-on-demand race) — re-probed via curl (200) + re-navigated successfully; operator :3000 untouched (read-only). Capture ran on the connected 0.0.75 instance; the 0.0.76 pin loads at next session reconnect (caveat disclosed + documented).

## Honest limitations

- The binding §1c gate enforces from the NEXT session's Q/A spawns (roster snapshot); this step's own qa runs the old snapshot.
- The bumped 0.0.76 server is pinned but not yet the connected instance (stdio no-respawn) — first 0.0.76 connect happens next session; the delta is patch-level with zero flag changes, so risk is minimal.
- Figma integration is docs/workflow only — no Figma file was created for the cockpit (the code-to-design first run is a natural follow-on when the operator wants a design review).
