# Live Check -- phase-91.9
Step: internal phase-tracking label "(phase-25.C7)" leaks into user-facing Data Freshness page copy

## Required evidence (per masterplan step 91.9's `verification.live_check`)
"Playwright screenshot of the observability/Data Freshness page showing the subtitle with no phase-tag"

## Evidence

Playwright MCP navigation to `http://localhost:3000/observability` behind the real, authenticated
NextAuth session (`.playwright-mcp/storage-state.json`, minted via
`scripts/qa/mint_playwright_storage_state.py`). URL after navigation confirmed as the target page
(`http://localhost:3000/observability`), **not** a `/login` redirect -- a redirect would mean the
session cookie didn't take and the capture would be no evidence per this project's Playwright
doctrine (`.claude/rules/frontend.md`).

Waited for the page to fully settle (`Computed at 2026-08-20T20:49:59.547889+00:00` visible, all 6
freshness-source rows populated, `Overall: Fresh`) before capturing, rather than screenshotting a
loading/transient state.

**Screenshot:** `handoff/current/captures_91.9/observability_no_phase_tag_v2_settled.png`

The Data Freshness page subtitle reads exactly:
> Per-table age + SLA bands across the warehouse

No `(phase-25.C7)` or any other internal phase-tracking reference is visible in the rendered text.
This matches the fix at `frontend/src/app/observability/page.tsx:114-121`, which relocated the
`phase-25.C7` provenance into a JSX comment immediately above the subtitle (per the research
brief's Pitfall 3 -- relocate, don't delete) rather than erasing it outright.

A second, earlier capture from the same session
(`handoff/current/captures_91.9/observability_no_phase_tag.png`, taken immediately after the
initial fix, before the Pitfall-3 amendment) is retained for provenance; this file supersedes it
as the live_check evidence since it was taken after the contract amendment and shows the fully
settled page.

## Verification command re-run (post-amendment)
```
$ grep -rnE '\(phase-[0-9]' frontend/src/app frontend/src/components --include='*.tsx' | grep -v '\.test\.tsx' | grep -vE '^[^:]+:[0-9]+: *(//|\*|\{/\*)'
(no output, exit 1)
```
Zero hits, confirming the relocated comment does not reintroduce the leak the immutable command checks for.
