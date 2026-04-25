---
step: phase-16.33
title: #9 partial -- sovereign_route.js audit + lighthouse --url wrapper + sovereign sidebar entry
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.33

## Research-gate summary

`handoff/current/phase-16.33-research-brief.md`. tier=simple, 6 in-full, 13 URLs, recency scan, gate_passed=true.

## Key research findings

1. **`sovereign_route.js` pattern is `sovereign_consistency.js` directly.** Three checks, JSON output, `process.exit(0/1)`. Stdlib only (`http`, `fs`, `path`).

2. **CRITICAL: `/sovereign` is NOT in `NAV_SECTIONS`.** Sidebar.tsx lines 21-54 have 4 sections (Analyze, Reports, Trading, System) — no sovereign entry. If we just create the audit script, `sidebar_entry_exists` check FAILs. **Must ALSO add the sidebar entry.** This is a real shipping gap surfaced by trying to make the verification command work.

3. **Lighthouse v13.1.0 takes URL positionally** — no `--url` flag exists in any lighthouse release. Wrapper approach: ~20-LOC `lighthouse-wrapper.js` that translates `--url X` → positional `X` and passes other flags through untouched.

4. **`package.json` change**: `"lighthouse": "lighthouse"` → `"lighthouse": "node scripts/audit/lighthouse-wrapper.js"`.

5. **Scope fits one cycle**: ~100 LOC across 2 new files + 1 package.json line change + Sidebar.tsx entry. Zero new npm deps.

## Hypothesis

Four small additions ship together:
- `frontend/scripts/audit/sovereign_route.js` (NEW, ~80 LOC) — 3 checks
- `frontend/scripts/audit/lighthouse-wrapper.js` (NEW, ~25 LOC) — `--url` translator
- `frontend/package.json` (1-line change to `lighthouse` script)
- `frontend/src/components/Sidebar.tsx` (add `/sovereign` entry to NAV_SECTIONS)

After: `node scripts/audit/sovereign_route.js` exits 0 with `{check_route_reachable: PASS, check_sidebar_entry: PASS, check_page_shell: PASS}`. `npm run lighthouse -- --url http://localhost:3000 --output json --output-path X` works as written.

## Success Criteria (verbatim, immutable)

```
test -f frontend/scripts/audit/sovereign_route.js && cd frontend && node scripts/audit/sovereign_route.js && cd .. && cd frontend && npm run lighthouse -- --url http://localhost:3000 --output json --output-path handoff/lighthouse_smoke.json --quiet 2>&1 | tail -3 || true
```

- sovereign_route_js_exists
- audit_script_passes
- lighthouse_url_flag_works

## Plan steps

1. Read `sovereign_consistency.js` (template), `package.json` (current lighthouse script), `Sidebar.tsx` (NAV_SECTIONS), `sovereign/page.tsx` (page shell shape)
2. Create `lighthouse-wrapper.js` (~25 LOC stdlib `child_process.spawnSync`)
3. Update `package.json` lighthouse script
4. Add sovereign entry to `Sidebar.tsx` NAV_SECTIONS
5. Create `sovereign_route.js` (~80 LOC, 3 checks, JSON output)
6. Run `node scripts/audit/sovereign_route.js` standalone — confirm 3/3 PASS
7. Run the verbatim verification command — confirm exits 0
8. Run vitest to confirm no regression on Sidebar test (if any)
9. Spawn Q/A

## What Q/A must audit

1. Both new scripts exist + are executable Node files
2. `sovereign_route.js` checks: route_reachable + sidebar_entry + page_shell — all 3 verified independently
3. Lighthouse wrapper: `--url X` translates to positional `X` correctly; other flags pass through
4. Sidebar.tsx now has `/sovereign` entry
5. vitest no regression
6. The 3rd broken verification command (10.5.0 `cd backend && pytest` calendar shadow) is NOT part of this cycle — that's 16.34 separately
