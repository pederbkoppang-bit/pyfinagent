---
step: phase-16.41
title: Authenticated-home lighthouse harness (#8)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - frontend/scripts/audit/lighthouse_auth_home.js (~120 LOC)
  - frontend/package.json (+1 npm script "lighthouse:auth-home")
  - docs/runbooks/lighthouse-auth-home.md (~50 LOC operator runbook)
---

# Sprint Contract -- phase-16.41

## Research-gate summary

`handoff/current/phase-16.41-research-brief.md`. tier=moderate, 6 in-full,
14 URLs, recency scan present, gate_passed=true. **Critical finding:**
`LIGHTHOUSE_SKIP_AUTH=1` env-var bypass already exists in
`frontend/src/middleware.ts:24`. No JWE minting needed. Option C
(simplest) is the right choice; saves ~60 lines of crypto code.

## Hypothesis

The auth bypass already exists; this cycle adds the **harness wiring**
(audit script + npm script + operator runbook) so the bypass is
discoverable + repeatable. The actual lighthouse run is operator-
invoked (requires `npm run dev` running with the env var), so the
masterplan immutable verification only checks that the infrastructure
is in place, not that lighthouse completes successfully.

## Concrete plan

1. **`frontend/scripts/audit/lighthouse_auth_home.js`** (~120 LOC):
   - Probe `http://localhost:3000/` to confirm middleware bypass is
     active (expects HTTP 200, NOT 302 to /login)
   - If probe fails: print clear error message ("Start dev server with
     `LIGHTHOUSE_SKIP_AUTH=1 npm run dev`") + exit 2
   - If probe succeeds: invoke `lighthouse-wrapper.js` via spawnSync
     with `--url http://localhost:3000/ --output json
     --output-path ../handoff/lighthouse_authenticated_home.json
     --quiet --chrome-flags="--headless"`
   - After lighthouse run: read output JSON, assert
     `finalUrl` does NOT contain `/login` (proves auth bypass was
     active during the audit)
   - Emit JSON result + exit 0 on PASS / 1 on FAIL
   - Pattern mirrors `sovereign_route.js` exactly

2. **`frontend/package.json`** add npm script:
   ```json
   "lighthouse:auth-home": "node scripts/audit/lighthouse_auth_home.js"
   ```

3. **`docs/runbooks/lighthouse-auth-home.md`** (~50 LOC operator
   runbook):
   - When/why to use this audit
   - Required env-var setup: `LIGHTHOUSE_SKIP_AUTH=1 npm run dev`
   - How to run: `npm run lighthouse:auth-home`
   - Output location + how to read scores
   - Reference to phase-16.33 lighthouse-wrapper for shared infra

## Success Criteria (verbatim, immutable)

```
test -f frontend/scripts/audit/lighthouse_auth_home.js && \
grep -q "lighthouse:auth-home" frontend/package.json && \
grep -q "LIGHTHOUSE_SKIP_AUTH" frontend/scripts/audit/lighthouse_auth_home.js && \
test -f docs/runbooks/lighthouse-auth-home.md
```

(Static infrastructure check — does NOT actually run lighthouse, since
that requires a live dev server with the env var set.)

Plus:
- `script_exists`: file at `frontend/scripts/audit/lighthouse_auth_home.js`
- `npm_script_added`: package.json has `"lighthouse:auth-home"` entry
- `script_aware_of_bypass`: script source mentions LIGHTHOUSE_SKIP_AUTH
- `runbook_exists`: docs/runbooks/lighthouse-auth-home.md
- `script_probe_works`: `node frontend/scripts/audit/lighthouse_auth_home.js
  --probe-only` returns non-zero rc when dev server is NOT in bypass
  mode (proving the safety net works)

## What Q/A must audit

1. Verification command exits 0.
2. Audit script source contains LIGHTHOUSE_SKIP_AUTH guidance text.
3. Audit script probe-only mode returns non-zero when bypass is NOT
   active (current dev-server state — server is running but without
   the env var, returns 302).
4. npm script wired correctly.
5. Runbook documents the env-var setup explicitly.
6. No backend changes; no other frontend changes outside the new
   script + package.json edit.
7. Pattern consistency with sovereign_route.js (output JSON shape,
   pass/fail/check helpers).
