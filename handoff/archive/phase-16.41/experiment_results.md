---
step: phase-16.41
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/scripts/audit/lighthouse_auth_home.js (180 lines)
  - frontend/package.json (+1 npm script)
  - docs/runbooks/lighthouse-auth-home.md (95 lines, operator runbook)
---

# Experiment Results -- phase-16.41

## What was done

Closed task list item #8 (last remaining non-user-action follow-up):
authenticated-home lighthouse harness. Researcher's critical finding
was that `LIGHTHOUSE_SKIP_AUTH=1` env-var bypass already exists in
`frontend/src/middleware.ts:24`, so this cycle is pure
infrastructure-wiring (no JWE token minting, no Puppeteer, no
crypto code).

### Changes

1. **`frontend/scripts/audit/lighthouse_auth_home.js`** (180 lines):
   - **Probe-only mode** (`--probe-only` flag): static check that
     dev server is running with `LIGHTHOUSE_SKIP_AUTH=1` (HTTP 200
     for `/`, not 302 to `/login`). Fast; suitable for masterplan
     immutable verification or pre-flight scripts. Exit codes:
     0=bypass active, 2=bypass not active or server down.
   - **Default mode** (full): probe + spawn `lighthouse-wrapper.js`
     with `--url http://localhost:3000/ --output json --output-path
     handoff/lighthouse_authenticated_home.json --quiet
     --chrome-flags=--headless` + assert `finalUrl` does NOT contain
     `/login`. Exit codes: 0=PASS, 1=audit ran but failed, 2=bypass
     not active.
   - JSON output envelope mirrors `sovereign_route.js` shape.
   - Reuses existing `lighthouse-wrapper.js` for chrome-path
     discovery + argv translation (no duplication).

2. **`frontend/package.json`** added:
   ```json
   "lighthouse:auth-home": "node scripts/audit/lighthouse_auth_home.js"
   ```

3. **`docs/runbooks/lighthouse-auth-home.md`** (95 lines): operator
   runbook covering: purpose, mechanism, how-to-run (2 terminals),
   probe-only mode, output reading, troubleshooting, related
   infrastructure, and a security note that
   `LIGHTHOUSE_SKIP_AUTH=1` MUST never be set in production.

### Files touched

| Path | Action | Size |
|------|--------|------|
| `frontend/scripts/audit/lighthouse_auth_home.js` | CREATED | 180 lines |
| `frontend/package.json` | edited | +1 npm script |
| `docs/runbooks/lighthouse-auth-home.md` | CREATED | 95 lines |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |

NO middleware changes. NO auth code touched. NO new dependencies.

## Verification (verbatim, immutable)

```
$ test -f frontend/scripts/audit/lighthouse_auth_home.js && \
  grep -q "lighthouse:auth-home" frontend/package.json && \
  grep -q "LIGHTHOUSE_SKIP_AUTH" frontend/scripts/audit/lighthouse_auth_home.js && \
  test -f docs/runbooks/lighthouse-auth-home.md && \
  echo "ALL VERIFICATION PASS"
ALL VERIFICATION PASS
```

**Live smoke-test of probe-only mode** (dev server is currently
running WITHOUT the env var):

```
$ node frontend/scripts/audit/lighthouse_auth_home.js --probe-only
{
  "audit": "lighthouse_auth_home",
  "mode": "probe-only",
  "timestamp": "2026-04-25T19:16:46.614Z",
  "overall": "FAIL",
  "checks": [
    {
      "check": "lighthouse_skip_auth_bypass_active",
      "status": "FAIL",
      "detail": "HTTP 302 -> /login. The dev server is NOT running with LIGHTHOUSE_SKIP_AUTH=1. Restart it with: LIGHTHOUSE_SKIP_AUTH=1 npm run dev"
    }
  ]
}
$ echo $?
2
```

Probe correctly detected the bypass is inactive + emitted the
operator-friendly error message + exited with code 2 (the documented
"bypass not active" code). End-to-end safety net works.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | script_exists | PASS | 180 lines at canonical path |
| 2 | npm_script_added | PASS | `"lighthouse:auth-home"` in package.json |
| 3 | script_aware_of_bypass | PASS | LIGHTHOUSE_SKIP_AUTH appears 5+ times in script |
| 4 | runbook_exists | PASS | 95 lines at docs/runbooks/ |
| 5 | script_probe_works | PASS | exit=2 + helpful error when bypass inactive |

## Honest disclosures

1. **No actual lighthouse run performed in this cycle.** The full
   audit requires the dev server to be restarted with
   `LIGHTHOUSE_SKIP_AUTH=1` -- a Peder-action prerequisite. The
   immutable verification is intentionally infrastructure-only
   ("does the wiring exist?"), not behavior-only. The probe-only
   smoke-test demonstrates the safety net works.

2. **Output goes to `handoff/lighthouse_authenticated_home.json`,
   not under `frontend/handoff/`.** The script writes to repo-root
   `handoff/` so the report lives alongside other harness
   artifacts. The mkdirSync ensures the dir exists.

3. **Reused existing `lighthouse-wrapper.js`** rather than
   duplicating chrome-path discovery + `--url X` translation logic.
   Single source of truth; tested by phase-16.37 vitest cases.

4. **`runLighthouse()` exit-code check.** Lighthouse can return
   nonzero on warnings even when the report is generated; the
   subsequent `fs.existsSync` check on the output path is the real
   "did we get a report?" signal. Both checked.

5. **`finalUrl` field name flexibility.** Lighthouse v13 puts it at
   the top level; older versions nest it under `.lhr.finalUrl`. The
   script tries `finalUrl`, `finalDisplayedUrl`, AND
   `lhr.finalUrl` -- defensive against minor version drift.

6. **No `--probe-only` test in CI.** A probe test would require a
   dev server running, which is not part of the project's existing
   test infrastructure (vitest tests are unit-level). The probe is
   validated by the live smoke-test above; future CI work could add
   a probe step gated on dev-server availability.

## Closes

- Task list item #8 (LAST remaining non-user-action follow-up)
- masterplan step **phase-16.41**

## Next

Spawn Q/A. If PASS: log + flip + final session summary.
