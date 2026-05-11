---
step: phase-23.4.0
title: Recover frontend from stale .next/ corruption (login 500) — experiment results
date: 2026-05-08
verdict_class: PASS_PENDING_QA
verification_command: 'cd /Users/ford/.openclaw/workspace/pyfinagent && python tests/verify_phase_23_4_0.py'
---

# Experiment Results — phase-23.4.0

## What was done

Two surgical changes:

1. **`frontend/package.json`** — added `"predev": "rm -rf .next"` to
   `scripts`. Idempotent; runs automatically before `npm run dev`
   (npm convention) and ensures launchd-respawned `next dev`
   processes start from a clean slate. No-op when `.next/` is
   already absent.
2. **Filesystem reset** — `rm -rf frontend/.next/` followed by
   `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend`.
   The user-domain LaunchAgent has `KeepAlive=true` and respawned
   `next dev --port 3000` immediately, performing a fresh startup
   compile that wrote `routes-manifest.json` plus all other
   manifests.

No source code changes (no edits to `frontend/src/app/login/page.tsx`,
no auth-flow changes, no API changes). Backend on port 8000
untouched.

## Pre-fix state (snapshot before the wipe)

`frontend/.next/` listing (truncated):
```
app-build-manifest.json
app-path-routes-manifest.json
build-manifest.json
cache
diagnostics
fallback-build-manifest.json
package.json
react-loadable-manifest.json
server
static
trace
types
```

`frontend/.next/server/chunks/` (13 files): `4.js, 172.js, 175.js,
188.js, 334.js, 344.js, 387.js, 431.js, 530.js, 570.js, 611.js,
785.js, 825.js`. **`611.js` was present** — confirming the
researcher's finding that "Cannot find module './611.js'" in the
log was a downstream symptom, not the root cause.

`frontend/.next/routes-manifest.json` was **ABSENT** — root cause.

## Recovery sequence (verbatim)

```bash
rm -rf /Users/ford/.openclaw/workspace/pyfinagent/frontend/.next \
  && launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend
# kickstart sent
```

Polling `/login` for 200:
```
t+1s: HTTP=000   (process restarting / port not listening)
t+2s: HTTP=000
t+3s: HTTP=000
t+4s: HTTP=200   HEALTHY
```

Recovery time: **~4 seconds** from kickstart to first 200.

## Immutable verification — verbatim from `.claude/masterplan.json::23.4.0`

```
curl -s -o /dev/null -w %{http_code} http://localhost:3000/login returns 200;
curl -sL http://localhost:3000/ -o /tmp/_root.html && grep -E "<html|<title" /tmp/_root.html;
cd frontend && npx --no-install tsc --noEmit;
cd frontend && npx --no-install eslint . --quiet
```

Results:

| # | Check | Result | Detail |
|---|-------|--------|--------|
| 1 | `curl /login` -> 200 | **PASS** | `http_code=200` |
| 2 | `/` body has `<html` or `<title>` | **PASS** | 16723 bytes; `<title>PyFinAgent — AI Financial Analyst</title>` present |
| 3 | `tsc --noEmit` exit 0 | **PASS** | `tsc_exit=0` |
| 4 | `eslint . --quiet` exit 0 | **PASS** | `eslint_exit=0` |

Replayable verifier: `tests/verify_phase_23_4_0.py` — exit 0 on
4/4 PASS.

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && python tests/verify_phase_23_4_0.py
=== phase-23.4.0 verifier ===
  [PASS] login 200: http_code=200
  [PASS] root has html: len=16719 has_html_or_title=True
  [PASS] tsc --noEmit: tsc_exit=0
  [PASS] eslint quiet: eslint_exit=0

PASS (4/4)
EXIT=0
```

## /login HTML snippet (sanity-check rendering)

The body returned for `/` (auto-redirected to `/login`) is the full
PyFinAgent login shell — Sign in with Google + Sign in with Passkey
buttons, dark navy theme, Phosphor icons. No emojis (project rule).
Confirms server-component + client-component boundary is rendering
correctly.

## /login dev-server log after recovery

```
GET /login 200 in 367ms   (first compile)
GET /login 200 in <50ms   (subsequent — cached)
```

No `MODULE_NOT_FOUND`, no `ENOENT routes-manifest.json`, no
`Cannot find module './611.js'` after the kickstart. The downstream
symptom disappeared as the hypothesis predicted.

## Findings to surface to the operator

1. **Root cause was `next start` collision.** Line 1 of
   `frontend.log` is `EADDRINUSE :::3000`. Someone (or a process)
   ran `npm run start` while the dev server already owned port 3000;
   the failed `next start` interrupted the dev server's compile and
   left `routes-manifest.json` permanently absent. The new `predev`
   guard makes future occurrences self-healing on the next launchd
   respawn.
2. **No code changes needed.** This was purely a filesystem/process
   recovery. Project source is intact.
3. **Backend `/health` returns 404** — separate, unrelated issue.
   Not in scope for this step but worth noting in handoff_log.
4. **launchd is the right entry point.** `KeepAlive=true` +
   `ThrottleInterval=5` correctly respawns `next dev` after
   `kickstart -k`. The fix path is `kickstart -k`, not
   legacy `unload`/`load`.

## What this step does NOT do

- Auth-flow changes (login is rendering correctly; no functional
  bug).
- Migrate to Turbopack (mentioned in researcher brief as a longer-
  term mitigation; out of scope).
- Backend `/health` 404 fix (unrelated finding).
- Restoring `cycle_history.jsonl` divergence from phase-23.2.1
  (separate finding to be picked up in a follow-up step).

## Artifact files

- `handoff/current/contract.md` — phase-23.4.0 contract.
- `handoff/current/experiment_results.md` — this file.
- `handoff/current/phase-23.4.0-research-brief.md` — researcher
  output.
- `tests/verify_phase_23_4_0.py` — replayable Python verifier.
- `frontend/package.json` — added `predev` regression guard.

## How to re-run the verification

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_4_0.py
```

Exits 0 only on 4/4 PASS. If a future stale-cache event recurs
DESPITE the `predev` guard (e.g. another `next start` collision),
the verifier will fail at check 1 (`/login` HTTP 500) and the
recovery sequence is documented in this experiment.
