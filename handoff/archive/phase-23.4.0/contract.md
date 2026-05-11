---
step: phase-23.4.0
title: Recover frontend from stale .next/ corruption (login 500)
cycle_date: 2026-05-08
harness_required: true
verification: 'curl -s -o /dev/null -w %{http_code} http://localhost:3000/login returns 200; curl -sL http://localhost:3000/ -o /tmp/_root.html && grep -E "<html|<title" /tmp/_root.html; cd frontend && npx --no-install tsc --noEmit; cd frontend && npx --no-install eslint . --quiet'
research_brief: handoff/current/phase-23.4.0-research-brief.md
---

# Contract — phase-23.4.0

## Hypothesis

`GET /login` returns HTTP 500 because the running `next dev` server
(launchd job `com.pyfinagent.frontend`, PIDs 86232/86274) is serving
from a corrupted `frontend/.next/` directory: `routes-manifest.json`
is absent. Cause: a prior `next start` collided with the running dev
server (`EADDRINUSE :::3000` is line 1 of `frontend.log`) and
interrupted the dev server mid-compile between its initial write
(22:11) and an HMR re-emit cycle (22:42), leaving the startup-only
`routes-manifest.json` permanently absent.

The downstream `Cannot find module './611.js'` error is a SYMPTOM,
not the cause: per the researcher's audit `611.js` IS present in
`frontend/.next/server/chunks/`. The router bootstrap fails before
it can execute the chunk because the manifest needed to register
the route is missing.

The fix is to wipe `frontend/.next/` and let launchd's
`KeepAlive=true` respawn `next dev`, which performs a full compile
on fresh startup and writes `routes-manifest.json` along with every
other manifest. We additionally add a `predev` script to
`frontend/package.json` to prevent recurrence: any future
interrupted-write that leaves `.next/` partial gets nuked on the
next launchd respawn cycle.

## Research-gate summary

`researcher` agent `aeda4de214c83a7bc` ran tier=moderate and
returned `gate_passed: true` with:
- 7 external sources fetched in full via WebFetch (≥5 floor cleared)
- 7 snippet-only + 7 read-in-full = 14 URLs (clears the ≥10 floor)
- Recency scan 2024-2026 performed (Apr 2026 stale-chunk playbook +
  Mar 2025 GitHub issue #76766 against 15.1.7; no 15.5.x-specific
  regression found)
- Three-query discipline followed (current-year, last-2-year,
  year-less)
- 10 internal files inspected (incl. `frontend/.next/`,
  launchd plist, `frontend/package.json`, `frontend.log`)

Brief: `handoff/current/phase-23.4.0-research-brief.md`.

**Researcher's recommended sequence:** Option A (full wipe +
`launchctl kickstart -k`). Option B (HMR-only) is ruled out by
architecture — HMR does not regenerate startup-only manifests.
Option C (`next build` first) is wasted work and risks dev/prod
artifact mixing.

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.4.0.verification`:

```
curl -s -o /dev/null -w %{http_code} http://localhost:3000/login returns 200;
curl -sL http://localhost:3000/ -o /tmp/_root.html && grep -E "<html|<title" /tmp/_root.html;
cd frontend && npx --no-install tsc --noEmit;
cd frontend && npx --no-install eslint . --quiet
```

Decoded into deterministic checks the GENERATE step + Q/A must perform:

1. `curl http://localhost:3000/login` returns HTTP **200** (not 500,
   not redirect — the login page must actually render).
2. `curl -sL http://localhost:3000/` (follow redirects) returns
   non-empty HTML containing at least one of `<html` or `<title>`
   tags. (Confirms the app shell is rendering, not a bare error
   string.)
3. `cd frontend && npx --no-install tsc --noEmit` exits 0 (no
   TypeScript regression introduced by the fix).
4. `cd frontend && npx --no-install eslint . --quiet` exits 0 (no
   error-level lint introduced; pre-existing warnings tolerated per
   project convention).

## Plan steps

1. (DONE — RESEARCH) Researcher returned brief, `gate_passed: true`,
   recommended Option A.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Snapshot the current `frontend/.next/` listing for the
      experiment_results record (lines + file count).
   b. Edit `frontend/package.json` — add `"predev": "rm -rf .next"`
      to `scripts`. This is the regression guard: idempotent (no-op
      when `.next/` is absent), zero-cost on subsequent launches.
   c. `rm -rf /Users/ford/.openclaw/workspace/pyfinagent/frontend/.next/`
   d. `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend`
      — sends SIGTERM to PIDs 86232/86274 and immediately re-fires
      `next dev --port 3000` with all plist env vars. No sudo
      needed (user-domain LaunchAgent).
   e. Wait for the dev server to recompile (poll `curl -s -o /dev/null
      -w %{http_code} http://localhost:3000/login` until it returns
      200; 30s budget).
   f. Run all 4 immutable verification commands verbatim. Capture
      output.
   g. Write `tests/verify_phase_23_4_0.py` (a 4-check Python
      verifier) so Q/A can replay deterministically.
   h. Write `handoff/current/experiment_results.md` with verbatim
      output.
4. **EVALUATE phase:** spawn fresh `qa` agent. 5-item harness-
   compliance audit FIRST (per `feedback_qa_harness_compliance_first.md`),
   then deterministic re-verification (curl + tsc + eslint), then LLM
   judgment.
5. **LOG phase:** append `handoff/harness_log.md` AFTER Q/A returns
   PASS / CONDITIONAL. Flip `.claude/masterplan.json` 23.4.0 +
   phase-23.4 status only after the log append (log-last per
   `feedback_log_last.md`).

## Anti-patterns guarded (≥2)

1. **Treating the symptom (`611.js` chunk error)** instead of the
   root cause (`routes-manifest.json` absent). Selectively
   re-creating chunks would not fix the manifest; the manifest
   re-creation requires a full `next dev` startup compile.
2. **Soft-kick (touch a file, hope HMR rebuilds)** — Option B in the
   researcher's enumeration. Ruled out: HMR file-watcher rebuilds do
   NOT regenerate startup-only manifests like `routes-manifest.json`.
   The dev server cannot self-heal this corruption.
3. **`next build` before `next dev` restart** — Option C. Wasted
   work and risks mixing prod build artifacts (BUILD_ID-keyed) with
   dev server cache.
4. **Forgetting the regression guard** — without `predev` cleanup,
   any future `next start` collision (or any other process that
   interrupts a `.next/` write) will re-corrupt the cache. The
   `predev` script makes recovery automatic on the next launchd
   respawn.
5. **Killing the launchd job's PID directly without using
   kickstart** — `launchctl bootout` followed by `bootstrap` is the
   modern API; `unload`/`load` is legacy. `kickstart -k` is the
   current best practice (cited in launchd.info + rakhesh.com 2025).

## Out of scope

- Auth-flow changes (`/login` page logic). The 500 is a build-cache
  corruption, not a code defect.
- Migrating to Turbopack. Turbopack has different cache semantics
  but adopting it is a separate decision (cited in researcher brief
  as a possible long-term mitigation).
- Backend changes (port 8000 is fine; backend `/health` 404 is a
  separate unrelated finding).
- Restoring `cycle_history.jsonl` divergence (phase-23.2.1 finding).

## Backwards compatibility

- `predev` script is purely additive. `npm run dev` semantics change
  to "wipe `.next/` then start" — this is the standard pattern in
  the Next.js practitioner literature and adds <1s of overhead per
  launch.
- No code changes. No frontend route or component edits. No backend
  edits. No migration scripts.

## Risk

- **Recompile time on respawn** — `next dev` startup compile may
  take 10-30s before the first request returns 200. The verification
  must allow for this (poll-with-budget loop in the verifier).
- **launchd respawn loop** — if the FIRST startup also fails for any
  reason, `KeepAlive=true` + `ThrottleInterval=5` will retry but
  loop on the same error. Mitigation: capture the new
  `frontend.log` tail in `experiment_results.md` so any persistent
  failure is visible.
- **The `611.js` symptom may persist transiently** during the first
  recompile. The hypothesis predicts it disappears once the new
  manifest set is written. Q/A must verify by tailing the log AFTER
  the 200 response is observed.
- **No backup** of `.next/` is taken. This is intentional — `.next/`
  is fully derivable from source + `node_modules/`. Per
  `.gitignore`, it is an artifact directory.

## References

- Research brief: `handoff/current/phase-23.4.0-research-brief.md`
  (researcher `aeda4de214c83a7bc`, 7 sources read in full, gate
  passed, Option A recommended).
- Masterplan: `.claude/masterplan.json::23.4.0` — verification
  copied verbatim above.
- Anthropic harness-design: https://www.anthropic.com/engineering/harness-design-long-running-apps
  ("verification criteria are immutable").
- Next.js module-not-found: https://nextjs.org/docs/messages/module-not-found
- Next.js deploying (manifests are startup-only):
  https://nextjs.org/docs/app/getting-started/deploying
- vercel/next.js#76766 (manifest-not-found in dev):
  https://github.com/vercel/next.js/issues/76766
- launchctl kickstart pattern: https://rakhesh.com/mac/macos-launchctl-commands/
- Stale chunk playbook (Apr 2026): https://medium.com/@tarequlislam105150/next-js-stale-chunk-cache-mismatch-resolution-playbook-cdcf556b594c
