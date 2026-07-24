# live_check -- Step 75.12 (frontend-01/02/03/05/06, fe-ts-01, frontend-09)

Date: 2026-07-24. Verbatim captures; exit codes via rc=$? immediately.

## 1. Immutable verification command (exit 0) -- Main's independent verbatim run

```
$ <masterplan 75.12 verification.command: python3 source-scan asserts + cd frontend && npx tsc --noEmit>
verification_exit=0
```

## 2. Full frontend suite + change surface

```
Test Files  30 passed (30)
Tests       201 passed (201)          (baseline 24 files / 187 tests at 75.6)
$ git diff --stat HEAD -- frontend/ | tail -1
 13 files changed, 407 insertions(+), 66 deletions(-)   (+6 new test files)
```

## 3. Live operator instance after ALL edits (read-only; NO second server)

```
$ curl -s -o /dev/null -w 'login=%{http_code} ' http://localhost:3000/login; curl -s -o /dev/null -w 'root=%{http_code}\n' http://localhost:3000/
login=200 root=302
```

## 4. Playwright capture (criterion 1) + non-discrimination disclosure

Navigated (read-only) to http://localhost:3000/agents -> redirected to
/login (auth wall intact); /login rendered STABLE across ~30s of
navigate/screenshot/snapshot interactions (a reload loop would thrash the
URL). Capture: `handoff/current/captures_75.12/agents_authwall_75.12.png`.
Accessibility snapshot showed the Google + Passkey sign-in UI; the single
console error is the pre-existing queued 75.6.2 triage item.

**DISCLOSURE (binding, from the research gate): DEV_LOCALHOST_BYPASS=1 is
active in the running backend, so no live capture on this box can
discriminate the auth-transport fixes (authed endpoints return 200 without
credentials here). The vitest behavioral suite is the discriminating
evidence: both-directions 401-redirect tests, zero-polls-on-/login,
stop-at-exactly-5 fake-timer breaker tests, not_initialized render test.**

## 5. UI conventions

New UI elements (partial-failure notice, stale segments): Phosphor icons
via @/lib/icons, rose-tinted error styling per frontend-layout section 8,
ASCII text, no emoji. No flag-gated live-loop behavior (frontend-only step).
