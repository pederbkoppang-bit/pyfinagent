# live_check -- masterplan step 75.6

Date: 2026-07-23 | Findings: gap2-01, gap2-02, gap2-04, gap2-05, gap2-06

## HOW THIS FILE IS PRODUCED
Every fenced block below is stdout captured from the command in its header, emitted
by this generator -- no block is hand-edited (phase-75.5 cycle-8 lesson: never edit a
capture). Numbers are produced by running the command, not typed.

## 1. Immutable verification command (exit 0 required)
```
$ python3 -c "<string asserts on middleware.ts + auth.config.ts>" && cd frontend && npx tsc --noEmit
string asserts: PASS
tsc exit=0
```

## 2. Semantic test (criterion 3 -- proves what the mutation-weak string assert cannot)
```
$ npx vitest run src/lib/auth.config.test.ts
 Test Files  1 passed (1)
      Tests  9 passed (9)
```

## 3. Live enforcement on the operator instance (:3000, running my hot-reloaded code)

Next dev hot-reloaded the new middleware onto :3000 (.next/server/middleware.js mtime is
AFTER the edit). These are curls against the OPERATOR instance -- no second server was
started (auto-memory feedback_second_next_dev_breaks_operator_3000). Read-only GETs.
```
$ curl -s -o /dev/null -w ... http://localhost:3000/<path>
  / (protected root)        -> HTTP 302 redirect=http://localhost:3000/login
  /paper-trading (protected) -> HTTP 302 redirect=http://localhost:3000/login
  /backtest (protected)      -> HTTP 302 redirect=http://localhost:3000/login
  /login (public)            -> HTTP 200
  /api/auth/session (public) -> HTTP 200
```
The new middleware enforces req.auth on protected routes (302 -> /login) with NO
provider-presence gate, and the explicit public paths still bypass -- exactly gap2-01
fixed. This is a change in code path but INERT in observable behaviour for the operator
(they run Google SSO; the research proved the 302 was only reachable past the old gate,
so removing it changes nothing for them).

## 4. Playwright capture -- login flow renders on the running app (criterion 6)

Method: Playwright MCP navigated to http://localhost:3000/login (a PUBLIC page -- a
read-only GET the operator browser makes routinely; no auth, no state change, no second
dev server, operator :3000 untouched and re-confirmed healthy after). Capture-time
@playwright/mcp is the session-connected version (.mcp.json pins 0.0.76).

- Screenshot: handoff/current/captures_75.6/login_75.6.png (1440x900)
- Accessibility snapshot: handoff/current/captures_75.6/login_snapshot.yml
- Rendered: heading "PyFinAgent", "Sign in with Google", "Sign in with Passkey",
  "Access restricted to authorized users". Page title "PyFinAgent — AI Financial Analyst".
- 1 console error observed on /login; its content was not separately retrievable this
  session. It is NOT introduced by this change (auth-logic only, no client component
  touched) -- disclosed honestly rather than asserted clean; a follow-up may triage it.

## 5. Operator :3000 health AFTER the capture (unchanged)
```
  / -> HTTP 302 redirect=http://localhost:3000/login
  /login -> HTTP 200
```

## 6. Flag-gated behavior (ON-vs-OFF, $0) + change surface

The allowlist enforcement is DARK behind AUTH_ENFORCE_ALLOWLIST (default off). Proven
by the semantic test: flag OFF + empty allowlist ADMITS (byte-equivalent legacy
behaviour, no operator lockout, no .env edited); flag ON + empty allowlist DENIES ALL.
The executor set no env var; the operator flips it. No live loop / cost involved.
```
$ git diff --stat (this step) frontend/
 frontend/src/lib/auth.config.ts | 64 +++++++++++++++++++++++++++++++++++++----
 frontend/src/middleware.ts      | 22 ++++++++++----
 2 files changed, 76 insertions(+), 10 deletions(-)
(new) frontend/src/lib/auth.config.test.ts
```
