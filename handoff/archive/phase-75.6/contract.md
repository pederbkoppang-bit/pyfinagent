# Contract -- masterplan step 75.6

**Step id**: `75.6`
**Name**: Audit75 S6 -- frontend auth fail-closed + session hardening
**Phase**: phase-75 | **Priority**: P0 | **Cycle**: 1 | **Date**: 2026-07-23
**BOUNDARY (from the step)**: no `.env` edits; allowlist enforcement ships DARK behind
`AUTH_ENFORCE_ALLOWLIST` default-off so the operator is NEVER locked out.

---

## 1. Research gate

**PASSED.** Run `wf_1e89656d-6c4`, two legs (researcher + adversarial verifier).
Envelope: `tier=complex`, `external_sources_read_in_full=10`, `snippet_only=18`,
`urls_collected=28`, `recency_scan_performed=true`, `internal_files_inspected=7`,
`gate_passed=true`. Brief: `handoff/current/research_brief_75.6.md`.

### 1a. Claim verdicts (adversarially re-derived)

All five CONFIRMED with verbatim file:line. Corrections that bind this plan:

1. **(a) line attribution**: `hasAuthProvider` is *defined* at `middleware.ts:7` and *used*
   at `:24`. Both occurrences must be removed or the `'hasAuthProvider' not in mw` assert
   fails. Keep `LIGHTHOUSE_SKIP_AUTH`. **Do NOT rename `middleware.ts`** -- the proxy.ts
   rename is Next.js 16+; this project is Next.js 15 and the verification command reads
   `frontend/src/middleware.ts`.
2. **(c) email_verified is authoritatively an OIDC `profile` claim, boolean** (Auth.js
   Google-provider doc: "Google also returns a `email_verified` boolean property in the
   OAuth profile"; OIDC Core). The current code reads it off `account` (aliased as
   `profile` at `auth.config.ts:41`), which is **always undefined** -- the rejection is
   dead code. Fix: destructure the real `profile` param, read `profile?.email_verified`,
   and **keep the "claim present but not truthy" guard** (OIDC permits providers to omit
   the claim -- `undefined` must be allowed, not rejected). **Cast for tsc**: the Auth.js
   `Profile` type may not declare `email_verified`, so a naive read breaks `npx tsc
   --noEmit`; cast e.g. `(profile as { email_verified?: boolean | string } | undefined)`.
3. **(b) backend mirror VERIFIED**: 75.1 shipped (commit `40111b03`, masterplan
   `status=done`). The real flag is `auth_enforce_allowlist` / env `AUTH_ENFORCE_ALLOWLIST`
   (`settings.py:574`, default `False`). Semantics: empty+flag = deny-all;
   empty+no-flag = admit + loud warn; non-empty = enforce. The frontend flag must mirror
   the **env name exactly** and the **empty-list fail-open default**, and accept the same
   truthy forms pydantic does (`true/1/yes/on`...) so the two layers don't desync.
4. **(e)** leave **NO residual `30 * 24`** anywhere, including comments, or the `'30 * 24'
   absent` assert fails -- replace, don't annotate. Prisma adapter is already wired
   (`auth.ts:14`), so the DB-session follow-up is a config change, not new infra.

### 1b. LOCKOUT ANALYSIS -- the P0 concern, resolved

**No part of this change can lock the operator out at executor defaults, not even
transiently -- and the config was PROVEN, not assumed.** Live probe: `curl :3000/` ->
`302 -> /login`, `curl :3000/login` -> `200`. Under the current middleware that 302 is
emitted only at `:29-32`, reachable **only** when the `:24` early-return did NOT fire --
i.e. `hasAuthProvider === true` -> Google creds ARE configured. **The operator runs Google
SSO and it works**, so removing `hasAuthProvider` (fix a) is **inert** for their live
config. `(e)` at worst forces one recoverable re-login (Google works). The only
hard-lockout path is **operator-initiated**: `AUTH_ENFORCE_ALLOWLIST=on` + empty
`ALLOWED_EMAILS` -- which is the intended fail-closed behavior, behind a default-off flag
the executor does not set.

### 1c. VERIFICATION-COMMAND WEAKNESS (must-flag for Q/A)

The immutable command's assert `'profile' in ac.split('signIn(')[1][:250]` is
**mutation-weak**: it ALREADY PASSES on the current buggy file (measured -- `const profile
= account` sits ~86 chars into that window). The string assert passes on a no-op, so it is
**not** evidence that (c) is fixed. **The criterion is immutable and NOT amended**; its
intent is satisfied and additionally proven **semantically** by the new test (see §4),
which asserts the callback destructures `profile` from its params AND reads
`email_verified` off `profile`, never `account`. Recorded here so Q/A does not read the
string assert's pass as sufficient.

---

## 2. Hypothesis

Five independent frontend-auth weaknesses, each a *silent* fail-open: route protection
disabled by provider-absence inference, an empty allowlist admitting anyone, a dead
unverified-email check, an emailless-principal wave-through, and a 30-day
non-revocable session on a kill-switch cockpit. Repairing them makes the frontend layer
fail *closed* and mirror the backend (75.1), with the operator-risking parts behind a
default-off flag.

---

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

> 1. middleware.ts enforces req.auth on protected routes regardless of which providers are configured (no Google-cred-presence early-return); the only bypasses are the explicit LIGHTHOUSE_SKIP_AUTH flag and (optionally) a named DEV_DISABLE_AUTH-style opt-in -- never inferred provider absence
> 2. auth.config.ts: with AUTH_ENFORCE_ALLOWLIST unset the current admit behavior is byte-equivalent (no operator lockout, executor edits no env files) but an empty allowlist logs a loud warning; with the flag set, empty allowlist rejects all sign-ins
> 3. signIn reads email_verified from the OIDC profile parameter (account alias removed) so unverified Google emails are actually rejected -- covered by a unit or manual-evidence check recorded in experiment_results.md
> 4. With a non-empty allowlist, a principal lacking an email is rejected (no && user.email fall-through to return true)
> 5. Session: maxAge is exactly 7 days with an explicit updateAge; the JWT-revocation limitation and database-session follow-up are documented in the code comment or security.md
> 6. npx tsc --noEmit passes and the operator login flow is re-verified against the running app per the frontend live-UI protocol (Playwright capture referenced in the live evidence file)

**Command** (immutable):
```
cd /Users/ford/.openclaw/workspace/pyfinagent && python3 -c "mw=open('frontend/src/middleware.ts').read(); assert 'hasAuthProvider' not in mw...; ..." && cd frontend && npx tsc --noEmit
```
**live_check**: `handoff/current/live_check_75.6.md` -- verbatim command output (exit 0) +
`git diff --stat`; ON-vs-OFF $0 diff for the flag-gated allowlist behavior; **Playwright
capture** of the operator login flow (UI-touching). Findings: gap2-01, -02, -04, -05, -06.

*No criterion amended. Criterion 3's evidence is the semantic test (§1c). Criterion 6's
Playwright capture uses the :3100 skip-auth rig per `frontend.md` -- the operator's :3000
is never touched.*

---

## 4. Plan

### middleware.ts (a / gap2-01)
Remove `hasAuthProvider` (both `:7` def and `:24` use). Bypass block keeps
`LIGHTHOUSE_SKIP_AUTH === "1"` and adds an **explicit** `DEV_DISABLE_AUTH === "1"`
opt-in (default-off, never inferred). `req.auth` enforcement (`:29-32`) unchanged.

### auth.config.ts (b, c, d, e)
- **flag**: `enforceAllowlist` = `AUTH_ENFORCE_ALLOWLIST` parsed against pydantic's truthy
  set (`true/1/yes/on/t/y`). Module-load `console.warn` (loud) when `allowedEmails` is
  empty -- one message for flag-on (deny-all), one for flag-off (admit + how to restrict).
- **(c)** `signIn({ user, account, profile })`; read
  `(profile as {email_verified?: boolean|string}|undefined)?.email_verified`; reject only
  when the claim is **present and not truthy** (`undefined` allowed).
- **(b+d)** allowlist block: if `enforceAllowlist && allowedEmails.length === 0` -> `return
  false` (fail-closed). If `allowedEmails.length > 0`: `if (!user.email) return false`
  (gap2-06), then membership check.
- **(e)** `session.maxAge: 7 * 24 * 60 * 60`, `updateAge: 24 * 60 * 60`; comment documents
  the JWT-revocation limit + the `strategy:"database"` Prisma follow-up. No `30 * 24`
  literal anywhere.

### Test (criterion 3 semantic proof)
New `frontend/src/lib/__tests__/auth.config.test.ts` (or a node-run assertion if vitest is
absent -- verify tooling first). Asserts BEHAVIOURALLY via the exported signIn callback:
empty+flag rejects; empty+no-flag admits (+warn); `email_verified:false` on `profile`
rejects; `email_verified:undefined` admits; non-empty allowlist + no email rejects;
non-member rejects; member admits. AST/structural check that `email_verified` is read from
`profile`, not `account`.

---

## 5. Mutation matrix (mandatory)

Per the cycle-131 durable rule + the new qa.md §4b. Each must FAIL a test:
M1 restore the `hasAuthProvider` early-return; M2 alias `account as profile` again (the
(c) dead-code bug) -- the SEMANTIC test must catch this even though the immutable string
assert cannot; M3 drop the `!user.email` reject; M4 make empty+flag `return true`;
M5 revert maxAge to 30d; M6 remove `updateAge`; M7 accept `email_verified` off `account`.
**M2 is the load-bearing mutant** -- it is the exact defect the immutable command misses.

## 6. Risks

- **R1 (flag desync)**: frontend `'1' vs 'true'` vs backend pydantic. Mitigation: accept
  pydantic's full truthy set.
- **R2 (tsc)**: `Profile` type may not declare `email_verified` -> cast (planned).
- **R3 (login-time vs request-time)**: the frontend signIn change gates only NEW logins;
  live sessions are re-checked by the backend (75.1). Do NOT claim the frontend change
  revokes a live session.
- **R4 (passkey path)**: WebAuthn users also traverse signIn (`account.provider==='passkey'`),
  skipping the Google email_verified branch; the allowlist + no-email reject still apply to
  their linked email -- verify a passkey user has a stored email so (d) doesn't reject them.
- **R5 (visual)**: no pixels change (auth logic only); the Playwright capture proves the
  login flow still renders + redirects, not a new view.

## 7. References
`handoff/current/research_brief_75.6.md` (`wf_1e89656d-6c4`); Auth.js v5 middleware +
signIn-callback + session docs; OIDC Core `email_verified`; OWASP session-management +
authn cheat sheets; `.claude/rules/frontend.md` (live-UI :3100 protocol);
`.claude/rules/security.md`; backend 75.1 (`settings.py:574`, commit `40111b03`);
new `.claude/agents/qa.md` §4b (claim auditing).
