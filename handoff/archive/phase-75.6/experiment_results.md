# Experiment Results -- masterplan step 75.6

**Step**: 75.6 -- Audit75 S6, frontend auth fail-closed + session hardening
**Cycle**: 1 | **Date**: 2026-07-23 | **Priority**: P0
**Contract**: `handoff/current/contract.md` | **Research**: `research_brief_75.6.md`
(gate `wf_1e89656d-6c4`, PASSED)

---

## 1. Verbatim verification command output

```
$ python3 -c "<string asserts>" && cd frontend && npx tsc --noEmit
string asserts: PASS
tsc exit=0
```

Full transcript + the live curls + the Playwright capture are in
`handoff/current/live_check_75.6.md` (generated, not hand-edited).

---

## 2. What changed (measured)

`git diff --stat HEAD` (frontend surface):
- `frontend/src/middleware.ts` -- gap2-01
- `frontend/src/lib/auth.config.ts` -- gap2-02, -04, -05, -06
- `frontend/src/lib/auth.config.test.ts` -- NEW, 9 semantic tests

### (a) gap2-01 -- middleware fail-closed
Removed the `hasAuthProvider` gate (both the `:7` definition and the `:24` use). Route
protection no longer infers "auth off" from Google-credential absence -- a passkey-only or
misconfigured deployment used to silently disable auth for **every** route. The only
bypasses are now the explicit, default-off `LIGHTHOUSE_SKIP_AUTH` and a named
`DEV_DISABLE_AUTH` opt-in. `req.auth` enforcement is unchanged.

**Comment-token trap I hit and fixed**: my first middleware comment documented the removed
gate by naming the literal `hasAuthProvider`, which tripped the immutable command's
`'hasAuthProvider' not in mw` string assert. Reworded the comment to describe it without
the token. (Noted because it is the same string-brittleness class as phase-75.5 -- the
verification command is naive; the comment must avoid the forbidden literal.)

### (b) gap2-02 -- allowlist fail-closed, DARK behind AUTH_ENFORCE_ALLOWLIST
`enforceAllowlist` parses `AUTH_ENFORCE_ALLOWLIST` against pydantic's truthy set
(`true/1/yes/on/t/y`) so the frontend flag cannot desync from the backend bool. Empty
allowlist + flag ON -> deny all; empty + flag OFF -> admit (byte-equivalent legacy
behaviour) + a **loud** module-load `console.warn`. Mirrors 75.1's backend
`auth_enforce_allowlist` (`settings.py:574`, default off). **No `.env` edited** -- the
executor never sets the flag; the operator flips it, and the two layers only fail closed
together when both are on.

### (c) gap2-04 -- unverified-email rejection made live
`signIn` now destructures `profile` and reads
`(profile as { email_verified?: boolean | string } | undefined)?.email_verified`. The
prior code aliased `account as profile`, so `email_verified` was always `undefined` and
the rejection was **dead code**. Rejects only an *explicitly* unverified email; an omitted
claim (`undefined`) is admitted per OIDC. Cast added so `npx tsc --noEmit` stays green
(the Auth.js `Profile` type may not declare the claim).

**This is the criterion the immutable command cannot verify** (contract §1c): its
`'profile' within 250 chars of 'signIn('` assert ALREADY PASSED on the buggy pre-fix file.
Proven instead by the semantic test's load-bearing case -- see §3, M2.

### (d) gap2-06 -- emailless principal rejected under an active allowlist
`if (!user.email) return false;` inside the `allowedEmails.length > 0` block, replacing the
`&& user.email` short-circuit that waved a no-email principal through to `return true`.

### (e) gap2-05 -- session hardening
`maxAge: 7 * 24 * 60 * 60` (was 30 days), explicit `updateAge: 24 * 60 * 60`. No `30 * 24`
literal remains anywhere (including comments). The comment documents the JWT-revocation
limitation and the `strategy:"database"` Prisma follow-up (adapter already wired in
`auth.ts`). OWASP's 4-8h absolute-timeout target is noted as the real endpoint; 7d is an
improvement, not the destination.

---

## 3. Mutation matrix -- 6/6 killed, 0 survived

```
baseline: 9 passed

M2   KILLED (2 failed)   alias account as profile (the gap2-04 dead-code bug the string assert misses)
M3   KILLED (1 failed)   drop the `!user.email` reject
M4   KILLED (2 failed)   make empty-allowlist+flag return true instead of false
M5   KILLED (1 failed)   revert maxAge to 30 days
M6   KILLED (1 failed)   remove updateAge
M7   KILLED (1 failed)   reject when email_verified is undefined (breaks OIDC omitted-claim allowance)

6/6 killed; 0 survived
post-restore: 9 passed
```

**M2 is the load-bearing mutant.** It re-introduces the exact aliasing bug the immutable
verification command's string assert cannot detect (the assert passes on the bug). The
semantic test's *"reads email_verified from profile NOT account"* case -- `false` on
`account`, `true` on `profile`, expecting ADMIT -- fails on the bug and passes on the fix.
This is the anti-vacuous-guard discipline (harness_log Cycle 131 + new qa.md §4b) applied:
the string assert is not evidence, so the behaviour is proven directly.

Scoped claim: **these 6 mutations were killed.** Not "the suite has no vacuous guards".

---

## 4. Live evidence (criterion 6) + honesty

- **tsc**: exit 0 (test files are type-checked -- tsconfig includes `**/*.tsx`/`**/*.ts`).
- **Full frontend suite**: 24 files / 187 tests passed (was 178; +9 mine). 0 regressions.
- **Live enforcement on :3000**: Next dev hot-reloaded the new middleware onto the operator
  instance (`.next/server/middleware.js` mtime after the edit). Protected routes
  (`/`, `/paper-trading`, `/backtest`) -> `302 -> /login`; public paths (`/login`,
  `/api/auth/session`) -> `200`. Measured by curl, in the live_check.
- **Playwright capture**: `handoff/current/captures_75.6/login_75.6.png` (+ snapshot yml) --
  the login page renders with Google + Passkey buttons and "Access restricted to authorized
  users". Navigated to `http://localhost:3000/login` (public, read-only GET); **no second
  dev server started** (auto-memory `feedback_second_next_dev_breaks_operator_3000`);
  operator :3000 re-confirmed healthy (`/`->302, `/login`->200) after.
- **Console error disclosure**: 1 console error observed on `/login`; its content was not
  separately retrievable this session. It is not introduced by this change (auth-logic
  only; no client component touched) -- stated as observed, not asserted clean.

### The division of proof, stated plainly
- **Playwright + curl** prove the app still WORKS (renders, redirects) -- regression safety.
- **The semantic unit tests** prove the new fail-CLOSED logic (deny-on-empty+flag,
  emailless reject, profile-email_verified, membership). You cannot Playwright-test an
  allowlist rejection without a real Google login, so the behaviour is proven at the unit
  boundary and the app-integrity at the live boundary. Neither substitutes for the other.

### Lockout -- ruled out and PROVEN, not assumed
Research live-probed the operator config: the `302 -> /login` on `:3000/` is emitted only
past the old provider gate, so Google SSO is configured and working. Removing the gate is
inert for them. The only hard-lockout path is operator-initiated (`AUTH_ENFORCE_ALLOWLIST`
on + empty `ALLOWED_EMAILS`), behind a default-off flag the executor never sets.

### Not done / follow-ups
- Passkey users must have a stored email or the (d) reject would deny them under an active
  allowlist (research R4) -- true of the backend allowlist too; not a regression, but worth
  a note when the allowlist is first enabled.
- `strategy:"database"` migration for true JWT revocation -- documented follow-up.
- ALLOWED_EMAILS must be kept consistent across `frontend/.env.local` and `backend/.env`
  when the operator enables enforcement, or a split (frontend login OK, backend 401) occurs.

---

## 5. Handoff note
75.5 is parked at CERTIFIED_FALLBACK (retry budget exhausted; awaiting operator
adjudication). Its handoff record is preserved at `handoff/archive/phase-75.5/` (contract,
experiment_results, evaluator_critique with all 7 verdicts, live_check, research_brief) so
this step's rolling files do not clobber it. 75.5 is committed at `3a7942cf`.
