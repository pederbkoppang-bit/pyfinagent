# Research Brief -- Masterplan step 75.6

## Audit75 S6 -- frontend auth fail-closed + session hardening

Tier: **complex** (P0, auth-critical, operator-lockout risk). NOT audit-class.
Researcher: Layer-3 Harness MAS Researcher. Access date: 2026-07-23.
Executor target: opus-4.8/xhigh. BOUNDARY: no .env edits; allowlist enforcement
ships DARK behind `AUTH_ENFORCE_ALLOWLIST` default-off.

---

## 0. Step summary (what 75.6 wants)

Five sub-items on the Next.js-15/Auth.js-v5 auth layer, all behind default-off flags:

- (a) gap2-01 -- middleware `hasAuthProvider` gate disables auth for a passkey-only
  config; enforce `req.auth` unconditionally, keep only `LIGHTHOUSE_SKIP_AUTH` (+ opt-in `DEV_DISABLE_AUTH`).
- (b) gap2-02 -- empty `ALLOWED_EMAILS` admits ANY Google account; deny-all-on-empty
  behind `AUTH_ENFORCE_ALLOWLIST` (default off, loud warn), mirroring 75.1 backend flag.
- (c) gap2-04 -- `signIn` aliases `account` as `profile`; `email_verified` never checked.
- (d) gap2-06 -- with active allowlist a no-email principal must be REJECTED, not waved through.
- (e) gap2-05 -- session maxAge 30d, no updateAge; set 7d + updateAge, document JWT-revocation limit.

**Bottom line: all five CONFIRMED. One line-number correction (a). One verification
weakness flagged (c). No item can permanently lock the operator out at executor defaults.**

---

## 1. Source table

### Read in full via WebFetch (10; >=5 required; counts toward the gate)
| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 1 | https://authjs.dev/reference/nextjs | 2026-07-23 | official doc | `NextAuthRequest.auth: null \| Session`; middleware `auth()` wrapper; "Authentication is done by the `callbacks.authorized` callback". req.auth is the decrypted session or null. |
| 2 | https://authjs.dev/getting-started/session-management/protecting | 2026-07-23 | official doc | Canonical middleware pattern **enforces auth unconditionally** for non-public paths (`if (!req.auth && path!="/login") redirect`); no provider-presence gate. Caveat: "do not rely on the proxy exclusively... verify session close to data fetching". |
| 3 | https://authjs.dev/getting-started/providers/google | 2026-07-23 | official doc | **"Google also returns a `email_verified` boolean property in the OAuth profile."** Example: `if (account.provider==="google") return profile.email_verified && ...` -- read via **`profile`**, not `account`. DEFINITIVE for (c). |
| 4 | https://authjs.dev/concepts/session-strategies | 2026-07-23 | official doc | JWT: "Expiring a JSON Web Token before its encoded expiry is not possible -- doing so requires... a server-side blocklist." DB sessions "can be at any time modified server-side"; signout deletes the row. DEFINITIVE for (e). |
| 5 | https://authjs.dev/getting-started/migrating-to-v5 | 2026-07-23 | official doc | v5 middleware = `export { auth as proxy } from "@/auth"`; **`middleware.ts`->`proxy.ts` rename is Next.js 16+** (N/A on Next 15). Stricter OIDC compliance. No signIn breaking change. |
| 6 | https://next-auth.js.org/configuration/options | 2026-07-23 | official doc (v4) | `maxAge` = "how long until an idle session expires" (default 30d). `updateAge` = throttle for session refresh (default 24h). (v4 note "ignored if using JWT" is legacy -- see s5(e).) |
| 7 | https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html | 2026-07-23 | OWASP (canonical) | Idle timeout 2-5 min high-value / 15-30 min low-risk; **absolute timeout 4-8 h**. "The shorter the session interval is, the lesser the time an attacker has to use the valid session ID." |
| 8 | https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html | 2026-07-23 | OWASP (canonical) | Covers MFA/password/error-handling; deny-by-default is an ASVS/Access-Control principle, not explicit here (noted). |
| 9 | https://openid.net/specs/openid-connect-core-1_0.html | 2026-07-23 | OIDC spec (canonical) | `email_verified`: Boolean, part of the `email` scope claims (profile), OPTIONAL (providers MAY omit). |
| 10 | https://github.com/nextauthjs/next-auth/issues/12116 | 2026-07-23 | GH issue | Confirms the v5 signIn callback receives `{ user, account, profile, email, credentials }` (docs were wrong). |
| +| https://auth0.com/docs/get-started/apis/scopes/openid-connect-scopes | 2026-07-23 | vendor doc | Corroborates: `email_verified` "a boolean indicating whether the email address was verified", in the `email` scope. |

### Snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| github.com/.../discussions/8961 (middleware chaining) | GH | tangential |
| github.com/.../discussions/11550 (signIn extra values) | GH | tangential |
| github.com/.../issues/11164 (overwrite signIn v5) | GH | corroborated by #12116 |
| github.com/.../issues/12167 (migrating to v5) | GH | covered by migration doc |
| dev.to/peterlidee callbacks explained | blog | lower tier |
| clerk.com nextjs session management | vendor blog | JWT-revocation corroboration only |
| blog.logrocket.com best auth library nextjs 2026 | blog | recency context |
| authjs.dev/getting-started/adapters/prisma | doc | DB-session follow-up ref |
| rajeshrnair.com nextauth v5 2026 guide | blog | lower tier |
| medium/@kamilmatejuk, iamraghuveer, nextjslaunchpad | blogs | lower tier setup guides |

**URLs collected: ~28 unique** (10 fetched + ~18 snippet).

---

## 2. Queries run (three-variant discipline)
- **Current-year / 2025 frontier:** "Auth.js NextAuth v5 breaking changes 2025 middleware signIn callback migration profile parameter".
- **Last-2-year window:** "next-auth jwt session strategy revocation limitation database session Prisma adapter" (2025-2026 results incl. logrocket-2026, clerk).
- **Year-less canonical:** OWASP Session_Management + Authentication cheat sheets; OIDC Core spec; authjs.dev `/concepts/session-strategies`, `/providers/google`; "NextAuth v5 signIn callback parameters user account profile email email_verified".

---

## 3. Recency scan (last 2 years, 2024-2026)
Searched v5 changes affecting middleware + signIn. Findings:
- **`middleware.ts` -> `proxy.ts` rename is Next.js 16+ ONLY.** pyfinagent is **Next.js 15** (CLAUDE.md stack), so the file **stays `middleware.ts`** and the verification command reads that path. **Executor MUST NOT rename it.**
- v5 uses `export { auth } from "@/auth"` wrapper (already the project's pattern, middleware.ts:1-9). No change needed to the import shape.
- v5 has **stricter OAuth/OIDC spec-compliance** than v4 -- reinforces that `email_verified` arrives as a proper profile claim (the (c) fix aligns with v5's stricter profile handling).
- A widely-copied blog claims "the session callback is no longer invoked" in v5 -- **IMPRECISE**: the project's `session` callback (auth.config.ts:60-65) demonstrably runs (populates `session.user.id`). Do not act on that blog claim.
- No new finding supersedes OWASP/OIDC canonical guidance; both still current.

---

## 4. Internal evidence table (verbatim quotes, file:line)

### frontend/src/middleware.ts (37 lines total)
| Line | Verbatim | Note |
|------|----------|------|
| 7 | `const hasAuthProvider = !!(process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET);` | **DEFINITION** of the gate. Step says ":24" -- **line correction: :24 is the USE site, :7 is the definition.** |
| 13-18 | public paths: `startsWith("/api/auth")` OR `=== "/login"` OR `startsWith("/_next")` OR `startsWith("/favicon")` -> `return` | public allowlist |
| 22-26 | `// Skip auth redirect if no provider is configured (dev mode)...` then `if (!hasAuthProvider \|\| process.env.LIGHTHOUSE_SKIP_AUTH === "1") { return; }` | **the gate USE (:24)**. `!hasAuthProvider` short-circuits auth entirely. |
| 29-32 | `if (!req.auth) { const loginUrl = new URL("/login", req.url); return Response.redirect(loginUrl); }` | the REAL enforcement -- only reached when hasAuthProvider is TRUE |
| 36 | `matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"]` | runs on all non-static routes |

Passkey-only / no-Google trace: `hasAuthProvider=false` -> :24 `!hasAuthProvider` true -> `return` -> **every route open**. CONFIRMED (a).

### frontend/src/lib/auth.config.ts (71 lines total)
| Line | Verbatim | Note |
|------|----------|------|
| 10-13 | `const allowedEmails = (process.env.ALLOWED_EMAILS \|\| "").split(",").map(...).filter(Boolean);` | parse. Step ":10" **CORRECT**. |
| 30-33 | `session: { strategy: "jwt", maxAge: 30 * 24 * 60 * 60, // 30 days }` | **no updateAge**. `30 * 24` present. CONFIRMED (e). |
| 39 | `async signIn({ user, account }) {` | destructures `{user, account}` only -- **no `profile`**. CONFIRMED (c). |
| 41 | `const profile = account as Record<string, unknown>;` | **aliases `account` as `profile`** -- the bug. |
| 42 | `if (!profile.email_verified && profile.email_verified !== undefined) { return false; }` | `account.email_verified` is always undefined -> `!== undefined` guard always false -> **check never fires**. CONFIRMED (c). Guard LOGIC is fine; the SOURCE object is wrong. |
| 46 | `if (allowedEmails.length > 0 && user.email) {` | `&& user.email` short-circuits: no-email principal skips the block -> `return true`. CONFIRMED (d). |
| 47-49 | `if (!allowedEmails.includes(user.email.toLowerCase())) { return false; }` | membership check |
| 60-68 | `session()` sets `session.user.id`; `authorized({auth}) { return !!auth?.user; }` | consumed by middleware `req.auth` |

### frontend/src/lib/auth.ts (24 lines) -- FULL (node) config
- `NextAuth({ ...authConfig, providers: [...(authConfig.providers||[]), Passkey], adapter: PrismaAdapter(prisma), experimental: { enableWebAuthn: true } })` (:11-23).
- **Prisma adapter IS wired** -> the (e) DB-session follow-up is a config change (`session.strategy:"database"`), not new infra. Sessions today are stateless JWTs (`strategy:"jwt"`); the adapter stores WebAuthn credentials + account links, not sessions.

### Backend 75.1 mirror -- SHIPPED (commit 40111b03, masterplan 75.1 status=done)
| Location | Verbatim | Semantics |
|----------|----------|-----------|
| settings.py:571 | `allowed_emails: str = Field("", ...)` | env `ALLOWED_EMAILS` |
| settings.py:574 | `auth_enforce_allowlist: bool = Field(False, "...True makes an EMPTY allowed_emails reject ALL authenticated users (fail-closed). Default False preserves fail-open legacy...")` | env `AUTH_ENFORCE_ALLOWLIST`, default False |
| api/auth.py:213-219 | `allowed=[...split(",") if e.strip()]`; `if not allowed and settings.auth_enforce_allowlist: raise 401`; `if allowed and email.lower() not in allowed: raise 401` | empty+flag=deny-all; empty+noflag=admit; non-empty=enforce membership (empty-email naturally rejected: `"" not in allowed`) |
| main.py:121-141 | `_warn_if_allowlist_empty` logs WARNING either way | loud startup warn |

**Backend enforces on EVERY API request; frontend `signIn` gates only NEW logins.**

---

## 5. Per-item verdicts (a)-(e)

- **(a) gap2-01 -- CONFIRMED (with line correction).** The gate is at `middleware.ts:7` (def) / `:24` (use), not `:24` for the definition. A no-Google config opens all routes. **Fix is correct:** JWT session validation needs NO provider -- `req.auth` is derived from the JWE session cookie (needs only `AUTH_SECRET`); providers matter only at LOGIN. Enforce `req.auth` unconditionally, keep `LIGHTHOUSE_SKIP_AUTH`, add an explicit `DEV_DISABLE_AUTH`-style opt-in for a deliberate dev-open mode. **Remove BOTH `hasAuthProvider` references (:7 and :24)** so the verification `'hasAuthProvider' not in mw` passes.

- **(b) gap2-02 -- CONFIRMED.** Empty `ALLOWED_EMAILS` + no flag admits any Google account (auth.config.ts:46 gate only fires when `length>0`). Fix: read `process.env.AUTH_ENFORCE_ALLOWLIST === "true"`; deny-all when flag ON + empty list; loud `console.warn` when empty (either mode). Mirror the backend exactly (s7). Default off = byte-identical to today.

- **(c) gap2-04 -- CONFIRMED (authoritatively).** `email_verified` is a **boolean profile claim** (Auth.js Google doc + OIDC spec + Auth0). Current code reads it off `account` (never present) so the unverified-email rejection is dead. Fix: `async signIn({ user, account, profile })` and read `profile?.email_verified`. **KEEP the `!== undefined` guard** -- OIDC allows providers to omit the claim (reject only on explicit `false`). **tsc note:** the `Profile` type may not declare `email_verified`; cast to keep `npx tsc --noEmit` green, e.g. `const p = profile as { email_verified?: boolean } | undefined`.

- **(d) gap2-06 -- CONFIRMED.** `&& user.email` (auth.config.ts:46) waves through a no-email principal when an allowlist is active. Fix: inside the enforcement block, `if (!user.email) return false;` (satisfies the `'!user.email'` assert) before the membership check. Mirrors the backend's natural `"" not in allowed` rejection.

- **(e) gap2-05 -- CONFIRMED.** `maxAge: 30*24*60*60`, no `updateAge`. Fix: `maxAge: 7 * 24 * 60 * 60`, add explicit `updateAge` (e.g. `24 * 60 * 60`). Document: JWT sessions cannot be revoked before expiry (blocklist required); the Prisma-adapter DB-session strategy (`session.strategy:"database"`) is the tracked follow-up for true revocation. **Ensure NO `30 * 24` remains anywhere (incl. comments)** -- the assert `'30 * 24' absent` fails otherwise. OWASP context: even 7d far exceeds the 4-8h absolute-timeout guidance for a privileged cockpit, so 7d is a real improvement but the DB-session/idle-timeout follow-up is where OWASP alignment lands.

---

## 6. v5 signIn callback signature confirmation (for c)
Auth.js v5 `signIn` callback receives **`{ user, account, profile, email, credentials }`** (GH #12116; migration guide shows no signIn breaking change). `email_verified` is a **boolean on `profile`** (Auth.js Google provider doc: "Google also returns a `email_verified` boolean property in the OAuth profile"), NOT on `account` (which holds OAuth token/provider data). The (c) fix -- destructure `profile`, read `profile?.email_verified` -- is CORRECT against the real v5 signature.

## 7. 75.1 backend-mirror check (for b)
75.1 SHIPPED (done). Backend flag is **`auth_enforce_allowlist`** (env `AUTH_ENFORCE_ALLOWLIST`), default **False**. Semantics the frontend must mirror EXACTLY:
- empty list + flag OFF -> **admit** (fail-open legacy) + loud WARNING.
- empty list + flag ON -> **reject all** (fail-closed).
- non-empty list -> enforce membership (empty-email rejected).
Frontend parity: same env names, `=== "true"` bool parse, deny-all-on-empty when ON, loud `console.warn` when empty. The two layers then fail closed together only when the operator flips BOTH .env files.

---

## 8. LOCKOUT ANALYSIS (P0)

**Operator's live config PROVEN:** `curl :3000/ -> HTTP 302 -> /login` (and `/login -> 200`).
That 302 is emitted ONLY at middleware.ts:29-32, which is reachable ONLY when the :24 early-return did NOT fire, i.e. **`hasAuthProvider === true` -> `AUTH_GOOGLE_ID` + `AUTH_GOOGLE_SECRET` ARE configured**. The operator runs Google SSO and it works. (`.env.local` is permission-blocked from this session, so this live probe is the ground truth.)

| Item | Can it lock the operator out at executor defaults? | Reasoning |
|------|---|---|
| (a) enforce req.auth unconditionally | **NO** | hasAuthProvider is already TRUE, so the :24 early-return already does NOT fire for the operator. Removing `!hasAuthProvider` leaves live behavior IDENTICAL (unauth->302 /login; valid JWT->pass). LIGHTHOUSE_SKIP_AUTH kept. **Highest-risk item, but proven inert for the live config.** The only config that would break is a no-provider/passkey-only dev setup (operator is NOT running it) -- the mandatory `DEV_DISABLE_AUTH` opt-in is the safety valve for that path. |
| (b) AUTH_ENFORCE_ALLOWLIST | **NO** (default off) | Flag defaults OFF -> today's admit behavior byte-identical. Lockout only if operator later flips it ON with an EMPTY ALLOWED_EMAILS -- a deliberate, documented operator action (no .env edits by executor). |
| (c) email_verified from profile | **NO** | Only rejects when Google returns `email_verified===false`; the operator's real Google account is verified (true). |
| (d) reject no-email | **NO** | Only active with a non-empty allowlist/flag; the operator's Google login always yields an email. |
| (e) maxAge 30d->7d + updateAge | **NO (permanent)**; at most ONE re-login | JWT rolling session: on next request the existing 30d cookie is decoded (still valid) and re-issued with exp=now+7d -> silent roll-down, no logout. Worst case (if old cookie's iat > new maxAge and Auth.js recomputes from iat) = a single forced re-login, trivially recoverable because Google SSO works. |

**VERDICT: NONE of (a)-(e) at executor defaults (`AUTH_ENFORCE_ALLOWLIST` off, `DEV_DISABLE_AUTH` unset) can PERMANENTLY lock the operator out.** Any forced re-auth is recoverable via the working Google login. The single hard-lockout path is operator-initiated: `AUTH_ENFORCE_ALLOWLIST=on` + empty `ALLOWED_EMAILS` (the intended fail-closed switch, mirrored on the backend, out of scope for the executor).

**Cross-layer nuance:** for an ALREADY-logged-in operator, (b)/(c)/(d) don't re-evaluate until next login (signIn is login-time). The backend (75.1) re-checks the allowlist on EVERY API call -- so an operator-flipped empty+ON flag would 401 API calls immediately even while the page shell renders. That is intended fail-closed behavior.

---

## 9. Consumer list (what could break)

- **middleware.ts `req.auth`** (via `authorized` callback) -- the only middleware consumer of auth state. Unchanged by the fix except the gate removal.
- **Sidebar.tsx:248,364-376** -- `useSession()` reads `session.user.image/name/email`. The stricter `signIn` **drops NO session fields** (jwt/session callbacks untouched); it only gates whether a login SUCCEEDS. No session-shape change -> Sidebar unaffected as long as login succeeds.
- **Backend `get_current_user`** (api/auth.py) decrypts the SAME JWE cookie and checks its OWN allowlist. Frontend and backend allowlists are independent env reads -- **operator must keep `ALLOWED_EMAILS` consistent across `frontend/.env.local` and `backend/.env`** or a user allowed by one but not the other gets split behavior (frontend login OK, backend 401). Doc this.
- **Passkey/WebAuthn path** (auth.ts:13, `enableWebAuthn`): passkey logins ALSO traverse `signIn`. `account.provider` = "passkey" -> the Google `email_verified` branch is skipped; the allowlist branch still applies to the passkey user's linked email (stored via PrismaAdapter at registration). The (d) no-email reject is the correct tightening here. Passkey-only is exactly the config (a) protects -- but the operator has Google.
- **Tests:** NO existing middleware/auth.config tests (grep clean). Runner = vitest 4.1.4 (`npm test` -> `node scripts/run-test.mjs`). The immutable verification is string-asserts + `npx tsc --noEmit` (no unit test required). A behavioral vitest test for the signIn matrix (empty+flag deny, no-email reject, email_verified=false reject) is RECOMMENDED but not gate-required; if added it must actually be able to fail (per the mutation-test-guard doctrine).

---

## 10. Risks

1. **VERIFICATION WEAKNESS (HIGH -- flag for Q/A):** the assert `'profile' in ac.split('signIn(')[1][:250]` is **mutation-weak** -- the CURRENT unfixed code already has `const profile = account` ~79 chars after `signIn(`, so `'profile'` is present in that window TODAY, before any fix. The string assert passes on a no-op. Q/A MUST verify (c) SEMANTICALLY: `profile` destructured from the callback params AND `email_verified` read from `profile` (not `account`). Do not treat the assert as proof of the fix.
2. **tsc failure risk (c):** `Profile` type may not declare `email_verified`; a naive `profile.email_verified` can fail `npx tsc --noEmit`. Cast defensively.
3. **`30 * 24` leftover (e):** any residual `30 * 24` (even in a comment) fails the `'30 * 24' absent` assert. Replace, don't annotate.
4. **`hasAuthProvider` residue (a):** remove BOTH the :7 definition and the :24 use; keep `LIGHTHOUSE_SKIP_AUTH` (asserted present). Do NOT rename `middleware.ts` (Next 15 -> stays middleware.ts; proxy.ts is Next 16+).
5. **Flag-parity drift (b):** frontend must read the SAME `AUTH_ENFORCE_ALLOWLIST` name + same empty-list-fail-open default as the backend; a mismatched bool-parse ("1" vs "true") could desync layers. Match the documented `=true` form.
6. **Login-time vs request-time gap:** frontend signIn changes only bite NEW logins; the backend is the enforcement point for existing sessions. Don't claim the frontend change alone revokes a live session.
7. **DEV_DISABLE_AUTH must be default-off and explicit** -- if the executor makes it default-on or infers it, it reintroduces the (a) hole.

---

## 11. JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 18,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All five items (a)-(e) CONFIRMED against current files with verbatim file:line. (a) line-correction: hasAuthProvider is defined at middleware.ts:7, used at :24. (c) email_verified is a boolean PROFILE claim (Auth.js Google doc + OIDC + Auth0) -- current code reads it off account (always undefined) so the check is dead; fix = destructure profile. 75.1 backend flag auth_enforce_allowlist SHIPPED (default False); frontend must mirror env names + empty-fail-open default exactly. LOCKOUT: operator proven to run Google SSO (live :3000/ -> 302 /login), so NONE of (a)-(e) at executor defaults can permanently lock them out; (a) is inert for the live config, (e) at worst forces one recoverable re-login. HIGH-VALUE Q/A flag: the 'profile in signIn(...)[:250]' assert is mutation-weak -- passes on the unfixed code -- so (c) must be verified semantically. tsc risk on Profile.email_verified (cast). Do not rename middleware.ts (Next 15).",
  "brief_path": "handoff/current/research_brief_75.6.md",
  "gate_passed": true
}
```
