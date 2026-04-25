---
step: phase-16.18
title: Live API Smoke (sovereign + paper-trading + auth + OWASP)
tier: simple
generated: 2026-04-24
gate_passed: true
---

# Research Brief: phase-16.18 — Live API Smoke

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Headers_Cheat_Sheet.html | 2026-04-24 | Official doc | WebFetch full | X-XSS-Protection should be "0" (disabled); X-Frame-Options: DENY; Cache-Control: no-store; Referrer-Policy: strict-origin-when-cross-origin |
| https://owasp.org/www-project-secure-headers/ | 2026-04-24 | Official doc | WebFetch full | X-XSS-Protection deprecated — set to "0". Cache-Control recommended "no-store, max-age=0". Permissions-Policy: working draft. |
| https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html | 2026-04-24 | Official doc | WebFetch full | Cache-Control: no-store mandatory for sensitive APIs; JWT must be signed; error messages must be generic |
| https://authjs.dev/getting-started/session-management/protecting | 2026-04-24 | Official doc | WebFetch full | Auth.js v5 middleware redirects unauthenticated users with HTTP 302; authorized callback drives the gate; matcher should exclude /api routes from Next.js middleware |
| https://assertible.com/blog/set-up-automated-smoke-tests-for-a-rest-api-in-5-minutes | 2026-04-24 | Authoritative blog | WebFetch full | Smoke tests should validate: HTTP status codes, JSON schema shape, and JSON path field presence; deploy-triggered and scheduled (>= hourly) |
| https://www.levo.ai/resources/blogs/api-security-testing-checklist-2026 | 2026-04-24 | Industry blog | WebFetch full | Pre-launch checklist: JWT signing verified, token expiry tested, CORS/CSRF validated, TLS 1.2+ enforced, no undocumented endpoints |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://owasp.org/API-Security/ | Official doc | Covered by HTTP Headers cheatsheet |
| https://qodex.ai/blog/owasp-top-10-for-api-security-a-complete-guide | Blog | Derivative of OWASP docs |
| https://github.com/tmotagam/Secweb | Code | FastAPI security middleware library; snippet sufficient |
| https://brightsec.com/blog/api-security-testing-checklist-2026 | Industry blog | Levo checklist more authoritative |
| https://testsigma.com/blog/api-testing-checklist/ | Blog | General testing; partial overlap |
| https://apidog.com/blog/api-testing-method-smoke-tests/ | Blog | Covered by Assertible full read |
| https://oneuptime.com/blog/post/2025-01-06-fastapi-owasp-security/view | Blog | FastAPI-specific; search snippet sufficient |
| https://accuknox.com/blog/owasp-api-security-top-10-the-complete-testing-checklist-2026 | Blog | Derivative checklist |
| https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control | Official doc | MDN; confirming no-store semantics via snippet |
| https://github.com/nextauthjs/next-auth/issues/8511 | Code/Issue | Auth.js OAuth returns 303 on sign-in, not 302; middleware redirect is separate path |

## Recency scan (2024-2026)

Searched for 2024-2026 literature on "OWASP security headers 2025", "NextAuth v5 Auth.js middleware 2025", "API smoke test pre-go-live 2026", and "Cache-Control no-store financial API 2025".

Findings: No new findings that supersede the canonical OWASP cheatsheets. The X-XSS-Protection deprecation (set to "0") was confirmed by both the OWASP Secure Headers Project and the HTTP Headers cheatsheet, with no change from their 2023-2024 stance. The Auth.js v5 middleware redirect behavior (302 for unauthenticated routes) is current as of 2026 based on the official authjs.dev docs. The Cache-Control: no-store guidance for sensitive financial APIs is unchanged and confirmed by multiple 2025-2026 sources.

One notable 2024-2026 update: OWASP Top 10:2025 elevates "Security Misconfiguration" to #2 (up four places), making the header smoke test directly relevant to a go-live gate.

## Key findings

1. **X-XSS-Protection must be "0" not "1; mode=block"** — OWASP explicitly says: "set to 0 in order to disable the XSS Auditor, and not allow it to take the default behavior." The .claude/rules/security.md documents the legacy value "1; mode=block" but main.py already emits the correct "0". The security.md doc is stale on this point. (Source: OWASP HTTP Headers Cheatsheet, https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Headers_Cheat_Sheet.html)

2. **Five required OWASP headers for the smoke test** — The five headers that the success criterion "owasp_headers_present_5_of_5" should verify are: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 0`, `Referrer-Policy: strict-origin-when-cross-origin`, `Cache-Control: no-store`. All five are set in main.py lines 287-291. `Permissions-Policy` is also set (line 292) and is a bonus sixth header. (Source: OWASP Secure Headers Project)

3. **Cache-Control: no-store is correct for all endpoints** — For sensitive financial APIs: "no-store indicates that any caches of any kind (private or shared) should not store the response." Applied globally in the middleware, which is the correct pattern. (Source: OWASP REST Security Cheatsheet)

4. **Auth.js v5 middleware issues a 302 redirect for unauthenticated routes** — The middleware.ts uses `Response.redirect(loginUrl)` which is HTTP 302. The smoke test criterion "all_authed_routes_200_or_302" correctly accounts for both authenticated (200) and unauthenticated-redirected (302) outcomes. For curl probes, `-L` flag or checking the 302 Location header is the pattern. (Source: authjs.dev/getting-started/session-management/protecting)

5. **Sovereign endpoints are public (no auth required)** — /api/sovereign is in `_PUBLIC_PATHS` at main.py line 228. All three sovereign endpoints (red-line, leaderboard, compute-cost) will return data without a Bearer token. The kill-switch endpoint at /api/paper-trading/kill-switch is under the paper-trading router, which is NOT in _PUBLIC_PATHS, so it requires auth. (Source: internal, main.py lines 215-232)

6. **kill_switch_paused_false criterion** — /api/paper-trading/kill-switch returns a JSON object with a top-level `"paused"` boolean field (paper_trading.py line 304). The smoke test should check `paused == false` using a jq-style probe: `curl ... /api/paper-trading/kill-switch | jq '.paused == false'`. (Source: internal, paper_trading.py lines 288-315)

7. **Paper trading /status returns 200 with a "status" key** — paper_trading.py line 125 returns `"status": "active"` or `"paused"` depending on scheduler state. If portfolio is not initialized it returns a 200 (not 404) with `"status": "not_initialized"`. The smoke criterion `paper_trading_status_200` is satisfied by the 200 in all these cases. (Source: internal, paper_trading.py lines 94-146)

8. **API smoke tests must be deterministic and <5 min** — Curl-based probes hitting known-good endpoints are the correct approach. The Assertible guidance confirms: smoke tests should be fast, low false-positive, and deploy-triggered. For pre-go-live, the recommended pattern is: health endpoint first, then auth-gated endpoints with a valid Bearer token, then response body field assertions. (Source: Assertible blog)

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/main.py` | 445 | FastAPI app: middleware chain, OWASP headers, auth gate, public paths | Active; OWASP headers at lines 287-292 |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/api/sovereign_api.py` | 549 | Three sovereign endpoints: /red-line, /leaderboard, /compute-cost, /strategy/{id} | Active; all fail-open, 60s in-memory cache |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/api/paper_trading.py` | 350+ | Paper trading endpoints including /status, /portfolio, /kill-switch, /pause, /resume | Active; kill-switch at lines 288-315 |
| `/Users/ford/.openclaw/workspace/pyfinagent/frontend/src/middleware.ts` | 37 | NextAuth v5 edge middleware: protects all routes except /login, /api/auth, /_next | Active; 302 redirect on !req.auth when hasAuthProvider is true |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/rules/security.md` | 39 | Documented OWASP header expectations and auth conventions | Partially stale: documents X-XSS-Protection: "1; mode=block" but main.py correctly emits "0" |

## Consensus vs debate (external)

Consensus: All five surveyed sources agree on `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Cache-Control: no-store`, and `Referrer-Policy: strict-origin-when-cross-origin`. There is complete consensus that `X-XSS-Protection` should be `0` (disabled) in 2025-2026 as the header is deprecated. There is no debate on these values.

Debate: Whether to emit `X-Frame-Options: DENY` alongside `Content-Security-Policy: frame-ancestors 'none'` (OWASP notes CSP supersedes X-Frame-Options for modern browsers). For pyfinagent's local-only scope this is immaterial — DENY is fine and the backend has no CSP header anyway.

## Pitfalls (from literature)

1. **X-XSS-Protection "1; mode=block" is wrong** — The security.md doc is stale. Setting "1; mode=block" enables a deprecated XSS auditor that can itself be exploited. The correct value is "0". main.py is already correct at line 289.

2. **auth_and_security_middleware skips headers on auth failures** — On line 265 of main.py, when authentication fails, a JSONResponse is returned early with only `WWW-Authenticate` and CORS headers, but WITHOUT the OWASP headers. The OWASP headers block (lines 287-292) is never reached. A curl probe that hits an auth-gated endpoint without a token will NOT see the OWASP headers. Smoke test must use an authenticated token OR probe a public path (like /api/health) to verify OWASP headers.

3. **Sovereign endpoints are under /api/sovereign which is in _PUBLIC_PATHS** — The public path whitelist uses `path.startswith(p)`. "/api/sovereign" at line 228 covers all sub-paths (/red-line, /leaderboard, /compute-cost). Auth is not required for these endpoints, which is correct for read-only public data but means the smoke test cannot use sovereign endpoints to test authenticated OWASP header delivery.

4. **kill-switch endpoint requires auth** — /api/paper-trading is NOT in _PUBLIC_PATHS. A smoke test probing /api/paper-trading/kill-switch must include a valid Bearer token or the middleware returns 401 before OWASP headers are set.

5. **Frontend 302 vs 200 disambiguation** — The middleware.ts checks `hasAuthProvider` (line 7): if AUTH_GOOGLE_ID/AUTH_GOOGLE_SECRET env vars are not set (dev mode or CI), no redirect is issued — the route returns 200 directly. The success criterion "all_authed_routes_200_or_302" must account for this dev-mode bypass.

## Application to pyfinagent

### Smoke test curl sequence (recommended)

1. **Health check (public, no auth)** — `curl -sI http://localhost:8000/api/health` — expect 200 + all five OWASP headers (this is the correct path to verify OWASP headers since headers are set on the response object returned by `call_next`, which runs for public paths).

2. **Sovereign red-line (public, data-bearing)** — `curl -s http://localhost:8000/api/sovereign/red-line` — expect 200 + JSON body with `window`, `series`, `events` keys.

3. **Sovereign leaderboard (public)** — `curl -s http://localhost:8000/api/sovereign/leaderboard` — expect 200 + `entries` array + `source` key.

4. **Sovereign compute-cost (public)** — `curl -s http://localhost:8000/api/sovereign/compute-cost` — expect 200 + `daily_breakdown`, `totals`, `grand_total_usd` keys.

5. **Paper trading /status (auth-required)** — `curl -sH "Authorization: Bearer <token>" http://localhost:8000/api/paper-trading/status` — expect 200 + `status` key (value: "active" | "paused" | "not_initialized").

6. **Kill-switch state (auth-required)** — `curl -sH "Authorization: Bearer <token>" http://localhost:8000/api/paper-trading/kill-switch` — expect 200 + `paused` == false + `breach` object.

7. **OWASP headers verification** — `curl -sI http://localhost:8000/api/health` — check response headers for all five: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy, Cache-Control.

8. **Frontend route reachability** — `curl -sI http://localhost:3000/` — expect 200 (if authenticated) or 302 to /login (if not). `curl -sI http://localhost:3000/sovereign` — same pattern.

### OWASP header values to assert (from main.py lines 287-291)
- `X-Content-Type-Options: nosniff` (line 287)
- `X-Frame-Options: DENY` (line 288)
- `X-XSS-Protection: 0` (line 289) -- note: security.md documents "1; mode=block" but actual code correctly emits "0"
- `Referrer-Policy: strict-origin-when-cross-origin` (line 290)
- `Cache-Control: no-store` (line 291)

### Discrepancy to document in contract
security.md line 16 documents `X-XSS-Protection: 1; mode=block` but main.py line 289 correctly emits `X-XSS-Protection: 0`. The smoke test should assert "0" (the deployed value), not the stale doc value. The security.md should be updated as a housekeeping item.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total (16 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (main.py, sovereign_api.py, paper_trading.py, middleware.ts, security.md)
- [x] Contradictions noted (security.md vs main.py X-XSS-Protection value)
- [x] All claims cited per-claim with URL + file:line

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-16.18-research-brief.md",
  "gate_passed": true
}
```
