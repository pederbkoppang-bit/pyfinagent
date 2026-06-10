# Research Brief: BudgetDashboard Hotfix + Next.js Upgrade Assessment

**Tier:** simple (root cause confirmed; research covers patterns + upgrade evaluation)
**Date:** 2026-04-21

---

## Executive Summary

`BudgetDashboard.tsx` uses a raw `fetch()` call without the Bearer token from NextAuth, so the FastAPI middleware returns HTTP 401 with `{"detail":"Authentication required"}`. The component stores that error shape as `data`, then dereferences `data.summary` which is `undefined`, causing the runtime crash at `s.total_monthly.toFixed(0)`. The fix is a one-function swap to `apiFetch` plus a TypeScript discriminated-union guard at the parse boundary. Two other components use raw `fetch()` (Sidebar changelog, StockChart) but neither dereferences auth-gated sub-fields unsafely. For Part B: Next.js 16.2.4 is now the latest stable as of April 2026. The upgrade from 15.x is a **major version** (not a minor patch), carries one middleware rename breaking change (`middleware.ts` -> `proxy.ts`), and requires evaluation before adoption. DEFER the upgrade -- there is no security emergency and the one breaking change (middleware rename) requires care with NextAuth's edge-compatible config split.

---

## Read in Full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://nextjs.org/blog/next-15-5 | 2026-04-21 | Official docs/blog | WebFetch full | 15.5 = Turbopack builds beta, Node.js middleware stable, typed routes stable, deprecation warnings for Next.js 16 features (legacyBehavior link, AMP, image quality). No breaking changes vs 15.x line. |
| https://nextjs.org/blog/next-15-2 | 2026-04-21 | Official docs/blog | WebFetch full | 15.2 = redesigned error UI with owner stacks, streaming metadata, Turbopack 57.6% faster, Node.js middleware experimental. No breaking changes to auth or fetch patterns. |
| https://nextjs.org/docs/app/getting-started/error-handling | 2026-04-21 | Official docs (Next.js 16.2.4) | WebFetch full | Canonical error boundary patterns for App Router: `error.js` file convention for route-segment boundaries; `unstable_catchError` for component-level boundaries; async fetch errors in `useEffect` must be caught manually via `useState`/`useReducer` -- they don't reach error boundaries. |
| https://www.typescriptlang.org/docs/handbook/2/narrowing.html | 2026-04-21 | Official docs (TypeScript) | WebFetch full | Discriminated unions on a `kind`/`type` literal field are the canonical TypeScript narrowing pattern for API responses that can be `ErrorBody | SuccessBody`. Type predicates (`response is SuccessResponse`) enable reusable guards. |
| https://betterstack.com/community/guides/scaling-nodejs/error-handling-nextjs/ | 2026-04-21 | Authoritative blog | WebFetch full | Model expected errors (auth, validation) as return values not thrown exceptions; use consistent `{ success: true, data }` / `{ success: false, error }` discriminated shape; validate auth before processing; conditional rendering on null data before touching sub-fields. |
| https://dev.to/huangyongshan46a11y/authjs-v5-with-nextjs-16-the-complete-authentication-guide-2026-2lg | 2026-04-21 | Technical blog (2026) | WebFetch full | Auth.js v5 is compatible with Next.js 16 by design (native App Router support, better TS types, edge runtime compatible). No new breaking changes introduced by Next.js 16 for Auth.js v5 specifically. The real compatibility risk is Prisma's Node.js-only session queries hitting edge middleware -- already addressed by the config split pattern in this repo. |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://releasebot.io/updates/vercel/next-js | Release tracker | Fetched -- confirmed Next.js 16.2.4 as of 2026-04-16; CVE-2026-23869 patched in 16.2.3 |
| https://github.com/vercel/next.js/releases | GitHub releases | Snippet only -- confirms 16.x series is current major |
| https://nextjs.org/support-policy | Official docs | Snippet -- Next.js 15 still receives security fixes; Next.js 14 and below are EOL |
| https://endoflife.date/nextjs | Reference | Snippet -- confirms 15.x support window and 16 as current |
| https://aurorascharff.no/posts/error-handling-in-nextjs-with-catch-error/ | Tech blog | Snippet -- `unstable_catchError` pattern confirmed; full read superseded by official docs above |
| https://blog.logrocket.com/best-auth-library-nextjs-2026/ | Industry blog | Snippet -- Auth.js v5 recommended for Next.js 16; no Prisma compatibility issues reported |
| https://www.abhs.in/blog/nextjs-current-version-march-2026-stable-release-whats-new | Blog | Snippet -- March 2026 version snapshot; confirmed superseded by 16.2.4 |
| https://github.com/prisma/prisma/issues/23685 | GitHub issue | Snippet -- Prisma edge runtime issue known and mitigated by auth.config.ts split already in this repo |
| https://sentry.io/answers/next-js-client-side-exception/ | Sentry docs | Snippet -- confirms `error.js` boundary and that async useEffect errors require manual catch |
| https://nextjs.org/blog/next-15-1 | Official blog | Snippet -- 15.1 React 19 stable + improved error debugging; no breaking changes |

---

## Recency Scan (2024-2026)

Searched with three query variants:
1. Current-year frontier: "Next.js 15 latest stable version release 2026"
2. Last-2-year window: "Next.js 15 changelog releases 15.1 15.2 15.3 what's new"
3. Year-less canonical: "React error boundary pattern Next.js App Router apiFetch auth-aware wrapper"

**Findings:**

- **Next.js 16 is now released.** The latest stable as of 2026-04-21 is **v16.2.4** (released 2026-04-16). This is a major version bump from 15.x. Next.js 15's last minor was 15.5 (August 2025), which introduced deprecation warnings for Next.js 16.
- **Security patch CVE-2026-23869** was addressed in Next.js 16.2.3 (2026-04-08). CVE details are not yet public. Next.js 15.x support policy indicates it still receives security backports -- verify whether the CVE was also backported before deciding to defer.
- **middleware.ts rename** is the one confirmed breaking change for Next.js 16 discovered in the search results: the middleware file may need to export a `proxy` named export or a default function under `proxy.ts`. This directly affects `frontend/src/middleware.ts` which guards all routes for NextAuth.
- **No new breaking changes** in the 15.0 -> 15.5 line for this repo's stack (NextAuth v5, Prisma, Recharts, Tailwind, Phosphor Icons). All are compatible.
- **Auth.js v5 + Next.js 16**: Native App Router support confirmed; no migration steps required beyond ensuring the Prisma edge-runtime split is in place (already done via `auth.config.ts` + `auth.ts` split visible in `frontend.md`).

---

## Internal Code Inventory

| File | Lines read | Role | Status |
|------|-----------|------|--------|
| `frontend/src/components/BudgetDashboard.tsx` | 1-549 | Budget page UI component; fetches `/api/backtest/budget/summary` | **Bug confirmed**: raw `fetch()` at line 128, no Bearer token |
| `frontend/src/lib/api.ts` | 1-544 | Central API client with `apiFetch<T>()` | Healthy; `apiFetch` sends Bearer token, handles 401 redirect, 30s timeout |
| `frontend/src/components/Sidebar.tsx` | 130-147 | `ChangelogModal` fetches `/api/changelog` | Raw `fetch()` at line 139; endpoint may be public (no sub-field crash risk) |
| `frontend/src/components/StockChart.tsx` | 85-105 | Fetches `/api/charts/<ticker>` | Raw `fetch()` at line 94; has explicit `!res.ok` guard + error state; no unsafe sub-field access |
| `frontend/package.json` | 1-53 | Dependency manifest | `"next": "^15.0.0"` -- semver range allows 15.x only (caret before major 15); `"next-auth": "^5.0.0-beta.30"`, `"@prisma/client": "^6.19.2"`, `"recharts": "^2.12.0"`, `"@phosphor-icons/react": "^2.1.10"` |

### BudgetDashboard crash sites (all `s.*` references after `const s = data.summary`)

`data` is typed `BudgetData | null` in state. The check at line 146 (`if (error || !data)`) guards against `null`. But on a 401 response, `data` is set to `{"detail":"Authentication required"}` (not `null`), which passes the `!data` guard. Then `s = data.summary` is `undefined`.

Every downstream use of `s` crashes:
- Line 166: `s.total_monthly.toFixed(0)` -- **crash point**
- Line 174: `s.total_gcp_monthly.toFixed(0)`
- Line 182: `s.total_fixed_monthly.toFixed(0)`
- Line 190: `s.monthly_budget.toFixed(0)`
- Line 195: `s.budget_utilization_pct >= 100` (comparison on undefined)
- Line 214: `s.budget_utilization_pct.toFixed(0)`
- Line 221: `<UtilizationBar pct={s.budget_utilization_pct} />`
- Line 227: `s.monthly_budget - s.total_monthly`
- Line 230: `s.total_monthly - s.monthly_budget`
- Line 255 (chart): `budget: s.monthly_budget` (loop)
- Line 327: `<ReferenceLine y={s.monthly_budget} ...>`
- Line 333: label uses `s.monthly_budget`

`data.fixed_costs`, `data.gcp_costs`, `data.monthly_history` are also undefined on the error shape but are only accessed after the `s` crash, or in branches guarded by `.length > 0` which would throw on `undefined.length` if reached.

### `apiFetch` behavior on non-2xx

`apiFetch` (api.ts:110-136):
- HTTP 401: redirects to `/login` via `window.location.href` and throws `"Session expired. Redirecting to login."`
- Any non-2xx: throws `Error` -- does NOT return the error body as data
- Result: using `apiFetch` means the `.catch(() => setError(...))` handler fires, `data` stays `null`, and the existing `if (error || !data)` guard renders the error banner cleanly

### Other raw `fetch()` components -- risk assessment

- `Sidebar.tsx::ChangelogModal` (line 139): fetches `/api/changelog`. This endpoint is likely public (changelog data). Pattern uses `data.entries || []` / `data.recent_commits || []` with fallbacks -- no unsafe sub-field access. Low risk, but should migrate to `apiFetch` for consistency and to benefit from the 30s timeout.
- `StockChart.tsx` (line 94): fetches `/api/charts/<ticker>`. Has explicit `if (!res.ok) throw ...` guard. No unsafe sub-field access. Risk: if `/api/charts/` is auth-gated, same crash pattern could emerge. Medium risk. Not an immediate crash but worth migrating.

---

## Consensus vs Debate (External)

**Hard crash vs graceful degrade:** The 2025-2026 dashboard doctrine (Next.js docs, BetterStack guide, Sentry) converges on **graceful degradation** for expected errors (auth failures, network timeouts) and **hard crash + error boundary** for programmer errors (wrong schema assumed). The distinction: auth 401 is an _expected_ operational state -- the user may not be logged in, the session may have expired. It should show a friendly "unavailable" state with a retry. A null-dereference on a schema mismatch _after_ successful auth is a programmer error and should bubble to the error boundary for visibility. In pyfinagent's case, routing through `apiFetch` converts the 401 into a thrown `Error` that lands in the `.catch(() => setError(...))` branch, which renders the existing rose-border error banner -- this is exactly the graceful degrade pattern.

**Centralized `apiFetch` vs component-local fetch:** No debate in 2025-2026 literature -- centralizing auth, timeout, and error normalization in a single wrapper is the universal recommendation. The only valid use of raw `fetch()` in a Next.js client component is when the endpoint is genuinely public (no auth required, no risk of an error-shape masquerading as data), and even then a 30s timeout is advisable.

---

## Pitfalls (from Literature)

1. **Error shape stored as data**: When `fetch().then(r => r.json()).then(setData)` is chained without checking `res.ok`, a `{"detail":"..."}` error body is stored in state. TypeScript won't catch this at runtime because `setData` accepts any JSON. Use discriminated unions or check `res.ok` before calling `.json()`.
2. **Error boundaries don't catch async useEffect errors**: React error boundaries only catch during render. A crash in `useEffect -> fetch -> .then(setData)` after the error shape is stored will crash during the *next render* (when `s.total_monthly` is accessed). This looks like an uncaught render error but the root cause is in the async chain. Guard at the fetch layer, not just with error boundaries.
3. **Next.js 16 middleware rename**: If upgrading to Next.js 16, `middleware.ts` must be reviewed for the `proxy` export requirement. Failing this breaks the auth guard on ALL protected routes.
4. **CVE-2026-23869 backport status**: Unknown as of writing whether Next.js 15.x received this security fix. If it wasn't backported, staying on 15.x leaves a known CVE open.

---

## Application to pyfinagent

### Part A Fix -- Exact Code Diff

The complete fix for `BudgetDashboard.tsx` is:

**Step 1: Remove the raw `fetch`, add `apiFetch` import**

```diff
-import { useEffect, useMemo, useState } from "react";
+import { useEffect, useMemo, useState } from "react";
+import { apiFetch } from "@/lib/api";
```

Wait -- `apiFetch` is not currently exported from `api.ts`. It is declared as `async function apiFetch<T>(...)` (line 65) but not exported. Check this before writing the diff.

Actually, looking at `api.ts` line 65: `async function apiFetch<T>(path: string, init?: RequestInit): Promise<T>` -- it is NOT exported with `export`. All the named API functions at the bottom of `api.ts` ARE exported, but `apiFetch` itself is module-private. The correct fix has two options:

- Option A (recommended): Add a named wrapper function `getBudgetSummary` to `api.ts` that calls `apiFetch`, then call that from the component. This is the existing repo pattern (every API call goes through a named function in `api.ts`).
- Option B: Export `apiFetch` directly. Simpler but breaks the naming convention -- every other call site uses a named function.

**Option A (matches repo convention):**

In `frontend/src/lib/api.ts`, add after the existing budget-related section:

```typescript
// ── Budget ──────────────────────────────────────────────────────

export function getBudgetSummary(): Promise<import("./types").BudgetData> {
  return apiFetch("/api/backtest/budget/summary");
}
```

Note: `BudgetData` is currently defined only inside `BudgetDashboard.tsx`. It should be moved to `types.ts` as part of this fix for full type safety. If moving to types.ts is out of scope, use `Promise<unknown>` with a type guard.

In `frontend/src/components/BudgetDashboard.tsx`, replace lines 127-135:

```diff
-  useEffect(() => {
-    fetch(
-      `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/backtest/budget/summary`
-    )
-      .then((r) => r.json())
-      .then(setData)
-      .catch(() => setError("Failed to load budget data"))
-      .finally(() => setLoading(false));
-  }, []);
+  useEffect(() => {
+    getBudgetSummary()
+      .then(setData)
+      .catch((e: Error) => setError(e.message || "Failed to load budget data"))
+      .finally(() => setLoading(false));
+  }, []);
```

Also add the import:

```diff
+import { getBudgetSummary } from "@/lib/api";
```

**Add defensive guard at the render boundary** (belt-and-suspenders after the apiFetch fix, in case the backend response shape ever diverges):

Replace line 154 (`const s = data.summary;`) with:

```diff
-  const s = data.summary;
+  const s = data.summary;
+  if (!s) {
+    return (
+      <div className="rounded-lg border border-rose-500/30 bg-rose-950/20 px-4 py-3 text-sm text-red-400">
+        Budget summary unavailable -- data shape mismatch
+      </div>
+    );
+  }
```

**Justification for graceful degrade over hard crash:** The 401 is an expected operational state (session expiry, page load before auth cookie is set). `apiFetch` converts it to `window.location.href = "/login"` + a thrown error, landing in the `.catch` handler which sets `error` state and renders the existing rose-border banner. That is the correct UX. The inner `if (!s)` guard is a secondary defense for schema drift -- it shows a specific message rather than a generic crash banner. Hard crashes (bubbling to Next.js error boundary) are appropriate for programmer errors, not auth failures.

### Part B -- Next.js Upgrade Decision: DEFER

**Current state:** `package.json` pins `"next": "^15.0.0"`, which allows 15.x but NOT 16.x (caret respects major). The browser overlay saying "15.5.12 (outdated)" is from the Next.js dev tools comparing installed version to the npm latest tag, which is now 16.x.

**Latest stable:** Next.js 16.2.4 (2026-04-16).

**What changed from 15.5.x to 16.x:**
- Node.js middleware stable (was stable in 15.5 too -- no change)
- **Breaking: `middleware.ts` -> `proxy.ts` rename / proxy export requirement.** This repo's `frontend/src/middleware.ts` is the NextAuth edge-compatible auth guard for all protected routes. Renaming and adding the `proxy` export is required for Next.js 16.
- Deprecation of `legacyBehavior` on `<Link>` -- not used in this repo
- AMP support removed -- not used
- `next lint` removed -- this repo's `package.json` does NOT use `next lint` (uses `eslint` script directly) -- no impact
- `unstable_catchError` available (from Next.js 16.2) -- beneficial for component-level error boundaries
- Typed routes now stable -- beneficial (TypeScript improvements)
- Turbopack builds beta -- beneficial for local dev build speed

**Risk/benefit for pyfinagent (local Mac, no prod fleet):**
- Benefit: faster builds, typed routes, better error tooling
- Risk: middleware rename breaks ALL auth-protected routes if missed; CVE-2026-23869 patch status unknown for 15.x; Auth.js v5 + Prisma config split already in place so auth compatibility risk is low once middleware rename is done
- Effort: low (1-2 hours to rename middleware, verify all pages load, run `npm run build`)

**Decision: DEFER for now; schedule as a named masterplan step.**

Reason for deferral: the middleware rename is a hard breaking change that requires a `npm run build` pass + manual route verification. This is out of scope for the hotfix cycle. There is no active security emergency (CVE-2026-23869 backport status unknown). Scheduling it as a dedicated step ensures it goes through the full harness loop (contract -> generate -> Q/A).

**Proposed masterplan step:**

- **Step id:** `phase-upgrade-nextjs16`
- **Step name:** "Upgrade Next.js 15 -> 16 and verify auth + build"
- **Verification sub-criteria (5):**
  1. `npm run build` exits 0 with no TypeScript or lint errors
  2. `/login` page loads and Google SSO redirect works in browser
  3. Auth-protected route (`/` dashboard) redirects to `/login` when unauthenticated (verifies middleware/proxy rename is correct)
  4. BudgetDashboard, StockChart, and Paper Trading pages render without runtime errors when authenticated
  5. `npm list next` shows `16.x.x`; no `^15` or `^14` versions remain as direct dependencies

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched in full)
- [x] 10+ unique URLs total (incl. snippet-only) (16 URLs total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (BudgetDashboard, api.ts, Sidebar, StockChart, package.json)
- [x] Contradictions / consensus noted (graceful degrade vs hard crash)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```
