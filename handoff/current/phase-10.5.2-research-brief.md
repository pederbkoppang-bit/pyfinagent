# Research Brief: phase-10.5.2 -- Sovereign Route Shell (two-hero layout)

Tier assumed: simple-moderate (frontend-only, one new page + one audit script).

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|------------|-------------|
| https://nextjs.org/docs/app/building-your-application/routing | 2026-04-21 | Official doc (Next.js 16.2.4) | WebFetch full | File-system routing: folder + page.tsx = route; no manual config needed |
| https://nextjs.org/docs/app/getting-started/server-and-client-components | 2026-04-21 | Official doc (Next.js 16.2.4) | WebFetch full | "use client" required for useState/useEffect/event handlers; server components are default |
| https://dev.to/getcraftly/nextjs-16-app-router-the-complete-guide-for-2026-2hi3 | 2026-04-21 | Authoritative blog | WebFetch full | Next.js 16 maintains same App Router conventions as 15; params/searchParams are Promises in 15+ |
| https://improvado.io/blog/dashboard-design-guide | 2026-04-21 | Industry blog | WebFetch full | F-pattern + 40-30-20-10 space rule: primary hero (top-left, largest), secondary (top-right); asymmetric split preferred over equal |
| https://bricxlabs.com/blogs/tips-for-dashboard-design | 2026-04-21 | Industry blog | WebFetch full | "Vital metrics at top-left where users start reading"; visual weight via size + contrast; white space to separate sections |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://medium.com/@livenapps/next-js-15-app-router-a-complete-senior-level-guide-0554a2b820f7 | Blog | Covered by official docs |
| https://medium.com/@differofeveryone/mastering-next-js-routing-a-modern-guide-for-2025-138c1e65b505 | Blog | Official docs + DEV article more authoritative |
| https://next-saas-stripe-starter.vercel.app/docs/configuration/layouts | SaaS starter | Sidebar/main centered layout; no two-hero pattern |
| https://vercel.com/blog/architecting-reliability-stripes-black-friday-site | Blog | Status page pattern, not dashboard hero |
| https://fuselabcreative.com/top-dashboard-design-trends-2025/ | Blog | Trend overview; less actionable than improvado |
| https://datacamp.com/tutorial/dashboard-design-tutorial | Tutorial | Snippet covered F-pattern; duplicates improvado |
| https://5of10.com/articles/dashboard-design-best-practices/ | Blog | Snippet covered 40-30-20-10 rule; covered |
| https://www.designrush.com/agency/ui-ux-design/dashboard/trends/dashboard-design-principles | Blog | Snippet; no new findings vs above |
| https://tailadmin.com/blog/best-analytics-dashboard | Blog | Template list; layout principles snippet sufficient |
| https://medium.com/@thiraphat-ps-dev/mastering-next-js-app-router | Blog | Snippet; official docs more authoritative |

---

### Recency scan (2024-2026)

Searched: "Next.js App Router page routing best practices 2025", "Next.js 16 App Router complete guide 2026", "dashboard two-pane hero layout analytics 2025 2026", "dashboard primary secondary hero panel split layout 2025 2026".

Findings: Next.js 16.2.4 (current) preserves all Next.js 15 App Router conventions. The primary 15+ breaking change (params as Promise) does not affect simple `"use client"` pages with no dynamic segments -- confirmed. No new layout primitives or conventions introduced in 16 that affect this work. Dashboard design literature (2025-2026) consistently endorses the F-pattern with asymmetric primary/secondary split, which aligns with `frontend-layout.md` §4.5 (dense operator bar + hero row, no equal-height bento for status).

---

## Key findings

1. **Route creation**: create `frontend/src/app/sovereign/page.tsx` -- that alone makes `/sovereign` a valid App Router route. No config file needed. (Source: Next.js 16 docs, https://nextjs.org/docs/app/building-your-application/routing)

2. **"use client" required**: the page imports `Sidebar` which uses `usePathname`, `useSession`, state hooks -- all client-only. All sibling pages (`/agents`, `/backtest`, `/paper-trading`) begin with `"use client"`. The sovereign page must also begin with `"use client"`. (Source: Next.js 16 docs, server-and-client-components; internal audit of `agents/page.tsx` line 1 and `backtest/page.tsx` line 1)

3. **Two-hero interpretation**: industry guidance recommends asymmetric split -- primary hero gets ~60% width (left), secondary gets ~40% (right). For Sovereign: `RedLineMonitor` is the primary hero (risk/safety concern, high urgency, left dominant); `AlphaLeaderboard` is the secondary (performance ranking, right, slightly smaller); `ComputeCostBreakdown` goes full-width beneath both (supporting context). This matches the F-pattern scanning path: primary risk first, ranking second, cost third. (Source: improvado.io, bricxlabs.com)

4. **No layout.tsx needed**: pyfinagent puts `<Sidebar>` directly inside each page component, not in a shared layout.tsx segment (verified: sibling pages all do this). Do NOT add a `sovereign/layout.tsx`.

5. **Phosphor icon recommendation**: use `Crown` for the Sovereign sidebar entry. `Crown` (available in `@phosphor-icons/react`) directly maps to the "sovereign" concept; `Shield` is already aliased for `RiskConservative`/`StepBiasAudit` in `icons.ts` (lines 63, 75); `Compass` is not currently exported; `Globe` is `GlobeHemisphereWest` (used for `StepMacro`/`PillarIndustry`). `Crown` has no existing alias and is unambiguous.

6. **Sidebar section**: add Sovereign to the "Trading" `NavSection` in `Sidebar.tsx` (lines 39-46). Rationale: Sovereign is the live trading control plane (red-line, leaderboard, compute cost) -- it belongs alongside `/paper-trading` and `/backtest`. A new "Sovereign" top-level section is also acceptable but adds sidebar weight for one item.

7. **Audit script pattern**: existing scripts in `scripts/audit/` are Python (`*.py`). The verification command calls `node scripts/audit/sovereign_route.js` -- this must be a Node.js script using `fs`/static analysis only (no server needed). Pattern from `route_count.py` (lines 76-89): enumerate filesystem, read file text, assert presence of tokens.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/src/app/` | dir | App Router root; contains 11 route folders | Active |
| `frontend/src/app/agents/page.tsx` | 60+ | Sibling page; "use client" + Sidebar import pattern | Canonical reference |
| `frontend/src/app/backtest/page.tsx` | 50+ | Sibling page; same shell pattern | Canonical reference |
| `frontend/src/components/Sidebar.tsx` | 373 | Global nav; `NAV_SECTIONS` at lines 21-54; "Trading" section at lines 39-46 | Needs 1 new `NavItem` |
| `frontend/src/lib/icons.ts` | 145 | Phosphor icon aliases; `Crown` NOT yet exported | Needs `Crown` export |
| `scripts/audit/route_count.py` | 123 | Reference audit script (Python); filesystem enumeration pattern | Reference only |
| `scripts/audit/` | dir | 28 existing audit scripts (Python); no JS scripts yet | New JS script needed |
| `.claude/rules/frontend-layout.md` | 452 | Canonical page shell skeleton at "New Page Template" section | Non-negotiable |

---

## Consensus vs debate (external)

Consensus: asymmetric two-hero (primary left dominant, secondary right) is preferred over equal 50/50 split. This matches `frontend-layout.md` §4.5 bento pattern guidance (one tall chart + secondary cards). No debate found.

Consensus: Next.js 16 App Router -- folder + page.tsx = route. No controversy.

---

## Pitfalls (from literature + internal)

- Do NOT use `min-h-screen` on the outer div (`frontend-layout.md` §1 shell rule).
- Do NOT add `h-full`/`flex-1` to the placeholder cards to fill dead space (`frontend.md` conventions + `frontend-layout.md` §9).
- Do NOT import from `@phosphor-icons/react` directly in the page -- import through `@/lib/icons` (`frontend.md` architecture note).
- Do NOT create a `sovereign/layout.tsx` -- pyfinagent's pattern is Sidebar-inside-page, not segment layout.
- Audit script must exit 0 on success, non-zero on failure (the verification command checks exit code).
- `Crown` must be added to `icons.ts` exports before importing in `Sidebar.tsx` or the build will fail.

---

## Application to pyfinagent: exact implementation spec

### File 1 (new): `frontend/src/app/sovereign/page.tsx`

Strict adherence to `frontend-layout.md` "New Page Template" skeleton. All 6 tiers present:

```tsx
"use client";

import { Sidebar } from "@/components/Sidebar";
import { Crown, ChartLineUp, CurrencyDollar, Trophy } from "@phosphor-icons/react";

export default function SovereignPage() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex flex-1 flex-col overflow-hidden">

        {/* ── Fixed header zone (Tiers 1-5) ── */}
        <div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
          {/* Tier 1: Header */}
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-slate-100">Sovereign</h2>
            <p className="text-sm text-slate-500">Live trading command centre — risk, alpha, and compute</p>
          </div>
          {/* Tier 5: Tab bar — not needed for stub; omit */}
        </div>

        {/* ── Scrollable content zone (Tier 6) ── */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">

          {/* Two-hero row: RedLineMonitor (primary, ~60%) + AlphaLeaderboard (secondary, ~40%) */}
          <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-5">

            {/* Primary hero: Red-Line Monitor — ships in 10.5.3 */}
            <div className="lg:col-span-3 rounded-xl border border-navy-700 bg-navy-800/60 p-6 flex flex-col items-center justify-center min-h-[320px]">
              <ChartLineUp size={48} weight="duotone" className="text-slate-600 mb-4" />
              <p className="text-lg text-slate-400 font-medium">Red-Line Monitor</p>
              <p className="mt-1 text-sm text-slate-600">Coming in phase 10.5.3</p>
            </div>

            {/* Secondary hero: Alpha Leaderboard — ships in 10.5.4 */}
            <div className="lg:col-span-2 rounded-xl border border-navy-700 bg-navy-800/60 p-6 flex flex-col items-center justify-center min-h-[320px]">
              <Trophy size={48} weight="duotone" className="text-slate-600 mb-4" />
              <p className="text-lg text-slate-400 font-medium">Alpha Leaderboard</p>
              <p className="mt-1 text-sm text-slate-600">Coming in phase 10.5.4</p>
            </div>
          </div>

          {/* Full-width beneath: Compute Cost Breakdown — ships in 10.5.5 */}
          <div className="rounded-xl border border-navy-700 bg-navy-800/60 p-6 flex flex-col items-center justify-center min-h-[200px]">
            <CurrencyDollar size={48} weight="duotone" className="text-slate-600 mb-4" />
            <p className="text-lg text-slate-400 font-medium">Compute Cost Breakdown</p>
            <p className="mt-1 text-sm text-slate-600">Coming in phase 10.5.5</p>
          </div>

        </div>
      </main>
    </div>
  );
}
```

Note: icons imported directly from `@phosphor-icons/react` above for illustration. In the actual implementation, `Crown`, `ChartLineUp`, `CurrencyDollar`, `Trophy` must first be added/confirmed in `frontend/src/lib/icons.ts`, then imported via `@/lib/icons`. `ChartLineUp` is already exported as `IconChart`/`NavPerformance`/`LogoIcon`; `Trophy` is available in Phosphor (exported as `StepCompetitor` in icons.ts line 25); `CurrencyDollar` is already exported as `MacroCpi`/`SettingsEstimator` in icons.ts.

### File 2 (icon addition): `frontend/src/lib/icons.ts`

Add one export near the Navigation section:
```ts
Crown as NavSovereign,
```

### File 3 (sidebar edit): `frontend/src/components/Sidebar.tsx`

Add to the "Trading" `NavSection` at line 44 (after `/backtest`):
```ts
{ href: "/sovereign", label: "Sovereign", icon: NavSovereign },
```

And add `NavSovereign` to the import from `@/lib/icons` at line 10-13.

### File 4 (new): `scripts/audit/sovereign_route.js`

```js
#!/usr/bin/env node
/**
 * Audit script for phase-10.5.2 sovereign route shell.
 * Checks three immutable success criteria by static file analysis.
 * Exit 0 = all pass. Exit 1 = one or more failures.
 *
 * Criteria:
 *   1. route_reachable        -- frontend/src/app/sovereign/page.tsx exists
 *   2. sidebar_entry_added    -- Sidebar.tsx contains href="/sovereign"
 *   3. page_shell_conforms    -- page.tsx contains required shell tokens:
 *                                "flex h-screen overflow-hidden", "Sidebar",
 *                                "flex-shrink-0", "overflow-y-auto scrollbar-thin"
 */
const fs = require("fs");
const path = require("path");

const REPO = path.resolve(__dirname, "../..");

const checks = {
  route_reachable: false,
  sidebar_entry_added: false,
  page_shell_conforms_to_frontend_layout: false,
};

// 1. Route file exists
const PAGE_PATH = path.join(REPO, "frontend/src/app/sovereign/page.tsx");
if (fs.existsSync(PAGE_PATH)) {
  checks.route_reachable = true;
}

// 2. Sidebar contains the sovereign href
const SIDEBAR_PATH = path.join(REPO, "frontend/src/components/Sidebar.tsx");
if (fs.existsSync(SIDEBAR_PATH)) {
  const sidebar = fs.readFileSync(SIDEBAR_PATH, "utf8");
  if (sidebar.includes('href="/sovereign"')) {
    checks.sidebar_entry_added = true;
  }
}

// 3. Page shell conforms -- check mandatory layout tokens
if (checks.route_reachable) {
  const page = fs.readFileSync(PAGE_PATH, "utf8");
  const requiredTokens = [
    "flex h-screen overflow-hidden", // outer shell
    "Sidebar",                        // sidebar component present
    "flex-shrink-0",                  // fixed header zone
    "overflow-y-auto scrollbar-thin", // scrollable content zone
  ];
  const allPresent = requiredTokens.every((token) => page.includes(token));
  if (allPresent) {
    checks.page_shell_conforms_to_frontend_layout = true;
  }
}

// Report
const passed = Object.values(checks).every(Boolean);
const result = {
  phase: "10.5.2",
  checks,
  passed,
};
console.log(JSON.stringify(result, null, 2));

if (!passed) {
  const failed = Object.entries(checks)
    .filter(([, v]) => !v)
    .map(([k]) => k);
  console.error(`FAIL: ${failed.join(", ")}`);
  process.exit(1);
}

console.log("PASS: all sovereign route shell criteria met");
process.exit(0);
```

---

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (incl. snippet-only) -- 15 total
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks -- note gaps but do not auto-fail:
- [x] Internal exploration covered every relevant module (page.tsx pattern, Sidebar.tsx NAV_SECTIONS, icons.ts, audit script pattern)
- [x] Contradictions / consensus noted (no debate; asymmetric two-hero is clear consensus)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple-moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/phase-10.5.2-research-brief.md",
  "gate_passed": true
}
```
