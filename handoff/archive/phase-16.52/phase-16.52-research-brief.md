# Research Brief: phase-16.52 — UX Audit Pass C: Settings Deep-Dive + Tab-Pattern Regression

**Tier:** simple (internal-only, per established pure-UI cycle precedent: 16.43, 16.46–16.49)
**Authority:** `.claude/rules/frontend.md` + `.claude/rules/frontend-layout.md`

---

## Internal Sources Read in Full

| File | Lines | Role | Status |
|------|-------|------|--------|
| `.claude/rules/frontend.md` | 48 | Frontend conventions authority | Read in full |
| `.claude/rules/frontend-layout.md` | 496 | Layout blueprint authority | Read in full |
| `frontend/src/app/settings/page.tsx` | 1242 | Canonical reference page | Read in full |
| `frontend/src/components/ReportTabs.tsx` | 57 | Shared tab component | Read in full |
| `frontend/src/app/reports/page.tsx` | 604 | Multi-tab page (16.49 fixed) | Read in full |
| `frontend/src/app/sovereign/page.tsx` | 167 | No-tab page | Read in full |
| `frontend/src/app/backtest/page.tsx` | lines 630–780 | Multi-tab page (16.49 fixed) | Read critical sections |
| Shell greps: login, signals, performance, home, agents, paper-trading | — | 16.48 / 16.49 drift check | grep output reviewed |

---

## 1. Canonical Patterns the Settings Page Demonstrates

The settings page (`frontend/src/app/settings/page.tsx`) is the canonical reference. However, it does **NOT** use the standard two-zone shell (`flex-shrink-0` header zone + `flex-1 overflow-y-auto` scrollable zone). Instead it uses a **single-zone layout**:

```tsx
// settings/page.tsx line 537
<main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">
  {/* Header, tab bar, AND tab content all in one scrollable zone */}
</main>
```

This is the **old pattern** — everything scrolls together including the header and tab bar. The new canonical pattern (established in phases 16.48/16.49) is the two-zone shell where the header and tab bar are pinned in a `flex-shrink-0` zone.

### What settings DOES demonstrate correctly

| Pattern | Location | Classes |
|---------|----------|---------|
| Outer shell | L535 | `flex h-screen overflow-hidden` |
| scrollbar-thin on scrollable zone | L537 | `overflow-y-auto scrollbar-thin` |
| Page header with action buttons | L538–565 | `flex items-center justify-between` |
| Tab bar pill pattern | L568–582 | `flex gap-1 rounded-lg bg-slate-800/50 p-1` |
| BentoCard groupings | L586–L773 | `BentoCard` component throughout |
| Tab content rendering | L585, L778, L962 | `{activeTab === "..." && (...)}`|
| Active tab: `bg-slate-700 text-slate-100 shadow-sm` | L574 | (vs canonical `bg-sky-500/10 text-sky-400`) |

### Key deviation: Settings uses a different active-tab color

`settings/page.tsx` L574: `bg-slate-700 text-slate-100 shadow-sm`
`frontend-layout.md` §5 canonical: `bg-sky-500/10 text-sky-400`
`reports/page.tsx` L238: `bg-sky-500/10 text-sky-400` (matches canonical)
`backtest/page.tsx` L751: `bg-sky-500/10 text-sky-400` (matches canonical)

Settings deviates from its own siblings on active-tab color.

---

## 2. Violations Table

### settings/page.tsx — Canonical Reference (REGRESSION FOUND)

| # | File:Line | Violation | Severity |
|---|-----------|-----------|----------|
| S-1 | `settings/page.tsx` L537 | **STRUCTURAL**: Main uses single-zone layout (`flex-1 overflow-y-auto ... p-6`) — both tab bar and header scroll with content. Does NOT use the two-zone `flex-shrink-0` + `flex-1` split that every other page uses after 16.48/16.49. | **SEVERE** — Tab bar scrolls off-screen on long content |
| S-2 | `settings/page.tsx` L511–531 | Loading state uses same single-zone layout (`flex-1 overflow-y-auto ... p-8`) instead of two-zone shell | MODERATE |
| S-3 | `settings/page.tsx` L574 | Active tab: `bg-slate-700 text-slate-100 shadow-sm` — deviates from canonical `bg-sky-500/10 text-sky-400` used by reports and backtest | COSMETIC |
| S-4 | `settings/page.tsx` L47–51 | SETTINGS_TABS definition has NO icon field — diverges from canonical tab definition (`{ id, label, icon }`) defined in `frontend-layout.md` §5 | MODERATE — inconsistent pattern |
| S-5 | `settings/page.tsx` L568 | Tab bar has `max-w-fit` constraint — not in any other tab bar. Harmless visually but non-standard | COSMETIC |

### reports/page.tsx — 16.49 FIXED PAGE

| # | File:Line | Violation | Severity |
|---|-----------|-----------|----------|
| R-1 | None found | Two-zone shell correct (L222–250): `flex h-screen overflow-hidden` > `main flex flex-1 flex-col overflow-hidden` > `flex-shrink-0` header > `flex-1 overflow-y-auto scrollbar-thin` content | PASS |
| R-2 | None found | Tab bar in fixed-header zone (L233–248), canonical `bg-sky-500/10 text-sky-400` active color | PASS |
| R-3 | `reports/page.tsx` L233 | Tab bar background uses `bg-navy-800/60` (the navy variant) while settings uses `bg-slate-800/50`. Minor inconsistency between siblings, neither violates the rule | COSMETIC |

### sovereign/page.tsx — No tabs, but shell check

| # | File:Line | Violation | Severity |
|---|-----------|-----------|----------|
| Sv-1 | None found | Two-zone shell correct (L114–166): `flex h-screen overflow-hidden` > `flex flex-1 flex-col overflow-hidden` > `flex-shrink-0` header > `flex-1 overflow-y-auto scrollbar-thin` content | PASS |
| Sv-2 | None found | No tab bar — correct, sovereign has no tabs | N/A |

### 16.48 Pages (login, signals, performance, home) — Drift Check

| # | File:Line | Status |
|---|-----------|--------|
| login/page.tsx | L35: `flex h-screen items-center justify-center overflow-hidden` | PASS — login is a centered form, no sidebar needed |
| signals/page.tsx | L88–102: Two-zone shell correct | PASS |
| performance/page.tsx | L41–64: Two-zone shell correct | PASS |
| home/page.tsx | L160–180: Two-zone shell correct | PASS |

No drift found on 16.48 pages.

### 16.49 Pages (backtest, reports, agents, paper-trading) — Drift Check

| # | File:Line | Status |
|---|-----------|--------|
| reports/page.tsx | Two-zone shell correct (see above) | PASS |
| agents/page.tsx | L289–368: Two-zone shell correct | PASS |
| paper-trading/page.tsx | L396–482: Two-zone shell correct | PASS |
| backtest/page.tsx | L630–765: Two-zone shell correct | PASS (with one note below) |

**Backtest note:** The `ingestResult` banner (L687–704) is rendered inside the `flex-shrink-0` fixed-header zone, inside the tab bar's parent `div`. Per `frontend-layout.md` §3, status banners (Tier 2) belong in the **scrollable content zone**, not the fixed header. However, a comment at L683–684 explicitly notes this was intentionally kept in the header (`phase-16.49: error banners moved to scrollable zone` note mentions the *error* banner was moved, but the ingest result banner was not). This is a minor Tier-placement deviation.

| # | File:Line | Violation | Severity |
|---|-----------|-----------|----------|
| B-1 | `backtest/page.tsx` L687–704 | `ingestResult` banner rendered in `flex-shrink-0` fixed-header zone instead of scrollable zone (Tier 2 placement rule) | MINOR |

---

## 3. Tab Pattern Consistency Audit

| Page | Tab implementation | Tab bar zone | Active color | Icons in tabs | Canonical? |
|------|-------------------|-------------|-------------|---------------|-----------|
| `settings/page.tsx` | Inline, no icons in SETTINGS_TABS | WRONG: inside single `overflow-y-auto` zone — scrolls | `bg-slate-700 text-slate-100` | No (tabs defined without icons) | NO |
| `reports/page.tsx` | Inline, full canonical pattern | CORRECT: `flex-shrink-0` zone | `bg-sky-500/10 text-sky-400` | Yes | YES |
| `backtest/page.tsx` | Inline, full canonical pattern | CORRECT: `flex-shrink-0` zone | `bg-sky-500/10 text-sky-400` | Yes | YES |
| `sovereign/page.tsx` | No tabs | N/A | N/A | N/A | YES |

**Shared tab component `ReportTabs.tsx`:** Exists at `frontend/src/components/ReportTabs.tsx`. Uses `bg-sky-500/15 text-sky-400` for active (slightly different opacity from the `bg-sky-500/10` used inline). The component is **not used by any of the main page routes** — it appears to be a sub-component used inside a report viewer context. Each page implements tabs inline rather than using this shared component.

**Conclusion on shared component:** There is NO single shared tab component used across all multi-tab pages. Reports, backtest, and settings all implement tabs inline. `ReportTabs.tsx` exists but is not the canonical cross-page component. This is an inconsistency but not a visual bug.

---

## 4. Decisive Findings Summary

**Total violations: 7**
- **Severe (visual layout broken): 1** — S-1: settings page uses single-zone layout; tab bar scrolls off-screen
- **Moderate: 2** — S-2 (settings loading state), S-4 (SETTINGS_TABS missing icon field)
- **Minor: 1** — B-1 (backtest ingest banner in wrong tier zone)
- **Cosmetic: 3** — S-3, S-5, R-3 (active-tab color inconsistency, max-w-fit, bg-navy vs bg-slate)

**Is there a shared tab component all pages should use?** `ReportTabs.tsx` exists but is not currently used by any page route. Pages implement tabs inline. The inline pattern is acceptable per `frontend-layout.md` §5 (it provides the exact inline code pattern to copy). No migration to a shared component is required, but the component should either be adopted universally or its active-tab color (`bg-sky-500/15`) harmonized with the inline pattern (`bg-sky-500/10`).

---

## 5. Proposed Fixes Table

| # | File:Line | Current | Target | Priority |
|---|-----------|---------|--------|---------|
| F-1 | `settings/page.tsx` L534–L1241 | Single-zone: `<main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">` contains header + tabs + content | Split into two-zone: `flex-shrink-0` zone with header + tab bar; `flex-1 overflow-y-auto scrollbar-thin` zone with tab content | HIGH |
| F-2 | `settings/page.tsx` L511–531 | Loading state also uses single-zone layout | Apply same two-zone split to the loading-state early return | MEDIUM |
| F-3 | `settings/page.tsx` L574 | `bg-slate-700 text-slate-100 shadow-sm` | `bg-sky-500/10 text-sky-400` to match canonical and siblings | LOW |
| F-4 | `settings/page.tsx` L47–51 | `SETTINGS_TABS` has no icon field | Add icon field per `frontend-layout.md` §5 tab definition pattern | LOW |
| F-5 | `backtest/page.tsx` L687–704 | `ingestResult` banner in fixed-header zone | Move to scrollable content zone (after `flex-1 overflow-y-auto`) | LOW |

---

## Research Gate Checklist

Internal-only tier per 16.43/16.46/16.47/16.48/16.49 precedent.

Hard blockers:
- [x] Project authority files read in full (frontend.md + frontend-layout.md)
- [x] All target page files read in full or via grep
- [x] file:line anchors for every claim
- [x] Canonical pattern documented with line-level evidence

Soft checks:
- [x] All 9 routes checked (login, home, signals, performance, backtest, reports, agents, paper-trading, settings + sovereign)
- [x] Tab-pattern consistency audited across all multi-tab pages
- [x] No external sources needed — project rules are authoritative for this UI audit

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 10,
  "gate_passed": true,
  "note": "Internal-only gate per established pure-UI cycle precedent (16.43, 16.46, 16.47, 16.48, 16.49). The project's own rules files (frontend.md, frontend-layout.md) are the sole authority for this layout audit. External literature does not add signal."
}
```
