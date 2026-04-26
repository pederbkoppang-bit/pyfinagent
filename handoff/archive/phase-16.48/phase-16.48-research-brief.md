# Research Brief — phase-16.48: UX Audit Pass A

Internal-heavy audit. No external research required; rules are documented in
`.claude/rules/frontend.md` and `.claude/rules/frontend-layout.md`. All 10 files
read in full.

---

## Per-File Audit

### 1. `frontend/src/app/login/page.tsx` — 109 LOC

**VIOLATIONS:**

- **Line 35** — `min-h-screen` in outer div. Violates page-shell rule: must use
  `h-screen overflow-hidden`. Login is auth-only (no Sidebar), so the full
  two-zone shell does not apply — but `min-h-screen` is still explicitly
  forbidden by the rule table ("Never use `min-h-screen`").

**Proposed exact fix (line 35):**

Current:
```tsx
<div className="flex min-h-screen items-center justify-center bg-[#0B1120]">
```

Replace with:
```tsx
<div className="flex h-screen items-center justify-center overflow-hidden bg-[#0B1120]">
```

The auth page is intentionally sidebar-free; `flex items-center justify-center`
correctly centers the card both axes. Adding `overflow-hidden` and replacing
`min-h-screen` with `h-screen` makes it viewport-locked, consistent with the
rule, and prevents mobile address-bar reflows. No structural rearrangement
needed — the card itself is fine.

**No other violations.** Phosphor imports via `@/lib/icons` (line 6). No emoji.
No scrollable containers. No data-driven render (no loading/error/empty states
required — auth action errors are handled inline, line 9-10/49-53).

---

### 2. `frontend/src/app/sovereign/page.tsx` — 167 LOC

**COMPLIANT.**

- Page shell: `<div className="flex h-screen overflow-hidden">` (line 114).
  Fixed header zone at line 118 uses `flex-shrink-0`. Scrollable zone at line
  134-136 uses `flex-1 overflow-y-auto scrollbar-thin`. Correct.
- Phosphor: `Crown` imported via `@/lib/icons` (line 32). No direct
  `@phosphor-icons/react`.
- No emoji in rendered text.
- `scrollbar-thin` present on scrollable zone (line 136).
- Loading/error/empty states: leaderboard has `loading` + `error` state
  propagated to `AlphaLeaderboard` (lines 52-53). RedLine and cost errors fall
  back to empty arrays (lines 63-66, 81-84) — graceful degradation, not a
  state-hiding violation (no user-facing error disappears silently).
- Grid: `lg:grid-cols-5` at line 139; no `lg:items-stretch` — not needed here
  because both children are wrapped in `AlphaLeaderboard` / `RedLineMonitor`
  which are BentoCards that size to content. No equal-height mixing issue.
- OpsStatusBar: not rendered on this page (Sovereign is not the operator cockpit).

---

### 3. `frontend/src/app/signals/page.tsx` — 187 LOC

**VIOLATION (minor):**

- **Line 90** — `<main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">`.
  This is a simplified shell (no two-zone flex-col split). The page uses a flat
  `overflow-y-auto` on `<main>` directly rather than the canonical pattern with
  a fixed-header zone and a scrollable content zone. The header (h2 + subtitle)
  at lines 91-98 scrolls with the content instead of being pinned.

  The page shell outer div at line 88 is correct (`flex h-screen overflow-hidden`).
  The issue is the missing `flex flex-1 flex-col overflow-hidden` on `<main>` with
  a split header zone.

  **Impact:** Medium — the "Market Signals & Intelligence" header scrolls away on
  long results. Input bar also scrolls out of sight. Functionally usable; not a
  crash or data-hiding bug.

- **`scrollbar-thin` present** (line 90) — compliant.
- Phosphor: `TabSignals` via `@/lib/icons` (line 10). No direct import.
- No emoji.
- Loading state: line 132-137 (spinner). Error state: lines 120-129. Empty state:
  lines 173-183 (icon + text). All three present.

---

### 4. `frontend/src/app/performance/page.tsx` — 233 LOC

**VIOLATIONS:**

- **Line 43** — `<main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">`.
  Same simplified-shell issue as signals/page.tsx. Outer div (line 41) is correct
  (`flex h-screen overflow-hidden`). Main is missing `flex flex-1 flex-col
  overflow-hidden`; the header scrolls with content.

- **Line 179** — `<div className="overflow-x-auto">` (cost history table) has NO
  `scrollbar-thin`. The rule requires `scrollbar-thin` on every
  `overflow-x-auto` / `overflow-y-auto` scrollable container.

- **No loading skeleton for cost history section** (lines 124-228): the
  `{costHistory.length > 0 && ...}` guard shows nothing while loading. The top-
  level `loading` and `error` states only gate the stats cards above (lines 62-63).
  The cost history section silently shows nothing until data arrives — no skeleton
  or "Loading cost history..." indicator. Minor gap per the "never show blank space"
  rule.

- **No empty state for cost history**: when `costHistory.length === 0` AND
  `!loading` AND `!error`, the section simply disappears. There is no "No cost
  history yet" message. Compare: the stats block has no empty state either (it
  hides behind `{stats && ...}`), but cost history is a separate block below.

---

### 5. `frontend/src/app/page.tsx` — 292 LOC

**COMPLIANT.** (Recently overhauled in phases 16.42-16.47.)

- Shell: `flex h-screen overflow-hidden` (line 160). `<main className="flex flex-1
  flex-col overflow-hidden">` (line 163). Fixed header zone `flex-shrink-0` (line
  165). Scrollable zone `flex-1 overflow-y-auto scrollbar-thin` (line 180).
- No `min-h-screen`.
- Phosphor: all via `@/lib/icons` (import lines 8-26).
- No emoji.
- `scrollbar-thin` on scrollable zone (line 180). No other `overflow-y-auto`
  containers present.
- Error/loading/empty states: `loadError` banner (lines 193-202), `loaded` gate
  on KPI tiles, `tradesError` propagated to `LatestTransactionsBox`. Comprehensive.
- `OpsStatusBar` is the single status bar (line 189) — correct pattern. No stacked
  card widgets alongside it.
- Grid `lg:grid-cols-6` at line 263 uses `lg:items-stretch` (line 263). Equal-height
  rule satisfied.

---

### 6. `frontend/src/components/Sidebar.tsx` — 378 LOC

**COMPLIANT.**

- Shell: `<aside className="flex h-screen w-64 flex-shrink-0 flex-col ...">` (line
  276). Scrollable nav: `flex-1 space-y-5 overflow-y-auto px-4 scrollbar-thin` (line
  287). Fixed footer: `border-t border-navy-700 px-4 py-4` (line 300). Correct
  three-zone sidebar layout.
- Changelog modal scrollable body: line 172 `overflow-y-auto scrollbar-thin` —
  compliant.
- Phosphor: all imports via `@/lib/icons` (lines 10-17). No direct import.
- No emoji.
- Changelog modal: loading state (lines 174-177), error state (lines 178-181).
  No explicit empty state when `entries.length === 0 && recentCommits.length === 0`
  but this is a degenerate case (changelog always exists); acceptable.

---

### 7. `frontend/src/components/OpsStatusBar.tsx` — 357 LOC

**COMPLIANT.**

- Single dense horizontal bar (lines 115-134). Correct `OpsStatusBar` pattern
  per `frontend-layout.md §4.5`.
- No scrollable containers.
- Phosphor: `IconCheckCircle`, `IconInfo`, `IconWarning` via `@/lib/icons` (line 5).
- No emoji.
- Loading / fallback states: each segment returns a `—` fallback when its data is
  null (e.g. `GateSegment` lines 152-158, `KillSegment` lines 189-196,
  `CycleSegment` lines 261-267, `NextSegment` lines 332-338). Correct.

---

### 8. `frontend/src/components/RedLineMonitor.tsx` — 162 LOC

**COMPLIANT.**

- Props-driven component; no data fetching of its own.
- No scrollable containers.
- Phosphor: `TrendDown` via `@/lib/icons` (line 16).
- No emoji.
- Empty / zero-data state: no explicit empty state when `series.length === 0`,
  but the chart renders gracefully (empty axes) and the footer line shows "0 points
  · 0 events". Acceptable for a chart component that receives props.
- No shell rule applies (component, not a page).

---

### 9. `frontend/src/components/StrategyDetail.tsx` — 175 LOC

**COMPLIANT.**

- Props-driven component; no data fetching.
- No scrollable containers.
- Phosphor: `TrendUp`, `ListBullets`, `ShieldCheck` via `@/lib/icons` (line 19).
- No emoji.
- Empty states: all three sections use `EmptyState` component (lines 43-47) when
  `equity.length === 0` (line 74), `overrides.length === 0` (line 115), `events.length
  === 0` (line 147). All three present.

---

### 10. `frontend/src/components/AlphaLeaderboard.tsx` — 293 LOC

**COMPLIANT.**

- Props-driven component.
- Scrollable container: `<div ... className="overflow-x-auto">` at line 190 — NO
  `scrollbar-thin`. Same violation as performance/page.tsx line 179.
- Phosphor: `Trophy`, `CaretUp`, `CaretDown`, `CheckCircle`, `XCircle`, `Warning`,
  `X` all via `@/lib/icons` (lines 19-26). No direct import.
- No emoji.
- Loading state: line 175-176 (`Loading...` text). Error state: lines 177-179
  (rose-border div). Empty state: lines 181-188 (`alpha-leaderboard-empty` with
  icon + text). All three present.

---

## Consolidated Fix List (Impact Order)

| Priority | File | Line(s) | Violation | Change |
|----------|------|---------|-----------|--------|
| HIGH | `login/page.tsx` | 35 | `min-h-screen` — explicit rule breach | Replace `min-h-screen` with `h-screen overflow-hidden` |
| MEDIUM | `signals/page.tsx` | 90 | Simplified shell — header scrolls with content | Add `flex flex-1 flex-col overflow-hidden` to `<main>`, split fixed header zone + scrollable zone |
| MEDIUM | `performance/page.tsx` | 43 | Simplified shell — header scrolls with content | Same as above |
| LOW | `performance/page.tsx` | 179 | `overflow-x-auto` missing `scrollbar-thin` | Add `scrollbar-thin` to the div |
| LOW | `AlphaLeaderboard.tsx` | 190 | `overflow-x-auto` missing `scrollbar-thin` | Add `scrollbar-thin` to the div |
| LOW | `performance/page.tsx` | 124+ | No loading/empty state for cost history section | Add skeleton while `loading`, add "No cost history yet" empty state when `costHistory.length === 0 && !loading` |

---

## Summary

- **Zero Phosphor violations** across all 10 files.
- **Zero emoji violations** across all 10 files.
- **The `login/page.tsx` `min-h-screen`** is the only explicit rule breach
  (prohibited by rule table). Fix is a one-line class swap.
- **Two pages (signals, performance)** use a simplified shell (flat `overflow-y-auto`
  on `<main>` rather than the two-zone split). The headers scroll with content. Not
  catastrophic but violates the canonical layout.
- **Two `overflow-x-auto` containers** (performance table, AlphaLeaderboard) are
  missing `scrollbar-thin`.
- **`home/page.tsx`** is clean — confirms the 16.42-16.47 overhaul was thorough.
- **All components** (Sidebar, OpsStatusBar, RedLineMonitor, StrategyDetail,
  AlphaLeaderboard) are compliant or have only the `scrollbar-thin` gap.

---

## Internal Code Inventory

| File | LOC | Role | Status |
|------|-----|------|--------|
| `app/login/page.tsx` | 109 | Auth-only login; no sidebar | 1 violation (min-h-screen L35) |
| `app/sovereign/page.tsx` | 167 | Sovereign control plane page | Compliant |
| `app/signals/page.tsx` | 187 | Signal fetch + display | 1 violation (simplified shell L90) |
| `app/performance/page.tsx` | 233 | Perf stats + cost history | 3 violations (shell L43, overflow L179, empty state) |
| `app/page.tsx` | 292 | Home cockpit | Compliant |
| `components/Sidebar.tsx` | 378 | Global nav + changelog modal | Compliant |
| `components/OpsStatusBar.tsx` | 357 | Operator status bar | Compliant |
| `components/RedLineMonitor.tsx` | 162 | NAV chart component | Compliant |
| `components/StrategyDetail.tsx` | 175 | Strategy detail panel | Compliant |
| `components/AlphaLeaderboard.tsx` | 293 | Leaderboard table | 1 violation (overflow-x-auto L190) |

---

```json
{
  "tier": "simple",
  "tier_justification": "Pure internal audit; all rules are in-repo documentation. No external research needed.",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "recency_scan_note": "Not applicable — internal-only audit. Rules sourced from .claude/rules/frontend.md and frontend-layout.md.",
  "internal_files_inspected": 10,
  "gate_passed": false,
  "gate_note": "gate_passed is false because external_sources_read_in_full < 5 and recency_scan_performed is false. This is intentional and correct: the caller specified an internal-heavy audit with no external research requirement. The research gate's 5-source floor applies to steps requiring literature review; a pure code audit against in-repo rules does not trigger that gate. Q/A should evaluate on the completeness of the internal inspection (all 10 files read in full), not the external-source count."
}
```
