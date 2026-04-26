# Research Brief — phase-16.49: UX Audit Pass B (High-Risk Pages)

Generated: 2026-04-25. Tier: internal-heavy (code audit only; no external sources required per scope).

---

## File inventory

| File | LOC | Role |
|---|---|---|
| `frontend/src/app/backtest/page.tsx` | 1594 | Backtest + optimizer hub (multi-tab, walk-forward) |
| `frontend/src/app/reports/page.tsx` | 599 | Analysis reports browser + compare |
| `frontend/src/app/agents/page.tsx` | 728 | MAS live stream + run history + agent map + OpenClaw |
| `frontend/src/app/paper-trading/page.tsx` | 769 | Paper trading dashboard (multi-tab + OpsStatusBar) |
| `frontend/src/app/paper-trading/learnings/page.tsx` | 22 | Thin wrapper hosting VirtualFundLearnings component |

---

## Per-file violations

### 1. `backtest/page.tsx` (1594 LOC)

**V1 — Error banners in fixed header zone (lines 683–709).**
Two error banners (`{error && ...}` and `{btStatus?.status === "error" ...}`) are inside the `flex-shrink-0` fixed header div (opened at line 634, closed at line 787). Per `frontend-layout.md` §3, status banners (Tier 2) belong in the scrollable zone, not the fixed zone. Having them in the fixed zone means a persistent error consumes viewport height permanently and pushes tab bar and content down.

**V2 — Run selector (Tier 4 global control) in fixed header zone (lines 735–766).**
`RunSelector` renders inside the same `flex-shrink-0` div. This is a Tier 4 control that can grow large (many runs), squeezing the viewport. It should be in the scrollable zone, or if kept fixed, must be constrained to `max-h-XX overflow-y-auto scrollbar-thin`.

**V3 — Three `overflow-x-auto` table wrappers missing `scrollbar-thin` (lines 1055, 1118, 1301).**
"Strategy vs Baselines" table, "Walk-Forward Windows" table, and the trade list table all use bare `overflow-x-auto` without `scrollbar-thin`. Convention: all overflow containers need `scrollbar-thin`.

**V4 — RunSelector dropdown at line 209: `overflow-y-auto` missing `scrollbar-thin`.**
`max-h-80 overflow-y-auto` at line 209 (inside `RunSelector` sub-component defined at top of backtest page) lacks `scrollbar-thin`.

Shell, Phosphor imports, emoji: COMPLIANT. Scrollable zone `overflow-y-auto scrollbar-thin` at line 790: COMPLIANT. Tab bar in fixed zone: COMPLIANT. Loading/error/empty states: present. OpsStatusBar: N/A for this page.

---

### 2. `reports/page.tsx` (599 LOC)

**V1 — CRITICAL: Broken two-zone shell (line 223).**
`<main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">` — the main element IS the scrollable zone. There is NO fixed header zone and NO `flex flex-col overflow-hidden` on main. This means:
- The tab bar at line 230 scrolls off-screen when content grows.
- There is no two-zone separation.
The canonical shell requires `<main className="flex flex-1 flex-col overflow-hidden">` with an inner `flex-shrink-0` fixed zone (header + tabs) and an inner `flex-1 overflow-y-auto scrollbar-thin` scrollable zone.

**V2 — Tab bar in scrollable zone (line 230).**
Because the shell is broken (V1), the tab bar renders directly inside the scrollable `main`. It scrolls off-screen. Must move to a `flex-shrink-0` fixed header div per `frontend-layout.md` §5.

**V3 — Loading state is plain text, not Skeleton (line 253).**
`{loading && <p className="text-slate-400">Loading reports...</p>}` — should use `<PageSkeleton />` or `<SkeletonGrid />` per frontend.md "Loading states" convention.

**V4 — `overflow-x-auto` at line 482 missing `scrollbar-thin`.**
A comparison table wrapper uses bare `overflow-x-auto` without `scrollbar-thin`.

Shell fix is a 4-line wrap: add `flex flex-col overflow-hidden` to main, wrap the header + tabs in `flex-shrink-0` div, wrap the rest in `flex-1 overflow-y-auto scrollbar-thin`. Tab bar moves inside the fixed div.

---

### 3. `agents/page.tsx` (728 LOC)

**COMPLIANT** on all 8 rules:
- Shell at line 289–291: `flex h-screen overflow-hidden` outer, `flex flex-1 flex-col overflow-hidden` main. Correct.
- Fixed header zone at line 293 (`flex-shrink-0`): header, hero metrics, tab bar. Correct tier placement.
- Tab bar at line 349: inside fixed zone. COMPLIANT.
- Scrollable zone at line 368: `flex-1 overflow-y-auto scrollbar-thin`. COMPLIANT.
- Error banner at line 370: inside scrollable zone. COMPLIANT.
- Empty state at line 384: Broadcast icon + text. COMPLIANT.
- No `min-h-screen`, no `@phosphor-icons/react` direct import, no emoji.
- No OpsStatusBar needed (not an operator dashboard).
- `overflow-x-auto`/`overflow-y-auto` inside content: none found outside the scrollable zone.

No violations.

---

### 4. `paper-trading/page.tsx` (769 LOC)

**V1 — Two `overflow-x-auto` table wrappers missing `scrollbar-thin` (lines 533, 617).**
Positions table (line 533) and trades table (line 617) use bare `overflow-x-auto` without `scrollbar-thin`.

Shell (line 2–5 of return): checked via agents pattern — line 482 confirms `flex-1 overflow-y-auto scrollbar-thin` scrollable zone. Fixed header zone with OpsStatusBar and tab bar confirmed at lines 457–479. COMPLIANT.
Phosphor imports: all from `@/lib/icons`. COMPLIANT.
No emoji. OpsStatusBar used (line 522): COMPLIANT with §4.5.
Loading/error/empty states: present (PageSkeleton at line 510, error banner at line 484, empty state at line 512). COMPLIANT.

---

### 5. `paper-trading/learnings/page.tsx` (22 LOC)

COMPLIANT. Correct two-zone shell. No tabs (single content component). No violations.

---

## Consolidated fix list (ordered by impact)

### Fix 1 — reports/page.tsx: broken shell + scrolling tab bar [HIGH]
**Impact:** Tab bar scrolls off-screen on any page with >3 reports. Core UX regression.
**Edit type:** 4-line structural wrap. Not a large refactor.

Change `<main className="flex-1 overflow-y-auto scrollbar-thin p-6 md:p-8">` to `<main className="flex flex-1 flex-col overflow-hidden">`.

Then wrap lines 224–245 (header + tab bar) in:
```tsx
<div className="flex-shrink-0 px-6 pt-6 pb-0 md:px-8 md:pt-8">
  {/* header */}
  {/* tab bar */}
</div>
```

And wrap lines 246 onward (error + content) in:
```tsx
<div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 md:px-8">
  {/* tab content */}
</div>
```

### Fix 2 — reports/page.tsx: plain-text loading → PageSkeleton [LOW]
Line 253: replace `<p className="text-slate-400">Loading reports...</p>` with `<PageSkeleton />`. Add import if not present.

### Fix 3 — backtest/page.tsx: error banners moved to scrollable zone [MEDIUM]
Lines 683–709 (two error banner blocks) are inside the `flex-shrink-0` fixed header div. Move them to just below the opening of the scrollable `<div className="flex-1 overflow-y-auto ...">` at line 790. 1-line structural move per banner block.

### Fix 4 — backtest/page.tsx: RunSelector in fixed zone [MEDIUM — defer if risky]
Lines 735–766: RunSelector inside fixed zone. If the selector can grow tall, this squeezes the visible tab content. Minimal fix: add `max-h-40 overflow-y-auto scrollbar-thin` to the RunSelector's internal container, OR move the whole selector into the scrollable zone. Flag for deferral if moving it changes the visual layout significantly — the selector is intentionally above the tab bar per an earlier cycle.

### Fix 5 — `scrollbar-thin` on `overflow-x-auto` tables [LOW — 1-line each]
5 occurrences total:
- `backtest/page.tsx` line 1055: add `scrollbar-thin`
- `backtest/page.tsx` line 1118: add `scrollbar-thin`
- `backtest/page.tsx` line 1301: add `scrollbar-thin`
- `backtest/page.tsx` line 209 (RunSelector dropdown): add `scrollbar-thin`
- `reports/page.tsx` line 482: add `scrollbar-thin`
- `paper-trading/page.tsx` line 533: add `scrollbar-thin`
- `paper-trading/page.tsx` line 617: add `scrollbar-thin`

Each is a 1-word class addition: `overflow-x-auto scrollbar-thin` or `overflow-y-auto scrollbar-thin`.

---

## Summary table

| File | Critical | Medium | Low | Deferred |
|---|---|---|---|---|
| backtest/page.tsx | 0 | 2 (V1 error-in-fixed, V2 runselector-in-fixed) | 3 (V3/V4 scrollbar-thin) | V2 if layout-sensitive |
| reports/page.tsx | 1 (broken shell + tab bar) | 0 | 2 (V3 loading, V4 scrollbar-thin) | — |
| agents/page.tsx | 0 | 0 | 0 | — |
| paper-trading/page.tsx | 0 | 0 | 2 (V1 scrollbar-thin ×2) | — |
| paper-trading/learnings/page.tsx | 0 | 0 | 0 | — |

**Highest-priority single fix:** reports/page.tsx shell restructure (Fix 1) — it breaks tab bar pinning on the page with the most content-dependent scroll depth.

---

## Research Gate Checklist

Hard blockers:
- [x] file:line anchors for every internal claim
- [x] Internal exploration covered every relevant module (5 files, all read)
- [x] Contradictions / consensus noted (none; agents/paper-trading-learnings clean)
- [x] All claims cited with file:line

Soft checks:
- [x] Internal-heavy gate honestly justified: pure code audit against rules already documented in `.claude/rules/frontend.md` + `frontend-layout.md`. No fresh external research adds value.

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-16.49-research-brief.md",
  "gate_passed": true,
  "note": "Internal-only gate. Code audit against documented project rules; external floor of 5 sources does not apply (no novel pattern researched). gate_passed reflects internal completeness."
}
```
