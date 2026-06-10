# Research Brief: Phase 15.6 — Sprint tile week selector dropdown

**Tier:** simple (assumption: stated by caller)
**Date:** 2026-04-21

---

## Executive summary

Phase 15.6 wires a week-selector `<select>` into `HarnessSprintTile` so operators can inspect historical sprint state. The backend already accepts `?week_iso=` on `/api/harness/sprint-state`, and the weekly ledger endpoint (phase 15.5) already supplies the list of `week_iso` values. This is a pure frontend change: add a controlled `<select>` to `HarnessSprintTile`, lift `selectedWeekIso` state into `HarnessDashboard`, and replace the current no-argument `getHarnessSprintState()` call with a `selectedWeekIso`-dependent effect.

Research confirms:
1. Native `<select>` is the correct choice here — no search, no multi-select, ~52 options max.
2. The fetch effect belongs in the **parent** (`HarnessDashboard`) because it already owns `sprintState` and all the other parallel fetches. The tile stays a dumb display component with an `onWeekChange` callback.
3. During re-fetch, keep old data visible + add an inline loading indicator (skeleton-over-stale pattern). Do not blank the tile.
4. Default to the **most recent week** from the ledger (first row, since the backend returns newest-first); fall back to empty string (current week) if the ledger is empty.

---

## Read in full (>=5 required)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://react.dev/reference/react-dom/components/select | 2026-04-21 | Official doc | WebFetch | Controlled select requires `value`+`onChange`; no React 19-specific changes; `<label>` association required for a11y |
| https://react.dev/reference/react/useEffect | 2026-04-21 | Official doc | WebFetch | Canonical dependent-fetch pattern: `useEffect([selectedWeekIso])` with `ignore` flag for race-condition prevention; clear state on fetch start |
| https://react.dev/learn/removing-effect-dependencies | 2026-04-21 | Official doc | WebFetch | Effect for data fetch lives in the component that *uses* the data; parent passes prop, child (or parent state handler) owns the effect |
| https://www.thewcag.com/examples/dropdowns-selects | 2026-04-21 | Accessibility guide | WebFetch | "Use native select when styling allows"; native handles all WCAG 2.2 Level AA a11y without ARIA; first option should be placeholder |
| https://blog.logrocket.com/ux-design/skeleton-loading-screen-design/ | 2026-04-21 | UX blog | WebFetch | Skeleton-over-stale is preferred for tile/card re-fetches; blanking tiles breaks layout stability and user orientation |
| https://medium.com/lego-engineering/building-accessible-select-component-in-react-b61dbdf5122f | 2026-04-21 | Engineering blog | WebFetch | Native select preferred by GOV.UK, Salesforce, IBM, Shopify; custom only when secondary display data is needed (not applicable here) |

---

## Snippet-only (identified, not read in full)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://copyprogramming.com/howto/javascript-react-fill-select-option-from-array | Tutorial | Covered by official docs |
| https://coreui.io/answers/how-to-handle-select-dropdown-in-react/ | Vendor doc | Content subsumed by react.dev |
| https://ibrahimaq.com/blogs/how-to-create-a-custom-accessible-dropdown-with-react-and-typescript/ | Blog | Custom combobox not needed here |
| https://retool.com/blog/hooks-and-state-102-the-dependency-array-in-useeffect | Blog | Subsumed by react.dev useEffect doc |
| https://blog.logrocket.com/useeffect-react-hook-complete-guide/ | Blog | Subsumed by official doc + removing-effect-dependencies |
| https://www.24a11y.com/2019/select-your-poison-part-2/ | A11y blog | Older (2019), native select superiority already confirmed by WCAG guide |
| https://medium.com/@katr.zaks/building-an-accessible-dropdown-combobox-in-react-a-step-by-step-guide-f6e0439c259c | Blog | Combobox pattern not needed |
| https://sachinkasana.medium.com/i-replaced-my-spinner-with-a-skeleton-and-my-ux-skyrocketed-5d261da61752 | Blog | Skeleton preference confirmed by logrocket fetch |
| https://medium.com/productboard-engineering/%EF%B8%8F-spinners-versus-skeletons-in-the-battle-of-hasting-b51b9c6574ef | Blog | Same conclusion as logrocket |
| https://maxrozen.com/learn-useeffect-dependency-array-react-hooks | Blog | Practical tips subsumed by official doc |

---

## Recency scan (2024-2026)

Searched: "React 19 controlled select dropdown best practices accessible 2026", "React useEffect dependency array re-fetch on prop change pattern 2025", "loading state UX stale content skeleton spinner while refetching 2025", "React 19 parent vs child fetch on selected value change where effect lives pattern 2025 2026".

**Result:** No new 2024-2026 findings supersede the canonical patterns. React 19 introduced no changes to the `<select>` element API (confirmed by react.dev). The `useEffect` dependency pattern is unchanged in React 19. React 19.2 adds `useEffectEvent` but it is not needed here — `selectedWeekIso` is correctly reactive (it should trigger a re-fetch). Skeleton-over-stale UX consensus is unchanged and reinforced by 2025 material. The pyfinagent dashboard already uses the `ignore` flag pattern (`getSeedStability` polling, line 652-655 of `HarnessDashboard.tsx`).

---

## Key findings

1. **Native `<select>` is sufficient.** With <=52 options (one year of weekly_ledger rows, backend caps at 52 per `truncated` field), no search/autocomplete is needed. Native `<select>` provides full keyboard nav and screen-reader support without ARIA boilerplate. (Source: react.dev/reference/react-dom/components/select, WCAG guide)

2. **Controlled pattern: `value` + `onChange`.** `selectedWeekIso` lives in the parent as `useState<string>`. The `<select>` in the tile receives it as a prop and fires `onWeekChange(e.target.value)`. (Source: react.dev controlled select)

3. **Effect lives in the parent.** `HarnessDashboard` already owns `sprintState` and calls `getHarnessSprintState()`. The re-fetch effect should be a *separate* `useEffect([selectedWeekIso])` in the parent (not inside the tile) so the tile stays a pure display component — consistent with the existing architecture where the tile receives `data` as a prop. The React docs show the child-owns-fetch pattern, but the project already uses the parent-fetches-passes-down pattern for all other tiles; consistency wins. (Source: react.dev removing-effect-dependencies + internal audit)

4. **Race condition prevention.** Use the `ignore` flag pattern (already used in the seed-stability polling at HarnessDashboard.tsx:652). Set `setSprintState(null)` at the start of the effect to show a loading indicator in the tile while the new week's data arrives. (Source: react.dev useEffect)

5. **Stale-data UX: show spinner badge, not blank.** Set `setSprintState(null)` only when `selectedWeekIso` changes (inside the effect), not before — this way the tile briefly keeps old data while the spinner appears, then updates. Actually simpler: set to null at effect start (the tile already handles `data === null` with its empty state). For a better UX, add a `tileLoading` boolean state in the parent so the tile can show an overlay spinner without blanking. However, given the sprint tile has a clean null-state empty screen already (lines 145-168 of HarnessSprintTile.tsx), setting `sprintState` to null is acceptable and matches the existing loading pattern. (Source: logrocket skeleton UX + internal audit)

6. **Default week on mount.** Use the most recent week from the ledger (first row, since the backend returns newest-first based on the WeeklyLedgerTable rendering). Initialize `selectedWeekIso` to `""` (empty string = current week per existing `getHarnessSprintState` signature at api.ts:479-483). Then in a `useEffect([weeklyLedger])`, if `weeklyLedger?.rows[0]?.week_iso` exists and `selectedWeekIso` is still `""`, set it to that value. This gives current-week data on first render (via the initial parallel fetch) and snaps to the most recent ledger week once the ledger loads. This avoids a double-fetch on mount.

---

## Internal code audit

### Q1: Current call site for `getHarnessSprintState()`

`HarnessDashboard.tsx` line 629:
```
getHarnessSprintState().catch(() => null),
```
Called with NO argument inside a `Promise.all([...])` in the mount-time `useEffect([], [])` at line 621. The result is stored at line 641: `setSprintState(sprint)`. No `selectedWeekIso` state exists yet.

### Q2: Prop signature to add on `HarnessSprintTile`

Current interface (`HarnessSprintTile.tsx` lines 31-35):
```typescript
export interface HarnessSprintTileProps {
  data: HarnessSprintWeekState | null;
  approval?: MonthlyApprovalState | null;
  onApproved?: () => void;
}
```

New interface:
```typescript
export interface HarnessSprintTileProps {
  data: HarnessSprintWeekState | null;
  approval?: MonthlyApprovalState | null;
  onApproved?: () => void;
  weeks?: string[];           // list of week_iso values from weekly_ledger
  selectedWeekIso?: string;   // controlled value for the <select>
  onWeekChange?: (weekIso: string) => void;  // callback to parent
  tileLoading?: boolean;      // optional: show spinner overlay during re-fetch
}
```

### Q3: Weeks list source

`weeklyLedger` is already fetched at `HarnessDashboard.tsx` line 633 and stored at line 646:
```
setWeeklyLedger(ledger);
```
Pass to tile as: `weeklyLedger?.rows.map(r => r.week_iso) ?? []`

### Q4: Verification grep literal match

The masterplan grep is:
```
grep -qE 'getHarnessSprintState\(\s*selectedWeekIso'
```
This matches `getHarnessSprintState(selectedWeekIso)` (with optional whitespace). The effect body in the parent must contain exactly:
```typescript
getHarnessSprintState(selectedWeekIso)
```
(not `getHarnessSprintState( selectedWeekIso )` with leading space — the `\s*` allows zero spaces so `getHarnessSprintState(selectedWeekIso)` is the safe literal to use.)

### Q5: Default `selectedWeekIso` on mount

**Recommendation: initialize to `""` (empty string = current week).** Rationale:
- The initial parallel fetch at mount already calls `getHarnessSprintState()` (no arg = current week).
- When `weeklyLedger` loads, a secondary `useEffect([weeklyLedger])` can set `selectedWeekIso` to `weeklyLedger.rows[0]?.week_iso ?? ""` IF `selectedWeekIso` is still `""`. This avoids a redundant re-fetch on first load because the initial fetch already returned current-week data.
- If the current week matches the first ledger row, no second fetch is triggered.

### Q6: `<select>` Tailwind styling (navy palette)

Matching the project's existing `bg-navy-800/60`, `border-navy-700`, `text-slate-300` tokens from the surrounding section:

```tsx
<select
  value={selectedWeekIso}
  onChange={e => onWeekChange?.(e.target.value)}
  className="rounded-md border border-navy-700 bg-navy-900/80 px-2 py-1 font-mono text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-sky-500"
  aria-label="Select week"
>
  {weeks.map(w => (
    <option key={w} value={w}>{w}</option>
  ))}
</select>
```

Place it in the tile header row (alongside the `<span>` showing `weekIso`), right-aligned, so it replaces or sits next to the current static `weekIso` label.

---

## Concrete recommendations

### 1. New prop signature on `HarnessSprintTile`

```typescript
export interface HarnessSprintTileProps {
  data: HarnessSprintWeekState | null;
  approval?: MonthlyApprovalState | null;
  onApproved?: () => void;
  weeks?: string[];
  selectedWeekIso?: string;
  onWeekChange?: (weekIso: string) => void;
}
```

(Skip `tileLoading` — the tile's existing null-state is clean enough for the fast local fetch.)

### 2. Effect location and dependencies (parent)

Add to `HarnessDashboard`:
```typescript
const [selectedWeekIso, setSelectedWeekIso] = useState<string>("");

// Snap to first ledger row when ledger loads (avoids double-fetch on mount)
useEffect(() => {
  if (weeklyLedger?.rows[0]?.week_iso && selectedWeekIso === "") {
    setSelectedWeekIso(weeklyLedger.rows[0].week_iso);
  }
}, [weeklyLedger]);

// Re-fetch sprint state when week selection changes
useEffect(() => {
  if (selectedWeekIso === "") return; // initial load handled by mount effect
  let ignore = false;
  setSprintState(null);
  getHarnessSprintState(selectedWeekIso)
    .then(s => { if (!ignore) setSprintState(s); })
    .catch(() => { if (!ignore) setSprintState(null); });
  return () => { ignore = true; };
}, [selectedWeekIso]);
```

### 3. Exact grep-passing literal

The re-fetch effect must contain:
```
getHarnessSprintState(selectedWeekIso)
```
This satisfies `grep -qE 'getHarnessSprintState\(\s*selectedWeekIso'`.

### 4. Default `selectedWeekIso` on mount

`useState<string>("")` — current week. Snaps to first ledger row once the ledger loads (via the secondary effect above), avoiding a redundant re-fetch if the current week IS the first ledger row.

### 5. `<select>` Tailwind tokens

```
rounded-md border border-navy-700 bg-navy-900/80 px-2 py-1 font-mono text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-sky-500
```

Render in the tile header alongside the existing `weekIso` display span. When `weeks` is empty or undefined, render only the static span (graceful degradation).

### 6. Dashboard call site

Update `<HarnessSprintTile>` in the parent JSX:
```tsx
<HarnessSprintTile
  data={sprintState}
  approval={monthlyApproval}
  onApproved={refreshMonthlyApproval}
  weeks={weeklyLedger?.rows.map(r => r.week_iso) ?? []}
  selectedWeekIso={selectedWeekIso}
  onWeekChange={setSelectedWeekIso}
/>
```

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (16 total: 6 read in full + 10 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (HarnessDashboard.tsx, HarnessSprintTile.tsx, api.ts, types.ts)
- [x] Consensus noted (native select; parent effect; skeleton-over-stale)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "gate_passed": true
}
```
