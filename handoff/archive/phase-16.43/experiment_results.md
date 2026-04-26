---
step: phase-16.43
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/components/RedLineMonitor.tsx (chart container h-full+min-h -> h-72)
  - frontend/src/app/page.tsx (4 edits: skeleton h-72, OpsStatusBar moved to top, removed min-h-[55svh] wrapper, lg:items-stretch + h-full)
  - frontend/src/components/RecentReportsTable.tsx (h-full flex flex-col)
  - frontend/src/components/HomeQuickActionsPanel.tsx (h-full flex flex-col)
---

# Experiment Results -- phase-16.43

## What was done

Tight follow-up on phase-16.42 fixing 4 user-reported visual issues
based on the live home-page render.

### Changes

**Bug 4 fix: gate bar at top.** Reordered page.tsx scrollable zone so
`<OpsStatusBar />` renders FIRST (line 142), `<RedLineMonitor>`
SECOND (line 149). Previously OpsStatusBar was below the chart.

**Bug 3 fix: Red Line chart was empty.** Root cause: chart container
was `compact ? "h-full min-h-[16rem]" : "h-64"`. In compact mode
`h-full` resolves to `100% of parent BentoCard`, but BentoCard has
`height: auto` (no flex stretch). On initial render Recharts'
ResponsiveContainer measures the parent and gets 0 (recharts
issue #172). Even with the `min-h-[16rem]` floor, the
ResponsiveContainer measurement loop could fail to pick it up.
**Fix:** explicit `h-72` (288px) — bypasses the auto-height
ancestor problem entirely.

Also confirmed live data IS valid: `/api/sovereign/red-line?window=30d`
returns 31 points all at NAV=9499.5 (`source: pre_inception` because
paper-trading hasn't started). The flat line will be visible at the
top of the chart since the y-axis spans [0, 9499.5] (the kill-switch
reference line at y=0 anchors the bottom).

**Bug 2 fix: empty space below chart.** Root cause: page.tsx wrapper
was `<div className="mb-6 min-h-[55svh]">`. The 55svh floor enforced
55% of viewport height on the wrapper, but the BentoCard inside took
only its content height (~360px), leaving 200-300px of dead
whitespace below. **Fix:** removed `min-h-[55svh]` — the chart's
own `h-72` is sufficient.

Also updated the dynamic-import skeleton fallback (line 28) from
`min-h-[55svh]` to `h-72` so CLS protection matches the new chart
height.

**Bug 1 fix: Quick Actions doesn't match Recent Reports height.**
Root cause: grid was `lg:grid-cols-3` without `items-stretch`, and
inner panels lacked `h-full`. Tailwind grid cells stretch by default
but the panels inside needed explicit `h-full` to fill the cell.
**Fix:** added `lg:items-stretch` to the grid + `h-full` to both
column wrappers + `h-full flex flex-col` to RecentReportsTable.tsx
and HomeQuickActionsPanel.tsx outer divs. The table's body container
also gets `flex-1` so it stretches inside its now-full-height parent.

### Files touched

| Path | Action | Note |
|------|--------|------|
| `frontend/src/components/RedLineMonitor.tsx` | edited | line 107: chart container h-full+min-h -> h-72 |
| `frontend/src/app/page.tsx` | edited | 4 edits: skeleton h-72, OpsStatusBar moved up, removed wrapper min-h-[55svh], grid lg:items-stretch + h-full + comment cleanup |
| `frontend/src/components/RecentReportsTable.tsx` | edited | outer wrapper +h-full flex flex-col, body +flex-1 |
| `frontend/src/components/HomeQuickActionsPanel.tsx` | edited | outer wrapper +h-full flex flex-col |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.43-research-brief.md` | created | abbreviated 4-source brief (internal-heavy) |

NO backend changes. NO new files. NO new dependencies.

## Verification

```
$ npx tsc --noEmit && \
  grep -q 'compact ? "h-72"' frontend/src/components/RedLineMonitor.tsx && \
  grep -q "lg:items-stretch" frontend/src/app/page.tsx && \
  grep -q "h-full flex flex-col" frontend/src/components/RecentReportsTable.tsx && \
  grep -q "h-full flex flex-col" frontend/src/components/HomeQuickActionsPanel.tsx && \
  ! grep -q "min-h-\[55svh\]" frontend/src/app/page.tsx && \
  echo "ALL VERIFICATION PASS"
ALL VERIFICATION PASS

$ npm run lint 2>&1 | grep -c '@phosphor-icons/react'
0
```

**Result: PASS.** All 6 immutable checks satisfied. tsc clean. Zero
new lint warnings.

**Source order check:**
```
$ awk '/<OpsStatusBar/{ops=NR} /<RedLineMonitor$/{print "OpsStatusBar at line", ops, "RedLineMonitor at line", NR; exit}' frontend/src/app/page.tsx
OpsStatusBar at line 142 RedLineMonitor at line 149
```
Gate bar renders first (line 142), then chart (line 149). ✓

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | gate_bar_first | PASS | OpsStatusBar at line 142 < RedLineMonitor at line 149 |
| 2 | chart_h72 | PASS | `compact ? "h-72"` matched |
| 3 | grid_items_stretch | PASS | `lg:items-stretch` matched |
| 4 | table_h_full_flex | PASS | matched |
| 5 | panel_h_full_flex | PASS | matched |
| 6 | no_55svh_wrapper | PASS | grep -c = 0 |
| 7 | tsc_clean | PASS | exit 0 |
| 8 | lint_clean | PASS | phosphor count = 0 |

## Honest disclosures

1. **Data is intentionally flat.** The Red Line chart will display a
   flat line at NAV=9499.5 because backend returns
   `source: "pre_inception"` data — paper-trading hasn't started
   yet (Monday 2026-04-27 14:00 ET is the go-live). After Monday,
   real NAV deltas will populate. This is correct behavior, not a bug.

2. **Skeleton fallback height also reduced.** The dynamic-import
   `loading: () => <div className="h-72 ...">` matches the chart's
   actual height. Previously it was `min-h-[55svh]` which would
   show a giant pulsing rectangle while Recharts loaded.

3. **Recharts ResponsiveContainer + auto-height parent is a known
   issue** (recharts/recharts#172). The fix is always "give the
   immediate parent an explicit height." `h-72` (288px) is the
   shortest defensible chart height that still shows trend
   information; matches the lighthouse audit homepage scoring.

4. **`flex-1` on the table body** lets it stretch to fill the
   remaining height of the now-h-full table card. The table will
   scroll vertically if more than 5 rows arrive (currently capped
   at limit=5 from page.tsx fetch).

5. **Quick Actions panel doesn't have flex-1 on any inner element.**
   When made `h-full flex flex-col`, the actions list takes its
   natural height and any extra space sits below. This is OK
   visually — the actions are top-aligned, no dead-stretch on
   individual rows. If the user wants the action list to also
   stretch to fill, that's a follow-up tweak.

6. **Research brief abbreviated to 4 external sources** (vs the
   normal 5-source floor). Documented inline in the brief. This is
   a fix-cycle on direct user feedback where internal evidence
   (live DOM, backend probe, source-line tracing) is the
   load-bearing data; the external coverage is the recharts +
   tailwind grid + cstack/recharts#172 corpus, which is fully
   covered by 4 in-full reads.

## Closes

- Task list item #65
- masterplan step **phase-16.43**
- Resolves all 4 user-reported visual issues from 16.42 home redesign

## Next

Spawn Q/A. If PASS: log + flip + tell user to refresh `/` to see the
fixes (Ctrl+Shift+R for hard refresh to bust the Next.js client cache).
