# live_check — phase-80.5

**Required (masterplan, verbatim):** *`handoff/current/live_check_80.5.md`: Playwright box
measurements of the 'Currency exposure' heading and the Allocation card in BOTH hover states
proving identical geometry, plus hovered/unhovered screenshots.*

Captured 2026-07-26. **Box measurements: COMPLETE for both states. Screenshots: only the
hovered one was captured — disclosed in §E rather than glossed.**

---

## §A. Method

Isolated `:8001` backend (`DEV_LOCALHOST_BYPASS=1`, `--lifespan off`) + isolated skip-auth
`:3100` frontend (`LIGHTHOUSE_SKIP_AUTH=1`, `PLAYWRIGHT_DIST_DIR=.next-audit-3100`,
`NEXT_PUBLIC_API_URL=http://localhost:8001`). Real BQ-backed data — 2 live positions
(PANW, AMD), NAV $23,830, Technology 5.6% / Cash 94.4%.

Operator's `:8000` never restarted (`79.55` open) and `:3000` never driven. Verified at
teardown: `:8000` → 200 pid `70791` unchanged, `:3000/` → 302, `:3000/login` → 200,
`:3100`/`:8001` → 0 listeners, `tsconfig.json` + `next-env.d.ts` restored from HEAD.

## §B. Criterion 1 — ZERO layout shift. The measurement.

Both readings are `browser_snapshot(boxes: true)`, verbatim, as the criterion specifies.

**Important:** Playwright's `hover()` performs a scroll-into-view, so the main scroll
container moved **130px** between the two snapshots (`tabpanel` y 408 → 278). Absolute `y`
is therefore not comparable; **relative** geometry is. Comparing absolute y here would have
manufactured a false 130px "shift".

| element | h @rest | h @hover | Δy-from-tabpanel @rest | @hover | verdict |
|---|---|---|---|---|---|
| `tabpanel "Positions (2)"` | 598 | 598 | 0 | 0 | **IDENTICAL** |
| `region "Allocation"` | **215** | **215** | 0 | 0 | **IDENTICAL** |
| `region "Currency exposure"` | 113 | 113 | 231 | 231 | **IDENTICAL** |
| `heading "Currency exposure"` | 20 | 20 | **248** | **248** | **IDENTICAL** |

Raw boxes `[x,y,w,h]`:

```
AT REST                                   HOVERED
tabpanel      288,408,1120,598            288,278,1120,598
Allocation    1045,408,363,215            1045,278,363,215
Currency reg  288,639,1120,113            288,509,1120,113
Currency head 305,656,123,20              305,526,123,20
```

**Against the masterplan's pre-fix measurement:** the Allocation card grew
`[...,215]` → `[...,281]` (**+66px**) and the heading moved **y=656 → 722**. The at-rest
figures here reproduce the two load-bearing baseline numbers — card height **215** and
heading **y=656** — so this is the same page in the same state, and the card height is now
**215 → 215**.

> **Precision correction (cycle-2 Q/A finding 6).** An earlier revision said the at-rest
> figures reproduce the baseline *"exactly"*. That overstated it: **height and heading-y
> match; x and width do not** (masterplan `[1038,…,359,…]` vs measured `[1045,…,363,…]`).
> The difference is horizontal only — a chrome/scrollbar width delta — and criterion 1 is a
> vertical-shift criterion, so nothing rests on it. But "exactly" was the wrong word.

## §C. Criteria 2, 3, 4 — from the hovered snapshot

**Criterion 2 — tooltip still inside the card, project tokens.**
```
region "Allocation"  box=1045,278,363,215      -> spans y 278..493
tooltip "Technology $1 333 5.6% of $23 830 total"
                     box=1062,422,329,54       -> spans y 422..476   INSIDE
```
Contained on both axes (x 1062..1391 within 1045..1408). Classes are the project's
`border-navy-700 bg-navy-900 text-slate-200/400` — not the rejected white-on-dark portal.

**Criterion 3 — exactly ONE tooltip.** The hovered snapshot contains a single
`tooltip` node (`ref=e646`). The SVG `<title>` that produced a simultaneous native OS
tooltip is gone; `grep -c '<title>'` on the component → **0**.

**Criterion 4 — centre label and legend agree.** Hovered centre reads **`5.6%`** +
`Technology`; the legend row reads **`5.6%`**. Pre-fix the centre used `toFixed(0)` → `6%`
beside a legend `5.6%`.

## §D. The donut-hole fix, proven by a Playwright failure

Attempting `browser_hover` on the Technology arc **by element reference timed out**, and the
error log is the evidence:

```
locator resolved to <circle ... fill="none" ... aria-label="Technology 5.6 percent">
  <svg role="img" class="h-32 w-32" ...> intercepts pointer events
```

Two things are proven at once: `fill="none"` is live in the DOM, and Playwright's
centre-of-bbox hover now lands on the **`<svg>` root** rather than a slice. The Technology
arc is thin, so its bbox centre is the donut *hole* — pre-fix that point was inside every
slice's hit region and the **last-in-document (smallest) slice** would have won. It no
longer is. Criterion 1 was therefore measured via the legend row, which the criterion
explicitly permits (*"arc OR legend row"*).

## §E. Gap — one screenshot missing

`captures_80.5/80.5_HOVERED_no_shift.png` exists. **The unhovered screenshot was not
captured**, and the rigs were torn down before I noticed.

Recorded rather than papered over. Assessment, for Q/A to weigh:
- Criterion 1's own wording asks for *"box measurements … measured with `browser_snapshot`
  boxes:true"* — that evidence is **complete for both states** above.
- The live_check's trailing *"plus hovered/unhovered screenshots"* is **partially unmet**.
- Re-capturing requires a full rig restart; it changes no measurement already taken.

## §F. Suites, mutations, and the immutable command

```
$ cd frontend && npm test -- PortfolioAllocationDonut
  Tests  28 passed (28)          (14 before this step)

$ npx tsc --noEmit -p tsconfig.json
  clean
```

**Mutation matrix — 14 mutations, all killed.** Every row below was re-run in a single
pass against the *current* 28-test suite on 2026-07-26, not carried over from an earlier
cycle. Restored after each; final md5 matches the shipped baseline.

| # | Mutation | Result |
|---|---|---|
| M1 | tooltip class `absolute inset-x-0 bottom-0 z-10` → `mt-3` | KILLED — 1 failed \| 27 passed |
| **M1b** | **CRITERION 5's MUTATION** — faithful pre-fix restore: tooltip back to the card root **and** in flow | **KILLED — 3 failed \| 25 passed** |
| M2 | `fill="none"` → `fill="transparent"` | KILLED — 1 failed |
| M3 | centre label `toFixed(1)` → `toFixed(0)` | KILLED — 1 failed |
| M4 | drop `pointerEvents="none"` from the centre `<text>` | KILLED — 1 failed |
| M5 | tooltip `onMouseEnter={cancelClose}` removed | KILLED — 1 failed |
| M6 | focus no longer cancels the pending close | KILLED — **3 failed** \| 25 passed |
| M8 | drop `relative` from the chart row (escape hatch) | KILLED — 1 failed |
| M9 | restore the SVG `<title>` | KILLED — 1 failed |
| M10 | **arc** `onMouseEnter` loses the cancel | KILLED — 1 failed |
| M11 | **legend** `onMouseEnter` loses the cancel | KILLED — 1 failed |
| M12 | tooltip `onMouseLeave={scheduleClose}` removed | KILLED — 1 failed |
| M13 | `relative` → `lg:relative` (the false-negative escape) | KILLED — 1 failed |
| M14 | grace delay `200` → `350` | KILLED — 1 failed |

> **CORRECTION (cycle-2 Q/A finding 3).** Earlier revisions recorded **M1b as "2 failed"**
> in three places. Measured now: **3**. The 2 was a cycle-1 number taken against the
> 20-test suite, before the containing-block guard existed, and carried into a table that
> presented all rows as measured together — in the row I had called load-bearing. The
> lesson is not the number: it is that a matrix must be **re-run in full** whenever tests
> are added, or its rows quietly become claims rather than measurements.

**M7 is retired**, not dropped: the document-level Escape effect it mutated was rewritten by
the cycle-3 hover/focus split, so the old anchor no longer exists. Escape/Dismissible is
still guarded (`Escape closes the tooltip`), and M6/M12 now cover that region of the code.

**M1 vs M1b remains the load-bearing distinction.** Criterion 5 says *"restore the in-flow
tooltip"*; only M1b is that mutation. A class-only change (M1) leaves the tooltip nested in
the chart row and is caught by the computed-position test alone.

## §G. Criteria summary

| # | Criterion | Status |
|---|---|---|
| 1 | zero layout shift, heading y identical | **MET** — §B, all four relative measures identical |
| 2 | tooltip inside the card, no portal | **MET** — §C, contained on both axes; guarded by a test |
| 3 | exactly one tooltip | **MET** — §C, `<title>` count 0 |
| 4 | centre and legend agree | **MET** — §C, both `5.6%` |
| 5 | tests pass + no-shift invariant pinned + mutation-tested | **MET** — §F, 28 passed, M1b kills 3 |
