# Experiment Results — phase-80.5

**Step:** `80.5` (P1) — hovering the Allocation donut pushed the whole page down.
**Tier:** T2 (Opus 5 / `high`) — assigned in `contract_80.5.md` **before** GENERATE.
Date 2026-07-26. Contract: `handoff/current/contract_80.5.md`.
Gate: `handoff/current/research_brief_80.5.md` (`gate_passed: true`, 10 sources read in
full, 38 URLs, recency scan performed).

---

## What was built

All four defects are in one file, `frontend/src/components/PortfolioAllocationDonut.tsx`.
Sole consumer: `paper-trading/positions/page.tsx:163`. Sole test file:
`PortfolioAllocationDonut.test.tsx`.

### 1. The layout shift (the operator's report)

The chart-row wrapper gains `relative`; the tooltip moves **inside it** as
`absolute inset-x-0 bottom-0 z-10`, dropping `mt-3`.

It was an **in-flow sibling of the card's flex column**, so it participated in layout:
54px box + 12px `mt-3` = exactly the **+66px** measured. The card sits in
`positions/page.tsx:151` — `grid grid-cols-1 gap-4 lg:grid-cols-3 **items-stretch**` — so
the row grew to the tallest item and everything below moved. Out-of-flow children
contribute nothing to a grid item's intrinsic height, so card height is now
hover-invariant **without touching the grid** (the `items-stretch` is deliberate, per the
comment at `:137-144`).

**Deliberately not a portal and not the Popover API.** The research checked both: Tremor's
portaled tooltip escaped the card with white-on-dark styling and the operator **rejected**
it (cycle-69); the Popover API's top layer "spans the entire viewport and sits on top of
all other layers" — functionally the same thing. The card has **no `overflow-hidden`**
(`:163`), so plain `absolute` suffices and the whole "escaping a clipping ancestor"
question is moot. CSS anchor positioning was also rejected: several 2026 blog posts claim
Baseline support, but `api.webstatus.dev` returns `baseline.status: "limited"`,
`browser_implementations: null`.

### 2. Duplicate tooltip

Deleted the SVG `<title>` (`:244`). The browser rendered it as its own unstyled native OS
tooltip **simultaneously** with the styled `role="tooltip"` div — two tooltips per hover,
visible in the operator's screenshot.

### 3. Contradictory rounding

`:258` `pct.toFixed(0)` → `toFixed(1)`. It was the lone outlier — `:176`, `:242` and `:293`
already used one decimal — so the centre read `6%` beside a legend `5.6%`, ~8px apart.

### 4. Donut-hole hit-testing

`fill="transparent"` → `fill="none"` on the slices (`:224`) **and the track ring**
(`:210`), plus `pointerEvents="none"` on both centre `<text>` elements.

Under `pointer-events: visiblePainted`, an interior is hit-testable whenever fill is
anything **other than `none`** — and `transparent` is a *colour with alpha 0*, not `none`.
So every slice's full disc (r=15.915) was hover-active including the hole; all slices
overlapped there and the **last in document order** won. Since `data` is sorted DESCENDING
(`:132`), that was the **smallest** slice. Hovering dead centre asserted the wrong sector
*and* fired the reflow. Both values paint no interior (SVG 2 §painting: `none` = "No paint
is applied"), so this is **visually identical** — no screenshot churn.

### 5. WCAG SC 1.4.13 — coupled to defect 1, not optional

The comment at `:302` **claimed** compliance the code did not deliver:

- **Hoverable FAILED outright** — `onMouseLeave` unmounted synchronously, so a pointer
  could never travel from the slice onto the tooltip.
- **Dismissible passed only VACUOUSLY**, through the exception for content that "does not
  obscure or replace other content" — true *only while the tooltip was in flow*. **Moving
  it to an overlay revokes that exception**, so a real dismiss mechanism became required.

**A layout-only fix would therefore have introduced a second AA failure.** Added a 200ms
grace-timer close cancelled by the tooltip's own `onMouseEnter` (Hoverable), and a
**document-level** Escape listener gated on hover (Dismissible) — the original `onKeyDown`
at `:157-161` never fired on the mouse path because focus is elsewhere. The grace timer
also prevents a flicker loop the overlay would otherwise create against the legend rows.
`pointer-events: none` on the tooltip would stop the flicker too but breaks Hoverable by
construction — not used. The false comment was replaced with one describing what ships.

## Files changed

| File | Change |
|---|---|
| `PortfolioAllocationDonut.tsx` | 4 defect fixes + the WCAG mechanism + corrected comment |
| `PortfolioAllocationDonut.test.tsx` | 2 existing tests converted to guards, 14 new tests (14 → 28) |

## Verbatim verification output

```
$ cd frontend && npm test -- PortfolioAllocationDonut
  Test Files  1 passed (1)
       Tests  28 passed (28)

$ npx tsc --noEmit -p tsconfig.json
  (clean)
```

## Tests

jsdom **cannot measure geometry** — measured, not assumed:
`getBoundingClientRect` returns all zeros even with inline width/height, `offsetHeight` 0,
`elementFromPoint`/`isPointInFill` undefined. But `getComputedStyle` **does** resolve
classes from a test-injected `<style>`. So the no-shift invariant is pinned **structurally**
and the real geometry is measured in the Playwright live_check.

- **The headline invariant (class-independent):** the card root's direct element children
  are identical at rest and hovered. Pre-fix 3 → 4; now 3 → 3. No strings, no CSS.
- **Mechanism:** inject `.absolute{position:absolute}`, hover, assert computed
  `position === "absolute"`. Ships *alongside* the structural test, not instead — it
  couples to a literal class name.
- **Anti-portal guard:** the tooltip must be a DOM descendant of `[role="region"]` and must
  not sit under `[popover]`.
- **Hit-testing:** every `<circle>` has `fill="none"`; every centre `<text>` has
  `pointer-events="none"`.
- **Rounding:** scoped to the centre `<text>` deliberately. **Correction (cycle-2 Q/A
  finding 5):** the earlier blanket claim that an unscoped `container.textContent` assertion
  "passes vacuously" was **half-right, and the Q/A measured it**. The *positive* assertion
  (`toContain("56.0%")`) does pass unscoped, because the legend supplies that string — but
  the *negative* one (`not.toContain("56%")`) still fails, so an unscoped test would not be
  fully vacuous. The shipped test is correctly scoped either way; the rationale as written
  was too strong.
- **WCAG Hoverable:** entering the tooltip within the grace window cancels the close.

**Two existing tests were converted, not deleted:**
- *"each slice has a `<title>` child"* asserted `titles.length === 2` → **inverted to 0**,
  becoming defect 2's regression guard.
- *"tooltip dismisses on mouseleave"* asserted removal **synchronously** → the grace timer
  breaks it. Moved to fake timers, and it now asserts the tooltip **survives** the grace
  window before advancing past it. Asserting synchronous removal would assert the very
  behaviour that breaks Hoverable.

## Mutation matrix

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

## Scope honesty

- **One artifact is missing:** the live_check spec asks for hovered **and** unhovered
  screenshots; only the hovered PNG was captured before teardown. The **box measurements are
  complete for both states**, which is what criterion 1 itself specifies. Disclosed in
  `live_check_80.5.md` §E for Q/A to weigh rather than quietly omitted.
- No backend change; paper book untouched; no `.env`, no flag flips.
- Operator's `:8000` never restarted, `:3000` never driven (200/302/200, pid `70791`
  unchanged at teardown).
- The `items-stretch` grid at `positions/page.tsx:151` was **not** modified — the fix makes
  the card hover-invariant instead, which is what that layout needs.
