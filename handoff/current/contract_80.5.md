# Contract — phase-80.5

**Step id:** `80.5` (phase-80, priority **P1**, `harness_required: true`)
**Title:** *Hovering the Allocation donut pushes the whole page down* — operator-reported,
2026-07-25 screenshot pair.

## TIER (assigned explicitly, before GENERATE)

| field | value |
|---|---|
| **Tier** | **T2** |
| Model | Opus 5 (`opus` alias) |
| Effort | `high` (Main session runs `xhigh` per settings.json; Q/A gate at `max`) |
| Rationale | P1, non-money-surface. The goal assigns T3 to `80.7`/`80.8`/`80.9` (money surfaces) and **T2 to the rest of wave 4**. Not T1: multi-file, has a research gate, and the criteria are not mechanically checkable. |

Recorded here because the phase-80.4 tier ledger found that a tier never written down is a
tier that silently defaults. See `handoff/current/tier_ledger_2026-07-26.md`.

## Research gate

`handoff/current/research_brief_80.5.md` — **`gate_passed: true`**, 10 sources read in full
(floor 5), 38 URLs, recency scan performed, 3-query-variant discipline visible.

Findings that changed the plan, each re-verified by Main against source before use:

- **The card has NO `overflow-hidden`** (`:163` = `h-full flex flex-col rounded-xl border
  border-navy-700 bg-navy-800/70 p-4`). The entire "which technique escapes a clipping
  ancestor" question is moot — plain `absolute` inside `relative` suffices.
- **CSS anchor positioning: do NOT adopt.** Multiple 2026 blog posts claim Baseline/91%
  support; `api.webstatus.dev` returns `baseline.status: "limited"`,
  `browser_implementations: null`. It also still requires `position: absolute`, so it buys
  nothing here.
- **Popover API: HARD STOP.** It *is* Baseline, so support is not the objection — the top
  layer is. It "spans the entire viewport and sits on top of all other layers", i.e.
  functionally the portal the operator already rejected.
- **`fill="none"` vs `fill="transparent"` confirmed twice** (MDN + SVG 2 §interact): interior
  hit-testing is conditioned on fill being "other than `none`", and `transparent` is a
  colour. SVG 2 §painting: `none` = "No paint is applied", `transparent` paints alpha-0 —
  **identical pixels, zero visual change**. Hover still works because `visiblePainted` also
  targets the *stroke*, and §13.5.7 builds the stroke shape from the dash pattern.
- **Defects 1 and 3 are COUPLED.** `Dismissible` currently passes only *vacuously*, via the
  exception "does not obscure or replace other content" — true only because the tooltip is
  in-flow. **Moving it to an overlay revokes that exception**, so a layout-only fix would
  introduce a second WCAG AA failure. `Hoverable` already FAILS today.
- **jsdom cannot measure geometry at all** — measured, not assumed:
  `getBoundingClientRect` → all zeros even with inline width/height; `offsetHeight` 0;
  `elementFromPoint`/`isPointInFill` undefined. But `getComputedStyle` DOES resolve classes
  from a test-injected `<style>`. So the no-shift invariant must be pinned structurally.

## Hypothesis

The tooltip renders as an **in-flow sibling** of the card's flex column (`:305-320`), so it
participates in layout: 54px box + `mt-3` (12px) = exactly the +66px measured. The card sits
in `positions/page.tsx:151` = `grid grid-cols-1 gap-4 lg:grid-cols-3 **items-stretch**`, so
the whole row grows and everything below shifts — `Currency exposure` y=656 → 722.

Out-of-flow children contribute nothing to a grid item's intrinsic height, so making the
tooltip `absolute` inside a `relative` wrapper makes card height **hover-invariant** without
touching the grid.

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. `Hovering any donut slice (arc OR legend row) causes ZERO layout shift: the y coordinate of the 'Currency exposure' heading is IDENTICAL hovered vs not, measured with browser_snapshot boxes:true`
2. `The tooltip still renders inside the card border with the project's navy/slate tokens -- the rejected portaled/escaping-tooltip behaviour is NOT reintroduced`
3. `Exactly ONE tooltip appears per hover (the native SVG <title> duplicate is gone or the custom tooltip is removed -- not both showing)`
4. `The centre label and the legend show the SAME rounded percentage for the same slice`
5. `The component's existing tests still pass and a test pins the no-shift invariant; MUTATION-TEST it by restoring the in-flow tooltip and confirming the test fails`

**Verification command (immutable):**
```
cd frontend && npm test -- PortfolioAllocationDonut 2>&1 | tail -30
```

**live_check (immutable):** *handoff/current/live_check_80.5.md: Playwright box measurements of the 'Currency exposure' heading and the Allocation card in BOTH hover states proving identical geometry, plus hovered/unhovered screenshots.*

## Plan

1. `:197` chart-row wrapper `flex items-center gap-4 flex-1` → add **`relative`**. (NOT
   `:198`, which wraps only the 128px SVG.)
2. Move the tooltip **inside** that wrapper as `absolute inset-x-0 bottom-0 z-10`, dropping
   `mt-3`. Card-root direct children then stay **3 → 3** across hover (today 3 → 4).
3. `fill="transparent"` → `fill="none"` at **`:224`** (slices) **and `:210`** (track ring).
4. `pointer-events="none"` on the two centre `<text>` elements (`:249-270`) — they sit last
   in the SVG and intercept at every circle's bbox centre.
5. Delete the SVG `<title>` at `:244` (defect 2 — the native OS tooltip duplicate).
6. `:258` `pct.toFixed(0)` → `toFixed(1)` (defect 3). `:176`, `:242`, `:293` already use
   `toFixed(1)`; `:258` is the lone outlier.
7. **WCAG (coupled to step 2):** grace-timer close (~200ms) cancelled by the tooltip's own
   `onMouseEnter` → satisfies *Hoverable* and prevents the overlay/legend flicker loop;
   document-level `Escape` listener gated on hover → satisfies *Dismissible*, which the
   overlay now genuinely requires. The existing `onKeyDown` at `:157-161` never fires on the
   mouse path because focus is elsewhere. Do **not** put `pointer-events: none` on the
   tooltip — that would break *Hoverable*.
8. Update `:302`'s comment to match the delivered behaviour.

## Tests

- **T2 (headline, class-independent):** card-root direct element children identical at rest
  and hovered. Today 3→4; after fix 3→3; **under the criterion-5 mutation 3→4 = FAIL.**
- **T1:** inject `<style>.absolute{position:absolute}</style>`, hover, assert
  `getComputedStyle(tooltip).position === "absolute"`. Ships *alongside* T2, not instead —
  it couples to a literal class name.
- **T3 (anti-portal guard):** tooltip must be a DOM descendant of `[role="region"]`.
- Two existing tests must change, not be deleted:
  - `:200-214` asserts `titles.length === 2` → **invert to 0**, becoming defect 2's guard.
  - `:181-198` asserts synchronous removal on mouseleave → grace timer breaks it; move to
    fake timers / `waitFor`. **This is the easy-to-miss one.**
- Defect 3 test must scope to the centre `<text>` node — asserting on
  `container.textContent` passes vacuously because the legend already contains `75.0%`.

## Do-no-harm

Frontend-only; paper book untouched. No `.env`, no flag flips. UI evidence from the
isolated skip-auth `:3100` rig with its own `distDir` — never the operator's `:3000`;
restore `tsconfig.json` + `next-env.d.ts` after.

**HARD STOP:** reintroducing a portaled/escaping tooltip (operator-rejected, cycle-69), or
any change making the component less accessible than today.
