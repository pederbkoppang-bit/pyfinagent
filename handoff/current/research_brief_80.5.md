# Research Brief -- masterplan step 80.5 (PortfolioAllocationDonut, 4 defects)

Tier: **T2** (moderate depth, Opus 5 / effort high). Not audit-class (`coverage.audit_class = false`).
Date: 2026-07-26. Write-first: this file was created BEFORE any source was read and grown incrementally.

Questions (from caller):
- **A.** Layout-shift-free tooltip visually inside a bounded card (portal is operator-REJECTED).
- **B.** SVG `fill="none"` vs `fill="transparent"` hit-testing under `pointer-events: visiblePainted`.
- **C.** WCAG 2.2 SC 1.4.13 normative text; is the `:302` compliance claim true?
- **D.** How to pin a "hovering causes zero layout shift" invariant in **vitest + jsdom** (no Playwright runner).
- **E.** Internal inventory: component, tests, frontend rules, grid-row propagation.

---

## Search queries run (3-variant discipline, `.claude/rules/research-gate.md`)

| Variant | Query |
|---|---|
| Current-year (2026) | `CSS anchor positioning baseline browser support 2026 tooltip` |
| Last-2-year (2025) | `accessible tooltip hoverable dismissible persistent implementation grace period 2025` |
| Year-less canonical | `SVG circle fill none vs fill transparent hover hit testing donut chart` |
| Year-less canonical | `jsdom getBoundingClientRect returns zeros not implemented layout` |
| Year-less canonical | `tooltip layout shift absolute positioning inside card reserve space CLS` |
| Year-less canonical | `SVG stroke-dasharray hit testing pointer-events dashes gaps not hit-testable spec` |

---

## Read in full (10; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_anchor_positioning/Using | 2026-07-26 | official doc (MDN) | WebFetch | Anchored element "needs to be absolutely or fixed positioned". Popover API creates an *implicit* anchor reference. Page is silent on overflow-clipping escape. |
| 2 | https://developer.mozilla.org/en-US/docs/Web/CSS/pointer-events | 2026-07-26 | official doc (MDN) | WebFetch | **Defect 4 CONFIRMED.** `visiblePainted` targets the interior "**and the `fill` property is set to a value other than `none`**". `transparent` is a colour, not `none`. |
| 3 | https://www.w3.org/WAI/WCAG22/Understanding/content-on-hover-or-focus.html | 2026-07-26 | normative + W3C WG note | WebFetch | Verbatim SC text (3 bullets + exception) -- see §C. Hoverable intent: "the pointer can be moved directly from the trigger onto the new content" without it disappearing. |
| 4 | https://svgwg.org/svg2-draft/interact.html | 2026-07-26 | spec (SVG 2) | WebFetch | `pointer-events` initial value = `auto`; applies to "container elements, graphics elements and 'use'"; **inherited: yes**. `visiblePainted` prose confirms the fill!=none condition. Text hit-tests "on a character cell basis". |
| 5 | https://svgwg.org/svg2-draft/painting.html | 2026-07-26 | spec (SVG 2) | curl + tag-strip (WebFetch truncated the page) | `<paint>` value `none` = "**No paint is applied in this layer.**" §13.5.7: "The **stroke shape** of an element is the shape that is filled by the `stroke` property" and its algorithm consumes the dash pattern -- so a dashed stroke's *gaps* are outside the stroke shape. |
| 6 | https://developer.mozilla.org/en-US/docs/Web/API/Popover_API/Using | 2026-07-26 | official doc (MDN) | WebFetch | Shown popover "is put into the **top layer** so it will sit on top of all other page content". `manual` cannot be light-dismissed; `auto`/`hint` can. |
| 7 | https://developer.mozilla.org/en-US/docs/Glossary/Top_layer | 2026-07-26 | official doc (MDN) | WebFetch | Top layer "spans the entire width and height of the viewport and sits on top of all other layers"; promoted elements "generate a new stacking context". |
| 8 | https://github.com/jsdom/jsdom/issues/3621 | 2026-07-26 | project issue tracker | WebFetch | "Implement getBoundingClientRect" -- **closed as duplicate of #3729**; still unimplemented. A full implementation "requires complex layout calculations". |
| 9 | https://api.webstatus.dev/v1/features/anchor-positioning + /features/popover | 2026-07-26 | official Baseline data (W3C WebDX) | curl (JSON) | **anchor-positioning baseline status = `"limited"`, `browser_implementations: null`** -- i.e. NOT Baseline. Popover = `"newly"`, `low_date 2025-01-27`. |
| 10 | https://raw.githubusercontent.com/mdn/browser-compat-data/main/css/properties/anchor-name.json | 2026-07-26 | official compat data (MDN BCD, `main`) | curl (JSON) | `anchor-name`: Chrome 125, **Firefox 147**, Safari 26, Edge/iOS mirror. |

Sources 5, 9, 10 were fetched with `curl` + tag-strip / JSON parse because the pages are truncated or JS-rendered under `WebFetch`; the full text was extracted and read (per the `feedback_gcloud_docs_fetch` precedent). They count as read-in-full.

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.nexgismo.com/blog/css-anchor-positioning-replace-javascript-tooltip-library-2026 | blog | **Claims contradicted by source #9/#10** -- see "Contradiction" below |
| https://pockit.tools/blog/css-anchor-positioning-api-complete-guide/ | blog | same false-Baseline claim |
| https://ultimatedesigntools.com/blog/how-to-use-css-anchor-positioning/ | blog | same |
| https://brainstormsandraves.com/css/css-anchor-positioning/ | blog | superseded by MDN + BCD |
| https://dev.to/parsajiravand/youre-using-floating-ui-to-position-your-tooltip-the-browser-does-it-natively-now-g00 | blog | superseded |
| https://here.luisavasquez.com/thoughts/svg-fill-none-transparent | blog | corroborates #2/#5; community tier |
| https://css-tricks.com/building-a-donut-chart-with-vue-and-svg/ | blog | prior-art donut pattern (year-less) |
| https://dev.to/mustapha/how-to-create-an-interactive-svg-donut-chart-using-angular-19eo | blog | prior-art interactive donut |
| https://akzhy.com/blog/create-animated-donut-chart-using-svg-and-javascript/ | blog | prior-art |
| https://css-tricks.com/almanac/properties/p/pointer-events/ | blog | MDN + SVG2 are authoritative |
| https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/stroke-dasharray | doc | covered by #5 |
| https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/fill-opacity | doc | covered by #5 |
| https://wiki.allizom.org/SVG:Pointer-events | implementer wiki | **HTTP 502** |
| http://dschulze.com/blog/articles/7/faster-hit-testing-in-svg | SVG-WG member blog | **TLS handshake failure** (SNI) |
| https://www.w3.org/Graphics/SVG/WG/track/actions/3279 | WG tracker | tracker stub only |
| https://github.com/jsdom/jsdom/issues/3729 | issue | canonical dup target of #8 |
| https://github.com/jsdom/jsdom/issues/3178 | issue | Range.getBoundingClientRect, off-topic |
| https://github.com/jsdom/jsdom/issues/1504 | issue | older dup |
| https://github.com/capricorn86/happy-dom/issues/1416 | issue | different DOM impl (not our runner) |
| https://www.ditdot.hr/en/debugging-cls-cumulative-layout-shift | blog | CLS background |
| https://css-tricks.com/fixed-height-cards-more-fragile-than-they-look/ | blog | argues AGAINST option (ii) |
| https://speedvitals.com/blog/reserve-space-prevent-layout-shift/ | blog | reserve-space technique |
| https://testparty.ai/blog/wcag-1-4-13-content-on-hover-or-focus-2025-guide | blog (2025) | recency scan |
| https://blog.greeden.me/en/2025/05/14/accessible-tooltip-design-guide-based-on-wcag-1-4-13-a-practical-approach/ | blog (2025) | recency scan |
| https://www.alephaccessibility.net/resources/accessible-tooltips | practitioner | corroborates ESC + aria-describedby |
| https://accessiblyapp.com/blog/tooltip-accessibility/ | practitioner | corroborates |
| https://accessibleweb.com/question-answer/why-is-tooltip-text-required-to-be-hoverable/ | practitioner | Hoverable rationale |
| https://www.wcag.com/authors/1-4-13-content-on-hover-or-focus/ | practitioner | corroborates |

**URLs collected: 38** (10 read in full + 28 snippet-only).

---

## Recency scan (last 2 years, 2024-2026)

Performed. **Two findings that change the plan, one that does not.**

1. **CHANGES THE PLAN -- CSS anchor positioning is NOT Baseline, despite a wave of 2026 blog posts saying it is.** Multiple 2026 posts (nexgismo, pockit, ultimatedesigntools) assert "Baseline 2026 ... 91% ... Chrome 125+, Firefox 132+, Safari 18.2+". The official W3C WebDX Baseline feed disagrees: `api.webstatus.dev/v1/features/anchor-positioning` returns `baseline.status = "limited"` with `browser_implementations: null` (accessed 2026-07-26). MDN BCD `main` gives the real per-engine numbers: Chrome 125, **Firefox 147**, Safari 26 -- not Firefox 132 / Safari 18.2. Firefox 147 is a ~2026 release, so the cross-engine window is months old, not years. **Do not adopt anchor positioning on blog authority.**
2. **CHANGES THE PLAN -- Popover API is Baseline "newly available" since 2025-01-27** (`api.webstatus.dev/v1/features/popover`), i.e. it is now genuinely shippable. That makes option A(iv) *technically* available -- which is precisely why it must be explicitly ruled out on product grounds (§A(iv)), not on support grounds.
3. **NO CHANGE** -- 2025 SC 1.4.13 practitioner guidance (TestParty 2025 guide, greeden 2025-05-14, Aleph) is unanimous with the 2018-era W3C Understanding doc: ESC-dismiss + hoverable bridge + no auto-timeout. No newer normative text; WCAG 2.2 kept 1.4.13 unchanged from 2.1. SVG hit-testing semantics are unchanged since SVG 1.1.

---

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `frontend/src/components/PortfolioAllocationDonut.tsx` | 323 | the component under repair | 4 defects confirmed by reading |
| `frontend/src/components/PortfolioAllocationDonut.test.tsx` | 215 | the ONLY test file for it | 15 tests; **2 will break** (see below) |
| `frontend/src/app/paper-trading/positions/page.tsx` | :163 | the ONLY consumer (grep-verified) | passes `slices/totalNav/liveBand/liveAgeSec/title` |
| `frontend/src/app/paper-trading/positions/page.tsx` | :151 | the grid row | `grid grid-cols-1 gap-4 lg:grid-cols-3 items-stretch` |
| `frontend/src/app/paper-trading/positions/page.tsx` | :137-144 | comment justifying `items-stretch` | deliberate (operator-flagged 2026-05-26) |
| `frontend/src/app/paper-trading/positions/page.tsx` | :177 | `MultiCurrencyNavBreakdown` "Currency exposure" | the element the caller measured shifting 656 -> 722 |
| `.claude/rules/frontend.md` | rule 3 | JIT-safe literal classes | cites this component's `DOT_BG_CLASS` as **the canonical example** |
| `.claude/rules/frontend-layout.md` | §4.5 | equal-height grid rows | forbids mixing short+tall at `items-stretch` |
| `frontend/vitest.config.ts` | -- | `environment: "jsdom"`, `globals: true` | no Playwright runner present |
| `frontend/vitest.setup.ts` | -- | jest-dom matchers + ResizeObserver shim | no geometry shim |

**Grid-row propagation: YES, confirmed.** `positions/page.tsx:151` is `lg:grid-cols-3 items-stretch`. CSS grid row height = the tallest item, so the donut card growing 215 -> 281px stretches `RiskMonitorCard` and `SectorBarList` too and pushes `MultiCurrencyNavBreakdown` (:177) down. That exactly reproduces the caller's measured 656 -> 722 shift. The `items-stretch` is intentional (`:137-144`) -- **do not touch the grid**; making the donut card's height hover-invariant is the correct fix and it is what `items-stretch` needs.

**`DOT_BG_CLASS`: yes, already present** (`:64-81`, consumed at `:276`), alongside `SLICE_STROKE_CLASS` (`:83-100`, consumed at `:215`). Any class the fix adds must be a **literal** string (`absolute`, `inset-x-0`, `bottom-0`, `z-10` all qualify) -- never a template concatenation.

### Existing tests a fix will break

| Test | Line | Breaks because |
|---|---|---|
| `"each slice has a <title> child for native SVG hover tooltips"` | :200-214 | Asserts `titles.length === 2`. **Defect 2's fix deletes `<title>`** -> fails. Must be INVERTED to assert zero `<title>` under the SVG; the inverted test then becomes the regression guard for defect 2. |
| `"tooltip dismisses on mouseleave"` | :181-198 | Asserts the tooltip is gone **synchronously** after `fireEvent.mouseLeave`. If the SC 1.4.13 "Hoverable" fix adds a close **grace timer** (§C), the tooltip is still mounted at that instant -> fails. Must move to `vi.useFakeTimers()` + `vi.advanceTimersByTime(...)`, or `await waitFor(...)`. **This is the easy-to-miss one.** |

Survive unchanged: `:112-128` (circle count 4 -- `fill` value does not change node count), `:130-144` (stroke classes), `:146-157` (no tooltip at rest), `:159-179` (tooltip content on hover), and all eight pre-cycle-70 tests.

### Measured jsdom capability probe (not asserted -- run in-repo, then deleted)

Ran a scratch vitest file against the project's own config. **jsdom 29.0.2 / vitest 4.1.4 / @testing-library/react 16.3.2**:

```
gBCR                                  = {"w":0,"h":0,"top":0,"bottom":0}   # even with inline width:300/height:200
offsetHeight                          = 0
clientHeight                          = 0
computed.position(inline)             = "absolute"
computed.position(class via <style>)  = "absolute"     # <-- KEY
computed.marginTop(class via <style>) = "0.75rem"
computed.position(default)            = "static"
svg circle gBCR.width                 = 0
typeof isPointInFill                  = "undefined"
typeof getTotalLength                 = "undefined"
elementFromPoint                      = "undefined"
```

Consequences: **no geometry assertion of any kind is possible** (every rect is 0), **no hit-test simulation is possible** (`elementFromPoint`, `isPointInFill` absent) -- but **`getComputedStyle` DOES resolve a class rule from a `<style>` element injected by the test**. That single fact is what makes a real (non-string-matching) out-of-flow assertion possible in jsdom.

---

## A. Layout-shift-free tooltip inside a bounded card -- RANKED

First, a fact that removes a constraint: **the card has no `overflow-hidden`.** `containerClass` (`:163`) is `h-full flex flex-col rounded-xl border border-navy-700 bg-navy-800/70 p-4`. There is no clipping ancestor to escape, so the entire "which technique escapes `overflow: hidden`" axis is **moot for this component**. Anchor positioning and the top layer exist to solve a problem this card does not have.

Second: `items-stretch` (E) means the card is a **grid item**, and a grid item establishes a containing block for `position: absolute` descendants as soon as it is `position: relative`. Grid stretch sizing is unaffected by out-of-flow children -- an abspos child contributes nothing to intrinsic height, which is exactly the property we want.

### #1 (RECOMMEND) -- `position: absolute` inside a `position: relative` ancestor within the card

Add `relative` to the chart-row wrapper at `:197` and render the tooltip as its last child with `absolute inset-x-0 bottom-0`.

- Escapes `overflow-hidden`? No -- irrelevant here (none exists). If one is ever added, the tooltip clips to the card, which is the *desired* behaviour ("visually inside the card border").
- Works inside a `items-stretch` grid row? **Yes** -- out-of-flow children do not contribute to the grid item's content height, so the row height stops depending on hover.
- Support: universal (CSS2.1 positioning).
- Reintroduces the rejected portal? **No** -- the node stays a DOM descendant of the card and inherits the project's navy/slate tokens.
- Trade-off: the tooltip **overlays** the bottom of the chart row (donut lower arc + last legend rows). That is normal chart-tooltip behaviour, but it has two consequences: it triggers the real SC 1.4.13 *Dismissible* obligation (§C), and it creates a **hover-flicker hazard** because the legend `<li>`s at `:281-282` also set `hoverIdx` -- tooltip covers an `<li>` -> `li` fires `mouseleave` -> tooltip unmounts -> `li` fires `mouseenter` -> loop. Both are solved by the *same* grace-timer + tooltip-`onMouseEnter` mechanism the Hoverable fix requires (§C). Do not paper over it with `pointer-events: none`, which would break Hoverable.

Why the chart-row wrapper (`:197`) rather than the card root: it keeps the card root's direct-element-child count **constant** across hover, which is the strongest mutation-killable jsdom invariant available (§D, T2). Putting the tooltip directly on the card root works visually but forfeits that test.

Note `:198` already has a `relative` div, but it wraps only the 128px SVG -- too small a containing block. Use `:197` (`flex items-center gap-4 flex-1`), adding `relative`.

### #2 -- permanently reserve the tooltip's height

Always render the tooltip box with a fixed min-height; show placeholder/empty content at rest.

- Zero shift by construction; simplest to reason about; no overlay, so no flicker and no new Dismissible obligation.
- **But** it costs +66px permanently, and because the row is `items-stretch`, that 66px of dead space is inflicted on **all three cards** in the row. `frontend-layout.md` §4.5 explicitly names dead whitespace in equal-height rows as the anti-pattern this page was already fixed for. CSS-Tricks "Fixed-Height Cards: More Fragile Than They Look" argues the same. Ranked below #1 for that reason, but it is the zero-risk fallback if the overlay is judged visually unacceptable.

### #3 -- CSS anchor positioning (`anchor-name` / `position-anchor` / `position-area`)

- **Not Baseline.** `api.webstatus.dev` reports `"limited"` (2026-07-26). MDN BCD: Chrome 125 / Firefox 147 / Safari 26 -- cross-engine but recent. The widely-circulated "Baseline 2026, Firefox 132+, Safari 18.2+" figure in 2026 blog posts is **wrong**.
- The anchored element still must be `absolute`/`fixed` (MDN #1), so it buys nothing over #1 unless you need to escape a clipping ancestor -- which this card does not have.
- Recommendation: **do not adopt.** Cost with no benefit here.

### #4 -- Popover API + `::backdrop` -- **HARD STOP**

- Popover *is* Baseline newly-available (2025-01-27), so the objection is not support.
- The objection is that showing a popover "puts it into the **top layer**", which "spans the entire width and height of the viewport and sits on top of all other layers" (MDN #6, #7). **This is functionally identical to the portal the operator rejected**: the element leaves the card's visual containment, renders above all page content, and picks up UA default popover styling (border + background + `inset: 0` centring) that must be fully overridden -- which is exactly the "escaped the card with white-on-dark styling" failure recorded in the component header at `:5-10` and re-stated at `:300-304`.
- `::backdrop` would additionally dim the whole viewport behind the tooltip. Wholly wrong for a hover affordance.
- **Flagged per the caller's instruction: adopting popover/top-layer here re-creates the rejected behaviour. Do not.**

---

## B. The two-character fix -- `fill="transparent"` -> `fill="none"`

**Confirmed by spec, twice, and there is no visual change.**

- Hit-testing. MDN `pointer-events` (#2), `visiblePainted`: the element is a target "when the `visibility` property is set to `visible` and e.g., when a mouse cursor is over the interior (i.e., 'fill') of the element **and the `fill` property is set to a value other than `none`**, or when a mouse cursor is over the perimeter (i.e., 'stroke') of the element and the `stroke` property is set to a value other than `none`." SVG 2 §interact (#4) says the same normatively: "it is over the interior (i.e., fill) of the element and the fill property **has an actual value other than none**". `transparent` is a `<color>` with alpha 0 -- a value, not `none`. So today every slice's full disc (r = 15.915, `:115`) is hit-testable including the hole, all slices overlap, and the last in document order wins; `data` is sorted DESC (`:132`), so that is the **smallest** slice. Caller's diagnosis is exactly right.
- No visual change. SVG 2 §painting (#5) defines the paint value `none` as "**No paint is applied in this layer.**" `transparent` paints alpha-0, i.e. also nothing observable. Identical pixels; only the hit region changes.
- **Hover still works after the fix** -- and this is the part worth checking before shipping. `visiblePainted` also targets the **stroke**, and the slices set `stroke` via `SLICE_STROKE_CLASS` (`:215`, e.g. `stroke-blue-500`), which is "a value other than `none`". Per SVG 2 §13.5.7 (#5), "the **stroke shape** of an element is the shape that is filled by the `stroke` property", and the stroke-shape algorithm consumes the dash pattern -- so with `stroke-dasharray` (`:228`) each slice's hit region collapses to **its own arc segment**, not the whole ring. That is precisely the desired behaviour: hover the blue arc, get the blue slice. Caveat from the same spec section: "implementations are given some leeway to deviate from this description for performance reasons" -- so per-dash arc hit-testing should be confirmed in the Playwright live_check, not assumed.
- Also fix the **track ring at `:210`** (same `fill="transparent"`). It is painted first, so it currently sits under every slice, but it is still a full hit-testable disc and it has no handlers -- leave it as `transparent` and it silently swallows nothing, change it to `none` and it is unambiguously inert. Change it; it is free.

### `<text>` non-interactivity

`pointer-events="none"` on the two centre `<text>` elements (`:249-270`) is the correct mechanism. Per SVG 2 §interact (#4) `pointer-events` **applies to graphics elements and is inherited**, and `none` means "The given element does not receive pointer events." Note the MDN caveat (#2): with `none` on a parent "its subtree could be kept targetable by setting `pointer-events` to some other value" -- not a concern here, `<text>` has no children.

**Accessibility-tree impact: none.** `pointer-events` is a hit-testing/paint property; it has no ARIA or accessibility-tree semantics (unlike `display:none`, `visibility:hidden`, or `aria-hidden`). The centre `<text>` nodes remain in the accessibility tree. *(Confidence: high, but this is an absence-of-statement inference -- neither MDN #2 nor SVG 2 #4 mentions the a11y tree at all. If Q/A wants a positive citation, that is the gap.)* In practice it does not matter for this component: the SVG carries `role="img"` + a full `aria-label` (`:202-203`), which makes its subtree a leaf for AT regardless.

---

## C. WCAG 2.2 SC 1.4.13 -- the `:302` claim is **NOT** true, and the layout fix makes it worse

### Normative text, verbatim (W3C, #3)

> Where receiving and then removing pointer hover or keyboard focus triggers additional content to become visible and then hidden, the following are true:
>
> **Dismissible** - A mechanism is available to dismiss the additional content without moving pointer hover or keyboard focus, unless the additional content communicates an input error or does not obscure or replace other content;
>
> **Hoverable** - If pointer hover can trigger the additional content, then the pointer can be moved over the additional content without the additional content disappearing;
>
> **Persistent** - The additional content remains visible until the hover or focus trigger is removed, the user dismisses it, or its information is no longer valid.
>
> Exception: The visual presentation of the additional content is controlled by the user agent and is not modified by the author.

### Audit of the current component

| Requirement | Current state | Verdict |
|---|---|---|
| **Persistent** | Stays until `mouseleave`/`blur`. | **PASS** |
| **Hoverable** | `onMouseLeave` on the circle (`:237`) unmounts immediately, and the tooltip is a *separate* box below the chart. Moving the pointer toward it leaves the circle first -> it vanishes. W3C intent: "the pointer can be moved **directly from the trigger onto the new content**". | **FAIL** -- and it fails today, before any change. |
| **Dismissible** | ESC is handled (`:157-161`) but only via `onKeyDown` on the container, which fires only when focus is inside it. On the **mouse** path focus is elsewhere, so ESC never reaches it. **However**, the exception applies: the tooltip is in-flow and therefore "does not obscure or replace other content". | **Vacuously PASS today** |

So `:302`'s claim ("hoverable via mouseenter; dismissible via Escape ...; persistent") is wrong on Hoverable, and right on Dismissible only by accident of the very in-flow layout that defect 1 removes.

**The critical interaction the fix must not miss:** moving the tooltip to an absolute overlay (§A #1) makes it obscure other content, which **revokes the Dismissible exception**. A fix that only addresses layout would take Dismissible from vacuously-passing to genuinely-failing. Defect 1 and SC 1.4.13 are coupled; they must be fixed in the same change.

### Which is cheaper -- fix the claim or fix the behaviour?

**Fix the behaviour.** Fixing only the comment is cheaper in lines but leaves a real WCAG AA failure (Hoverable) *and* introduces a second one (Dismissible) as a side effect of defect 1's fix. The project's contrast targets already aim at AAA (`.claude/rules/frontend.md` rule 6), so shipping a knowing AA regression is inconsistent.

Minimal correct pattern (~25-35 lines):

1. **Grace timer.** Replace `onMouseLeave={() => setHoverIdx(null)}` with a scheduled close (`setTimeout`, ~150-250ms into a ref). Give the tooltip `onMouseEnter` = cancel the pending close, `onMouseLeave` = schedule it. That single mechanism satisfies **Hoverable** *and* kills the legend-overlay flicker hazard from §A #1. No normative grace-period value exists (the W3C Understanding doc specifies none); 150-250ms is the practitioner consensus. Clear the timer on unmount.
2. **Real ESC.** Attach a `document`-level `keydown` listener in a `useEffect` gated on `hovered !== null`, instead of (or in addition to) the container `onKeyDown` at `:157-161`. That makes **Dismissible** hold on the mouse path once the tooltip starts obscuring content.
3. Narrow the `:300-304` comment to describe what is actually implemented, and drop the blanket "per WCAG SC 1.4.13" phrasing in favour of naming the three bullets it satisfies.

Do **not** add an auto-hide timeout -- that breaks Persistent (2025 practitioner guidance is unanimous; W3C: content remains "until the hover or focus trigger is removed, the user dismisses it, or its information is no longer valid").

---

## D. Pinning "hovering causes zero layout shift" -- what jsdom CAN and CANNOT do

**Cannot** (measured above, not assumed): any geometry. `getBoundingClientRect` returns all zeros even for explicitly-sized elements; `offsetHeight`/`clientHeight` are 0; `elementFromPoint` and `isPointInFill` do not exist. jsdom has no layout engine and the request to add one is closed (#8 -> jsdom#3729). CLS via `PerformanceObserver` is doubly unavailable (no layout, and layout-shift entries are a browser-only source). **Playwright `boundingBox()` is the only way to assert real pixels, and this repo has no Playwright test runner** -- Playwright exists here solely as an MCP for manual capture.

**Can**: `getComputedStyle` resolving class rules from a test-injected `<style>` element (measured: `.absolute{position:absolute}` -> `"absolute"`; default -> `"static"`), and full DOM structure.

### Recommended jsdom test set -- three invariants, all mutation-killable

Mutation to beat (criterion 5): restore `{hovered && <div className="mt-3 ...">}` as a direct in-flow child of the card root.

- **T2 -- the headline invariant (class-independent).** Snapshot the card root's direct **element** children at rest, hover a slice, snapshot again, assert identical length and identical node identity.
  - Today: 3 at rest (`:188` header div, `:194` `<p>`, `:197` chart row) -> **4** on hover. Under the recommended fix the tooltip nests inside the chart row, so it stays **3 -> 3**.
  - Under the mutation: 3 -> 4. **FAILS.** No string matching, no CSS -- pure DOM structure. This is the strongest available and the one to lead with.
- **T1 -- out-of-flow (computed style).** In the test, `document.head.appendChild(<style>.absolute{position:absolute}.relative{position:relative}</style>)`; hover; assert `getComputedStyle(tooltip).position === "absolute"`.
  - Under the mutation the tooltip has `mt-3` and no `absolute` -> resolves to `"static"`. **FAILS.**
  - Honest caveat for the contract: this does couple the test to the literal class name `absolute`. It is stronger than a `className.includes("absolute")` string match (it goes through the CSS cascade and would also catch an override), but it is not fully class-independent. Ship it *alongside* T2, not instead of it.
- **T3 -- anti-portal regression guard.** Assert the tooltip is a DOM descendant of the `[role="region"]` card root (`container.querySelector('[role="region"]').contains(tooltip)`), and that `document.body`'s direct children gained nothing on hover.
  - Under a portal/popover regression (the operator-rejected shape) the node is not inside the region. **FAILS.** This is the guard that keeps §A #4 from creeping back in.

Optionally **T0** for defect 3: hover, then scope to the centre `<text>` node specifically and assert its content is `"75.0%"` not `"75%"`. Do **not** assert on `container.textContent` -- the legend already contains `"75.0%"` (`:293`), so an unscoped assertion passes under the mutation and is a fake guard.

### Must be left to the Playwright live_check

1. **Real geometry**: donut card `boundingBox()` height identical at rest and on hover (215 -> 215, per the caller's baseline), and the "Currency exposure" heading's `y` unchanged (656 -> 656, i.e. `MultiCurrencyNavBreakdown` at `positions/page.tsx:177` does not move). This is the criterion-1 evidence and jsdom provably cannot produce it.
2. **Hit-testing correctness** (defect 4): `mouse.move()` onto the largest arc and assert the tooltip names the largest slice; move into the donut hole and assert no tooltip. jsdom has neither `elementFromPoint` nor `isPointInFill`, so there is no unit-test substitute. This also empirically settles the per-dash stroke hit-testing question flagged in §B.
3. **No duplicate native tooltip** (defect 2) -- an OS-rendered `title` tooltip is not in the DOM at all; a screenshot after a hover dwell is the only evidence. (The DOM half -- zero `<title>` elements -- is unit-testable via the inverted `:200-214` test.)
4. Visual confirmation that the overlay tooltip stays inside the card border and reads correctly on navy (`.claude/rules/frontend.md` rule 5 makes visual verification mandatory for chart work).

---

## Consensus vs debate

- **Consensus**: absolute positioning is the standard answer for tooltip-induced layout shift ("Overlay small dynamic content with absolute positioning ... instead of shifting them"); `fill="none"` vs `fill="transparent"` hit-testing is settled and identical across MDN, SVG 2, and practitioner write-ups; SC 1.4.13 guidance has not moved in 2024-2026.
- **Debate / contradiction**: the 2026 blog wave on CSS anchor positioning states a Baseline status the official W3C WebDX feed contradicts. Resolved in favour of `api.webstatus.dev` + MDN BCD (source hierarchy: official data > blogs). Flagging it because three separate 2026 posts repeat the same wrong numbers, so it is a live misinformation trap for this step.
- **Debate**: reserve-space vs overlay. CSS-Tricks argues fixed-height cards are fragile; the CLS literature prefers overlay for small dynamic content. Both agree the in-flow tooltip is wrong.

---

## Pitfalls (ranked, for the contract)

1. Fixing defect 1 **without** fixing SC 1.4.13 Dismissible -- the overlay revokes the "does not obscure" exception (§C). Coupled defects.
2. The **legend hover-flicker loop** the overlay creates against `:281-282` -- solved by the same grace timer as Hoverable, but it must be deliberate.
3. `"tooltip dismisses on mouseleave"` (`:181-198`) **silently breaking** under a grace timer -- must move to fake timers / `waitFor`.
4. The `<title>` test (`:200-214`) breaking -- invert it rather than delete it.
5. Asserting defect 3 against `container.textContent` -- passes under the mutation; scope to the `<text>` node.
6. Reaching for popover / anchor positioning -- re-creates the rejected portal (#4) or adopts a non-Baseline feature for no benefit (#3).
7. Template-string Tailwind classes -- `.claude/rules/frontend.md` rule 3 names this very file as the canonical counter-example.
8. Touching `items-stretch` at `positions/page.tsx:151` -- it is deliberate (`:137-144`); the fix is to make the card's height hover-invariant, not to change the grid.

## Application to pyfinagent -- change sites

| Defect | Site | Change |
|---|---|---|
| 1 layout shift | `PortfolioAllocationDonut.tsx:197` and `:305-320` | add `relative` to the chart-row wrapper; move the tooltip inside it; swap `mt-3` for `absolute inset-x-0 bottom-0 z-10` (literal classes) |
| 1 (coupled) | `:157-161`, `:236-239` | document-level ESC while open; grace-timer close + tooltip `onMouseEnter`/`onMouseLeave` |
| 2 duplicate tooltip | `:244` | delete the `<circle>` `<title>`; `aria-label` at `:242` already carries the same string |
| 3 rounding | `:258` | `toFixed(0)` -> `toFixed(1)` (`:176`, `:242`, `:293` already use `toFixed(1)`; `:258` is the lone outlier) |
| 4 hit-testing | `:224` (slices) and `:210` (track) | `fill="transparent"` -> `fill="none"` |
| 4 (centre text) | `:249-270` | `pointerEvents="none"` on both `<text>` elements |
| doc | `:300-304` | rewrite the comment to match implemented behaviour |
| tests | `PortfolioAllocationDonut.test.tsx:181-198`, `:200-214` | update per §E; add T1/T2/T3 (+T0) per §D |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (10)
- [x] 10+ unique URLs total (38)
- [x] Recency scan (2024-2026) performed + reported (2 plan-changing findings)
- [x] Full pages/specs read, not abstracts (SVG 2 chapters extracted via curl where WebFetch truncated)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered the component, its test file, its sole consumer, both frontend rule files, and the vitest config
- [x] Contradictions noted (anchor-positioning Baseline: blogs vs official feed)
- [x] Claims cited per-claim
- [x] jsdom capability claims **measured in-repo**, not asserted

---

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 28,
  "urls_collected": 38,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "All four defects confirmed against spec. Defect 4 is real: MDN + SVG 2 both require fill != none for interior hit-testing, and 'transparent' is a colour, so fill=\"none\" at :224/:210 is a no-visual-change fix; stroke hit-testing survives because SVG 2 13.5.7 builds the stroke shape from the dash pattern. Recommended tooltip fix is plain absolute-in-relative inside the chart row at :197 -- the card has NO overflow-hidden, so anchor positioning (officially Baseline 'limited', NOT the 'Baseline 2026' that 2026 blogs claim) buys nothing, and popover/top-layer is a HARD STOP because it re-creates the operator-rejected portal. Critical coupling: the overlay revokes SC 1.4.13's 'does not obscure' exception, so Dismissible becomes genuinely required; Hoverable already FAILS today, so the :302 claim is false. One grace-timer fixes Hoverable and the legend-overlay flicker together. jsdom measured in-repo: all rects 0, no elementFromPoint -- but getComputedStyle resolves injected <style> class rules, enabling three mutation-killable invariants (card-root child-count stability being the strongest). Two existing tests break: :200-214 and :181-198.",
  "brief_path": "handoff/current/research_brief_80.5.md",
  "gate_passed": true
}
```
