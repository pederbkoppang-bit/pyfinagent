---
name: frontend-test-env-and-ui-specs-80-5
description: Measured jsdom 29 capability limits for frontend tests + three UI-spec traps (anchor-positioning is NOT Baseline, popover==rejected portal, SVG fill=transparent is hit-testable)
metadata:
  type: project
---

Findings from the phase-80.5 research gate (2026-07-26, PortfolioAllocationDonut).
Brief: `handoff/current/research_brief_80.5.md`.

## jsdom capability probe -- MEASURED in-repo, not assumed

Ran a scratch vitest file against `frontend/vitest.config.ts` (jsdom 29.0.2 /
vitest 4.1.4 / @testing-library/react 16.3.2), then deleted it:

```
getBoundingClientRect            = {w:0,h:0,top:0,bottom:0}  # even with inline width/height
offsetHeight / clientHeight      = 0
document.elementFromPoint        = undefined
SVGGeometryElement.isPointInFill = undefined
getTotalLength                   = undefined
getComputedStyle(position) from a class rule in a test-injected <style>  = resolves correctly
getComputedStyle(position) default = "static"
```

**Why:** jsdom has no layout engine; the request to add `getBoundingClientRect`
is closed (jsdom#3621 -> #3729). But `getComputedStyle` DOES run the CSS cascade
over `<style>` elements the test appends.

**How to apply:** never design a frontend test around geometry, CLS, or
hit-testing -- those belong in the Playwright live_check. The strongest
layout-ish invariants jsdom CAN pin are (a) DOM structure (e.g. "the container's
direct element-child count is identical at rest and on hover" -- fully
class-independent and mutation-killable), and (b) computed style via an injected
`<style>` rule (stronger than a `className.includes()` string match, but still
couples to the literal class name -- pair it with (a), see
[[mutation-test-guards-and-fixtures]]).

## Three UI-spec traps

1. **CSS anchor positioning is NOT Baseline (as of 2026-07-26).**
   `api.webstatus.dev/v1/features/anchor-positioning` -> `baseline.status:
   "limited"`, `browser_implementations: null`. MDN BCD `main` ->
   Chrome 125 / **Firefox 147** / Safari 26. Multiple confident 2026 blog posts
   claim "Baseline 2026, 91%, Firefox 132+, Safari 18.2+" -- all wrong. Always
   check `api.webstatus.dev/v1/features/<id>` (curl; the HTML site is
   JS-rendered) + `raw.githubusercontent.com/mdn/browser-compat-data/main/...`
   before quoting support. See [[gcloud-docs-fetch]] for the same curl-not-
   WebFetch pattern.

2. **Popover API / top layer == the portal the operator rejected.** Popover IS
   Baseline (newly available 2025-01-27), so support is never the objection --
   but a shown popover is promoted to the top layer, which "spans the entire
   viewport and sits on top of all other layers". For any pyfinagent card-
   contained tooltip that is a hard stop: it re-creates the cycle-69 Tremor
   failure (tooltip escaped the card with white-on-dark styling).

3. **SVG `fill="transparent"` is hit-testable; `fill="none"` is not.** Under the
   default `pointer-events: visiblePainted`, MDN and SVG 2 both condition
   interior hit-testing on the fill being "a value other than `none`" --
   `transparent` is a colour with alpha 0, i.e. a value. Both paint nothing, so
   swapping is a zero-visual-change fix. Stroke hit-testing survives because
   SVG 2 §13.5.7 builds the *stroke shape* from the dash pattern (so a
   `stroke-dasharray` donut hit-tests per-arc) -- with the spec caveat that
   implementations "are given some leeway to deviate ... for performance".

## WCAG SC 1.4.13 coupling (generalizes beyond this component)

The Dismissible bullet has an exception: it does not apply when the additional
content "does not obscure or replace other content". An **in-flow** tooltip
therefore satisfies Dismissible *vacuously*. Converting it to an absolute
overlay (the standard layout-shift fix) **revokes that exception** and makes
Dismissible a real requirement. Any "remove the tooltip from flow" step must
ship the ESC handler in the same change.
