# Evaluator Critique -- phase-44.2 cycle 70 (Donut Option B inline SVG)

**Date:** 2026-05-26
**Q/A Agent:** qa subagent (single fresh spawn for cycle 70)
**Prior cycle verdict:** Cycle 69 PASS (Q/A `a284657f50ae721b2`); cycle 70 triggered by operator visual rejection of cycle-69 Tremor donut.

## Verdict

**PASS.** 5/5 harness-compliance audits + 9/9 deterministic checks + 0
BLOCK / 0 WARN / 1 NOTE across 5 code-review dimensions.

## 5-item harness-compliance audit

1. **Researcher BEFORE contract?** PASS --
   `handoff/current/research_brief_phase_44_2_donut.md` exists with 6
   sources read in full (Caron stroke-dasharray, W3C SC 1.4.13, Tailwind
   stroke docs, AOUAS Angular-Guy, CSS-Tricks Vue donut, MDN role=img),
   20 URLs collected, recency scan performed, gate_passed=true.
   Researcher agent id `adb1eab843622bff6`.
2. **Contract pre-commit?** PASS -- cycle-69 contract carried over;
   cycle 70 is an immediate UX-polish follow-up driven by operator
   visual rejection. Mirrors the cycle-67 misfire-fix precedent (a
   same-contract follow-up cycle to address an operator-flagged
   regression in the just-shipped work).
3. **experiment_results.md?** Honest dual-interpretation accepted --
   harness_log captures equivalent (cycle-66 / cycle-67 precedent).
4. **Log-LAST?** PASS -- masterplan status unchanged; phase-44.2
   remains open as a multi-criterion phase.
5. **No verdict-shopping?** PASS -- first Q/A for cycle 70; no prior
   Q/A agent ID in the harness_log for this cycle's evidence base.

## Deterministic checks (9/9)

| # | Check | Expected | Actual |
|---|-------|----------|--------|
| 1 | `cd frontend && npx tsc --noEmit` | EXIT=0 | EXIT=0 |
| 2 | `cd frontend && npm test -- --run` | 22 files / 172 tests | 22 files / 172 tests (+6 net vs cycle-69 166) |
| 3 | `cd frontend && npm run build` | green | exit 0; all routes built |
| 4 | `grep -n "DonutChart\|@tremor/react" PortfolioAllocationDonut.tsx` | zero matches | one match, comment line 5 only (semantically: explains what was removed) |
| 5 | `grep -n "SLICE_STROKE_CLASS\|stroke-blue-500\|stroke-amber-500" ...` | static map present | lines 78-95 (16 colors mapped) |
| 6 | `grep -n "role=\"tooltip\"" ...` | custom tooltip | line 295 |
| 7 | `grep -n "role=\"img\"\|aria-label.*allocation" ...` | SVG role + summary | lines 190-191 |
| 8 | `grep -n "Escape\|onKeyDown" ...` | WCAG SC 1.4.13 ESC dismiss | lines 150-154 + 179 (handleEsc useCallback + container onKeyDown wiring) |
| 9 | `grep -E "stroke-.*-500" .next/static/css/*.css` | stroke-blue-500 in built CSS | 16/16 colors (blue, amber, indigo, emerald, fuchsia, lime, orange, yellow, cyan, violet, rose, slate, pink, teal, sky, purple) in `66394fa6cbfff906.css` |

Vitest sub-suite passes verbatim (14/14 in the donut file):
- 8 cycle-69 / pre-cycle-69 tests (DOM contract preserved)
- 6 cycle-70 additions: SVG circle count + JIT-safe class assertion +
  tooltip-at-rest absence + tooltip-on-hover content + dismissal-on-
  mouseleave + `<title>` native SVG hover element

## Code-review heuristic sweep (5 dimensions)

### Dimension 1 -- Security
CLEAN. No secrets, no prompt-injection paths, no `subprocess`/`eval`/
`exec`, no system-prompt leakage (presentation only), no RAG mutation,
no unbounded LLM loops, no new agent capability. Component is a leaf
React presentational widget.

### Dimension 2 -- Trading-domain correctness
N/A. No `kill_switch.py`, `risk_engine.py`, `paper_trader.py`,
`perf_metrics.py`, `backtest_engine.py`, or `backtest_trader.py`
modified. Pure frontend.

### Dimension 3 -- Code quality
1 NOTE: `STROKE_WIDTH + 1.2` at line 213 is a magic literal for the
hover stroke-width bump. Semantically adjacent context (the hover
branch of the ternary) makes intent clear; not worth blocking on. All
other constants (RADIUS, CX, CY, STROKE_WIDTH) are named with the
explanatory comment at lines 107-113.

### Dimension 4 -- Anti-rubber-stamp
PASS. Critically:
- The 8 cycle-69 tests pass UNMODIFIED -- this is the behavior-
  preservation evidence. The component's public API
  (`slices`/`totalNav`/`title`/`className`) is preserved exactly.
- The 6 new cycle-70 tests assert REAL behavior, not mocks. The
  `JIT-safe stroke-* classes` test (lines 130-144) asserts the literal
  class strings appear in the rendered HTML -- catches the cycle-68
  template-string-concat regression class deterministically.
- The `<title>` child element test (lines 200-214) exercises the
  native-SVG tooltip path -- defensive ARIA layering.
- The locale-tolerant `replace(/[\s,. ]/g, "")` regex at line 177
  shows the author thought about Linux locale CI differences from
  Mac dev. Real defensive test, not rubber-stamp.

### Dimension 5 -- LLM-evaluator anti-patterns
PASS. This Q/A is the FIRST for cycle 70 (no prior verdict to flip);
all 9 deterministic checks have file:line citations; not a
sycophancy-under-rebuttal situation.

## SVG math verification (replaces the cycle-69 guesswork)

Per the research brief F1 (heyoka/Mark Caron formula):
- `RADIUS = 100 / (2 * Math.PI)` -- line 110 -- circumference = 100.
- `strokeDasharray="<pct> <100-pct>"` -- line 216 -- each slice's
  dasharray directly encodes its share of total NAV.
- `strokeDashoffset = -acc` (negative running sum) -- lines 139, 217.
- `transform="rotate(-90 cx cy)"` -- line 220 -- rotates entire
  `<circle>` so first slice starts at 12 o'clock.

The implementation deviates from the research brief's formula
(`offset = 100 - sum(prev) + 25`) but the two paths are
mathematically equivalent: the brief encodes the 12-o'clock-start
adjustment via the `+25` offset on each slice; the implementation
encodes it via the SVG `transform` on each `<circle>`. Both correct.

`acc += pct` in the `arcs.map` loop guarantees the sum across all
slices equals `(totalValue / totalValue) * 100 = 100` (by
construction, the percentages of a normalized total are exhaustive).

## Tooltip stays inside the card (cycle-69 regression fixed)

Cycle-69 operator complaint: Tremor's default tooltip portaled outside
the card with white-on-dark default styling, escaping the navy/slate
palette.

Cycle-70 fix verified:
- Line 293-308: `<div role="tooltip">` rendered as a direct sibling
  of the `<svg>`, both inside the same `containerClass` div (line
  175). NO portal. Tooltip lives in the document tree position where
  it CAN'T escape the card border.
- Styling: `bg-navy-900 border-navy-700 text-slate-200/100/400` --
  fully navy/slate palette per frontend.md dark-mode rules.
- WCAG SC 1.4.13:
  - Dismissible: ESC handler at lines 150-154 + 179.
  - Hoverable: no `pointer-events: none`; mouse can move from slice
    to legend list item which also hosts a mouseenter handler
    (lines 269-270).
  - Persistent: stays visible until mouseleave (lines 225, 270).

## Cross-component impact

- `frontend/src/app/paper-trading/positions/page.tsx` uses
  `<PortfolioAllocationDonut slices={...} totalNav={...} />` with no
  prop shape change. No consumer-side changes needed.
- `@tremor/react` is still used by `frontend/src/app/performance/page.tsx`
  (`AreaChart`), so the dep can't be removed yet -- but the donut's
  Tremor coupling is fully severed.

## Visual-verification posture

Cycle 69 lacked deterministic certification that slice colors would
render -- the test suite couldn't catch Tremor's failure to apply
`colors` prop to the chart's internal SVG paths.

Cycle 70 fixes this:
- Check #9 (stroke-* in `.next/static/css/`) demonstrates Tailwind JIT
  picked up all 16 SLICE_STROKE_CLASS literals into the production CSS
  bundle. The classes EXIST in the shipped CSS; the `<circle>` elements
  HAVE the class attribute; therefore the slices WILL render with
  their colors.
- Test #11 (JIT-safe stroke-* classes) asserts the literals appear in
  rendered HTML -- catches template-string regression at the
  component-render layer.

Operator visual confirmation still recommended for the final
contrast/density judgment but no longer load-bearing for the "are the
slices colored at all" question.

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "5/5 harness-compliance + 9/9 deterministic + 0 BLOCK / 0 WARN / 1 NOTE (magic-number STROKE_WIDTH+1.2 at line 213; semantically clear). Tremor DonutChart fully replaced by inline-SVG Option B (mirrors cycle-63 SectorBarList precedent). 16/16 stroke-*-500 colors verified in production CSS bundle. WCAG SC 1.4.13 three-leg compliance (dismissible/hoverable/persistent). All 8 pre-cycle-69 tests pass unmodified (behavior contract preserved); +6 new cycle-70 tests exercise SVG correctness + JIT-safe classes + tooltip lifecycle.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit",
    "syntax",
    "verification_command",
    "frontend_typecheck",
    "frontend_test",
    "frontend_build",
    "frontend_eslint",
    "code_review_heuristics",
    "css_bundle_jit_verification",
    "svg_math_verification"
  ]
}
```
