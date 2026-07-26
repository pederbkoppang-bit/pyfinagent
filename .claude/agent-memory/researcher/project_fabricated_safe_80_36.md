---
name: project-fabricated-safe-80-36
description: phase-80.36 measured facts — absence-becomes-affirmative is a REPEATING class in the paper-trading cockpit (?? 0 and ?.x !== "literal"); zero cockpit-helpers test coverage; per-row (not per-card) unknown derivation is mandatory
metadata:
  type: project
---

Measured 2026-07-26 while researching phase-80.36 (Risk Monitor renders `SAFE` with
zero data). Facts that are NOT derivable by reading the defect report:

**1. "Absence becomes affirmative" is a CLASS in this cockpit, not one bug.** Four
distinct instances, three of which the defect report did not list:
- `cockpit-helpers.tsx:309` `perf?.max_drawdown_pct ?? 0` → `0 > -10` → `SAFE`
- `cockpit-helpers.tsx:310` `portfolio?.total_nav ?? 10000` → concentration % computed
  against a FABRICATED $10k fund, so the "honest" Max-position row lies too in the
  warm-failure path
- `layout.tsx:215` `status?.status !== "not_initialized"` → **TRUE when status is null**,
  which is the only reason the cockpit renders beside its own error banner
- `cockpit-helpers.tsx:256-257` same `?? 0` in `PaperVsBacktestCard` (route
  `/paper-trading/reality-gap`) where the NUMBER renders `—` but the COLOUR is emerald
When auditing any pyfinagent surface for this, grep BOTH `?? 0` and
`?\.\w+ !== "` — the optional-chain-vs-literal comparison is the sneakier half.

**2. The fix must derive unknown PER ROW, never per card.** `Position size` and
`Sector concentration` read `positions`+`portfolio`; only `Kill switch` and `Drawdown`
read `perf`. `getPaperPerformance()` has its own `.catch(() => null)`
(`layout.tsx:193`), so perf-null-with-good-portfolio is a REAL state — a card-level
`if (!perf)` bail hides a live `HIGH (>20%)` breach.

**3. Discriminate on PRESENCE, never on value.** `max_drawdown_pct === 0`,
`benchmark_return_pct === 0`, and `position_count === 0` are all legitimate healthy
readings. `if (!x)` / `x || null` flips genuine SAFE to UNKNOWN. `Math.abs(null) === 0`
in JS, so half-applied fixes silently keep working.

**4. `cockpit-helpers.tsx` has ZERO test coverage** (621 lines, 6 exported components,
no `.test.tsx`). The only paper-trading layout test is `layout.test.tsx:52` covering the
`not_initialized` payload. Nothing anywhere asserts `SAFE`/`OK`/`WARNING`/`DANGER`.

**5. The unknown-state idiom ALREADY EXISTS in-repo** — don't invent one:
`FreshnessBand = "green"|"amber"|"red"|"unknown"` declared 3x
(`paper-trading-utils.ts:33`, `types.ts:1243`, `live-portfolio-context.tsx:51`), and
`states/StaleDataState.tsx` has an `isUnknown` → `"no data"` branch with `role="status"`.
Copy its SHAPE, not its tokens — it uses `zinc`, which violates
`.claude/rules/frontend.md` §1 (navy/slate only).

**6. `.claude/rules/frontend.md` already binds this**: "Color coding: green=bullish,
red=bearish, amber=neutral, **gray=error/unavailable**". The rule existed; the widget
violated it. Cite the in-repo rule before reaching for external authority.

**Why:** phase-80.36 was framed as one widget bug; it is a systemic null-handling class
whose blast radius spans 7 paper-trading sub-routes.

**How to apply:** on any pyfinagent "widget shows wrong/absent data" step, first grep the
whole surface for the two absence-becomes-affirmative shapes, check whether the row's
inputs actually come from the null source, and check for test coverage before promising
a regression net. Best external anchors for the argument:
FAA AC 25-11B Tables 4-1/4-2 (misleading ≥ loss in hazard class),
Grafana No Data/Error defaults, IBM Carbon `Unknown` vs `Normal`.

Related: [[project_frontend_test_env_and_ui_specs_80_5]],
[[project_nan_json_leak_80_1]], [[project_session_stampede_donut_80_11]]
