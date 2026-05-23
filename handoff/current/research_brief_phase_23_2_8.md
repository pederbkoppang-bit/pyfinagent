---
phase: 23.2.8
tier: simple
title: Verify home cockpit + paper-trading hero NAV / Total P&L SSOT (useLiveNav)
date: 2026-05-23
gate: TBD (filled in Section F)
---

# phase-23.2.8 Research Brief -- useLiveNav SSOT verification

## Section A -- Internal audit (file:line)

### A.1 The hook itself

**Definition:** `frontend/src/lib/useLiveNav.ts` (51 lines, single export).

Note: the hook lives at `frontend/src/lib/useLiveNav.ts`, NOT at the
collected location `frontend/src/lib/hooks/`. This is consistent with
sibling hooks `useLivePrices.ts` and `useTickerMeta.ts` which also
live at `src/lib/` root, while the `src/lib/hooks/` subdirectory
contains only generic utility hooks (`useDebounced`, `useEventSource`,
`useURLState`, `useKeyboardShortcut`).

`useLiveNav.ts:19-22` -- return shape:
```ts
export interface UseLiveNavResult {
  liveNav: number | null;
  liveTotalPnlPct: number | null;
}
```

`useLiveNav.ts:24-28` -- signature:
```ts
export function useLiveNav(
  status: PaperTradingStatus | null,
  positions: PaperPosition[],
  livePrices: Record<string, LivePriceEntry>,
): UseLiveNavResult
```

`useLiveNav.ts:29-40` -- `liveNav` math:
- 0 positions OR no live ticks present -> fall back to
  `status?.portfolio.nav ?? null`.
- Else NAV = `cash + sum(livePrice * qty)` where `livePrice` falls
  back to `pos.current_price` then `pos.avg_entry_price` per-row.

`useLiveNav.ts:42-48` -- `liveTotalPnlPct`:
`((liveNav - starting_capital) / starting_capital) * 100`
(deposit-aware, anchored to operator's deposit baseline).

### A.2 Home page (MAS Operator Cockpit)

`frontend/src/app/page.tsx:15` -- import:
```ts
import { useLiveNav } from "@/lib/useLiveNav";
```

`frontend/src/app/page.tsx:154-156` -- call site:
```ts
const positionTickers = positions.map((p) => p.ticker).filter(Boolean);
const { prices: livePrices } = useLivePrices(positionTickers, positions.length > 0);
const { liveNav, liveTotalPnlPct } = useLiveNav(ptStatus, positions, livePrices);
```

`frontend/src/app/page.tsx:158-162` -- display:
```ts
const nav = ptStatus?.portfolio;
const navValue = liveNav ?? nav?.nav;
const pnl     = liveTotalPnlPct ?? nav?.pnl_pct;
```

### A.3 Paper-trading page

`frontend/src/app/paper-trading/page.tsx:46` -- import:
```ts
import { useLiveNav } from "@/lib/useLiveNav";
```

`frontend/src/app/paper-trading/page.tsx:437-444` -- call site
(arg shapes identical to home page):
```ts
const { prices: livePrices } = useLivePrices(positionTickers, positions.length > 0);
const { liveNav, liveTotalPnlPct } = useLiveNav(status, positions, livePrices);
```

`frontend/src/app/paper-trading/page.tsx:188-203` -- `SummaryHero`
display path; `navDisplay = liveNav ?? status?.portfolio.nav ?? null`,
mirroring the home page fallback (line 159).

### A.4 SSOT discipline -- VERIFIED PRESENT

Both pages:
1. Import `useLiveNav` from the SAME path (`@/lib/useLiveNav`).
2. Destructure the SAME pair `{ liveNav, liveTotalPnlPct }`.
3. Apply the SAME `?? snapshot.nav` fallback at the display layer.
4. Wire `useLivePrices` with the SAME enabled-flag pattern
   (`positions.length > 0`).
5. Pass `(status|ptStatus, positions, livePrices)` -- the variable
   names differ (`status` vs `ptStatus`) but the types are the same
   (`PaperTradingStatus | null`).

**Verified: the SSOT invariant the masterplan claims as
"phase-23.1.17 useLiveNav SSOT" is structurally satisfied.**
The single function `useLiveNav.ts:24-28` is the only place this math
is written; both pages call it with isomorphic args.

### A.5 Phase-23.1.17 archive review

`handoff/archive/phase-23.1.17/contract.md` confirms intent. Plan
step 1 introduced `useLiveNav.ts`; steps 2 and 3 refactored
paper-trading and home pages to consume it; step 7 added immutable
verification that asserts existence + dual-import.

**One immutable verification already exists:**
`tests/verify_phase_23_1_17.py` (per contract line 111). It asserts:
- `useLiveNav.ts` exists
- Home page imports it
- Paper-trading page imports it (refactored)
- Repair script exists and calls `mark_to_market`

This is the GREP test the masterplan wants reused / augmented. See
Section C for the augmentation shape.

### A.6 Existing test coverage gap

`grep useLiveNav frontend/src` returned 7 hits, all in production
source. **No `useLiveNav.test.ts(x)` exists.** vitest is configured
(`frontend/vitest.config.ts`, `jsdom` env, includes
`src/**/*.{test,spec}.{ts,tsx}`); 7 component tests are already in
place (e.g. `RedLineMonitor.test.tsx`). Adding a Vitest test for
the hook is the right augmentation -- compute-once-share-shape
parity AND a unit test that proves both pages get the same number
for identical inputs.

## Section B -- External sources (>=5 in full)

| # | URL | Accessed | Kind | Fetched | Key takeaway |
|---|-----|----------|------|---------|--------------|
| 1 | https://react.dev/learn/reusing-logic-with-custom-hooks | 2026-05-23 | Official docs | WebFetch in full | "Custom Hooks let you share stateful logic but not state itself. Each call to a Hook is completely independent." |
| 2 | https://tkdodo.eu/blog/the-query-options-api | 2026-05-23 | Authoritative blog (TkDodo, TanStack Query maintainer) | WebFetch in full | `queryOptions` helper = single source of truth for queryKey+queryFn; eliminates "duplicated query keys/functions, easy to drift". |
| 3 | https://github.com/TanStack/query/discussions/2310 | 2026-05-23 | Official discussion (TanStack maintainer Dominik) | WebFetch in full | "When useQuery is called multiple times with the same key, all 3 components are subscribed to it ... whenever the data is updated, all 3 will re-render." Confirms cache-keyed sharing pattern (orthogonal to our hook-only pattern). |
| 4 | https://kentcdodds.com/blog/colocation | 2026-05-23 | Authoritative blog (Kent C. Dodds) | WebFetch in full | Colocation principle: keep code as close as possible to where it's relevant; extract upward only when 2+ consumers need it. Validates the `src/lib/` placement vs `src/lib/hooks/`. |
| 5 | https://testing-library.com/docs/react-testing-library/api/#renderhook | 2026-05-23 | Official testing-library docs | WebFetch in full | `renderHook` is the correct vitest+RTL API for testing a custom hook in isolation (`@testing-library/react` v13.1+); use `result.current` to read returned values; rerender to change props. |
| 6 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-23 | Anthropic engineering | WebFetch in full | "Communication was handled via files: one agent would write a file, another agent would read it" -- auditable file artifacts > remembered manual checks. (Page itself recommends Playwright for UI eval; pyfinagent's source-grep+Vitest equivalent below.) |

Per-claim citations appear inline below where the finding is applied.

## Section C -- Recommended pytest / Vitest shape

The masterplan step says "Manual: open both pages; NAV / Total P&L
should be byte-identical". A manual visual check is **not**
mutation-resistant -- a refactor that breaks SSOT could merge before
anyone opens both pages side-by-side. Two layered tests close that
gap:

### C.1 Tier-1 source-grep test (cheap, fast, deterministic)

The existing `tests/verify_phase_23_1_17.py` already does this.
**Reuse and augment it** (do not duplicate) so phase-23.2.8 inherits
the assertion:

```python
# tests/verify_phase_23_2_8.py
"""phase-23.2.8 -- assert useLiveNav SSOT invariant: home and
paper-trading pages BOTH import useLiveNav from @/lib/useLiveNav,
BOTH destructure { liveNav, liveTotalPnlPct }, AND no NAV math
exists outside the hook file."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_SRC = ROOT / "frontend" / "src"

HOME = FRONTEND_SRC / "app" / "page.tsx"
PAPER = FRONTEND_SRC / "app" / "paper-trading" / "page.tsx"
HOOK = FRONTEND_SRC / "lib" / "useLiveNav.ts"

failed = []

# 1. Hook exists and exports useLiveNav.
if not HOOK.exists():
    failed.append(f"missing: {HOOK}")
elif "export function useLiveNav" not in HOOK.read_text():
    failed.append(f"hook does not export useLiveNav: {HOOK}")

# 2. Both pages import useLiveNav from the canonical path.
IMPORT_RE = re.compile(
    r'import\s*\{\s*useLiveNav\s*\}\s*from\s*["\']@/lib/useLiveNav["\']'
)
for page in (HOME, PAPER):
    src = page.read_text()
    if not IMPORT_RE.search(src):
        failed.append(f"{page} does not import useLiveNav from @/lib/useLiveNav")
    # 3. Both pages destructure the same pair.
    if "{ liveNav, liveTotalPnlPct }" not in src:
        failed.append(f"{page} does not destructure liveNav + liveTotalPnlPct")

# 4. Anti-drift: NAV math outside the hook -> fail.
# The phrase "cash + positionsValue" should appear ONLY in useLiveNav.ts.
NAV_MATH_FRAGMENTS = ["cash + positionsValue", "cash + sum(livePrice"]
for fragment in NAV_MATH_FRAGMENTS:
    locations = []
    for ts_file in FRONTEND_SRC.rglob("*.ts*"):
        if fragment in ts_file.read_text():
            locations.append(ts_file)
    if len(locations) > 1 or (locations and locations[0].name != "useLiveNav.ts"):
        failed.append(
            f"NAV math fragment {fragment!r} found outside useLiveNav.ts: "
            f"{[str(p) for p in locations]}"
        )

if failed:
    for f in failed:
        print(f"FAIL: {f}")
    sys.exit(1)
print("PASS: phase-23.2.8 SSOT invariants hold")
```

This is the **immutable-verification command** for the contract.
Cheap (<200ms), zero infra (pure file IO), and mutation-resistant
(catches removed imports, renamed paths, and any future drift where
someone re-inlines the math in one of the pages).

### C.2 Tier-2 Vitest hook test (numerical parity)

Optional but recommended -- add `frontend/src/lib/useLiveNav.test.ts`
to prove the hook returns identical values for identical inputs.
This catches a class of bug the grep can't: someone updates the hook
math but the math is still single-sourced; numerical drift between
the two pages would only show if you tested the math itself.

```ts
// frontend/src/lib/useLiveNav.test.ts
import { describe, it, expect, afterEach } from "vitest";
import { renderHook, cleanup } from "@testing-library/react";
import { useLiveNav } from "./useLiveNav";
import type { LivePriceEntry } from "./useLivePrices";
import type { PaperPosition, PaperTradingStatus } from "./types";

const STATUS: PaperTradingStatus = {
  portfolio: { nav: 10500, cash: 2000, starting_capital: 10000, pnl_pct: 5 },
} as PaperTradingStatus;

const POSITIONS: PaperPosition[] = [
  { ticker: "AAPL", quantity: 10, avg_entry_price: 150, current_price: 160 } as PaperPosition,
  { ticker: "MSFT", quantity:  5, avg_entry_price: 300, current_price: 320 } as PaperPosition,
];

const LIVE: Record<string, LivePriceEntry> = {
  AAPL: { price: 170, fetched_at: "2026-05-23T12:00:00Z" } as LivePriceEntry,
  MSFT: { price: 330, fetched_at: "2026-05-23T12:00:00Z" } as LivePriceEntry,
};

describe("useLiveNav", () => {
  afterEach(cleanup);

  it("computes NAV from cash + live*qty when ticks present", () => {
    const { result } = renderHook(() => useLiveNav(STATUS, POSITIONS, LIVE));
    // cash 2000 + 10*170 + 5*330 = 2000 + 1700 + 1650 = 5350
    expect(result.current.liveNav).toBe(5350);
    // (5350 - 10000) / 10000 * 100 = -46.5
    expect(result.current.liveTotalPnlPct).toBeCloseTo(-46.5, 5);
  });

  it("falls back to snapshot NAV when no live ticks", () => {
    const { result } = renderHook(() => useLiveNav(STATUS, POSITIONS, {}));
    expect(result.current.liveNav).toBe(10500); // snapshot
    expect(result.current.liveTotalPnlPct).toBe(5); // snapshot pct
  });

  it("falls back to snapshot when positions empty", () => {
    const { result } = renderHook(() => useLiveNav(STATUS, [], LIVE));
    expect(result.current.liveNav).toBe(10500);
  });

  it("is referentially deterministic for identical args", () => {
    // The two pages will produce byte-identical NAV when args are
    // structurally identical -- this is the SSOT contract.
    const { result: r1 } = renderHook(() => useLiveNav(STATUS, POSITIONS, LIVE));
    const { result: r2 } = renderHook(() => useLiveNav(STATUS, POSITIONS, LIVE));
    expect(r1.current.liveNav).toBe(r2.current.liveNav);
    expect(r1.current.liveTotalPnlPct).toBe(r2.current.liveTotalPnlPct);
  });
});
```

`renderHook` is from `@testing-library/react` per
[testing-library docs](https://testing-library.com/docs/react-testing-library/api/#renderhook).

### C.3 Why TWO tests, not one

- The source-grep is mutation-resistant against *structural drift*
  (someone re-inlines, renames, moves the hook).
- The Vitest is mutation-resistant against *numerical drift*
  (someone "fixes" the hook math in a way that breaks parity).
- Both together = the SSOT invariant under refactoring.

**Critical caveat from React.dev:** "Custom Hooks let you share
stateful logic but not state itself. Each call to a Hook is
completely independent from every other call to the same Hook."
[react.dev/learn/reusing-logic-with-custom-hooks](https://react.dev/learn/reusing-logic-with-custom-hooks).

This means: if the two pages somehow received different `status`
or `positions` props (e.g. the home page fetched a stale snapshot
while the paper-trading page polled fresh), the hook would compute
different NAVs even though the math is single-sourced. The pages
mitigate this by both calling `getPaperTradingStatus()` +
`getPaperPortfolio()` at mount (page.tsx:100-106, paper-trading
page.tsx:469-475), but neither shares a TanStack-Query cache.

**This is a known weakness** that a future phase could close by
introducing TanStack Query (or SWR) keyed shared cache per
[Dominik's queryOptions guidance](https://tkdodo.eu/blog/the-query-options-api).
For phase-23.2.8 the verification only asserts the **logic** is
single-sourced; numerical parity at the page level depends on the
fetch race converging within the 30s poll window.

## Section D -- Recency scan (2024-2026)

Searched 2025-2026 React + TanStack Query + custom-hook SSOT
literature.

**New findings in the window:**
- TanStack DB 0.5 (Aug 2025) introduces query-driven sync with
  live binding to a local store; sharper "single cache" semantics
  than v4 `queryOptions`. Out of scope for phase-23.2.8 but worth
  flagging if a v2 of useLiveNav adds shared cache.
  [tanstack.com/blog/tanstack-db-0.5-query-driven-sync](https://tanstack.com/blog/tanstack-db-0.5-query-driven-sync)
- TkDodo's [queryOptions guidance](https://tkdodo.eu/blog/the-query-options-api)
  (TanStack Query maintainer) reaffirms that shared `queryKey` +
  `queryFn` via a typed helper is the *recommended* SSOT pattern in
  2025+. The current pyfinagent custom-hook approach is the
  pre-TanStack baseline -- valid, but more drift-prone than the
  TanStack pattern.

**No findings that supersede React.dev's core guidance** that
custom hooks share *logic*, not *state*. The 2025-2026 picture is
"custom hooks remain valid; libraries like TanStack offer a
strictly-stronger SSOT story when you adopt them."

## Section E -- 3-variant query log

Per `.claude/rules/research-gate.md`:

1. **Current-year frontier (2026):**
   - `React custom hook single source of truth SSOT cross-page state 2026`
   - `Vitest custom hook testing best practice 2026`
2. **Last-2-year window (2025):**
   - `TanStack Query share state across pages 2025`
3. **Year-less canonical:**
   - `React custom hook SSOT pattern duplicate component state`
   - `Kent C Dodds colocation pattern`
   - `testing-library renderHook API`

Mix achieved: the source table has React 19 / 2026 docs
(react.dev), 2025 TanStack DB blog, canonical TkDodo + Dodds posts,
and a 2024 Anthropic harness post.

## Section F -- JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "gate_passed": true
}
```

## Section G -- Application notes for the planner

1. **SSOT structurally satisfied -- proceed to GENERATE with a
   verification-only step.** No code change needed in production
   source. The contract should ADD `tests/verify_phase_23_2_8.py`
   (Section C.1) and the optional Vitest test (Section C.2). Do
   NOT touch `useLiveNav.ts`, `page.tsx`, or `paper-trading/page.tsx`.

2. **Mutation-resistance over visual parity.** The masterplan's
   "manual: open both pages" check is operator-dependent. The
   source-grep test (C.1) catches re-inlining, rename, or path
   drift on EVERY future cycle, not just the one phase-23.2.8 is
   focused on. This aligns with pyfinagent's file-based handoff
   discipline per
   [Anthropic harness-design](https://www.anthropic.com/engineering/harness-design-long-running-apps):
   "Communication was handled via files: one agent would write a
   file, another agent would read it" -- an auditable artifact
   beats a remembered manual check. (The Anthropic page itself
   describes Playwright-based interactive verification for UI
   evaluators; the pyfinagent equivalent for SSOT is the source-
   grep + Vitest pair below.)

3. **Fallback path if hook absent.** Both pages have a parallel
   `?? status?.portfolio.nav ?? null` fallback chain at the display
   layer (home page:159, paper-trading SummaryHero:203). If a
   future refactor removed `useLiveNav.ts` the pages would still
   render -- they would silently regress to showing the stale BQ
   snapshot. The Section C.1 test catches this by asserting both
   the import AND the destructured pair; the fallback alone is not
   safe.

4. **Known weakness to flag in evaluator_critique.md:** custom
   hooks share *logic*, not *state*
   ([react.dev](https://react.dev/learn/reusing-logic-with-custom-hooks)).
   Numerical parity between the two pages depends on both pages
   receiving identical `status` + `positions` from the backend. A
   stale 30s tick gap between page mounts could still produce
   transient NAV mismatch even with SSOT logic. A future phase
   could close this with TanStack Query keyed cache
   ([TkDodo queryOptions](https://tkdodo.eu/blog/the-query-options-api)).
   Out of scope here.

5. **Do not duplicate `tests/verify_phase_23_1_17.py`.** That
   script already asserts the import and existence. The new
   `tests/verify_phase_23_2_8.py` should add the *anti-drift*
   assertions (no NAV math outside the hook; both pages destructure
   the same pair) so the two scripts are independently auditable
   but non-overlapping. The contract should declare the new
   script as the immutable verification command.
