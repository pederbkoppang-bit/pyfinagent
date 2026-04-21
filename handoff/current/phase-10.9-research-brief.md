# Research Brief — phase-10.9: Harness-tab sprint-state tile

**Tier:** moderate  
**Date:** 2026-04-20

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://nextjs.org/docs/app/guides/testing/vitest | 2026-04-20 | Official doc | WebFetch | jsdom env + `vitest.config.mts` pattern; `vitest run <positional>` is the CLI, no native `--filter` flag |
| https://www.wisp.blog/blog/setting-up-vitest-for-nextjs-15 | 2026-04-20 | Blog (practitioner) | WebFetch | `globals:true`, `setupFiles`, `include` glob pattern; `@testing-library/jest-dom` import in setup |
| https://testing-library.com/docs/queries/byrole/ | 2026-04-20 | Official doc | WebFetch | `queryAllByRole` returns `[]` on no match — correct for read-only guard; `queryByRole('button')` for single absence |
| https://kentcdodds.com/blog/common-mistakes-with-react-testing-library | 2026-04-20 | Authoritative blog | WebFetch | `query*` variants are the only safe way to assert absence; `expect(screen.queryByRole('button')).not.toBeInTheDocument()` |
| https://oneuptime.com/blog/post/2026-01-15-unit-test-react-vitest-testing-library/view | 2026-04-20 | Blog (2026) | WebFetch | Read-only components: no `userEvent.setup()` needed; verify rendered data directly; props-as-data pattern |
| https://vaskort.medium.com/bulletproof-react-testing-with-vitest-rtl-deeaabce9fef | 2026-04-20 | Blog (practitioner) | WebFetch | Three-tier pattern (hook / component / state); describe blocks per UI state; `vi.fn()` only for callbacks |

---

## Identified but snippet-only (does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://vitest.dev/guide/filtering | Official doc | Summary retrieved via search; positional-arg behavior confirmed |
| https://vitest.dev/guide/cli | Official doc | CLI reference; snippet confirmed no `--filter` flag |
| https://vitest.dev/guide/browser/component-testing | Official doc | Browser mode — not relevant (jsdom is used here) |
| https://medium.com/@samueldeveloper/react-testing-library-vitest-the-mistakes-that-haunt-developers | Blog | Covered by Kent C. Dodds source above |
| https://dev.to/kevinccbsg/react-testing-setup-vitest-typescript-react-testing-library-42c8 | Blog | Setup steps covered by wisp.blog source |
| https://medium.com/@dilit/building-a-modern-application-2025-a-complete-guide-for-next-js | Blog | Tailwind v4 / shadcn focus; not relevant to this component |
| https://makersden.io/blog/guide-to-react-testing-library-vitest | Blog | 404 at fetch time; snippet confirmed role-query priority |

---

## Search queries run (three-variant discipline)

1. **Current-year frontier:** `vitest React Testing Library tile component read-only assertion 2026`
2. **Last-2-year window:** `vitest jsdom queryAllByRole button read-only component test assertion 2025`
3. **Year-less canonical:** `React Testing Library screen.queryAllByRole no mutation controls read-only component verification pattern`
4. **Supplemental:** `vitest --filter positional filename argument file pattern matching 2025`
5. **Supplemental:** `React 19 Next.js 15 dashboard tile card component design pattern Tailwind 2025`
6. **Recency:** `vitest React Testing Library read-only dashboard tile 2025 2026`

---

## Recency scan (2024-2026)

Searched with both year-locked (`2026`, `2025`) and year-less queries.

Findings: Vitest 4.x (released 2025) is confirmed in use in this repo (`"vitest": "^4.1.4"` in `package.json`). No `--filter` flag exists in Vitest 4.x CLI — the positional filename substring is the only native mechanism. The `scripts/run-test.mjs` shim already bridges this gap (see Internal code inventory). No new 2024-2026 literature changes the read-only assertion pattern: `queryAllByRole('button')` + `expect([]).toHaveLength(0)` remains the canonical approach.

---

## Key findings

1. **`--filter` is NOT a native Vitest flag** — Vitest accepts a positional filename-substring argument (`vitest run HarnessSprintTile`). The masterplan command `npm run test -- --filter=HarnessSprintTile` is valid because `frontend/scripts/run-test.mjs` translates `--filter=X` to a positional arg. This is already the established pattern for `AutoresearchLeaderboard` and `VirtualFundLearnings`. No contract mismatch. (Source: vitest.dev/guide/filtering; `run-test.mjs` line 22)

2. **Vitest config is jsdom-based, globals enabled** — `frontend/vitest.config.ts` sets `environment: "jsdom"`, `globals: true`, `setupFiles: ["./vitest.setup.ts"]`, `include: ["src/**/*.{test,spec}.{ts,tsx}"]`. Setup file is a single line: `import "@testing-library/jest-dom/vitest"`. (Source: `frontend/vitest.config.ts:1-18`, `frontend/vitest.setup.ts:1`)

3. **Existing test files are co-located with components** — `AutoresearchLeaderboard.test.tsx` sits alongside `AutoresearchLeaderboard.tsx` in `frontend/src/components/`. `VirtualFundLearnings.test.tsx` follows the same pattern. The new test file must therefore be `frontend/src/components/Harness/SprintStateTile.test.tsx` (or flat in `components/` if no `Harness/` subdirectory is created).

4. **Read-only guard assertion pattern** — Kent C. Dodds canonical guidance: use `query*` (not `get*`) to assert absence. For buttons and textboxes: `expect(screen.queryAllByRole('button')).toHaveLength(0)` and `expect(screen.queryAllByRole('textbox')).toHaveLength(0)`. This covers `read_only_no_mutation_controls` criterion without relying on `data-testid`. (Source: kentcdodds.com/blog/common-mistakes-with-react-testing-library)

5. **No existing `HarnessSprint*` types in `types.ts`** — `frontend/src/lib/types.ts` defines `HarnessCycle` (line 915) and `HarnessValidation` (line 928) but no `HarnessSprintState` interface. The new type must be declared in `types.ts` or inline in the component file.

6. **No `harness_learning_log` API route or BQ write exists yet** — grep across the entire repo found zero matches for `harness_learning_log`, `sortino_delta`, `sprint_state`, or `slot_accounting`. The data shape for the tile is net-new; the parent component will need a new API function in `api.ts` or the tile should accept `data` as a prop (parent-owns-fetching, as specified in the brief). Props-only design is correct.

7. **Existing data-fetching pattern is plain `fetch` via `apiFetch`** — `frontend/src/lib/api.ts` uses a raw `apiFetch` helper (Bearer auth, 30s timeout, no-store). No React Query or SWR. The tile's `data` prop means no fetch at all inside the component — consistent with the `AutoresearchLeaderboard` pattern which also accepts `candidates` as a prop.

8. **`HarnessDashboard.tsx` imports icons directly from `@phosphor-icons/react`** — this violates the stated convention (import from `@/lib/icons.ts`). The new `SprintStateTile` must import from `@/lib/icons.ts` only. Suitable existing exports: `IconCheckCircle`, `IconXCircle`, `IconWarning`, `IconTrendUp`, `NavBacktest` (ClockCounterClockwise).

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/package.json` | 53 | Dep manifest; test runner = `node scripts/run-test.mjs`; vitest 4.1.4 | Current |
| `frontend/scripts/run-test.mjs` | 38 | Shim: translates `--filter=X` to `vitest run X` positional arg | Current, load-bearing |
| `frontend/vitest.config.ts` | 18 | jsdom env, globals true, setupFiles, include glob | Current |
| `frontend/vitest.setup.ts` | 1 | `import "@testing-library/jest-dom/vitest"` | Current |
| `frontend/src/components/HarnessDashboard.tsx` | 501 | Existing Harness tab main component; plain `useEffect`+`useState` fetch | Current |
| `frontend/src/components/AutoresearchLeaderboard.test.tsx` | 134 | Canonical test pattern: `describe/it`, `render/cleanup`, `document.querySelector`, `data-col`/`data-cell` attrs | Current |
| `frontend/src/components/VirtualFundLearnings.test.tsx` | 115 | Same pattern; `data-section`/`data-testid`/`data-row-index` attrs | Current |
| `frontend/src/lib/api.ts` | ~495 | `apiFetch` helper + harness API fns (lines 471-495) | Current |
| `frontend/src/lib/types.ts` | ~950 | `HarnessCycle` (915), `HarnessValidation` (928); no SprintState type | Current |
| `frontend/src/lib/icons.ts` | 145 | Centralized Phosphor exports; `IconCheckCircle`, `IconXCircle`, `IconWarning`, `IconTrendUp` available | Current |

---

## Consensus vs debate

**Consensus:** `queryAllByRole` + `toHaveLength(0)` is the universally recommended pattern for asserting no interactive controls exist. No debate on this.

**Consensus:** Vitest positional-arg is the file filter mechanism; `--filter` does not exist natively. The `run-test.mjs` shim resolves the masterplan command cleanly.

**Minor debate:** Whether to co-locate the test file (`components/SprintStateTile.test.tsx`) or put it in `components/Harness/SprintStateTile.test.tsx`. Existing tests are flat in `components/`; a `Harness/` subdirectory would be new. Flat is safer for consistency.

---

## Pitfalls (from literature)

- Using `getByRole` instead of `queryAllByRole` for absence assertions — will throw instead of returning empty. (Kent C. Dodds)
- Importing icons from `@phosphor-icons/react` directly — `HarnessDashboard.tsx` already does this and it is an existing violation; do not replicate it.
- `data-testid` as primary selector — acceptable as fallback but role/text queries preferred per RTL docs.
- Not calling `cleanup()` in `afterEach` — both existing test files call `cleanup()` explicitly; match this pattern (even though `globals:true` + jest-dom should auto-cleanup, being explicit is the established convention).

---

## Application to pyfinagent (file:line anchors)

| Recommendation | Anchor |
|----------------|--------|
| Test file: `frontend/src/components/SprintStateTile.test.tsx` | Matches pattern at `AutoresearchLeaderboard.test.tsx:1` |
| Component: `frontend/src/components/SprintStateTile.tsx` | New file; sibling to `HarnessDashboard.tsx` |
| Type: add `HarnessSprintState` to `frontend/src/lib/types.ts` after line 931 | `types.ts:928-931` |
| No API function needed inside component; parent owns fetch | `api.ts:471-495` (existing harness fns pattern) |
| Icon imports from `@/lib/icons.ts`, not `@phosphor-icons/react` directly | `icons.ts:1-145` |
| `--filter=HarnessSprintTile` resolves to `vitest run HarnessSprintTile` via shim | `run-test.mjs:22` |
| Read-only guard: `expect(screen.queryAllByRole('button')).toHaveLength(0)` | RTL canonical; matches criterion `read_only_no_mutation_controls` |

---

## Final recommendations

**Component path:** `frontend/src/components/SprintStateTile.tsx`

**Props interface (declare in `types.ts` after line 931):**
```typescript
export interface HarnessSprintWeekState {
  weekIso: string;
  thu: { batchId: string; candidatesKicked: number } | null;
  fri: { promotedIds: string[]; rejectedIds: string[] } | null;
  monthly: { sortinoDelta: number; approvalPending: boolean; approved: boolean } | null;
}
```
Component signature: `function SprintStateTile({ data }: { data: HarnessSprintWeekState | null })`

**Test file location:** `frontend/src/components/SprintStateTile.test.tsx`

**`--filter` resolution:** NOT a native Vitest flag. `npm run test -- --filter=HarnessSprintTile` works correctly because `frontend/scripts/run-test.mjs` strips `--filter=` and passes `HarnessSprintTile` as a positional substring argument to `vitest run`. Vitest then matches any test file whose path contains `HarnessSprintTile`. Zero tooling mismatch — this is identical to how the existing `AutoresearchLeaderboard` and `VirtualFundLearnings` tests are run.

**Test structure (three criteria from masterplan):**
1. `tile_renders_weekly_state` — render with non-null `thu`/`fri`, assert `data-section="weekly-state"` text contains weekIso and batch/promoted counts
2. `tile_renders_monthly_sortino_delta` — render with non-null `monthly`, assert `data-cell="sortino-delta"` shows the numeric value
3. `read_only_no_mutation_controls` — render with full data, assert `screen.queryAllByRole('button').length === 0` and `screen.queryAllByRole('textbox').length === 0`

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (13 collected: 6 full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim
