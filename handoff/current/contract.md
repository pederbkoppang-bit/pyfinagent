# Contract — `goal-market-filter-in-gate-bar` (Cycle 34)

**Date:** 2026-06-01. **Tier:** simple. **Type:** goal-slug (UI control relocation),
same shape as `goal-multimarket-ux` / `goal-browser-mcp`.

## N* delta (N* = Profit − Risk − Burn)

**Burn↓ (speculative, real):** removes one full horizontal row of chrome above the
fold on the paper-trading cockpit, tightening toward the §4.5 "ONE dense bar, not
stacked rows" doctrine. No P delta. No R delta (no trading-path code touched; pure
presentational move of an existing, already-functional control). Articulable ⇒ not
DEFERRED.

## Research-gate summary

`researcher` ran first (gate **PASSED**: 6 sources read in full, 20 URLs, recency scan
done, 7 internal files audited). Brief: `handoff/current/research_brief.md`. Three
decisive findings:
1. **§4.5 endorses this.** The repo's own `frontend-layout.md` §4.5 literally says
   "fold its signal into the bar itself" and ships a `Next run` segment — folding a
   *global* view-scope control + a status dot into `OpsStatusBar` is in-doctrine, not a
   violation. Peer precedent: GitHub "unified filter bar" (Apr-2026), Grafana 12
   conditional-render clutter reduction (May-2025).
2. **a11y is safe ONLY because `OpsStatusBar` is `<section>`, not `role="toolbar"`.**
   W3C APG (toolbar pattern) warns: do not nest a control needing arrow keys (a
   radiogroup) inside a `role="toolbar"` — the toolbar steals Left/Right. The bar is a
   plain `<section aria-label="Paper-trading operator status">` (grep: zero
   `role="toolbar"` in the codebase), so `MarketFilter`'s native roving-tabindex +
   four-arrow + selection-follows-focus model (`MarketFilter.tsx:44-58`) moves in
   verbatim. **HARD RULE: do not promote the bar to `role="toolbar"`.**
3. **The open/closed dot folds in with zero hydration risk** if the existing
   mount-guarded `useState<Date|null>` (`MarketSessionStrip.tsx:24-29`) is lifted into
   the new segment — React's documented two-pass pattern, unchanged in React 19.

Confirmed (file:line): `OpsStatusBar` renders at exactly `page.tsx:360` (homepage, only
`nextRunAt`) + `layout.tsx:478` (cockpit). `MarketFilter` only at `layout.tsx:484`,
`MarketSessionStrip` only at `layout.tsx:489`. **No test references any of the three
components** (`layout-tablist.test.tsx` is a misnamed DataTable smoke test). Bonus:
`MARKET_BENCHMARK_LABEL` (`format.ts:38`) is the `vs SPY/DAX/KOSPI` label criterion 3
asserts.

## Hypothesis

Adding an optional, prop-gated **Market** segment to `OpsStatusBar` and deleting the
standalone filter row in `layout.tsx` will (a) place the `All·US·EU·KR` radiogroup
inside the status bar, (b) remove one row of vertical chrome, (c) preserve the filter's
full function + a11y, and (d) leave the homepage status bar byte-identical — with no
hydration warning and a green build.

## Immutable success criteria (verbatim from the goal prompt — do NOT edit)

1. The All·US·EU·KR radiogroup renders INSIDE the OpsStatusBar
   `<section aria-label="Paper-trading operator status">` (DOM containment, not pixels);
   the standalone row at `layout.tsx:483-490` no longer exists.
2. One fewer row: gate-bar-bottom→NAV-tile-top distance strictly less than today.
3. Live Playwright (skip-auth Path A): EU still flips VS SPY→VS DAX, scopes
   table/allocation/sector; All restores combined view. Reset to All + RESTORE auth gate
   (`launchctl unsetenv LIGHTHOUSE_SKIP_AUTH` + kickstart; verify 302) per
   `docs/runbooks/browser-mcp.md`.
4. Homepage status bar (`page.tsx`) structurally identical to before (no Market segment,
   no market props).
5. Open/closed session state still visible in the cockpit; no hydration warning.
6. `cd frontend && npm run build` green; existing tests pass (incl
   `layout-tablist.test.tsx`); zero emoji; zero new cockpit console errors.

## Plan steps

1. **`MarketFilter.tsx`** — add an optional `sessionOpen?: Record<string, boolean>`
   prop. When provided, color each non-`All` pill's dot emerald (open) / slate (closed)
   via a literal ternary; when absent, keep today's `MARKET_DOT_CLASS` per-market dot
   (so the homepage / any other future caller is unaffected). Keep the exchange-name
   `title` (`:80`), the radiogroup role, and the roving-tabindex/arrow code (`:44-58`)
   exactly as-is.
2. **`OpsStatusBar.tsx`** — add optional props `markets?: string[]`,
   `activeMarket?: string`, `onMarketChange?: (m: string) => void`. Add a
   `MarketSegment` helper that owns the mount-guarded `useState<Date|null>` (lifted from
   `MarketSessionStrip.tsx:24-29`), computes `sessionOpen` via `isMarketOpen`, and
   renders `SegmentLabel "Market"` + `<MarketFilter ... sessionOpen={...} />`. Render it
   as the **left-most** child of the `<section>` + a `<Divider/>` before `GateSegment`,
   gated on `markets && activeMarket && onMarketChange` all present. **Keep the
   `<section>` role — do NOT add `role="toolbar"`.**
3. **`layout.tsx`** — delete the standalone row (`483-490`); pass
   `markets={availableMarkets} activeMarket={activeMarket} onMarketChange={setActiveMarket}`
   into the cockpit `<OpsStatusBar>` (`:478`). Drop the now-unused `MarketFilter` /
   `MarketSessionStrip` imports if no longer referenced. Keep the filtered note
   (`499-504`).
4. **`page.tsx`** — UNCHANGED (homepage `OpsStatusBar` keeps only `nextRunAt`).
5. **`MarketSessionStrip.tsx`** — retire (delete) once its signal is folded into the
   pills and it has no remaining importer; verify `isMarketOpen` still has a live
   consumer (the new `MarketSegment`).
6. **Verify:** `npm run build`; run frontend test suite; emoji grep; Playwright
   skip-auth Path A click-through (EU→`vs DAX`, reset to All, homepage bar unchanged,
   console clean) → **restore auth gate (verify 302)**.

## Guardrails (from research + frontend.md/§4.5)

- No emoji (Phosphor icons / colored dots + text only). Navy/slate palette, never zinc.
- JIT-safe classes: `MARKET_DOT_CLASS` (`format.ts:100`) + literal emerald/slate
  ternary; never `bg-${...}`.
- Dense-bar §4.5: bar stays ONE `flex flex-wrap items-center gap-x-6 gap-y-3` row that
  wraps gracefully; Market segment must not force a permanent 2nd line ≥1280px beyond
  today's existing `Next` wrap.
- Mount-guarded time read (no hydration warning). Keep radiogroup a11y intact.

## Risks (carried from brief)

R1 homepage regression (mitigate: gate on all 3 props) · R2 hydration warning (mitigate:
mount guard) · R3 a11y break if `role="toolbar"` added (mitigate: keep `<section>`) ·
R4 wrap/density (mitigate: existing flex-wrap; visual check at 1440px) · R5 dead
`isMarketOpen` export (mitigate: new segment is the consumer) · R6 visual-only
correctness ⇒ Playwright click-through is the real acceptance evidence.

## References

- `handoff/current/research_brief.md` (6 sources in full; W3C APG radio + toolbar;
  react.dev hydrateRoot; GitHub unified filter bar; Grafana 12; Tailwind JIT).
- `handoff/current/goal_market_filter_in_gate_bar.md` (goal prompt).
- `.claude/rules/frontend-layout.md` §4.5 + §3; `.claude/rules/frontend.md` rules 1/3/5.
- `docs/runbooks/browser-mcp.md` (skip-auth Path A + mandatory restore).
- Code: `OpsStatusBar.tsx`, `MarketFilter.tsx`, `MarketSessionStrip.tsx`,
  `layout.tsx:478/483-490/499-504`, `page.tsx:360`, `format.ts:38/51/77/100/128/212`.
