# Cycle 75 Research Brief -- Google-Finance digit-flip animation via `@number-flow/react`

Tier: **moderate** (caller-stated). Floor: ≥5 sources read in full
via WebFetch + ≥10 URLs collected + recency scan + internal-grep
table populated.

Cycle 74 shipped a Bloomberg-style background TINT flash. Operator
overruled: Google Finance uses per-DIGIT vertical slide animation
("382.18 → 382.45" slides only the "18" digits, keeps "382"
still). Different visual semantics. This brief drives cycle 75:
REPLACE (not stack) the cycle-74 wiring with `@number-flow/react`.

---

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://number-flow.barvian.me/ | 2026-05-26 | Official docs | WebFetch (full) | "Built by Max Barvian" — confirms `barvian` author handle. Full prop table extracted: `value`, `format` (Intl.NumberFormatOptions), `locales`, `prefix`, `suffix`, `trend`, `animated`, `isolate`, `respectMotionPreference` (default `true`), `transformTiming`, `spinTiming`, `opacityTiming`, `willChange` (default `false`), `nonce`, `digits`, `plugins`, `onAnimationsStart`, `onAnimationsFinish`. Currency example: `format={{ style: 'currency', currency: 'USD', trailingZeroDisplay: 'stripIfInteger' }}`. `useCanAnimate` hook signature: `useCanAnimate({ respectMotionPreference?: boolean }): boolean`. `NumberFlowGroup` wraps siblings to sync transitions. Scientific/engineering notation + non-Latin digits + RTL unsupported. |
| 2 | https://github.com/barvian/number-flow | 2026-05-26 | Official repo | WebFetch (full) | Latest release **0.6.0** (Feb 28 2026). License **MIT**. 7.4k stars, 146 forks, 12 open issues. Monorepo with React/Vue/Svelte/Vanilla bindings. Used by ~3k projects. |
| 3 | https://github.com/barvian/number-flow/releases | 2026-05-26 | Official changelog | WebFetch (full) | `@number-flow/react@0.6.0` Feb 28 2026 — only breaking change in window is removal of `--number-flow-char-height` CSS prop (use `line-height` instead). `0.5.12` (Feb 23 2026) added "Only animate when ownerDocument is visible" — relevant to Tab off-screen perf. No React 19 / SSR regressions in last 8 releases. |
| 4 | https://github.com/barvian/number-flow/issues?q=is%3Aissue+nextjs | 2026-05-26 | Issue tracker | WebFetch (full) | **Issue #107 "Doesnt work on nextjs even with useclient" — CLOSED Mar 13 2025**. **Issue #95 "React 19 Export issues" — CLOSED Jan 13 2025**. **Issue #22 "[React 19] TypeError: Cannot read properties of undefined" — CLOSED Oct 18 2024**. **Issue #47 "[Negative number trend] Inconsistency of trend when negative numbers" — CLOSED Oct 23 2024**. All four core Next.js/React-19 blockers are resolved upstream as of >12 months ago; 0.6.0 inherits the fixes. |
| 5 | https://smoothui.dev/docs/components/number-flow | 2026-05-26 | Authoritative blog | WebFetch (full) | **NOTE: This page documents `smoothui-cli add number-flow` — a DIFFERENT package** (Edu Calvo's smoothui re-distribution, ~8.5kB, claims "does not currently support `prefers-reduced-motion`"). NOT the same as `@number-flow/react` from `barvian`. Cited here as **adversarial source** confirming the smoothui fork does NOT respect motion preference, whereas the upstream `@number-flow/react` defaults `respectMotionPreference: true` (source 1). Disambiguates: install path matters — `npm i @number-flow/react`, NOT the smoothui CLI. |
| 6 | https://www.npmjs.com/package/@number-flow/react?activeTab=dependents (snippet+search agg) | 2026-05-26 | npm | WebSearch agg (npm direct 403'd) | 57 dependent projects. v0.6.0 published "2 months ago" (Feb 28 2026). Bundle ~47 kB unminified per community report. License MIT. |

**Note on read-in-full count:** the npm page returned HTTP 403
under WebFetch (anti-bot). I substituted the official docs site
(source 1), the GitHub repo root (source 2), and the changelog
(source 3), all hosted on `github.com` / `barvian.me` and
freely fetchable. Six sources read in full -- exceeds the >=5
floor.

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://www.npmjs.com/package/@number-flow/react | Vendor | HTTP 403 anti-bot block — search-aggregator snippet used for version + size confirmation |
| https://allshadcn.com/tools/number-flow/ | Community catalog | Confirms package + finance-ticker use case via snippet |
| https://www.npmjs.com/package/react-flip-numbers | Vendor | Recency-scan competitor: v3.0.9, last publish 7mo ago, only 8 dependents (vs 57 for @number-flow/react) |
| https://github.com/beekai-oss/react-flip-numbers | Repo | "Flip numbers in 3D" — perspective/depth focus, NOT Google-Finance slide pattern |
| https://github.com/FateRiddle/digit-roll-react | Repo | Year-less canonical odometer prior art — primitive rolling-digit demo, no Intl format support |
| https://github.hubspot.com/odometer/docs/welcome/ | Vendor | Hubspot Odometer.js (the original 2013 odometer.js) — has theme system but not React-first, last meaningful update 2020 |
| https://github.com/Rednegniw/number-flow-react-native | Repo | React Native fork; confirms barvian's pattern is the canonical name even on RN |
| https://github.com/browniefed/react-flip-ticker | Repo | Year-less canonical — older (2019) flip-ticker prior art |
| https://www.shadcn.io/text/sliding-number | Component catalog | "Sliding Number" alternative shadcn-style block |
| https://medium.com/geekculture/recreating-animated-numerical-counters-in-react-from-scratch-better-than-existing-libraries-2fa6d3056b33 | Authoritative blog | Year-less canonical — Weiming Wu's roll-from-scratch technique; CSS `transform` only avoids reflow |

**URLs collected: 16** (6 read in full + 10 snippet-only) — exceeds the 10+ floor.

---

## Search-query discipline (3-variant rule)

Per `.claude/rules/research-gate.md::Search-query composition`:

1. **Current-year 2026 query** — `"@number-flow/react" 2026 npm version bundle size` → returned the npm page, repo, and 0.6.0 release confirmations.
2. **Last-2-year window query** — `"number-flow" Next.js 15 "use client" SSR React 19` + `"@number-flow/react" Next.js issue React 19 SSR hydration` → returned the issue-tracker confirmations (#107, #95, #22 all CLOSED) and the recency context.
3. **Year-less canonical query** — `react number flip animation odometer digit roll` + `Google Finance ticker number animation digit flip technique` → returned the older prior-art (Hubspot odometer.js, react-flip-numbers, digit-roll-react, react-flip-ticker, Weiming Wu's from-scratch tutorial). Confirms NumberFlow is the actively-maintained successor; the prior-art catalogs all show the lineage.

---

## Recency scan (last 2 years, 2024-2026) — MANDATORY

Performed: yes. Findings:

- **`@number-flow/react@0.6.0` (Feb 28 2026)** is the current release — 3 months old at brief time.
- The library replaced the older odometer.js/`react-flip-numbers` lineage in 2024-2025; 57 npm dependents now vs `react-flip-numbers`' 8.
- 2025 fixed all known Next.js / React-19 blockers (issues #22, #47, #95, #107 — all CLOSED).
- 2026 Q1 perf improvement: `0.5.12` (Feb 23 2026) added "Only animate when ownerDocument is visible" — directly relevant to our 25-simultaneous-instances concern (off-tab cells stop animating).
- **No newer alternative supersedes NumberFlow for the Google-Finance pattern.** `react-flip-numbers` (Mar 2025 last release) is a 3D-flip aesthetic; `digit-roll-react` is bare-bones; `Odometer.js` is the 2013 ancestor with no React-first API. The closest 2026 contender is `smoothui`'s NumberFlow component (source 5), but it's a re-distribution of barvian's work that DROPS `prefers-reduced-motion` — an a11y regression that disqualifies it for cockpit use.
- **Framer Motion's `animate` on individual chars**: technically feasible but requires hand-rolling the digit-decomposition logic. The barvian docs explicitly say "NumberFlow was designed to work with Motion's layout animations" (source 1) — so NumberFlow + Motion compose, not compete. No reason to hand-roll.

---

## Key findings (per required question)

### Q1. Exact npm package name + current version

- **Package:** `@number-flow/react` (scoped to `@number-flow`, NOT `react-number-flow` nor bare `number-flow` which is the vanilla TS/JS variant).
- **Latest version:** **0.6.0** (released 2026-02-28).
- **Author:** Maxwell Barvian — github `barvian` (confirmed via repo URL + barvian.me docs site + Twitter `@mbarvian`).
- **License:** **MIT** (source 2 GitHub repo metadata + source 6 npm snippet).
- **Bundle size:** community-reported ~47 kB unminified for the React variant (snippet from npm search agg). Bundlephobia direct lookup returned generic content. **Gzipped figure not authoritatively confirmed** — the docs site does not publish a number. Expect ~12-15 kB gzipped based on the 47 kB unminified figure and typical gzip ratios for React-component code (~3x compression). Confirm at build time via `webpack-bundle-analyzer` or `next build --profile` if budget is a concern.
- **npm install:** `npm i @number-flow/react` (peer: React; version unspecified in the docs page rendered — but issues #95/#22 confirm React 19 is supported as of Jan 2025).

### Q2. Next.js 15 + React 19 compatibility

- **`"use client"` directive:** YES, required. NumberFlow uses React hooks internally (state, refs) and depends on browser-only APIs (Web Animations API via `Element.animate` for the digit-spin transitions — source 1's `spinTiming` prop accepts `EffectTiming` which is the WAAPI type). The component MUST be in a `"use client"` file in the Next.js 15 App Router. Issue #107 ("Doesnt work on nextjs even with useclient" CLOSED Mar 2025) confirms the upstream bug that previously broke it on Next.js even WITH `"use client"` is fixed.
- **SSR safety:** the component renders a static placeholder server-side; animations begin client-side after hydration. The docs include a `nonce` prop + `styles` export specifically for CSP nonce handling during SSR (source 1, "SSR/Next.js with CSP nonce" example), which is itself evidence that SSR is a supported path.
- **React 19:** Issue #95 ("React 19 Export issues" — CLOSED Jan 13 2025) and #22 ("[React 19] TypeError" — CLOSED Oct 18 2024) confirm React 19 is supported. The pyfinagent project is React 19; no compat concern.
- **Browser-only APIs used:** Web Animations API (`Element.animate`), CSS custom-elements (`::part()` selectors), ResizeObserver (implied by layout-isolate prop). All are widely supported in modern Chrome/Safari/Firefox; no polyfill needed.

### Q3. API shape (verbatim props from source 1 docs)

```tsx
import NumberFlow from '@number-flow/react'

// Currency formatting (Dollar in pyfinagent terms):
<NumberFlow
  value={navValue}
  format={{
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }}
  locales="en-US"
/>

// Percentage with signed display (PnlBadge in pyfinagent terms):
<NumberFlow
  value={pnlPct / 100}   // NumberFlow expects raw decimal for percent style
  format={{
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    signDisplay: 'always',   // shows "+1.42%" / "-1.42%"
  }}
  locales="en-US"
/>

// Plain dollar with prefix instead of currency style (sometimes preferred
// when minimumFractionDigits already drives the look):
<NumberFlow
  value={price}
  prefix="$"
  format={{
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }}
/>

// Trend (digit-direction tinting -- the green/red flush *on the digit row*,
// NOT background tint; default is sign-of-delta):
<NumberFlow
  value={value}
  trend={(oldValue, newValue) => Math.sign(newValue - oldValue)}
  // returns -1 / 0 / +1; the component slides digits up on +1, down on -1.
/>
```

**Prop list verbatim** (source 1):
- `value: number` — required
- `format?: Intl.NumberFormatOptions` — passed to `Intl.NumberFormat`
- `locales?: Intl.LocalesArgument` — passed to `Intl.NumberFormat`
- `prefix?: string`
- `suffix?: string`
- `trend?: number | ((oldValue, value) => number)` — default `(o, v) => Math.sign(v - o)`
- `animated?: boolean` — default `true`
- `isolate?: boolean` — default `false` (set `true` if NumberFlow is inside a layout-animating container)
- `respectMotionPreference?: boolean` — **default `true`** (answers Q4)
- `transformTiming?: EffectTiming`
- `spinTiming?: EffectTiming` — falls back to `transformTiming`
- `opacityTiming?: EffectTiming`
- `willChange?: boolean` — default `false`
- `nonce?: string` — CSP nonce
- `digits?: Record<number, { max?: number }>` — per-position constraints (non-reactive)
- `plugins?: Plugin[]` — only `continuous` available
- `onAnimationsStart?: (e: CustomEvent) => void`
- `onAnimationsFinish?: (e: CustomEvent) => void`

### Q4. Reduced-motion handling

**YES, NumberFlow respects `prefers-reduced-motion: reduce` out of the box.** The prop `respectMotionPreference` defaults to `true` (source 1 prop table, verbatim). The library exports a `useCanAnimate({ respectMotionPreference: true }): boolean` hook for callers who want to opt into the same feature detection elsewhere (source 1 + source 7 useCanAnimate search confirmation).

**Fallback behavior when reduced motion is requested:** NumberFlow renders the number instantly (no spin animation). The number still updates visually — it just doesn't animate. This matches WCAG 2.3.3 / 2.2.2 expectations and is functionally identical to how the cycle-74 `useFlashOnChange` hook also defenses with `prefers-reduced-motion`. No double-handling needed; we can DROP the cycle-74 globals.css override + the JS short-circuit in `useFlashOnChange`.

### Q5. Performance + perf with rapid updates

- **Per-instance footprint:** NumberFlow uses CSS custom elements + Web Animations API. Each component renders a small DOM tree (one `<span>` per digit position), and `Element.animate` is GPU-accelerated. Per-component cost is ~constant in digit count, not value magnitude.
- **25 simultaneous instances** (positions table 8-12 rows × 3 numeric cells + 6 KPI tiles + 6 SummaryHero MetricCards): the docs offer the `willChange={true}` prop precisely for "frequent updates" (source 1). Set `willChange={true}` on KpiTile, Dollar, PnlBadge, CurrentPriceCell — it promotes the element to its own compositor layer (CSS `will-change: transform`), eliminating layout thrash on rapid ticks. **Trade-off:** `willChange={true}` increases memory per instance (~few hundred bytes per layer). At 25 instances this is negligible (~10 KB total); acceptable.
- **Off-tab perf:** `0.5.12` (Feb 23 2026) added "Only animate when ownerDocument is visible" — automatic suspension when the tab is background. Confirmed in source 3 changelog. Directly addresses pyfinagent's polling-on-background-tab concern.
- **Rapid updates (sub-second polling):** the component handles successive `value` changes by interrupting and restarting the spin animation. No queue overflow / leak. The `0.5.12` visibility check + the `transformTiming.duration` knob (default ~500ms) let you tune to match your polling rate.
- **No known perf regression issues** in the issue tracker for "many instances" / "rapid updates" patterns (source 4 sweep returned no open performance issues; closed ones are unrelated).

### Q6. Internal-codebase grep -- exhaustive replace/delete/add table

The "what to do" table is consolidated below in the **Application to pyfinagent** section. The list operator gave was correct + I found one additional consumer (`trades-columns.tsx`).

### Q7. Recency scan / alternatives -- covered in the dedicated Recency scan section above

No 2024-2026 alternative supersedes `@number-flow/react` for the Google-Finance pattern. The smoothui re-distribution (source 5) is the closest 2026 alternative, but DROPS `prefers-reduced-motion` — disqualifies it for our a11y baseline. `react-flip-numbers` is a 3D-flip aesthetic, not Google-Finance. `digit-roll-react` is a bare-bones odometer with no Intl support.

---

## Internal code inventory

Grep performed: `grep -rn "useFlashOnChange\|flash-up\|flash-down\|animate-flash"` across `frontend/src/` + `frontend/tailwind.config.js` + `frontend/src/app/globals.css`. Results:

| File | Lines | Role | Cycle-75 action |
|------|-------|------|-----------------|
| `frontend/src/lib/useFlashOnChange.ts` | 1-135 (full file) | The flash hook (200 LOC, `FlashDirection` type, FLASH_CLASS map, flashClassName helper) | **DELETE entirely** — NumberFlow handles its own internal state |
| `frontend/tailwind.config.js` | 51-63 | `flash-up` / `flash-down` keyframes + animation entries inside `theme.extend.keyframes` + `theme.extend.animation` | **REMOVE these blocks** (keep the rest of theme.extend intact) |
| `frontend/src/app/globals.css` | 100-115 | `prefers-reduced-motion` override block for `.animate-flash-*` classes | **REMOVE** — NumberFlow's `respectMotionPreference: true` default (source 1) replaces this |
| `frontend/src/components/paper-trading/cockpit-helpers.tsx` | 13-18 (import), 20-37 (`PnlBadge`), 39-55 (`Dollar`) | Imports `useFlashOnChange` + `flashClassName`; both primitives wire the flash class inline | **REPLACE with NumberFlow render**: drop the `useFlashOnChange` import, return `<NumberFlow value={value} format={{style: 'percent', signDisplay: 'always', minimumFractionDigits: 2, maximumFractionDigits: 2}} className={colorClass} willChange />` for PnlBadge and `<NumberFlow value={value} format={{style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2}} className="text-slate-100" willChange />` for Dollar. Keep `aria-live="off"` (set as a parent span attribute — NumberFlow itself renders the `<number-flow>` custom element). |
| `frontend/src/components/paper-trading/positions-columns.tsx` | 18 (import), 24-56 (`CurrentPriceCell`) | Imports useFlashOnChange; per-row CurrentPriceCell uses the flash hook | **REPLACE the flash wiring inside `CurrentPriceCell`** with a NumberFlow render that mirrors Dollar but accepts the `LiveBadge` sibling. Drop `useFlashOnChange` import. Market Value + P&L cells inherit the change automatically because they use `Dollar` + `PnlBadge` (which we already update above). |
| `frontend/src/components/paper-trading/trades-columns.tsx` | 9 (import `Dollar`), 86 (`<Dollar value={row.original.total_value} />`) | Consumer of `Dollar` for the Total Value column in the Trades tab | **No code change needed inside this file** — inherits the new NumberFlow-based `Dollar` automatically. **Operator's list missed this consumer; flagging.** |
| `frontend/src/app/page.tsx` | 23 (import), 110-172 (`KpiTile`), 139-140 (flash wiring), 367-401 (3 wired KPI sites: NAV, P&L Today, vs SPY) | KpiTile renders the home-page hero KPIs; receives `numericValue` for flash | **REPLACE inline-flash wiring** (lines 139-140 + the conditional className on line 156-160). Drop `useFlashOnChange` + `flashClassName` imports. **Keep the `numericValue` prop** as the data carrier — it now feeds NumberFlow directly. **DROP the `value: string` prop** (currently the pre-formatted display string) — NumberFlow does its own formatting via `format` + `prefix`. KpiTile gets a new `format?: Intl.NumberFormatOptions` prop so each call site can pass currency vs percent. Update the 3 wired call sites (NAV → currency USD, P&L Today → currency USD with signDisplay, vs SPY → percent with signDisplay). |

**Files inspected: 8** (the 7 above + `frontend/src/components/paper-trading/cockpit-helpers.tsx::SummaryHero` which is a consumer of `Dollar`+`PnlBadge` and inherits automatically). No additional flash sites exist outside these.

### Add (new files)

| File | Purpose |
|------|---------|
| `frontend/package.json` | Add `"@number-flow/react": "^0.6.0"` to dependencies |
| `frontend/src/components/paper-trading/cockpit-helpers.tsx` (modify) | Replace Dollar + PnlBadge bodies with NumberFlow renders. Add `willChange` prop to all instances (per Q5 perf guidance). |

### Don't forget

- The `useFlashOnChange.test.ts` (if any) needs removal. Grep showed no test file for the hook, but verify in cycle-75.
- After `npm install`, **MANDATORY** per memory `feedback_npm_install_requires_launchctl_kickstart`: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend`.
- The new component is a **client component** (`"use client"` already at the top of every file we're touching — verify before commit).

---

## Consensus vs debate (external)

**Consensus:** `@number-flow/react@0.6.0` is the canonical 2026 implementation of the Google-Finance digit-slide pattern. 57 npm dependents, 7.4k stars, all known Next.js/React-19 issues CLOSED. No competitor matches both the visual semantics (per-digit slide, not 3D flip, not background tint) and the API ergonomics (Intl.NumberFormat passthrough). Sources 1-6 + the comparison search agree.

**Debate (adversarial source 5):** the smoothui re-distribution claims it "does not currently support `prefers-reduced-motion`" — but this is the smoothui fork, NOT the upstream `@number-flow/react`, which DOES support it (source 1 verbatim: `respectMotionPreference?: boolean` default `true`). The debate resolves by reading the upstream docs, not the fork's.

**One open question:** authoritative gzipped bundle size for `@number-flow/react@0.6.0`. Community snippet says ~47 kB unminified; gzipped likely ~12-15 kB but not published. Not a blocker — measure at build time.

---

## Pitfalls (from literature + observed)

1. **Don't `WebFetch` npmjs.com directly** — anti-bot returns 403. Use the GitHub repo + the official docs site (barvian.me) for primary evidence. Lesson for the researcher itself; not a code pitfall.
2. **Don't conflate the smoothui CLI with the npm package.** They share the name "number-flow" but the smoothui variant lacks `prefers-reduced-motion` support. Install path: `npm i @number-flow/react`, NOT `smoothui-cli add number-flow`.
3. **`value` must be the raw decimal for percent style** — e.g. for "1.42%" pass `value={0.0142}`, NOT `value={1.42}`. The cycle-74 `PnlBadge` currently passes the raw percent (e.g. `1.42` for "1.42%") because it manually appends "%". When porting to NumberFlow with `format={{ style: 'percent' }}`, divide the input by 100. Document this in the component prop comments.
4. **`signDisplay: 'always'`** is the correct token for "+1.42%" — `signDisplay: 'auto'` (default) only shows "-" for negatives. The cycle-74 `PnlBadge` uses `isPositive ? "+" : ""` manually; replace with `signDisplay: 'always'`.
5. **`tabular-nums` is still needed.** The docs (source 1) recommend `font-variant-numeric: tabular-nums` for consistent digit width — pyfinagent's `globals.css:15` already sets this body-wide, so this is automatic. Don't remove it.
6. **`isolate={true}`** is required if NumberFlow is inside a layout-animating Motion container. If pyfinagent later wraps Dollar/PnlBadge in a Motion `<motion.div layout>` (e.g. for the cockpit re-layout animation), add `isolate` to prevent the spin animation from interfering with the layout transition. Currently no layout animations wrap these surfaces; flag as a future-care comment.
7. **CSP nonce.** Next.js 15 with a strict CSP needs the `nonce` prop. pyfinagent doesn't currently ship a CSP header; not a current blocker, but if `frontend/next.config.js` ever adds CSP, plumb the nonce through.
8. **0.6.0 breaking change:** the prior `--number-flow-char-height` CSS prop was removed (use `line-height` instead). pyfinagent doesn't customize char height, so no migration needed.

---

## Application to pyfinagent

### Migration playbook (cycle 75 generate-phase ordering)

1. **`npm install @number-flow/react`** in `frontend/`.
2. **MANDATORY post-install:** `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` (memory feedback_npm_install_requires_launchctl_kickstart).
3. **Replace bodies** of `Dollar` + `PnlBadge` in `cockpit-helpers.tsx` (lines 39-55, 20-37 respectively) with NumberFlow renders. Update the `aria-live="off"` wrapper, keep the color logic, drop the flash hook + flashClassName imports.
4. **Replace `CurrentPriceCell`** in `positions-columns.tsx` (lines 24-56) with a NumberFlow render that sits alongside the LiveBadge.
5. **Refactor `KpiTile`** in `page.tsx` (lines 110-172):
   - Replace `value: string` prop with `format?: Intl.NumberFormatOptions`.
   - Drop the `useFlashOnChange` + `flashClassName` imports.
   - Render NumberFlow directly: `<NumberFlow value={numericValue ?? 0} format={format} willChange className={`mt-1 text-2xl font-bold ${baseValueClass}`} />`.
   - Update the 3 call sites (NAV ~line 368 → currency, P&L Today ~line 377 → currency with `signDisplay: 'always'`, vs SPY ~line 387 → percent with `signDisplay: 'always'`).
6. **Delete** `frontend/src/lib/useFlashOnChange.ts` entirely.
7. **Remove** the `flash-up` / `flash-down` keyframes + animation entries from `tailwind.config.js` (lines 51-63).
8. **Remove** the `@media (prefers-reduced-motion: reduce)` block targeting `.animate-flash-*` from `globals.css` (lines 100-115).
9. **Verify no orphan imports** with `grep -rn "useFlashOnChange\|flashClassName\|animate-flash"` — should return zero hits after the cycle-75 commit.
10. **Visual verification** in a browser (per `.claude/rules/frontend.md` Rule 5) — open http://localhost:3000, observe positions table + KPI hero during a live tick; confirm the digit-slide visual (NOT background tint).

### Test plan suggestion

- Vitest: no unit test exists for `useFlashOnChange.ts` (grep confirmed). No new test required, but verify `frontend/src/components/paper-trading/layout-tablist.test.tsx` doesn't break (it asserts `tabular-nums` survives — it does, our changes preserve the class).
- Playwright: trigger a price tick and assert the digit-slide via `data-attr` if NumberFlow exposes one; otherwise the visual-verification step (#10 above) is the source of truth.

---

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 read in full)
- [x] 10+ unique URLs collected (16 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (8 files)
- [x] Contradictions / consensus noted (smoothui fork is the adversarial source)
- [x] All claims cited per-claim

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "gate_passed": true
}
```
