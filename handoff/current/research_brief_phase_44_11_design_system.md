# Research Brief — phase-44.11: Design-system enforcement layer (semantic tokens + shared ui components)

**Tier:** moderate · **Cost:** FREE (frontend; no project LLM spend) ·
**Date:** 2026-05-29 · **Researcher:** Layer-3 MAS
**Step:** phase-44.11 (Cycle 5, production-ready+money push; priority-7 UX)
**Goal:** Validate the ADDITIVE plan to create a semantic design-token
module + first shared `ui/` components WITHOUT regressing the ~120
existing pages. This is the W2 enforcement layer in `ux_roadmap.md`.

Held to `.claude/rules/frontend.md` + `frontend-layout.md` throughout:
navy/slate palette (never zinc), Phosphor icons only (no emoji),
Recharts-only for charts, JIT-safe literal class strings.

---

## PART A — INTERNAL CODE AUDIT (the Explore half)

### A.0 Headline: what ALREADY exists vs what is MISSING

phase-44.1 ("Foundation: design tokens + states lib + hooks + Cmd-K")
is DONE (commit `db1e6208`, 2026-05-22). Critically, **its "design
tokens" deliverable was NOT a `design-tokens.ts` module** — the commit's
file list (verified via `git show --stat db1e6208`) shows it created the
`states/` lib, `lib/hooks/` lib, `CommandPalette.tsx`, `featureFlags.ts`,
and a Sidebar/layout refresh. The phrase "design tokens" in its title
referred to the WCAG/palette baseline work, not a typed token module.

| Artifact phase-44.11 proposes | Already exists? | Evidence |
|---|---|---|
| `@/lib/design-tokens.ts` semantic maps | **NO — genuinely missing** | `grep -rl design-tokens src/` → 0 hits; not in `git show --stat db1e6208` |
| `ui/` component directory | **NO — missing** | `ls src/components/ui/` → no such dir |
| `ui/Button.tsx` | **NO — missing** | no Button component anywhere; 40 files hand-roll `<button>` |
| `ui/StatusBadge.tsx` | **NO — missing** | `LiveBadge.tsx` + `CitationBadge.tsx` exist but are single-purpose, not a variant-driven status badge |
| `ui/TextInput.tsx` / `ui/Select.tsx` | **NO — missing** | DataTable hand-rolls its filter `<input>` (zinc-violation) |
| states lib (Loading/Empty/Error/Offline/StaleData) | **YES — phase-44.1** | `src/components/states/` (5 components + `index.ts` barrel) |
| hooks lib (useDebounced/useKeyboardShortcut/useURLState/useEventSource) | **YES — phase-44.1** | `src/lib/hooks/` (+ `useEnrichmentSignals`, barrel) |
| Cmd-K command palette | **YES — phase-44.1** | `CommandPalette.tsx` (cmdk@^1.1.1), mounted in `layout.tsx` |
| BentoCard (card primitive) | **YES — pre-existing** | `src/components/BentoCard.tsx` |
| Tailwind token primitives (navy palette, radius, shadow) | **YES** | `tailwind.config.js:27-44` |
| Framer-Motion preset lib | **YES but ORPHANED** | `@/lib/motion.ts` — `grep` confirms ZERO imports (only its own docstring) |

**Conclusion: the phase-44.11 plan is purely ADDITIVE and does NOT
duplicate phase-44.1.** The semantic token TS module and `ui/` directory
do not exist. Proceed to create them.

### A.1 Token primitives already defined (the source of truth to reference)

`frontend/tailwind.config.js`:
- `colors.navy`: 900 `#020617`, 800 `#0f172a`, 700 `#1e293b`,
  600 `#1a2744`, 500 `#243352` (lines 28-34). Slate is Tailwind default.
- `borderRadius`: `card: 12px`, `button: 8px`, `badge: 6px` (36-40).
- `boxShadow`: `card`, `card-hover` (41-44).
- `darkMode: "selector"` (line 9); `content` includes
  `./node_modules/@tremor/**` (line 19).

`frontend/src/app/globals.css` (158 lines):
- CSS vars `--bg-primary #020617`, `--bg-card rgba(15,23,42,0.7)`,
  `--border-card #1e293b` (lines 7-9).
- Keyframes: `shimmer` (skeleton), `pulse-glow` / `.alpha-score-glow`
  (3s), `spin-slow`, NumberFlow tint `pyfa-tint-up/down` (127-150).
- `@media (prefers-reduced-motion: reduce)` already guards the
  NumberFlow tints (lines 153-157) — the project ALREADY honors reduced
  motion in CSS; the motion.ts adoption must do the same in JS.

The semantic `design-tokens.ts` should map MEANING → these literal
classes (e.g. `text.primary = "text-slate-100"`), NOT redefine the
palette. It is a typed convenience layer over tailwind.config.js.

### A.2 The orphaned motion lib (`@/lib/motion.ts`)

6 presets: `springSnappy`, `springGentle`, `fadeIn`, `slideUp`,
`staggerContainer`, `staggerItem`, `hoverTap`. Imports
`from "motion/react"` (line 5) — i.e. the modern **`motion`** package
(`motion@^12.38.0` in package.json), NOT legacy `framer-motion`. Zero
consumers. phase-44.11 does NOT need to wire animations into pages (that
is W6), but the Button component CAN bake a `whileTap` micro-interaction
using `motion/react` so the components ship animation-ready. If the
cycle wants to stay maximally low-risk, Button can use a CSS
`active:scale-95 transition-transform` instead — both are acceptable;
note the tradeoff for the contract.

### A.3 Variant-API grounding (existing hand-rolled buttons)

The Button variant API should be reverse-engineered from the real sites
it will replace (40 files contain `<button>`; ~30 with bg- styling):

- **danger** — `GoLiveGateWidget.tsx:70`:
  `rounded bg-rose-900/40 px-3 py-1 text-xs text-rose-200 hover:bg-rose-900/60`
- **secondary (emerald action)** — `OpsStatusBar.tsx:240` (Resume):
  `rounded-md border border-emerald-500/30 px-2 py-1 text-[10px] font-medium text-emerald-300 hover:bg-emerald-900/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 disabled:cursor-not-allowed disabled:opacity-40 min-h-[24px] min-w-[24px]`
- **danger (Flatten)** — `OpsStatusBar.tsx:260`: same shape, rose tint.
- The existing EmptyState CTA (`states/EmptyState.tsx:45-57`) is a
  sky-tinted hand-rolled button with focus ring — a good "primary"
  reference.

**Observed canonical idiom** (already WCAG-correct in OpsStatusBar):
`focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400`
+ `disabled:opacity-40 disabled:cursor-not-allowed` + `min-h-[24px]
min-w-[24px]` (WCAG 2.5.8 target size). The Button component MUST bake
exactly this so every call site inherits the accessible focus ring
phase-44.1 established. Variants needed: `primary | secondary | ghost |
danger` (danger = Pause/Flatten/Stop), per ux_roadmap.

### A.4 The zinc violations to fix this cycle (tiny, additive)

`grep zinc- src/components src/app` → 52 occurrences across ~12 files.
The two ux_roadmap-named offenders:

- **`states/EmptyState.tsx`** — lines 35 (`text-zinc-500 dark:text-zinc-400`),
  40 (`text-zinc-700 dark:text-zinc-300`), 42 (`text-zinc-500 dark:text-zinc-400`).
  Fix → slate equivalents (`text-slate-400` container, `text-slate-200`
  heading, `text-slate-400` desc). Actively imported by 3 pages
  (`performance`, `reports`, `StrategyDetail.tsx`) so this is a real
  visible fix, low risk.
- **`DataTable.tsx`** filter input — line 80 carries the forbidden
  `border-zinc-200 dark:border-navy-700 bg-white dark:bg-navy-900`
  light-mode base (frontend.md rule #2: a `bg-white` base can't be
  reliably overridden). Lines 88/109/110/116/139/148/163 also carry
  zinc light-fallbacks. Minimal fix: drop the zinc/white light bases,
  keep the dark tokens (project is dark-only).

NOTE: scope discipline — phase-44.11 should fix ONLY the two
ux_roadmap-named offenders (EmptyState; optionally the DataTable input
since it is the named filter-input violation). The full 52-occurrence
zinc sweep + the ~120-file token migration is **W5, a later cycle**.
Fixing all 52 now would balloon scope and risk regressions. State this
boundary explicitly in the contract.

### A.5 Dependency reality (no new deps needed)

package.json (relevant): `clsx@^2.1.0` ✓, `motion@^12.38.0` ✓,
`cmdk@^1.1.1` ✓, `@number-flow/react@^0.6.0` ✓, `recharts@^2.12.0` ✓,
`@phosphor-icons/react@^2.1.10` ✓, `tailwindcss@^3.4.0` ✓,
`@tremor/react@^3.18.7` (present; /performance offender, out of scope).
**Absent: `cva` / `class-variance-authority`, `tailwind-variants`,
`tailwind-merge`.** Therefore the variant system should be a
**hand-rolled `Record<Variant, string>` lookup composed with `clsx`** —
this matches the existing `EmptyState.tsx` idiom (uses clsx) and the
canonical `PortfolioAllocationDonut.tsx::DOT_BG` JIT-safe map pattern
named in frontend.md §1.3. **No `npm install` is required**, so NO
launchctl frontend kickstart is needed (per memory
`feedback_npm_install_requires_launchctl_kickstart` — kickstart applies
only after `npm install`). Build-verify is `cd frontend && npm run build`.

### A.6 JIT-safety confirmation

frontend.md §1.3: Tailwind v3 JIT scans for LITERAL class strings; it
does NOT compile `` `bg-${x}-500` ``. The token module + Button variant
map MUST therefore store FULL literal strings per key, e.g.:

```ts
export const BUTTON_VARIANT: Record<ButtonVariant, string> = {
  primary:   "bg-sky-500/10 text-sky-300 hover:bg-sky-500/20 border border-sky-500/30",
  secondary: "border border-navy-600 text-slate-200 hover:bg-navy-700/50",
  ghost:     "text-slate-300 hover:bg-navy-700/40 hover:text-slate-100",
  danger:    "bg-rose-900/40 text-rose-200 hover:bg-rose-900/60 border border-rose-500/30",
};
```

No template concatenation, no `bg-${color}`. The canonical reference is
`PortfolioAllocationDonut.tsx::DOT_BG_CLASS` (named in frontend.md). This
is the single most important correctness constraint for this step.

### A.7 Internal files inspected (anchors)

| File | Lines | Role | Status |
|---|---|---|---|
| `frontend/tailwind.config.js` | 1-48 | token primitives (navy, radius, shadow) | source-of-truth; reference, do not edit |
| `frontend/src/app/globals.css` | 1-158 | CSS vars + keyframes + reduced-motion guard | source-of-truth; reference |
| `frontend/src/lib/motion.ts` | 1-63 | 6 motion/react presets | ORPHANED (0 imports); Button may adopt whileTap |
| `frontend/src/components/states/EmptyState.tsx` | 1-61 | empty-state affordance | zinc-violation (35,40,42) → fix to slate |
| `frontend/src/components/DataTable.tsx` | 74-163 | table + filter input | zinc/white light-base (80,88,...) → drop light fallbacks |
| `frontend/src/components/OpsStatusBar.tsx` | 240,260 | Resume/Flatten buttons | canonical accessible-button idiom; Button must match |
| `frontend/src/components/GoLiveGateWidget.tsx` | 70 | danger button | variant grounding |
| `frontend/src/components/BentoCard.tsx` | (full) | card primitive | EXISTS; ux_roadmap wants +isHoverable/isPressable (W5, not now) |
| `frontend/src/components/states/index.ts` | 1-10 | states barrel | mirror this barrel pattern for `ui/index.ts` |
| `frontend/package.json` | deps | clsx ✓, motion ✓; no cva/tailwind-merge | hand-rolled variants + clsx |
| `handoff/archive/phase-44.1/contract.md` | 1-117 | proves 44.1 scope | confirms no design-tokens.ts shipped |

---

## PART B — EXTERNAL RESEARCH

### B.0 Search-query discipline (3-variant per rules/research-gate.md)

| Topic | Current-year (2026) | Last-2-yr (2025) | Year-less canonical |
|---|---|---|---|
| Semantic tokens in Tailwind | "...typed token maps best practices 2026" | n/a (covered by canonical) | "design tokens semantic naming convention layers primitive alias component" |
| Variant component patterns | n/a | "tailwind-variants vs cva button variants 2025" | (cva.style / tailwind-variants docs surfaced) |
| Motion v12 in App Router | "Next.js 15 React 19 ... Framer Motion use client boundary 2026" | — | "Framer Motion motion React ... prefers-reduced-motion bundle size" |
| JIT-safety (correctness) | — | — | "Tailwind CSS v3 JIT safelist dynamic class names not generated full string" |
| Focus a11y | — | — | "WCAG 2.2 focus-visible ring accessible button focus indicator" |

### B.1 Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://motion.dev/docs/react-reduce-bundle-size | 2026-05-29 | Official doc | WebFetch (full) | Full `motion` component = **34kb min**, can't tree-shake smaller (declarative API). `m`+`LazyMotion` → **4.6kb** initial; `domAnimation` +15kb (animations+variants+tap/hover/focus gestures); `domMax` +25kb (adds pan/drag/layout). |
| 2 | https://motion.dev/docs/react-motion-component | 2026-05-29 | Official doc | WebFetch (full) | `import { motion } from "motion/react"` is the **current canonical** import (not "framer-motion"). RSC path: `import * as motion from "motion/react-client"`. `whileTap`/`whileHover` are declarative gesture props (`whileTap={{scale:0.95}}`). `useReducedMotion` hook exists. |
| 3 | https://www.w3.org/WAI/WCAG22/Understanding/focus-appearance.html | 2026-05-29 | Official spec | WebFetch (full) | SC 2.4.13 Focus Appearance is **Level AAA**. Min area = 2 CSS-px-thick perimeter; **3:1 contrast change** focused vs unfocused. "Easiest…way to meet: solid outline ≥2px." (Note: 2.4.11 Focus Not Obscured + 2.4.7 Focus Visible are the AA criteria; this project's existing ring already targets these.) |
| 4 | https://www.sarasoueidan.com/blog/focus-indicators/ | 2026-05-29 | Authoritative blog | WebFetch (full) | `:focus-visible` shows ring on **keyboard focus only** (suppresses for mouse) — exactly the pattern to bake into Button. `outline:none` w/o replacement = "Do. Not. Do. This." Box-shadow/ring acceptable but **pair with `outline`** for forced-colors/HCM survival. |
| 5 | https://dev.to/webdevlapani/cva-vs-tailwind-variants-choosing-the-right-tool-for-your-design-system-12am | 2026-05-29 | Practitioner blog | WebFetch (full) | cva = framework-agnostic; tailwind-variants = requires Tailwind, adds slots/compound-slots/responsive variants. **Both are conveniences, not requirements** — neither is needed for a simple 4-variant Button; a typed object + clsx is sufficient (article assumes a lib but names no blocker to hand-rolling). |
| 6 | https://medium.com/eightshapes-llc/naming-tokens-in-design-systems-9e86c7444676 | 2026-05-29 | Authoritative (Nathan Curtis) | WebFetch (full) | Canonical token taxonomy: category→property→concept + modifiers (variant/state/scale/mode). **Color tokens by property role**: `color-text-primary`, `color-background-warning`, `color-border-neutral`. Semantic/alias layer aliases generic (`color-red-36`) → purposeful (`color-feedback-error`) so "one red value" can be retuned safely. |
| 7 | https://www.maviklabs.com/blog/design-tokens-tailwind-v4-2026/ | 2026-05-29 | Practitioner (2026) | WebFetch (full) | 3-tier architecture: base (primitives) → **semantic ("Never skip semantic tokens—they're what makes refactors safe")** → component. v3 = config-based compile-time; v4 = CSS-first `@theme` runtime vars. Project is on **v3** (`tailwindcss@^3.4.0`), so config-based + a TS semantic layer is the right v3 fit. |

### B.2 Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://cva.style/docs/getting-started/variants | Official doc | cva API confirmed via #5; project won't add the dep |
| https://www.webtoolshub.online/blog/css-variables-design-tokens-dark-mode-system-2026 | Blog | Token theory dup of #6/#7 |
| https://hexshift.medium.com/how-to-build-a-design-token-system-for-tailwind-that-scales-forever-... | Blog | "fewer than 150 semantic tokens" snippet captured; theory dup |
| https://github.com/tailwindlabs/tailwindcss/discussions/6763 | Official GH | JIT "complete unbroken strings" rule confirmed via snippet |
| https://blogs.perficient.com/2025/08/19/understanding-tailwind-css-safelist-... | Blog | safelist alternative; not needed (we use literal maps not safelist) |
| https://www.hemantasundaray.com/blog/use-framer-motion-with-nextjs-server-components | Blog | "use client" wrapper pattern confirmed via snippet + motion.dev #2 |
| https://motion.dev/docs/react-lazy-motion | Official doc | LazyMotion detail dup of #1 |
| https://www.alwaystwisted.com/articles/design-token-naming-conventions | Blog | naming dup of #6 |
| https://www.a11y-collective.com/blog/focus-indicator/ | Blog | focus dup of #3/#4 |
| https://testparty.ai/blog/wcag-focus-appearance-minimum | Blog | WebAIM "78% of pages fail focus" stat; dup of #3 |
| https://stevekinney.com/courses/react-typescript/tailwind-cva-typed-variants | Course | cva typing dup of #5 |
| https://www.frontendtools.tech/blog/tailwind-css-best-practices-design-system-patterns | Blog | general Tailwind DS patterns; dup |

**URLs collected total: ~25 unique** (7 read-in-full + ~18 snippet/identified). Floor (10+) cleared.

### B.3 Recency scan (2024-2026) — MANDATORY

Searched the 2026 + 2025 windows for: semantic Tailwind tokens, variant
libraries, and Motion-in-App-Router. **Findings (3 new, all
COMPLEMENT — none supersede — the additive plan):**

1. **Tailwind v4 CSS-first `@theme`** (2026 sources #7, webtoolshub) is
   the emerging direction, but it is NOT relevant: this project is
   pinned to `tailwindcss@^3.4.0`. The v3-correct approach is exactly
   what is planned: keep palette in `tailwind.config.js`, add a typed
   TS semantic layer on top. **Do not migrate to v4 in this cycle** (a
   v3→v4 migration is a large separate undertaking, out of scope).
2. **Motion v12 namespace** (2026, motion.dev) confirms `"motion/react"`
   is current and `motion.ts` already uses it correctly — no migration
   debt. `react-client` import exists for RSC if ever needed.
3. **WCAG 2.2 focus criteria** (W3C, current) — 2.4.11/2.4.7 (AA) +
   2.4.13 (AAA). phase-44.1's `focus-visible:ring-2 ring-sky-400` is on
   the right track; the new Button standardizes it everywhere. No newer
   criterion changes the approach.

No 2024-2026 finding contradicts the additive token-module + hand-rolled
variants plan. The plan is consistent with the current frontier.

### B.4 Key findings (per-claim cited)

1. **Semantic tokens are the layer that makes refactors safe** — the
   exact value proposition for this step. "Never skip semantic
   tokens—they're what makes refactors safe" (Mavik Labs 2026,
   maviklabs.com). The 3-tier model base→semantic→component is the
   consensus (Curtis EightShapes; Mavik). Our `design-tokens.ts` is the
   semantic layer over the existing tailwind.config.js primitives.
2. **Color tokens should be keyed by property role** —
   `text-*`/`background-*`/`border-*` (Curtis, eightshapes-llc). Maps
   cleanly to ux_roadmap's proposed `text`/`surface`/`hover`/`focus`/
   `status` semantic groups.
3. **No variant library is required** — cva/tailwind-variants are
   conveniences; a typed `Record<Variant,string>` + `clsx` is
   sufficient for a 4-variant Button (dev.to #5). Avoiding a new dep
   honors /goal "NO new deps w/o research + owner approval" AND avoids
   the launchctl-kickstart hazard.
4. **`:focus-visible` is the correct primitive** — ring on keyboard
   focus only (Soueidan). Bake `focus-visible:outline-none
   focus-visible:ring-2 focus-visible:ring-sky-400` into Button; this
   matches the existing OpsStatusBar idiom and clears WCAG 2.4.7 (AA).
   For HCM robustness, an `outline` fallback is ideal but the ring
   alone matches current project convention — acceptable for this cycle.
5. **Motion is animation-ready at low cost** — `whileTap={{scale:0.95}}`
   is one prop (motion.dev). But the full `motion` component is 34kb;
   for a single button micro-interaction a CSS `active:scale-95
   transition-transform` is zero-kb and equally valid. Recommend CSS
   for Button this cycle; reserve `motion.ts`/LazyMotion for the W6
   page-level rollout where `m`+`LazyMotion` (4.6kb) should be used.

### B.5 Consensus vs debate (external)

- **Consensus:** 3-tier token model; semantic layer is non-optional;
  color tokens by role; `:focus-visible` over `:focus`; Motion requires
  a client boundary.
- **Debate:** cva vs tailwind-variants vs hand-rolled — genuine split.
  For THIS project (v3, no existing variant lib, only a 4-variant
  Button, dep-averse) the hand-rolled clsx map wins on simplicity and
  zero-dep. Revisit a lib only if slots/compound-variants are later
  needed (W5+).
- **Debate:** CSS keyframes vs Motion for micro-interactions — both
  valid; CSS is lighter for a single whileTap, Motion is better for
  orchestrated page/stagger transitions (the W6 use case).

### Pitfalls (from literature)

- **JIT purge eats dynamic classes** — `` `bg-${x}` `` is silently
  dropped (tailwindlabs GH #6763: "only…complete unbroken strings").
  Mitigation: literal-string variant maps (A.6).
- **`outline:none` without replacement** breaks keyboard a11y
  (Soueidan). Never strip the ring; always pair with `focus-visible`.
- **Box-shadow/ring dies in forced-colors mode** unless paired with
  `outline` (Soueidan). Minor for a dark-only internal tool; note it.
- **34kb Motion tax** if `motion` (not `m`) is imported broadly
  (motion.dev). Don't sprinkle `motion.*` across pages without
  LazyMotion — but that is a W6 concern, not this cycle.
- **Tailwind v4 temptation** — do not migrate; project is v3 (recency
  scan #1).

---

## PART C — VALIDATED ADDITIVE PLAN (the deliverable)

**Verdict: the ux_roadmap W2 plan is sound, additive, and
regression-free as scoped. Proceed.** Concrete shape below.

### C.1 NEW files to create

1. **`frontend/src/lib/design-tokens.ts`** — typed semantic maps over
   the tailwind.config.js primitives. JIT-safe full-literal strings
   only. Suggested shape (keyed by property role per Curtis finding):
   ```ts
   export const text = {
     primary: "text-slate-100", default: "text-slate-200",
     secondary: "text-slate-300", tertiary: "text-slate-400",
     dim: "text-slate-500",
   } as const;
   export const surface = {
     card: "bg-navy-800/70", cardSolid: "bg-navy-800",
     raised: "bg-navy-700",
   } as const;
   export const border = { card: "border-navy-700", subtle: "border-navy-700/50" } as const;
   export const hover = { row: "hover:bg-navy-700/40" } as const;
   export const focusRing =
     "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-navy-800" as const;
   export const transition = { fast: "transition-all duration-150", base: "transition-all duration-200", slow: "transition-all duration-300" } as const;
   export const status = {
     success: "bg-emerald-500/15 text-emerald-300",
     warning: "bg-amber-500/15 text-amber-300",
     danger:  "bg-rose-500/15 text-rose-300",
     neutral: "bg-slate-500/15 text-slate-300",
   } as const;
   ```
   (Mirrors the values already in frontend.md §6 contrast targets +
   frontend-layout.md §4 card tokens — it CODIFIES existing rules, does
   not invent new colors.)

2. **`frontend/src/components/ui/Button.tsx`** — variants
   `primary|secondary|ghost|danger`, sizes `sm|md`, `clsx`-composed
   from a literal `Record<ButtonVariant,string>` map (A.6). Bakes
   `focusRing` + `disabled:opacity-40 disabled:cursor-not-allowed` +
   WCAG `min-h-[24px] min-w-[24px]` + CSS `active:scale-95` micro-press
   (zero-kb; finding B.4#5). Forwards ref + all native button props +
   optional `icon` (Phosphor) + `loading` spinner. `danger` = the
   Pause/Flatten/Stop tint.

3. **`frontend/src/components/ui/StatusBadge.tsx`** — variants
   `success|warning|danger|neutral|info`, reading from `status` map.
   Pill: `inline-flex items-center gap-1 rounded-badge px-2 py-0.5
   text-xs font-medium`. Distinct from single-purpose `LiveBadge`/
   `CitationBadge` (kept).

4. **`frontend/src/components/ui/index.ts`** — barrel (mirror
   `states/index.ts`).

5. *(Optional, if time)* `ui/TextInput.tsx` + `ui/Select.tsx` —
   navy/slate input tokens (fixes the DataTable zinc-input pattern at
   the component level). If deferred, say so; Button+StatusBadge+tokens
   is the minimum viable W2.

### C.2 Tiny in-place fixes (named offenders only)

- **`states/EmptyState.tsx`** lines 35/40/42: zinc → slate
  (`text-slate-400` / `text-slate-200` / `text-slate-400`). Optionally
  refactor its inline CTA to use the new `<Button variant="primary"
  size="sm">` to dogfood the component (low risk; 1 import).
- **`DataTable.tsx`** line 80 (and optionally 88/109/110/116/139/148/
  163): drop the `bg-white`/`border-zinc-200` light bases, keep the
  dark navy/slate tokens. This is the ux_roadmap-named "filter input
  border-zinc-200" fix. If the broader table-cell zinc fallbacks feel
  scope-creepy, fix ONLY line 80 (the input) this cycle and leave the
  rest for W5.

### C.3 Explicit scope boundary (keeps it regression-free)

- **DOES NOT migrate the ~120 existing sites** to the new tokens/Button
  (that is **W5**). Existing pages keep their hand-composed classes and
  render identically — additive files can't regress what doesn't import
  them.
- **DOES NOT touch** tailwind.config.js or globals.css (token
  primitives unchanged → zero risk to every existing class).
- **DOES NOT wire animations into pages** (W6) and **does NOT add the
  `motion`/LazyMotion page rollout**. Button uses CSS `active:scale-95`.
- **DOES NOT add any npm dependency** → **no `npm install` → no
  launchctl kickstart needed** (memory
  `feedback_npm_install_requires_launchctl_kickstart`).
- **DOES NOT do the full 52-occurrence zinc sweep** — only the 2
  named offenders.

### C.4 Variant API (validated)

```ts
type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md";
// primary:   bg-sky-500/10 text-sky-300 hover:bg-sky-500/20 border border-sky-500/30
// secondary: border border-navy-600 text-slate-200 hover:bg-navy-700/50
// ghost:     text-slate-300 hover:bg-navy-700/40 hover:text-slate-100
// danger:    bg-rose-900/40 text-rose-200 hover:bg-rose-900/60 border border-rose-500/30
```
All literal strings (JIT-safe). Grounded in the real
GoLiveGateWidget/OpsStatusBar/EmptyState sites (A.3).

### C.5 Build-verify

- `cd frontend && npm run build` (Next.js 15 production build — the
  canonical gate; catches TS + JIT issues).
- `cd frontend && npx tsc --noEmit` for type-only check (note:
  pre-existing playwright.config.ts error is NOT this step's, per
  phase-44.1 contract gate #2).
- Optional `npx vitest run` if a Button/StatusBadge test is added
  (vitest is configured per memory `project_474_vitest_leaderboard`).
- **No backend changes** → no pytest impact.
- **Visual verification** (frontend.md rule #5): since this ships
  pixels, mark visual review pending OR open the dev server and probe
  EmptyState + one Button site. Unit test + grep is necessary but not
  sufficient for visual correctness.

### C.6 Application to pyfinagent (external → internal mapping)

| External finding | Internal application (file:line) |
|---|---|
| Semantic layer over primitives (Mavik, Curtis) | NEW `lib/design-tokens.ts` over `tailwind.config.js:27-44` |
| Color tokens by role (Curtis) | `text`/`surface`/`border`/`status` maps; codifies `frontend.md §6` |
| No variant lib needed (dev.to) | hand-rolled map + `clsx@^2.1.0`; mirrors `EmptyState.tsx:14` |
| `:focus-visible` ring (Soueidan) | `focusRing` const; matches `OpsStatusBar.tsx:240` |
| JIT needs literal strings (tailwindlabs) | literal `Record` maps; `PortfolioAllocationDonut.tsx::DOT_BG` pattern |
| CSS micro-interaction beats 34kb Motion (motion.dev) | Button `active:scale-95`; defer `motion.ts` to W6 |
| Don't migrate to Tailwind v4 (recency) | keep `tailwindcss@^3.4.0` config-based |

---

## Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7: 3 official-doc/spec, 2 authoritative, 2 practitioner)
- [x] 10+ unique URLs total incl. snippet-only (~25)
- [x] Recency scan (last 2 years) performed + reported (B.3; 3 complementary findings)
- [x] Full papers/pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Part A tables)

Soft checks:
- [x] Internal exploration covered every relevant module (tokens, motion, states, DataTable, button sites, deps)
- [x] Contradictions/consensus noted (B.5)
- [x] All claims cited per-claim (B.4)
- [x] 3-variant query discipline shown (B.0)

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 12,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/research_brief_phase_44_11_design_system.md",
  "gate_passed": true
}
```
