# Research Brief — phase-53.2 (UX elevation: ONE consistent layout/design/animation across ALL pages + WCAG AA)

Tier: **complex** | Researcher session | Date: 2026-06-10 | Gate: **PASSED** (`gate_passed: true`)

THE TASK: phase-47.5 shipped `design-tokens.ts` + `ui/Button` + `ui/StatusBadge`; phase-44.1
shipped a `states/` library (LoadingState/EmptyState/ErrorState/OfflineState/StaleDataState).
phase-53.2 elevates the WHOLE surface toward ONE consistent layout/design/animation + WCAG AA.
"One consistent surface across ALL pages" is potentially an unbounded refactor — this brief
makes it BOUNDED: a documented all-pages audit that names the TOP, highest-value, lowest-risk
unification deltas (where pages diverge from the tokens + `ui/` primitives) + the concrete
WCAG-AA gaps, as a finite, grep-verifiable worklist. DO-NO-HARM: every touched page keeps its
behavior + error/loading/empty states; no data/behavioral regression.

## HEADLINE FINDING (drives the whole scope)

**The phase-47.5 + 44.1 foundation was BUILT but is almost entirely UN-ADOPTED.** The consistent
surface already exists in code; the pages just don't import it. Grep-proven:

- `tokens` from `@/lib/design-tokens` — imported by **0 pages**; only the 2 `ui/` primitives
  consume it (design-tokens.ts itself; ui/Button.tsx:16; ui/StatusBadge.tsx:12).
- `ui/Button` + `ui/StatusBadge` — imported by **0 consumers** outside the `ui/index.ts` barrel,
  despite Button.tsx:2-4 documenting it should replace "~30 hand-composed inline button class
  strings (page.tsx, OpsStatusBar.tsx, GoLiveGateWidget.tsx, Sidebar.tsx)".
- `states/` library — `LoadingState`/`ErrorState` imported by **0 pages**; `EmptyState` adopted
  by exactly **2** (`performance/page.tsx:7`, `reports/page.tsx:26`). The other ~11 pages still
  inline their empty/error/loading states.

So 53.2 is NOT "design a new consistent surface" — it is **adopt the surface that 47.5/44.1
already built**. That reframing is what makes the step bounded, low-risk, and high-value: each
delta is a mechanical swap of a hand-rolled string for an existing, tested primitive/token, with
a grep that proves it. The phase-43.0 DoD audit (commit 0d4ddcbe, 2026-06-01) independently
scores UX at **0/12** — consistent with "foundation present, adoption absent".

**RECOMMENDED BOUNDED 53.2 SCOPE (one line):** Land the **inline-error-banner → `ErrorState`
migration** (the single largest, most mechanical, most grep-verifiable consistency win: ~36
inline rose-banner divs across 13 pages → one `<ErrorState>`), the **`ui/Button` adoption on the
highest-traffic action sites**, and a **focus-ring + `tokens.focusRing` sweep on the 9 pages whose
buttons lack `:focus-visible`** (the one true WCAG-AA gap). Defer the full ~1246-site slate-token
migration, DataTable on /backtest+/cron, and all operator-visual items to documented follow-ups.

---

## External literature

### Read in full (>=5 required; counts toward the gate) — 7 read

| # | URL | Accessed | Kind | Fetched how | Key finding |
| --- | --- | --- | --- | --- | --- |
| 1 | https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/ | 2026-06-10 | W3C official (normative index) | WebFetch full | Authoritative AA-vs-AAA split for the new 2.2 criteria. **AA (required):** 2.4.11 Focus Not Obscured (Min), 2.5.7 Dragging, 2.5.8 Target Size (Min), 3.3.8 Accessible Auth. **AAA:** 2.4.12, **2.4.13 Focus Appearance**, 3.3.9. |
| 2 | https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum | 2026-06-10 | W3C Understanding (official) | WebFetch full | SC 2.5.8 = **"at least 24 by 24 CSS pixels"**. Exceptions: spacing (a 24px-diameter circle centered on each undersized target must not intersect another), equivalent control, inline (line-height-constrained), user-agent, essential (maps/dataviz). |
| 3 | https://www.w3.org/WAI/WCAG22/Understanding/non-text-contrast.html | 2026-06-10 | W3C Understanding (official) | WebFetch full | SC 1.4.11 = **3:1** for UI components + graphical objects + **focus indicators** ("the visual focus indicator must have sufficient contrast against the adjacent background"). Button **boundaries** need 3:1 only when the border is the SOLE identifier of the control (text/contrasting-icon controls are exempt). Hover-only treatments are not "required to identify". |
| 4 | https://www.w3.org/WAI/WCAG22/Understanding/focus-not-obscured-minimum | 2026-06-10 | W3C Understanding (official) | WebFetch full | SC 2.4.11 AA = focused component **not ENTIRELY hidden** by author content (sticky header/footer/banner). Partial obscuring is OK at AA (full visibility is the AAA 2.4.12). Dashboard fix: **CSS `scroll-padding`** so the sticky header/sidebar never fully covers a focused element. |
| 5 | https://www.w3.org/WAI/WCAG22/Understanding/focus-visible.html | 2026-06-10 | W3C Understanding (official) | WebFetch full | SC 2.4.7 AA = "keyboard focus indicator is visible". **ANY visible indicator suffices for AA — no size/contrast spec at AA** (size/contrast detail is the AAA 2.4.13). `C45: Using CSS :focus-visible` is a SUFFICIENT technique. => the project's `tokens.focusRing` (`:focus-visible ring-2 ring-sky-400`) is the correct, AA-sufficient idiom. |
| 6 | https://www.deque.com/axe/axe-core/ | 2026-06-10 | Official (Deque, axe-core engine) | WebFetch full | axe-core supports **WCAG 2.0/2.1/2.2 at A/AA/AAA**; "commitment to zero false positives" => uncertain cases are flagged for manual review, NOT auto-failed. Does NOT cover native mobile. Confirms automated tooling is a floor, not a ceiling. |
| 7 | https://designsystem.digital.gov/design-tokens/ | 2026-06-10 | Official (USWDS, US gov design system) | WebFetch full | The consistency thesis: "maximize design efficiency... with design tokens: the discrete palettes of values from which we base ALL our visual design" — "like the presets on a car radio, not every option". Constrain to a small semantic set (e.g. 7 measure tokens) rather than arbitrary values. **=> highest-leverage consistency lever = make pages import the existing token set instead of hand-composing strings.** |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://www.w3.org/TR/WCAG22/ | W3C normative spec | Canonical reference; the per-SC Understanding pages (#1-5) carry the actionable detail. |
| https://www.w3.org/WAI/WCAG22/Understanding/focus-appearance.html | W3C 2.4.13 (AAA) | AAA, not in AA scope; surfaced to CONFIRM 2.4.13 is out of the AA requirement (prompt listed it under AA — corrected). |
| https://inclly.com/resources/axe-vs-lighthouse | practitioner comparison | WebFetch **timed out (60s)**; key numbers captured via WebSearch snippet (axe ~96 rules, Lighthouse 57; automated catches ~30-40% of WCAG issues). |
| https://designmd.cc/benchmarks/stripe | practitioner (Stripe token teardown) | Snippet: Stripe = **4px base unit**, generous whitespace, sohne-var font, 300/400 weights; section padding 64px. Corroborates USWDS constrained-scale thesis. |
| https://webaim.org/articles/contrast/ | authoritative (WebAIM) | Snippet: text AA = **4.5:1** normal / 3:1 large (>=24px or >=18.66px bold); contrast is the **#1 web a11y violation (83.6% of sites, WebAIM 2024 Million)**. |
| https://www.smashingmagazine.com/2023/11/creating-accessible-ui-animations/ | authoritative blog | Snippet: prefers-reduced-motion best practice — prefer opacity/color over movement; don't blunt-strip ALL motion. |
| https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion | official (MDN) | Snippet: mature, widely supported; reads OS-level setting. |
| https://www.eleken.co/blog-posts/modern-fintech-design-guide | practitioner | Snippet: fintech 2026 = trust/clarity/consistency; a robust design system "embeds accessibility and compliance into the foundation". |
| https://www.designsystemscollective.com/design-tokens-in-2026-beyond-colors-and-spacing-... | practitioner | Snippet: 2026 tokens "encode meaning, behavior, and context" — semantic > raw, exactly the `tokens.text.primary` role-naming the repo already uses. |
| https://accessibilityassistant.com/blog/.../how-to-apply-wcag-22-colour-contrast-... | practitioner | Snippet: dark mode must INDEPENDENTLY meet 4.5:1 text / 3:1 UI — recalc per theme. |
| https://blog.pope.tech/2025/12/08/design-accessible-animation-and-movement/ | practitioner (Dec 2025) | Snippet: 2025 reduced-motion guidance; audit modals/drawers/hovers individually. |

**URLs collected: 18** (7 read-in-full + 11 snippet-table). Gate floor (10+) cleared.

### Recency scan (2024-2026) — PERFORMED

Searched the 2024-2026 window explicitly (queries tagged `2026`, `2025`, `2024`). Findings:

1. **WCAG 2.2 is the current normative standard** (W3C Rec 2023-10, still current 2026) — the AA
   criteria new since 2.1 (2.4.11 focus-not-obscured, 2.5.7 dragging, 2.5.8 target-size 24px,
   3.3.8 accessible-auth) are the live delta a "WCAG AA in 2026" claim must address. The repo's
   `frontend.md:42` cites "WCAG 2.2 AAA" contrast targets and the `axe` npm script tags only
   `wcag2a,wcag2aa,wcag21a,wcag21aa` — **it does NOT include `wcag22aa`**, so the automated check
   currently UNDER-tests vs the 2.2 AA bar. (Autonomous fix: add `wcag22a,wcag22aa` to the tag
   list in `frontend/package.json:14`.)
2. **WebAIM 2024 Million**: color contrast is the #1 a11y violation (83.6% of sites). Reinforces
   that the contrast leg (slate-400/500 on risk numbers) is worth a targeted pass — but the repo's
   token comments already certify slate-400 ~7:1 and slate-500 ~4.6:1 on bg-navy-800/70, both
   passing 4.5:1 AA for normal text, so this is a PRECISION finding (risk-relevant numbers only),
   not a blanket failure.
3. **Axe ML coverage rising** (2025): automated detection up to ~57% by volume, projected ~70% by
   end-2025 — but the structural ceiling (keyboard order, reading order, alt-text quality) still
   needs manual/operator testing. Confirms the autonomous-vs-operator split below.
4. **Design tokens 2026** (Design Systems Collective): tokens now "encode meaning, behavior,
   context" (semantic naming) — exactly the `tokens.text.primary`/`tokens.status.error` role model
   the repo built in 47.5. No reason to redesign the token vocabulary; the gap is adoption.
5. **prefers-reduced-motion** (Pope Tech Dec-2025; Dacey Nolan May-2026): mature; the repo already
   honors it in `globals.css` + 2 files. Not a 53.2 blocker.

No 2024-2026 source overturns the W3C AA criteria or the "constrain to a token palette" thesis.
Net: the recency scan REINFORCES "adopt the existing tokens + states + ui/ primitives + a
focus-ring sweep" and corrects two prompt assumptions (2.4.13 is AAA; the axe tag list misses 2.2).

### 3-query evidence (mandatory variants)

- **Current-year (2026):** "design system consistency tokens... 2026 financial dashboard Linear
  Stripe" -> Eleken fintech-2026 + Design Systems Collective 2026 + DesignMD Stripe teardown;
  "WCAG 2.2 AA focus 2.4.13 / target 2.5.8" -> W3C 2.2 (current normative).
- **Last-2-year (2025/2024):** "axe-core vs lighthouse... 2025" -> inclly 2026 + Kairos 2025 +
  axe ML 57%->70% 2025; "WCAG focus contrast 3:1 dashboard... 2024" -> WebAIM 2024 Million 83.6%;
  Pope Tech Dec-2025 reduced-motion.
- **Year-less canonical:** "Understanding 1.4.11 Non-text Contrast" / "2.4.7 Focus Visible" /
  "2.5.8 Target Size" -> the W3C WAI Understanding pages (the founding normative docs);
  "USWDS design tokens" -> designsystem.digital.gov; "prefers-reduced-motion MDN/CSS-Tricks" ->
  the canonical CSS-feature docs. The source table is a deliberate mix of 2026/2025/2024 frontier
  and the year-less W3C/USWDS canon.

---

## Key findings (per-claim cited)

1. **For WCAG AA, focus = "any visible indicator" (2.4.7) + "not entirely obscured" (2.4.11).
   Size/contrast of the indicator is AAA (2.4.13), NOT AA.** "Any keyboard operable user interface
   has a mode of operation where the keyboard focus indicator is visible"; `:focus-visible` (C45)
   is a sufficient technique (W3C 2.4.7, accessed 2026-06-10). The prompt listed 2.4.13 under AA —
   it is **AAA** (W3C new-in-22, accessed 2026-06-10). => The project's existing
   `tokens.focusRing = ":focus-visible ring-2 ring-sky-400"` (design-tokens.ts:40) is exactly the
   AA-sufficient idiom; the gap is pages that don't apply it, not the idiom.
2. **The focus ring must still meet 1.4.11's 3:1 to count as "visible".** "the visual focus
   indicator for a component must have sufficient contrast against the adjacent background" — 3:1
   (W3C 1.4.11, accessed 2026-06-10). `ring-sky-400` on `bg-navy-800/70` clears 3:1, so the
   existing ring is compliant; new focus rings must reuse it, not invent a dimmer color.
3. **Target size AA = 24x24 CSS px, with a 24px-circle spacing exception.** "at least 24 by 24 CSS
   pixels"; undersized targets pass if a 24px-diameter circle centered on each does not intersect
   another target (W3C 2.5.8, accessed 2026-06-10). => `ui/Button` already bakes
   `min-h-[24px] min-w-[24px]` (Button.tsx:52) and OpsStatusBar action buttons already carry it
   (OpsStatusBar.tsx:309/319/329) — so adopting `ui/Button` is also the target-size fix.
4. **Sticky chrome must not ENTIRELY hide a focused element (2.4.11 AA); fix with `scroll-padding`.**
   Partial obscuring is allowed at AA (W3C 2.4.11, accessed 2026-06-10). => The fixed page-header /
   sidebar shell (frontend-layout.md §1) should set `scroll-padding-top` on the scroll container so
   a Tab-focused element below the fixed header is not covered — a low-risk global CSS addition.
5. **The highest-leverage consistency lever is a constrained semantic token palette that every
   surface imports.** "the discrete palettes of values from which we base ALL our visual design...
   like presets on a car radio, not every option" (USWDS, accessed 2026-06-10); Stripe = a single
   4px base scale (DesignMD snippet). => The repo already HAS this (`tokens.*`); consistency comes
   from ADOPTING it, exactly the 47.5 design intent (design-tokens.ts:1-12).
6. **Automated a11y tooling catches ~30-40% of WCAG issues; the rest needs manual keyboard/
   screen-reader testing.** axe runs ~96 rules, Lighthouse 57; both miss keyboard interaction,
   reading order, alt-text quality (axe-vs-lighthouse snippet + Deque #6, accessed 2026-06-10).
   => Credible 53.2 evidence = axe-core + Lighthouse on `/login` (the only unauthenticated route)
   AUTONOMOUSLY, plus an explicit OPERATOR-TO-CONFIRM keyboard-nav + authed-route section — the
   masterplan's own `live_check_53.2.md` shape.
7. **Dark mode must independently meet 4.5:1 text / 3:1 UI; contrast is the #1 violation.** WebAIM
   2024 (83.6% of sites). The repo's token comments certify the slate tiers on bg-navy-800/70:
   slate-100 >=13:1, slate-200 >=12:1, slate-300 >=10:1, slate-400 >=7:1, slate-500 >=4.6:1
   (design-tokens.ts:17-21) — all clear 4.5:1 for normal text. => Contrast is NOT a blanket AA
   failure here; the only at-risk case is slate-400/500 used on RISK-RELEVANT numbers (frontend.md:46
   already forbids that), which is a targeted grep, not a global swap.

---

## WCAG-AA concrete checklist (dark dashboard) — status in THIS repo

| SC | Level | Requirement | Repo status (file:line) | 53.2 action |
| --- | --- | --- | --- | --- |
| **1.4.3** Contrast (text) | AA | 4.5:1 normal / 3:1 large | PASS by token (design-tokens.ts:17-21 certifies all slate tiers >=4.6:1). Risk: slate-400/500 on risk numbers. | Targeted grep only (precision, not blanket) |
| **1.4.11** Non-text contrast | AA | 3:1 UI + focus + graphics | Focus ring `ring-sky-400` clears 3:1 (design-tokens.ts:40). | Reuse the ring; don't invent dimmer ones |
| **2.1.1** Keyboard | A | All function via keyboard | Native buttons/links OK; `<details>` tabs OK; custom click-divs are the risk. | Operator keyboard-nav confirm (manual-only) |
| **2.4.1** Bypass blocks | A | Skip link | **PASS** — skip-link present (layout.tsx:25-28, `#main` sr-only + focus ring). | None (do not re-add) |
| **2.4.7** Focus visible | AA | Any visible focus indicator | **PARTIAL** — 9 page files have `onClick` buttons with NO `:focus-visible` in-file (agents, cron, backtest, observability, paper-trading/manage, performance, settings, signals, sovereign). | **FOCUS-RING SWEEP** (apply `tokens.focusRing`) |
| **2.4.11** Focus not obscured | AA | Focused el not entirely hidden by sticky chrome | Fixed header/sidebar shell could cover a focused element below the fold. | Global `scroll-padding-top` on scroll zone (low-risk) |
| **2.5.8** Target size | AA | 24x24 CSS px (+ exceptions) | `ui/Button` + OpsStatusBar buttons already >=24px (Button.tsx:52; OpsStatusBar.tsx:309+). Hand-rolled `py-0.5`/`py-1` buttons may be <24px. | Adopt `ui/Button` on action sites (also fixes size) |
| **3.1.1** Language of page | A | `lang` on `<html>` | **PASS** — `lang="en"` (layout.tsx:20). | None |
| **4.1.2** Name/role/value | A | aria on custom controls | Tab bars vary: /reports has `role=tablist` (44.4); others lack it. | Mirror the /reports tablist pattern (follow-up) |
| (no-emoji) | repo rule | Phosphor only, no emoji | **PASS** — grep found ZERO true emoji / ZERO ✓✗⚠ symbol-emoji. Every non-ASCII hit is a typographic arrow (`→ ↔ ↑↓ ↵`) in copy/keyboard-hints, which is valid typography (and `↑↓ ↵` in CommandPalette.tsx:184 are good a11y hints). | **NONE — do not manufacture an emoji item** |

**Note on the prompt's "emoji in UI (grep)" item:** the audit DISPROVES it. The earlier broad
codepoint range matched arrows; the narrow emoji + dingbat ranges return 0 hits across all of
`src/`. Reporting an emoji-removal task would be a false finding.

---

## Internal code inventory (file:line anchors)

| File | Lines | Role | Status |
| --- | --- | --- | --- |
| `frontend/src/lib/design-tokens.ts` | 14-57 | THE token source (47.5). `text` (5 contrast tiers), `surface`, `border`, `hover`, **`focusRing`** (:40 `:focus-visible ring-2 ring-sky-400`), `transition` (base/state/icon), `status` (5 pre-attentive pills). Every value a complete literal class string (JIT-safe). | THE consistency vocabulary — **adopt it** |
| `frontend/src/components/ui/Button.tsx` | 1-65 | Canonical Button. 4 variants (primary/secondary/ghost/danger), bakes `tokens.focusRing` + `min-h/w-[24px]` (:52, the 2.5.8 fix) + `active:scale-95`. Doc (:2-4) names ~30 inline-button sites it should replace. | **0 consumers** — adopt on action sites |
| `frontend/src/components/ui/StatusBadge.tsx` | 1-32 | Canonical status pill, consumes `tokens.status`. | **0 consumers** — adopt where inline pills exist |
| `frontend/src/components/states/index.ts` + `LoadingState`/`EmptyState`/`ErrorState`/`OfflineState`/`StaleDataState` | barrel | 44.1 states library. | `EmptyState` used by 2 pages (performance:7, reports:26); `LoadingState`/`ErrorState` by **0** |
| `frontend/src/app/layout.tsx` | 20, 25-28 | Root layout: `lang="en"` + `dark` (3.1.1 PASS); skip-link `#main` sr-only+focus-ring (2.4.1 PASS). | Reference — do NOT re-add these |
| `frontend/src/app/globals.css` | (prefers-reduced-motion present) | Honors reduced-motion globally + `scrollbar-thin` defn. | Add global `scroll-padding-top` for 2.4.11 |
| `frontend/src/components/OpsStatusBar.tsx` | 304-329 | Action buttons (pause/flatten/resume) ALREADY carry `focus-visible:ring-2 ring-sky-400` + `min-h/w-[24px]` — the proven idiom design-tokens.ts:38 references. Status pills (:251/291) are display-only spans (`py-0.5`), NOT interactive targets — so NOT a 2.5.8 violation. | Reference idiom; the UX-2 "target-size violation" is stale/resolved |
| Page shell consistency | all `app/**/page.tsx` | `h-screen overflow-hidden` + `scrollbar-thin` present on every top-level page; `paper-trading/*` sub-routes correctly inherit it from `paper-trading/layout.tsx:326/439`; `/login` is intentionally shell-less. | **CONSISTENT already** — no shell work needed |
| Inline error banners | 13 pages, ~36 divs | `border-rose-*` banner divs: backtest(10), cron(5), settings(4), paper-trading/layout(3), sovereign(2), signals(2), performance(2), manage(2), page.tsx(2), strategy(1), reports(1), observability(1), agents(1). | **MIGRATE -> `ErrorState`** (biggest mechanical win) |
| Inline animate-pulse | 10 files | page.tsx, settings, agents, backtest + 6 components (excl. Skeleton.tsx/states which legitimately use it). | -> `LoadingState`/`SkeletonPulse` (follow-up) |
| Zinc strays (rule-1 violation) | 4 real | AnalysisProgress.tsx(20), CommandPalette.tsx(13), DataTable.tsx(7), LiveBadge.tsx(2) + states/LoadingState(3)/StaleDataState(1) + reports(1)/performance(1). | **SWAP zinc->navy/slate** (small, grep-verifiable) |
| Focus-ring gaps | 9 pages | agents, cron, backtest, observability, paper-trading/manage, performance, settings, signals, sovereign have `onClick` buttons but NO `:focus-visible` in-file. | **FOCUS-RING SWEEP** (the true AA gap) |
| DataTable (UX-6) | 3 of 4 | Adopted: trades, positions, reports. Gap: /backtest, /cron still raw `<table>`. | Follow-up (higher-risk, larger diff) |
| Slate token-able sites | ~1246 | text-slate-100(72)/200(126)/300(259)/400(372)/500(417) + hover:bg-navy-700/40(21). | **DEFER** — full migration is the unbounded part |
| `frontend/package.json` | 14 | `axe` script tags `wcag2a,wcag2aa,wcag21a,wcag21aa` — **missing `wcag22a,wcag22aa`**. | **ADD 2.2 tags** (1-line, raises the bar to spec) |

---

## Prioritized all-pages unification worklist

Ranked by (value x grep-verifiability) / risk. Each: delta -> fix -> autonomous? -> risk.

| # | Delta (where it diverges) | Fix | Autonomous + grep-verifiable? | Risk |
| --- | --- | --- | --- | --- |
| **P1** | ~36 inline `border-rose-*` error-banner divs across 13 pages (vs the `ErrorState` primitive that 0 pages use) | Replace each with `<ErrorState message=.. onRetry=.. />`, preserving the existing retry handler + dismiss | **YES** — grep `border-rose-` in `src/app` trends toward 0; build+tsc prove it | LOW-MED — must preserve each banner's retry/dismiss behavior + the exact error string (DO-NO-HARM) |
| **P2** | 9 pages with `onClick` buttons but no `:focus-visible` (2.4.7 AA gap) | Add `tokens.focusRing` (or adopt `ui/Button`) on the interactive buttons | **YES** — grep `focus-visible:ring` per file; axe on /login | LOW — additive class, no behavior change |
| **P3** | `axe` script under-tests (no `wcag22aa`) | Add `wcag22a,wcag22aa` to package.json:14 tag list | **YES** — diff is one string; `npm run axe` runs the 2.2 rules | NONE |
| **P4** | 4 zinc-palette files (rule-1 violation): AnalysisProgress(20), CommandPalette(13), DataTable(7), LiveBadge(2) | Swap `zinc-*` -> matching `navy-*`/`slate-*` token | **YES** — grep `zinc-` in `src/` trends to 0 | LOW — visual-only; operator confirms shade |
| **P5** | `ui/Button` has 0 consumers despite ~30 inline button strings | Adopt `<Button variant=..>` on the top action sites (OpsStatusBar already matches the idiom; start with page.tsx/GoLiveGateWidget action buttons) | **YES** — grep `from "@/components/ui"` consumer count rises | MED — must preserve onClick/disabled/aria; visual diff -> operator |
| **P6** | 2.4.11: fixed header/sidebar can fully cover a focused element below fold | Global `scroll-padding-top` on the `.scrollbar-thin` scroll zone (globals.css) | Partial — CSS is autonomous; "not obscured" needs operator Tab-through to confirm | LOW |
| P7 | `LoadingState` 0 adopters; ~10 files inline `animate-pulse` | Swap inline spinners -> `<LoadingState>` / `SkeletonPulse` | YES (grep) | MED — larger diff; defer |
| P8 | DataTable on /backtest + /cron (UX-6 4-of-4) | Wrap raw `<table>` in `<DataTable>` | YES (grep `<DataTable`) | HIGH — large refactor of complex tables; defer |
| P9 | ~1246 hand-rolled slate/hover strings not using `tokens.*` | Migrate to `tokens.text.*`/`tokens.hover.*` | YES but huge | HIGH churn — DEFER (the unbounded part) |
| P10 | Tab-bar aria inconsistency (only /reports has role=tablist) | Mirror the 44.4 tablist pattern app-wide | Partial | MED — defer to a dedicated a11y step |
| OP1 | Lighthouse a11y >=95 / perf on AUTHED routes | Operator runs `npm run lighthouse:auth-home` behind NextAuth | **NO — operator only** (NextAuth wall) | n/a |
| OP2 | Keyboard nav + reading order + screen-reader on every route | Operator manual pass (the ~60% automation can't see) | **NO — operator only** | n/a |

---

## Recommended BOUNDED 53.2 scope (land THIS cycle)

Land **P1 + P2 + P3 + P4** (and P6 if cheap). Rationale: these are the highest consistency value,
are fully grep-verifiable + build/tsc-checkable autonomously, carry LOW-to-MEDIUM risk, and
directly satisfy the four success_criteria:

- **SC-1** (audit + unification deltas vs design-tokens.ts + ui/ primitives): this brief IS the
  documented all-pages audit; P1-P5 ARE the named deltas.
- **SC-2** (no emoji [already PASS], Recharts dark theme [unchanged], scrollbar-thin [already
  consistent], error/loading/empty preserved): P1 migrates error banners to `ErrorState` WHILE
  preserving each retry/dismiss + message (the DO-NO-HARM core).
- **SC-3** (build + tsc pass; a11y check recorded; no regression): `npm run build` + `npx tsc
  --noEmit` + `npm run axe` (now with 2.2 tags via P3) on /login = the autonomous a11y evidence;
  P2's focus-ring sweep is the concrete AA improvement.
- **SC-4** (`live_check_53.2.md`: build/types + a11y evidence + operator-to-confirm visual): P1-P4
  give the autonomous half; OP1/OP2 are the explicit operator section.

DEFER (documented follow-ups, NOT this cycle): P5 (partial — only top sites), P7 (LoadingState
sweep), P8 (DataTable /backtest+/cron — large), P9 (full slate-token migration — the unbounded
part), P10 (app-wide tablist aria). These map cleanly to the existing pending steps **44.2 / 44.9**
("Mobile + a11y + states-library global polish") so the follow-up home already exists in the plan.

**Why bounded works here:** because the foundation is already built, each P1-P4 delta is a swap to
an EXISTING tested primitive/token, not net-new design. The unbounded risk (rewriting 1246 sites)
is explicitly deferred. This is the "presets, not every option" discipline (USWDS) applied to the
scope itself.

---

## DO-NO-HARM risks

1. **Error-banner migration must preserve behavior (P1).** Each inline rose banner often wires a
   retry callback, a dismiss button, and a specific error string. `ErrorState` must receive the
   SAME `message` + `onRetry` so the page's error UX is identical. Do NOT drop the retry path —
   the frontend.md error-state rule + the polling-failure-limit rule (frontend.md:56-57) depend on
   it. Verify each touched page still renders its error/loading/empty states.
2. **Focus-ring additions are additive only (P2).** Adding `tokens.focusRing` must not change
   onClick/disabled/aria. Reuse the EXISTING `ring-sky-400` (1.4.11 3:1-compliant); do not invent
   a dimmer ring that fails contrast.
3. **Zinc->navy swaps are visual (P4).** A wrong shade reads "off" (frontend.md rule 1). These are
   in the OPERATOR-visual bucket for final confirmation even though the grep is autonomous.
4. **`ui/Button` adoption (P5) changes rendered markup.** Defer broad adoption; if any is landed,
   keep it to the top action sites and preserve every prop. Visual diff -> operator live_check.
5. **No backend/data/behavior change.** 53.2 is a frontend-consistency + a11y step: no API shape
   change, no Recharts data change, no polling/auth change. The page-shell + scrollbar-thin are
   ALREADY consistent — do not "fix" what passes (avoid churn).
6. **Don't manufacture findings.** Emoji (0 true emoji), skip-link (present), `lang` (present),
   page-shell (consistent), animation-duration (4 values, token-mappable, not a problem) are all
   already-satisfied — the brief flags them so the GENERATE phase doesn't waste effort or claim a
   false fix.
7. **axe runs on /login only (NextAuth wall).** Autonomous axe/Lighthouse evidence is limited to
   the one unauthenticated route; authed-route a11y + all keyboard/reading-order checks are the
   operator's `live_check_53.2.md` section (W3C #6: automation catches only ~30-40%). Frame the
   contract's a11y evidence honestly as "axe clean on /login + operator-to-confirm on authed
   routes", not "WCAG AA verified app-wide".

---

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] **>=5 authoritative external sources READ IN FULL via WebFetch** — 7 (W3C new-in-22; W3C
  2.5.8; W3C 1.4.11; W3C 2.4.11; W3C 2.4.7; Deque axe-core; USWDS design-tokens). 5 are W3C/official
  top-of-hierarchy.
- [x] **10+ unique URLs total** — 18 (7 full + 11 snippet).
- [x] **Recency scan (last 2 years) performed + reported** — yes, 5 findings; corrects 2 prompt
  assumptions (2.4.13 is AAA; axe tag list misses 2.2). No source overturns the W3C AA canon.
- [x] **Full pages read (not abstracts)** for the read-in-full set — yes (W3C Understanding pages,
  Deque, USWDS read in full).
- [x] **3 query variants (2026 / 2025 / year-less)** — documented above; source table mixes 2026
  frontier and year-less W3C/USWDS canon.
- [x] **file:line anchors for every internal claim** — yes (design-tokens.ts:14-57/40;
  ui/Button.tsx:2-4/52; ui/StatusBadge.tsx:12; states/index.ts; layout.tsx:20/25-28;
  OpsStatusBar.tsx:251/291/304-329; paper-trading/layout.tsx:326/439; CommandPalette.tsx:184;
  package.json:14; plus the per-page grep tables).

Soft checks:
- [x] Internal exploration covered every relevant module (23 page routes, the 3 ui/ + 5 states/
  primitives, design-tokens, root layout, globals.css, OpsStatusBar, DataTable, the a11y scripts).
- [x] Contradictions / corrections noted (2.4.13 AAA-not-AA; axe under-tagged; emoji/skip-link/
  lang/shell are FALSE-or-already-done findings explicitly flagged).
- [x] All claims cited per-claim (URL + access date inline in Key Findings; file:line inline in
  the inventory + worklist).

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 11,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "gate_passed": true
}
```
