# Research Brief -- step 91.13

**Topic:** Dashboard KPI stat-card visual consistency -- when a highlighted/glow
treatment is appropriate on ONE card vs a uniform KPI row (hero-metric emphasis
vs uniform-row conventions in dashboard design).

**Tier:** simple (caller-specified). Audit-class: NO (`coverage.dry` not required).

**Internal scope (caller-specified):**
- `frontend/src/components/CostDashboard.tsx` -- Total Cost box vs its 3 siblings
- `frontend/src/components/BentoCard.tsx` -- the `glow` prop -> `alpha-score-glow` class
- `frontend/src/app/globals.css` -- the pulse-glow animation definition

---

## STATUS ENVELOPE (phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 25,
  "urls_collected": 32,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 3,
    "dry": false
  },
  "gate_passed": true
}
```

---

## Search-query composition (three-variant discipline)

| Variant | Query run |
|---|---|
| Current-year frontier (2026) | `dashboard KPI card visual hierarchy hero metric emphasis 2026` |
| Last-2-year window (2025) | `dashboard card glow animation accessibility distraction 2025 KPI emphasis consistency` |
| Year-less canonical | `dashboard design uniform KPI row visual consistency emphasis one card` |
| Year-less canonical | `Nielsen Norman Group dashboard visual hierarchy emphasis consistency cards` |
| Year-less canonical | `when to visually emphasize one KPI card hero metric versus uniform card set data visualization guidance` |

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://www.w3.org/WAI/WCAG22/Understanding/pause-stop-hide.html | 2026-08-20 | Official standard (W3C/WAI) | WebFetch, full | SC 2.2.2 normative: *"For any moving, blinking or scrolling information that (1) starts automatically, (2) lasts more than five seconds, and (3) is presented in parallel with other content, there is a mechanism for the user to pause, stop, or hide it unless ... essential."* Scope covers *"content in which the visible content conveys a sense of motion ... animations"*. Fetch confirms: **"The five-second threshold would not exempt indefinite animations."** "Essential" exception applies only if removal *"would fundamentally change the information or functionality"*. |
| https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion | 2026-08-20 | Official docs (MDN) | WebFetch, full | `reduce` means the user *"prefers an interface that removes, reduces, or replaces motion-based animations"*. Accessibility concerns: *"Such animations can trigger discomfort for those with vestibular motion disorders"*; *"Animations such as scaling or panning large objects can be vestibular motion triggers."* Canonical pattern is a later-in-source-order `@media (prefers-reduced-motion: reduce)` override, same specificity. **Baseline Widely available since January 2020.** |
| https://www.smashingmagazine.com/2025/09/ux-strategies-real-time-dashboards/ | 2026-08-20 | Authoritative blog (2025 -- recency anchor) | WebFetch, full | Endorses glow **as a transient change-signal, not a permanent state**: *"A soft pulse around a changing metric can signal activity without overwhelming the viewer"*, with *"smooth transitions of 200 to 400 milliseconds"*. Persistent emphasis is done with *"bold, large font"* high-contrast cards for *"Primary KPIs"* -- typography, not motion. Explicit directive: *"Animations follow motion-reduction preferences to support users with vestibular sensitivities."* Don'ts include *"Rely on animation as the only signal for priority."* |
| https://data-goblins.com/power-bi/kpi-templates | 2026-08-20 | Industry practitioner | WebFetch, full | *"One key consideration for KPI visuals is to not have too many"*; *"Having too many of these callouts is counter-productive, as it dilutes their value."* Emphasis should be **conditional/exception-driven** -- neutral when on-target, *"a deep red when off-target"* -- and color used only *"to steer attention"*. Caps a KPI strip at 3-4 callouts. |
| https://www.activecampaign.design/docs/components/data-visualization/kpi-card | 2026-08-20 | Official design-system docs | WebFetch, full | KPI-bar consistency is enforced at the token level: a mandated supplementary-color order (*"AC Blue, followed by Dusk, Maroon, and Violet in that order"*) and a scarcity rule -- *"only one red metric should appear in any KPI bar"*. **The spec does not provide any mechanism for styling an individual card differently within a KPI bar set** -- i.e. a mature design system offers uniformity + semantic color, not a per-card decorative variant. |
| https://www.nngroup.com/articles/dashboards-preattentive/ | 2026-08-20 | Authoritative blog (NN/g) | WebFetch, full | Pre-attentive channels ranked: *"we are quite adept at estimating how lengths compare, and we can also accurately estimate position in a 2D space"*; area/angle are weak -- *"we are not good at judging how much bigger the large rectangle is."* Color/shape carry **categorical, not quantitative**, meaning. **[PARTIAL-NEGATIVE / adversarial-adjacent] The article offers NO guidance on motion or flicker as an attention channel and does not address KPI-card repetition** -- so the canonical NN/g pre-attentive literature does not license a motion-based emphasis. |
| https://www.smashingmagazine.com/2021/11/dashboard-design-research-decluttering-data-viz/ | 2026-08-20 | Authoritative blog (year-less canonical) | WebFetch, full | **[ADVERSARIAL] Pushes back on reflexive uniformity**: *"decluttering for the sake of decluttering is a poor design maxim."* Complexity should be matched to user need via research, not trimmed on principle. Prescribes *"Clear information hierarchy"*, *"Two to three colors"*, *"White space, plenty of it"*. Warns color semantics mislead (red *"is often associated with danger, failure, and poor performance"*). Notably contains **no warning about decorative shadows/glows** -- the anti-ornament case is not universal in the literature. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://carbondesignsystem.com/data-visualization/dashboards/ | Official design system | **Fetch ATTEMPTED and FAILED** -- returned "Content truncated due to length", no extractable text. Not counted. |
| https://m3.material.io/styles/elevation/applying-elevation | Official design system | **Fetch ATTEMPTED and FAILED** -- JS-rendered; only the page title returned. Not counted. |
| https://www.epcgroup.net/power-bi-kpi-visuals-dashboard-guide-2026 | Industry | Vendor-consultancy SEO tier; superseded by data-goblins + ActiveCampaign |
| https://tabulareditor.com/blog/kpi-card-best-practices-dashboard-design | Industry | Same ground as data-goblins (read in full) |
| https://nastengraph.substack.com/p/anatomy-of-the-kpi-card | Practitioner blog | Card anatomy only; no emphasis-vs-uniformity rule |
| https://www.intelligentgraphicandcode.com/design/dashboard-design/dashboard-layout | Blog | Snippet already yielded the position-decay finding |
| https://www.setproduct.com/blog/dashboard-ui-design | Blog | Layout-metrics focus (240-280px sidebar, 4-6 KPI strip) |
| https://www.uxpin.com/studio/blog/dashboard-design-principles/ | Blog | General principles, no card-emphasis rule |
| https://uxpilot.ai/blogs/dashboard-design-principles | Blog | Restates consistency principle |
| https://figr.design/blog/dashboard-design-best-practices | Blog | Restates consistency principle |
| https://www.aufaitux.com/blog/dashboard-design-principles/ | Blog | Listicle tier |
| https://thedan.design/insights/dashboard-design-principles-best-practices-to-enhance-your-data-analysis/ | Blog | Listicle tier |
| https://www.elevenspace.co/blog/eleven-rules-to-follow-to-improve-your-dashboard-design | Blog | Listicle tier |
| https://artofstyleframe.com/blog/visual-hierarchy-ui-design-principles/ | Blog | Generic visual-hierarchy, not dashboard-specific |
| https://www.datwin.com/blog/the-dashboard-as-a-product-treating-analytics-like-user-interfaces | Blog | Product-framing, not card spec |
| https://www.datacamp.com/tutorial/dashboard-design-tutorial | Tutorial | Tool-specific |
| https://www.datacamp.com/tutorial/power-bi-kpi | Tutorial | Tool-specific |
| https://rishandigital.com/power-bi/creating-kpi-and-card-visuals/ | Tutorial | Tool-specific |
| https://gallery.fanruan.com/kpi-card-example | Vendor gallery | Marketing tier |
| https://improvado.io/blog/dashboard-design-guide | Vendor blog | Marketing tier |
| https://keomarketing.com/marketing-analytics-attribution-guide-150191-2 | Vendor blog | Marketing-domain, off-topic |
| https://www.clearpointstrategy.com/blog/kpi-dashboard-best-practices | Vendor blog | Marketing tier |
| https://www.designstudiouiux.com/blog/dashboard-ui-design-guide/ | Agency blog | Marketing tier |
| https://www.aidesigner.ai/blog/how-to-design-a-dashboard-ui | Blog | AI-generated content tier |
| https://www.animbits.dev/docs/animations/cards/hover-glow | Component library | Implementation recipe; **hover**-triggered glow (transient), not an infinite loop -- confirms the transient framing |

**URL tally:** 7 read in full + 25 snippet-only = **32 unique URLs**.

## Recency scan (2024-2026)

**Performed.** Query: `dashboard card glow animation accessibility distraction 2025 KPI emphasis consistency`,
plus the 2026 frontier query above.

**Result: 3 new findings in the 2024-2026 window, and they SHARPEN rather than
supersede the canonical guidance.**

1. Smashing Magazine 2025-09 (read in full) is the only source in the corpus that
   addresses glow-on-a-metric-card directly. It licenses a *"soft pulse"* but frames
   it as a **200-400ms transient tied to a data change**, and explicitly assigns
   *persistent* primary-KPI emphasis to typography/contrast instead.
2. The same 2025 source states motion-reduction compliance as a flat requirement,
   not a nice-to-have -- consistent with MDN's Baseline-since-2020 status for
   `prefers-reduced-motion`.
3. The 2026 frontier results converge on a **4-6 card uniform KPI strip** as the
   standard top-of-dashboard pattern, with emphasis delivered by **position**
   (first/top-left) rather than by decoration. No 2024-2026 source advocates a
   permanently animated card in a uniform row.

Nothing in the window contradicts the older canonical material (Few/Tufte-lineage
consistency, NN/g pre-attentive ranking).

## Key findings

1. **A uniform row already encodes emphasis positionally -- decoration is
   redundant.** *"When the same element repeats across a dashboard, i.e., a row of
   KPI cards ... attention is strongest at the first item and drops off from left
   to right"* (intelligentgraphicandcode.com, snippet). Total Cost is **already
   first** in the pyfinagent row.
2. **Consistency across a card set is the default rule.** A design system's answer
   to "make one card special" is semantic color with a scarcity rule
   (*"only one red metric should appear in any KPI bar"*) -- not a per-card
   decorative variant, which the spec simply does not offer
   (activecampaign.design, read in full).
3. **Emphasis dilutes when overused.** *"Having too many of these callouts is
   counter-productive, as it dilutes their value"* (data-goblins, read in full).
   The corollary holds in reverse: a single always-on glow with no state meaning
   trains the eye to ignore it.
4. **Motion is not a supported emphasis channel in the pre-attentive literature.**
   NN/g's pre-attentive article ranks length/position/color and is **silent on
   motion** (nngroup.com, read in full) -- the emphasis case must be made on
   typography/position/color, not animation.
5. **Glow is legitimate as a TRANSIENT change-signal.** *"A soft pulse around a
   changing metric can signal activity without overwhelming the viewer"* at
   *"200 to 400 milliseconds"* (smashingmagazine 2025, read in full). pyfinagent's
   is `3s infinite` -- an order of magnitude longer and unbounded.
6. **An infinite decorative animation is in WCAG 2.2 SC 2.2.2 scope and has no
   exception here.** It starts automatically, exceeds five seconds (unbounded), and
   is presented in parallel with other content; *"the five-second threshold would
   not exempt indefinite animations"* (w3.org, read in full). The "essential"
   exception fails because removing the glow changes no information.
7. **Reduced-motion is a hard requirement, not optional.** *"Animations follow
   motion-reduction preferences"* (smashingmagazine 2025); MDN confirms the feature
   is Baseline Widely available since Jan 2020, so there is no support argument for
   omitting the guard.
8. **[ADVERSARIAL] Uniformity is not an absolute.** *"Decluttering for the sake of
   decluttering is a poor design maxim"* (smashingmagazine 2021, read in full). A
   genuinely dominant metric MAY be differentiated -- but the differentiation should
   be hierarchy-bearing (size/contrast/span), which is exactly what the project's own
   `frontend-layout.md` §4 already prescribes.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| `frontend/src/components/CostDashboard.tsx` | 216 | Cost & Token Usage tab. 4-card KPI row at `:84-112`. `<BentoCard glow>` on **Total Cost only** (`:85`); siblings Total Tokens (`:91`), LLM Calls (`:97`), Deep Think Calls (`:103`) are plain. All four use identical `font-mono text-3xl font-bold` values (`:87`, `:93`, `:99`, `:105`). | LIVE -- the inconsistency under review |
| `frontend/src/components/BentoCard.tsx` | 26 | Shared card primitive. `glow?: boolean` (`:9`, `:13`) -> `glow && "alpha-score-glow"` (`:19`). Base `rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg` (`:18`). | LIVE, 23 importing files |
| `frontend/src/app/globals.css` | 183 | `@keyframes pulse-glow` (`:61-64`), `.alpha-score-glow { animation: pulse-glow 3s infinite ease-in-out; }` (`:66-68`). Sky-400 `rgba(56,189,248,...)`; box-shadow 20px -> 40px + 60px. | LIVE -- **no `prefers-reduced-motion` guard** |
| `frontend/src/app/performance/page.tsx` | -- | 2nd glow site (`:178`): Win Rate card, `col-span-12 md:col-span-4` (`:177`) -- **same span and same `text-5xl` as its Avg Return sibling** (`:189`, `:193`). | LIVE -- same 1-of-N-uniform pattern |
| `frontend/src/components/GlassBoxCards.tsx` | -- | 3rd glow site (`:31`): `AlphaScoreCard`, `text-7xl` (`:36`) -- the only site with genuine size hierarchy, and the class's namesake. | **DEAD -- 0 render sites** (see below) |
| `frontend/src/app/reports/[ticker]/page.tsx` | -- | Imports only `InvestmentThesisCard, RisksCard, ScoringMatrixCard` from GlassBoxCards (`:21-25`) -- **not** `AlphaScoreCard`. | LIVE |
| `frontend/src/components/PdfDownload.tsx` | -- | Imports `InvestmentThesisCard, RisksCard, ScoringMatrixCard` (`:12`) -- **not** `AlphaScoreCard`. | LIVE |
| `.claude/rules/frontend-layout.md` | -- | §4 "Metric Grids" + §9 hierarchy table already binding on this decision (see Application). | Project rule |

### The three `glow` call sites (complete -- grep over `*.tsx`/`*.ts`/`*.css`)

| Site | Context | Row shape | Hierarchy-bearing? |
|---|---|---|---|
| `CostDashboard.tsx:85` | Total Cost, 1st of 4 in `grid-cols-1 sm:grid-cols-2 lg:grid-cols-4` (`:84`) | 1-of-4 **uniform** | **No** -- identical `text-3xl`, identical span |
| `app/performance/page.tsx:178` | Win Rate, `md:col-span-4` in a `grid-cols-12` (`:176-177`) | 1-of-N **uniform** | **No** -- identical `text-5xl`, identical span |
| `GlassBoxCards.tsx:31` | `AlphaScoreCard`, `text-7xl` (`:36`) | **DEAD CODE** | Yes -- but unreachable |

### Dead-code finding (measured, not inferred)

`AlphaScoreCard` is exported at `GlassBoxCards.tsx:25` and has **0 render sites**
repo-wide (`grep -rn "<AlphaScoreCard" frontend/src --include="*.tsx"` -> 0; its
three siblings return 2 each). Both consumers of the module
(`app/reports/[ticker]/page.tsx:21-25`, `components/PdfDownload.tsx:12`) import the
other three exports only.

**Consequence:** the CSS class `.alpha-score-glow` is named for the one card where
the treatment was hierarchy-bearing, and that card no longer renders. Every
*surviving* use of the glow is a 1-of-N card in a uniform row with no accompanying
size/span/contrast differentiation. The effect has outlived its rationale.

### Motion-preference drift (measured)

`globals.css` guards `prefers-reduced-motion: reduce` for the NumberFlow tint
**only** (`:178-183`, `animation: none !important`), and its comment at `:147-151`
reasons explicitly about WCAG SC 2.2.2/2.3.3 for that one animation -- concluding
*"SC 2.2.2 satisfied (900ms << 5s ceiling)"*. **That same reasoning inverts for
`pulse-glow`, which is `3s infinite`.** Four infinite animations carry no guard:
`shimmer` (`:48-58`), `pulse-glow` (`:61-68`), `spin-slow` (`:71-77`),
`gemini-bounce` (`:101-104`). Repo-wide, `prefers-reduced-motion` appears in only 4
places (`globals.css:147` comment, `globals.css:178` the one live rule,
`app/page.tsx:24` comment, `components/paper-trading/cockpit-helpers.tsx:38`
comment) -- exactly **one enforced rule**. Tailwind's `motion-safe:` /
`motion-reduce:` variants are used **nowhere** in `frontend/src`.

## Consensus vs debate (external)

**Consensus (5 of 7 read-in-full sources):** a KPI strip is a uniform component
set; emphasis belongs to position, size/contrast, and semantically-scarce color;
persistent decorative differentiation of one card is not a supported pattern in
mature design systems.

**Debate:** Smashing 2021 [ADVERSARIAL] warns against uniformity-as-dogma
(*"decluttering for the sake of decluttering is a poor design maxim"*), and
Smashing 2025 does license a pulse -- but as a **200-400ms transient tied to a data
change**, which is a different mechanism from a 3s infinite loop. The two positions
reconcile: differentiation is legitimate when it is hierarchy-bearing and
information-bearing; an always-on ambient glow is neither.

**Not in debate:** WCAG SC 2.2.2 scope and the reduced-motion requirement. No source
in the corpus disputes either.

## Pitfalls (from literature)

- **Diluted emphasis** -- *"Having too many of these callouts is counter-productive,
  as it dilutes their value"* (data-goblins). An always-on glow with no state meaning
  becomes chrome.
- **Motion as the sole priority signal** -- explicit don't in Smashing 2025.
- **Vestibular harm** -- MDN: animations *"can trigger discomfort for those with
  vestibular motion disorders"*; a 20px->60px box-shadow bloom is a scaling-class
  trigger.
- **Silent WCAG regression** -- unbounded auto-starting motion with no pause
  mechanism fails SC 2.2.2 and no automated check in this repo tests for it.
- **Over-correction** -- removing all differentiation is also a documented failure
  mode (Smashing 2021).

## Application to pyfinagent

1. **`CostDashboard.tsx:85` is the weakest of the three uses.** The glow is the
   *only* differentiator: Total Cost shares `text-3xl` and grid span with all three
   siblings (`:87` vs `:93`/`:99`/`:105`). Under finding 1, position already gives it
   first-item salience, so the animation adds no hierarchy -- only motion.
2. **The project's own rules already decide this.** `frontend-layout.md` §4 puts
   emphasis in typography (`text-2xl font-bold text-slate-100`) and color tokens; §9
   lists *"Consistent micro-interactions -- Material Design 3, Apple HIG -- Same
   scrollbar, same hover states, same transitions everywhere"*. A one-off infinite
   animation on one card in a uniform row is the inconsistency that rule names.
   External consensus (finding 2) corroborates the internal rule.
3. **Two coherent options for PLAN; both are defensible, and they differ in scope.**
   - **(a) Uniform row.** Drop `glow` at `CostDashboard.tsx:85` (and, for the same
     reason, `performance/page.tsx:178`). Keep the `glow` prop on `BentoCard.tsx:13`
     for a future hierarchy-bearing hero. Smallest diff; matches findings 1-4.
   - **(b) Real hero treatment.** If Total Cost genuinely is the hero, differentiate
     with span + type scale per `frontend-layout.md` §4 (e.g. wider grid span,
     larger value) instead of motion -- per finding 8 and Smashing 2025's
     "persistent emphasis = bold, large font".
   Option (a) is the lower-risk reading of the caller's "visual consistency" framing;
   (b) is available if Main judges Total Cost to be a true hero metric.
4. **Independent of (a)/(b): the `prefers-reduced-motion` gap is a real defect and
   is WIDER than this step's scope.** `.alpha-score-glow` (`globals.css:66-68`) is
   `3s infinite` with no guard -> WCAG 2.2 SC 2.2.2 (finding 6). Note that if option
   (a) removes both live call sites, the class becomes **unreferenced** and the WCAG
   exposure goes to zero without a CSS change -- but `shimmer`, `spin-slow` and
   `gemini-bounce` remain unguarded. **Recommend Main queue the motion-guard sweep as
   a separate masterplan step** rather than widening 91.13; the fix pattern already
   exists in-repo at `globals.css:178-183`.
5. **Dead code to queue, not to fix here.** `AlphaScoreCard`
   (`GlassBoxCards.tsx:25-53`) has 0 render sites. Removing the glow from the two
   live sites would leave the class referenced only by dead code -- worth recording
   in the contract so a later cleanup does not read the glow as load-bearing.
6. **UI verification will be required.** Per CLAUDE.md's Playwright bullet and
   `.claude/rules/frontend.md`, any claim about the rendered card row needs a live
   capture behind the NextAuth wall (`scripts/qa/mint_playwright_storage_state.py`),
   not a code reading. **This brief makes no rendered-appearance claim** -- every
   internal statement above is a file:line fact.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **7** (2 further fetches attempted and failed; both listed, neither counted)
- [x] 10+ unique URLs total (incl. snippet-only) -- **32**
- [x] Recency scan (last 2 years) performed + reported -- 3 findings, section above
- [x] Full pages read (not abstracts) for the read-in-full set -- no PDFs/arXiv in this corpus, so the html->ar5iv->pdfplumber chain did not apply
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module -- all 3 caller-scoped files read in full, plus all 3 `glow` call sites and both `GlassBoxCards` consumers (8 files)
- [x] Contradictions / consensus noted -- see "Consensus vs debate"; 1 source tagged [ADVERSARIAL], 1 [PARTIAL-NEGATIVE]
- [x] All claims cited per-claim (URL or file:line inline, not footer-only)
- [ ] **Gap disclosed:** two official-design-system sources (IBM Carbon, Material 3) could not be extracted -- both are JS-rendered. The design-system tier is therefore represented by ActiveCampaign only. This weakens tier-2 coverage but the floor is met without them and their absence does not change any finding.
