---
name: glow-emphasis-91-13
description: Step 91.13 -- .alpha-score-glow is named after DEAD code (AlphaScoreCard, 0 render sites); every surviving glow site is a 1-of-N uniform row; globals.css has exactly ONE enforced prefers-reduced-motion rule out of 5 infinite animations
metadata:
  type: project
---

Step 91.13 (dashboard KPI stat-card visual consistency) measured findings.

**The glow's namesake is dead code.** `.alpha-score-glow`
(`frontend/src/app/globals.css:66-68`) is named for `AlphaScoreCard`
(`GlassBoxCards.tsx:25`), which has **0 render sites** repo-wide -- its three
module siblings have 2 each, and both consumers
(`app/reports/[ticker]/page.tsx:21-25`, `PdfDownload.tsx:12`) import only those
three. So the ONE site where the glow was hierarchy-bearing (`text-7xl`) is
unreachable, and both surviving sites (`CostDashboard.tsx:85`,
`app/performance/page.tsx:178`) are 1-of-N cards in a **uniform** row with
identical type scale and identical grid span as their siblings. The effect
outlived its rationale. Corollary that matters for scoping: removing `glow` from
those two call sites makes the CSS class **unreferenced**, so the WCAG exposure
goes to zero with no CSS edit -- do not bundle a CSS fix into a JSX-only step.

**Motion-guard drift is measurable and asymmetric.** `globals.css` has exactly
**one** enforced `prefers-reduced-motion` rule (`:178-183`, NumberFlow tint).
Five infinite animations exist; four are unguarded: `shimmer` (`:48-58`),
`pulse-glow` (`:61-68`), `spin-slow` (`:71-77`), `gemini-bounce` (`:101-104`).
The file's own comment at `:147-151` reasons "SC 2.2.2 satisfied (900ms << 5s
ceiling)" for the guarded one -- **that reasoning inverts for a `3s infinite`**,
so the comment is a ready-made argument against its neighbours. Tailwind
`motion-safe:`/`motion-reduce:` variants appear **nowhere** in `frontend/src`.

**External corpus shape (useful if this topic recurs).** Design-system tier is
hard to fetch: `carbondesignsystem.com` returns "Content truncated due to
length" and `m3.material.io` returns only a page title -- both JS-rendered, both
unusable via WebFetch. `activecampaign.design` DOES extract and is a good
substitute (its KPI-card spec offers no per-card decorative variant at all, and
carries a scarcity rule: "only one red metric should appear in any KPI bar").
NN/g's pre-attentive article is **silent on motion** -- it ranks
length/position/color only, so it cannot be cited to license animated emphasis.
Smashing 2025-09 is the only source that addresses glow-on-a-card directly and
frames it as a **200-400ms transient tied to a data change**, not a persistent
state.

**Why:** the operator flagged the Total Cost box as visually inconsistent with
its 3 siblings; the question was whether one-card emphasis is ever right.

**How to apply:** on any pyfinagent frontend emphasis question, check whether the
differentiator is hierarchy-bearing (span/type scale) or decoration-only -- and
grep the render sites before assuming a styled component is live. See
[[feedback_url_count_must_be_re_derived]] for the URL-count discipline used in
this brief.
