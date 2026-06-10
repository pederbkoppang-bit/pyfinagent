## Research: phase-10.12 — Visual alignment of HarnessSprintTile.tsx with navy palette

Tier assumption: simple (pure CSS/styling refactor; behavior preservation)

---

### Search queries run (three-variant discipline)

1. **Current-year frontier**: "dashboard card empty state best practices 2026 compact vs hero UX"
2. **Last-2-year window**: "NNG Nielsen Norman Group empty state compact dashboard card size 2024 2025", "PatternFly empty state design guidelines compact size variations 2025"
3. **Year-less canonical**: "Shneiderman reduce short-term memory load dashboard design information density", "Tailwind UI dark dashboard card design system tokens", "Grafana 12 dashboard dark theme card empty state"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.nngroup.com/articles/empty-state-interface-design/ | 2026-04-20 | Official doc (NN/G) | WebFetch full | "A simple message communicates the state of the system and increases user confidence." Three guidelines: status, learning cues, direct pathways. No hero sizing. |
| https://www.patternfly.org/components/empty-state/design-guidelines/ | 2026-04-20 | Official design system doc | WebFetch full | Four size tiers: XS = "inside cards or when space-constrained"; S = tables/wizards/modals; L = full-page; XL = first-use/congratulations only. Card-context empty states center within container with lg spacer padding only. |
| https://cloudscape.design/patterns/general/empty-states/ | 2026-04-20 | Official design system doc (AWS) | WebFetch full | Three required elements: heading, optional description, action. Emphasizes action to prevent confusion. Does not prescribe large vertical padding in container contexts. |
| https://www.cs.umd.edu/users/ben/goldenrules.html | 2026-04-20 | Academic primary source | WebFetch full | Rule 8: "Reduce short-term memory load — avoid interfaces in which users must remember information from one display." Supports compact info density; dense bar > stacked expansive cards. |
| https://hexshift.medium.com/creating-a-dark-mode-aware-dashboard-with-tailwind-css-and-react-c63bdb5f3f4b | 2026-04-20 | Authoritative blog | WebFetch full | Tailwind dark-mode card implementation: `bg-white dark:bg-gray-700`, `shadow-md rounded-lg p-4`. Confirms dark-only design avoids dual-mode classes; recommendation: use semantic custom tokens (navy) not zinc gray defaults. |
| https://www.eleken.co/blog-posts/empty-state-ux | 2026-04-20 | Industry blog | WebFetch full | "Keep a copy of one strong sentence and, if needed, a line of supporting text." Endorses minimalist approach; "generous spacing" for full-page, not card context. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.pencilandpaper.io/articles/empty-states | Blog | Fetched but contained only strategic content, no dimension specs |
| https://wise.design/patterns/empty-state-pattern | Design system | Fetched but lacked sizing/padding specs |
| https://grafana.com/blog/2025/05/07/dynamic-dashboards-grafana-12/ | Vendor blog | Fetched; covers dynamic layout feature not card-level token specs |
| https://www.saasframe.io/categories/empty-state | Gallery | Visual gallery only, no accessible text |
| https://carbondesignsystem.com/patterns/empty-states-pattern/ | Design system | Budget limit; PatternFly covers same ground in more detail |
| https://mobbin.com/glossary/empty-state | Design reference | Gallery only |
| https://tech-insider.org/tailwind-css-tutorial-dashboard-v4-2026/ | Tutorial | Covers Tailwind v4 generally, not custom navy token patterns |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on dashboard card empty states, dark-mode design tokens, and compact empty-state sizing. Result: PatternFly's 2025 XS/S/L/XL size taxonomy was the most directly applicable new finding — it codifies the "card context = XS size" rule that existed as informal practice. No finding from 2024-2026 contradicts or supersedes the canonical Shneiderman (density) or NN/G (informative default) sources. Tailwind v4 CSS-variable token approach (2025) reinforces the custom `navy-*` token strategy this project already uses.

---

### Key findings

1. **Empty states in card/tile context use XS size, not full-page size** — "Use an extra small empty state inside cards or when there are space constraints." (PatternFly 2025, https://www.patternfly.org/components/empty-state/design-guidelines/) This directly contradicts the current `py-12` + `size={40}` implementation in the null branch.

2. **Container empty states center within their container with a single lg spacer on all sides** — not an expanded hero treatment. (PatternFly 2025, ibid.) Equivalent: `py-6 px-4` not `py-12 px-6`.

3. **Icon in card-context empty states should be small, not dominant** — PatternFly XS variant places a small icon above left-aligned (or centered) text; `size={32}` or smaller is appropriate. The current `size={40}` is PatternFly "small" range. (PatternFly 2025, ibid.)

4. **Information density: compact surfaces beat expansive dead whitespace** — Rule 8 ("reduce short-term memory load") and the NN/G "informative default" pattern both argue for recognizable tile state, not a large empty region that draws attention to its own emptiness. (Shneiderman, https://www.cs.umd.edu/users/ben/goldenrules.html)

5. **Dark-only app should use semantic custom tokens, not dual-mode zinc/white classes** — The hex shift article confirms that `bg-white dark:bg-zinc-900` + `border-zinc-200 dark:border-zinc-800` is a pattern for dual-mode apps. This project is dark-only; using the bare zinc dual-mode classes adds dead specificity and clashes visually. Use `bg-navy-800/60 border-navy-700` directly. (Dark-mode Tailwind guide, https://hexshift.medium.com)

6. **NN/G rule: always communicate system status** — "A simple message communicates the state of the system." The current empty-state text is good; only the visual scale needs reducing. (NN/G, https://www.nngroup.com/articles/empty-state-interface-design/)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/src/components/HarnessSprintTile.tsx` | 189 | Sprint-state tile (phase-10.9) | MISALIGNED — uses `rounded-2xl border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 p-6`; empty state uses `py-12` + `size={40}` |
| `frontend/src/components/HarnessDashboard.tsx` | 513 | Parent page; places HarnessSprintTile at top | CORRECT — uses navy tokens throughout: `border-navy-700 bg-navy-800/80 bg-navy-900/50 hover:bg-navy-700/40 divide-navy-700/50 bg-navy-800/60` |
| `frontend/src/components/BentoCard.tsx` | 26 | Shared card wrapper used by all other HarnessDashboard cards | CORRECT — `rounded-2xl border border-navy-700 bg-navy-800/70 p-6 backdrop-blur-lg` |
| `frontend/src/components/OpsStatusBar.tsx` | 60+ | Canonical dense-bar pattern (frontend-layout §4.5) | CORRECT — uses `border-navy-700 bg-navy-800/60` tokens |
| `frontend/tailwind.config.js` | 31 | Custom color token definitions | navy-900=#020617, navy-800=#0f172a, navy-700=#1e293b, navy-600=#1a2744, navy-500=#243352 |
| `frontend/src/components/HarnessSprintTile.test.tsx` | 98 | Vitest tests (phase-10.9) — 5 tests | LOAD-BEARING — assertions on `data-section="weekly-state"`, `data-cell="thu-candidates"`, `data-cell="fri-promoted-count"`, `data-cell="sortino-delta"`, `aria-label="Sprint state"`, `data-week-iso`, no `<button>`, no `<input>`, "No sprint activity yet" text |

---

### Consensus vs debate (external)

Consensus: all five design systems and academic sources agree that empty states in tile/card contexts should be compact — not hero-sized. The informative-default principle (NN/G) does NOT require large vertical space; it requires clear text. There is no debate on the dark-only token approach: zinc dual-mode classes are a pattern for apps supporting both modes.

---

### Pitfalls (from literature)

- Removing `py-12` completely risks an unreadable 1-line empty state with no visual breathing room. PatternFly recommends `lg spacer` inside the container, equivalent to `py-6` or `py-8`. Do not go below `py-6`.
- The icon should stay (`aria-hidden="true"` preserved) — NN/G says learning cues help; removing the icon entirely loses the informative-default signal.
- Reducing icon from `size={40}` to `size={32}` is within the XS/S boundary from PatternFly. Do not go below `size={24}` or it becomes non-perceivable.

---

### Application to pyfinagent (mapping external findings to file:line anchors)

| Change | File:line | Rationale | Risk to tests |
|--------|-----------|-----------|---------------|
| Outer `<section>` class: replace `rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm dark:border-zinc-800 dark:bg-zinc-900` with `rounded-xl border border-navy-700 bg-navy-800/60 p-5` | HarnessSprintTile.tsx:35 (null branch) and :68 (data branch) | Aligns with BentoCard.tsx:17 and HarnessDashboard.tsx:98 `border-navy-700` token | NONE — tests assert on `aria-label="Sprint state"` and `data-week-iso` attribute, not CSS classes |
| Empty-state icon: `size={40}` → `size={32}` | HarnessSprintTile.tsx:38 | PatternFly XS = card context, smaller icon | NONE — tests only assert `.textContent` and absence of buttons |
| Empty-state vertical padding: `py-12` → `py-8` | HarnessSprintTile.tsx:37 | Reduces ~400px dead space; PatternFly lg spacer for container context | NONE — no test asserts on padding classes |
| Inner sub-cards: `rounded-xl border border-zinc-200 bg-zinc-50 p-4 dark:border-zinc-800 dark:bg-zinc-950/40` → `rounded-lg border border-navy-700/50 bg-navy-900/40 p-4` | HarnessSprintTile.tsx:82, :103, :129 | Matches HarnessDashboard.tsx:322 `rounded-lg bg-navy-900/50 p-3` and :342 `border border-navy-700` | NONE — tests assert on `data-cell` text content, not class strings |
| Title `<h3>` label: already `text-slate-400` — keep | HarnessSprintTile.tsx:71 | Correct per frontend-layout §4 card anatomy | NONE |
| Value cells `text-2xl font-bold text-slate-100` — keep | HarnessSprintTile.tsx:93, :114 | Correct per frontend-layout §4 card anatomy | NONE |
| `text-emerald-400`, `text-amber-400`, `text-slate-400` for monthly color — keep | HarnessSprintTile.tsx:58-62 | Correct per frontend.md color coding | NONE |

**Zero test-breaking changes.** All 5 phase-10.9 assertions are on `data-*` attributes, `.textContent`, element roles (no buttons), and text content — none on class strings.

**Safe sub-card pattern from HarnessDashboard.tsx (lines 322-336):**
```
rounded-lg bg-navy-900/50 p-3 text-center
```
and table containers (line 342):
```
overflow-hidden rounded-xl border border-navy-700
```

**Proposed exact JSX skeleton for the outer `<section>` (both branches):**
```tsx
className="rounded-xl border border-navy-700 bg-navy-800/60 p-5"
```

**Proposed empty-state inner div:**
```tsx
<div className="flex flex-col items-center justify-center py-8 text-center">
  <IconTimer size={32} weight="duotone" className="text-slate-600" aria-hidden="true" />
  <p className="mt-3 text-sm text-slate-400">No sprint activity yet</p>
  <p className="mt-1 text-xs text-slate-600">
    Thursday batch and Friday promotion will appear here once the first sprint runs.
  </p>
</div>
```

**Proposed inner sub-card class (replace all three instances at lines 82, 103, 129):**
```tsx
className="rounded-lg border border-navy-700/50 bg-navy-900/40 p-4"
```

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched in full)
- [x] 10+ unique URLs total (incl. snippet-only) — 13 unique URLs
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (HarnessSprintTile, HarnessDashboard, BentoCard, OpsStatusBar, tailwind.config.js, test file)
- [x] Contradictions/consensus noted (none found — all sources agree on compact XS for card context)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/phase-10.12-research-brief.md",
  "gate_passed": true
}
```
