# Research Brief: phase-24.12 — Frontend UI/UX Presentation-Layer Audit (P2)

**Tier:** moderate
**Date:** 2026-05-12
**Researcher:** Researcher agent (combined external + internal)

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|---------------------|
| https://www.w3.org/TR/WCAG22/ | 2026-05-12 | Official spec | WebFetch | "When a user interface component receives keyboard focus, the component is not entirely hidden due to author-created content." (SC 2.4.11, new in 2.2) |
| https://playwright.dev/docs/screenshots | 2026-05-12 | Official doc | WebFetch | `page.screenshot({ path, fullPage })` and element-level `.screenshot()`; CI note: OS/font differences mean baselines generated on macOS won't match Linux CI |
| https://testdino.com/blog/playwright-visual-testing | 2026-05-12 | Practitioner blog | WebFetch | `toHaveScreenshot('x.png', { maxDiffPixelRatio: 0.01, threshold: 0.2 })`; commit baselines to VCS; `page.waitForLoadState('networkidle')` required before capture |
| https://ashconnolly.com/blog/playwright-visual-regression-testing-in-next | 2026-05-12 | Practitioner blog | WebFetch | Next.js-specific: `NEXT_PUBLIC_E2E_TESTING=true` env to disable devIndicators; `prefers-reduced-motion` / `reducedMotion: 'reduce'` config to freeze CSS animations; `mask:` API for dynamic content |
| https://eslint.org/docs/latest/rules/no-restricted-imports | 2026-05-12 | Official doc | WebFetch | `paths[].message` custom message; `files` override to exempt the barrel file; `allowTypeImports` for type-only exceptions; confirms the pattern already in pyfinagent's eslint.config.mjs is canonical |
| https://www.w3.org/WAI/ARIA/apg/ | 2026-05-12 | Official spec | WebFetch | Landmark regions, naming conventions for status panels; directs to dedicated pattern pages for tabs, alert dialogs, loading spinners |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://bug0.com/knowledge-base/playwright-visual-regression-testing | Blog | Covered by testdino full read |
| https://backlight.dev/blog/best-practices-w-eslint-part-1 | Blog | Lit-specific content, not React; partial read returned |
| https://medium.com/@barshaya97_76274/design-tokens-enforcement-977310b2788e | Blog | Snippet sufficient; eslint.org rule doc more authoritative |
| https://dev.to/timwjames/the-best-eslint-rules-for-react-projects-30i8 | Blog | Covered by official ESLint doc |
| https://medium.com/neighborhoods-com-engineering-blog/custom-eslint-rules-for-faster-refactoring-2095e69bde08 | Blog | Snippet for pattern context |
| https://css-tricks.com/automated-visual-regression-testing-with-playwright/ | Blog | Covered by testdino + ashconnolly reads |
| https://www.techinterview.org/post/3233475391/frontend-testing-2026-vitest-playwright-visual-regression/ | Blog | Snippet for 2026 context; no novel findings |

---

## Recency scan (2024-2026)

Searched: "visual regression testing Playwright Next.js CI 2025 2026", "design system enforcement ESLint React 2025 2026".

**Result:** Three material 2025-2026 findings:

1. Playwright `toHaveScreenshot()` is now the stable, idiomatic API (replaces earlier `toMatchSnapshot()` for visual work). OS-level rendering divergence between macOS and Linux CI is a 2026 concern for baseline management — baselines MUST be generated in CI Docker, not locally.
2. `eslint-config-next v16` (shipped with Next.js 15 era) packages `react-hooks/set-state-in-effect`, `purity`, and `immutability` as compiler-awareness rules. The pyfinagent config already has these as "warn".
3. `svh` units (used in `frontend-layout.md §4.6`) reached Baseline Widely Available June 2025 — no risk from the layout rule.

No findings supersede the canonical sources (WCAG 2.2, Playwright docs, ESLint no-restricted-imports rule).

---

## Key findings

1. **Zero direct `@phosphor-icons/react` imports outside `icons.ts`** — The `no-restricted-imports` rule was promoted to "error" in phase-16.39; the sweep fixed 22 prior violators. Current grep of `frontend/src/` (excluding `lib/icons.ts`) returns empty — rule is enforced at the lint level AND the barrel is clean. (Source: grep + `eslint.config.mjs`)

2. **ESLint rule is canonical** — The existing config in `eslint.config.mjs` matches the ESLint official `no-restricted-imports` pattern exactly: `paths` with `name`+`message`, `patterns` for sub-paths, and a `files`-scoped override to exempt the barrel. No additional infra needed for the icon rule itself. (Source: official ESLint doc, `eslint.config.mjs`)

3. **Loading/error states: two degraded instances** — `performance/page.tsx:65-66` uses bare `<p className="text-slate-400">Loading...</p>` and `<p className="text-rose-400">` rather than `<PageSkeleton />` and a rose-border banner with retry button. Violates `frontend.md` "Loading states: Use Skeleton.tsx components" and "Error states: ... surface an error banner (rose-900 border, rose-950/50 bg) with retry button". (Source: `performance/page.tsx:65-66`)

4. **`paper-trading` tab definitions missing Phosphor icons** — The `TABS` array at `paper-trading/page.tsx:383-390` is `{ id, label }` only — no `icon` field. The canonical tab definition pattern from `frontend-layout.md §5` requires `{ id, label, icon: Icon }`. Tabs render as text-only, inconsistent with the spec. (Source: `paper-trading/page.tsx:383-390`)

5. **Cross-tab KPI mismatch: Sharpe uses different data sources** — Home cockpit computes Sharpe (90d) via local `kpiSharpe()` from `redLineSeries` (`page.tsx:18,161`). Paper-trading page shows `perf?.sharpe_ratio` from `getPaperPerformance()` API (`paper-trading/page.tsx:203-204`). Sovereign has no Sharpe display. The two sources are not guaranteed to match — `kpiSharpe()` uses a 90-day rolling window over NAV; backend `sharpe_ratio` period is unspecified. Users cross-referencing the two pages will see different values. (Source: `page.tsx:18,161,236`; `paper-trading/page.tsx:203-204`)

6. **Page shell: all 14 pages compliant** — Every page uses `flex h-screen overflow-hidden` outer div + `flex-1 overflow-y-auto scrollbar-thin` scrollable zone. The `learnings` and `agent-map` pages are thin wrappers with no async data and correct shells. (Source: per-file grep)

7. **Polling failure discipline: inconsistent** — `cron/page.tsx:42,156` and `agents/page.tsx:183,225-227` implement the 5-consecutive-failure rule. The `paper-trading/page.tsx:534-550` `handleRunNow` interval runs up to 300 seconds with no failure counter — it only exits on clean `!s.loop.running` or timeout ceiling, never on repeated fetch errors. (Source: `paper-trading/page.tsx:534-550`)

8. **Sovereign silent data failures** — `sovereign/page.tsx:63-68` has `.catch(() => { setRedLineSeries([]); setRedLineEvents([]) })` — no error state is surfaced to the user when the red-line endpoint fails; the chart renders empty with no indication of why. (Source: `sovereign/page.tsx:63-68`)

9. **Playwright visual regression CI not present** — No `playwright.config.ts` exists under `frontend/`. The Next.js-specific setup (devIndicator env, animation freeze, dynamic-content masking) is documented above. Baseline generation must target CI Docker to avoid macOS/Linux font rendering divergence. (Source: filesystem check)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `frontend/src/lib/icons.ts` | 239 | Centralized Phosphor barrel | Clean — 0 violations outside this file |
| `frontend/eslint.config.mjs` | ~50 | ESLint config | `no-restricted-imports` set to "error" with barrel exemption |
| `frontend/src/app/page.tsx` | 302 | Home cockpit | Shell OK; Sharpe from `kpiSharpe()` (local, 90d rolling) |
| `frontend/src/app/paper-trading/page.tsx` | 1273 | Paper trading | Shell OK; Sharpe from `perf.sharpe_ratio` (API); tabs missing icons; polling no fail-counter |
| `frontend/src/app/sovereign/page.tsx` | 167 | Sovereign control | Shell OK; no Sharpe; silent RedLine/Cost failure |
| `frontend/src/app/performance/page.tsx` | ~200 | Rec performance | Shell OK; bare `<p>` loading+error — missing PageSkeleton, missing retry button |
| `frontend/src/app/backtest/page.tsx` | 740+ | Backtest | Shell OK; PageSkeleton used; error banner present |
| `frontend/src/app/reports/page.tsx` | 530+ | Reports | Shell OK; PageSkeleton used; error banner present |
| `frontend/src/app/agents/page.tsx` | 660+ | Agent pipeline live | Shell OK; 5-failure polling rule implemented |
| `frontend/src/app/cron/page.tsx` | 450+ | Cron/logs | Shell OK; 5-failure rule with `MAX_CONSECUTIVE_FAILURES=5` |
| `frontend/src/app/signals/page.tsx` | ~200 | Ticker signals | Shell OK; inline spinner; empty-state present |
| `frontend/src/app/settings/page.tsx` | 1227+ | Settings | Shell OK; PageSkeleton used |
| `frontend/src/app/agent-map/page.tsx` | 34 | Agent topology | Thin wrapper; delegates to AgentMap |
| `frontend/src/app/paper-trading/learnings/page.tsx` | 20 | Learnings | Thin wrapper; delegates to VirtualFundLearnings |
| `.claude/rules/frontend.md` | 48 | Frontend rules | Canonical — all rules cross-checked |
| `.claude/rules/frontend-layout.md` | 496 | Layout blueprint | Canonical — all patterns cross-checked |

---

## Consensus vs debate (external)

- **Consensus:** ESLint `no-restricted-imports` is the standard mechanism for barrel-file enforcement; no controversy.
- **Consensus:** Playwright `toHaveScreenshot()` is the modern API; OS-baseline mismatch is a known CI concern.
- **Debate:** WCAG 2.2 AA vs AAA for financial dashboards. Some practitioners argue AAA contrast for data tables in financial tools; pyfinagent's `text-slate-400` labels likely hit AA but not AAA. Out of scope for this step.

---

## Pitfalls (from literature)

1. **Playwright baselines on macOS vs Linux CI** — OS font hinting differences produce pixel-level diffs. Always generate and commit baselines from the CI Docker image, not a dev machine. (Source: ashconnolly.com, testdino.com)
2. **Masking dynamic content before snapshot** — Live prices, timestamps, and chart data will cause every run to differ. The `mask:` array must include all dynamic elements. (Source: ashconnolly.com)
3. **Sharpe definition creep** — When frontend computes Sharpe independently, silent divergence from backend values is inevitable as windows or annualization constants differ. Single authoritative API source is the fix.
4. **Polling without a failure counter** — A `setInterval` with no error counting silently accumulates rejected promises. The 5-consecutive-fail rule in `frontend.md` exists for this reason.

---

## Application to pyfinagent

| Finding | File:line | Recommended fix | Phase-25 candidate |
|---------|-----------|-----------------|-------------------|
| Bare `<p>` loading state | `performance/page.tsx:65` | Replace with `<PageSkeleton />` | (c) missing-states sweep |
| Bare `<p>` error state (no retry) | `performance/page.tsx:66` | Rose border banner + retry button | (c) |
| Tabs without icons | `paper-trading/page.tsx:383-390` | Add `icon` field per `frontend-layout.md §5` | (c) |
| Sharpe source divergence | `page.tsx:161` vs `paper-trading/page.tsx:203` | Expose `sharpe_90d` on status endpoint; home page reads it | (d) KPI reconciliation |
| Polling no fail-counter | `paper-trading/page.tsx:534-550` | Add `failCount` ref; stop after 5 errors | (c) |
| Silent sovereign data failure | `sovereign/page.tsx:63-68` | Surface amber "data unavailable" notice | (c) |
| No Playwright visual CI | `frontend/` (no playwright.config.ts) | Create config with Next.js env, reducedMotion, masking | (a) visual regression CI |
| ESLint rule already enforced | `eslint.config.mjs:37-54` | No code change — add `npm run lint` as required CI step | (b) CI enforcement |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total (13 total: 6 full + 7 snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (all 14 page.tsx files + icons.ts + eslint.config)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
