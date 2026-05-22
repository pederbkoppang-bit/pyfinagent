# phase-44.0 -- Frontend UX/AI Design Master Plan (super-planning, plan-only)

**Step id:** `phase-44.0` (new; appended to masterplan in GENERATE step alongside 44.1-44.10)
**Date:** 2026-05-22
**Mode:** OVERNIGHT planning -- one harness pass; deep-tier research + per-page expansion; NOT execution.
**Author:** Main (Claude Opus 4.7, this Claude Code session)
**Cycle in `handoff/harness_log.md`:** Cycle 11 (after Cycle 10 phase-33.2 master roadmap)

---

## Research-gate summary

The researcher subagent (id `a63a24f40308f8093`, effort `deep`/max-tier) produced
`handoff/current/research_brief.md` (775 lines) covering:

- **Section A:** internal per-page audit -- 15 pages + 58 components + Sidebar + 8 lib files + 2 frontend rules. Per-page report: lines, components imported, API calls, state management, loading/empty/error, mobile/a11y status, charts, most-painful UX issue.
- **Section B:** external 2026 patterns synthesis -- 10 sources read in full across 10 dimensions (production design systems, AI transparency UX, financial dashboards, real-time SSE, data-dense tables, command palette, WCAG 2.2 a11y, AI cockpit transparency, charting, dark-mode color systems). Pattern -> step mapping in B.11.
- **Section C:** per-page gap table -- one row per page with top-2 gaps + 2026 pattern + effort tier.
- **Section D:** reusability matrix -- 14 NEW components reduce cross-page duplication (CommandPalette, TraceTree, DataTable, Sparkline, Drawer, EmptyState, ErrorBanner, LiveBadge, TimeRangeSelector, Breadcrumb, KeyboardShortcuts, useEventSource, useURLState, useDebounced).
- **Section E:** recency scan -- 2024-2026 supersessions (Tremor=Vercel-owned, cmdk=de-facto, WCAG 2.2=EU mandate 2026, TanStack v8=replaces v7, SSE=dominant for LLM streaming).
- **Section G:** `gate_passed: true` -- 10 external sources / 8-source floor exceeded.

**Researcher headline:** Four 2026-frontier gaps dominate:
1. ZERO Cmd-K command palette across 15 routes -- biggest miss vs Linear/Vercel/Stripe.
2. WCAG 2.2 a11y gap -- single-digit aria-* count per page, OpsStatusBar inline buttons failing 24x24 target-size rule (EU AAA mandate 2026).
3. /agents Live Stream is a flat log not a LangSmith-style trace tree (backend already emits the needed span types).
4. /settings duplicates /paper-trading Manage tab -- DRY violation.

**Top-3 pages needing most work:**
- `/paper-trading` (1284 LoC, cockpit) -- raw tables, Manage tab dupes /settings, tablist a11y
- `/backtest` (1594 LoC, 7 tabs across 4 domains) -- Budget+Harness don't belong, RunSelector heavy
- `/agents` (728 LoC, only SSE page) -- flat log vs trace-tree, hand-coded SVG topology

---

## Hypothesis

> If we produce `handoff/current/frontend_ux_master_design.md` containing
> per-page expanded plans where each step's success criteria are derived
> from the researcher's Section A audit (current state, file:line) AND
> Section B external 2026 patterns (specific citation per pattern), AND
> we append phase-44 entries to `.claude/masterplan.json` with 11 steps
> (44.0 audit + 44.1 foundation + 44.2 cockpit + 44.3 trail-drawer + 44.4
> reports + 44.5 trading-non-paper + 44.6 analyze + 44.7 system + 44.8
> settings/login + 44.9 mobile-a11y + 44.10 SSE) all status `in-progress`
> per `feedback_masterplan_status_flip_order`, THEN the next session can
> walk each step with explicit success criteria backed by research and
> known 2026 patterns -- the high-standard bar is preserved by design.

If true: phase-44.0 closes with one master design doc + 11 masterplan
phases (1 parent + ~31 step children, depending on how the 11 are split)
that any future session executes directly. Q/A verifies per-page
coverage (15 of 15), external-source backing (>= 8 cited in steps), and
masterplan JSON validity.

If false: either per-page coverage misses (Q/A returns CONDITIONAL/FAIL
on a missing page), the master doc lacks file:line citations, or JSON
inserts fail to parse. Fix + fresh Q/A. Max 2 retries -> `blocked`.

---

## Immutable success criteria (VERBATIM from /goal directive)

The user directive was: "deep research codebase and top production ready
UX/AI design 2026 with our researcher for each of the frontend design
steps and then expand each steps with founds so we get a high standard
on all pages. do not make any adjustment to frontend yet as this is only
for planning phase. also research each page to understand current setup"

Decomposed into mechanically-verifiable acceptance criteria:

1. **Deep research executed:** researcher subagent fired at `deep` tier with `gate_passed: true` (>= 8 external sources read in full). VERIFIED -- 10 fetched-in-full.

2. **Per-page synthesis:** each of the 15 pages has its own per-page expanded plan in `frontend_ux_master_design.md` covering: (a) current setup with file:line, (b) gaps vs 2026 standard, (c) specific 2026 patterns to apply (citing research_brief Section B), (d) per-page success criteria (mechanically verifiable), (e) effort + risk class.

3. **High-standard across all pages:** every page has at least 3 success criteria from the research_brief's gap table (Section C) + at least 1 cross-cutting reusable-component application from Section D.

4. **Plan-only -- NO frontend code changes:** `git diff --stat frontend/src/` returns empty across the whole phase-44.0 cycle.

5. **Sub-bars + Settings included:** Sidebar's 4 sections (Analyze / Reports / Trading / System) + the standalone Settings + Login route ALL appear in the master design doc with explicit verdicts and per-page success criteria. The sidebar itself gets a treatment (because keyboard nav + collapse + active-route highlight need work per A.1).

6. **Masterplan inserts valid:** `python -c "import json; json.load(open('.claude/masterplan.json'))"` exits 0. Phase-44 + 11 step entries land with `status: in-progress` (parent) / `status: pending` (steps); NO `done` flips.

7. **Coverage check before status flip:** every one of 15 pages from `research_brief.md` Section A.3-A.17 + Sidebar (A.1) + Lib (A.2) + Components (A.18) MUST appear in `frontend_ux_master_design.md`. Anything not addressed = silent drop = QA CONDITIONAL.

8. **Honest current setup citation:** at least 1 file:line citation from the brief per page in the master design doc (`frontend/src/app/<page>/page.tsx:NNN`).

---

## Plan steps (within this phase-44.0 work)

| # | Step | Tool / Artifact | Status |
|---|---|---|---|
| 1 | Researcher gate -- deep-tier audit + 2026 patterns | `handoff/current/research_brief.md` (775 lines, gate_passed=true) | DONE |
| 2 | Write this contract | `handoff/current/contract.md` | IN FLIGHT |
| 3 | GENERATE: per-page expanded master design doc | `handoff/current/frontend_ux_master_design.md` | NEXT |
| 4 | GENERATE: append phase-44 + 11 step entries to masterplan | edits to `.claude/masterplan.json` | NEXT |
| 5 | Coverage-check (15 pages + sidebar + lib + components against master doc) | grep + cross-list | NEXT |
| 6 | Spawn Q/A ONCE (max 2 retries) | qa subagent | NEXT |
| 7 | Append cycle 11 to `handoff/harness_log.md` | edit | NEXT |
| 8 | Flip phase-44 + step 44.0 status to done (auto-commit + push, prefix `phase-44.0:`) | masterplan.json | NEXT |

---

## Hard guardrails (verbatim from user directive + project rules)

- **NO frontend code changes anywhere this cycle.** Plan-only. `git diff --stat frontend/src/` must be empty.
- **NO new dependencies in `frontend/package.json` proposed** without explicit pattern justification cited from research_brief Section B.
- **NO emojis** -- Phosphor Icons only per `feedback_no_emojis`. The master design doc itself contains zero emojis.
- **`feedback_masterplan_status_flip_order`:** all new phase-44 steps land with `status: in-progress` / `pending`; NO `done` flips in the GENERATE phase. Only the final 44.0 flip-to-done after Q/A PASS + harness_log append.
- **`feedback_log_last`:** harness_log.md append BEFORE status flip.
- **`feedback_qa_harness_compliance_first`:** Q/A prompt starts with the 5-item protocol audit.
- **No `AskUserQuestion`** within this cycle -- the directive is the directive.

---

## References

- `handoff/current/research_brief.md` -- the 775-line per-page audit + 2026-pattern synthesis
- `handoff/current/master_roadmap_to_production.md` (cycle 10 deliverable) -- parallel-track backend roadmap; this phase-44 plan runs alongside it
- `.claude/rules/frontend.md` + `.claude/rules/frontend-layout.md` -- frontend conventions (mandatory re-read by executor)
- `.claude/masterplan.json` -- to be edited in GENERATE step
- All 15 page.tsx files + 58 components -- the per-page citations live there
- External anchors (>= 10, cited per page in the master design doc):
  - Linear dashboard best practices
  - Stripe Apps Design Patterns
  - Vercel cmdk + Tremor v3
  - WCAG 2.2 (EU 2026 mandate)
  - TanStack Table v8
  - SSE 2026 streaming UX
  - LangChain Agent Observability (trace-tree pattern)
  - Bloomberg Terminal complexity-concealment
  - Tremor / Recharts wrappers
  - Apple HIG dark-mode + Linear color palette
