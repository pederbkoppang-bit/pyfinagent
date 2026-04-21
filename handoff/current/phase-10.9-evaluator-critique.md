# Q/A Critique — phase-10.9 (Harness-tab sprint-state tile)

**Step:** 10.9 (FINAL phase-10 step) **Date:** 2026-04-20
**Q/A id:** qa_109_v1 **Verdict:** PASS

## 5-item harness-compliance audit

| # | Check | Result |
|---|-------|--------|
| 1 | Researcher: ≥5 full sources + recency + gate_passed | PASS — 6 read-in-full, 13 URLs, 2024-2026 recency scan present, three-variant search query discipline honoured, `gate_passed=true` (brief §"Research Gate Checklist") |
| 2 | Contract mtime ≤ results mtime | PASS — contract 1776721574, results 1776721682 (108s gap, correct order) |
| 3 | Immutable criteria quoted verbatim | PASS — contract §"Immutable success criteria" lines 22-26 quote masterplan verbatim; test `it()` names match case-sensitive |
| 4 | Log-is-last (no appended 10.9 cycle yet) | PASS — last harness_log entry closes phase-10.8 with "Next: phase-10.9"; no 10.9 cycle appended |
| 5 | Cycle v1 (no verdict-shopping) | PASS — first Q/A spawn for this step; no prior evaluator-critique |

## Deterministic checks

| # | Check | Command / anchor | Result |
|---|-------|------------------|--------|
| A | Immutable CLI | `cd frontend && npm run test -- --filter=HarnessSprintTile` | PASS — 5 tests passed, 897ms, Vitest 4.1.4 |
| B | TypeScript | `npx tsc --noEmit 2>&1 \| grep -E "HarnessSprint\|error TS"` | PASS — empty output (no new type errors) |
| C | 3 handoff files present | contract / results / research-brief | PASS — all three at `handoff/current/phase-10.9-*.md` |
| D.1 | No `<button>`/`<input>`/`<select>`/`<textarea>`/`<form>` in component | grep on `HarnessSprintTile.tsx` | PASS — zero matches |
| D.2 | No `onClick`/`onChange`/`onSubmit` handlers | grep on component | PASS — zero matches |
| D.3 | No `useState`/`useEffect`/hooks | grep on component | PASS — zero matches (pure presentational) |
| D.4 | Icons imported from `@/lib/icons` only | lines 13-18 | PASS — `IconCheckCircle, IconWarning, IconTimer, IconChart` from `@/lib/icons`; no `@phosphor-icons/react` direct import |
| D.5 | Props signature `{ data: HarnessSprintWeekState \| null }` | line 22 | PASS |
| E.1 | `queryAllByRole('button').toHaveLength(0)` present | test file lines 67, 80 | PASS — canonical RTL guard appears in both `read_only_no_mutation_controls` and `renders empty state` tests |
| E.2 | Test names match masterplan verbatim | `it(...)` at lines 23, 44, 63 | PASS — all three criterion names case-exact |
| F | Type used, not orphaned | `types.ts:936` declared → `HarnessSprintTile.tsx:19` imported → `:22` used | PASS |

## LLM judgment

### Contract alignment
The tile satisfies each immutable criterion with a durable attribute-based surface, not a flimsy text match: `data-section="weekly-state"`, `data-cell="sortino-delta"`, `data-cell="thu-candidates"`, `data-cell="fri-promoted-count"`. Tests assert on these attributes plus text content, so a careless refactor that dropped the attribute would fail the test. This is the right coupling strength — tight enough to catch regressions, loose enough to survive layout tweaks.

### Read-only contract at source level (strongest form)
The spec asked "is the read-only contract enforced at the component level, not just at the test level?" — yes. The component source contains only `<section>`, `<div>`, `<h3>`, `<p>`, `<span>` tags. A reviewer reading the file cold would see no interactive elements. This is structurally stronger than a test-only guard because the invariant holds even if someone deletes the test.

### Convention compliance (`frontend.md` / `frontend-layout.md`)
- BentoCard token: `rounded-2xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900 p-6` matches exactly
- Empty state: `IconTimer` + guidance text + sub-text, `py-12` centered — matches §8
- Color semantics: emerald=approved, amber=pending, slate=null/missing — matches §9
- Icon convention: honoured; the researcher flagged that `HarnessDashboard.tsx` violates this and the new code does not replicate the mistake (disciplined)

### Scope honesty
Three carry-forwards, all disclosed explicitly in both contract §"Carry-forwards" and results §"Carry-forwards":
1. Wire into `HarnessDashboard.tsx` — legitimate integration step
2. Backend API `GET /api/harness/sprint-state?week=YYYY-Www` — legitimate backend scope
3. Fix `HarnessDashboard` icon-import violation — unrelated cleanup

None of these are scope-hiding; the parent-owned-fetch design pattern (documented in researcher brief §7) is the recognized React Testing Library convention for pure view components.

### Research-gate compliance
Contract §"References" cites `phase-10.9-research-brief.md` plus the 6 file anchors the researcher produced (component pattern at `AutoresearchLeaderboard.tsx`, insertion point at `types.ts:928-931`, icon source at `icons.ts`, frontend rules). Gate properly traversed.

### Mutation tests (skipped — Q/A tool constraint)
Q/A is Edit-forbidden, so M1/M2/M3 were not executed. Structural evidence substitutes: the component grep for `<button>`/form elements returns zero, and the test file contains the canonical `queryAllByRole('button').toHaveLength(0)` guard at line 67. If a future refactor added a `<button>`, the guard would fail on next run. The mutation resistance of the test suite is therefore evident by construction.

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_109_v1",
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "researcher_gate",
    "mtime_contract_before_results",
    "verbatim_criteria",
    "log_last_not_appended",
    "cycle_v1",
    "immutable_cli",
    "tsc_noemit",
    "handoff_files_exist",
    "no_mutation_elements_in_source",
    "no_event_handlers",
    "no_react_hooks",
    "icon_import_convention",
    "props_signature",
    "canonical_queryAllByRole_guard",
    "test_names_verbatim",
    "type_used_not_orphaned",
    "bentocard_convention",
    "scope_honesty_carry_forwards"
  ],
  "certified_fallback": false,
  "reason": "All 3 immutable criteria met (5/5 Vitest tests passed). Read-only enforced at component source level (zero interactive elements). TypeScript clean. HarnessSprintWeekState declared at types.ts:936 and consumed by the component. BentoCard and icon-import conventions honoured. Research gate properly traversed (6 full sources, 13 URLs, recency scan, gate_passed=true). Carry-forwards are legitimate deferrals with disclosed scope."
}
```

## Recommendations for follow-on steps

Not blockers — informational only:

1. When wiring into `HarnessDashboard`, prefer a parent `useEffect`-fetch + prop-pass pattern over moving the fetch inside the tile, to preserve the tile's current "pure-view, untestable-mutation-surface" property.
2. The backend `GET /api/harness/sprint-state` endpoint should map the phase-10.8 `harness_learning_log` rows into the `HarnessSprintWeekState` shape server-side (do NOT reshape in the frontend), so the type remains a clean contract boundary.
3. The separate `HarnessDashboard` icon-import cleanup ticket is worth closing before phase-11 so the Harness-tab surface has a consistent import story.
