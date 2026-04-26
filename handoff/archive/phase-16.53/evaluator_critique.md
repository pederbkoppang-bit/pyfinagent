---
step: phase-16.53
title: Settings full-width content fix -- drop max-w-4xl from 3 tab grids
cycle_date: 2026-04-26
verdict: PASS
qa_agent: qa (merged qa-evaluator + harness-verifier)
---

# Q/A Critique -- phase-16.53

## 5-item harness-compliance audit

1. Researcher spawn -- PASS. `phase-16.53-research-brief.md` present, simple tier, internal-only basis explicitly tied to pure-UI cycle precedent (16.43/16.46/16.47/16.48/16.49/16.52). `gate_passed: true`, `internal_files_inspected: 5`, recency scan performed.
2. Contract pre-commit -- PASS. `contract.md` carries `step: phase-16.53` and `verification: cd frontend && npx tsc --noEmit`. Research brief is referenced.
3. Results document -- PASS. `experiment_results.md` exists with header `step: phase-16.53`.
4. Log-last -- PASS. `handoff/harness_log.md` has 0 entries for `phase=16.53` -- correct ordering (log appended only after Q/A PASS).
5. No verdict-shopping -- PASS. First Q/A spawn for phase-16.53; no prior critique to overturn.

## Deterministic checks

| Check | Command | Result |
|-------|---------|--------|
| A. tsc --noEmit | `cd frontend && npx tsc --noEmit` | exit=0 |
| B. ESLint | `npm run lint` | 0 errors, 34 pre-existing warnings (no new) |
| C. max-w-4xl removed | `grep -c 'max-w-4xl' frontend/src/app/settings/page.tsx` | 0 |
| D. max-w-fit retained | `grep -c 'max-w-fit' frontend/src/app/settings/page.tsx` | 1 (tab bar preserved) |
| E. No regression on other pages | `grep -l 'max-w-4xl' frontend/src/app/*/page.tsx` | empty |

## LLM-judgment leg

- **Diagnosis correct**: Yes. Operator screenshot showed dead whitespace on right; `max-w-4xl` (~896px) on a wide viewport explains the ~30% waste. Removing the wrapper constraint allows the inner `lg:grid-cols-2` to fill the parent container.
- **Fix minimal**: Yes. Three single-line edits to the 3 grid wrappers (Models L601, Cost & Weights L794, Performance L978). No change to card internals, tab bar, or page shell. Matches the canonical "page shell + scrollable main" pattern from `frontend-layout.md`.
- **Tab bar preserved**: Yes. `max-w-fit` on the tab bar correctly retained (verified by grep returning 1).
- **Out-of-scope discipline**: Per the contract's "Out of scope" -- card internals untouched. Verified by the surgical 3-line diff scope.
- **Material defects**: None blocking masterplan flip.

## Verdict

**PASS** -- safe to append `harness_log.md` block then add the phase-16.53 entry to `.claude/masterplan.json` with `status: done`.

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": "",
  "certified_fallback": null,
  "checks_run": ["harness_compliance_audit", "tsc_noemit", "eslint", "grep_max_w_4xl_removed", "grep_max_w_fit_retained", "grep_other_pages_no_regression", "contract_header", "research_brief_gate", "log_last_ordering"]
}
```
