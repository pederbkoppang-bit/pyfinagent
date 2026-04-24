# Q/A Critique — phase-11 audit (qa_id: qa_11_v3)

**Verdict: PASS**
**Cycle:** 3 (cycle-2 remediation on materially changed JSON blocks)
**Date:** 2026-04-20

## 5-item harness-compliance audit (FIRST)

1. **Researcher gate** — PASS. `phase-11-audit-brief.md` Part E lists 6 read-in-full sources (>=5 floor), 16 total URLs, recency scan 2024-2026 present, research-gate checklist all hard blockers checked (lines 336-347).
2. **Contract pre-commit** — PASS. `phase-11-contract.md` (mtime 20:15) predates `phase-11-experiment-results.md` (mtime 20:16) and the updated brief (mtime 20:19).
3. **Results exist** — PASS. `phase-11-experiment-results.md` present with cycle-2 summary.
4. **Log-append deferred** — OK. Main explicitly defers `harness_log.md` append until after Q/A PASS and before masterplan flip (correct order).
5. **No verdict-shopping** — OK. This cycle-3 Q/A is on materially CHANGED evidence: JSON blocks in the brief were rewritten (11.10 semantics, new 10.8.1, three strengthened verification.commands). qa_11_v2's CONDITIONAL was specifically "JSON blocks don't match prose"; Main has now edited the JSON blocks. Fresh Q/A on new evidence is the documented canonical cycle-2 flow (CLAUDE.md harness-protocol block), not shopping.

## Deterministic spot-checks (6/6 PASS)

| # | Item | Expected | Actual | Pass |
|---|------|----------|--------|------|
| 1 | 11.10 renamed to Observability | 1 block, mentions `perf_tracker`/`/api/observability/latency` | 1 block present at line 262-274; 3 matches for perf_tracker/observability-latency in verification.command | yes |
| 2 | 10.8.1 separate block | 1 block with `log_slot_usage` + `inspect.getsource` | 1 block at line 280-296; `log_slot_usage` occurs 10x across brief; `inspect.getsource` 1 match | yes |
| 3 | 11.1 UI greps | both `CostBudgetWatcherTile` and `getCostBudgetToday` greps | both present in 11.1 verification.command (line 141) | yes |
| 4 | 11.3 POST round-trip + onClick | `curl -s -X POST` + `onClick=` approve grep | both present on line 169 | yes |
| 5 | 11.6 selectedWeekIso grep | grep present in 11.6 verification | 2 occurrences (plain grep + `getHarnessSprintState\(\s*selectedWeekIso` pattern on line 211) | yes |
| 6 | Old 11.10 sprint-tile content removed | 0 matches for "Sprint tile - wire real slot-usage data" in 11.10 block | 0 matches anywhere in file | yes |

**No-code invariant:** `git status --porcelain` outside `handoff/current/` and `.claude/` shows 0 `.py`/`.ts`/`.tsx` changes. Pure docs cycle - confirmed.

## What changed vs qa_11_v2 (CONDITIONAL)

qa_11_v2 correctly refused to PASS because the prose cycle-2 plan in the contract/results was not reflected in the paste-ready JSON blocks. That critique has now been addressed in the JSON itself:

- 11.10 semantics flipped from "sprint tile slot-usage wiring" to "observability wiring for phase-11 endpoints" (line 263).
- The slot-usage wiring is correctly re-homed as phase-10.8.1 under "Additional (moved out of phase-11 - belongs to phase-10 backend)" (lines 278-296).
- 11.1, 11.3, 11.6 verification.commands now enforce UI-side evidence (component/export greps, POST round-trip, state-hook grep) - closing the qa_11_v2 gap where endpoint-only checks could pass with no UI wiring.

## LLM judgment

**Contract alignment:** The immutable success criteria in the 10 JSON blocks + 1 10.8.1 block align verbatim with the prose plan in contract/results. No drift.

**Mutation resistance:** The strengthened verification.commands (11.1, 11.3, 11.6, 11.10) would now detect the failure modes the prose plan promises to close. If a future Main implements 11.1 as a pure backend endpoint with no UI wiring, the grep would fail the step. Good.

**Scope honesty:** The 10.8.1 re-homing is the correct scope move - it's a phase-10 backend wiring task wearing phase-11 clothing. Explicit "moved out of phase-11" heading is honest disclosure.

**Research-gate compliance:** Part E shows 6 read-in-full sources + recency scan + 16 total URLs. All hard blockers checked. Exceeds the >=5/>=10 floor.

**Anti-rubber-stamp:** No planted-violation mutation test was run (not applicable to a docs-only cycle). Counter-evidence would be a JSON parse failure in the blocks; I visually inspected the JSON structure - brackets balanced, commas in place.

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_11_v3",
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "file_existence",
    "11.10_observability_rename",
    "10.8.1_separate_block",
    "11.1_ui_grep_pair",
    "11.3_post_and_onclick",
    "11.6_selectedWeekIso",
    "old_11.10_sprint_removed",
    "no_code_invariant",
    "research_gate_compliance",
    "harness_5_item_audit"
  ],
  "certified_fallback": false,
  "reason": "All 6 cycle-2 remediation items verified deterministically. 11.10 block now correctly names Observability wiring with perf_tracker + /api/observability/latency in verification.command; 10.8.1 present as separate block with log_slot_usage + inspect.getsource check; 11.1 enforces CostBudgetWatcherTile + getCostBudgetToday UI greps; 11.3 enforces POST round-trip + onClick approve grep; 11.6 enforces selectedWeekIso grep; old 11.10 sprint-tile content fully removed (0 matches). No code changes outside handoff/ (docs-only cycle). Research gate: 6 read-in-full + 16 URLs + recency scan. Cycle-3 spawn on materially changed JSON blocks - documented cycle-2 flow, not verdict-shopping."
}
```

## Next action for Main

PASS granted. Main may now:
1. Paste the 10 phase-11 sub-steps (11.1-11.10) and phase-10.8.1 into `.claude/masterplan.json`.
2. Append phase-11 audit closure entry to `handoff/harness_log.md` (log-last rule).
3. Flip `phase-11-audit` masterplan status to `done` AFTER the log append.
