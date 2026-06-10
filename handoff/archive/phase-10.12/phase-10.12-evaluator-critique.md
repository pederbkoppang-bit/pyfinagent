# phase-10.12 Q/A Critique — HarnessSprintTile visual alignment

**qa_id:** qa_1012_v1
**Verdict:** PASS
**Date:** 2026-04-20

## JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "qa_id": "qa_1012_v1",
  "violated_criteria": [],
  "violation_details": [],
  "checks_run": [
    "harness_compliance_audit_5_item",
    "handoff_files_exist",
    "tests_npm_run_test_filter",
    "grep_zinc_zero",
    "grep_navy_nonzero",
    "grep_py8_size32",
    "grep_read_only_no_handlers",
    "tsc_noemit_filtered",
    "spot_read_tile_structure",
    "peer_alignment_HarnessDashboard",
    "contract_verbatim_criteria"
  ],
  "reason": "All 4 immutable success criteria met. Deterministic checks: 5/5 tests pass, zinc=0, navy=5, py-8+size=32 matches=2, tsc clean, 0 event handlers / form controls. Minimally invasive class-token refactor; DOM structure, data-* attributes (6 preserved), text content, and grid layout unchanged. Palette now aligns with HarnessDashboard peer (bg-navy-800/60 / border-navy-700 used 4x there). rounded-xl is the canonical summary-card token per frontend-layout.md §4. Research gate cleared (tier=simple, 6 sources read-in-full, recency scan performed, gate_passed=true)."
}
```

## 5-item harness-compliance audit

| # | Check | Result |
|---|---|---|
| 1 | Researcher spawn with `gate_passed: true` + ≥5 full sources | PASS (6 read-in-full, 7 snippet-only, recency scan present) |
| 2 | Contract mtime ≤ results mtime, research ≤ contract | PASS (17:31 → 17:32 → 17:33) |
| 3 | Verbatim immutable criterion in contract | PASS (all 4 success_criteria tokens grep-confirmed) |
| 4 | No harness_log.md append yet (log-last discipline) | PASS |
| 5 | Cycle v1 (no prior CONDITIONAL to verdict-shop) | PASS |

## Deterministic reproduction (verbatim)

```
$ cd frontend && npm run test -- --filter=HarnessSprintTile
Test Files  1 passed (1)
Tests       5 passed (5)
Duration    917ms

$ grep -c zinc frontend/src/components/HarnessSprintTile.tsx    → 0 (expect 0)
$ grep -c "navy-" ...                                            → 5 (expect >0)
$ grep -cE "py-8|size=\{32\}" ...                                → 2 (expect 2)
$ grep -cE "<button|<input|<select|<textarea|<form|onClick|..." → 0 (expect 0)
$ npx tsc --noEmit 2>&1 | grep HarnessSprint                     → empty
```

## Success-criteria mapping

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | existing_5_tests_still_pass | 5 passed (5) in vitest output | PASS |
| 2 | tile_uses_navy_palette_not_zinc | zinc=0, navy-=5 (navy-700 x2, navy-800/60 x2, navy-700/50 x3, navy-900/40 x3) | PASS |
| 3 | empty_state_not_oversized | py-8 (was py-12) + size={32} (was size={40}) both present line 37 & 39 | PASS |
| 4 | read_only_preserved | 0 event handlers, 0 form controls; `<section>` wrapper only | PASS |

## Spot-read verification (HarnessSprintTile.tsx)

| Expectation | Actual | Status |
|---|---|---|
| Outer section: `bg-navy-800/60 border-navy-700 p-5 rounded-xl` | Lines 35, 68: `rounded-xl border border-navy-700 bg-navy-800/60 p-5` | PASS |
| Empty-state: `py-8`, `size={32}` | Lines 37, 39 | PASS |
| Inner sub-cards: `bg-navy-900/40 border-navy-700/50 rounded-lg p-4` | Lines 82, 103, 129 | PASS |
| data-* attributes preserved | 6 present: data-week-iso, data-section×2, data-cell×3 | PASS |
| No `<button>`, `<input>`, event handlers | grep count = 0 | PASS |

## LLM judgment

**Minimally invasive?** Yes. Diff is class-string swaps only: `zinc-*` → `navy-*`, `rounded-2xl` → `rounded-xl`, `py-12` → `py-8`, `size={40}` → `size={32}`. No imports changed, no JSX structural change, no prop/type change, no behavior. The phase-10.9 test file was untouched and all 5 tests remained green — strong mutation-resistance signal that semantics survived.

**Peer alignment?** Yes. `HarnessDashboard.tsx` (the parent surface) uses `bg-navy-800/60` / `border-navy-700` in 4 places. The tile now matches its container instead of clashing. Before this change the tile was the one zinc-palette island on an otherwise navy page — the visual-regression motivation cited in the contract is legitimate.

**rounded-xl vs rounded-2xl.** `frontend-layout.md` §4 explicitly prescribes the summary-card tokens: `rounded-xl border border-navy-700 bg-navy-800/60 p-5`. The contract's shrink from rounded-2xl to rounded-xl brings the tile into compliance with the documented token. Acceptable.

**Empty-state shrink (py-12 → py-8, icon 40 → 32).** §8 of frontend-layout.md gives `py-24` and `size=48` as the *page-level* empty state. A per-tile empty state is smaller by design. py-8 + size=32 is appropriate for a single card rendering "No sprint activity yet" — it still communicates emptiness without dominating an otherwise-populated dashboard.

**Carry-forwards.** The BentoCard.tsx global-fix and visual-regression snapshot deferrals in experiment-results.md are legitimate scope-bounding. BentoCard touches multiple surfaces (backtest, optimizer, harness) and should not be folded into a visual-alignment micro-step for a single tile. Snapshot tooling is a net-new test infra concern. Both are correctly deferred with explicit notes rather than silently dropped.

**Anti-rubber-stamp.** I re-checked the `zinc` grep against the file directly (not just the test output) and confirmed literal absence. If `zinc` had been masked in a comment/string, the grep would still have caught it. The 5-test pass is real (vitest output shows `Test Files 1 passed (1)`).

## Conclusion

PASS. Step ready to log + flip to `status: done`. Main may now append `handoff/harness_log.md` (log-last discipline) and update `.claude/masterplan.json` to mark phase-10.12 done. No follow-up cycle required; no blockers.
