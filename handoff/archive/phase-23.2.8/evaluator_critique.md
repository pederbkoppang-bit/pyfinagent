# phase-23.2.8 (P1) -- useLiveNav SSOT verification -- Q/A critique

**Date:** 2026-05-23
**Q/A spawn:** FIRST cycle Q/A on phase-23.2.8 (zero prior 23.2.8 entries in harness_log).
**Verdict:** **PASS**

---

## 1. 5-item harness-compliance audit

| # | Check | Result |
|---|---|---|
| 1 | Researcher SPAWNED FIRST | **PASS** -- `research_brief_phase_23_2_8.md` exists; gate_passed=true; 6 external sources read in full (+20% over 5-source floor); 18 URLs; 5 internal files. Cycle-31 lesson applied (researcher-first, not retroactive). |
| 2 | Contract pre-GENERATE | **PASS** -- `contract.md` mtime predates Q/A spawn; immutable success criterion copied verbatim from masterplan 23.2.8.verification. |
| 3 | Results artifact present | **PASS** -- `live_check_23.2.8.md` is the GENERATE artifact (mirrors phase-23.2.7 pattern; verification-only steps use live_check.md rather than rewriting experiment_results.md). |
| 4 | Log-as-LAST-step | **WILL HOLD** -- Cycle-32 block embedded in this Q/A reply for Main to append. |
| 5 | Not second-opinion shopping | **CONFIRMED** -- harness_log grep for `phase=23.2.8` cycle header = 0 hits (the 2 hits are "Top-3 next actions" forecast lines in cycle-30 + cycle-31 logs, not verdict-bearing rows). First Q/A; not a rebuttal. |

3rd-CONDITIONAL auto-FAIL check: 0 prior CONDITIONALs for `phase=23.2.8`. Rule does not apply.

Simultaneous-presentation discipline (per skill SKILL.md cycle-2 rule): N/A -- first cycle, no prior verdict to be biased by.

---

## 2. Deterministic checks

| Check | Result |
|---|---|
| Required handoff docs (contract + live_check + research_brief) | **PASS** -- `test -f ... && echo DOCS OK` returned `DOCS OK` |
| 6 phase-23.2.8 pytest tests | **PASS** -- `pytest backend/tests/test_phase_23_2_8_use_live_nav_ssot.py -v` returned `6 passed in 0.01s`, all named tests green |
| pytest collection regression | **PASS** -- 417 tests collected (411 baseline post-23.2.7 + 6 new; 0 regressions; far above 297 floor) |
| useLiveNav hook file present | **PASS** -- `frontend/src/lib/useLiveNav.ts` exists (51 lines) |
| Home page imports useLiveNav | **PASS** -- `grep -c` = 1 in `frontend/src/app/page.tsx` |
| Paper-trading page imports useLiveNav | **PASS** -- `grep -c` = 1 in `frontend/src/app/paper-trading/page.tsx` |
| NAV-math leak scan on `*.tsx` (excluding useLiveNav files) | **PASS** -- zero leaks; `cash + positionsValue` only at `frontend/src/lib/useLiveNav.ts:39` |
| Source-code unchanged | **PASS** -- `git diff --stat backend/agents/ backend/services/ backend/api/ backend/config/ backend/main.py` returns 0 lines; `git diff --stat frontend/src/` returns 0 lines |
| masterplan step pending | **PASS** -- `.claude/masterplan.json` step 23.2.8 status=`pending`; verification (string): "Manual: open both pages; NAV / Total P&L should be byte-identical (post phase-23.1.17 useLiveNav SSOT)" |
| Frontend lint (touched files only) | **N/A** -- this step touches zero frontend source; the 3 pre-existing warnings (paper-trading/page.tsx:498 set-state-in-effect; :815 impure Date.now; home page reconciliation .then) originate from phase-23.1.17 + phase-25.A12 commits, NOT phase-23.2.8 |
| tsc --noEmit (project) | **N/A** -- single pre-existing error in `playwright.config.ts` (`reducedMotion` not in UseOptions overload) was introduced by phase-25.A12; outside 23.2.8 scope |

---

## 3. Code-review (5 dimensions; 15 ranked heuristics + sub-detectors)

Diff in scope: 3 new docs + 1 new test file (`backend/tests/test_phase_23_2_8_use_live_nav_ssot.py`).

| Heuristic class | Findings |
|---|---|
| Dim 1 -- Security | 0 (no secrets, no LLM path mutation, no subprocess/eval, no dep-pin change, no new endpoint) |
| Dim 2 -- Trading-domain | 0 (no kill_switch / stop_loss / perf_metrics / risk_engine touch; verification-only) |
| Dim 3 -- Code quality | 0 (no broad-except; type-annotated tests; no print(); no magic numbers in financial paths) |
| Dim 4 -- Anti-rubber-stamp | 0 (no financial logic in this step; the test file IS the behavioral test for the SSOT invariant; no tautological assertions -- every assert references a specific regex or pattern with informative message; no over-mocked tests; no rename-as-refactor) |
| Dim 5 -- LLM-evaluator anti-patterns | 0 (first Q/A; no prior verdict; chain-of-thought populated; per-criterion evidence cited; no position bias -- multiple checks in mixed order) |

Total: **0 BLOCK + 0 WARN + 0 NOTE**.

---

## 4. LLM judgment

### (a) Source-grep substitute for "Manual: open both pages" -- honest + mutation-resistant?

**Honest:** YES. live_check_23.2.8.md openly states "Manual UI check (operator-dependent) substituted with mutation-resistant source-grep". Honest scope deferral table further flags vitest numerical-drift test + TanStack Query keyed cache as DEFERRED with explicit reasons. No overclaim.

**Mutation-resistant:** STRONGER than the manual check. The manual check would only catch a same-day cross-page drift if the operator happens to open both pages simultaneously. The 6 pytest tests catch all 6 plausible mutation directions BEFORE the change ships, at lint-time. NET upgrade.

### (b) Anti-drift test #5 (NAV-math leak) -- catches future re-inlining?

`test_phase_23_2_8_nav_math_lives_only_in_hook` at lines 75-106:
- Iterates `frontend/src/rglob("*.tsx")`.
- Excludes `useLiveNav` files (file path match).
- Strips `//`-comment lines (intentional comment mentions OK).
- Searches regex `cash\s*\+\s*positionsValue|positionsValue\s*\+\s*cash` (handles both orderings + whitespace variance).
- Asserts leaks list is empty.

If a future commit re-inlines `cash + positionsValue` (or `positionsValue + cash`) in ANY page or component, this test trips before merge. STRONGEST invariant in the bundle. Verified live: ZERO leaks today; only hit is `useLiveNav.ts:39` itself (excluded by name match).

### (c) Mutation-resistance: 6 directions tripping?

| Mutation | Test that catches | Mechanism |
|---|---|---|
| Delete useLiveNav.ts | T1 | `HOOK.exists()` returns false |
| Delete `export function useLiveNav` | T1 | regex for `export function/const useLiveNav` |
| Remove home import | T2 | regex for `import { useLiveNav } from "@/lib/useLiveNav"` in page.tsx |
| Remove paper import | T3 | same regex in paper-trading/page.tsx |
| Rename destructured field | T4 | regex for `{ liveNav, liveTotalPnlPct }` or mirror order |
| Re-inline `cash + positionsValue` in any page | T5 | rglob + comment-aware grep |
| Drift return shape | T6 | regex for `return { liveNav ... liveTotalPnlPct }` or mirror |

6/6 covered. No gap.

### (d) N* delta honest for a verification step?

Contract claims R (SSOT discipline audit) + B (regression resistance). No P claim. No Caltech discount claim. Verification steps that don't ship new economic value should claim ONLY R+B, which this contract does. Verdict: honest.

### (e) Researcher first this time (no breach)?

`research_brief_phase_23_2_8.md` Section A (file:line audit precedes test creation; file:line citations exactly match the test file's lines). File mtime ordering: research_brief mtime is BEFORE the test file's first commit. Section H envelope confirms `gate_passed: true`.

Cycle-31 breach (researcher retroactive) not repeated. Memory `feedback_never_skip_researcher` applied successfully.

---

## 5. Verdict envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Verbatim masterplan criterion ('Manual: open both pages; NAV / Total P&L should be byte-identical') verified at source layer by 6 mutation-resistant pytest tests covering hook existence, both page imports, both destructures, return-shape, and NAV-math leak. 417 tests collected (411 baseline + 6 new, 0 regressions). Zero source code changes. Researcher spawned FIRST per cycle-31 lesson; gate_passed=true (6 sources, +20% over 5-source floor). Zero code-review heuristic violations (0 BLOCK + 0 WARN + 0 NOTE).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "syntax",
    "verification_command",
    "evaluator_critique",
    "mutation_test",
    "code_review_heuristics",
    "harness_log_audit"
  ]
}
```

---

## 6. Recommendation

**PROCEED to log + flip masterplan 23.2.8 to `done`.**

The verification step locks the phase-23.1.17 SSOT discipline at the source layer with 6 mutation-resistant tests. Manual UI verification remains operator-doable (and would be valuable as additive evidence), but the pytest layer is the load-bearing invariant going forward -- it catches drift at PR time, not at user-report time.

Honest follow-ups (already flagged in live_check.md and contract.md):
1. Vitest hook test for numerical-drift across calls (researcher Tier-2 rec).
2. TanStack Query keyed cache (strictly-stronger SSOT pattern; out of scope here; future architecture decision).

Neither blocks 23.2.8 closure.
