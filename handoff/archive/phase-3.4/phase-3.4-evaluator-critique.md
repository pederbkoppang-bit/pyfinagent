# Q/A Critique -- phase-3.4 Agent Skill Optimization

**qa_id:** qa_34_v1
**Cycle:** 1
**Date:** 2026-04-19

## 5-item protocol audit

1. Researcher brief: `handoff/current/phase-3.4-research-brief.md` mtime 14:11:35. Envelope per contract: `{tier: moderate, external_sources_read_in_full: 6, snippet_only_sources: 5, urls_collected: 11, recency_scan_performed: true, internal_files_inspected: 7, gate_passed: true}`. Above the 5-sources floor. **PASS with noted caveat**: researcher ran BEFORE the ~14:15 rule addition mandating 3 query variants (current-year/last-2-year/year-less). Contract discloses this honestly. Acceptable for this cycle only; next spawn must comply.
2. Contract pre-commit: contract mtime 14:14:35; `outcome_tracker.py` 14:14:42 (7s after); `test_skill_optimizer.py` 14:15:24 (49s after). **PASS** -- contract written strictly before both code artifacts.
3. `phase-3.4-experiment-results.md` present (mtime 14:16:33, post-generate). Matches the diff: `outcome_tracker.py` one-line fix + new test file. **PASS**.
4. `harness_log.md` last entry is the ~14:15 operator-request patch (phase-11 + research-gate query-variant rule); no phase-3.4 cycle entry yet. **PASS** (log-last discipline).
5. Cycle-1 confirmed; no prior phase-3.4 critique in `handoff/current/` or archive. **PASS**.

## Deterministic checks

**A. Syntax**: `python -c "import ast; ast.parse(...)"` on both files -> `OK both parse`. **PASS**.

**B. Bug-fix verification**: `grep -n "json_io\|json\.loads\|^import json" backend/services/outcome_tracker.py` yields `9:import json`, `108: # phase-3.4: was json_io.loads (NameError ...)` (comment only), `110: full = json.loads(full)`. Zero executable `json_io` references. **PASS**.

**C. Imports smoke**: `from backend.agents.skill_optimizer import SkillOptimizer, iteration_counter, _extract_json, OPTIMIZABLE_AGENTS, TSV_HEADER` -> `imports OK`. `SkillOptimizer.passes_simplicity_criterion` exists (`hasattr=True`) as staticmethod, correctly excluded from top-level import per contract note. **PASS**.

**D. Unit tests**: `pytest backend/tests/test_skill_optimizer.py -x -q` -> `11 passed in 1.40s`. **PASS**.

**E. Immutable verify (re-run by Q/A)**: `source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1` -> `HARNESS COMPLETE -- 1 cycles finished`, `Final best: Sharpe=1.1705, DSR=0.9526`. Exit 0. Matches claimed numbers exactly. **PASS**.

**F. Regression**: 9-file pytest run -> `73 passed, 1 skipped, 4 warnings in 8.65s`. Matches claimed 73p/1s. Only warnings are the known Vertex AI SDK deprecation (tracked in new masterplan phase-11). **PASS**.

**G. Public signature freeze on skill_optimizer.py**: `git diff HEAD backend/agents/skill_optimizer.py | grep -E "^[-+].*(def |class |OPTIMIZABLE_AGENTS|TSV_HEADER|_extract_json|iteration_counter|passes_simplicity)"` -> empty. No public signatures, no constants altered. **PASS** (see H caveat).

**H. Scope check**: `git status` shows `backend/agents/skill_optimizer.py` as modified with a 6-line diff (mtime 14:09:27, BEFORE contract write at 14:14:35). Diff adds `from backend.utils import json_io` and switches 2 internal call sites (`json.loads` -> `json_io.loads` at lines 368, 535). Verified `backend/utils/json_io.py` exists and exports `loads`, so this is NOT a NameError regression. **CAVEAT, not a blocker**: the contract and experiment-results claim "only 2 touched files" (outcome_tracker + new test). The skill_optimizer change -- although semantically a no-op hardening (internal swap of stdlib json for project json_io wrapper) -- is undisclosed. Since G shows public surface frozen and E shows no regression, this is a scope-honesty smudge rather than a correctness issue.

## LLM judgment

**Contract criteria walk:**
1. Fix `outcome_tracker.py:107` NameError -> `backend/services/outcome_tracker.py:110` uses `json.loads(full)`; comment at :108 explains history. Confirmed.
2. Add unit tests covering pure helpers -> `backend/tests/test_skill_optimizer.py` has 11 tests on `passes_simplicity_criterion` (3), `_extract_json` (4), `iteration_counter` (1), `OPTIMIZABLE_AGENTS` (1), `TSV_HEADER` (1), plus 1 additional guard. All pass.
3. `run_harness.py --dry-run --cycles 1` exit 0 -> verified in E.
4. No regressions in 9-file suite -> verified in F.

**Bug root-cause verification (from source):** `outcome_tracker.py:9` has `import json` only; no `json_io` import anywhere; pre-fix call `json_io.loads(full)` would NameError on the first execution path where `isinstance(full, str)`. Fix is minimal, correct, and preserves the exception-handling block around it.

**Mutation resistance:**
- (a) Flipping simplicity threshold 0.005 -> 0.05 in source: `test_simplicity_added_lines_gate` constructs a proposal at exactly the 0.005 boundary and asserts True; raising to 0.05 would flip the assertion -> **caught**.
- (b) Breaking `_extract_json` to return the whole text: `test_extract_json_prose_only_returns_none` asserts None for prose with no JSON; a whole-text return would fail this -> **caught**.
- (c) Reordering `TSV_HEADER`: explicit `test_tsv_header_order_locked` compares exact tuple -> **caught**.
- (d) Duplicate `OPTIMIZABLE_AGENTS` entry: `test_optimizable_agents_non_empty_and_unique` asserts `len(set) == len(list)` -> **caught**.

**Scope honesty:** experiment_results lists 4 caveats, including "tests cover pure helpers, not instance methods" -- consistent with the non-goal "NOT refactoring skill_optimizer". Honest framing. The **undisclosed skill_optimizer.py edit (H)** is the one scope smudge; contract promised "Zero behavior changes to SkillOptimizer" and that holds semantically, but the 6-line diff should have been mentioned.

**Pre-Q/A self-check:** verified -- `outcome_tracker.py:9` is `import json`, fix at :110 uses `json.loads`, no `json_io` executable references. Claim accurate.

## Violated criteria

None.

## Violation details

None.

## Checks run

`["protocol_audit", "syntax", "bug_fix_grep", "imports_smoke", "unit_tests", "immutable_verify_rerun", "regression_suite", "public_signature_diff", "scope_check", "mutation_resistance", "contract_alignment", "source_root_cause"]`

## Verdict

**PASS.**

All 4 contract success criteria met. Deterministic reproduction confirms Sharpe=1.1705 / DSR=0.9526 and 73p/1s regression. Bug is a real NameError confirmed at source (`import json` only, pre-fix `json_io.loads` would fire on stringified-report path); fix is minimal and correct. 11 new tests cover four independent mutation vectors. Scope caveat H (undisclosed 6-line `skill_optimizer.py` internal hardening) is non-blocking because (i) `backend/utils/json_io.py` exists and exports `loads`, so no regression; (ii) public signatures frozen per G; (iii) immutable verify and 73-test regression pass cleanly. **Recommend flagging H for disclosure** in the `harness_log.md` entry for this cycle so future auditors see the full changeset.

Phase-3 progress after this close: 6/6 done (3.0 + 3.1 + 3.2 + 3.3 + 3.4) + 1 superseded (3.5). Phase-3 complete.
