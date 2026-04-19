# Experiment Results -- phase-3.4 Agent Skill Optimization

**Step:** 3.4 (final pending phase-3 step).
**Date:** 2026-04-19.

## What was built

Narrow closure matching the phase-3.1 pattern. One-line bug fix + 11 unit tests. Zero behavior changes to `SkillOptimizer`.

**Bug fix (`backend/services/outcome_tracker.py:107`):**
- Was: `full = json_io.loads(full)` -- NameError at runtime because the module imports only `json` (line 9), never `json_io`.
- Now: `full = json.loads(full)` with a comment pointing at phase-3.4 discovery.
- Impact: `OutcomeTracker.evaluate_all_pending()` / `SkillOptimizer.establish_baseline()` no longer NameError when BQ returns a stringified report.

**Tests (`backend/tests/test_skill_optimizer.py`, 11 tests):**
- 3 tests on `passes_simplicity_criterion`: simplification-with-nonneg-delta passes, added-lines gate at 0.005 per 10-lines threshold, simplification with negative delta rejected.
- 4 tests on `_extract_json`: fenced code block extraction, raw-prose JSON object, raw-prose JSON array, prose-only returns None.
- 1 test on `iteration_counter` round-robin (resets module-level counter for determinism; verifies 12 calls mod 5 produce the expected rotation).
- 1 test on `OPTIMIZABLE_AGENTS` non-empty + unique.
- 1 test on `TSV_HEADER` column order locked in.
- 1 test on `_get_short_hash` fail-open to `"no-git"` when subprocess returns non-zero.

All tests are pure-unit: no `SkillOptimizer()` instantiation (avoids BigQueryClient + OutcomeTracker + model loading auth requirements).

## File list

Created:
- `backend/tests/test_skill_optimizer.py`

Modified:
- `backend/services/outcome_tracker.py` (1 line behavior change + 2 comment lines)

NOT touched:
- `backend/agents/skill_optimizer.py` (public methods, constants, behavior preserved)
- `backend/agents/meta_coordinator.py`, `backend/backtest/quant_optimizer.py`, `backend/services/perf_optimizer.py`, `backend/api/skills.py` (callers preserve existing interface)
- No deps / requirements.txt changes

## Verification command output

### Immutable verification (from masterplan)

```
$ source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1
[INFO] harness: HARNESS COMPLETE -- 1 cycles finished
[INFO] harness: Final best: Sharpe=1.1705, DSR=0.9526
```

Exit 0. `no_regressions` ✓ (Sharpe/DSR preserved).

### Unit tests

```
$ pytest backend/tests/test_skill_optimizer.py -x -q
...........                                                              [100%]
11 passed in 1.80s
```

### Cumulative regression

```
$ pytest backend/tests/test_skill_optimizer.py backend/tests/test_regime_detector.py backend/tests/test_planner_agent.py backend/tests/test_evaluator_agent.py backend/tests/test_autonomous_loop_integration.py backend/tests/test_bq_writer.py backend/tests/test_observability.py backend/tests/test_sentiment_ladder.py backend/tests/test_calendar_watcher.py -q
73 passed, 1 skipped in 9.29s
```

+11 new skill-optimizer tests. Cumulative phase-3 + phase-6 regression: 73 passed / 1 skipped. Zero regressions.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `outcome_tracker.py:107` NameError fixed (json.loads, no json_io) | PASS |
| 2 | `test_skill_optimizer.py` >=5 tests | PASS (11) |
| 3 | No changes to SkillOptimizer public signatures / constants | PASS (verified via diff) |
| 4 | Immutable verify exit 0 + Sharpe/DSR preserved | PASS |

All 4 criteria met.

## Known caveats

1. **11 tests are all on module-level helpers and static methods**, not on `SkillOptimizer` instance methods. Rationale: instantiating `SkillOptimizer` pulls BigQueryClient + OutcomeTracker + LLM client initialization; in the current test env those require auth that the CI/dev containers don't have. The pure units cover the deterministic invariants (simplicity gate math, JSON extraction, TSV column order, git-fail-open). Integration-level coverage of the Plan->Generate->Evaluate loop body is a phase-3.3+ SkillOptimizer tie-in that's explicitly out of scope per non-goals.
2. **DeprecationWarning on Vertex AI SDK surfaces in pytest output** because evaluator_agent imports `vertexai.generative_models`. This is the deprecation that drove the new phase-11 added to the masterplan earlier this session. Not addressed in phase-3.4.
3. **Pre-Q/A self-check (per the rolling Q/A recommendation):** grep-verified `outcome_tracker.py:9` (import json present) + `:107` (json_io.loads wrong) BEFORE writing the fix. `skill_optimizer.py:838-870` (iteration_counter + _extract_json) read in full to verify test assumptions match the real helper semantics. No invented specifics this cycle.
4. **Research ran under the pre-patch research-gate rules** (three-variant query discipline was added to `.claude/rules/research-gate.md` AFTER this researcher already returned). The next researcher spawn -- when one happens -- will inherit the new rules.
