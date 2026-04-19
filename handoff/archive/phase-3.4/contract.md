# Sprint Contract -- phase-3.4 Agent Skill Optimization

**Written:** 2026-04-19 PRE-commit.
**Step id:** `3.4` in phase-3 (last pending step; 3.5 is superseded).
**Immutable verification:** `source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1` with `success_criteria: [evaluator_critique_pass, no_regressions]`.

## Research-gate summary

Researcher envelope `{tier: moderate, external_sources_read_in_full: 6, snippet_only_sources: 5, urls_collected: 11, recency_scan_performed: true, internal_files_inspected: 7, gate_passed: true}`. Brief: `handoff/current/phase-3.4-research-brief.md`.

**Note on research-gate discipline:** this researcher ran BEFORE the new "Search-query composition (mandatory)" section was added to `.claude/rules/research-gate.md`. Its query variants may not include all three (current-year/last-2-year/year-less) per the new rule; the next researcher spawn will comply.

Key research findings:
- `backend/agents/skill_optimizer.py` (866 lines) implements the Karpathy-autoresearch pattern (already formalized in phase-2.10 audit as the absorber).
- **Concrete bug at `backend/services/outcome_tracker.py:107`**: uses `json_io.loads(full)` but the module only imports `json` at line 9, NOT `json_io`. This is a real `NameError` that fires on every `establish_baseline()` call when a stringified report exists in BQ. Verified by grep.
- Zero tests for `SkillOptimizer` (same zero-test gap phase-3.1 had).
- External 2024-2026: OPRO (ICLR 2024), DSPy optimizers, MASS (arXiv 2502.02533), AutoPDL (arXiv 2504.04365). No prior-art supersession; autoresearch pattern is validated and more widely adopted than when phase-2.10 looked.

Staked rec (adopted): narrow closure -- fix the `json_io` NameError, add 5-7 unit tests, document + flip. Same pattern as phase-3.1 joint close.

## Hypothesis

Fixing the `outcome_tracker.py:107` NameError + adding mutation-resistant tests for the `SkillOptimizer` public methods closes the bureaucratic cycle and eliminates a latent production bug that would fire on any skill-optimizer run with a stringified report in BQ.

## Success criteria

**Functional:**
1. `backend/services/outcome_tracker.py:107` bug fixed: `json_io.loads(full)` -> `json.loads(full)` (the module already imports `json`; no new import needed).
2. New tests `backend/tests/test_skill_optimizer.py` with >=5 tests covering:
   - `passes_simplicity_criterion` threshold logic (simplicity gate).
   - `_get_agent_experiments` TSV-round-trip (write + read via monkey-patched RESULTS_TSV).
   - `get_status` returns the expected shape (not-running idle state).
   - `_extract_json` helper correctly isolates a JSON block from surrounding prose.
   - `iteration_counter` modular arithmetic invariant (1-based).
   - `OPTIMIZABLE_AGENTS` constant non-empty + no duplicates.
3. NO changes to `SkillOptimizer` public method signatures, `OPTIMIZABLE_AGENTS`, or `TSV_HEADER`. Behavior-preserving only.
4. Immutable `run_harness.py --dry-run --cycles 1` exit 0 + Sharpe/DSR preserved (1.1705 / 0.9526).

**Correctness verification commands:**
- Syntax: `python -c "import ast; ast.parse(open('backend/services/outcome_tracker.py').read()); ast.parse(open('backend/tests/test_skill_optimizer.py').read()); print('ok')"` -> `ok`.
- Import smoke: `python -c "from backend.agents.skill_optimizer import SkillOptimizer, iteration_counter, _extract_json, OPTIMIZABLE_AGENTS, TSV_HEADER; print('ok')"` -> `ok`.
- Unit tests: `pytest backend/tests/test_skill_optimizer.py -x -q` -> all pass.
- Regression: `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` -> zero failures (excludes the large paper-trading test file that's not part of the phase-3 regression baseline).
- Immutable verify: `source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1` -> exit 0, HARNESS COMPLETE, Sharpe=1.1705 / DSR=0.9526.

**Non-goals:**
- NOT refactoring `revert_modification`'s `git checkout HEAD~1` fragility (research called out but scoped out per research rec).
- NOT adding OPRO/DSPy/MASS pattern integration (future phase if ever).
- NOT touching `meta_coordinator.py`, `quant_optimizer.py`, `perf_optimizer.py` (callers preserve existing interface contracts).
- NOT adding new deps.
- NOT adding `json_io` as an import to `outcome_tracker.py` -- the fix is `json.loads(full)` because the module already has `import json` at line 9 and the file doesn't use `json_io` anywhere else.

## Plan steps

1. Fix `backend/services/outcome_tracker.py:107`.
2. Write `backend/tests/test_skill_optimizer.py` (>=5 tests).
3. Run verification commands.

## References

- `handoff/current/phase-3.4-research-brief.md`
- `backend/agents/skill_optimizer.py:1-866`
- `backend/services/outcome_tracker.py:107` (the NameError)
- `backend/config/prompts.py` (SKILLS_DIR, load_skill)
- External read-in-full: OPRO, DSPy, MASS arXiv 2502.02533, AutoPDL arXiv 2504.04365, Karpathy autoresearch, DSPy via TDS.

## Researcher agent id

`a00a0d2e4d08c442e`
