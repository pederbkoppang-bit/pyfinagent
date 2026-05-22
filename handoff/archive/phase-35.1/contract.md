# phase-35.1 -- Learn-Loop Writer Wiring (live-verify, code change)

**Step id:** `phase-35.1`
**Date:** 2026-05-22
**Mode:** EXECUTION (NOT plan-only). One harness pass. Backend code change behind feature flag.
**Author:** Main (Claude Opus 4.7, this Claude Code session)
**Cycle in `handoff/harness_log.md`:** Cycle 13 (after Cycle 12 phase-45.0 CLOSURE).

---

## North-star delta (mandated by /goal)

**Terms:** R (immediate) + P (speculative).

**R (immediate):** persisting `outcome_tracking` rows on every stop_loss_trigger SELL means future cycles can compute MAE-aware exit metrics (give-back ratio, capture-ratio drift, holding-day percentiles). Reduces R by surfacing the cost of late exits before they compound into next cycle's positions.

**P (speculative):** writing `agent_memories` lessons enables BM25 retrieval on next cycle's analyze step. The 28-agent pipeline already builds situation descriptions and retrieves memories (per `backend/agents/memory.py::FinancialSituationMemory`). With non-empty memories, future analyses gain priors from past closures. Magnitude unknown until index has >=5 rows; conservatively estimate +0.05-0.20 Sharpe improvement over a 60-day window assuming lesson-quality ~50% (Caltech adversarial finding: LLM agents systematically deviate from human traders; conservative discount applied).

**How measured:** Pre-step `outcome_tracking` row count = 0; post-step manual-cycle row count >= 1. `agent_memories` row count = 0 -> >= 1 after operator flips the flag and one stop_loss_trigger close fires. Long-run: 60-day forward Sharpe delta vs pre-35.1 baseline (deferred measurement to phase-43.0 DoD).

---

## Research-gate decision

**Researcher SKIPPED** (justified per /goal conditional clause "Researcher if new external OR roadmap tags 'refresh-on-touch'").

Justification:
- `closure_roadmap.md` §3 already documents the writer-gap diagnosis with file:line precision (`outcome_tracker.py:74` + `outcome_tracker.py:189`).
- `closure_roadmap.md` §9 already documents the fix path (writer fan-out + idempotency on outcome_id).
- 2026 best-practice frame already cited in research_brief.md (cycle 12): event-sourcing-lite idempotent UPSERT pattern (the BQ `save_outcome` + `save_agent_memory` writers already exist; the gap is at the DISPATCHER level, not the writer level).
- BM25 cold-start pattern: the existing `backend/agents/memory.py::FinancialSituationMemory` handles empty-index retrieval (returns empty list); no new pattern needed.

No new external dependencies. No new BQ migration (tables already exist with correct schemas). Pure dispatcher fix.

---

## Hypothesis

> If we modify `backend/services/autonomous_loop.py::_learn_from_closed_trades`
> to (1) call `tracker._generate_and_persist_reflections(outcome, full_report)`
> AFTER `evaluate_recommendation` returns a non-None outcome, AND (2) add
> a fallback write path when `evaluate_recommendation` early-returns None
> (due to yfinance flake or missing analysis_date) -- both behind feature
> flag `PAPER_LEARN_LOOP_ENABLED` default OFF -- AND we add a pytest test
> that exercises both paths with mocked BQ; THEN flipping the flag to true
> and triggering one cycle that produces a stop_loss_trigger SELL will
> result in >= 1 `outcome_tracking` row + >= 1 `agent_memories` row in BQ.

If true: phase-35.1 closes. agent_memories has lessons that BM25 will retrieve on phase-35.1's next pickup.

If false: either yfinance still flakes silently, idempotency check breaks the second-fire scenario, or the lesson-generation LLM call (Gemini-2.5-pro per phase-34.1) fails — diagnose + fix + fresh Q/A.

---

## Immutable success criteria (verbatim from masterplan 35.1.verification)

1. `outcome_tracking_has_at_least_one_row_from_autonomous_loop_after_real_close`
2. `agent_memories_bm25_retrieve_returns_at_least_one_lesson_on_next_cycle`
3. `live_check_quotes_the_outcome_row_and_the_loaded_lesson`

Plus the 10 integration gates per /goal:
4. `pytest_backend_count_at_least_297` (baseline locked at phase-45.0)
5. `ts_build_unchanged_no_frontend_edits`
6. `feature_flag_PAPER_LEARN_LOOP_ENABLED_default_OFF_in_settings_py_and_env_example`
7. `bq_no_new_migration_required_existing_tables_outcome_tracking_and_agent_memories`
8. `env_var_documented_in_backend_env_example_and_CLAUDE_md`
9. `contract_has_north_star_delta` (this document, above)
10. `zero_emojis_in_changed_files`
11. `ascii_only_loggers_in_changed_files`
12. `single_source_of_truth_no_duplicate_writer_logic_outcome_tracker_remains_authoritative`
13. `harness_log_cycle_13_appended_BEFORE_status_flip_to_done`

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Pre-cycle health check + writer-gap callsite located (`autonomous_loop.py::_learn_from_closed_trades` line 1714; `outcome_tracker.py::evaluate_recommendation` line 35 with early-return at line 45; `_generate_and_persist_reflections` line 152 only called by `evaluate_all_pending`) | DONE |
| 2 | Researcher decision (SKIP justified) | DONE |
| 3 | Write this contract | IN FLIGHT |
| 4 | Add `paper_learn_loop_enabled: bool = Field(False, ...)` to `backend/config/settings.py` + `PAPER_LEARN_LOOP_ENABLED=` in `backend/.env.example` + CLAUDE.md env block note | NEXT |
| 5 | Modify `_learn_from_closed_trades` in `autonomous_loop.py`: gate new fan-out behind flag; call `_generate_and_persist_reflections` after non-None outcome; add fallback path for None outcome (use trade fields directly) | NEXT |
| 6 | Add pytest test `backend/tests/test_phase_35_1_learn_loop_writer.py` exercising both paths with mocked BQ | NEXT |
| 7 | Run `pytest backend/ --collect-only -q` -- confirm count >= 297 (target: 297 + ~3 new tests = ~300) | NEXT |
| 8 | Run `python -c "import ast; ast.parse(open(f).read()) for f in changed_files"` -- syntax OK | NEXT |
| 9 | Grep changed files for emojis + non-ASCII logger strings -- zero hits | NEXT |
| 10 | Write `handoff/current/live_check_35.1.md` documenting: pytest PASS, code change summary, the flag-OFF rationale, and the operator's path to enable + verify (one `/run-now` trigger with PAPER_LEARN_LOOP_ENABLED=true) | NEXT |
| 11 | Spawn Q/A ONCE with 5-item compliance audit | NEXT |
| 12 | If Q/A PASS: append harness_log Cycle 13 FIRST, then flip 35.1 status to done LAST (auto-commit + push, prefix `phase-35.1:`) | NEXT |

---

## Files this step will touch

- `backend/config/settings.py` (+1 Field declaration, ~2 lines)
- `backend/.env.example` (+1 env line + 1 comment line)
- `CLAUDE.md` (+1 env-block note line, optional)
- `backend/services/autonomous_loop.py` (modify `_learn_from_closed_trades` function only; ~30-50 lines changed)
- `backend/tests/test_phase_35_1_learn_loop_writer.py` (NEW, ~80-120 lines)
- `handoff/current/contract.md` (this file)
- `handoff/current/live_check_35.1.md` (created post-test)
- `handoff/current/evaluator_critique.md` (Q/A overwrite at end)
- `handoff/harness_log.md` (Cycle 13 append)
- `.claude/masterplan.json` (status flip 35.1 + 35 parent at very end)

**Out of scope this step:**
- `backend/services/outcome_tracker.py` (NOT modified -- single source of truth; the dispatcher fix lives in autonomous_loop)
- `backend/services/paper_trader.py` (NOT modified -- the closure_roadmap §3 said "paper_trader.py" but the actual writer-gap is in the autonomous_loop dispatcher; correcting the closure_roadmap's slight imprecision)
- Any frontend file

---

## References

- `handoff/current/closure_roadmap.md` §3 (BQ-probe findings), §9 (audit_basis upgrade verbatim), §11 (execute-prompt skeleton)
- `handoff/current/research_brief.md` (cycle 12, 11 sources, gate_passed=true)
- `backend/services/autonomous_loop.py:1714-1784` (current `_learn_from_closed_trades`)
- `backend/services/outcome_tracker.py:35-85, 152-198` (existing writer functions, NOT modified)
- `.claude/masterplan.json` step 35.1.audit_basis (upgraded in phase-45.0)
- /goal directive (10 integration gates + circuit breakers)
- Caltech arxiv:2502.15800 (adversarial finding cited in closure_roadmap §3 -- "LLM agents systematically deviate from human traders") -- conservative discount applied to N* P estimate
