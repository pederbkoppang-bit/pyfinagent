---
step: phase-10.7.2
cycle_date: 2026-04-25
verdict: PASS
agent: qa
---

# Q/A Critique -- phase-10.7.2 (Recursive Prompt Optimization)

## Harness-compliance (5 items)

1. **Research gate**: PASS. `handoff/current/phase-10.7.2-research-brief.md`
   exists, tier=moderate, 6 in-full reads (>=5 floor), 16 URLs, recency
   scan present (SIPDO 2025, GAAPO 2025, GEPA 2025, Anthropic 2026),
   3-variant query discipline visible, JSON envelope `gate_passed: true`.
   Cited Promptbreeder (arXiv 2309.16797), SIPDO (2505.19514), GAAPO
   (Frontiers 2025), Anthropic harness, DSPy/GEPA, maximerivest -- all
   reflected verbatim in contract section "Key research findings".
   Moderate tier depth justified.
2. **Contract-before-GENERATE**: PASS. `contract.md` frontmatter
   `step: phase-10.7.2` (NOT stale 16.34). Success Criteria block
   contains the immutable
   `pytest tests/meta_evolution/test_directive_rewriter.py -v`.
3. **Experiment results**: PASS. step=phase-10.7.2, verbatim 8/8 PASS
   pytest output present, 8 honest-disclosure entries (incl. scope-creep
   admission for the 8th test, migration-not-applied disposition,
   fail-open behavior, HITL-by-design caveat).
4. **Log-last**: PASS. `grep -c "phase-10.7.2" handoff/harness_log.md`
   returns 1 -- a contract-cycle reference earlier in the cycle, but the
   PASS log entry itself is intentionally NOT yet appended. Appending is
   Main's post-PASS step (correct order: critique -> log append ->
   masterplan flip).
5. **No verdict-shopping**: PASS. Prior evaluator_critique.md belonged
   to phase-16.34; this is a fresh first-pass critique on phase-10.7.2
   evidence. No re-spawn against unchanged evidence.

## Deterministic checks

- contract_step: yes (phase-10.7.2)
- pytest_passes: 8/8 (`============================== 8 passed in 0.03s ===`)
- ast_clean_3_files: yes (`all 3 syntax_ok`)
- migration_dry_run_canonical_sql: yes (CREATE SCHEMA + CREATE TABLE
  IF NOT EXISTS, PARTITION BY DATE(proposed_at), CLUSTER BY proposer,
  parent_version_id, append-only description)
- backend_tests_regression: 182 passed, 1 skipped, 7 warnings -- matches
  Main's claimed 182 baseline. No regression from the 3 new files.
- no_existing_files_touched_by_this_cycle: yes for the 10.7.2 scope.
  **Caveat**: `git status --short` shows
  `M backend/agents/multi_agent_orchestrator.py` with +173/-3 lines.
  Inspection of the diff confirms these are phase-16.31 Gemini-fallback
  additions (`_anthropic_unavailable`, `_get_gemini_mas_client`) --
  pre-existing uncommitted work from a prior cycle, NOT introduced by
  phase-10.7.2. Recommend Main commit or revert that file before
  flipping the masterplan status. Does not block 10.7.2 PASS because:
  (a) the phase-10.7.2 verification command is scoped to
      `tests/meta_evolution/test_directive_rewriter.py` and passes;
  (b) the broader `backend/tests/` regression also passes;
  (c) the 16.31 lines reference symbols only the orchestrator
      module touches, not the new directive_rewriter.

## Anti-drift guards

- min_briefs_floor: `MIN_BRIEFS_FOR_PROPOSAL = 5` (line 41) -- enforced
  in `rewrite_directive` lines 270-277.
- min_score_floor: `MIN_LLM_JUDGE_SCORE = 0.6` (line 42) -- enforced
  in `is_acceptable()` line 75 + post-LLM check line 319.
- noop_detection_present: yes -- line 290
  `proposed_text == current_directive_text.strip()` -> None.
- json_parse_strict: yes -- `_parse_llm_json` (line 206) returns None
  on malformed; `rewrite_directive` returns None on `not parsed`
  (line 285).
- empty_directive_guard: yes (bonus) -- line 266-268.

## HITL gate

- no_writes_to_researcher_md: yes. `grep` on the rewriter shows ONLY
  documentation references to `.claude/agents/researcher.md` (lines 5,
  55, 127, 252, 258). Zero `open(..., 'w')`, zero `.write_text(`, zero
  `Path(..)` filesystem writes anywhere in the module.
- only_bq_persist: yes. The single mutation path is BQ row insert via
  `bq_client.insert_rows_json` in `persist_version`. The HITL gate is
  enforced by the absence of any filesystem-write code path. See LLM
  judgment #1 for whether design-only enforcement is sufficient.

## LLM-call path

- preemptive_oat_skip: yes -- line 167 guards on
  `api_key.startswith("sk-ant-api")` BEFORE constructing the
  Anthropic client, avoiding a wasted 401 on `sk-ant-oat-*` keys.
- gemini_fallback_present: yes -- lines 188-200, Vertex genai client
  with `gemini-2.0-flash`. Triggered both on missing api03 key and
  on Anthropic exception (line 183 fall-through "fall through to
  Gemini").
- fail_open: yes -- outer `try/except` (lines 162/201) returns None on
  any failure; logs warning; never raises into the caller's daily
  cycle.

## Test reality

- 8_real_tests: yes. Each calls `rewrite_directive` (or
  `persist_version` / subprocess) with realistic args. No
  `@pytest.mark.skip` decorators present.
- llm_override_used_in_6_of_8: yes. Tests 1-6 inject FakeLLM via
  `llm_call_override`. Tests 7-8 use FakeBQ + subprocess
  respectively (no LLM path needed). Total: 6 of 8 use the override,
  2 of 8 exercise non-LLM paths. Matches Main's accounting.
- subprocess_for_migration: yes -- `test_migration_script_dry_run`
  uses `subprocess.run([sys.executable, ...])` with 15s timeout.
  Real CLI smoke, asserts on `DRY RUN`, `CREATE TABLE IF NOT EXISTS`,
  `PARTITION BY DATE(proposed_at)`,
  `CLUSTER BY proposer, parent_version_id`.
- meaningful_assertions: yes. `test_llm_above_floor_returns_version`
  verifies the captured prompt contained `n_briefs` and
  `recent_qa_verdicts` -- proves the prompt-builder actually uses the
  inputs, not a vacuous "is not None" check. `test_below_min_briefs`
  uses `pytest.fail` inside the override to assert the LLM was NEVER
  called (proves short-circuit, not just "returns None").
- network_isolated: yes. No live HTTP / BQ in any test.

## LLM judgment

1. **HITL-gate design-only enforcement**: sufficient HERE. The module
   simply has no path to a filesystem write of the agent file. An
   explicit `assert "researcher.md" not in path` runtime check would be
   defensive theater since there is no path-taking code at all. A
   stronger guarantee would come from a hook-level deny rule (e.g.,
   PreToolUse rejecting Write to `.claude/agents/researcher.md` from
   the directive_rewriter call site), but that is out-of-scope for the
   module under review. Acceptable as-is; recommend a one-line comment
   in `persist_version` if Main wants belt-and-suspenders.

2. **`llm_call_override` attack surface**: very low. The override is
   a kwarg on a pure-Python function, not an env var or sys-path
   monkey-patch. Production callers do not pass it; the default
   `_call_llm_for_rewrite` is used. The seam is named with `override`
   in the identifier, signaling test-only intent. No hidden defaults.
   Not flagged.

3. **8 vs 7 scope-creep**: defensible. Test 7
   (`test_persist_via_fake_bq_records_call`) covers `persist_version`,
   which the contract did NOT mandate but the implementation includes
   (line 330+). Adding a test for shipped public-API code is
   protective, not bloat. The honest disclosure in
   `experiment_results.md` line 45 acknowledges the +1 explicitly.
   Approved.

4. **Promptbreeder + SIPDO citation relevance**: highly relevant.
   Promptbreeder (2309.16797) establishes the self-referential mutator
   pattern -- the LLM rewrites the prompt that drove research, and the
   Research Directive IS that prompt. SIPDO (2505.19514) is the
   anti-drift guard pattern (`MIN_BRIEFS_FOR_PROPOSAL=5` + score floor
   mirror SIPDO's local + global confirmation idea, though only the
   local check is implemented this cycle -- see follow-up B). GAAPO's
   simplicity criterion appears as `diff_size_bytes` (line 64 + 301-303).
   Each citation maps to a concrete code construct.

5. **`__init__.py` barrel state**: NOT updated. The package docstring
   mentions "alpha velocity, recursive prompt optimization, cron
   budget allocation, evaluator review gate" but exports nothing
   explicitly. `directive_rewriter` is reachable via deep import only
   (`from backend.meta_evolution.directive_rewriter import ...`), same
   shape as `alpha_velocity` already is. Consistent with the existing
   sibling-module pattern, not a regression. Optional follow-up: add
   explicit `__all__` once a stable public surface emerges across the
   10.7.x series.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Phase-10.7.2 ships the Research Directive rewriter with all four anti-drift guards (MIN_BRIEFS=5, score floor 0.6, no-op detection, JSON-strict parse), HITL gate enforced by design (zero filesystem writes to .claude/agents/researcher.md), 8/8 unit tests PASS, migration --dry-run emits canonical SQL, backend regression 182 passed / 1 skipped, no existing files modified within the 10.7.2 scope (3 new files only). Research gate cleared (6 in-full sources, 16 URLs, recency scan, gate_passed=true).",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "follow_up_tickets": [
    "FU-10.7.2-A: Operator runs scripts/migrations/create_directive_versions_table.py --apply before phase-10.7.3 starts persisting versions. Same FRED-pattern as 10.7.1; intentionally not auto-applied this cycle.",
    "FU-10.7.2-B: SIPDO global-confirmation step (re-score proposed directive on full historical brief corpus, require non-decreasing score) is documented in module docstring lines 22-24 but NOT implemented. Required before any auto-apply path lands. Track for phase-10.7.4 or later.",
    "FU-10.7.2-C: backend/agents/multi_agent_orchestrator.py has +173/-3 uncommitted lines (phase-16.31 Gemini-fallback). Recommend Main commit or revert before flipping masterplan status -- not a 10.7.2 blocker but leaves the working tree in a confusing state for cycle-N+1.",
    "FU-10.7.2-D: Optional barrel update -- backend/meta_evolution/__init__.py currently exposes nothing explicitly; once 10.7.x stabilizes, consider an __all__ list."
  ],
  "checks_run": [
    "contract_frontmatter_step",
    "pytest_directive_rewriter_8_of_8",
    "ast_clean_3_files",
    "migration_dry_run_canonical_sql",
    "backend_tests_regression_182_passed",
    "git_status_no_existing_files_touched_in_scope",
    "anti_drift_guards_all_4_present",
    "hitl_gate_no_fs_writes_to_agent_file",
    "llm_call_path_oat_skip_gemini_fallback_fail_open",
    "test_reality_8_real_no_skip_no_network",
    "research_gate_6_in_full_16_urls_recency_scan_gate_passed_true",
    "log_last_discipline_pre_status_flip"
  ]
}
```

PASS. Main may proceed to append `handoff/harness_log.md` (cycle entry
with `result=PASS`) and then flip phase-10.7.2 to `status: done` in
`.claude/masterplan.json`. Strongly recommend addressing FU-10.7.2-C
(orchestrator diff) before starting phase-10.7.3 to avoid masking a
future regression in the multi_agent_orchestrator path.
