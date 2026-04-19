# phase-11.2 Q/A Critique — qa_id=qa_112_v1

**Step:** phase-11.2 — Migrate trivial Vertex callers (evaluator_agent, skill_optimizer, debate) to google-genai.
**Cycle:** 1.
**Verdict:** **PASS** (0 violated_criteria).

## 5-item protocol audit

1. **Research gate**: `handoff/current/phase-11.2-research-brief.md` present (mtime 14:53:28, 12.2 KB). Assumed gate_passed=true per brief contents; three-query discipline applied.
2. **Contract PRE-commit**: `phase-11.2-contract.md` mtime **14:54:22** < all 4 changed files (evaluator_agent 14:54:53, test 14:55:02, skill_optimizer 14:55:46, debate 14:56:12). PASS.
3. **Experiment results present**: `phase-11.2-experiment-results.md` (14:57:44, 7.4 KB). PASS.
4. **Log-last discipline**: `handoff/harness_log.md` last entry is "Cycle N+50 … phase=11.1 result=PASS". No 11.2 entry yet. PASS.
5. **Cycle-1, no verdict-shopping.** PASS.

## Deterministic (A-I)

- **A. Immutable verify**: `pytest backend/tests/test_evaluator_agent.py -q` → `6 passed`. `python -W error::DeprecationWarning -c "from backend.agents import evaluator_agent"` → EXIT=0. **Vertex DeprecationWarning absent.** (Only a pytest-time `_UnionGenericAlias` warning from google/genai/types.py fires during test collection — NOT during the bare import path, and it is Python-3.17-stdlib-deprecation, not Vertex.) PASS.
- **B. Syntax**: all 4 files ast-parse clean. PASS.
- **C. Live `from vertexai.generative_models` imports**: zero. Only match is `backend/agents/debate.py:18` — the expected "removed dead import" comment. PASS.
- **D. `GENAI_AVAILABLE` / `VERTEX_AVAILABLE`**: 5 GENAI hits (module + tests), zero VERTEX. PASS.
- **E. New API surface**: `get_genai_client` + `client.models.generate_content` present in both `evaluator_agent.py` (L43, L97, L286) and `skill_optimizer.py` (L107-108, L366, L546). PASS.
- **F. `pytest -x -q`**: 6 passed. PASS.
- **G. Cumulative regression**: `pytest backend/tests/ -q --ignore=test_paper_trading_v2.py` → **79 passed, 1 skipped**. Matches contract. PASS.
- **H. Scope check**: `git diff --name-only` shows orchestrator.py and risk_debate.py modified. **Cross-referenced against harness_log cycle N+50**: these are declared pre-existing session-dirt from phase-11.1, explicitly flagged for separate-commit handling. Not introduced by this cycle. **Non-blocking** — but Main MUST stage only the 4 in-scope files + 4 handoff artifacts when committing 11.2. Recording as cycle-close requirement, not a violation.
- **I. risk_debate / orchestrator diff-stat non-empty**: diff is pre-existing, not from this cycle (see H). Accepted.

## LLM judgment

- **Inventory discrepancy (5 vs 3)**: accepted as **Known-Caveat disclosure**. Original phase-11.0 broader grep counted `GenerativeModel(...)` usages + `vertexai.init`; this cycle's narrow grep is imports-only. The migration is complete (zero live vertex imports in the 3 target files); the contract's predicted count was off by a grep-scope mismatch, not a missed file. Not CONDITIONAL.
- **`_get_model` signature change**: grepped `_get_model(` across `backend/` — 3 hits, all inside `skill_optimizer.py` (L98 def, L359, L539 internal callers). Zero external callers. Safe.
- **debate.py:18 comment**: pure style annotation; survives grep-C only because the comment quotes the removed import verbatim. Not a real issue.
- **Fail-open paths**: `evaluator_agent.py:98` + `:280` guard `self.model is None` → mock path. `skill_optimizer.py:360` + `:540` guard `client is None` → `return None`. All 3 no-crash paths verified at source.
- **DeprecationWarning gate**: re-ran `python -W error::DeprecationWarning -c "from backend.agents import evaluator_agent"` → EXIT=0. Clean.

## Violated criteria

None.

## Violation details

None.

## certified_fallback

False.

## checks_run

`["protocol_audit_5", "syntax_4files", "immutable_verify_tests", "immutable_verify_import_dep_error", "grep_vertex_imports", "grep_genai_flags", "grep_new_api", "grep_get_model_callers", "pytest_regression_79", "fail_open_source_inspection", "scope_diff_crosscheck", "contract_pre_commit_mtime"]`

## Close-out note to Main

Stage-selectively on commit: include only `backend/agents/{evaluator_agent,skill_optimizer,debate}.py`, `backend/tests/test_evaluator_agent.py`, and `handoff/current/phase-11.2-*.md`. Do NOT sweep in `orchestrator.py` / `risk_debate.py` — those are phase-11.3+ scope and still carry the pre-existing dirt from cycle N+50.
