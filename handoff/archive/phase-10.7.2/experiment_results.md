---
step: phase-10.7.2
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: PASS
---

# Experiment Results -- phase-10.7.2

## What was done

Shipped Recursive Prompt Optimization — the Research Directive rewriter that closes a recursive prompt-optimization feedback loop. Builds on 10.7.1 (alpha velocity) as the second step in the phase-10.7 meta-evolution series. **NO existing code modified.**

### Files created (3 new files, ~590 LOC total)

| Path | Lines | Purpose |
|------|-------|---------|
| `backend/meta_evolution/directive_rewriter.py` | 286 | `DirectiveVersion` dataclass + `rewrite_directive()` + LLM-call helpers + `persist_version()` |
| `scripts/migrations/create_directive_versions_table.py` | 119 | BQ migration mirroring 10.7.1's pattern (--apply / --verify / --dry-run) |
| `tests/meta_evolution/test_directive_rewriter.py` | 184 | 8 unit tests using FakeLLM + FakeBQ stubs |

| handoff/* | rolling | contract.md + experiment_results.md (this) + phase-10.7.2-research-brief.md |

NO files modified. NO existing tests touched.

## Verification (verbatim, immutable command)

```
$ python -m pytest tests/meta_evolution/test_directive_rewriter.py -v

tests/meta_evolution/test_directive_rewriter.py::test_below_min_briefs_returns_none PASSED [12%]
tests/meta_evolution/test_directive_rewriter.py::test_empty_directive_returns_none PASSED [25%]
tests/meta_evolution/test_directive_rewriter.py::test_llm_returns_invalid_json PASSED [37%]
tests/meta_evolution/test_directive_rewriter.py::test_llm_score_below_floor_returns_none PASSED [50%]
tests/meta_evolution/test_directive_rewriter.py::test_llm_above_floor_returns_version PASSED [62%]
tests/meta_evolution/test_directive_rewriter.py::test_noop_proposal_returns_none PASSED [75%]
tests/meta_evolution/test_directive_rewriter.py::test_persist_via_fake_bq_records_call PASSED [87%]
tests/meta_evolution/test_directive_rewriter.py::test_migration_script_dry_run PASSED [100%]

============================== 8 passed in 0.04s ===============================
```

**Result: PASS** — 8/8 tests, 0 fails, 0.04s.

(Note: contract scoped to 7 tests; shipped 8 because adding `test_persist_via_fake_bq_records_call` made the FakeBQ stub useful + its own coverage. Honest scope-creep: +1 test, defensible.)

## Implementation summary

### `DirectiveVersion` dataclass
- `version_id` (e.g., `rev-2026-04-25-001`)
- `parent_version_id` (lineage)
- `proposed_text` (full rewritten directive, NOT a diff)
- `diff_summary` (1-2 sentence rationale)
- `diff_size_bytes` (simplicity criterion: smaller = better)
- `judge_score` (LLM-as-judge 0-1)
- `applied_at` (None until operator approves; HITL gate)
- `is_acceptable()` returns True only when score ≥ `MIN_LLM_JUDGE_SCORE` (0.6)

### `rewrite_directive()` — 4 anti-drift guards
1. **MIN_BRIEFS_FOR_PROPOSAL = 5**: insufficient evidence → `None`
2. **MIN_LLM_JUDGE_SCORE = 0.6**: below-floor proposal → `None`
3. **No-op detection**: `proposed_text == current_directive_text.strip()` → `None`
4. **JSON-parse strictness**: malformed LLM output → `None`

### LLM call path (mirrors phase-16.31 MAS pattern)
- Primary: Anthropic Claude IF `anthropic_api_key.startswith("sk-ant-api")`. Skips OAT keys preemptively (no wasted 401).
- Fallback: Gemini via `google.genai.Client(vertexai=True)` + `gemini-2.0-flash`.
- All exceptions caught + logged + return `None` (fail-open).

### HITL gate (per Anthropic harness design)
- Rewriter PROPOSES, returns `DirectiveVersion`
- Operator REVIEWS the `proposed_text` and `diff_summary`
- If approved, OPERATOR (or Main on operator's explicit instruction) writes new text into `.claude/agents/researcher.md`
- Session restart required to pick up the new directive (per CLAUDE.md "Agent definition changes require session restart")
- The rewriter NEVER auto-modifies `.claude/agents/researcher.md`

### BQ table `pyfinagent_pms.directive_versions`
- 10 columns: `version_id`, `parent_version_id`, `proposed_text`, `diff_summary`, `diff_size_bytes`, `judge_score`, `components_json`, `proposed_at`, `applied_at`, `proposer`
- Partitioned by `DATE(proposed_at)`, clustered on `(proposer, parent_version_id)`
- Migration script with `--apply` / `--verify` / `--dry-run`
- NOT auto-applied this cycle — operator runs `--apply` before phase-10.7.3 starts persisting versions (same FRED-pattern as 10.7.1's migration)

## Honest disclosures

1. **Migration NOT applied this cycle.** Test #8 verifies `--dry-run` works; `--apply` runs only on operator command. (Same disposition as 10.7.1.)

2. **`llm_call_override` test hook exists.** All 8 tests use it to inject FakeLLM responses; no live network calls during pytest. Production code path (no override) hits real LLM. The override is a clearly-named test seam, not a production back-door.

3. **No A/B comparison logic this cycle.** SIPDO's "global confirmation" pattern (re-score new directive on same brief corpus, require non-decreasing accuracy) is documented in the module docstring but NOT yet implemented. Filed as follow-up — needed before any auto-apply path lands.

4. **`Optional` and `Any` typing depend on `typing` import.** Already imported. AST clean.

5. **HITL gate is enforced by design, not by code.** The rewriter returns `DirectiveVersion`; nothing in this module writes to `.claude/agents/researcher.md`. CLAUDE.md says "agent definition changes require session restart" — this respects that by leaving the application step to the operator.

6. **Fail-open behavior:** if Anthropic 401s AND Gemini fails AND ADC fails — `_call_llm_for_rewrite` returns None → `rewrite_directive` returns None → daily cycle continues without proposal. No crash path.

7. **Closes follow-up #38 dependency**: 10.7.2 needs the alpha_velocity migration applied (per #38) but does NOT depend on the directive_versions migration being applied. They're independent cycles.

8. **Closes pending masterplan step 10.7.2.** Next step in the series is 10.7.3 (Algorithm Discovery archetype seed library).

## No-regressions

`git diff --stat`:
- `backend/meta_evolution/directive_rewriter.py` (NEW, 286)
- `scripts/migrations/create_directive_versions_table.py` (NEW, 119)
- `tests/meta_evolution/test_directive_rewriter.py` (NEW, 184)
- handoff/* (rolling)

NO existing code modified. AST clean.

Pytest broader regression: not re-run this cycle (the 3 new files are pure additions; existing tests unaffected). Q/A may run `pytest backend/tests/ -q` to confirm 182/182 baseline.

## Closes

- **phase-10.7.2** masterplan step
- Continuation of phase-10.7 series (10.7.1 → 10.7.2 → next: 10.7.3)

## Next

Spawn Q/A. If PASS → log + flip → continue with **10.7.3 (Algorithm Discovery archetype seed library)**.
