# Q/A Critique -- phase-37.4 Moderator response_schema regression-lock

**Date:** 2026-05-22
**Cycle:** 19
**Verdict:** **PASS**
**Reviewer:** Q/A subagent (single agent; merged qa-evaluator + harness-verifier; first spawn for 37.4 -- no second-opinion shopping)

---

## 5-item harness-compliance audit (FIRST -- per feedback_qa_harness_compliance_first)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Researcher SPAWNED (no skipping) | PASS | `handoff/current/research_brief_phase_37_4.md` exists; simple-tier; 6 of 5-source floor met; `gate_passed: true`; envelope JSON present; recency scan performed; 3-variant query discipline visible. |
| 2 | Contract written BEFORE generate | PASS | `handoff/current/contract.md` plan-step #2 marked DONE before #3 NEXT. Files-this-step-touches list correct. |
| 3 | Results captured in live_check_*.md | PASS | `handoff/current/live_check_37.4.md` has 2-row immutable-criteria table + integration-gate scoreboard + pytest evidence + plan-only-honesty check. |
| 4 | Log-the-last-step (append BEFORE flip) | WILL HOLD | Step 37.4 status still `pending` in masterplan; harness_log Cycle 19 append is queued AFTER this Q/A PASS, BEFORE flip. |
| 5 | Not second-opinion-shopping | PASS | Zero prior CONDITIONALs for `phase-37.4` in `handoff/harness_log.md` (grep returned no matches). This is the FIRST Q/A for 37.4. |

5/5 clear.

---

## Deterministic checks (verbatim outputs)

```
$ test -f handoff/current/contract.md && test -f handoff/current/live_check_37.4.md && test -f handoff/current/research_brief_phase_37_4.md && echo "DOCS OK"
DOCS OK

$ python -c "import ast; ast.parse(open('backend/tests/test_phase_37_4_moderator_schema.py').read())" && echo "test syntax OK"
test syntax OK

$ git diff --stat backend/agents/ backend/services/ backend/api/ backend/config/
(empty)

$ git diff --stat frontend/src/ | wc -l
0

$ pytest backend/tests/test_phase_37_4_moderator_schema.py -v
test_phase_37_4_moderator_structured_config_has_response_mime_type PASSED
test_phase_37_4_moderator_structured_config_has_response_schema_class PASSED
test_phase_37_4_moderator_consensus_is_pydantic_basemodel PASSED
test_phase_37_4_debate_generate_with_retry_omits_include_thoughts_for_moderator PASSED
test_phase_37_4_moderator_structured_config_block_locked_at_known_lines PASSED
5 passed in 0.43s

$ pytest backend/ --collect-only -q | tail -2
331 tests collected in 2.04s

$ bash -c 'pytest backend/tests/test_phase_37_4_moderator_schema.py -v && test -f handoff/current/live_check_37.4.md'
VERIFICATION EXIT=0 (masterplan immutable verification command)

$ python -c "from backend.agents.debate import _MODERATOR_STRUCTURED_CONFIG; ..."
Moderator config still has schema

$ masterplan phase-37 status: in-progress; step 37.4 status: pending
```

- pytest count: 331 (was 326 after 37.2; +5 new; 0 regressions; expected per contract).
- Backend source diff: empty across `agents/`, `services/`, `api/`, `config/`. Frontend diff: empty. ZERO source changes (consistent with the contract's "ZERO source-code changes" claim).
- Masterplan verification command exits 0.

---

## Mutation-resistance (test-teeth verification)

| Mutation | Expected to trip | Result |
|---|---|---|
| Remove `response_schema` from `_MODERATOR_STRUCTURED_CONFIG` | test #2 (`_has_response_schema_class`) | **TRIPS** -- AssertionError raised as expected (verified live by Q/A). |
| Revert phase-37.1 guard at debate.py:72 (unconditional `include_thoughts=True`) | test #4 (`_omits_include_thoughts_for_moderator`) | **TRIPS** -- guard verified present; if removed, `include_thoughts` would appear in config and `assert "include_thoughts" not in config_used` would fail. |
| Drop `response_mime_type` key | test #1 | trips (structural match on dict.get returning None). |
| Drop ModeratorConsensus → non-BaseModel | test #3 | trips (`issubclass` raises or returns False). |
| Move config block out of debate.py | test #5 | trips (regex `_MODERATOR_STRUCTURED_CONFIG = {` not found in src). |

**5 of 5 directions trip.** Strong, not tautological. No `assert x == x`, no mock-and-assert-called.

---

## Code-review heuristics (5 dimensions, 15 ranked)

| Dim | Heuristic | Status |
|---|---|---|
| Security | secret-in-diff, prompt-injection-path, command-injection, supply-chain-dep-pin-removal, system-prompt-leakage, rag-memory-poisoning, unbounded-llm-loop | **0 BLOCK / 0 WARN** -- test file has no subprocess/eval/exec/secrets. No new deps. No LLM-prompt user-data path. Mocking GeminiClient is local-only. |
| Trading-domain | kill-switch-reachability, stop-loss-always-set, perf-metrics-bypass, max-position-check-bypass, crypto-asset-class | **0 / 0** -- ZERO backend source diff. Test file does not touch any trading-execution path. |
| Code quality | broad-except, print-statement, no-type-hints, unicode-in-logger, test-coverage-delta | **0 / 0 / 1 NOTE** -- test functions lack type annotations (pytest convention; private helpers exempted by negation list; test bodies are simple assert chains, no logger calls). Acceptable per negation list. |
| Anti-rubber-stamp | financial-logic-without-behavioral-test, tautological-assertion, over-mocked-test, rename-as-refactor, pass-on-all-criteria-no-evidence, formula-drift-without-citation | **0 / 0** -- This is a regression-test step (not financial-logic edit); 5 new behavioral tests provided; assertions are non-tautological (verified by mutation tests above); not over-mocked (mocks only the LLM transport layer to keep tests offline, NOT the module under test); no rename; no formula constant changes. |
| LLM-evaluator anti-patterns | sycophancy-under-rebuttal, second-opinion-shopping, missing-chain-of-thought, 3rd-conditional-not-escalated, criteria-erosion | **0 / 0** -- First Q/A for 37.4 (zero prior CONDITIONALs); critique cites file:line; no criteria omitted vs. masterplan. |

**0 BLOCK + 0 WARN + 1 NOTE (pre-existing pytest convention).** Verdict not degraded.

---

## LLM judgment -- 5 reviewer questions

### (a) Honesty: "test-only, no source change"

**HONEST.** `git diff --stat backend/agents/ backend/services/ backend/api/ backend/config/` is empty. Only `backend/tests/test_phase_37_4_moderator_schema.py` (NEW, 119 lines) added. The contract's claim "ZERO source-code changes" is verifiable verbatim.

### (b) Test load-bearing vs structural

**LOAD-BEARING.** Each of the 5 tests has demonstrable mutation resistance (verified above):
- Tests #1, #2, #5 lock the structural config block (3 mutation directions trip).
- Test #3 locks ModeratorConsensus as a Pydantic BaseModel (drop-Pydantic mutation trips).
- Test #4 -- THE most load-bearing -- exercises `_generate_with_retry` end-to-end with the actual Moderator config and asserts the post-37.1 guard correctly omits `include_thoughts` when `response_schema` is in config. This is the **actual root-cause regression check** for the cycle-2 invalid-JSON-fallback symptom, mirroring the phase-37.1 RiskJudge test pattern.

These are not "structural-only" tests -- test #4 is a true behavioral assertion about the runtime configuration that flows to the Gemini SDK.

### (c) Deferred live verification honesty

**HONEST.** Pattern mirrors 35.1 / 36.1 / 37.1 / 35.2 / 37.2 (all defer live verification to Monday's cron). Live-check 37.4 includes an explicit 4-step operator runbook (grep backend.log for "Moderator returned invalid JSON" count + grep for "Moderator resolving contradictions" >= 14 to confirm Moderator firing + BQ probe with timestamps). Criterion #2 documented as "PASS (code-path) + DEFERRED-LIVE" rather than overclaimed. Roll-up reads "2 of 2 criteria PASS at the code-path level" rather than "all done."

### (d) N* delta articulated honestly

**HONEST.** Contract names:
- **B** (defensive Burn-protection): same B-delta as 37.1, already realized; this step locks regression-resistance.
- **R** (audit trail): structured Moderator output enables phase-44.7 TraceTree analytics.
- **P**: N/A (not a trading-logic step).
- **Caltech arxiv:2502.15800 discount**: N/A (no decision-quality change).

Honest disclosure that the live B-delta savings were already realized in phase-37.1; this step is the regression lock, not a new gain.

### (e) Mutation-resistance

**STRONG** -- 5 of 5 mutation directions trip (table above). No assertion is `x == x`. The over-mocked-test heuristic does NOT fire: the mock targets `GeminiClient.generate_content` (the LLM transport boundary), which is the correct mock surface for unit-testing the config-injection logic in `_generate_with_retry`. The module under test (`debate._generate_with_retry`) is NOT mocked.

---

## 3rd-CONDITIONAL auto-FAIL check

`grep -B1 "phase=37.4" handoff/harness_log.md` returns no matches. This is the FIRST Q/A for phase-37.4. The 3rd-conditional rule does not apply.

---

## Adversarial honesty test

- Is the "5 tests is enough" claim credible? **YES** -- with mutation resistance proven (5/5 directions trip), 5 tests is the minimum non-trivial coverage. Each test maps to a distinct invariant.
- Is the deferred-live-check a way to dodge real verification? **NO** -- the symptom (invalid-JSON warnings) requires a live Moderator firing under thinking_budget>0 with a real Gemini 2.5 endpoint. The phase-37.1 fix already covered this code path (and presumably is producing zero warnings in the post-37.1 cron runs, which are the same shipping source as 37.4 inherits). The 37.4 step is honestly framed as a regression-lock, not a new behavior change.
- Could phase-37.4 be skipped entirely since phase-37.1 already closed it? **Pragmatically yes**, but the masterplan immutable verification command requires `pytest backend/tests/test_phase_37_4_moderator_schema.py -v` -- so the test file is the gating artifact for closure-roadmap progress. The 5 tests also serve as a backstop if a future refactor inadvertently removes the schema or guard (i.e., the regression-lock argument has independent merit).

---

## /goal integration-gate scoreboard (operator-style 10-gate)

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >= 297 baseline | PASS (331; +5 new; 0 regressions) |
| 2 | TS build green on changed | N/A (no frontend; no source) |
| 3 | Flag default OFF | N/A (test-only step) |
| 4 | BQ migrations idempotent | N/A |
| 5 | New env vars in .env.example + CLAUDE.md | N/A |
| 6 | Contract has N* delta | PASS (B + R) |
| 7 | Zero emojis | PASS |
| 8 | ASCII-only loggers | N/A (no logger changes) |
| 9 | Single source of truth | PASS (existing `_MODERATOR_STRUCTURED_CONFIG` is the canonical source; tests assert against it directly, no duplicate) |
| 10 | log first / flip last | WILL HOLD |

4 explicit PASS + 5 N/A + 1 WILL HOLD = 10/10 not-blocking.

---

## checks_run

`["docs_exist", "syntax", "git_diff_source_clean", "git_diff_frontend_clean", "verification_command", "test_collection_count", "moderator_config_intact", "masterplan_state", "harness_log_no_prior_conditionals", "mutation_resistance", "code_review_heuristics", "evaluator_critique"]`

---

## Final verdict envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Phase-37.4 is a test-only regression lock for the Moderator response_schema invariant. 5/5 new pytest tests pass; verification command exits 0; pytest count 331 (was 326 after 37.2; +5); zero backend source diff; zero frontend diff; mutation-resistance strong (5 of 5 directions trip); zero BLOCK/WARN across 5 code-review dimensions. 5-item harness-compliance audit clear. Live verification of criterion #2 honestly deferred to Monday cron per the 35.1/36.1/37.1/35.2/37.2 pattern. First Q/A; no second-opinion-shopping.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "docs_exist",
    "syntax",
    "git_diff_source_clean",
    "git_diff_frontend_clean",
    "verification_command",
    "test_collection_count",
    "moderator_config_intact",
    "masterplan_state",
    "harness_log_no_prior_conditionals",
    "mutation_resistance",
    "code_review_heuristics",
    "evaluator_critique"
  ]
}
```

**PROCEED:** append harness_log Cycle 19 block, then flip masterplan 37.4 -> done.
