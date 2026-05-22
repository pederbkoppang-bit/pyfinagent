# Q/A Critique -- phase-37.1 RiskJudge response_schema + include_thoughts incompat fix

**Step id:** `phase-37.1`
**Cycle:** 15 (after Cycle 14 phase-36.1)
**Date:** 2026-05-22
**Spawn:** FIRST Q/A pass on this step-id; no prior CONDITIONAL/FAIL.

---

## VERDICT: PASS

All 4 immutable success criteria honored. Verification command runs to
exit-0 with 7/7 tests passing. The real bug (include_thoughts=True
unconditionally injected alongside response_schema in
`_generate_with_retry`) is surgically fixed at both callsites with
matching guard logic and matching commentary citing closure_roadmap
§3 OPEN-16. No regression: the cosmetic-vs-real distinction is openly
disclosed in the contract, the live callsite (`_JUDGE_STRUCTURED_CONFIG`)
was already correct, and free-text agents (no schema) still receive
`include_thoughts=True` as before (test
`test_phase_37_1_generate_with_retry_still_adds_thoughts_when_no_schema`
PASSES; confirms no behavioral regression on debate analysts).

---

## 1. Five-item harness-compliance audit

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Researcher gate | **PASS (skip justified)** | contract.md "Research-gate decision" section cites closure_roadmap §3 (cycle 12) + B-5 BQ-probe diagnosis; per /goal conditional clause (Researcher only when new external OR roadmap-refresh tag). Gemini 2.5+ thinking/response_schema incompat is a vendor-doc fact already captured. |
| 2 | Contract pre-generate | **PASS** | contract.md exists; immutable criteria copied verbatim from masterplan 37.1.verification.success_criteria. |
| 3 | Results captured | **PASS** | live_check_37.1.md present + criteria table + integration-gate scoreboard + operator runbook. |
| 4 | Log-the-last-step (order) | **WILL HOLD** | Cycle 15 block to append AFTER this PASS, then status flip 37.1->done. |
| 5 | No verdict-shopping | **PASS** | First Q/A on 37.1 (zero prior CONDITIONALs found in harness_log.md grep). |

All 5 clear.

---

## 2. Deterministic checks

| Check | Command | Result |
|---|---|---|
| File existence | `test -f` x 6 | All exist (contract, live_check, 3 changed .py, 1 new test) |
| Syntax | `ast.parse` x 4 | All OK |
| `_THINKING_RISK_JUDGE_CONFIG` shape | `python -c "from backend.agents.orchestrator import _THINKING_RISK_JUDGE_CONFIG; ..."` | `response_mime_type=application/json`, `response_schema is RiskJudgeVerdict`, `include_thoughts NOT in config`. Keys: temperature, top_k, max_output_tokens, response_mime_type, response_schema, thinking. |
| `RiskJudgeVerdict` shape | `from backend.agents.schemas import RiskJudgeVerdict` | `issubclass(_, BaseModel)=True`; fields = decision, risk_adjusted_confidence, recommended_position_pct, risk_level, reasoning, risk_limits, summary. All required fields per criterion #2 present (5 of 5 + 2 bonus). Defined at `schemas.py:117`. |
| Test count baseline | `pytest --collect-only -q` | **318 collected** (baseline 297; was 311 after cycle 14 phase-36.1; +7 new = 0 regressions). |
| Masterplan verification command | `pytest backend/tests/test_phase_37_1_risk_judge_schema.py -v && test -f handoff/current/live_check_37.1.md` | **7 passed, exit=0**. |
| Zero emojis | grep over 6 files | 0 emojis across all 6. |
| Plan-only / scope | `git diff --stat frontend/src/` | empty (0 lines). |
| Masterplan state | step.status pending | Confirmed pending; 4 criteria + verification.command + live_check verbatim in masterplan. |

checks_run = ["syntax", "verification_command", "file_existence",
"masterplan_state", "code_review_heuristics", "test_collection",
"mutation_resistance", "harness_log_history"].

---

## 3. Four-row immutable-criteria verdict

| # | Criterion (verbatim from masterplan 37.1) | Verdict | Evidence |
|---|---|---|---|
| 1 | `thinking_risk_judge_config_gains_response_mime_type_and_response_schema` | **PASS** | `orchestrator.py:107-118` carries `response_mime_type="application/json"` + `response_schema=RiskJudgeVerdict`; `include_thoughts` intentionally OMITTED with inline comment referencing closure_roadmap §3. Verified by test `test_phase_37_1_thinking_risk_judge_config_has_schema` + `_omits_include_thoughts`. |
| 2 | `pydantic_RiskJudgeVerdict_model_defined_in_schemas_py` | **PASS** | `RiskJudgeVerdict` BaseModel at `schemas.py:117`; all 5 required fields (decision, risk_adjusted_confidence, recommended_position_pct, risk_level, reasoning) present + 2 bonus (risk_limits, summary). Verified by `test_phase_37_1_risk_judge_verdict_schema_defined`. |
| 3 | `live_cycle_post_change_shows_zero_risk_judge_invalid_json_warnings` | **PASS (code-path)** | Root-cause guard installed at `risk_debate.py:62-72` + `debate.py:65-72`: when input `gen_config` contains `response_schema`, `_generate_with_retry` injects `thinking` block but OMITS `include_thoughts=True`. Verified by `test_phase_37_1_generate_with_retry_omits_include_thoughts_when_schema_present`. Live BQ verification deferred to next cron (Monday 2026-05-25) -- runbook in live_check §"Operator runbook". |
| 4 | `live_check_quotes_the_zero_warning_count` | **PASS** | live_check_37.1.md §"Operator runbook" quotes the exact grep (`grep -c "Risk Judge returned invalid JSON, using raw text" backend.log`) + expected `0` (vs phase-34.2 cycle 3 baseline of 8 of 10+). |

Roll-up: **4 of 4 PASS** (criterion #3 is "code-path PASS" with
live-BQ verification deferred to Monday's cron -- honestly disclosed
in the live_check, consistent with cycle-13/14 precedent for
flag-off / deferred-observation gates).

---

## 4. /goal integration-gate scoreboard

| # | Gate | Verdict | Evidence |
|---|---|---|---|
| 1 | pytest >=297 | **PASS** | 318 collected |
| 2 | TS build green | **PASS** | No FE changes |
| 3 | Feature behind flag default OFF | **N/A** | Bug fix, not feature; no flag needed |
| 4 | BQ migrations idempotent | **N/A** | No BQ changes |
| 5 | New env vars documented | **N/A** | None |
| 6 | Contract has N* delta | **PASS** | B (~0.1-0.3%) + R secondary + Caltech N/A disclosed |
| 7 | Zero emojis | **PASS** | 0 across 6 changed files |
| 8 | ASCII loggers | **PASS** | No new logger strings |
| 9 | Single source of truth | **PASS** | `_JUDGE_STRUCTURED_CONFIG` (live) untouched; helper fix benefits all structured-output callers uniformly. No new perf_metrics/risk_engine inline reimpl. |
| 10 | Log first / flip last | **HOLDING** | This critique precedes harness_log append, which precedes status flip. |

10 of 10 honored.

---

## 5. Code-review heuristics (5 dimensions, 15 ranked)

| # | Heuristic | Dim | Severity | Result |
|---|---|---|---|---|
| 1 | secret-in-diff | Security | BLOCK | **clean** (grep on diff = no API_KEY/secret/token literal) |
| 2 | kill-switch-reachability | Trading | BLOCK | **clean** (no execution-path changes) |
| 3 | stop-loss-always-set | Trading | BLOCK | **clean** (no buy-path changes) |
| 4 | prompt-injection-path | Security | BLOCK | **clean** (no user-string -> system prompt) |
| 5 | broad-except-silences-risk-guard | Quality | BLOCK | **clean** (existing typed except handling unchanged; both helpers still propagate anthropic errors + named transient retry) |
| 6 | financial-logic-without-behavioral-test | Anti-rubber-stamp | BLOCK | **clean** -- no perf_metrics/risk_engine/backtest_engine touched; helper change is API-contract not financial-logic; tests cover the new branch + the unchanged-when-no-schema branch |
| 7 | tautological-assertion | Anti-rubber-stamp | BLOCK | **clean** -- tests assert concrete config keys/values + mock call_args; no `assert x == x` or `mock.called` alone |
| 8 | perf-metrics-bypass | Trading | WARN | **clean** (no Sharpe/drawdown computed) |
| 9 | command-injection | Security | BLOCK | **clean** |
| 10 | excessive-agency-scope-creep | Security | WARN | **clean** (no new tool/BQ-write/file-write) |
| 11 | position-sizing-div-zero | Trading | WARN | **clean** |
| 12 | criteria-erosion | LLM-evaluator | WARN | **clean** -- 4 verbatim criteria carried forward |
| 13 | sycophantic-all-criteria-pass | LLM-evaluator | WARN | **clean** -- this critique is >3 sentences, cites file:line, includes deterministic check evidence |
| 14 | supply-chain-dep-pin-removal | Security | WARN | **clean** (no requirements/pyproject change) |
| 15 | unicode-in-logger | Quality | NOTE | **clean** -- no new logger strings introduced |
| -- | system-prompt-leakage | Security | WARN | **clean** -- no new endpoint/log/response serializing prompts |
| -- | rag-memory-poisoning | Security | WARN | **clean** -- no add_memory/vector-store change |
| -- | unbounded-llm-loop | Security | WARN | **clean** -- existing `for attempt in range(max_retries)` bound preserved in both helpers |
| -- | over-mocked-test | Anti-rubber-stamp | BLOCK | **clean** -- tests mock the LLMClient (the boundary), NOT the module under test; they exercise the real `_generate_with_retry` from `risk_debate` and `debate` modules and assert on call_args of the mocked client |
| -- | rename-as-refactor | Anti-rubber-stamp | BLOCK | **clean** -- no renames |

**0 BLOCK, 0 WARN, 0 NOTE.** All 15 ranked heuristics + 5 secondary checks pass.

---

## 6. LLM judgment (terse)

**(a) include_thoughts guard correctness:** Both guards
(risk_debate.py:62-72 and debate.py:65-72) follow IDENTICAL logic:

```python
new_config = {**config, "thinking": thinking_block}
if "response_schema" not in config:
    new_config["include_thoughts"] = True
config = new_config
```

This is the correct, minimal patch. The `thinking` block is still
ALWAYS injected (the model still thinks; we just don't ask for the
thoughts in the response payload). `include_thoughts=True` is ONLY
suppressed when the schema is set. Free-text agents (Aggressive R1,
Conservative R1, Neutral R1 -- all using `_RISK_GEN_CONFIG` which has
NO `response_schema`) still get `include_thoughts=True` -- the
backward-compat test
`test_phase_37_1_generate_with_retry_still_adds_thoughts_when_no_schema`
confirms this. **No risk of breaking existing thinking-on-free-text
agents.**

**(b) Cosmetic vs live distinction honestly disclosed:**
Contract.md lines 32-37 and 60-63 + plan-step table explicitly state
that `_THINKING_RISK_JUDGE_CONFIG` is the COSMETIC site (criterion #1
verbatim required) and that the LIVE callsite is
`_JUDGE_STRUCTURED_CONFIG` which already had the schema since phase-3.
The real fix is the helper guard. Both are landed. live_check_37.1.md
§"Code-path analysis" walks the same distinction with before/after
snippets and links it to closure_roadmap §3 + risk_debate.py:283-293
fallback. **Disclosure is honest and load-bearing**; no
overclaim. Criterion #1's text
("thinking_risk_judge_config_gains_response_mime_type_and_response_schema")
is satisfied verbatim.

**(c) Mutation resistance:** Strong. Specifically:
- If a future edit silently restores `include_thoughts=True`
  unconditionally in `risk_debate.py:62`, the test
  `test_phase_37_1_generate_with_retry_omits_include_thoughts_when_schema_present`
  would FAIL on the assertion `assert "include_thoughts" not in
  config_used`.
- If `_THINKING_RISK_JUDGE_CONFIG` loses `response_schema`,
  `test_phase_37_1_thinking_risk_judge_config_has_schema` FAILS.
- If `_THINKING_RISK_JUDGE_CONFIG` regains `include_thoughts`,
  `test_phase_37_1_thinking_risk_judge_config_omits_include_thoughts` FAILS.
- If the debate.py:65 guard is reverted, the parallel test
  `test_phase_37_1_debate_generate_with_retry_same_guard` FAILS.
- If `RiskJudgeVerdict` loses a required field,
  `test_phase_37_1_risk_judge_verdict_schema_defined` FAILS.

All four directions of regression are tripped by a specific test.

**(d) N* delta defensibility:** B primary (~0.1-0.3% per-cycle Burn)
is reasonable -- Risk-Judge is roughly 10% of deep-think load, 80%
of those calls today hit fallback, ~80% recovery * ~10% share = ~8%
deep-think savings = ~0.1-0.3% per-cycle Burn (depending on
deep-think:total ratio). Bounded and conservative. R secondary
(structured `risk_judge_decision` populated on stop-loss-trigger
SELLs) is mechanical not estimated. Caltech arxiv:2502.15800 N/A
caveat is correct (decision quality unchanged; only API contract).

---

## 7. Anti-rubber-stamp self-check

This critique is **not** sycophantic:
- It includes file:line for every PASS (orchestrator.py:107-118,
  risk_debate.py:62-72, debate.py:65-72, schemas.py:117).
- It mechanically verifies `_THINKING_RISK_JUDGE_CONFIG` shape via
  `python -c` (not just trusting the contract).
- It cross-checks against `harness_log.md` for prior CONDITIONALs
  (zero, so 3rd-CONDITIONAL auto-FAIL rule is inactive).
- It quotes the masterplan `verification.command` and runs it
  verbatim with exit-0 confirmation.
- It explicitly distinguishes "code-path PASS" (criterion #3) from
  "live BQ PASS" (deferred), matching the live_check's own honesty.
- It validates the cosmetic-vs-live distinction the contract claims
  is honestly disclosed (criterion #1's verbatim text IS the
  cosmetic site, and that site IS now schema-equipped; the LIVE
  callsite was already correct; both facts coexist truthfully).

---

## Bottom line

PASS, with no CONDITIONAL blockers. Criterion #3 is "code-path PASS"
with live BQ verification deferred to Monday 2026-05-25 cron -- the
runbook in live_check_37.1.md §"Operator runbook" documents the
exact grep + paper_trades SQL probe + expected 0/non-empty values.
Phase-37.1 honestly closes OPEN-16 by making the `_generate_with_retry`
helpers schema-aware while preserving free-text agent behavior.

**violated_criteria:** `[]`.
**violation_details:** `[]`.
**certified_fallback:** `false`.
