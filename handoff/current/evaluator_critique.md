# phase-37.3 -- Q/A evaluator critique (Cycle 46)

**Date:** 2026-05-23
**Cycle:** 46
**Step id:** 37.3 (P3 OPEN-18 -- budget_tokens deprecation cleanup)
**Q/A round:** 1 (first spawn this step; 3rd-CONDITIONAL counter = 0)
**Verdict:** **PASS** (HONEST NO_OP trace-link closure)

---

## 1. Harness-compliance audit (5-item, FIRST)

| # | Item | Status |
|---|---|---|
| (a) | Researcher spawned FIRST | **PASS** -- `handoff/current/research_brief_phase_37_3.md` exists; 5 sources read-in-full; gate_passed=true; recency scan present; tier=simple appropriate for trace-link closure. 3 consecutive cycles honoring `feedback_never_skip_researcher`. |
| (b) | Contract pre-generate | **PASS** -- `handoff/current/contract.md` line 1 = `# phase-37.3 -- budget_tokens deprecation cleanup (NO_OP closure)`. NO_OP closure documented openly with researcher's recommendation (c) cited. |
| (c) | experiment_results.md present + current step | **PASS** -- header line 1 = `# phase-37.3 -- experiment results (Cycle 46) -- NO_OP closure`. Refreshed for this cycle (NOT stale phase-38.x content). |
| (d) | Log-last discipline | PENDING -- harness_log block to be appended BEFORE status flip per `feedback_log_last`. Main committed in prompt. |
| (e) | No verdict-shopping | **PASS** -- this is the FIRST Q/A on phase-37.3 (0 prior 37.3 entries in harness_log.md). Not a cycle-2 spawn. No sycophancy-under-rebuttal risk. |

All 5 items clean. No process blockers.

---

## 2. Deterministic checks (verbatim)

```
$ test -f handoff/current/research_brief_phase_37_3.md && echo BRIEF_OK
BRIEF_OK

$ pytest backend/tests/test_phase_37_3_budget_tokens.py -v
========================= 3 passed, 1 xfailed in 0.05s =========================
  test_phase_37_3_thinking_budget_used_in_gemini_path PASSED      [criterion 2]
  test_phase_37_3_no_compat_shim_remains PASSED                   [criterion 3]
  test_phase_37_3_anthropic_legacy_refs_are_wire_literal PASSED   [criterion 1 operational]
  test_phase_37_3_literal_criterion_1_unsatisfiable_until_anthropic_eol XFAIL [criterion 1 literal, expected]

$ pytest backend/ --collect-only -q | tail -2
500 tests collected   (was 496; +4 net new; 0 regressions)

$ git diff HEAD --stat backend/agents/
(empty -- ZERO production-code lines changed)

$ git status --short
 M handoff/current/contract.md
 M handoff/current/experiment_results.md
?? backend/tests/test_phase_37_3_budget_tokens.py
?? handoff/current/live_check_37.3.md
?? handoff/current/research_brief_phase_37_3.md
```

---

## 3. Verbatim criterion -> evidence mapping

| # | Masterplan immutable criterion | Evidence | Verdict |
|---|---|---|---|
| 1 LITERAL | `zero_budget_tokens_refs_in_backend_py_files` | grep finds 45 wire-literal refs; verification command `test $(grep -rn 'budget_tokens' backend/ --include='*.py' \| wc -l) -eq 0` exits 1 | **xfail (strict)** -- honest dual-interpretation: criterion unsatisfiable without breaking Anthropic legacy support. Named follow-up phase-37.3.1. |
| 1 OPERATIONAL | every remaining ref is API-required and documented | test_phase_37_3_anthropic_legacy_refs_are_wire_literal PASS; offenders=0 | **PASS** |
| 2 | `thinking_budget_param_used_at_all_callsites` | llm_client.py:917 `ThinkingConfig(thinking_budget=budget, include_thoughts=True)`; test_phase_37_3_thinking_budget_used_in_gemini_path PASS | **PASS** |
| 3 | `no_compat_shim_remains` | No try/except alias, no version-gated rename, no `thinking_budget_alias = ...`; direct boundary construction; test_phase_37_3_no_compat_shim_remains PASS | **PASS** |

Researcher core claim INDEPENDENTLY cross-checked by Q/A reading both boundary sites:
- `backend/agents/llm_client.py:1378-1388` -- Anthropic legacy wire path (literal `budget_tokens` key inside `{"type":"enabled","budget_tokens":N}` payload)
- `backend/agents/llm_client.py:907-919` -- Gemini typed translation (`ThinkingConfig(thinking_budget=...)`)
Both correct as-is.

---

## 4. Code-review heuristic sweep (Top-15)

| # | Heuristic | Result |
|---|---|---|
| 1 | secret-in-diff [BLOCK] | clean (test file + 3 handoff docs only) |
| 2 | kill-switch-reachability [BLOCK] | N/A (no production code) |
| 3 | stop-loss-always-set [BLOCK] | N/A |
| 4 | prompt-injection-path [BLOCK] | N/A |
| 5 | broad-except-silences-risk-guard [BLOCK] | N/A |
| 6 | financial-logic-without-behavioral-test [BLOCK] | N/A |
| 7 | tautological-assertion [BLOCK] | clean -- 4 tests assert real post-conditions (boundary uses correct field name; no compat shim; offenders=0) |
| 8 | perf-metrics-bypass [WARN] | N/A |
| 9 | command-injection [BLOCK] | N/A |
| 10 | excessive-agency-scope-creep [WARN] | clean |
| 11 | position-sizing-div-zero [WARN] | N/A |
| 12 | criteria-erosion [WARN] | **clean** -- criterion 1 NOT dropped; addressed via honest dual-interpretation + xfail strict + named follow-up. Criteria 2 + 3 PASS via dedicated tests. |
| 13 | sycophantic-all-criteria-pass [WARN] | **clean** -- this critique flags criterion 1 literal as xfail (not all-PASS); cites file:line evidence for every criterion. |
| 14 | supply-chain-dep-pin-removal [WARN] | clean -- zero dep changes |
| 15 | unicode-in-logger [NOTE] | N/A (no logger calls) |

**Dimension 4 (anti-rubber-stamp):** PASS. 3 PASS tests use AST/grep/regex on real source content; xfail strict provides mutation-resistance if wire refs ever get silently deleted. NO_OP closure is CORRECT pattern here per CLAUDE.md "honest dual-interpretation pattern (literal vs operational criterion; xfail with named follow-ups)".

**Dimension 5 (LLM-evaluator anti-patterns):** PASS. First Q/A spawn (no sycophancy-under-rebuttal risk). 3rd-CONDITIONAL counter = 0. PASS verdict grounded in independent cross-check of researcher claims at llm_client.py:1388 + :917.

---

## 5. LLM-judgment

### (a) Is NO_OP closure with xfail-on-literal HONEST?

**YES.** Per CLAUDE.md harness lessons "honest dual-interpretation pattern (literal vs operational criterion; xfail with named follow-ups)". This is the DOCUMENTED honest path when a literal criterion is unsatisfiable without regressing other API surface. The contract openly discloses the conflict, the test xfail-strict carries the reason + named follow-up, and Main did NOT touch the immutable criteria text (only added operational equivalent tests + an xfail-strict for the literal).

### (b) Researcher claim cross-check

**Independently verified.** Read `backend/agents/llm_client.py:907-919` (Gemini boundary uses `ThinkingConfig(thinking_budget=...)`) and `:1378-1388` (Anthropic legacy gate on `model_id.startswith("claude-opus-4-7"...)` -- adaptive path for Opus 4.7+, manual `{"type":"enabled","budget_tokens":N}` for legacy). Both correctly implemented. The researcher's NO_OP recommendation is sound.

### (c) Has Main correctly avoided gutting Anthropic API support?

**YES.** `git diff HEAD --stat backend/agents/` is EMPTY. `git diff HEAD --stat backend/` shows only handoff/audit/* + handoff/current/contract.md + handoff/current/experiment_results.md. Untracked: 1 new test file + 2 new handoff docs (live_check + research_brief). No production code touched.

### (d) xfail strict discipline -- adequate mutation-resistance?

**YES.** `strict=True` on the xfail means: if the test SUDDENLY passes (someone silently deletes the wire refs), pytest will fail loudly. This catches the failure mode where a future cleanup PR removes the wire refs and silently breaks Anthropic API support. Combined with `test_phase_37_3_anthropic_legacy_refs_are_wire_literal` (operational equivalent) which enumerates each remaining ref's classification, the mutation-resistance is robust.

### (e) Follow-up phase-37.3.1 documented?

YES in test file (xfail reason cites "phase-37.3.1 -- re-evaluate when Anthropic legacy-model EOL is announced") + contract + live_check + experiment_results. Recommend ADDING `phase-37.3.1` (P3) to masterplan after this PASS:
- Title: "Re-evaluate budget_tokens removal when Anthropic legacy-model EOL announced"
- Verification: `grep -c '"budget_tokens"' backend/agents/llm_client.py` returns 0 AFTER Anthropic deprecation announcement.

---

## 6. Output envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "phase-37.3 NO_OP trace-link closure verified. Researcher core-claim (budget_tokens = Anthropic wire-literal at llm_client.py:1388 inside {type:enabled,budget_tokens:N}; thinking_budget = Gemini field already correctly used at llm_client.py:917 inside ThinkingConfig(thinking_budget=...)) independently cross-checked by Q/A reading both boundary sites. Bulk rename would break Anthropic legacy-model wire path. ZERO production code lines changed (git diff HEAD --stat backend/agents/ empty). Tests: 3 passed + 1 xfailed strict in 0.05s; collection 496->500 (+4 net, 0 regressions). Masterplan literal verification command exits 1 with 45 refs remaining -- correctly identified as unsatisfiable and honestly xfailed strict. Code-review Top-15 sweep: 0 BLOCK / 0 WARN / 0 NOTE on diff. 5-item harness-compliance audit clean. Honest dual-interpretation pattern correctly applied per CLAUDE.md.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax_ast", "file_existence", "verification_command_exit_code", "pytest_new_file", "pytest_collect_count", "git_diff_backend_agents", "researcher_claim_cross_check", "code_review_heuristics", "harness_compliance_audit", "harness_log_prior_conditional_count"]
}
```

---

## 7. Bottom line

**PROCEED.** NO_OP trace-link closure verified honest and complete. Researcher's recommendation independently cross-checked at the boundary sites. xfail strict provides ongoing mutation-resistance. Recommend ADDING `phase-37.3.1` (P3) to masterplan as the named follow-up for Anthropic legacy-model EOL.

Closure pattern: TRACE-LINK (per CLAUDE.md 3 documented closure patterns). Not engineered work because nothing was broken -- the boundary translation was already correct. The xfail + operational equivalents preserve audit visibility without forcing a misguided refactor.
