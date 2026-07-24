# Evaluator critique — Step 75.5.12 (P1)

Cycle 159 | 2026-07-25 | Q/A launch: `.claude/workflows/qa-verdict.js` (Workflow
structured-output, model opus / effort max) | **Cycle-1 verdict: CONDITIONAL**

Main records this verdict; Main did NOT author it. Transcribed VERBATIM below.

## Verdict (verbatim)

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 3 immutable criteria are MET and were reproduced independently by Q/A (not taken from Main): the immutable command exits 0 with 13 passed; an in-memory mutation matrix run by Q/A gives M1-revert -> exactly one RED (test_bare_cc_rail_shape_contributes_zero) with test_cc_rail_rows_contribute_zero_both_shapes GREEN (criterion 3 exactly), M2-prefix -> RED on the cc_railway over-match test (criterion 1's \"not a prefix-wildcard\" half), M4-no-null-guard -> RED on test_agent_none_rows_are_included; the 3-shape inventory was re-derived from the writers by Q/A and no fourth shape exists. Harness compliance is 5/5 and no unintended production change occurred (flag verified False at RUNTIME). CONDITIONAL is issued for ONE prose defect, not a code defect: the justification Main gives for declining the masterplan step's suggested predicate -- \"adopting the step's wording would SILENTLY NEUTER the existing shape-2 guard; the test would keep passing while no longer testing anything\" -- is refuted by execution. Q/A applied that exact form and got 2 RED, including the very both-shapes test claimed to stay green. The mechanism half of the claim is true (the substring the fake keys on at test:85 drops out), but the consequence is inverted: the branch not firing means the rail row is INCLUDED, so the assertion FAILS loudly. This unreproduced behavioral prediction is stated as fact in all three artifacts (contract finding #1, experiment_results section 1, live_check section 3) and is the sole stated basis for deviating from an explicit instruction on a P1 money step. Fix is documentation-only in those three files; no code change is required, and the decision it justified was in fact CORRECT for a stronger reason Main did not give (see notes).",
  "violated_criteria": [
    "claim-auditing (qa.md 4b): load-bearing justification claim does not reproduce [WARN]"
  ],
  "violation_details": [
    {
      "violation_type": "Contradiction",
      "action": "Q/A applied the masterplan step's suggested predicate form (AND NOT (agent = 'cc_rail' OR agent LIKE 'cc_rail:%')) to spend.py in memory and ran backend/tests/test_phase_75_5_1_spend_metric.py",
      "state": "Observed: '2 failed, 11 passed'; RED = ['test_bare_cc_rail_shape_contributes_zero', 'test_cc_rail_rows_contribute_zero_both_shapes']. Claimed in contract_75.5.12.md finding #1, experiment_results_75.5.12.md section 1, and live_check_75.5.12.md section 3: 'Adopting the step's wording would silently neuter the existing shape-2 guard -- the test would keep passing while no longer testing anything.' The test does NOT keep passing; it goes RED. The fake's branch at test_phase_75_5_1_spend_metric.py:85 not firing means the cc_rail:synthesis row is INCLUDED in the aggregate, so the assertion fails loudly. Main never ran this counterfactual; it is an unverified behavioral prediction asserted as fact. SEVERITY: WARN -- caps the verdict at CONDITIONAL, no immutable criterion is violated.",
      "constraint": "qa.md 4b claim-auditing: every behavioral/quantified claim in the handoff must carry, or be able to re-derive, the exact command that produces it; a claim whose output does not reproduce is a Contradiction finding."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope (8 read-in-full >= 5, recency scan, gate_passed true)",
    "mtime_ordering (research 00:32:41 < contract 00:33:58 < code 00:36:13/14 < results 00:37:13)",
    "log_last (0 harness_log entries for 75.5.12; masterplan status still pending)",
    "no_verdict_shopping (no evaluator_critique_75.5.12.md; first Q/A on this step)",
    "immutable_verification_command (13 passed, true exit=0)",
    "ruff_lint_gate F821/F401/F811 on git-derived scope (3 files, non-empty asserted, exit=0)",
    "backend_runtime_smoke (module import OK; flag False at runtime)",
    "scoped_pytest (-k spend/observability/cost_budget/cost_truth: 80 passed)",
    "independent_mutation_matrix_M1_M2_M4_in_memory",
    "fixture_mutation_matrix_F1_F2_F3_F4 (evaluator mutated the FAKE, qa.md 4c)",
    "step_text_counterfactual_mutation (MSTEP)",
    "sql_3vl_oracle (sqlite ANSI NULL semantics)",
    "predicate_truth_table_5_agent_values",
    "shape_set_re_derivation (all log_llm_call writers, whole repo)",
    "git_scope_isolation (in-scope vs 76.9.3/78.0 parallel work)",
    "immutable_criteria_unchanged_vs_HEAD",
    "flag_default_unchanged",
    "code_review_heuristics",
    "claim_auditing_4b",
    "guard_vacuity_4c",
    "adversarial_worst_of_N_lenses"
  ],
  "harness_compliance_ok": true,
  "notes": "CRITERIA (all independently reproduced by Q/A, not accepted from Main):\n- C1 MET. spend.py:229-230 uses exact `agent != 'cc_rail'`, not a prefix. Q/A mutation M2 (swap to `NOT LIKE 'cc_rail%'`) -> RED on test_cc_railway_is_not_swallowed_by_the_bare_cc_rail_exclusion. New bare-shape fixture present.\n- C2 MET. Q/A re-derived the shape-set from `grep -rn --include='*.py' \"log_llm_call(\" backend scripts`: only TWO flat-fee writers exist -- claude_code_client.py:501-504 (shapes b/c via the ternary) and autonomous_loop.py:2298-2299 (shape a, the only provider='claude-code' writer). Every other writer (llm_client.py:1124/1277/1894/2322/2334, orchestrator.py:908) is a metered SDK path. scripts/ contains ZERO llm_call_log writers. No fourth shape. Docstring names all three with anchors.\n- C3 MET. M1 revert -> exactly one RED (bare-shape test); test_cc_rail_rows_contribute_zero_both_shapes stayed GREEN. That existing test is byte-untouched: `git diff -U0` on the test file shows 0 deleted lines.\n\n(a) THE PREDICATE -- deviation is not merely acceptable, it is CORRECT, and Main understated its own case. Q/A truth table over {NULL,'cc_rail','cc_rail:synthesis','cc_railway','synthesis'}: the two forms agree on all four non-NULL values but NOT on NULL. Taken literally without re-adding an IS NULL guard, the step-text form DROPS NULL-agent rows (3VL: NULL != 'cc_rail' -> NULL -> WHERE requires TRUE). NULL agents are the COMMON metered case (llm_client.py:1124 passes agent=generation_config.get(\"_role\"), and orchestrator.py:826-835 sets only _ticker, never _role), so the step text as written would have silently UNDER-counted real metered Gemini spend -- a false negative that opens the breaker late. Main's form is strictly safer. Main should have cited this, plus \"keeps the existing fake branch live\", instead of the refuted silent-neuter story.\n\n(c) THE M4 / THREE-VALUED-LOGIC BRANCH IS FAITHFUL, NOT TUNED. Three independent proofs: (i) sqlite ANSI oracle -- NULL row kept without the IS-NULL guard = 0, with guard = 1, exactly the BigQuery semantics Main describes; (ii) Q/A run F4 -- removing the new branch while production SQL is CORRECT changes NOTHING (all 4 target tests green), so the branch is inert on the correct path and only adds discriminating power under mutation, which is the signature of a fidelity fix rather than overfitting; (iii) Q/A run F3 reproduces Main's disclosed pre-fix state exactly (M4 did not kill; all green). The guard is materially load-bearing money logic. Main's disclosure of the initial M4 miss is honest and reproduces.\n\n(d) FIXTURE HONESTY -- the prefix branch is LEGITIMATE, stated plainly. Q/A run F1 is decisive: with that branch removed, M2 leaves test_cc_railway_is_not_swallowed_by_the_bare_cc_rail_exclusion GREEN. Without it, criterion 1's over-match guard is VACUOUS for the exact mutation it exists to catch (qa.md 4c shape #5, fixture that cannot represent the failure). The branch is dormant on correct SQL -- verified that \"NOT LIKE 'cc_rail%'\" is not a substring of \"NOT LIKE 'cc_rail:%'\" -- so it cannot mask a production bug. Not coverage inflation.\n\n(e) CONFIRMED via git diff -U0: zero deleted/changed lines in the test file; test_cc_rail_rows_contribute_zero_both_shapes is untouched, so criterion 3's \"stay green\" is a genuinely independent signal.\n\n(f) SCOPE CLEAN. In-scope changes are exactly spend.py + its test. test_phase_75_deps.py and scripts/autoresearch/requirements-autoresearch.txt belong to 76.9.3; the .claude/masterplan.json diff is unicode-escape churn plus 78.0/76.9.3 queue edits -- 75.5.12's success_criteria are byte-identical to HEAD and status is still `pending`. Audit JSONLs are hook-appended. cost_budget_use_llm_spend_enabled verified False at RUNTIME (not just in source), closing vacuity shape #9.\n\nREQUIRED TO CLEAR (documentation only, no code change):\n1. Correct the silent-neuter sentence in contract_75.5.12.md finding #1, experiment_results_75.5.12.md section 1, and live_check_75.5.12.md section 3 to the reproduced result: the step-text form yields 2 RED (including the both-shapes test) -- a loud failure, not a silent pass. The real hazard is narrower: the fake's :85 branch goes DEAD, so a naive repair could weaken the guard.\n2. Add the stronger, reproduced reason: the two forms are NOT equivalent for NULL, and the step text taken literally would drop NULL-agent metered rows and under-count spend. Include the truth table.\n\nNOTES (non-blocking, do not affect this verdict):\n- QUEUE-WORTHY ADJACENT DEFECT (per feedback_queue_discovered_defects_in_masterplan): phase-76.9.2's anthropic_max_bridge (127.0.0.1:18797) routes flat-fee Max traffic over the Anthropic Messages protocol. No llm_call_log writer is wired to it today, so no fourth shape exists NOW. But if any future caller reaches llm_client.py:1894 through that bridge, rows land as provider='anthropic' with agent=_role (frequently NULL) and this exclusion will NOT catch them -- phantom pricing returns in a new shape. This deserves its own masterplan step, not a prose mention.\n- The fake is SQL-TEXT-keyed by design (pre-dates this step). A semantically equivalent rewrite produces a false RED: Q/A mutation MX (`agent != 'cc_rail'` -> `agent NOT LIKE 'cc_rail'`) turns the bare-shape test red although the SQL means nearly the same thing. Accepted design property, documented in the module docstring; worth knowing before any future refactor of that predicate.\n- The new 3VL branch keys on the substrings \"agent != 'cc_rail'\" / \"NOT LIKE 'cc_rail\", so it does not model the NULL drop for the step-text form (that is why test_agent_none_rows_are_included stayed green under MSTEP). Harmless for the shipped SQL.\n- Verdict arrived at via the P1 worst-of-N-lenses rule: correctness PASS, scope-honesty PASS, does-it-reproduce CONDITIONAL -> min() = CONDITIONAL. This is the FIRST Q/A on 75.5.12 (0 prior harness_log entries), so the 3rd-CONDITIONAL auto-FAIL rule does not engage; retry_count 0 < max_retries 3, so certified_fallback is false.\n- Q/A was fully read-only. The .claude/hooks/qa-write-guard.sh hook (phase-75.20.1) correctly BLOCKED an attempted scratchpad Write; all mutation work was then executed in memory via stdin, and the working tree was re-verified unchanged afterwards (same 2 modified files, 90 insertions / 6 deletions).\n- No frontend/** in the diff, so gate 1b (eslint/tsc) and gate 1c (live UI capture) are not applicable. No BQ execute-query was run (deny-gated); the 30d measurement was corroborated instead against handoff/current/census_78.json + census_78.md, an artifact authored by a DIFFERENT step, which independently reports \"2,192 calls / 4,370,458 tok agent='cc_rail' model=claude-sonnet-4-6\" -- an exact match for the dominant line, and 2192+357=2549 / 4,370,458+500,651=4.87M reproduces the docstring arithmetic."
}
```
## Main follow-up (cycle-2) — what changed in response

The Q/A confirmed **all three immutable criteria MET**, reproducing every mutation
itself rather than accepting mine, and confirmed the two things I was most exposed on:
the three-valued-logic branch is a **faithful** model (three independent proofs,
including that removing it changes nothing on correct SQL — the signature of a fidelity
fix, not overfitting), and the prefix branch is **legitimate** (removing it leaves M2
green, so criterion 1's over-match guard would be vacuous without it).

It withheld PASS for **one prose defect, and it was a real one.**

**I asserted a behavioral counterfactual I never ran.** In all three artifacts I stated
that adopting the masterplan step's own suggested predicate "would silently neuter the
existing shape-2 guard — the test would keep passing while no longer testing anything."
The Q/A executed that exact form; so did I afterwards:

```
=== MSTEP: the masterplan step text's OWN suggested predicate ===
  2 failed, 11 passed
  reds: ['test_cc_rail_rows_contribute_zero_both_shapes',
         'test_bare_cc_rail_shape_contributes_zero']
```

The *mechanism* half was right — the substring the fake keys on does drop out — but the
*consequence* is inverted: with that branch not firing the rail row is **included**, so
the assertion fails loudly. This mattered because it was the **sole stated basis for
deviating from an explicit instruction on a P1 money step**, and it is precisely the
failure class `feedback_measure_dont_assert_claims` exists to catch. I have been
demanding measured claims from the Q/A all session and then shipped an unmeasured one.

**The decision to decline the step text was nonetheless correct, for a stronger reason
the Q/A supplied and I then verified.** Taken literally, without re-adding an
`agent IS NULL` guard, the step-text form **drops every NULL-agent row** through SQL
three-valued logic. ANSI oracle:

```
STEP TEXT: NOT (agent='cc_rail' OR agent LIKE 'cc_rail:%')   kept: ['cc_railway', 'synthesis']
MINE:      (agent IS NULL OR (agent!='cc_rail' AND ...))     kept: [None, 'cc_railway', 'synthesis']
```

NULL is the **common** metered case, not an edge case: `llm_client.py:1127` logs
`agent=config.get("_role")` and `"_role"` is set in exactly two places repo-wide
(`autonomous_loop.py:2722`, `:2762`). Measured blast radius from this cycle's own 30d
query: **226 Gemini calls / 232,090 tokens** plus 3 haiku calls carry a NULL agent and
would have been silently dropped from metered spend — an **under-count**, so the
breaker opens LATE. That is the more dangerous direction, and it makes the De-Morgan'd
form strictly safer rather than merely test-friendlier.

Fixed in `contract_75.5.12.md`, `experiment_results_75.5.12.md` and
`live_check_75.5.12.md`: the refuted claim is left visible with a `CORRECTION` block
above it rather than quietly rewritten, so the error and its correction are both
auditable. **No code changed** — the Q/A verified the implementation green on its own
runs, and the defect was entirely in my justification.
