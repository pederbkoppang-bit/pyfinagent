# Experiment results — Step 83.0.3: PBO false-pass on the MCP surface (PROOF step)

Date: 2026-08-07 (autonomous drain, cycle 169). Contract: `contract_83.0.3.md`.

## What was built

**One new test file, zero production code changes** (contract D1): `backend/tests/test_phase_83_0_3_pbo_false_pass.py` — **6 tests** (5 in cycle 1; the routing spy added in cycle 2). The routing fix itself shipped under commit `e5bb9f25` (phase-82.27); this step proves it and pins it against regression:

0. `test_c1_routing_spy_compute_pbo_checked_is_invoked` (cycle 2) — the routing made behaviourally observable: monkeypatches the analytics attribute the function-scoped import resolves at call time and asserts it is invoked; kills the inline-refusal impostor class (0 spy calls under it, measured).
1. `test_c1_c2_undersized_matrix_refuses_not_passes` — the refusal payload field-by-field via the shared `_assert_refusal`, plus the discriminator: raw `compute_pbo` returns 0.0 on the identical input (necessary-but-not-sufficient for routing — see the cycle-2 Follow-up).
2. `test_c3_reverting_to_raw_compute_pbo_fails_criterion_2` — the revert mutant (anchor count asserted ==1, `ast.parse`-validated, exec'd in-memory, never touches disk) reproduces the ORIGINAL false PASS (`ok: True, pbo: 0.0, vetoed: False` — non-equivalence pinned) and fails the SAME `_assert_refusal` set the criterion-2 test uses (`pytest.raises(AssertionError)`).
3. `test_c4_near_identical_columns_reported_not_diverse` — fixture precondition asserted FIRST (min pairwise corr > 0.99; the eps window is narrow, so seed drift fails as a fixture, never as a false verdict); `columns_diverse` False; `gate_grade` present and False (N=8 < 10 — deliberately NOT asserted to discriminate diversity).
4. `test_c4_mirror_independent_columns_reported_diverse` — the mandatory mirror guard (an always-False diagnostic would vacuously satisfy #3).
5. `test_refusal_survives_client_round_trip` — the refusal survives into `structured_content` over a real in-memory fastmcp Client (contract D4). The payload-vs-protocol `is_error` divergence is deliberately NOT asserted — owned by queued step 83.0.6.

## Verification (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_0_3_pbo_false_pass.py -q
.....
5 passed in 2.49s
```

Regression guard across the whole PBO surface: `pytest test_phase_83_0_3... test_phase_82_23... test_phase_82_27... -q` → **46 passed in 8.77s** (6 new + the 40 pre-existing; re-measured post-cycle-2 — the cycle-1 figure was 45/8.72s with 5 new).

## Follow-up — cycle 2 (2026-08-07, after Q/A CONDITIONAL wf_b361863d-3c4)

Cycle-1 verdict (verbatim in `evaluator_critique_83.0.3.md`): all 6 criteria MET; capped by one WARN — the artifacts credited criterion 1 to the behavioural refusal, which the Q/A's impostor probe refuted (an inline-refusal wrapper over raw `compute_pbo` passes every behavioural test; only the `test_c3` source anchor killed it). All three named fixes applied:

- **(a) Attribution corrected** in the test docstring and via an appended Correction in `contract_83.0.3.md` (original D1 text preserved as the record): behavioural refusal is necessary-but-not-sufficient; criterion 1 is carried by the routing spy + the `test_c3` anchor.
- **(b) Do-not-delete comment** added at the `test_c3` anchor assertion.
- **(c) Routing spy added**: `test_c1_routing_spy_compute_pbo_checked_is_invoked` monkeypatches `backend.backtest.analytics.compute_pbo_checked` (the function-scoped import resolves at call time) and asserts it is actually invoked. Measured discrimination proof: running the Q/A's impostor shape against the spy records **0 calls** while the impostor still refuses — the spy fails exactly the case the behavioural tests miss.

Suite after cycle 2: immutable command → **6 passed in 2.50s**.

## Follow-up — cycle 3 (2026-08-07, after Q/A CONDITIONAL #2 wf_2f31e904-f24)

Cycle-2 verdict (verbatim in `evaluator_critique_83.0.3.md`): the routing WARN is CLOSED (the Q/A re-proved it with three independently-built impostors — I1 inline-refusal, I2 local-function-named-compute_pbo_checked, I4 call-then-discard — all killed by the spy/refusal set); capped by a criterion-5 evidence-staleness WARN: the census block still cited the discriminator at `:86` after the cycle-2 spy insertion moved it to `:112`, the unqualified grep no longer reproduced (297 raw / 51 qualified vs the stated 50), and two cycle-1 note fixes (N1 qualified-grep, N3 serialization note) had not been applied. All five named fixes are now in: census re-run post-edit (`:112`), qualified grep committed (51, with the 297-unqualified explanation), guard lines corrected to `:192`/`:200` with their conditions, test count corrected to 6 with the spy enumerated and the regression figure re-measured (46 passed in 8.77s), and the live_check serialization note added (1e-7 `column_corr_mean` delta from 6dp JSON rounding; `pbo` bit-identical). Every number in this section was re-derived after the final edit of this cycle — nothing carried forward.

## Criterion 5 — raw `compute_pbo` call-site census (measured DELTA, sites enumerated)

AST call-node census (authoritative; `ast.Call` with `func.id == 'compute_pbo'`, `.venv` skipped), **RE-RUN 2026-08-07 AFTER the cycle-2 edit** (the cycle-1 block cited the discriminator at `:86`; the cycle-2 spy insertion moved it — cycle-2 Q/A finding, re-measured):

```
backend/backtest/analytics.py:240                      KEEP -- the checked wrapper's own delegation
backend/tests/test_phase_82_23_pbo_in_gate.py:37       KEEP -- deliberately pins the hazard (82.23)
backend/tests/test_phase_83_0_3_pbo_false_pass.py:112  NEW THIS STEP -- the C1/C2 discriminator, same class as the 82.23 pin
scripts/harness/run_82_3_candidate_backtests.py:206    guarded at :192 (`if len(series) < 2:`) / :200 (`if T < 32:`) (82.27)
tests/autoresearch/test_phase_48_2_backtest_adapter.py:106  test fixture
```

**DELTA: production call sites outside the wrapper — before = 0, after = 0.** Total enumerated sites 4 → 5; the one addition is this step's own test discriminator. Corroborating text census, qualified form committed per the cycle-1/2 findings:

```
$ grep -rnP --include='*.py' '\bcompute_pbo\b(?!_checked)' . | grep -v '/\.venv/' | wc -l
      51
```

(51 post-cycle-2 — the spy test's docstring added one matching line to the cycle-1 count of 50. The UNQUALIFIED form without `--include='*.py'` returns 297 because it sweeps .md/.jsonl handoff artifacts; the qualified command above is the committed one. Lines are overwhelmingly comments/docstrings, which is why the AST census is authoritative.)

## Criterion 6 — gate.py minimum-trials bypass (disclosure, verbatim source)

`backend/autoresearch/gate.py:43-53`, verbatim:

```python
        n_trials = trial.get("pbo_n_trials")
        if n_trials is not None:
            try:
                n_int = int(n_trials)
            except (TypeError, ValueError):
                return {"promoted": False, "reason": f"non_numeric_pbo_n_trials:{n_trials!r}",
                        "trial_id": trial.get("trial_id")}
            if n_int < self.min_pbo_trials:
                return {"promoted": False,
                        "reason": f"pbo_trials_below_min:{n_int}<{self.min_pbo_trials}",
                        "trial_id": trial.get("trial_id")}
```

A trial dict that OMITS `pbo_n_trials` skips the `min_pbo_trials` floor entirely — a deliberate, documented legacy carve-out (comment at :41-42: "Absent => unchanged legacy behaviour"), already proven live by `test_phase_82_27_pbo_sweep_producer.py:258-262` (stripping the key flips promoted False→True). **Actionable consequence for 83.5, recorded here as the criterion demands: any producer feeding the gate MUST always emit `pbo_n_trials`, or the floor is inert for it.** Disclosure only — thresholds are immutable and out of scope (step text).

## Corrections to the step's own text (measured)

- The step name's anchors `risk_server.py:142-143` are STALE: post-`e5bb9f25` those lines are comment prose about the old defect; the live checked call sits at the `compute_pbo_checked` import+call a few lines below. Criteria were all satisfiable as written.
- The step's premise ("the MCP does not use it") described the pre-82.27 tree; 83.0.3 is executed as a PROOF step per the contract.

## live_check

`live_check_83.0.3.md`: the "BEFORE" is the ast-validated mutant reproduction, explicitly labelled as such (the pre-fix tree state no longer exists — an honest historical capture is impossible); the AFTER is two verbatim `mcp__pyfinagent-risk__pbo_check` responses over the live MCP surface (refusal on T=10; `columns_diverse: false` + PBO 0.851 veto on the near-identical matrix).

## Files changed

`backend/tests/test_phase_83_0_3_pbo_false_pass.py` (new). Handoff: `contract_83.0.3.md`, `research_brief_83.0.3.md`, `live_check_83.0.3.md`, this file. Production code byte-unchanged (proof step).
