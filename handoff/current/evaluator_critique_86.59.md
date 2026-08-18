# evaluator_critique -- step 86.59

## Verdict ledger

| cycle | verdict | run id | recorded |
|---|---|---|---|
| 1 | **CONDITIONAL** | `wf_5a3bc88c-4e1` | 2026-08-18T03:08:02Z |

---

## Cycle 1 -- CONDITIONAL -- `wf_5a3bc88c-4e1`

*Transcribed verbatim from the captured Workflow return in the same turn it
landed. Main records the verdict and never authors it.*

**verdict:** `CONDITIONAL`
**ok:** `false`
**certified_fallback:** `false`
**harness_compliance_ok:** `true`
**research_needed:** `false`

**reason:**

> Criteria 1/3/4/5 MET and every reported number reproduced EXACTLY under my own
> independent re-runs (rho 0.9622/0.9319, 15.8%/day, 12 distinct + IT 72.0%, flag arms
> 15.8/28.4/22.1/17.9, effective shares 22.6/37.0/40.4, matrix 14/14 KILLED with control
> GREEN first and SHA-verified restore); immutable command exit=0 ("parses"), derived-scope
> ruff F821/F401/F811 clean over 4 files, zero production files in the step commit 15a817cc,
> 86.59 criteria byte-identical across that commit, harness compliance clean (gate PASSED 8
> sources/54 URLs, mtime order research<contract<artifacts, status still pending,
> attempt_number=1). Criterion 7 is NOT MET: "mutation-test EVERY new guard" fails because 2
> of the 19 new guards cannot be made red by any mutation -- `panel_is_us_only` is a literal
> `True` (AST-proven), and `baseline_arm_is_the_unflagged_ranking` asserts
> `len(x)==len(set(x))` on a list built as `sorted({...})`, so it is true for every input
> while its NAME claims criterion 4's load-bearing property. I proved this by EXECUTION, not
> argument: with the control GREEN first, poisoning FLAG_ARMS[0] to carry
> `sector_neutral=True` SURVIVED (baseline turnover 13.3%->46.7%, distinct 7->11, top-sector
> 75%->20%, becoming byte-identical to the sector_neutral arm so every criterion-4 delta
> would have read ZERO), and making the soft-diversity arm inert (w=0.0) also SURVIVED; a
> re-aimed positive control (measure_flags(1,W)) DID trip `flag_arms_all_ran`, so the probe
> reaches the path. The criterion-4 code path carries only 2 guards total -- its own output
> prints "OK: all 2 invariants hold" -- one of which is the tautology. Secondary: the sigma
> triple "~10.2/~19.4/~31.0" quoted in live_check, experiment_results and (durably)
> masterplan 86.117 does not reproduce from its own cited command (re-derived means
> 10.646/19.849/30.442) and is self-inconsistent with the artifact's own effective shares,
> which the re-derived means reproduce exactly. The step's PRODUCT is sound and nothing ships
> to production, so this is an evidence-layer gap, precisely fixable -- not a wrong result.

**violated_criteria:**
`criterion_7_mutation_test_every_new_guard`, `illusory-guard`, `anti-rubber-stamp`

### violation_details

**1. Circular_Reasoning**

- *action:* In-process mutation with control observed GREEN first: `FLAG_ARMS[0] =
  ("baseline", {"sector_neutral": True})`, then `measure_flags(4, 126)`
- *state:* SURVIVED -- no invariant raised. Both guards RAN (`_RAN ==
  ['flag_arms_all_ran','baseline_arm_is_the_unflagged_ranking']`). Baseline turnover
  13.3%->46.7%, distinct 7->11, top-sector share 75%->20%; the 'baseline' arm became
  byte-identical to the sector_neutral arm, so every criterion-4 delta (+12.6/+6.3/+2.1pp)
  would have been reported as zero and nothing detected it. Re-aimed positive control
  `measure_flags(1,126)` DID raise 'INVARIANT FAILED: flag_arms_all_ran', proving the probe
  reaches and can trip guards on this path.
- *constraint:* `scripts/qa/rank_stability_86_59.py:707-710` `base_d =
  out["baseline"]["distinct"]; _ok("baseline_arm_is_the_unflagged_ranking",
  out["baseline"]["n_distinct"] == len(set(base_d)), ...)` -- `distinct` is built at :692 as
  `sorted({t for s in slates for t in s})`, so `len(x)==len(set(x))` is TRUE for every
  possible input (verified on `[]`, `['A']`, `['A','B','C']`, `sorted({'A','B','A'})`).
  qa.md 4c: a guard that cannot fail when its subject is broken does not count; sole-coverage
  vacuity on a money-path criterion is BLOCKING. This is sole coverage -- `measure_flags()`
  carries exactly 2 guards and prints 'OK: all 2 invariants hold'.

**2. Overgeneralization**

- *action:* AST census of every `_ok(...)` invariant in
  `scripts/qa/rank_stability_86_59.py` vs the 14 cells in `scripts/qa/mutation_86_59.py`
- *state:* 19 new guards exist; 13 are named by a matrix cell. 6 are uncovered
  (`panel_is_us_only`, `dedup_actually_fired_on_this_panel`, `enough_sessions_for_window`,
  `baseline_arm_is_the_unflagged_ranking`, `price_only_multidim_arm_ran`,
  `displacements_are_tie_explained` -- the last covered transitively via `_tie_explained`).
  Two of the uncovered ones are UNKILLABLE: `:337 _ok("panel_is_us_only", True, ...)` is a
  literal-True constant (AST-proven, the only one in the file), and `:708` as above. The
  artifact presents '14 cells, 14 KILLED, 0 SURVIVED, 0 UNSCORABLE' as satisfying criterion 7.
- *constraint:* Immutable criterion 7: 'mutation-test every new guard: revert it and show the
  check goes red, with the control observed GREEN first and a byte-identical restore.' A
  matrix licenses only 'these N mutations were killed', never a global claim (qa.md 4c).
  MITIGATING and measured: the fetch SQL at `:113` does carry `WHERE market = 'US'`, and I
  verified the cached panel independently -- 0 of 513 tickers deviate from a US symbol shape
  -- so `panel_is_us_only` is decorative, not masking an error.

**3. Contradiction**

- *action:* Re-derived the cross-sectional sigmas from the script's own `--dispersion
  --cycles 20` table, which I re-ran and which reproduces line for line
- *state:* Claimed '~10.2 (1m), ~19.4 (3m), ~31.0 (6m)'. Re-derived means: 10.646 / 19.849 /
  30.442 (medians 10.375 / 19.775 / 29.870) -- neither statistic yields the quoted triple.
  Self-inconsistency: weight x sigma from the CLAIMED triple gives effective shares
  21.9/36.5/41.6, but the artifact's own headline is 22.6/37.0/40.4, which is exactly what
  the RE-DERIVED means give (22.6/36.9/40.4). The load-bearing ratio survives (2.86x measured
  vs '~3.0x' claimed), so no conclusion moves.
- *constraint:* qa.md 4b: every numeric claim must re-derive from the command cited for it; a
  number that does not reproduce is a Contradiction finding. Aggravating: the non-reproducing
  triple has been propagated into `.claude/masterplan.json` step 86.117's `audit_basis`,
  where a future step will read it as the measurement rather than as an approximation.

**4. Invalid_Precondition** *(severity: NOTE)*

- *action:* Ran verbatim the command the artifacts cite as evidence for 'no production file
  modified': `git status --short -- backend/`
- *state:* Returns THREE modified files (`backend/api/sovereign_api.py`,
  `backend/services/autonomous_loop.py`, `backend/services/experiments/perf_results.tsv`) --
  all from unrelated in-flight work (the autonomous_loop hunk is a /reports empty-summary UI
  fix). The SUBSTANTIVE claim is TRUE and I verified it at commit level: `git show
  --name-only 15a817cc | grep -E '^(backend|frontend)/'` returns NONE, and settings.py is
  untouched with all three flags still False/0.
- *constraint:* `experiment_results_86.59.md:26-27` and `live_check_86.59.md:180` present
  that command as the reproducing evidence. severity=NOTE -- claim-precision only; cite the
  commit-scoped command instead, since the working tree is shared with other steps.

### checks_run

`harness_compliance_audit_5_item`, `immutable_verification_command_exit_0`,
`git_scope_audit_of_the_step_commit`, `immutable_criteria_unchanged_across_commit`,
`python_lint_gate_ruff_F821_F401_F811_derived_scope`,
`independent_rerun_rank_stability_cycles_20`, `independent_rerun_flags_cycles_20`,
`independent_rerun_dispersion_cycles_20`, `independent_rerun_mutation_matrix_14_cells`,
`ast_guard_census_and_tautology_detection`,
`adversarial_mutation_baseline_arm_poisoned_SURVIVED`,
`adversarial_mutation_inert_flag_arm_SURVIVED`, `positive_control_reaimed_and_KILLED`,
`panel_provenance_reverified_from_pickle`, `sector_map_coverage_reverified`,
`sigma_triple_rederivation`, `criterion_5_flag_state_verification`, `code_review_heuristics`,
`prior_attempt_and_verdict_evidence`, `head_and_tree_recheck_before_return`

---

## Main's response -- cycle 1 (fix applied, see cycle 2)

**All four findings are accepted. Nothing is disputed.** Finding 1 is the one
that matters: the Q/A did not argue that my guard was weak, it **poisoned the
baseline arm and showed the run stayed green** while every criterion-4 delta
collapsed to zero. That is sole-coverage vacuity on this step's most
consequential result, and the criterion-4 table is exactly what the operator
asks are built on.

The irony is recorded rather than smoothed over: this step's own artifact
claims credit for finding two vacuous guards in its deliverable, and shipped a
third and a fourth in the same file. Finding a tautology does not immunise you
against writing another one, and "I already audited for this" is precisely the
assumption the audit is supposed to replace.

Fixes are listed in the cycle-2 section below.
