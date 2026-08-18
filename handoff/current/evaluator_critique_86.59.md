# evaluator_critique -- step 86.59

## Verdict ledger

| cycle | verdict | run id | recorded |
|---|---|---|---|
| 1 | **CONDITIONAL** | `wf_5a3bc88c-4e1` | 2026-08-18T03:08:02Z |
| 2 | **CONDITIONAL** | `wf_d1d01d57-0f6` | 2026-08-18T03:37:41Z |
| 3 | **CONDITIONAL** | `wf_2cc6808c-bea` | 2026-08-18T04:08:16Z |

**3rd-CONDITIONAL rule is now BINDING.** Three consecutive CONDITIONALs stand.
Per CLAUDE.md F1 the next Q/A pass **must return FAIL regardless of evidence**.
The step is therefore **PARKED** rather than re-spawned: a fourth attempt cannot
produce a PASS, so spending one would burn tokens to obtain a verdict the rule
has already determined. See `escalation_86.59_third_conditional.md`.

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

---

## Cycle 2 -- CONDITIONAL -- `wf_d1d01d57-0f6`

*Transcribed verbatim from the captured Workflow return in the same turn it
landed.*

**verdict:** `CONDITIONAL` | **ok:** `false` | **certified_fallback:** `false`
**harness_compliance_ok:** `true` | **research_needed:** `false`

**reason:**

> Every published number reproduces EXACTLY under my own independent re-runs (rho
> 0.9622/0.9319, turnover 15.8%/day, 3-of-19 zero-turnover sessions, 12 distinct + IT 72.0%
> with identical counts {20,72,8}, flag arms 15.8/28.4/22.1/17.9 with deltas +12.6/+6.3/+2.1pp,
> sigmas 10.646/19.850/30.441 and 2.86x now PRINTED by the script, effective shares
> 22.6/37.0/40.4, multidim 50/10139 = 0.493%, fidelity 80%, dedup 47,880/200,875, 10 split
> bars); immutable command exit=0 "parses"; derived-scope ruff F821/F401/F811 clean on both
> the step-commit scope and the working-tree scope; ZERO production files in BOTH step commits
> (15a817cc and 3e75c2d6, grep exit 1 each); criteria byte-identical across
> 15a817cc~1/15a817cc/3e75c2d6/HEAD; harness compliance clean (gate PASSED 8 sources/54 URLs,
> mtime order brief<contract<artifacts, status still pending, no harness_log result row). All
> four cycle-1 findings are genuinely fixed and I re-verified each by execution: the matrix is
> 20/20 KILLED with control GREEN FIRST on all three modes and a SHA-256-verified restore, the
> AST census reports full coverage and I proved it is not vacuous with a positive control,
> cycle-1's two exact mutations now KILL, and the sigma correction reached masterplan 86.117's
> audit_basis. Criteria 1/3/5 fully MET; 2 and 6 are degenerately satisfied and honestly
> flagged BY the author rather than claimed -- and I rule that criterion 2 does NOT require the
> reweighted term to be built here, because criterion 4 mandates measuring existing flags
> before new code (all three MOVE the slate), criterion 2's own DSR/PBO demand reads a table
> this step measured at 38% duplicate keys, and the obligation is carried forward verbatim into
> 86.117's criteria. THREE residuals cap this at CONDITIONAL. (1) A poisoned BASELINE reference
> SURVIVES all four criterion-4 guards at the PUBLISHED --cycles 20 setting: with control GREEN
> first and FLAG_ARMS[0] left byte-identical as ("baseline", {}), injecting momentum_52wh_tilt
> at the script's own replay_session seam flips min_k_sectors=3's reported delta from +2.1pp to
> -2.1pp -- a SIGN FLIP on the exact number ASK-1 rests on -- and a w=0.05 variant survives
> with every turnover delta reading exactly as published while baseline top-sector share
> silently moves 0.72->0.64. (2) live_check section 8 presents "coverage: 20 guards in target,
> 20 covered" in the same fenced block as "sha256 unchanged (9282ba866f2afc87...)"; at that
> exact sha the command prints 21/21 -- a spliced verbatim capture, and the second
> non-reproducing number in this step. (3) The _PREDICATE_FIXTURE that makes four of the twenty
> cells meaningful has no cell and no size assertion: emptying it leaves
> predicates_reject_known_bad_inputs GREEN. The PRODUCT is sound, nothing ships to production,
> and all three are precisely fixable in a few lines.

**violated_criteria:** `criterion_4_baseline_reference_has_no_behavioural_guard`,
`criterion_7_evidence_block_does_not_reproduce`, `illusory-guard`

### violation_details (abridged to the action/state/constraint triples)

**1. Unjustified_Inference** -- wrapped `rs.replay_session` so only the baseline
arm's call acquired `momentum_52wh_tilt=True, k=0.2`; `FLAG_ARMS[0]` left
byte-identical; `measure_flags(20, 126)`. **SURVIVED** -- all four guards ran and
none fired. Baseline turnover 15.8%->20.0%, distinct 12->13; reported deltas
became sector_neutral +8.4pp (published +12.6), soft_diversity +2.1pp (published
+6.3), min_k **-2.1pp (published +2.1pp) -- a SIGN FLIP on the figure ASK-1
cites**. A `w=0.05` variant also survived at 20 cycles with all three turnover
deltas reading exactly as published while baseline top-sector share silently
moved 0.72->0.64. A `w=0.15` variant survived at `--cycles 4` but was killed at
20, so kill/survive is cycle-count dependent and the matrix ran every cell at 4,
**making the matrix oracle weaker than the published run's**.
*Constraint:* `baseline_arm_applies_no_flags` asserts the arm DEFINITION, which
no downstream injection touches, while its own detail string claims the
behavioural property. *MITIGATING and measured:* `backend/tools/screener.py`
contains ZERO `settings.` references, so flags reach `rank_candidates` only as
explicit caller kwargs -- this is a code-edit risk in the measurement harness,
**not reachable from an operator promoting ASK-1/ASK-2**.

**2. Contradiction** -- `live_check` §8 claimed "coverage: 20 guards in target,
20 covered" in the same fenced block as "sha256 unchanged (9282ba866f2afc87...)";
at that exact sha the command prints **21/21**. A spliced capture. Direction is
conservative (it understates coverage) and 20 coincides with the CELL count,
"which is exactly the guards-vs-cells conflation that would mask a real coverage
gap." Second non-reproducing number in this step.

**3. Missing_Assumption** -- cleared `_PREDICATE_FIXTURE` and called
`_check_predicate_fixture()`: **SURVIVED**, because a loop over an empty list
yields `bad == []`. With `_us_only` additionally rebound always-true, fully
blind. The fixture is the entire cycle-2 remediation mechanism and the
load-bearing half of 4 of the 20 cells, yet carried no cell and no size
assertion -- and the AST census structurally cannot see it, because the fixture
is DATA, not an `_ok(...)` call.

---

## Main's response -- cycle 2 (fixes applied, cycle 3 below)

**All three accepted.** Finding 1 is the one that matters and it is the same
lesson one level deeper: cycle 1's guard checked a *value*, so I replaced it
with one that checks the *definition* -- and a definition is still not
behaviour. The evaluator injected at the seam, left the definition untouched,
and flipped the sign of the exact number ASK-1 rests on.

The fix stops asserting anything about how the baseline was *configured* and
instead **recomputes the baseline slate through a direct, unflagged
`rank_candidates` call** and requires the two to agree. An injection anywhere in
the replay path makes them diverge.

Finding 2 is the second non-reproducing number in this step, and both were mine.
The §8 block is now a single verbatim capture; nothing in it is assembled from
more than one run.

---

## Cycle 3 -- CONDITIONAL -- `wf_2cc6808c-bea`

*Transcribed verbatim from the captured Workflow return in the same turn it
landed.*

**verdict:** `CONDITIONAL` | **ok:** `false` | **certified_fallback:** `false`

**reason (abridged to the load-bearing half; the confirmatory half is in the
run record):**

> Every published number reproduces EXACTLY under my own independent re-runs [...]
> Mutation matrix reproduced end to end: control GREEN FIRST on all three modes, coverage 23
> guards/23 covered, KILLED 22/22, SURVIVED 0, UNSCORABLE 0, restore sha256 unchanged
> be0565ff3c9615da -- which also matches the sha in live_check section 8, so the
> previously-spliced block is genuinely one capture now. [...] ZERO production files across ALL
> FOUR step commits (15a817cc, 3e75c2d6, fb6f8a67, a4a5765c -- I checked all four, not the two
> the artifact cites) [...] Harness compliance 5/5 [...] Cycle-2 findings 2 and 3 are genuinely
> closed and I verified both by execution. ONE BLOCKING RESIDUAL, and it is the same class a
> third time, one seam over. delta = arm - baseline. The cycle-3 fix guards the `base` call at
> measure_flags():806, which feeds ONLY the min_k_sectors=3 ARM. The row every delta is
> subtracted FROM -- arms["baseline"], produced by a separate, structurally identical
> replay_session(...) inside `for name, kw in FLAG_ARMS` at :799-802 -- has no behavioural
> guard at all. With the CONTROL observed GREEN first and reproducing the published table to
> the digit, adding momentum_52wh_tilt=True,k=0.2 to that one line SURVIVES all six criterion-4
> guards while baseline goes 15.8%/12/IT 72% -> 20.0%/13/IT 70%, sector_neutral 28.4% -> 46.3%
> with its top sector changing Industrials->Financials, and min_k's delta FLIPS +2.1pp ->
> -2.1pp (the exact figure ASK-1 rests on) even though the min_k arm itself is provably
> unchanged. A w=0.05 variant also SURVIVES with every turnover delta reading EXACTLY as
> published while the baseline's top-sector share moves 0.72->0.64 -- verbatim the variant the
> cycle-3 narrative says was closed. Both were applied in memory only; the file on disk was
> never written and md5 is identical pre/post. Consequently the claim at
> rank_stability_86_59.py:816-818 and in live_check section 8 -- "An injection anywhere in the
> replay path -- at the seam, in the kwargs, in a wrapper -- makes these diverge" -- is FALSE.
> No criterion is missed and nothing ships to production, and the fix is a few lines [...]

**violated_criteria:** `criterion_4_baseline_ROW_has_no_behavioural_guard`,
`illusory-guard`, `guard_coverage_claim_exceeds_the_guard`

**violation 1 -- Missing_Assumption (BLOCKING).** In-memory mutant, disk never
written, null-mutant CONTROL first and reproducing the published table exactly.
Injected at `:799-802`, left `:806` and `FLAG_ARMS[0]` byte-identical.
**SURVIVED** -- all six guards ran, none fired. min_k's delta flipped +2.1pp ->
-2.1pp *while the min_k arm itself was provably unchanged*; soft_diversity's
delta dropped +6.3pp -> +3.2pp (ASK-2's figure). *Constraint:* "Two structurally
identical call sites; one guarded."

**violation 2 -- Overgeneralization (BLOCKING).** The shipped claim "An
injection anywhere in the replay path ... makes these diverge" is false; both
injections are at the replay seam and neither diverges. *"A guard licenses only
the variable it compares -- not the path it is described as covering."*

**violation 3 -- Unjustified_Inference (NOTE).** The contract commits to P3
(build the standardisation behind a default-OFF flag), P4 (DSR/PBO on it) and P6
(parity with it OFF). None was built. The reason is sound and recorded, "but it
steps over the CONTRACT's own plan, and the contract was never reconciled."

---

## Main's response -- cycle 3

**All three accepted; the first is the one that matters and it is my error four
times over.** The lesson arrived as: a *value* check (cycle 1), then a
*definition* check (cycle 2), then a *behavioural* check **on the wrong
variable** (cycle 3). Each fix moved the seam instead of closing the class. The
evaluator's phrasing is the right summary and I am keeping it: *two structurally
identical call sites; one guarded.*

**Fixed anyway, though the step is parking.** The oracle is now computed once
per cycle *before* the arms loop, and **both** unflagged slates -- the baseline
ROW and the min_k reference -- are required to equal it
(`baseline_ROW_matches_an_unflagged_direct_call`, cell M22). Both of the
evaluator's exact injections now KILL at the published `--cycles 20`, control
green and reproducing 15.8%/12 with min_k at +2.1pp, disk untouched. Matrix
23/23, coverage 24/24. The false coverage claim is narrowed in the code comment
and in both artifacts.

**Violation 3 is reconciled** in `experiment_results_86.59.md` rather than left
implicit: the contract's P3/P4/P6 were abandoned, for a measured reason, and
that is now stated as a **deviation** instead of being papered over by "the
criteria describe a measurement".

---

## Cycle 4 -- judge returned CONDITIONAL -> **FAIL** by the F1 3rd-CONDITIONAL rule -- `wf_a1e6c7bd-10c`

*Transcribed verbatim from the captured Workflow return in the same turn it
landed. Main records the verdict and never authors it.*

**judge verdict (unmodified):** `CONDITIONAL`
**applied verdict:** `FAIL` -- the CALLER applies CLAUDE.md F1. `enforceEscalation`
computed `consecutive_conditionals: 3` and
`would_auto_fail: true`; a fourth CONDITIONAL is converted, and the
counter RESETS on this FAIL.
**ok:** `false`
**certified_fallback:** `false`
**harness_compliance_ok:** `true`
**research_needed:** `false` (routing: no re-research)
**attempt_number:** `4` of `5`; **budget_exhausted:** `false`
**judge_was_told_consequence:** `false` -- the spawn carried no positional claim,
so the verdict is not consequence-primed.

**reason:**

> PRODUCT SOUND, EVIDENCE CAPPED. Every published number reproduced EXACTLY under my own independent in-process drive of the real module (rho 0.9622/0.9319, top-10 and top-5 turnover both 15.8%/day, 3-of-19 zero-turnover, 12 distinct with the exact ticker list, IT 72.0% counts {Industrials 20, IT 72, Health Care 8}, fidelity 79.6%, 18 live distinct, dedup 47,880/200,875, 513 tickers, sigmas 10.646/19.850/30.441 at 2.86x, effective shares 22.6/37.0/40.4, multidim 50/10,139 with 5 identical cycles, and the whole criterion-4 table 15.8/28.4/22.1/17.9 with deltas +12.6/+6.3/+2.1pp). Deterministic: immutable command exit=0 "parses"; derived-scope ruff F821/F401/F811 clean over the 4 .py files in the step commits (git-derived -- `git diff --name-only HEAD -- '*.py'` is EMPTY because the work is committed); ZERO production files across ALL step commits; `--verify` -> "OK: all 42 invariants hold"; mutation matrix re-run BY ME reproduces control-GREEN-first on all three modes, coverage 24/24, KILLED 23/23, SURVIVED 0, UNSCORABLE 0, restore sha256 16164dcb7e04f039... which I confirmed equals `git show HEAD:` of the target; my own AST census independently confirms 24 distinct guard names; the two guards a prior cycle proved unkillable are genuinely gone. Harness compliance clean (gate PASSED 8 sources/54 URLs/recency true; all 7 criteria byte-identical masterplan-vs-contract AND HEAD-vs-worktree; status still pending; evidence CHANGED since verdict 3 via commit 099414fe, so this is a legitimate fresh respawn, not verdict-shopping). CAP: I executed two ONE-INPUT changes that each INVERT the ASK-1 vs ASK-2 ordering at the published --cycles 20 while every guard stays green. (1) The min_k arm's label `arms["min_k_sectors=3"]` (:793) and its parameter `_min_k_sector_slice(base, ANALYZE_TOP_N, 3)` (:853) are two independent literals with nothing tying them: k=4 SURVIVES and the row still labelled "min_k_sectors=3" then reports +6.3pp (tying ASK-2), k=5 SURVIVES at +7.4pp (exceeding it) -- while ASK-1 recommends promotion explicitly "at the smallest turnover cost of the three arms (+2.1pp/day)". k=2 IS killed, but by accident (the arm degenerates to baseline and trips `flag_arms_are_distinguishable_from_baseline`), which is vacuity shape #11, mis-attributed kill mechanism -- it took k=4/k=5 to establish this. (2) `sector_map_covers_most_of_the_panel` floors coverage at 50% while the operating point is 502/513 = 97.9%; degrading the cached map to 401/513 = 78.2% (a realistic yfinance-lookup failure, far above the floor) leaves all 7 criterion-4 guards green while soft_diversity and min_k SWAP turnover cost (+6.3pp -> +2.1pp and +2.1pp -> +6.3pp), with baseline and every top-sector share unmoved so nothing in the report signals it. (3) live_check_86.59.md:222-223 prose says "the cell count (22) is lower than the guard count (23)" nine lines below its own fenced verbatim block at :213 reading "24 guards in target, 24 covered" -- I re-derived 24 -- and experiment_results_86.59.md:99-101 says 24, so the two artifacts disagree; this is the same class as the step's own cycle-3 item 10, the block regenerated and the authored prose beside it left stale. Criteria 1/3/5/7 MET. Criteria 2 and 6 are VACUOUSLY satisfied because their antecedent is false (no new or reweighted term, so no new behaviour to disable) -- I accept that reading because criterion 2 is conditional in form, criterion 4 explicitly contemplates "do not rebuild a mitigation that already exists" and criterion 5 forbids promoting a flag, and the reason for abandoning the contract's P3/P4/P6 is MEASURED (38% duplicate keys in historical_prices would poison any DSR/PBO figure) and filed as 86.117 BLOCKED-BY 86.116 rather than absorbed; the deviation is disclosed, which is the right handling. Criterion 4 is MET on substance and capped only by (1) and (2). Nothing ships to production, so all three findings are precisely fixable evidence-layer gaps, not a wrong result. I also retired one plausible finding by execution rather than reporting it: the oracle-and-arms-poisoned-together attack needs a coordinated 2-site edit and the code comment at :845-849 already narrows the guard's scope to exactly that, which is honest scoping.

**violated_criteria:**
`criterion_4_min_k_arm_parameter_decoupled_from_its_label`, `criterion_4_sector_map_guard_floor_cannot_protect_the_published_deltas`, `illusory-guard`, `evidence_number_does_not_reproduce_live_check_s8`

### violation_details

**1. Missing_Assumption**

- *action:* SEVERITY WARN. In-process, control observed GREEN first at the published --cycles 20: force _min_k_sector_slice(base, ANALYZE_TOP_N, k) to k=4 and k=5, simulating a one-literal change at scripts/qa/rank_stability_86_59.py:853 while the row label arms["min_k_sectors=3"] at :793 is untouched
- *state:* k=4 SURVIVED -- run green, 7 guards ran, row labelled 'min_k_sectors=3' reports turnover 22.1% / delta +6.3pp / distinct 15 / IT 49.0% against the published 17.9% / +2.1pp / 14 / 60.0%. k=5 SURVIVED -- 23.2% / +7.4pp / 15 / 47.0%. k=2 was KILLED, but by flag_arms_are_distinguishable_from_baseline firing on degeneracy-to-baseline, not on the label/parameter divergence (vacuity shape #11, mis-attributed kill mechanism). Repo sha256 unchanged at 16164dcb7e04f039 before and after.
- *constraint:* criterion 4 -- 'measure what ... paper_min_k_sectors_analyzed ... do to candidate turnover'. ASK-1 recommends promoting paper_min_k_sectors_analyzed = 3 explicitly 'at the smallest turnover cost of the three arms (+2.1pp/day)'. At k=4 the row ties ASK-2 (+6.3pp) and at k=5 exceeds it, so the ordering the operator ask rests on inverts with no guard able to fail. The AST census reports 24/24 coverage -- a completeness claim -- while the parameter that NAMES the row has no guard at all. Named fix: derive the label from the k variable, or add _ok("min_k_arm_used_the_labelled_k", ...) with its own cell. WARN not BLOCK because criterion 4 does carry genuine behavioural guards that kill (M14/M20/M22 all verified by me).

**2. Threshold_Not_Met**

- *action:* SEVERITY WARN. In-process, control observed GREEN first at --cycles 20: patch load_sectors() to a map degraded to 401/513 = 78.2% coverage (every 5th ticker alphabetically blanked), a realistic build_sector_map/yfinance failure mode and far above the guard's floor
- *state:* SURVIVED -- all 7 criterion-4 guards green, no error raised. soft_diversity_w0.30 turnover 22.1% -> 17.9% (delta +6.3pp -> +2.1pp) and min_k_sectors=3 turnover 17.9% -> 22.1% (delta +2.1pp -> +6.3pp): the two recommended arms swap their cost. sector_neutral +12.6pp -> +11.6pp. Baseline (15.8% / 12 distinct / IT 72.0%) and every top-sector share are UNMOVED, so nothing printed in the report signals the degradation.
- *constraint:* criterion 4 plus the criterion-7 guard-vacuity doctrine. sector_map_covers_most_of_the_panel asserts known >= 0.5 * len(tickers) while the operating point is 502/513 = 97.9%; the floor sits about 20pp below the level at which the published ASK-1/ASK-2 ordering inverts, so the guard nominally covers the input but cannot protect the number derived from it. Named fix: pin coverage near the operating point, or assert the criterion-4 table was computed on a map of stated completeness. WARN not BLOCK because the published figures are correct for the actual 97.9% input and that input's quality is disclosed in live_check.

**3. Contradiction**

- *action:* SEVERITY WARN. Read live_check_86.59.md:213 (inside the block the artifact labels 'a single verbatim capture of one mutation_86_59.py run') against live_check_86.59.md:222-223 (authored prose nine lines below), then re-derived the guard count myself by AST over a pristine `git show HEAD:` copy and counted the cell list in mutation_86_59.py
- *state:* the fenced block says 'coverage: 24 guards in target, 24 covered'; the prose says 'The cell count (22) is lower than the guard count (23)'. Re-derived: 24 distinct guard names (21 literal + 3 f-string prefixes) and 22 CELLS + 1 NEGATIVE_CONTROL = 23 scored cells. The guard count is 24, not 23 -- the prose is stale at the pre-M22 value. experiment_results_86.59.md:99-101 states it correctly ('23 cells ... coverage 24/24'), so the two handoff artifacts disagree with each other.
- *constraint:* qa.md 4b -- every numeric claim in the handoff must reproduce. This is the same defect class the step was already capped for at cycle 3 (its own item 10, 'the section 8 evidence block was spliced from two runs'): the block was regenerated and the authored summary beside it was not. WARN not FAIL because the stale number sits in prose outside the fence rather than inside a block labelled verbatim, the correct value is present in the same section, and no conclusion moves.

**checks_run:**

- harness_compliance_audit_5_item
- research_gate_envelope
- criteria_verbatim_vs_masterplan
- criteria_immutability_head_vs_worktree
- immutable_verification_command
- git_derived_change_scope
- production_file_scan_all_step_commits
- ruff_F821_F401_F811_derived_scope
- rank_stability_verify_rerun
- mutation_matrix_full_rerun
- mutation_matrix_control_green_first
- ast_guard_census_independent
- sha256_restore_verification
- independent_rederivation_criterion_1
- independent_rederivation_criterion_3
- independent_rederivation_criterion_4_table
- independent_rederivation_finding_a_dispersion
- adversarial_mutation_min_k_parameter_k2_k4_k5
- adversarial_mutation_sector_map_degradation
- guard_vacuity_check_4c
- claim_audit_4b
- worst_of_n_lenses
- code_review_heuristics
- evaluator_critique
- verdict_ledger_and_qa_wip_evidence

**notes:**

> SEQUENCE EVIDENCE (gathered, not applied). `python scripts/qa/verdict_history_86_21.py --step 86.59 --evidence-only` -> status "ok", "3 verdict(s) from the ledger", verdicts CONDITIONAL -> CONDITIONAL -> CONDITIONAL; aggregates suppressed by --evidence-only as required. `python scripts/qa/qa_wip.py 86.59 --spawned-at 2026-08-18T07:24:36Z` -> source_present true, attempt_number 4 (attempt_number_status "ok", attempt_number_is_lower_bound true), prior_attempts 3, records_retained 4 (a GAUGE per records_retained_unit, not used as a counter), records_pruned_known null. CROSS-CHECK: prior_attempts (3) == ledger rows (3), so the ledger is NOT stale for this step. Any threshold or escalation that follows is the caller's to compute; I neither applied one nor compensated for one.
>
> CONSEQUENCE-LEAK OBSERVATION, worth the harness's attention. The phase-86.78 scrub is honoured in my spawn prompt, but two artifacts I was INSTRUCTED to read re-deliver the payload verbatim: escalation_86.59_third_conditional.md:6-7 ("The next Q/A pass **must return FAIL regardless of evidence**") and evaluator_critique_86.59.md:12-13 (same sentence), plus experiment_results_86.59.md:332-337 and :378-382. That is the exact channel arXiv 2604.15224 measures, arriving through the evidence layer rather than the instruction layer, where prose scrubbing cannot reach it. I recorded it and discounted it in BOTH directions -- I did not apply the rule and did not compensate for it -- but ERRJ=0.000 means I cannot certify that from introspection, which is precisely why the channel is worth closing. Consider keeping escalation prose out of the artifacts a Q/A is pointed at, the same way qa_wip and verdict_history were given an --evidence-only view.
>
> METHOD DISCLOSURE. The write guard correctly blocked me from writing a driver script even to the session scratchpad (it permits only .claude/agent-memory/qa/), so every adversarial mutation ran through `python3 - <<'PY'` heredocs that patch module attributes IN PROCESS. No repo file was ever written: the target's sha256 was 16164dcb7e04f039... before and after every attack, verified in-run. I confirmed there is no multiprocessing on the exercised paths before running from stdin. I also deliberately did NOT import the target while the mutation matrix was mid-run -- I checked and it had a MUTANT marker on disk at that moment (md5 8959aed1) -- and used `git show HEAD:` copies for the AST census instead.
>
> TWO CANDIDATE FINDINGS RETIRED BY EXECUTION, recorded so a later cycle does not re-raise them. (a) "k=2 shows the min_k arm is unguarded" -- FALSE, k=2 is killed; only k=4/k=5 establish the finding, and the difference matters because the k=2 kill would otherwise read as proof the arm IS guarded. (b) "the oracle and the arms can be poisoned together" -- the code comment at rank_stability_86_59.py:845-849 already narrows the guard's scope to exactly that case, and defeating it needs a coordinated two-site edit rather than a one-site survivor; that is honest scoping, not an overclaim.
>
> TWO NOTE-LEVEL ITEMS THAT DO NOT DEGRADE THE VERDICT. (1) rank_stability_86_59.py:80-83 says the top-N values are "read from backend/config/settings.py rather than restated ... a number retyped into a script is a number that can go stale" directly above two hardcoded literals; the values are correct today (settings.py:406 Field(10), :407 Field(5)) so no number moves, but the comment claims a mechanism the code does not have. (2) Five of the 24 guards -- displacements_are_tie_explained, panel_carries_no_non_us_symbols, enough_sessions_for_window, dedup_actually_fired_on_this_panel, flag_arms_are_distinguishable_from_baseline -- are never themselves shown to go red; the covering cell fires a different guard, the predicate fixture guard. Each is DECLARED in COVERED_TRANSITIVELY with a reason, which is what the doctrine asks for, so it is a stated limit rather than a violation. It is, however, the exact seam that finding (1) landed in.
>
> SCOPE NOTE ON --verify. Reading main() at :1157-1197, `--verify` runs measure() only -- it does not call measure_flags() or measure_dispersion(), and defaults to 10 cycles -- so its 42 invariants contain ZERO criterion-4 guards. Main's re-run claim reproduces exactly as stated and no artifact overclaims it (the only artifact references are the matrix control line and an explicit "`--verify` green is a self-check, not a verdict"), but criterion-4 coverage rests entirely on the matrix, not on --verify.
>
> TOOLING PROVENANCE. scripts/qa/guardlib.py and guardlib_selftest.py were committed this session under feat(qa): (3e532fe4, ff3ed8c7); they are not a masterplan step and appear only in goal_next_2026-08-19.md. Main's census claim was produced with a tool Main authored in the same session, so I did not rely on it -- I ran my own AST census over a pristine HEAD copy and got the same 24 names, so the claim reproduces independently.
>
> WORKING TREE. The only uncommitted masterplan change is a peer session's additive 26-line insert of a NEW step 86.120; 86.59's criteria are byte-identical between HEAD and the working tree and its status is still pending. This step makes no UI claim, so the 1c live-capture gate does not apply. No backend module was changed (zero production files in any step commit), so 1d is satisfied vacuously.
>
> WRITE-FIRST RECORD: .claude/agent-memory/qa/verdicts/verdict_wip_86.59__20260818T072436Z.md (STATUS COMPLETE; it is a crash-survival record, not a verdict).


---

## Cycle-4 Follow-up -- what Main changed in response

All three cycle-4 findings were WARN-severity and evidence-layer; the Q/A
independently reproduced every published number and recorded ZERO production
files across all step commits. The fixes below touch `scripts/qa/` only.

### Finding 1 -- min_k row label decoupled from the k actually passed

**Structural fix, not a guard bolted on top.** `MIN_K_SECTORS = 3` is declared
once and `MIN_K_ARM = f"min_k_sectors={MIN_K_SECTORS}"` is DERIVED from it, so
the label and the parameter can no longer be edited apart by accident. The call
site records the k it received (`min_k_passed`) and the new guard
`min_k_arm_used_the_labelled_k` compares that against the integer parsed back
out of the label.

The guard is deliberately NOT a restatement of the derivation: its
counterexample is a divergent edit to EITHER site, and both directions have
their own cell, because a cell for one licenses nothing about the other.

- **M23** call site drifts (`_k = 4`) -> KILLED.
- **M24** label drifts (`MIN_K_ARM = "min_k_sectors=9"`) -> KILLED.

RED observed directly before shipping, at `--flags --cycles 4`:

```
AssertionError: INVARIANT FAILED: min_k_arm_used_the_labelled_k -- the row is
labelled 'min_k_sectors=3' but _min_k_sector_slice received [4] -- a criterion-4
row that reports a turnover cost under the wrong k silently inverts the
ASK-1/ASK-2 ordering
```

That is the exact k=4 edit the Q/A executed, which SURVIVED before this change.

### Finding 2 -- coverage floor could not protect the number derived from it

The floor was `known >= 0.5 * len(tickers)` while the operating point is
502/513 = 97.9%. It is now pinned at **0.95**, renamed
`sector_map_covers_the_panel_at_the_published_operating_point`, and the measured
coverage is PRINTED so a reader never has to infer it from a pass/fail guard.

0.95 is chosen rather than inherited: the published ordering was measured at
97.9% and was shown by the Q/A to invert by 78.2%, so the floor must sit between
the two; 0.95 leaves ~15 tickers of headroom for ordinary lookup churn.

- **M9** total collapse (`known = 1`) -> KILLED (the old floor caught this too).
- **M9b** degradation to **78.2%**, which is **28pp ABOVE the old floor** and is
  the level that actually swaps soft_diversity's and min_k's turnover cost ->
  KILLED. This is the discriminating cell; M9 alone would have proven nothing
  about the change.

RED observed directly before shipping:

```
AssertionError: INVARIANT FAILED:
sector_map_covers_the_panel_at_the_published_operating_point -- sector coverage
is 401/513 = 78.2%, below the 95% floor -- the criterion-4 turnover ordering is
not stable across coverage this degraded, so the table must not be published
from it
```

### Finding 3 -- artifact prose contradicted its own verbatim block

`live_check_86.59.md` prose said "the cell count (22) is lower than the guard
count (23)" nine lines below a block reading "24 guards in target, 24 covered".
The block and the prose are now **regenerated together from one run**, and the
prose explains why the two counts differ in BOTH directions (some guards carry
more than one cell; some are covered transitively). The stale figures in
`experiment_results_86.59.md` and `escalation_86.59_third_conditional.md` were
swept in the same pass and each marked as the cycle-3 record it is, with a
pointer to the current numbers -- a corrected capture that leaves its siblings
stale is the same defect one file over.

### Post-fix state, re-measured

```
control --verify       -> rc=0 GREEN
control --dispersion   -> rc=0 GREEN
control --flags        -> rc=0 GREEN
coverage: 25 guards in target, 25 covered by a cell or an explicit transitive entry
KILLED 26 / 26   SURVIVED 0   UNSCORABLE 0
restore verified: sha256 unchanged (39fc81f531c91cce...)
```

Independent AST census (`scripts/qa/guardlib.py`): **CENSUS OK** -- 22
literal-named guards each named by a cell, plus 3 parameterised guards covered
by the literal prefixes the matrix declares.

`--verify` -> `sector map coverage: 502/513 = 97.9%` then
`OK: all 42 invariants hold`.

**No production file was touched.** Under this session's R2 (evidence-only
changes do not buy an evaluation cycle) these fixes are recorded here rather
than spent on a re-grade; whether to spend 86.59's one remaining attempt is the
operator's call.
