# evaluator_critique -- step 86.118

## Verdict ledger

| cycle | verdict | run id | recorded |
|---|---|---|---|
| 1 | **CONDITIONAL** | `wf_29efd777-f0f` | 2026-08-18T09:53:03Z |

---

## Cycle 1 -- CONDITIONAL -- `wf_29efd777-f0f`

*Transcribed verbatim from the captured Workflow return in the same turn it
landed. Main records the verdict and never authors it.*

**verdict:** `CONDITIONAL`  **ok:** `false`  **certified_fallback:** `false`  **harness_compliance_ok:** `true`
**attempt_number:** `2` of `5`; **consecutive_conditionals:** `0`; **would_auto_fail:** `false`; **budget_exhausted:** `false`
**judge_was_told_consequence:** `false`
**research_needed:** `false`

**reason:**

> 5 of 7 immutable criteria MET, 2 PARTIAL. Everything I re-derived reproduced, some byte-for-byte: my own full-suite run gave `8 failed, 3684 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings in 518.53s` against the author's `...514.34s` with all 8 FAILED names matching the live_check §5 table 8/8 (criterion 6); I replicated `scripts/qa/mutation_86_118.py` myself (7/7 controls GREEN at the exact collected counts 44/18/25/7/13/8/77, KILLED 13/13, SURVIVED 0, UNSCORABLE 0, 7/7 SHA-256 restores, tree clean after); guardlib_selftest gave `cases passed: 45 FAILED: 0`; the pinned census reproduced exactly as {'dict':720,'str':126,'list':13,'none':24}; and the headline reproduced independently (Settings() reads True/True from backend/.env:83-84 while Settings.model_fields defaults are False/False). I ran the ONE mutation the author declared but deliberately did not automate -- flipping the DECLARED default in-memory, no file touched -- and BOTH re-aimed assertions went RED with the named assertion in the traceback (control: 2 passed), so the `Settings.model_fields[...].default` re-aim is NOT a tautology and NOT a change of subject (Main's item 1: sound). Item 2 (effort pin xhigh->max) is tracking a documented operator decision, not drift: .claude/settings.json:2 is "max" and CLAUDE.md:59 independently documents the 2026-08-04 raise, and the guard stays an equality assert. Item 3 (the single --deselect) is legitimate and I verified rather than accepted it: without it test_phase_75_17_verification_paths.py gives `1 failed, 44 passed` and the one FAILED is exactly the deselected name. Item 4 is honest scoping -- criterion 6 explicitly contemplates a non-green suite named with dispositions. Criterion 4 verified by derivation, not by claim: assert-line accounting over the commit is 4 removed / 5 added with a replacement for every removal and zero xfail/skip/deleted-assertion/widened-tolerance/pinned-seed. WHAT CAPS IT. (a) Criterion 2 requires each failing test classified into STALE-EVIDENCE / PRODUCT-DEFECT / ORDERING-ARTIFACT; live_check §5 leaves `test_phase_62_4_sentinel::test_infra_path_distinct_exit` EXPLICITLY unclassified ("Not yet classified STALE vs PRODUCT"), and six finer labels (ENV LEAKAGE, CLASSIFIER FALSE POSITIVE, PROXY ASSERTION, LIFECYCLE META-TEST, SUPERSEDED POLICY, census-vs-live-artifact) are never mapped to the three named buckets. (b) Criterion 5 requires "identify the shared state responsible"; the isolation half is proven and I reproduced it (test_phase_86_6_subprocess_channel passes alone -- 18 passed -- and is in my own full run's FAILED list), but the shared state behind the one ORDERING-ARTIFACT is never identified, though contract P2 planned exactly that, and the contract's opposite-direction masking anomaly (test_portfolio_swap passes when the 19 run together, fails alone and in full) is dropped from the evidence artifact. Both criteria are substantially, not materially, uncovered -- hence CONDITIONAL rather than FAIL. Harness compliance clean on all 5: research gate PASSED (11 sources read in full, 60 URLs, recency scan at brief:150, envelope COMPLETE, gate_passed true) with mtimes research 08:49:27 < contract 08:53:31 < experiment_results 11:33:35; no `phase=86.118 result=` row in harness_log and masterplan 86.118 still `pending`; attempt 1, no verdict-shopping. Immutable command reproduced: stdout `parses`, exit 0. Lint gate on the commit-derived 10-file .py scope (non-empty asserted): `All checks passed!` exit=0. No unintended production change -- the commit touches 6 backend/tests files, 4 scripts/qa files, masterplan (purely additive, only removed line is `-}`) and 2 handoff artifacts; zero production backend modules.

**violated_criteria:**
`criterion_2_classification_incomplete`, `criterion_5_shared_state_not_identified`

### violation_details

**1. Missing_Assumption**

- *action:* read handoff/current/live_check_86.118.md §3 classification table + §5 residual table, then cross-checked against criterion 2's three named buckets
- *state:* Row 7 `backend/tests/test_phase_62_4_sentinel.py::test_infra_path_distinct_exit` is labelled 'exit-code drift' in §3 and §5 states verbatim 'Not yet classified STALE vs PRODUCT'. It reproduces as FAILED in my own full-suite run, so it is a live member of the set criterion 2 governs. Six further rows carry labels outside the three buckets (ENV LEAKAGE, CLASSIFIER FALSE POSITIVE, PROXY ASSERTION, LIFECYCLE META-TEST, SUPERSEDED POLICY, census-vs-live-artifact) with no stated mapping.
- *constraint:* criterion 2: 'each failing test is classified with evidence into STALE-EVIDENCE ..., PRODUCT-DEFECT ..., or ORDERING-ARTIFACT ..., and the classification for each cites what was read or run to reach it'. Evidence is cited for every row; a BUCKET is not assigned for row 7 and is not derivable for six others.

**2. Missing_Assumption**

- *action:* read live_check_86.118.md §6 (criterion 5) and contract_86.118.md P2; independently re-ran the order-dependent test alone (`18 passed`) and confirmed it FAILED in my own full-suite run
- *state:* The one ORDERING-ARTIFACT found -- `test_phase_86_6_subprocess_channel.py::test_the_optin_IS_honoured_so_a_real_window_remains_possible` -- is proven to pass in isolation and fail in the full run, but no shared state is named; §6 substitutes a scope argument ('outside this step's named 12 files', handed to 86.119). Contract P2 had planned 'Identify the shared state behind the single victim'. Separately, the contract's recorded opposite-direction anomaly (test_portfolio_swap passes when the 19 are run together but fails alone and in the full suite -- itself order-dependence evidence) appears nowhere in the live_check.
- *constraint:* criterion 5: 'the ORDERING-ARTIFACT class is proven rather than asserted -- show the test passing in isolation AND failing in the full run, and identify the shared state responsible'. The first two clauses are discharged; the third is not, for the one member of the class this step produced.

**checks_run:**

- harness_compliance_audit_5_item
- immutable_verification_command_exit_0
- scope_derivation_git_show_name_only
- ruff_lint_gate_F821_F401_F811
- full_backend_suite_independent_reproduction
- mutation_matrix_independent_replication
- guardlib_selftest_45_cases
- independent_mutation_declared_default_in_memory
- independent_mutation_75_19_sibling_bound_reference
- environment_contingency_probe_M5
- deselect_scope_verification
- census_pin_reproduction
- env_leakage_headline_reproduction
- classifier_blast_radius_census_1153_commands
- bootstrap_oracle_evasion_probe
- archive_resolver_breadth_check
- assert_line_weakening_accounting
- filed_defect_steps_verification
- isolation_reproduction_order_dependent_test
- research_gate_envelope_check
- code_review_heuristics
- evaluator_critique

**notes:**

> SEQUENCE / ATTEMPT EVIDENCE. `python scripts/qa/qa_wip.py 86.118 --spawned-at 2026-08-18T09:34:46Z` -> source_present=true, attempt_number=1, attempt_number_status="ok", attempt_number_is_lower_bound=false, prior_attempts=0, prior_records=[]. `python scripts/qa/verdict_history_86_21.py --step 86.118 --evidence-only` -> status="no_rows_for_step", verdicts=(none). Cross-check prior_attempts (0) vs ledger rows (0): equal, so no staleness signal. Main's disclosure (research gate wf_628cc28c-e10 PASSED; no prior Q/A) is consistent with both sources and is advisory only.
>
> WRITE-FIRST RECORD (evidence for any respawn, never a verdict): /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_86.118__20260818T093446Z.md -- carries every command, exit code, matrix cell and criterion verdict below. No write was blocked.
>
> WHAT WOULD LIFT THE CAP (cheap, all four): (1) assign row 7 a bucket, or state explicitly that criterion 2 cannot be discharged for it and why, as a NAMED gap rather than a footnote; (2) state the mapping from the six finer labels to STALE-EVIDENCE / PRODUCT-DEFECT / ORDERING-ARTIFACT; (3) bisect the polluter behind the single ORDERING-ARTIFACT (it drives a subprocess via run_smoke, so inherited os.environ / CWD is the first place to look), or record criterion 5's shared-state clause as deliberately deferred to 86.119 and carry the contract's masking anomaly forward; (4) qualify the "13/13 KILLED" headline with the M5 finding below.
>
> MY OWN FINDING, NOT IN THE ARTIFACTS -- M5's kill is ENVIRONMENT-CONTINGENT. Simulating M5 exactly (base pin removed from _make_settings, explicit overrides still winning): with backend/.env as deployed the unpinned fixture flag reads True and the mutant is KILLED; with PAPER_RISK_JUDGE_REJECT_BINDING=false the flag reads False and the same mutant SURVIVES. The matrix reports 13/13 unqualified. In a step whose entire headline is that the suite is not hermetic, its own criterion-7 evidence inheriting that property is worth stating. This is NOT a criterion-7 failure -- every mechanic the criterion names (control GREEN first, equal collected counts, NAMED test failing, byte-identical restore) is satisfied as run, and the sibling declared-default assertion is killed environment-INDEPENDENTLY.
>
> I EXTENDED THE MATRIX ON ITS ONE GAP AND THE GUARD HELD. No cell NAMES the 75_19 sibling, so "one line held two tests red" was proven on only one consumer. I neutered ONLY the `alternative-arm-satisfied` return value in preflight_verify_masterplan's BOUND fp_reason reference (in-memory, nothing on disk) and `test_phase_75_19_preflight_calibration::test_live_masterplan_is_currently_clean` went RED. Disclosure: my first probe patched the wrong module identity (`sweep_absent_verification_paths` vs `scripts.qa.sweep_absent_verification_paths`, plus a bound-name import) and came back clean; I chased the clean answer instead of banking it.
>
> NOTE-LEVEL, no verdict effect. (a) The `||` classifier's blast radius over the whole live masterplan is exactly 2 token-instances, both the same token in step 86.31 -- so the repair excuses precisely the one command it was written for (1,153 verification commands walked, 14 contain `||`). Two shapes I constructed ARE wrongly excused -- `grep -q x CLAUDE.md || test -f MISSING` and `test -f MISSING && echo ok || cat CLAUDE.md` -- with zero live instances today; a latent false-negative window, not a live one. (b) Main's stated 3-row discrimination table reproduces only when the token sits in the arm NOT followed by `||`; in first-arm position a PRE-EXISTING negation_patterns rule returns 'absence-asserted' before the new block is reached. Not this step's code, but the claim's wording is narrower than stated. (c) The new bootstrap oracle has real rejection power (both my controls rejected) but three constructed evasions slip: `launchctl  bootstrap` (double space), `export HINT='launchctl bootstrap ...'` + `eval "$HINT"` (the `^VAR=` assign regex does not match `export VAR=`, so the hint var is never recorded -- the false-negative direction), and `LC=launchctl` + `$LC bootstrap`. Incompleteness on a hygiene guard with a genuine behavioural core (M7/M7b/M7c/M7d all kill), and strictly better than the "not a comment => executed" oracle it replaced. (d) contract:92 and research_brief:315 cite `19 failed / 3635 passed` while live_check cites 3672/3673; fully explained -- the peer session's two untracked test files collect 38 tests (measured), 3635+37=3672 and 3635+38=3673 -- and the contract predates the runs. (e) The live census is now 1136 vs the live_check's 1132; the masterplan moved by 4 during/after the step, which CONFIRMS the pin rationale. (f) The commit swept the peer session's masterplan filings 86.120/86.121/86.122 in under this step's subject via `git add -A`; purely additive, no content loss. (g) `1 xpassed` reproduces in my run -- the silent-XPASS finding is real and correctly filed to 86.124 rather than fixed here.
>
> UI GATE: not applicable -- no frontend/** in scope and the step makes no UI claim, so no Playwright capture was required or taken.
>
> CREDIT WHERE DUE, since it bears on anti-rubber-stamp: the step reports four defects the matrix caught in its OWN work (including a cell whose SURVIVAL was the finding), discards a plausible-and-wrong consolidation with a measurement (row 18 fails identically with both promotions neutralised, `1 failed in 0.23s` vs `0.24s`), and leaves eight tests red with named dispositions rather than silencing them. I found no wrong claim in the artifacts -- the two caps are clauses not delivered, not statements that failed to reproduce.


---

## Cycle-1 Follow-up -- what Main changed in response

Both caps were on **immutable criteria that were not discharged**, not on
instrumentation polish -- criterion 2 left one row explicitly unclassified with
six labels unmapped, and criterion 5 left its third clause ("identify the shared
state responsible") unanswered. Discharging them required measurement, not
rewording.

### Criterion 2 -- every row now maps to one of the THREE named buckets

`live_check_86.118.md` §3 opens with an explicit mapping table from each finer
label to STALE-EVIDENCE / PRODUCT-DEFECT / ORDERING-ARTIFACT, with the reason
for each. The finer label is kept because it is what makes a row actionable, but
it is now stated as a SUB-CLASS rather than offered as a substitute.

Two of the finer labels map to **PRODUCT-DEFECT**, and saying so matters:
CLASSIFIER FALSE POSITIVE (the code under test was genuinely wrong and was
fixed, not the test) and PROXY ASSERTION (the oracle was wrong on any input, not
merely out of date).

### Row 7 classified by DRIVING the sentinel

```
$ SENTINEL_TEST_BQ_FAIL=1 bash scripts/away_ops/sentinel.sh
"gates_failed": ["metered_source_unavailable", "flags_match_tokens"]
"warnings": [..., "unauthorized true flags: PAPER_SYNTHESIS_INTEGRITY_ENABLED"]
```

`sentinel.sh:159-160` exits 2 only when `gates_failed` is a **subset** of the
infra set. The injection adds its infra gate as designed, but a second,
NON-infra gate is already failing, so the subset test fails and it exits 1.
**STALE-EVIDENCE, env-dependent -- a third instance of the §2 headline class.**

**And it surfaced an operational finding that is not a test problem.** Run with
no injection at all the sentinel exits **1**, `ok: false`, because
`backend/.env:88` sets `PAPER_SYNTHESIS_INTEGRITY_ENABLED=true` with no matching
authorization token -- exactly the condition `flags_match_tokens` exists to
catch. This step does not touch it (flag promotion is operator-gated and
`backend/.env` is not written here); it is raised for the operator, and the
sentinel's own test being red is plausibly why an `ok: false` went unnoticed.

### Criterion 5 -- the shared state, identified rather than deferred

It is **not** in-process mutable state and there is no polluter test. From the
full-run traceback the child was **SIGKILLed at a 120s timeout** after reaching a
real preflight against `http://localhost:8000` with `binary:
/Users/ford/.local/bin/claude`, `rail_state: ON`. Timed both ways:

| context | result |
|---|---|
| alone | `1 passed in 6.80s` |
| full suite | `TimeoutExpired` at 120s, `returncode: -9` |

**The shared resource is WALL-CLOCK on a real external dependency** -- the
operator's running backend and the real `claude` CLI. Under whole-suite CPU
contention the same call exceeds its ceiling by more than 17x. Nothing another
test *wrote* is responsible; what is shared is the machine, the live backend and
the CLI. Consequently the usual order-dependence remedy (clean the shared state
-- Luo FSE'14 F.10: 74%) does not apply here, and that is stated.

### The opposite-direction anomaly, restored

The contract's masking observation (`test_portfolio_swap` passing when the 19 ran
together while failing alone and in the full suite) had been dropped from the
artifact. It is restored, and the wall-clock explanation makes both directions
consistent rather than contradictory. It is recorded as an **open observation,
not a conclusion** -- measured once during the contract's isolation sweep and not
re-measured since the repairs.

No production file was touched by this follow-up; no test was weakened; the
post-work suite counts are unchanged at `8 failed, 3684 passed`.
