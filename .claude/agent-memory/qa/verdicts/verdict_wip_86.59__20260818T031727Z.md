STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.59
WRITTEN: 2026-08-18T03:17:27Z

# Q/A write-first record -- step 86.59 (cycle 2 per Main's disclosure)

## Plan
- A. Harness-compliance audit (5 items)
- B. Deterministic: immutable command, git scope, lint, re-run the 4 rerunnable checks
- C. Mutation / vacuity attack on the NEW guards Main claims fixed
- D. Criterion-by-criterion MET/NOT MET

## Log
(appending as established)

## A. Harness compliance (5 items) -- CLEAN so far
- research gate: research_brief_86.59_rerun.md gate_passed TRUE, 8 sources read in full (>=5), 54 URLs (>=10), recency scan performed, envelope COMPLETE. Gate verdict file research_gate_86.59_rerun_verdict.md, run wf_ff8717e8-ccf.
- mtime order: brief 2026-08-14T02:43 < contract 2026-08-17T20:56 < scripts 2026-08-18T05:13/05:14 < experiment_results/live_check 2026-08-18T05:16. OK.
- experiment_results present (15,288 B), live_check present (12,570 B), evaluator_critique present (9,164 B).
- log-last: harness_log has ONLY "Cycle 1226 -- 2026-08-12 -- phase=86.59 result=GATE-FAILED"; no result= row for this cycle. masterplan 86.59 status=pending at HEAD. OK.
- no-verdict-shopping: evidence CHANGED. Fix commit 3e75c2d6 touches scripts/qa/mutation_86_59.py, scripts/qa/rank_stability_86_59.py, experiment_results, live_check, evaluator_critique, masterplan.

## Prior-attempt / prior-verdict evidence (gathered, not a trigger)
- qa_wip.py 86.59 --spawned-at 2026-08-18T03:17:27Z: source_present=true, attempt_number=2, attempt_number_status=ok, prior_attempts=1, records_retained=2 (gauge), prior record verdict_wip_86.59__20260818T025531Z.md
- verdict_history_86_21.py --step 86.59 --evidence-only: status=ok, "1 verdict(s) from the ledger", verdicts: CONDITIONAL
- CROSS-CHECK: prior_attempts (1) == ledger rows (1). Ledger is CURRENT, not stale.

## B. Deterministic
- IMMUTABLE CMD: `bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tools/screener.py\").read()); print(\"parses\")"'` -> stdout "parses", EXIT=0.
- Step-commit py scope (derived from git show --name-only on 15a817cc + 3e75c2d6): scripts/qa/mutation_86_59.py scripts/qa/rank_stability_86_59.py. ruff --select F821,F401,F811 -> "All checks passed!" exit=0.
- Working-tree py scope (git diff --name-only HEAD -- '*.py'): backend/api/sovereign_api.py backend/services/autonomous_loop.py (PEER SESSION work, not in either step commit). ruff exit=0.
- PRODUCTION SCOPE: `git show --name-only --format="" 15a817cc | grep -E '^(backend|frontend)/'` -> exit 1 (NO MATCH). Same for 3e75c2d6 -> exit 1. CONFIRMED: zero production files in either of this step's commits.
- IMMUTABLE CRITERIA: identical json hash across 15a817cc~1 / 15a817cc / 3e75c2d6 / HEAD. Command + live_check strings identical. Criteria text byte-matches what was passed into this spawn.
- masterplan 86.117 audit_basis NOW carries 10.646 / 19.850 / 30.441 and 2.86x -- the cycle-1 Contradiction finding IS corrected at durable storage.
- NOTE: `blocked_by` is not a schema key anywhere in masterplan.json (grep -c = 0). 86.117's dependency on 86.116 is expressed in its criteria text + notes, not a structured field. "explicitly BLOCKED-BY" is prose, substantively true.

## B2. Independent re-runs -- EVERY published number reproduces EXACTLY
- `--cycles 20`: rho mean 0.9622 / min 0.9319; top-10 turnover 15.8%/day; top-5 15.8%; ZERO-turnover sessions 3 of 19; distinct 12 (same list DD,DDOG,DELL,DVA,FTNT,HPE,HPQ,HUM,MU,PANW,SNDK,ZBRA); IT 72.0%; counts {Industrials 20, IT 72, Health Care 8}; fidelity mean overlap 80%; live distinct 18 (same list); dedup 47,880 of 200,875 (23.8%); split-shaped bars 10. "OK: all 71 invariants hold".
- `--flags --cycles 20`: 15.8 / 28.4 / 22.1 / 17.9; distinct 12/22/17/14; top-sector 72/20/40/60; deltas +12.6/+6.3/+2.1pp and -52.0/-32.0/-12.0pp. "OK: all 4 invariants hold" (was 2 in cycle 1 -- the 2->4 claim reproduces).
- `--dispersion --cycles 20`: mean sigma 1m 10.646 / 3m 19.850 / 6m 30.441, ratio 2.86x (now PRINTED by the script); effective 22.6/37.0/40.4; gaps -17.4/+2.0/+15.4pp; multidim 50 of 10139 (0.493%), 5 cycles identical. "OK: all 5 invariants hold".
- `python scripts/qa/mutation_86_59.py`: control --verify/--dispersion/--flags all rc=0 GREEN FIRST; KILLED 20/20, SURVIVED 0, UNSCORABLE 0; restore sha256 9282ba866f2afc87... unchanged. Target sha on disk before AND after = 9282ba866f2afc872435bda7e84bf833e37d98429ad27bf3c2577f0f7bd19d1d.
- Cycle-1's two exact mutations now KILL (M14 -> baseline_arm_applies_no_flags; M15 -> predicates_reject_known_bad_inputs). Confirmed in my own matrix run.
- Criterion 5 independently verified: grep of the 3 flag names in backend/.env -> exit 1 (positive control: 52 `^[A-Z_]+=` lines, so the grep can match); no env file in either step commit; settings.py defaults still False / False / 0.0 / 0; screener.py has ZERO `settings.` references.
- Residual sweep for the retired sigma triple: 3 hits, ALL inside labelled retraction notes (experiment_results:256, live_check:160, masterplan 86.117 audit_basis). Correction REPLACES rather than accompanies. Positive control "30.441" present 1x/2x.

## C. NEW FINDINGS (this cycle)

### F1 [WARN] live_check section 8 coverage line does NOT reproduce
Block claims `coverage: 20 guards in target, 20 covered` AND `restore verified: sha256 unchanged (9282ba866f2afc87...)` in the same fenced block. At that exact sha the command prints **21 guards in target, 21 covered**. Verified twice: (a) my full matrix run printed 21; (b) direct AST census via mutation_86_59.guard_names_in_target() returns len 21. Spliced/stale "verbatim" capture on the criterion that was NOT MET last cycle. Direction is conservative (understates coverage) and 20 coincides with the CELL count (20 rows), which is the conflation that would mask a real gap.

### F2 [WARN, the substantive one] a poisoned BASELINE reference survives all four criterion-4 guards at the PUBLISHED cycle count
Control observed GREEN first (4 guards ran). Poisoning applied at the target script's OWN seam (`rs.replay_session`), baseline arm only, FLAG_ARMS[0] left byte-identical as ('baseline', {}).
- `momentum_52wh_tilt=True, k=0.2` @ --cycles 20 -> **SURVIVED**, all 4 guards ran. Baseline turnover 15.8% -> 20.0%, distinct 12 -> 13. Reported deltas become sector_neutral +8.4pp (published +12.6), soft_diversity +2.1pp (published +6.3), **min_k_sectors=3 -2.1pp (published +2.1pp -- A SIGN FLIP on the number ASK-1 rests on)**.
- `soft_sector_diversity w=0.05` @ --cycles 20 -> **SURVIVED** with turnover deltas reading EXACTLY as published (+12.6/+6.3/+2.1) while baseline top-sector share silently moves 0.72 -> 0.64.
- `soft w=0.15` @ --cycles 4 -> SURVIVED (baseline 13.3% -> 40.0%, distinct 7 -> 10); the SAME mutant @ --cycles 20 is KILLED. The matrix runs every cell at `--cycles 4` (mutation_86_59.py:266), so kill/survive here is cycle-count dependent -- the matrix oracle is weaker than the published run's.
Root cause: `baseline_arm_applies_no_flags` asserts the arm DEFINITION (FLAG_ARMS[0] == ("baseline", {})), which a downstream injection does not touch; `flag_arms_are_distinguishable_from_baseline` detects DEGENERACY (an arm coinciding with baseline), not FLAGGEDNESS, so it fires only by coincidence. The property criterion 4 actually rests on -- "the reference arm IS the unflagged ranking" -- has no BEHAVIOURAL observer. Same harm the guard's own detail string claims to prevent, one seam from where the guard looks.
MITIGATING: backend/tools/screener.py has ZERO `settings.` references, so flags reach rank_candidates only as explicit caller kwargs -- this is a code-edit risk in the measurement script, NOT reachable from an operator promoting ASK-1/ASK-2.
NAMED FIX (~3 lines): recompute the baseline slate via an independent no-kwargs call and assert equality, or capture the kwargs actually passed at :768-771 and assert they were empty.

### F3 [WARN] the predicate FIXTURE is itself unguarded
`_PREDICATE_FIXTURE.clear()` leaves `predicates_reject_known_bad_inputs` GREEN (verified in-process; the loop over an empty list yields `bad == []`). Combined with an always-true `_us_only`, the pair is fully blind (MUT-B2 SURVIVED). The fixture is the entire cycle-2 remediation mechanism and the load-bearing half of 4 of the 20 cells (M15/M16/M17/M18), yet it has no cell and no size assertion, and the AST census cannot see it because the fixture is DATA, not an `_ok(...)` call. Fix: `assert len(_PREDICATE_FIXTURE) >= 15` + a cell.
NOT scored as vacuity: a single-edit mutation of any assertion into inertness survives by definition; MUT-B2 is a 2-property compound.

### F4 [NOTE] the AST coverage census over-accepts by unanchored bidirectional prefix match
POSITIVE CONTROL: a genuinely new name (`a_brand_new_unrelated_guard`) IS detected as uncovered -- the census is NOT vacuous. But `check_coverage`'s `g.startswith(n) or n.startswith(g)` also accepts `sector_map_present_and_fresh`, `baseline_arm_applies_no_flags_at_the_call_site`, and single characters `s` / `p` as covered. Ironically the guard that would close F2 would be auto-accepted without a cell.

### F5 [NOTE] live_check section 2 lists 6 of 20 sessions in the per-cycle table; the reproducing command is stated and the full output reproduces, and the full 18-name live distinct list is given. Presentation, not absence.
### F6 [NOTE] `blocked_by` is not a masterplan schema key (0 repo-wide). "86.117 explicitly BLOCKED-BY 86.116" is prose in 86.117's criterion 4 + notes; substantively true, not structurally enforced.
### F7 [NOTE] live_check ~:197 -- the cycle-2 correction blockquote runs into the body sentence ("The commit-scoped command can. The only files it authors are ..."), so body text renders inside the quote. Cosmetic.
### F8 [advisory only] Main's spawn-prompt claim "Residual-sweep for the old triple across all three files returns 0" does not reproduce literally (3 hits) -- all 3 are labelled retractions, so the INTENT (0 load-bearing uses) is true and verified. Not an artifact defect; the claim is not in any durable artifact.

## Code-review heuristics (5 dimensions) -- no BLOCK
- secret-in-diff: none (grep over both step commits' scripts/qa diff -> exit 1).
- command-injection: `subprocess.run([sys.executable, ...])` list form, shell=False -> safe per negation list. `eval(expr)` at rank_stability:171 operates on a FIXED module-level literal table with no external input (`# noqa: S307`) -> NOTE only; a (callable,args,expected) fixture would be less fragile and would also close F3.
- disk-mutating checker wiring: `mutation_86_59.py` writes to a real repo file, but grep shows it is NOT wired into any hook, settings.json or CI -- only invoked manually. OK.
- trading-domain: zero production files, no execution path, no kill-switch/stop-loss/perf-metrics surface touched.
- LLM-evaluator anti-patterns: this is a cycle-2 spawn on CHANGED evidence (fix commit 3e75c2d6); not verdict-shopping. Simultaneous-presentation applied: read updated experiment_results -> updated critique -> prior verdict -> re-derived independently.

## D. Criterion-by-criterion
1. MET -- measured, command stated, reproduces EXACTLY; premise partially refuted and reported anyway.
2. MET (degenerate, honestly disclosed) -- no new/reweighted term, so nothing to justify. RULING on Main's explicit question: criterion 2 does NOT require the term to be built here. (a) criterion 4 mandates measuring existing flags BEFORE new code so the step does not rebuild an existing mitigation, and all three MOVE the slate; (b) criterion 2 itself demands DSR/PBO, whose gates read a table this step MEASURED at 38% duplicate keys -- a gate number off that data would be fabricated; (c) the obligation is carried forward verbatim in 86.117's criterion 4 including the withhold-if-86.116-open clause; (d) criterion 1 explicitly licenses refuting the premise, and it partly did.
3. MET -- N=20 stated, 12 distinct / 100 slots (12.0%), IT 72.0%, live "before" 18 distinct.
4. MET -- table reproduces exactly under my own run; measured before any new code; window-sensitivity disclosed (6-cycle sign flip). RESIDUAL F2 on its tamper-resistance.
5. MET -- verified four independent ways.
6. MET (degenerate, honestly characterised) -- no new behaviour to disable; "zero production files in BOTH step commits" verified at commit level, which ENTAILS an unchanged live candidate list and is strictly stronger than the parity asked for. The ORACLE element is not demonstrated because there is nothing to disable.
7. MET WITH NAMED RESIDUALS -- 21/21 guards covered (re-derived by me), 20/20 KILLED, control GREEN first on all 3 modes, sha-256 restore verified, census positive-controlled. Cycle-1's two unkillable guards are gone and their replacements kill. Residuals F1 / F3 / F4.

## NOTE on contract-vs-delivery
Contract PLAN P3 (standardise behind a new default-OFF flag), P4 (criterion-2 OOS + DSR/PBO) and P6 (parity oracle) were NOT executed. The artifacts explain WHY (86.116 blocks the gates; criterion 4 says do not rebuild an existing mitigation) but never name it as a divergence from the contract's own plan. Judged legitimate course-correction on measured evidence, not scope-dodging -- but a reader diffing contract against results would see three unexecuted plan items with no explicit reconciliation.

## VERDICT REASONING
Worst-of-3-lenses (P1 money path): correctness PASS; does-it-reproduce CONDITIONAL (F1); scope-honesty PASS-with-NOTE. min = CONDITIONAL.
Severity dispatch: no BLOCK (no sole-coverage vacuity -- all guards are killable and killed). Worst finding is WARN (F2 executed surviving mutant with a sign flip on ASK-1's number; F1 non-reproducing number in a verbatim block; F3 unguarded fixture) -> CONDITIONAL.

COMPLETED: 2026-08-18T03:35:26Z
