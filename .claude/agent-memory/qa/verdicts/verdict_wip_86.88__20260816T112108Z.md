STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.88
WRITTEN: 2026-08-16T11:21:08Z
COMPLETED: 2026-08-16T11:35:43Z

# Q/A cycle 4 write-first record for step 86.88

Spawn started. Read qa.md in full. Beginning harness-compliance audit.

Prompt states: cycle 4, sequence [CONDITIONAL x3] as DATA (caller computes escalation).
Commits: 03386529 c1; 4e01f3b6 c2; a2ac7cca c3; 617ba2c0 CYCLE 4 (audit target).

Main's self-flagged weak points:
1) 5 of 12 cells exist only because they survived a prior cycle.
2) Real BigQuery writer NOT driven end-to-end.
3) Checker's syntactic rule cannot see an intermediate alias.

Judge-these questions posed by Main: A) third duplicated site guarded by one cell?
B) parametrised exactness test circular (derives key list from shipped constant)?
C) checker vacuous in other direction (provenance twice in SAME path)?
D) save_report-capturing stub vs real BQ writer.

## Findings log (appended as established)

### Prior-attempt / sequence evidence
- `qa_wip.py 86.88 --spawned-at 2026-08-16T11:21:08Z`: source_present=true,
  attempt_number=4 (status "ok", is_lower_bound=true), prior_attempts=3,
  records_retained=4 (GAUGE, not counter), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.88 --evidence-only`: status
  `no_rows_for_step`, verdicts `(none)`. CROSS-CHECK: attempt_number (4) >
  ledger verdict count (0 rows) => THE LEDGER IS STALE for this step.
  sequence: UNKNOWN from the authoritative source. Main's advisory disclosure
  says [CONDITIONAL x3]; recorded as advisory only, not as the sequence.

### Harness compliance (5 items)
1. research gate: research_brief_86.88.md exists, 38,587 bytes, mtime
   2026-08-16T12:08:43 -- BEFORE contract (12:11:20). OK
2. contract before generate: contract 12:11:20 < test file 13:18:31 <
   checker 13:18:50 < experiment_results 13:20:39. OK
3. experiment_results_86.88.md present (20,064 bytes), cycle-4 follow-up
   section present. OK
4. log-last: `grep -F "phase=86.88" handoff/harness_log.md` -> pending check
5. no-verdict-shopping: evidence CHANGED (commit 617ba2c0 modifies the test
   file +69 and the checker +17). OK

### Deterministic
- IMMUTABLE COMMAND `python scripts/qa/verify_lite_risk_seam_86_86.py`
  => EXIT 0, "checks emitted: 10 (PASS 10 / FAIL 0)", RESULT: OK. REPRODUCED.
- Direct `pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q`
  => 78 passed, 1 warning, exit 0. REPRODUCED (matches artifact's 78).
- Baseline sha256 autonomous_loop.py
  c68ebad5c45f281a88d17ec96c6061fa5a05b5f4b36d91c8096db384a4fe6799
- Cycle-4 commit 617ba2c0 touched NO production module: only
  backend/tests/..., scripts/qa/..., and handoff artifacts. Product code is
  unchanged since a2ac7cca (cycle 3).

### INDEPENDENT MUTATION MATRIX (my own; in-memory sys.modules injection,
### repo tree never written -- tree_untouched=true on every cell)
CONTROL: 78 passed | checker exit 0 -> GREEN (harness proven faithful:
matches the direct pytest run exactly)

| cell | mutation | pytest | checker | result |
|---|---|---|---|---|
| A_prov_2x_claude | move the GEMINI provenance block into the CLAUDE dict (count still 2, both on ONE path) | 1 failed | **exit 0 -- BLIND** | KILLED by the suite only |
| B_inline_only | neuter ONLY the inline `if risk_dict == _LITE_RISK_DEFAULT` in `_lite_position_pct` (helper untouched) | 4 failed | 0 | KILLED |
| B2_helper_only | `_lite_judge_produced_no_verdict` -> False (inline untouched) | 4 failed | 0 | KILLED |
| B3_helper_true | helper -> True (over-fire; the retired M10) | 5 failed | 0 | KILLED |
| D_persist_claude_only | drop persisted provenance @ CLAUDE only | 1 failed | exit 1 | KILLED |
| E_persist_gemini_only | drop persisted provenance @ GEMINI only (c3 survivor) | 1 failed | exit 1 | KILLED |
| F_pin_false_claude | pin persisted provenance False @ CLAUDE (the un-run mirror of M17) | 1 failed | 0 | KILLED |
| G_pin_false_gemini | pin persisted provenance False @ GEMINI (M17) | 1 failed | 0 | KILLED |
| H_superset_tolerant | `all(risk_dict.get(k)==v ...)` (M14) | 1 failed | 0 | KILLED |
| I_ignore_risk_limits | per-key subset ignoring the NESTED `risk_limits` | 2 failed | 0 | KILLED |
| **J_helper_bypassed** | **production expression stops calling the helper: `"judge_verdict_absent": risk_dict.get("reasoning") == _LITE_RISK_DEFAULT["reasoning"]`** | **78 passed** | **exit 0** | **SURVIVED** |

**J IS A REAL SURVIVOR.** The cycle-4 class-wide exactness test
(`test_the_equality_is_EXACT_over_EVERY_key_not_just_one`, line 1291) imports and
drives `_lite_judge_produced_no_verdict` DIRECTLY. It therefore pins the CLASS on
the HELPER, while the value that actually lands in the record is produced by a
separate expression at `_build_lite_risk_assessment`. Replace that expression with
any weaker rule that agrees on the three point-inputs the record tests use and the
whole suite plus the checker stay green. This is Main's own recurring class
("a guard covering N sites at once cannot see a regression in one") recurring at
the NEXT seam, and it directly contradicts known_weak_point 3's claim that
"the runtime value-equality guard covers it" for the intermediate-alias case.

**Answer to Main's question C: YES, the checker IS vacuous in that direction.**
Cell A moved the Gemini block into the Claude dict; `len(prov) == 2` still held and
the checker exited 0. It is a COUNT of the string constant anywhere in the module,
with no function/path attribution. Mitigated by a genuine behavioural guard
(`test_gemini_route_provenance_also_reaches_the_PERSISTED_payload` failed), so
WARN-level, not sole coverage.

**Answer to question B: the circularity is real but narrow.** Deriving the key
list from the shipped constant means a key REMOVED from `_LITE_RISK_DEFAULT`
silently shrinks the test's coverage. In practice the three subscripted keys
(`risk_level`, `risk_limits`, `recommended_position_pct`) would KeyError elsewhere,
so the exposure is the two non-subscripted keys. NOTE-level. The far bigger
problem is not circularity, it is that the parametrisation is attached to the
wrong subject (finding J).

### SECOND SURVIVOR -- the intermediate-alias case Main claimed was covered
| cell | mutation | pytest | checker | result |
|---|---|---|---|---|
| **L_alias_via_helper** | `"judge_verdict_absent": _lite_judge_produced_no_verdict({**risk_dict, "reasoning": _LITE_RISK_DEFAULT["reasoning"]})` -- STILL calls the helper, pre-normalises the ARGUMENT | **78 passed** | **exit 0** | **SURVIVED** |
| C0 route Claude no-JSON | `dict(...)` -> `{**default, "reasoning": ...}` | 3 failed | 0 | KILLED |
| C1 route Claude exception | same, per-route | 1 failed | 0 | KILLED |
| C2 route Gemini no-JSON | same, per-route | 2 failed | 0 | KILLED |
| C3 route Gemini exception | same, per-route | 1 failed | 0 | KILLED |
| K drop the in-memory key (M8) | | 5 failed | 0 | KILLED |
| M13-analog ignore `decision` | | 2 failed | 0 | KILLED |
| M5-analog over-fire on the helper | | 5 failed | 0 | KILLED |
| CHK 5th whole-dict route added | | 78 passed | **exit 1** "expected 4 ... found 5" | KILLED (checker SOLE detector) |
| CHK one route inlined away | | 78 passed | **exit 1** "expected 4 ... found 3" | KILLED (checker SOLE detector) |
| N1 (the step's own mutant) on the SHIPPED tree | | 2 failed, 76 passed | 0 | KILLED -- criterion 3 REPRODUCED |

Known_weak_point 3 says "the checker's syntactic rule still cannot see an
intermediate alias; **the runtime value-equality guard covers it**". MEASURED
FALSE: cell L is exactly an intermediate alias and the runtime guard does not see
it, because that guard calls the helper with its OWN arguments and never observes
the argument the production site passes.

### BEHAVIOURAL DIFFERENTIAL -- neither survivor is equivalent
`_build_lite_risk_assessment` -> `judge_verdict_absent`:

| input | shipped | L_alias | J_bypass |
|---|---|---|---|
| whole default (judge produced NOTHING) | True | True | True |
| REAL judge: defaults except its OWN reasoning | **False** | **True** | False |
| REAL judge: defaults except decision=REJECT | **False** | False | **True** |
| REAL judge: pct 7.0 + own reasoning | False | False | False |
| REAL judge: default reasoning but pct 7.0 | **False** | False | **True** |

Under L, a judge emitting APPROVE_REDUCED / 3.0 / MODERATE / standard limits with
its own reasoning sentence -- an entirely ordinary output for the shipped prompt
schema -- is persisted as "the judge produced nothing". That is the provenance
collapse this step exists to close, in the direction
`test_a_real_judge_verdict_is_NOT_recorded_as_absent` was written to protect, on
an input it does not use. NOT equivalent mutants.

### Criterion 7 -- INDEPENDENTLY RE-DERIVED (not read)
Pre-fix module (03386529~1) vs shipped, all 7 disclosure-table inputs, real
`decide_trades`, `paper_risk_judge_reject_binding` both True and False.
Order outcomes IDENTICAL on 7/7 x 2 states (0.0 -> no buy; 3.0/ABSENT/None/whole
default -> 1 buy; '0' and 'high' -> no buy). ALL SAME = True. MET.

### Claim audit -- Main's cycle-4 numbers REPRODUCE
Independently re-ran 7 of the 12 shipped cells; failed/passed counts EXACT matches:
M1 2/76, M8 5/73, M14 1/77, M15 1/77+checker1, M16 1/77+checker1, M17 1/77,
CONTROL 78. Every one of the 12 matrix rows sums to 78 = the shipped tree, so the
matrix was run against the shipped suite (the cycle-2 defect does not recur).
Test-count arithmetic across all four cycles reproduces: `def test_` counts
51/61/64/66/67 vs collected 62/72/75/77/78 -- deltas +10/+3/+2/+1 agree exactly.
Pre-fix suite refs to `_run_claude_analysis|_run_gemini_analysis` = **0**
(shipped = 7), corroborating criterion 1's root cause independently.

### Question D -- the save_report stub, CONTRACT-TESTED against the real type
Drove the real `_run_claude_analysis` -> real `_persist_analysis` with the
capturing stub, then bound the captured kwargs to the REAL
`BigQueryClient.save_report` signature:
- 17 kwargs captured; `set(captured) - set(real params)` = **[]**
- `inspect.signature(BigQueryClient.save_report).bind(...)` = **OK**
- the real `save_report` has **no** `**kwargs`, so a bogus kwarg WOULD be a
  production TypeError -- and none is present.
- provenance present in the persisted `full_report`; named columns
  rj_decision='APPROVE_REDUCED', risk_level='MODERATE', pct=3.0.
So the stub is NOT hiding a TypeError. Residual (NOTE): nothing BELOW
save_report (BQ serialisation / schema acceptance) is exercised.

### live_check_86.88.md IS STALE FOR CYCLE 4
`git log -- handoff/current/live_check_86.88.md` -> last touched at **a2ac7cca
(cycle 3)**; commit 617ba2c0 does NOT include it. Its verbatim capture says
`checks emitted: 9  (PASS 9 / FAIL 0)` while the shipped immutable command emits
**10**, and `CONTROL: 77 passed` while the shipped tree is **78**. This is the
same class cycle 3 was CONDITIONAL for (finding 2, "live_check was never
regenerated"), recurring at cycle 4. masterplan `verification.live_check` names
this file as the required evidence artifact.

### Criterion 8 audit
No `.env` / settings / masterplan file in the step diff (`git diff --name-only
03386529~1 617ba2c0 | grep -iE '\.env|settings|masterplan'` -> NONE). Only two
checker lines removed across the whole step: `unexpected = keys - RETAINED_KEYS`
became `... - {WHOLE_DICT_COPY}` -- a relaxation PAIRED with a strictly tighter
new assertion (`len(copies) == 4`), which I proved goes RED both ways (5 routes
and 3 routes). Zero assertions removed from the test file. Cycle 4 replaced
`test_the_equality_is_EXACT_not_a_subset_match` with a strictly stronger
parametrised version covering the same `reasoning` case. NOT a loosening.

### Other deterministic gates
- ruff F821,F401,F811 on the DERIVED scope (`git diff --name-only 03386529~1
  617ba2c0 -- '*.py'` = 3 files, non-empty): "All checks passed!", exit 0.
- runtime smoke: `import backend.services.autonomous_loop` OK; backend :8000
  /api/health -> {"status":"ok",...}; ast.parse OK on all 3 files.
- frontend gates 1b/1c NOT required: the step diff touches no `frontend/**`.
  The uncommitted `sovereign_api.py` + 5 frontend files have mtimes of
  **2026-08-14**, two days before this step's work (2026-08-16) -- pre-existing,
  out of scope, NOT introduced by this cycle.
- log-last: `grep -cF "phase=86.88" handoff/harness_log.md` -> **0** (positive
  control `phase=86.86` -> 1). masterplan 86.88 status = **pending**. Correct order.
- evaluator_critique_86.88.md ends at "Cycle 3 verdict: CONDITIONAL" -- no
  pre-authored cycle-4 verdict.
- ALL THREE FILE sha256s byte-identical before and after my entire mutation
  matrix (autonomous_loop c68ebad5..., tests 4ac05365..., checker 1a46b9d7...).

## VERDICT: CONDITIONAL
All 8 immutable criteria MET; harness compliance clean; no unintended production
change. Two blocking-quality findings, both fixable and named:
1. two NON-EQUIVALENT mutants (J, L) survive the shipped 78-test suite AND the
   shipped 10-check command at the production expression that computes
   `judge_verdict_absent`; the class-wide exactness guard is bound to the helper,
   not to the production call site or its argument. WARN (a genuine behavioural
   guard on three point inputs coexists, so not sole-coverage vacuity).
2. `live_check_86.88.md` was not regenerated for cycle 4 and now contradicts the
   shipped command (9 vs 10 checks) and the shipped tree (77 vs 78 tests).
Plus NOTEs: the checker's provenance count is path-blind (cell A), the exactness
key list is derived from the constant it guards, and nothing below save_report
is exercised.

