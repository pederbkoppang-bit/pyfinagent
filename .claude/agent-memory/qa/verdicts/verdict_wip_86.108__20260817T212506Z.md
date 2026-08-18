STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.108
WRITTEN: 2026-08-17T21:25:06Z

## Spawn context
- Launched via Workflow rail (qa-verdict.js), agentType qa, model opus, effort max.
- Prior WIP records visible on disk for 86.108: 20260817T204157Z (9,180B),
  20260817T210432Z (17,235B). Will run qa_wip.py + verdict_history for the
  authoritative evidence rather than counting files.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable cmd, git scope, ruff on DERIVED scope, pytest, runtime smoke
C. Claim auditing / guard-vacuity / mutation matrix independent re-run
D. Criterion-by-criterion MET / NOT MET

(findings appended below as established)

## Attempt / sequence evidence
- qa_wip.py 86.108 --spawned-at 2026-08-17T21:25:06Z: source_present=True,
  attempt_number=3 (status ok, is_lower_bound=True), prior_attempts=2,
  records_retained=3 (gauge, incl. mine), records_pruned_known=None.
- verdict_history_86_21.py --step 86.108 --evidence-only: status=ok,
  "2 verdict(s) from the ledger", verdicts: CONDITIONAL -> CONDITIONAL.
- CROSS-CHECK: prior_attempts (2) == ledger rows (2) -> ledger NOT stale.

## A. Harness compliance (5 items)
1. research gate: research_brief_86.108.md exists, 52,295B, envelope
   brief_status=COMPLETE, gate_passed=true, external_sources_read_in_full=15,
   urls_collected=35, recency_scan_performed=true, §"Recency scan (last 2
   years, 2024-2026) — PERFORMED". OK.
2. order: brief mtime 20:43:25 < contract 20:53:38 < code 23:21-23:22 <
   experiment_results 23:24:32. OK.
3. experiment_results_86.108.md present (12,039B). OK.
4. log-last: grep -F 86.108 handoff/harness_log.md -> 0 hits; masterplan
   86.108 status=pending. OK (not yet logged/flipped).
5. no verdict shopping: evidence CHANGED since prior spawns (prior WIPs
   20:41 / 21:04; artifacts + code re-written 23:20-23:24). OK.

## B. Deterministic
- IMMUTABLE CMD: `ast.parse(orchestrator.py)` -> "parses", EXIT=0. REPRODUCED.
- Derived scope `{git diff --name-only HEAD -- '*.py'; git ls-files --others
  --exclude-standard -- '*.py'} | sort -u` = 13 files. Matches Main's claim.
- RUFF (13 files, xargs -0 so the zsh word-split trap cannot fire):
  backend/agents/debate.py:16:20: F401 `typing.Callable` imported but unused
  Found 1 error. RUFF_EXIT=1.
  PRE-EXISTENCE INDEPENDENTLY PROVEN: `git show HEAD:backend/agents/debate.py`
  copy reproduces the same F401 (exit 1); `Callable` count is 1 in BOTH HEAD
  and worktree (import only); `git diff HEAD -- debate.py | grep Callable` is
  empty. So this step neither created nor can fix-by-accident that finding.
- pytest backend/tests/test_phase_86_108_parse_failure_ledger.py -q:
  37 passed in 2.05s (37 progress dots -- consistent, not spliced).
- REGRESSION SWEEP RE-RUN BY ME:
  `1 failed, 560 passed, 3068 deselected` -- the failure is
  test_phase_40_2_settings_json_still_valid_json_after_edit.

### FINDING E1 (evidence, CONFIRMED): live_check §8 carries the STALE sweep
live_check_86.108.md:383 -> "1 failed, 543 passed, 3068 deselected"
live_check_86.108.md:491 -> "(543 vs the then-current 552). Re-run; §8 now
                             carries the current figure."
experiment_results_86.108.md:61 -> "1 failed, 560 passed, 3068 deselected"
My re-run -> 560. So §8 still holds the CYCLE-1 capture (543) while §13 of the
SAME FILE asserts it was regenerated. The cycle-2 finding-3 remediation landed
in experiment_results only. The live_check is the criterion's named gate
artifact, and the false "now carries the current figure" sentence is inside it.

### FINDING E2 (evidence): "queued" is not reproducible
- live_check §8: the F401 "is queued below rather than fixed here". §12
  ("Defects discovered, to be queued") lists THREE items and the debate.py
  F401 is NOT one of them -- the pointer is dead.
- experiment_results:57 "queued below" (nothing below queues it) and :65
  "Queued as a defect" (past tense) for the effortLevel test.
- Masterplan walk: NO step exists for the F401 / Callable / effortLevel /
  xhigh defect (searched all 86.1xx ids and full-text). §12's own heading says
  "to be queued", which is honest; the past-tense wording elsewhere is not.

## C. MUTATION -- author's matrix REPRODUCED by me
`.venv/bin/python scripts/qa/mutation_86_108.py`
CONTROL rc=0 collected=37; M1..M17 all KILLED; SURVIVORS=none; UNSCORABLE=none.
Restore verified INDEPENDENTLY: md5 of all six touched files identical
before/after (ledger bc5dc0d9.., debate b6856c41.., llm_parse cd003f18..,
risk 7845abde.., orch de071796.., flags 66bcb677..).

## C2. MY OWN independent cells (not in the author's matrix)
Same strict rule (control green first, exit==1, same collect count, SHA-256
restore). CONTROL rc=0 collected=37 on every batch.

  Q1  KILLED   orchestrator._client_model_name -> constant  [test_client_model_name_unit]
  Q2  KILLED   risk_debate._effective_model_name -> constant (the twin M15 does
               NOT mutate)  [test_effective_model_name_resolution +
               test_run_risk_debate_records_the_real_client_model_on_every_agent]
  Q3  KILLED   in_force reports backend/.env instead of the RUNNING process
               [test_divergence_between_env_and_the_running_process_is_detected]
  Q4  KILLED   resolve_rail drops the failforward-ambiguity unknown
  Q5  KILLED   rail_basis becomes a CONSTANT "measured"
  Q8  KILLED   a whole KIND silently stops being recorded
  Q11 KILLED   a debate call site passes a LITERAL straight to the recorder
  Q7  **SURVIVED**  model_name=_client_model_name(None) at the Synthesis-Final
               orchestrator site. AST-legal (the kwarg value is a Call node, not
               ast.Constant) so test_every_parse_call_site_forwards_a_model_name
               passes. Real differential: _client_model_name(client) yields the
               client's model -> rail measured; _client_model_name(None) yields
               None -> rail "unknown" with basis "no_model_in_scope_at_emit_site",
               which is FALSE (a model was in scope). 37/37 stayed green.
  Q9  SURVIVED (weaker) `... or "claude-opus-4-8"` -- AST-legal BoolOp, would
               ACTIVELY misattribute, but only when the client's model is falsy.
  Q10 RETRACTED -- EQUIVALENT MUTANT, not a finding. Swapping
               self.synthesis_client for self.deep_think_client changes nothing
               because orchestrator.py:684-685 builds BOTH from the same
               `deep_model_name`. Tested, rejected; recorded so the record shows
               the differential was checked rather than assumed.

## D. Other reproductions (all GREEN unless noted)
- Settings 264 / FullSettings 45 / gated population 168 -- REPRODUCED exactly.
  No credential-shaped name among the 168.
- census --rotated-only: 2859; per-agent buckets sum to 2859 exactly
  (602+359+342+314+310+309+307+264+52). Moderator 359. REPRODUCED.
- census incl. live: 2874, Moderator 368. REPRODUCED.
- era_rail: ROTATED ONLY 2859 / INCL LIVE 2874 + the computed NOT-SUPPORTED
  bound. REPRODUCED.
- COMPLETENESS RECALL TEST on the emit sites: the census marker
  "returned invalid JSON" occurs at exactly 4 logger sites in backend/
  (debate:153, risk_debate:146, orchestrator:341, llm_parse:165). The 5th
  textual occurrence (claude_code_client.py:472) is a RAISED exception message
  and yields no census bucket. So "the four emit sites" IS the complete
  population behind the 2,859.
- Runtime smoke: all 8 changed backend modules import clean (exit 0).
- Live route state REPRODUCED against pid 41635 (started 15:57:16 local =
  13:57:16Z): /api/settings/flags 404, /api/observability/parse-failures 404,
  /api/observability/latency 200, /api/health 200.
- backend/.env mtime 2026-08-17T15:06:04, git-clean; SettingsUpdate NOT
  extended; risk_debate._judge_parse_fail_fallback UNTOUCHED by the diff.
- I DROVE the 3 wired emit sites myself: the discriminating Moderator row
  reproduces (live paper_use_claude_code_route=True, model gemini-2.5-flash ->
  rail=gemini_or_direct). Return values unchanged.
- No unintended production change: the peer files carry ZERO "86.108" markers;
  research_brief_86.69.md's modification is step 86.69's own cycle-2 work.
- Research gate: contract cites wf_8581f683-d24, 15 sources / 35 URLs /
  audit-class dry; the four transport quotes in live_check §4 trace verbatim to
  brief rows 3, 4, 15 and 2.

### FINDING P1 (guard coverage): the AST guard is defeated by any non-literal
test_every_parse_call_site_forwards_a_model_name rejects only ast.Constant at
the model_name= position, and it is the SOLE coverage for the orchestrator's 3
call sites. Q7 is an AST-LEGAL expression that survives 37/37 and degrades every
Synthesis-Final record's rail from measured to "unknown" with the basis
"no_model_in_scope_at_emit_site" -- false, since a model IS in scope. Named fix:
require the kwarg to be a Call to _client_model_name/_effective_model_name whose
argument is not None, or drive the synthesis loop. NOT vacuous (it kills
M16/M17/Q11) -- narrower than the defect class it is offered against.

### FINDING E3 (overclaim): experiment_results:14 says "37 tests. Every one
drives the REAL function or the REAL route handler." False for
test_every_parse_call_site_forwards_a_model_name, which only ast.parse()s three
source files. live_check §13 labels it correctly ("observes no behaviour"), so
the two artifacts contradict each other in the direction that overstates guard
strength.

### FINDING E4 (disclosure): an UNWIRED emit site listed as equivalent
live_check §5's table lists llm_parse.py:parse_llm_json beside the three wired
sites. VERIFIED: parse_llm_json has ZERO production callers -- every non-test
hit is the DIFFERENT `_parse_llm_json` in
backend/meta_evolution/directive_rewriter.py:214. The production docstring says
so; grep for "no production caller|not wired|75.5.5|surface uniformity" across
contract + experiment_results + live_check returns ZERO. Coverage of the ACTUAL
failure population is unaffected (all 9 census buckets are served by the 3 wired
sites), so this is scope-honesty, not coverage.

### NOTE (not capping): census_invalid_json_86_108.py's docstring says "2371
compact vs 501 json ... 17.4%"; current runs give 488/17.1% (rotated) and
503/17.5% (incl. live). That file is unmodified this cycle and the drift is
inherent to the growing live log; live_check §2 correctly uses --rotated-only.

## E. Criterion verdicts
C1 MET (disclosed, evidenced deviation: era-bucketed not per-event; rates shown
   non-derivable with a measured reason)
C2 MET
C3 MET (E4 is a disclosure gap on a site with no production traffic)
C4 MET-with-residual (route built + behaviourally guarded by Q3; the 404 on the
   running pid reproduced by me, disclosed with a positive control, deferred per
   the standing batched-restart rule, filed as ASK-3)
C5 MET
C6 MET as a process (17/17 reproduced, control green FIRST, byte-identical
   restore verified independently) -- P1 is the named coverage gap
Harness compliance: 5/5 clean.

## F. Verdict (worst-of-N lenses)
correctness   -> CONDITIONAL (P1 survivor, named fix)
reproduce     -> CONDITIONAL (E1: the ONE number that does not reproduce sits in
                 the gate artifact under a false "regenerated" claim)
scope-honesty -> CONDITIONAL (E1/E2/E3/E4)
min = **CONDITIONAL**. Not FAIL: no criterion is materially unaddressed and the
product is sound under 28 executed mutation cells. Not PASS: a false remediation
claim inside the artifact named by verification.live_check, plus a real
surviving mutant on the one seam with no behavioural driver.

COMPLETED: 2026-08-17T21:39:58Z

