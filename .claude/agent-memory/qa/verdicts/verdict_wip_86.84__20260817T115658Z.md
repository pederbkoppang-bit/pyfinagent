STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.84
WRITTEN: 2026-08-17T11:56:58Z

# Q/A cycle-10 evaluation of step 86.84 (write-first record)

## A. HARNESS COMPLIANCE (5 items) -- CLEAN
- research gate PASSED: handoff/current/research_brief_86.84.md envelope read directly --
  brief_status COMPLETE, external_sources_read_in_full 11, snippet_only 8, urls_collected 19,
  recency_scan_performed true, gate_passed true. Committed d9e9a35b 2026-08-14T19:18:58+0200.
- contract-before-generate: brief 2026-08-14 19:15 < contract 2026-08-17 12:20 <
  experiment_results + live_check 2026-08-17 13:38 (mtime); git: contract d69da099 12:35,
  generated artifacts cbbd1566 13:41. ORDER OK.
- experiment_results present (24,728 B; Cycle 10 section at :394). live_check §14 at :674.
- log-last: harness_log carries only result=IN-PROGRESS / result=EVIDENCE-ADDED for
  phase=86.84; masterplan status = "pending". OK.
- no-verdict-shopping: evidence CHANGED (HEAD cbbd1566: +130 experiment_results, +115
  live_check, +148 rail_turn_cap.py, +59/-10 mutate_rail_turn_cap.py).

## SEQUENCE EVIDENCE (gathered, not applied)
- qa_wip.py --spawned-at 2026-08-17T11:56:58Z: source_present=true, attempt_number=10,
  prior_attempts=9, attempt_number_status="ok", attempt_number_is_lower_bound=true,
  records_pruned_known=null, records_retained=10 (GAUGE, not used as a counter).
- verdict_history_86_21.py --evidence-only: status=ok, "9 verdict(s)".
  CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> FAIL -> NO_VERDICT -> FAIL ->
  FAIL -> CONDITIONAL. NO_VERDICT carried through as-is.
- Cross-check: prior_attempts (9) == ledger rows (9); attempt_number (10) exceeds by exactly
  this in-flight spawn. Ledger NOT demonstrably stale for this step.

## B. DETERMINISTIC -- ALL REPRODUCE
- `python3 scripts/qa/rail_turn_cap.py --verify` EXIT 0. My own read: 598 records / 1296
  spawns / 0 missing. qa cap=30 n=353 drop=39 @cap=39 >cap=0; researcher cap=40 n=107 drop=9
  @cap=9 >cap=0. Drop turn sets {30} and {40}. C1 1296>0 / 0 zero-with-lines; C2 0 exceed;
  detector 1228/1238 vs 1/48; C3 killed at [1,1,2,2,2,3,4,5,6,16], 0 at a cap.
  Uncapped 0/901, AT RISK 0/101 vs 12.2%. Uncensored sample: qa n=51 p50=41 p90=55 max=62
  >old-cap(30)=47, dropped=0 non-emitters=0 killed=0 errored=0 erased=2(non-emit 2);
  researcher n=14 p50=19 p90=35 max=38.
- MATRIX re-executed by me: cells=37, real survivors=0, known/equivalent BY OUTCOME=2
  (M6/M6b, disclosed), errors=0, kills {VERIFY 29, ORACLE 2, INJECTED_TRUTH 2,
  MUST_STAY_GREEN 2}, control GREEN first, byte-identical restore. S15 kills with the named
  problem. EXACTLY as the artifact claims.
- criterion 4: `node scripts/qa/verify_rail_retry.mjs` 38 passed / 0 failed, exit 0
  (section [F]: F1 exhausted retry yields NO value; F2 rethrows original; F3 research-gate
  RECOMPUTES gate_passed; F4 retry loop assigns no verdict field).
- criterion 7: `node scripts/qa/verify_research_gate_workflow.mjs` 124 passed / 0 failed,
  exit 0 (fail-closed on null envelope, gate_passed recomputed from brief on disk).
- criterion 5, re-derived at all three sources: rail_drop_rate.py docstring :20 "THE CAUSE IS
  NOW KNOWN, AND THE MODEL SPLIT ABOVE IS CONFOUNDED -- DO NOT CITE IT AS A MODEL EFFECT" +
  RUNTIME caveat printed under the by-model table (I ran it); qa-verdict.js:629 "say it was
  UNPROVEN; that is SUPERSEDED"; research-gate.js:880 "This block used to end ...".
  Quote-then-replace at each. MET.
- lint: DERIVED 9-file scope (git diff HEAD + d69da099..HEAD + untracked, sorted -u,
  non-empty asserted, piped through xargs) -> ruff F821/F401/F811 "All checks passed!" exit 0.
- frontend: npx tsc --noEmit exit 0; npx eslint src 0 errors / 55 warnings exit 0.
- SCOPE: `git status --short` over all 86.84 files -> EMPTY (all clean vs HEAD). Only dirty
  86.84 artifact is evaluator_critique_86.84.md (+291), Main transcribing cycle 9.
- Repo byte-identical after ALL my work: 7 md5s re-checked and unchanged.

## C. MY OWN EXECUTED MUTATIONS (temp mirror; repo never written)
Mirror = scripts/qa/rail_turn_cap.py + .claude/agents/*.md + .claude/workflows/*.js under a
mkdtemp root; module loaded by importlib with mod.REPO asserted == mirror root.

CONTROL           verify_ok=True problems=[] erased_qa=(2,2) unclassified=41
                  unclassified_post=0 render_md5=b55b5c69...

MUT-A  qa-verdict.js emitted header renamed to 'BINDING ACCEPTANCE TESTS for '+stepId+':'
       with the retired literal kept ONLY in two `//` comment lines.
       Measured: literal on a NON-comment line = 0; literal present in file bytes = True.
       -> verify_ok=True, problems=[]  ***SURVIVOR***
MUT-B  same rename with the literal REMOVED entirely.
       -> verify_ok=False, problems=["ORPHAN-CLASSIFIER COUPLING BROKEN: ..."]  KILLED
       (proves the pin is NOT inert; S15's kill is genuine)
MUT-C  erased_unclassified hardcoded to 0 in rail_turn_cap.py.
       -> verify_ok=True, problems=[], render byte-identical to CONTROL  ***SURVIVOR***

## FINDINGS (both WARN; both executed; neither overturns the diagnosis or the remedy)
F1  The cycle-10 coupling pin (:1021) asserts only that the marker literal appears in the
    workflow file's BYTES, not that it is the string EMITTED. A rename that leaves the old
    literal in a `//` comment keeps verify GREEN while every FUTURE qa orphan classifies
    role=None. Not speculative: qa-verdict.js ALREADY carries a comment reproducing a sibling
    prompt literal verbatim (:106 `// 'EVIDENCE / FILES TO READ: ' + evidence`), plus :609
    and :628 quoting retired text, and research-gate.js:880 -- and criterion 5 of this very
    step MANDATES quote-then-replace. Main applied this exact insight to the OTHER pin
    (live_check:307, cell M7b, "restore the pin with a note" via a `#` comment) but not to
    this one. qa.md 4c shapes #3 + #8. Named fix: require the literal on a non-comment line,
    or match against the prompt array region / the script's own rendered prompt.
F2  erased_unclassified / _post_removal (:758-761) are computed into the remediation dict and
    referenced NOWHERE else -- not rendered by render(), not asserted by verify(), no matrix
    cell. Reachable only via `--json`, which the immutable command does not print and no
    artifact quotes. MUT-C proves no mode (VERIFY/ORACLE/INJECTED_TRUTH/MUST_STAY_GREEN) can
    kill a mutation of it. The artifact's "role=None orphans are never invisible" is true only
    on the --json path. Named fix: print `unclassified=N` in the per-role remediation block --
    one line, and it immediately gains ORACLE coverage.
F1+F2 COMPOSE into the cycle-9 failure mode for future drift: rename the header per house
    style -> pin green -> future orphans role=None -> dropped by the per-role filter -> land
    in a counter the default report never prints. Narrower than cycle 9's (which needed no
    comment trick), so the fix DID improve things -- said plainly.

## CLAIMS RE-DERIVED (4b)
- "41 of 44 current orphans are pre-removal role=None bulk from two old runs" -> REPRODUCES
  exactly: Counter({(None,False):41, ('qa',True):2, ('qa',False):1}); role=None runs =
  {wf_03d6e7c4-fda: 40, wf_b9bbd4fd-978: 1} = two runs. post-removal unclassified = 0.
- "qa n=49 ... erased=2(non-emit 2)" -> today reads n=51; disclosed as "captured at write
  time (counts move with the corpus)". Reproduces in kind. Not a finding.
- qa orphans: wf_48465ea7-38e(15,pre,non-emit), wf_2fafe515-6a2(38,post,non-emit),
  wf_80376bff-7ae(10,post,non-emit) -> matches erased=2(non-emit 2) on the post-removal row.

## CRITERION MAP
1 MET · 2 MET · 3 MET · 4 MET · 5 MET · 6 MET IN SUBSTANCE (residual F1/F2 undisclosed)
7 MET · 8 MET IN FORM, GAP: erased_unclassified is a new cycle-10 addition with no cell and
no possible kill (MUT-C), and the pin's comment-evasion (MUT-A) was not tried or reported.

COMPLETED: 2026-08-17T12:07:53Z
