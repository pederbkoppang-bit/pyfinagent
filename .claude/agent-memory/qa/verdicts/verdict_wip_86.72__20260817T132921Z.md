STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.72
WRITTEN: 2026-08-17T13:29:21Z
COMPLETED: 2026-08-17T13:43:06Z

# Q/A cycle-2 evaluation of step 86.72 (research-on-demand routing)

Read qa.md in full (871 lines) at 13:29Z. Operating per its verification order.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command `node --check .claude/workflows/qa-verdict.js`; git status scope; lint gate; scoped tests
C. LLM judgment vs 8 immutable criteria; mutation matrix on new guards (I mutate, not just read the author's matrix)

## Findings (appended as established)

### ATTEMPT / SEQUENCE EVIDENCE (gathered, not applied)
- `qa_wip.py 86.72 --spawned-at 2026-08-17T13:29:21Z`: source_present=true,
  attempt_number=2 (status "ok", is_lower_bound false), prior_attempts=1,
  prior_records=[verdict_wip_86.72__20260817T130241Z.md], records_retained=2 (gauge).
- `verdict_history_86_21.py --step 86.72 --evidence-only`: status `ok`,
  "1 verdict(s) from the ledger", verdicts = `FAIL`.
- Cross-check: prior_attempts 1 == ledger count 1. No staleness contradiction
  detectable for this step-id. Sequence: FAIL (cycle 1).

### B1. IMMUTABLE VERIFICATION COMMAND
`bash -c 'source .venv/bin/activate && node --check .claude/workflows/qa-verdict.js && echo parses'`
-> `parses`, EXIT=0.

### md5 BASELINE (taken before any mutation)
- .claude/workflows/qa-verdict.js       94164d41f77b5d53d2cb5378fbd110a6
- scripts/qa/verify_prompt_render_86_90.mjs cab5b10bbcc93f80ad82759573d4d52f
- scripts/harness/research_router.py     c1afce6e27db48b29ad0a21c582aa859
- .claude/agents/qa.md                   0ca1b71c6f9b1743adc00f23f97ce150
- .claude/workflows/research-gate.js     e26dc258bc862beead7f4a336c978480

### CHECKER FAMILY (control, unmutated)
- verify_prompt_render_86_90.mjs  ALL GREEN: 136 passed, 0 failed
- verify_research_gate_workflow.mjs ALL GREEN: 124 passed, 0 failed
- verify_workflow_args_boundary.mjs ALL GREEN: 96 passed, 0 failed
All three EXIT 0 (read unpiped via PIPESTATUS).

### C3 -- ROUTER, INDEPENDENTLY DRIVEN BY ME
- `research_router.py --self-test` -> 11 checks, all ok, SELF-TEST PASSED, EXIT=0.
  (Author claimed "self-test 11 checks" -- reproduces exactly.)
- Attempt-gate audit stream DOES carry the input the counter needs:
  `handoff/audit/attempt_budget_audit.jsonl` row
  `{"ts":"2026-08-17T13:22:09Z","type":"attempt","step_id":"86.105",
    "workflow":"research-gate.js",...,"attempt_number_inclusive":1}`.
  27 rows total, 2 with "research-gate".
- I drove the router MYSELF against the REAL default ledger (no --audit-ledger)
  with my own distinctive stub spec:
    decision DISPATCH, rounds_used **1**, rounds_floor true,
    reason "round 2 of 2 for 86.105", scriptPath ".claude/workflows/research-gate.js",
    topic OBJ_DISTINCT_ZQ1, internal_scope "SCOPE_DISTINCT_ZQ3 -- BOUNDARIES: BOUND_DISTINCT_ZQ4",
    extra carrying FMT_DISTINCT_ZQ2.
  => the round counter is REAL on the production path, not only on the temp fixture.
     A third round would REFUSE at exit 3.
- Arm-B spawn wf_5eacb773-aa5 EXISTS and is a REAL researcher launch:
  meta.json `{"agentType":"researcher","spawnDepth":1,"model":"opus"}`,
  first user message = the research-gate stage-1 prompt for step **86.105**,
  TIER: moderate, INTERNAL SCOPE carries the router's literal
  `<tool_scope> -- BOUNDARIES: <task_boundaries>` concatenation shape.
  Run record .json not yet written (still running at eval time; journal has
  only `{"type":"started"}`) -- disclosed by the caller as expected tree motion.
- ARM A, RE-DRIVEN BY ME END TO END (not read from the artifact): I drove the
  SHIPPED qa-verdict.js with a no-signal judge return and captured its REAL
  output -> `research_routing: {research_needed: null, research_brief_spec: null,
  next_action_on_research_needed: null}`; piping THAT captured return into the
  router gives `NO_SIGNAL`, `launch: null`, EXIT 0. Both arms independently
  reproduced by the evaluator.

### C1 -- CONTROLS RE-DERIVED (mine, exact match to the artifact)
run_harness.py=5 | qa-verdict.js@fff6d8c4=0 | working tree=7 | research-gate.js=0
| negative control ZZZ_NO_SUCH_86_72=0. Plus a whole-directory sweep:
`git grep -c research_needed fff6d8c4 -- .claude/workflows/` returns NOTHING.
Brief reports 4 disagreements with audit_basis (incl. stale anchor :202->:394)
rather than adopting them. **MET.**

### C2 -- CENSUS INDEPENDENTLY RE-DERIVED (607 records, my own script)
Population: every `*/workflows/wf_*.json` under the project session dirs; role =
scriptPath contains research-gate.js|qa-verdict.js; step = args.step_id.
195 unclassified (other scripts), 6 classified-but-no-step_id. 326 qa / 80 researcher.
  86.85 qa=11 r=1 (2026-08-14T19:35) | 36.8 9/0 | 86.28 8/1 (2026-08-10T06:40)
  86.21 7/2 | 86.84 7/0 | 75.5 7/0 | 86.94 6/2 (08-16T21:08, 08-17T07:07) | 78.2 6/0
  86.97 r=2; **twelve** steps have >1 researcher launch; **86.59 has FOUR**.
Every published figure reproduces EXACTLY, including the two event timestamps and
the "twelve / 86.59=FOUR" claim. Only delta: 86.21 now also sits at qa=7 (its
cycle-8 qa spawn ran 13:14Z, AFTER Main's capture) -- tree motion, not a defect.
The claim is CORRECTED not confirmed, which the criterion explicitly permits. **MET.**

### C4 / C5 / C6
- C4 MET: caller-supplied tier justified in writing with cited evidence (Anthropic
  S1 verbatim "Agents struggle to judge appropriate effort..."; Triage S5: eta_E
  negative, 6.0-36.6% self-budget compliance, <8% infeasibility detection, 20
  models/4 benchmarks) AND demonstrated -- ~24 tier assertions in
  verify_research_gate_workflow.mjs drive the real machinery (UNSUPPORTED spawns
  ZERO agents; ABSENT defaults to moderate and still spawns; a supported
  non-default tier is APPLIED not downgraded).
- C5 MET: `VALID_TIERS = ['simple','moderate','complex']` -- 'deep' absent; the
  in-file note says "NOT IN SCOPE, deliberately ... pre-empt an open operator
  decision. Report the gap; do not close it unilaterally." ASK-1 in the contract
  carries the measured cost basis and parks the call at 86.73. NOTE: the
  CRITERION's own anchor ":190-200" is stale (that range is now
  `jsonLosslessViolation`; the note lives ~:374-411) -- the brief reports the drift.
- C6 MET: FLOOR_SOURCES=5 / FLOOR_URLS=10 unchanged; `recency_scan_performed !==
  true` violation intact; the ONLY research-gate.js change in the whole step
  (77f15b4d) is +8 lines, **all comments, zero deletions** (filtering non-comment
  '+' lines yields empty); cycle-2's commit does not touch the file at all;
  .claude/rules/research-gate.md last changed 2026-08-13 (pre-step).

### C7 -- DRIVEN BY ME ON THE SHIPPED SCRIPT
Judge returns FAIL + research_needed=true + spec -> returned object
`verdict: "FAIL", ok: false`, routing BESIDE not inside. The cycle-1 WARN on
`verdict_unmodified` is corrected IN PLACE in experiment_results and the
correction is ACCURATE: I re-confirmed the narrowness (LG-E pre-merge in-place
and LG-F post-computation both keep verdict_unmodified true). **MET.**

### C8 -- MUTATION WORK, EXECUTED BY ME
Control first: 136 / 124 / 96 all green, all EXIT 0.
All five cycle-1 survivors are permanent cells reporting KILLED, each with a
property-owning control (routed target / spawn instruction / stagnation / floors /
field-for-field spec echo vs DISTINCTIVE OBJ-FMT-SCOPE-BOUND-UNIQ-8672 fixtures).

RELOCATED-TREE PROBES (mkdtemp mirror of scripts/qa + .claude/workflows):
- NULL control: 132 passed / 2 failed -- the 2 are section [1]/[4] git reproduction
  in a non-git tmp dir (measured relocation cost, per run-a-null-mutant-first).
- P1 (cycle-1's :638 latent defect): I made one cell's mutant UNBUILDABLE. It now
  scores `UNSCORABLE: the mutant did not build (Unexpected token ';')` and the
  check goes **RED** (+1 vs control). The build-break-credited-as-KILL defect is
  genuinely FIXED, proven by execution.
- P2 (does LG-1 own the runtime leak-guard?): neutering
  `if (leakedR.length > 0)` turns LG-1 **RED** with "no throw -- the guard is
  inert" (+1 vs control). LG-1 is NOT vacuous.

MY OWN MATRIX ON THE NEW ROUTER (Main presented none): 10 cells + NULL control.
  NULL (relocated, unmutated): green -- relocation is not a confound.
  KILLED (8): R1 any-workflow-counts-as-a-round; R2 `>=`->`>`; R3 TMAX 2->99;
    R4 target repointed at qa-verdict.js; R5 spec fabricated; R7 research_needed
    False now dispatches; R8 counter hardwired to 0; R10 REFUSE downgraded to exit 0.
  UNSCORABLE (1): R6 all()->any() -- crashes loudly (KeyError 'tool_scope'),
    i.e. it cannot mis-validate silently. Licenses nothing either way.
  **SURVIVOR (1): R9** -- repointing the module-level `AUDIT_LEDGER` constant at a
    nonexistent path leaves all 11 self-test checks GREEN, because the self-test
    always injects a temp ledger and no re-runnable check pins the PRODUCTION
    sink. Vacuity shape 5. Failure direction is OPEN: a silently-zero counter
    never fires TMAX. MITIGATION that keeps this WARN not BLOCK: I verified the
    production path LIVE (rounds_used=1, "round 2 of 2") and 8 killed cells prove
    the guard is otherwise behavioural.

SECOND SURVIVOR, on the runtime leak-guard's PRECISION (LG-A, mine): the guard
excludes `research_needed`/`research_brief_spec` from the leak set UNCONDITIONALLY
by name. Executed: assigning the DERIVED routing values to those top-level keys on
a verdict where the judge set NEITHER does **not** throw, `verdict_unmodified`
stays true, and a caller-computed `research_needed: null` surfaces as top-level
judge output. One-line tightening: exclude only when `k in verdict`.

BYTE-IDENTICAL RESTORE: all five md5s unchanged after every mutation
(qa-verdict.js 94164d41..., checker cab5b10b..., router c1afce6e..., qa.md
0ca1b71c..., research-gate.js e26dc258...); `git status --short` on
.claude/workflows scripts/qa scripts/harness .claude/agents docs/runbooks CLAUDE.md
is EMPTY. **MET.**

### DETERMINISTIC GATES
- ruff F821,F401,F811 over a DERIVED non-empty scope (working-tree diff U step
  commits U untracked, 2 files: backend/api/sovereign_api.py,
  scripts/harness/research_router.py): "All checks passed!", exit 0.
- `npx tsc --noEmit` exit 0. `npx eslint src` exit 0. `npx eslint .` exit 1 with
  all 26 errors in `.next-audit-36-12/` + `.next-functional/` build output --
  PRE-EXISTING, already a queued defect; this step touched no frontend/**.
- Step scope (union of 1777cc8d + 77f15b4d, non-handoff): qa.md, qa-verdict.js,
  research-gate.js, CLAUDE.md, per-step-protocol.md, research_router.py,
  verify_prompt_render_86_90.mjs. No backend/**, no frontend/**, no masterplan,
  no test tree. NO UNINTENDED PRODUCTION CHANGE.
- No UI claims in this step -> gate 1c not applicable.

### HARNESS COMPLIANCE (5 items) -- CLEAN
1. research gate: brief COMPLETE, gate_passed true, 8 sources >=5, 28 URLs >=10,
   dedicated recency scan. brief 11:50 < contract 12:22 < results 15:24.
2. contract before generate: yes (mtime + commit order).
3. experiment_results present, with a cycle-2 GENERATE section.
4. LOG-last: `grep -cF "phase=86.72" handoff/harness_log.md` = 0; masterplan
   status "pending". Correct -- neither has happened yet.
5. no verdict-shopping: evidence CHANGED (commit 1777cc8d rewrote
   experiment_results +75 and live_check +57 since the cycle-1 FAIL).

### RESIDUALS -- EVIDENCE-QUALITY, QUEUEABLE (all criteria substantively MET)
R-a WARN: router R9 -- no re-runnable check pins the production AUDIT_LEDGER
    path/shape; the counter fails OPEN if the audit stream moves or renames
    `workflow`. Fix: one self-test cell asserting the default constant resolves to
    an existing file whose rows carry step_id+workflow.
R-b WARN: leak-guard by-name exclusion is unconditional; tighten to `k in verdict`.
R-c NOTE: no re-runnable cell asserts "FAIL + signal stays FAIL" on the whole
    script -- C7 rests on a one-off drive. ~6 lines in section [8] would also kill
    my LG-E/LG-F.
R-d NOTE: experiment_results section 4 still claims, past tense, "Flagged in
    harness_log for operator review" -- `grep -cF "phase=86.72"
    handoff/harness_log.md` = 0. Cycle-1 remediation item (e) closed its first half
    (the qa.md sentence, now TRUE and router-named) but not this. LOG legitimately
    runs after EVALUATE, so the defect is the TENSE; the note is genuinely owed at
    LOG time because this step edited .claude/agents/qa.md twice.
R-e NOTE: research_router.py hardcodes `tier: "moderate"` for every dispatched
    round -- the requesting judge cannot express difficulty. Consistent with the
    C4 design choice (caller owns tier) but undisclosed in the artifacts.

### VERDICT REASONING
Worst-of-3-lenses: correctness (drove router, script and checker myself -- right);
reproduces (every headline figure re-derived independently and matched); scope
honesty (the artifact CORRECTS its own prior claim rather than defending it,
discloses two of its own first-run mutation failures, discloses the stub-verdict
limit, and corrects the cycle-1 WARN in place). All 8 criteria MET. Residuals are
evidence-quality. VERDICT: PASS with five queued residuals.
