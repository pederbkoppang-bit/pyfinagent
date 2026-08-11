STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.33
WRITTEN: 2026-08-11T12:59:20Z

# Q/A write-first record -- step 86.33 (EVALUATE, cycle 1)

Spawned via Workflow rail. Read .claude/agents/qa.md in full (640 lines) at
2026-08-11T12:59:20Z -- operating instructions confirmed live from disk.

## A. HARNESS COMPLIANCE -- 5/5 CLEAN
- research-gate-before-contract: research_brief_86.33.md envelope
  brief_status=COMPLETE, gate_passed=true, external_sources_read_in_full=12
  (>=5), urls_collected=22 (>=10), recency_scan_performed=true (B3, method
  substituted + disclosed because WebSearch was exhausted 200/200 -- shortfall
  DISCLOSED, not hidden).
- contract-before-generate: mtimes research 14:43:23 < contract 14:49:33 <
  experiment_results 14:58:08. ORDER HOLDS.
- experiment_results present: 8615 bytes.
- log-last: `grep -cE 'phase=86\.33 +result=' handoff/harness_log.md` = 0, and
  masterplan status=pending. Nothing logged/flipped ahead of the verdict. OK.
- no-verdict-shopping: cycle 1, no prior evaluator_critique_86.33 exists,
  0 prior CONDITIONALs -> the 3rd-CONDITIONAL auto-FAIL rule does not bind.

## B. DETERMINISTIC
- IMMUTABLE COMMAND `bash -c 'bash -n .claude/hooks/qa-write-guard.sh && echo
  guard-parses'` -> stdout `guard-parses`, EXIT=0.
- git status: only researcher memory + audit/heartbeat/health logs modified.
  NO unintended production change. Step work committed 9a0a55bb / a6d4c42f /
  7fd1c131 / c88484be.
- ruff F821,F401,F811 over the derived superset scope (37 .py files,
  `git diff --name-only d23a981e HEAD -- '*.py'`, non-empty asserted):
  "All checks passed!" exit=0.
- guard md5 f90a01405e3f21577695ee4fedb800a2 BEFORE and AFTER every prover and
  the mutation matrix -- no residue.

## C. CRITERION 2 -- THE REAL MEASUREMENT (this spawn produced it)
My own first Write fired the real PreToolUse hook. Log row
2026-08-11T12:59:29.967328+00:00:

  agent_type = "qa"   agent_id = "afd21026f4056c9e0"
  payload_keys = [agent_id, agent_type, cwd, effort, hook_event_name,
                  permission_mode, prompt_id, session_id, tool_input,
                  tool_name, tool_use_id, transcript_path]   (12 keys)

Comparison set derived from the same log:
  - 12-key set: 1 row  = mine, the FIRST real subagent payload ever recorded.
  - 10-key set: 3 rows = MAIN-shaped writes; agent_type AND agent_id ABSENT.
  - 8/5/4/3-key sets (471 rows) = synthetic prover payloads.

ANSWER: **NO.** The installed Claude Code PreToolUse payload carries exactly
ONE role-ish identity field (`agent_type`) plus an opaque instance id
(`agent_id`). There is no `subagent_type`, no `agent_name`, no second field
that separates TYPE from NAME. The 4 keys Main's synthetic probe lacked
(effort, prompt_id, tool_use_id, transcript_path) are none of them role
attributes. Criterion 2's question is now MEASURED, not inferred.
Main's refusal to claim its synthetic echo as the answer was correct.

## C. CRITERION-BY-CRITERION
1. MET (with a WARN). Script re-runs, exit 0, 72 distinct agent_type over
   8319 rows; I re-derived 72 independently. WARN: prose says "72 distinct...
   covers 34 ... misses 37" -- 34+37=71. Mechanism found at
   derive_agent_type_population_86_33.py:153-154: `evade = ... if t and ...`
   silently drops the EMPTY agent_type (2072 rows, the LARGEST bucket, = Main's
   own writes) from the denominator while the headline 72 includes it. Also
   the script REIMPLEMENTS the guard predicate as
   startswith(("qa-","qa_","QA-","QA_")) instead of the guard's
   `.lower()` form -- divergent on e.g. "Qa-Mixed" (guard MATCHES, script
   reports EVADING). No effect on today's numbers; wrong shape.
   Also stale-by-growth: prose "quality-auditor (11)" measures 21 now; the
   last 10 rows landed 12:58:34Z, 26s AFTER experiment_results was written,
   from Main's own live_check run. Log is gitignored so no committed baseline.
2. MET -- by MY measurement above; Main built the instrument and honestly
   marked its own synthetic result PENDING rather than claiming it.
3. MET. Re-ran prove_researcher_rail_unbroken_86_33.py: 34 researcher
   spellings ALLOW, CONTROL (qa -> backend/main.py) BLOCK rc=2. Non-vacuous:
   under M3 the researcher leg goes RED, so the control is load-bearing.
4. MET. Re-ran mutation_matrix_86_33.py: control GREEN FIRST (4 checks),
   M1/M2/M3 all KILLED with expected-vs-actual red sets printed, restore
   byte-identical True, post-restore all green True, guard md5 unchanged.
   M3's actual red set (4 legs) exceeds its expected (1) -- correctly reported
   rather than trimmed.
5. MET, and it is RESTRAINT not evasion -- I proved it independently.
   612-cell differential (17 agent_types x 9 paths x 4 tools) between the
   pre-step guard (d23a981e, md5 19385cd0...) and HEAD: **0 decision
   differences**. Probe shown DISCRIMINATING first (15 ALLOW / 6 BLOCK), so
   the zero is not vacuous. The whole 86.33 delta is additive log-only.
   ASK #6 recorded with three costed options; the criterion literally says
   such a change must NOT be shipped in this step.
6. MET. Python body is inside `python3 -c '` (line 35) closing at line 132;
   apostrophes strictly inside the body = 0 (the 4 counted are the opener's
   printf '%s' and the closing quote). 86.33-added lines contain 0
   apostrophes. M3 proves the trap is DETECTED, not merely documented.

## Gate correction check (Main vs the researcher brief)
Brief lines 455/526 assert "32 SyntaxError traces ... demonstrably fired".
MEASURED: SyntaxError=5, NameError=1, Traceback=1. Main is RIGHT. The 32 is
the brief's own line-count of unparseable log LINES (line 121), re-quoted at
:455/:526 as a TRACE count -- a units error. All traces sit beside
`/T/qa_guard_probe_z02c0bak/g.sh` and `pythonNOPE3: command not found`, i.e.
a deliberately broken interpreter in a temp COPY of the guard. Probe
artifacts, not silent production breakage. Main's correction stands.

## FINAL: CONDITIONAL
Deciding fact: criterion 2's covering evidence is NOT in the handoff record.
experiment_results_86.33.md:60-64 and :165, and live_check_86.33.md:55, all
state the real answer is PENDING; only the synthetic echo is written down.
qa.md section 4 contract-completeness: a criterion with no covering evidence
in experiment_results.md CAPS the verdict. It is CONDITIONAL, not FAIL,
because the criterion is materially addressed -- the instrument was built
correctly, and the measurement now EXISTS (produced by this spawn, and it
reproduces: two real rows, 12:59:29.967328Z and 13:06:06.080792Z, both
agent_type='qa', agent_id='afd21026f4056c9e0', both 12 keys).

REMEDIATION (small, mechanical, no code change):
 R1. Transcribe into experiment_results §2 + live_check §4: the 12-key real
     payload, the 10-key Main-shaped payload with agent_type/agent_id ABSENT,
     and the answer NO -- one caller-chosen `agent_type` plus an opaque
     `agent_id`; no field separates TYPE from NAME. Cite the two log rows.
 R2. Fix criterion 1's partition sentence to state its normalization rule:
     72 distinct = 34 qa-role + 37 other named + 1 EMPTY (2072 rows,
     Main-shaped, silently dropped at derive_...86_33.py:154 by `if t`).
     Drop or date-stamp "quality-auditor (11)" -- it measures 21 now.
 R3. Optional: make derive_...86_33.py reuse the guard's `.lower()` predicate
     instead of startswith(("qa-","qa_","QA-","QA_")); they diverge on
     "Qa-Mixed" (guard MATCHES, script reports EVADING).

NOT findings (checked and cleared): criterion 5 restraint is correct, proved
by a 612-cell zero-diff differential with demonstrated discriminating power;
Main's correction of the gate's "32 SyntaxError traces" is right (5+1+1,
probe artifacts); the mutation matrix and researcher prover both reproduce
exactly and leave the guard byte-identical.

COMPLETED: 2026-08-11T13:08:00Z
