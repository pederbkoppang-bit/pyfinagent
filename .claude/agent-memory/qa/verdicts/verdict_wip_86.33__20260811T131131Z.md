STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.33
WRITTEN: 2026-08-11T13:11:31Z

CYCLE: 2 (cycle 1 = wf_87bc566d-64d, CONDITIONAL, remediated at 335257a8)
SCOPE PER SPAWN PROMPT: verify ONLY the two remediations + anything they broke.

## A. HARNESS COMPLIANCE -- CLEAN (5/5)
1. research-gate-before-contract: research_brief_86.33.md envelope brief_status=COMPLETE,
   external_sources_read_in_full=12 (>=5), urls_collected=22 (>=10), recency_scan_performed=true,
   gate_passed=true. mtime 14:43:23 < contract 14:49:33. OK
2. contract-before-generate: contract 14:49:33 < experiment_results 15:10:14 < live_check 15:10:39;
   commit order 8935be78/10a703db (PLAN) precede a6d4c42f/7fd1c131/c88484be (GENERATE). OK
3. experiment_results present (10,280 B). OK
4. log-last: `grep -F 86.33 handoff/harness_log.md` -> only forward-references from 86.31/86.32,
   NO cycle entry; masterplan status="pending". Not yet flipped. OK
5. no-verdict-shopping: evidence CHANGED between cycles -- 335257a8 touched
   experiment_results_86.33.md, live_check_86.33.md, derive_agent_type_population_86_33.py. OK
3rd-CONDITIONAL: `grep -cE "phase=86\.33 .*result=CONDITIONAL"` -> 0. No auto-FAIL trigger.

## B. DETERMINISTIC
- IMMUTABLE CMD: bash -c 'bash -n .claude/hooks/qa-write-guard.sh && echo guard-parses'
  -> "guard-parses"  exit=0
- ruff F821,F401,F811 over 15 .py files derived from `git diff --name-only 8935be78^ HEAD -- '*.py'`
  (non-empty set asserted, passed via xargs not an unquoted var) -> "All checks passed!" exit=0
- derive_agent_type_population_86_33.py exit=0 ; census_qa_write_guard_log_86_31.py exit=0
- git status --short: NO unintended production change (only researcher agent-memory owned by
  another agent, rotating audit/heartbeat logs, and this WIP file).
- GUARD LIVENESS (not just parse): my own Write at 2026-08-11T13:11:39.273140+00:00 produced a
  payload_keys row -> the single-quoted python body EXECUTED. The guard is ALIVE, not fail-open-dead.

## C. THE TWO REMEDIATIONS

### R1 criterion 2 is IN the record -- VERIFIED, and independently reproduced
- Cited rows 12:59:29.967328 and 13:06:06.080792 reproduce byte-exact: 12 keys
  [agent_id, agent_type, cwd, effort, hook_event_name, permission_mode, prompt_id, session_id,
  tool_input, tool_name, tool_use_id, transcript_path], agent_type='qa',
  agent_id='afd21026f4056c9e0'.
- INDEPENDENT reproduction by ME: my own write logged the IDENTICAL 12-key set with
  agent_type='qa' and agent_id='a88e054bb964e863b' -- a DIFFERENT id under the SAME type.
  That strengthens the answer: agent_id is per-instance, agent_type is the one role field.
- 10-key Main-shaped rows (3 found) confirmed: agent_type and agent_id genuinely absent from
  the key set, not merely empty.
- 'pending' sweep over contract/experiment_results/live_check: ZERO hits. The old
  "§4 ... the REAL answer is pending" block was REMOVED by the remediation, not left beside.
- NOTE (cosmetic): artifacts render the ts as ...967328Z, the log as ...967328+00:00. Same instant.

### R2 criterion 1 partition -- VERIFIED, with one wording correction
- Re-ran the script: total 78 / matched 36 / NOT matched 42 / "-> 36 + 42 = 78 (must equal 78;
  EMPTY is counted, not dropped)"; EMPTY printed as "(EMPTY -- Main-shaped writes) 2152".
- Predicate reuse verified against qa-write-guard.sh:120-121 (strip().lower() then
  =='qa' or startswith 'qa-'/'qa_'). Divergence claim is REAL, not hypothetical: 'Qa-Mixed'
  EXISTS in the log and is the ONLY value where guard != the old reimplementation
  (checked all 78; also present: 'QA-80-2', 'QA-Upper', which both agree).
- CORRECTION: commit 335257a8's message says "prints the partition and ASSERTS it sums".
  There is NO assert -- `grep -n "assert|raise|sys.exit"` yields only `sys.exit(main())`:185.
  It PRINTS "(must equal 78)". The sum is now guaranteed BY CONSTRUCTION (qa_like and evade are
  exact complements of the same set), so an assert would be tautological -- the structural fix is
  the correct one -- but the word "asserts" should not stand, or a later reader will rely on a
  guard that does not exist. NOTE-level, not blocking.

### R3 perishability disclosure -- ADEQUATE
- Reproduces: quality-auditor now 97 (matches the disclosed "97 an hour later"); EMPTY 2152 vs
  the artifact's 2,151 -- grew by 1 between the write and my read, exactly as disclosed.
- Every figure I found is date-stamped or rule-stamped (§1 "AS OF 2026-08-11T18:1x CEST";
  §3 "AS OF this run ... the log GROWS"). No frozen count is presented as durable.

## D. FINDING I RETIRED (recorded so it is not re-raised)
I filtered on a platform-minted agent_id to test §2's corroboration: 73 such rows, agent_type only
'qa'/'researcher', ZERO of the 66 name-shaped values. That looked like a falsification. It is not:
the pre-contamination slice (<2026-08-10T09:30:00Z) holds 3,012 rows and 65 distinct agent_types
INCLUDING qa-80-2-c2 / researcher-80-27, and carries NO agent_id at all -- the field was added by
this step's own P0 (8a9a4293). My filter was blind by construction. Probe indicted, finding withdrawn.
Residual NOTE only: §2's "2 definitions against 78 logged values" quotes the UNFILTERED count, of
which most are prover-fabricated; the real-spawn figure is 65 distinct in the clean slice. §1
already discloses that population problem and the conclusion is unaffected.

## E. BLOCKER -- criterion 1 / live_check record incompleteness
The masterplan live_check field REQUIRES live_check_86.33.md to carry "the re-derived agent_type
census (full distribution + outside-memory-dir counts)". The file (88 lines, §§1-5) carries NO
census: no distribution table, no per-identity outside-.claude/agent-memory/qa/ counts. Verified it
was never there (0 hits across BOTH committed revisions), so the remediation did not remove it.
Immutable criterion 1 further requires the derivation to report, for every value, how many
Write/Edit events targeted paths OUTSIDE .claude/agent-memory/qa/; recall validated against the
DERIVED CLASS ("20 events across 10 identities ... A method that misses ANY of them is rejected");
and "any census must take a --before cutoff and REPORT the excluded row count".
derive_agent_type_population_86_33.py does NONE of these (no argparse, no --before, no
outside-memory counts, no breach class), and experiment_results_86.33.md never cites the script
that does (grep for census_qa_write_guard_log_86_31 -> rc=1).
THE EVIDENCE EXISTS AND REPRODUCES TODAY -- I ran it:
  $ python scripts/qa/census_qa_write_guard_log_86_31.py --before 2026-08-10T09:30:00Z
  rows counted 3012 | rows excluded 6866 | 27 named qa-role identities
  113 Write/Edit events, 69 OUTSIDE .claude/agent-memory/qa/
  NO-SELF-EVAL BREACHES: 20 events across 10 identities   (exit 0)
This is the SAME defect shape cycle 1 raised and Main ACCEPTED for criterion 2 ("the answer existed
only in the log and the verdict, so criterion 2 had no covering evidence in the handoff record"),
surviving on criterion 1's other legs. The fix is transcription, not new work.

## F. CRITERION MAP
1 NOT MET (record) -- derivation + partition reproduce; outside-memory counts, 20/10 recall,
  --before cutoff + excluded-row count absent from the 86.33 record and from its script.
2 MET -- measured on the installed platform, reproduced by me independently. Answer NO.
3 MET -- live_check §2: 34 spellings ALLOW + control BLOCK rc=2 (cycle 1 reproduced).
4 MET -- live_check §3: control green FIRST, 3/3 KILLED, restore byte-identical (cycle 1).
5 MET -- ASK #6 recorded, no fail-closed shipped; guard edit is log-only (payload_keys, :75-86).
6 MET -- bash -n exit 0; M3 apostrophe cell KILLED; guard proven ALIVE by my own write.

VERDICT DIRECTION: CONDITIONAL (1st for this step-id in harness_log; no auto-FAIL).

COMPLETED: 2026-08-11T13:23:05Z
