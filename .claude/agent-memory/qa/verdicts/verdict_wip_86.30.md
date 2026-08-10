STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.30
WRITTEN: 2026-08-10T15:03:29Z
CYCLE: 2 (predecessor cycle-1 run DROPPED without StructuredOutput)

NOTE ON THIS FILE: the WIP path is FIXED per step, so creating this file per the
qa.md write-first rule OVERWROTE the predecessor's cycle-1 record in the working
tree. It is NOT lost -- it was committed in 5285699b and is recoverable with
`git show 5285699b:.claude/agent-memory/qa/verdicts/verdict_wip_86.30.md`. I read it
from git before writing anything further. Reported in `notes` as a mechanism hazard.

## A. Harness compliance (5 items) -- re-measured by me
1. research-gate-before-contract: research_brief_86.30.md mtime 13:02:06 < fix commit
   63074429 (13:47:35). Gate PASSED (wf_8dfd196f-3fa, 9 sources / 42 URLs). OK.
2. contract-before-generate: **BREACH CONFIRMED**. contract_86.30.md mtime 13:46:16,
   AFTER fix 13:44:17 and test 13:45:09. Self-disclosed in a head banner; single
   commit, nothing backdated. Un-repairable. harness_compliance_ok = FALSE.
3. experiment_results_86.30.md (14:02:38) + live_check_86.30.md (13:47:14) present.
4. log-last: `grep -F "phase=86.30" handoff/harness_log.md` -> ZERO. masterplan 86.30
   status=pending, retry_count=0. Correct ordering. 3rd-CONDITIONAL: 0 priors -> not triggered.
5. no-verdict-shopping: cycle-1 produced NO verdict (rail drop). Evidence CHANGED
   (5285699b: test file +81/-13, experiment_results +122). Not shopping.

## B. Deterministic
- IMMUTABLE CMD `pytest backend/tests/test_phase_86_27_live_origin_class.py -q`
  -> **exit 0, 50 passed in 9.79s**.
- `git diff 63074429 -- scripts/qa/live_backend_origin.py` EMPTY; md5
  2669cbe069b026e2d9590e37a5d275cd identical to 63074429. B5 claim TRUE.
- Frozen table test_phase_86_6_subprocess_channel.py md5 d9f3650c4054c2504c1bbfaccea25629
  == md5 at 63074429~1. BYTE-UNCHANGED (criterion 3).
- `git diff HEAD --stat`: no dirty production/test file. Only agent-memory WIP, audit
  jsonl, heartbeat, archive-baseline.
- ruff F821,F401,F811 over git-derived scope (3 .py, non-empty asserted) -> exit 0.
- Runtime smoke: module execs; enumerable=True; example.com:8000 False, 127.0.0.1:8000
  True, 192.0.2.55:8000 False. Normal path intact.

## C. Criterion 1 -- MY OWN reproduction (not the dropped record's)
Own addresses derived at runtime TWO ways: psutil=16, ifconfig=16, symmetric difference [].
Own GLOBAL IPv6 = 6.
  PRE-FIX (63074429^, psutil blocked, interfaces_enumerable()==False asserted):
    own-global-v6 REMOTE 6/6 | FULL-own REMOTE 6/16 | address_is_live_backend NOT-refused 6/6
    witness 2001:4654:6451:0:15d6:967b:7384:d0a | controls (127.0.0.1,::1,CF)=(True,True,False)
  POST-FIX (HEAD): 0/6, 0/16, 0/6 | controls (True,True,True)
=> criterion 1 MET, both predicates, independently.

## D. B2 -- corrected mechanism VERIFIED TRUE by me
  block-only  -> interfaces_enumerable = False (branch REACHED)
  evict-only  -> interfaces_enumerable = True  (branch NOT reached)
  both        -> False
The shipped `_NoPsutil` docstring (lines 67-89) now states exactly this. TRUE.
RESIDUAL: experiment_results_86.30.md:50 STILL states the OLD INVERTED claim as fact
("Evicting sys.modules['psutil'] is the load-bearing half") with NO supersession marker
at the point of the claim; the correction is ~100 lines later in the CYCLE 2 section.
Same class: the cycle-1 criterion table row 6 (line 42) still reads "3 cells, all KILLED
| MET", which cycle 2 itself proved incomplete. Main's claim "Corrected in all three
places" reproduces for the docstring and the contract, NOT for this file. -> WARN.

## E. Mutation matrix -- MINE, mirrored scratchpad tree (repo never written)
Mirror at scratchpad/mut/{backend/tests,scripts/qa}; the test resolves SRC via parents[2].
Anchor `and refusing is the safe answer.\n        return True` asserted unique (count==1).
HEALTHY env:
  C0-CONTROL                                rc=0  9 passed
  M1-REVERT (not ip.is_global)              KILLED  3 failed
  M5-REFUSE-ALL (predicate -> True)         KILLED  2 failed
  M6-NOT-V4-GLOBAL                          KILLED  1 failed   <- B3 FIX CONFIRMED
  M7-V6-OR-NOTGLOBAL                        KILLED  1 failed   <- B3 FIX CONFIRMED
  N1-NOT-MULTICAST   `not ip.is_multicast`      *** SURVIVED ***
  N2-NOT-RESERVED    `not ip.is_reserved`       *** SURVIVED ***
  N3-NOT-MCAST-OR-RSVD                          *** SURVIVED ***
  N4-NOT-DOCUMENTATION (2-addr literal list)    *** SURVIVED ***
BEHAVIOURAL DIFFERENTIAL (degraded mode, shipped==True everywhere):
  N1/N3 differ on 224.0.0.1, ff02::1, 239.255.255.250 -> call them REMOTE (allow)
  N2/N3 differ on 240.0.0.1, 255.255.255.255         -> call them REMOTE (allow)
  `is_live_backend("http://224.0.0.1:8000/api/health")` shipped=True; under N1 it allows.
NON-EMPTY differential => genuine survivors, not equivalent mutants. This CORRECTS the
dropped record, which called M8-NOT-MULTICAST "EMPTY differential -> equivalent": its
differential set contained no multicast address, so it could not discriminate.
CLASS: N1/N2/N3 are the SAME defect shape the step exists to remove -- an IP PROPERTY
test standing in for ownership -- and are natural-language-defensible, so a maintainer
could plausibly write one. Criterion 2's MANDATED assertion scope (own full set +
GENUINELY_REMOTE on the healthy path) is satisfied; its universal headline is not pinned.
NAMED FIX (one list): add a NON-own, NON-GENUINELY_REMOTE odd-class set
{224.0.0.1, ff02::1, 240.0.0.1, 255.255.255.255, 203.0.113.7, 100.64.0.1, fe80::dead:beef}
to the degraded assertion; kills N1-N4 at once.

## F. B4 -- fixed as claimed, with an UNDISCLOSED residual I measured
Verified with a process-wide sitecustomize __import__ block (precondition asserted:
`import psutil` raises). Suite -> **5 passed, 4 skipped, exit 0**. All four
TestDegradedBranchRefuses tests RUN, plus TestCriterion2FullAddressSet's full-set test
via the new ifconfig fallback. The 4 skips are the 3 TestNormalPathIsUntouched tests +
test_healthy_path_still_calls_remote_addresses_remote, all `interfaces not enumerable here`.
RESIDUAL, MEASURED:
  psutil-BLOCKED  C0-CONTROL     rc=0  5 passed, 4 skipped
  psutil-BLOCKED  M1-REVERT      rc=1  KILLED (3 failed)
  psutil-BLOCKED  M5-REFUSE-ALL  rc=0  *** SURVIVED *** 5 passed, 4 skipped
i.e. in the exact environment this fix targets, the suite CANNOT distinguish the fix
from a guard destroyed entirely -- because the anti-vacuity control is one of the skips.
experiment_results §"Criterion 4 -- the anti-vacuity control" names that very test as the
thing that stops "refuse everything unconditionally". Main's "the four remaining skips
are healthy-path tests, correctly inapplicable" is literally true but does not follow
through to this consequence. NAMED FIX, machinery already in the file: synthesise a
healthy path without psutil by setting mod._own_cache = frozenset(_own_addresses_via_ifconfig())
and mod._own_enumerable = True, then run the anti-vacuity control there. -> WARN.

## G. Criterion grading
1 MET (C, independently re-derived, both predicates)
2 MET over the mandated scope (0/16 full own set; GENUINELY_REMOTE remote on healthy path);
  WARN residual N1/N2/N3 (E)
3 MET (immutable cmd exit 0 / 50 passed; frozen table md5 identical to 63074429~1)
4 MET (positive control asserts interfaces_enumerable() is False; M1-REVERT KILLED in
  BOTH the healthy and the psutil-blocked environment)
5 MET (lsof reproduced by me byte-for-byte, same pid 43839, IPv4-only; psutil 7.2.2
  importable and declared in NO requirements file; "not reachable in practice" TRUE)
6 MET literally (revert mutated and killed; matrix honestly scoped -- Main's own
  "licenses 'these four were killed', nothing global"); WARN residual (E)

## H. Verdict issued: CONDITIONAL (ok=false, harness_compliance_ok=false)
Ceiling is CONDITIONAL because harness compliance is NOT clean (B1). Not FAIL: every
immutable criterion is MET on evidence I executed myself, and harness_log has 0 prior
86.30 entries so the 3rd-CONDITIONAL auto-FAIL rule is not triggered.
Three residual findings, all with named fixes: E (survivors), F (anti-vacuity absent
where the branch is live), D (stale inverted claim + stale criterion-6 row).

No `handoff/current/evaluator_critique_86.30.md` exists -- confirmed: this is the FIRST
verdict for 86.30, not a reversal of one.

COMPLETED: 2026-08-10T15:11:04Z
