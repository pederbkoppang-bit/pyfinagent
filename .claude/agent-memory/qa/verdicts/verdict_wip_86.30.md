STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.30
WRITTEN: 2026-08-10T11:48:07Z
COMPLETED: 2026-08-10T11:58:59Z

# Q/A write-first record -- step 86.30, cycle 1

Launch: Workflow structured-output rail. Read .claude/agents/qa.md in full at 11:48Z.

## A. Harness compliance (5 items)
1. research-gate-before-contract: research_brief_86.30.md mtime 13:02:06 < code 13:44:17. Gate PASSED
   (wf_8dfd196f-3fa, 9 sources / 42 URLs). OK.
2. contract-before-generate: **FAILS**. contract_86.30.md 13:45:50 is LAST, after fix 13:44:17 and
   test 13:45:09. SELF-DISCLOSED in a head banner; mtimes I re-measured match the banner exactly.
   Nothing backdated (single commit 63074429 at 13:47:35).
3. experiment_results_86.30.md present (13:47:03). live_check_86.30.md present (13:47:14).
4. log-last: harness_log has ZERO `phase=86.30 result=` entries; masterplan 86.30 status=pending,
   retry_count 0. Correct ordering. 3rd-CONDITIONAL rule: 0 priors -> not triggered.
5. no-verdict-shopping: cycle 1, no prior verdict. N/A.

## B. Deterministic
- IMMUTABLE CMD -> **exit 0, 50 passed in 9.82s** (re-run twice).
- backend/tests/test_phase_86_30_degraded_direction.py -> 9 passed, 0 skipped.
- git status: NO uncommitted production changes. Commit 63074429 touches exactly 5 files. Frozen table NOT in it.
- Frozen table md5 = d9f3650c4054c2504c1bbfaccea25629 -- MATCHES; last touched by 9bda4e6d (86.27).
- ruff F821,F401,F811 over commit-derived scope (2 .py, non-empty asserted) -> exit 0.
- Runtime smoke: module execs clean; normal path example.com:8000=False, 127.0.0.1:8000=True. conftest intact.
- Only `not ip.is_global` left in HEAD is inside a COMMENT (line 185).
- Regression -k selection: 81 passed (matches) / 3339 deselected (artifact said 3337; total collected
  3418 -> 3420 between 13:47 and 13:55 with NO test file changed -- some module's collection is
  state-dependent. Load-bearing half reproduces exactly; NOTE only).
- mutation_matrix_86_27.py: all 7 KILLED, tracked source sha-equal.
- Consumers grepped: conftest.py, test_86_6, test_86_27, smoke_cc_rail_e2e.py, reproduce_86_27_spellings.py
  -- all still import fine; 86.27:346 asserts ('192.0.2.55',8000) is False on the healthy path and passes.

## C. Independent reproduction (criterion 1) -- addresses derived by ME at runtime
psutil=16 addrs, ifconfig=16 addrs, symmetric difference []; own GLOBAL IPv6 = 6 by both, symdiff [].
PRE-FIX (git show 63074429^), psutil blocked, interfaces_enumerable=False:
  own GLOBAL v6 REMOTE by _is_this_machine 6/6 (witness 2001:4654:6451:0:15d6:967b:7384:d0a)
  own GLOBAL v6 NOT refused by address_is_live_backend 6/6 ; FULL own set REMOTE 6/16
POST-FIX (HEAD): 0/6, 0/6, 0/16. Controls both runs: 127.0.0.1->True ::1->True LAN->True;
  Cloudflare False->True. is_live_backend('https://example.com:8000') False->True in DEGRADED mode only.
Address quoted in live_check §1 (2001:4654:6451:0:31:6467:1ea6:1852) IS still present in ifconfig.

## D. Mutation matrix (mine, in-memory via pytest.main plugins; repo tree never written)
Anchored on the unique preceding comment line (`        return True` occurs 6x -- same ANCHOR-BAD hazard Main disclosed).
  C0-CONTROL                                  exit 0, 9 passed
  M1-REVERT (not ip.is_global)                KILLED 3 failed
  M2-DELETE-BRANCH (return False)             KILLED 3 failed
  M3-IS-PRIVATE                               KILLED 3 failed
  M4-IS-GLOBAL                                KILLED 1 failed
  M5-REFUSE-ALL (whole predicate True)        KILLED 2 failed (frozen_row + healthy_path control)
        -> the anti-vacuity control is REAL; "refuse everything always" does NOT pass.
  H1-NO-EVICTION (harness)                    SURVIVED 9 passed  -> EQUIVALENT (see E)
  H2-NO-IMPORTBLOCK (harness)                 KILLED 2 failed incl. test_the_branch_is_actually_reached
        -> the positive control is NOT vacuous.
  M6-NOT-V4-GLOBAL  `not (ip.version==4 and ip.is_global)`   *** SURVIVED *** 9 passed
  M7-V6-OR-NOTGLOBAL `ip.version==6 or not ip.is_global`     *** SURVIVED *** 9 passed
  M8-NOT-MULTICAST                            SURVIVED, EMPTY differential -> equivalent, not a finding
  BEHAVIOURAL DIFFERENTIAL of M6/M7 vs shipped, degraded mode:
    differ on 8.8.8.8, 1.1.1.1, 93.184.216.34 -> they call GLOBAL IPv4 "remote" (allow),
    which criterion 2 ("NEVER classifies any address as remote") forbids. Two independently
    constructed spellings agree -> a real class, not a construction artifact.
  ROOT CAUSE of the gap: `GENUINELY_REMOTE` (3 IPv4 + 1 IPv6) is asserted ONLY on the healthy path;
  degraded mode asserts over-refusal for exactly ONE address, the IPv6 Cloudflare one.
  NAMED FIX (one line): assert every GENUINELY_REMOTE entry is `_is_this_machine(a) is True` inside
  `_NoPsutil` too. Verified by construction that this kills M6 and M7.

## E. FINDING -- the stated mechanism is FALSE (H1 root-caused)
Claim, verbatim in experiment_results_86.30.md, contract P1, AND the SHIPPED `_NoPsutil` docstring:
  "a module already in sys.modules is served from cache and an __import__ hook never fires.
   Evicting sys.modules['psutil'] is the load-bearing half"
MEASURED with psutil force-imported and NO eviction: hook fired True, interfaces_enumerable() False
  -> degraded branch REACHED; the block was NOT inert.
MEASURED eviction only, no block: interfaces_enumerable() True -> branch NOT reached.
=> BLOCK is load-bearing, EVICTION is redundant. The artifacts state the exact inverse.
`import x` always calls builtins.__import__; sys.modules is consulted INSIDE it.
REAL mechanism that DOES defeat such a probe, demonstrated: warm the module-level `_own_enumerable`
cache (live_backend_origin.py:126-127) first, then block+evict -> interfaces_enumerable() stays True.

## F. Blast radius + target-environment behaviour (measured)
Degraded-mode over-refusal is correctly SCOPED TO PORT 8000:
  (CF,8000) refuse=True / (CF,443) False ; (8.8.8.8,8000) True / (8.8.8.8,53) False ;
  (127.0.0.1,8000) True / (127.0.0.1,5432) False.
With psutil unimportable PROCESS-WIDE (the env the fix targets): suite exits 0 with **4 passed, 5 SKIPPED** --
including test_degraded_mode_calls_NO_own_address_remote (criterion 2 full-set) and
test_healthy_path_still_calls_remote_addresses_remote (the anti-vacuity control). `_all_own_addresses()`
returns [] without psutil even though the file already carries an ifconfig-based derivation.

## G. Criterion grading
1 MET (independently re-derived, both predicates) | 2 MET as written, WARN residual (M6/M7) |
3 MET (50/50 exit 0, md5 match, matrix 7/7) | 4 MET (H2 proves the control; M1 kills the revert) --
  but the stated REASON the harness works is false (E) | 5 MET (lsof reproduces byte-for-byte, psutil
  7.2.2 importable and in no requirements file, IPv6 curl=000 vs IPv4 200 -> not reachable, claim TRUE) |
6 MET literally (revert mutated + killed) but the "no mutant survives" bar is not fully cleared.

## H. Verdict issued: CONDITIONAL (ok=false, harness_compliance_ok=false, certified_fallback=false)
Blockers: (1) contract-before-generate breach, (2) false mechanism claim in SHIPPED source,
(3) M6/M7 mutation survivors on criterion 2, (4) suite self-disables where the branch is live.
No criterion is materially unaddressed, so not FAIL. 0 prior CONDITIONALs -> auto-FAIL rule not triggered.
