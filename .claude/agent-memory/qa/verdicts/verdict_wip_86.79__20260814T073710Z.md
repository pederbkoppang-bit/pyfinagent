STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.79
WRITTEN: 2026-08-14T07:37:10Z

# Q/A write-first record -- step 86.79 (qa_wip.py attempt-counter off-by-one)

## A. Harness compliance (5 items) -- ALL CLEAN
- research gate: research_brief_86.79.md envelope brief_status=COMPLETE,
  gate_passed=true, external_sources_read_in_full=10 (floor 5),
  urls_collected=25 (floor 10), recency_scan_performed=true. Contract cites
  run wf_267244ab-91e + the brief path (contract:18-19,24,264). PASS.
- contract-before-generate (mtime chain, all UTC 2026-08-14):
  research 07:05:37 < contract 07:14:15 < qa_wip.py 07:18:56 <
  verify_counter 07:21:42 < mutation_matrix 07:24:57 < live_check 07:35:52 <
  experiment_results 07:36:39. ORDER CORRECT.
- experiment_results_86.79.md present (9,969 B).
- log-last: `grep -Fc 86.79 handoff/harness_log.md` = 0; masterplan
  status = "pending". LOG not run, step not flipped. CORRECT.
- no-verdict-shopping: attempt 1 (see D). Nothing to shop.

## B. Deterministic
- IMMUTABLE CMD reproduced BY ME -> `qa_wip-parses`, **exit=0**.
- `verify_counter_86_79.py` re-run BY ME: **42 checks, 0 failed, exit 0**.
- `mutation_matrix_86_79.py` re-run BY ME: control GREEN first (exit 0,
  42 checks), **7/7 KILLED**, subject sha256[:16] 146600b722a02481 before
  AND after -> tracked file UNCHANGED. Both match the transcription.
- scope: only `scripts/qa/qa_wip.py` (+220/-7) among source files; the other
  6 modified paths are hook-written audit streams + researcher MEMORY.md.
  NO unintended production change. `git diff --stat -- .claude/agents/qa.md`
  EMPTY and `git status --porcelain` EMPTY -> qa.md IS untouched as claimed.
- LINT GATE **RED**. Scope DERIVED (tracked-modified UNION untracked, COUNT=3
  asserted non-empty, xargs -0):
    F401 [*] `datetime` imported but unused
      --> scripts/qa/verify_counter_86_79.py:27:8
  Genuinely dead (`grep -c datetime` = 1, the import itself). qa_wip.py alone
  and mutation_matrix_86_79.py alone are both `All checks passed!` -- so the
  author's documented gate scope would have been GREEN; the union catches it.
  No lint/ruff claim appears anywhere in the handoff.
- 1b/1c/1d N/A (no frontend/**, no UI claim, no backend/** diff).

## C. Independent re-derivation
- ENUMERATION WIDENED: re-ran the prune grep with NO --include filter. Every
  extra hit is .md/.json prose. ZERO executable callers outside the allowlist.
  Claim SURVIVES a wider derivation. Stronger fact I add: `main()` exposes only
  {step_id, --body, --spawned-at} -- prune has NO CLI entry point at all.
- CONSUMER SURFACE (criterion 6): NOTHING parses report(). `.claude/hooks/**`
  references the memory dir only in qa-write-guard.sh's own constant;
  qa-verdict.js mentions qa_wip only inside PROMPT TEXT. No path from this
  change to verdict_gate.py / auto-commit. FAIL cannot become PASS.
- crash-window (context 4) CONFIRMED: `_record_loss` at :301 precedes the
  unlink loop :303-308. Crash between them OVER-counts -> escalates EARLY ->
  safe. Swallowed `except OSError: pass` gives the same safe over-count.
- context 5 CONFIRMED in code: the C2 probe requires BOTH "records_retained"
  AND "prior_attempts + 1" in the guidance string (verify_counter:147-149) --
  the strict form, i.e. the SUBJECT moved, not the probe. The
  RED-WRONG-REASON discrimination is live code (mutation_matrix:159-165) and
  my own cells exercised it.
- context 6 CONFIRMED: `qa_wip.py 86.32` -> records_retained 5, unchanged.
- context 3 CONFIRMED: §11.2 discloses that below the window the flag reports
  False and is "a claim about *automated* loss only". Disclosed, not papered.

## D. Attempt / verdict-sequence counters
- `qa_wip.py 86.79 --spawned-at 2026-08-14T07:37:10Z`: source_present=TRUE,
  records_retained=1, prior_records=[], prior_attempts=0, **attempt_number=1**.
- `verdict_history_86_21.py --step 86.79`: status=no_rows_for_step,
  verdicts=(none), consecutive=0, auto-FAIL armed=False.
- CROSS-CHECK: prior_attempts (0) == ledger count (0) -> ledger NOT stale here.
  harness_log grep = 0, agrees.
- => ATTEMPT 1 of F1b's 5. Prior-verdict sequence EMPTY. Rule NOT armed.

## E. MY OWN mutation matrix (independent of the author's 7)
Through the same PYFIN_QA_WIP_OVERRIDE seam; mutants in tempfile.mkdtemp only;
tracked file byte-identical after (asserted). CONTROL GREEN first.

  KILLED    N3-PRUNE-OFF-BY-ONE               6 fails
  KILLED    N4-ATTEMPT-NUMBER-EXCLUSIVE       4 fails
  KILLED    N5-DROP-LOSS-ADDBACK-MATCHED      3 fails
  SURVIVED  N1-LOWER-BOUND-HEURISTIC-DISABLED
  SURVIVED  N2-IDENTITY-SELECTION-DIRECTION
  SURVIVED  N6-DROP-LOSS-ADDBACK-UNMATCHED

Field-coverage census (grep of the checker / matrix):
  attempt_number 29/7, prior_attempts 4/2, attempt_number_status 4/0,
  attempt_number_guidance 2/0, records_pruned_known 2/0,
  records_retained_unit 2/1, **attempt_number_is_lower_bound 0/0**.

### Behavioural differentials for the survivors (all EXECUTED)
- **N6 (REAL HOLE).** `prior_attempts = lost_n + len(records)` -> `len(records)`
  on the `no_record_for_this_spawn` path. 6 real attempts pruned to keep=3
  (ledger lost=3): baseline prior_attempts=**6** -> attempt 7 -> ESCALATE;
  mutant prior_attempts=**3** -> attempt 4 -> CONTINUE. That is the exact
  under-count-suppresses-escalation defect this step exists to remove, on a
  branch this step created, on the path a post-drop recovery uses. Root cause
  = §4c vacuity shape 5: C2 exercises that branch with NO ledger (lost_n==0,
  verified: no prune/ledger line in verify_counter:126-153) and C3 exercises
  the ledger only on the MATCHED path. The two are never combined. Also
  survives verify_wip_retention_86_36.py.
- **N1 (REAL HOLE).** `attempt_number_is_lower_bound = True` -> `False`.
  Measured on the LIVE repo, step 86.32: baseline True -> mutant False, i.e.
  the mutant silently claims exactness in the one regime the field exists to
  flag. Zero assertions anywhere. Also survives the 86.36 gate.
- **N2 (NOT a system hole).** `for cand in reversed(records)` -> `for cand in
  records`: baseline recovering spawn 09:00 returns the 09:00 file, attempt 1;
  mutant returns the 11:00 file, attempt 3. Survives THIS checker but IS
  killed by verify_wip_retention_86_36.py on the NAMED assertion "spawn 1
  resolves to cycle 1's record" (exit 1, "RED -- 22 passed, 1 failed"). The
  kill attribution in the qa_wip.py:473-478 comment is therefore CORRECT.

## F. Criterion 4 -- the residual, and a member the author did not enumerate
`.claude/agents/qa.md:622` still reads "`records_retained` is the count of
prior Q/A spawns on this step -- the **attempt number**". Not made to agree.
Loudly disclosed via 3 channels, so NOT silent -- but "made to agree" did not
happen. INDEPENDENT FINDING: the SAME false statement is duplicated in
`.claude/workflows/qa-verdict.js:152` ("records_retained gives the ATTEMPT
number (authoritative)") and :147 -- the PRIMARY launch rail's prompt, and the
one that spawned me. The patch file / experiment_results / live_check mention
`qa-verdict` **0 / 0 / 0** times. That file is NOT under `.claude/agents/`, so
the separation-of-duties blocker the author cites does not apply to it: routes
A/B would half-fix the consumer class.

## G. NOTES
- `EXPECTED_CHECKS = 30` while 42 run -> 12 checks of slack; the comment says
  "raise it when adding checks" and it was not raised.
- Criterion 5's 3rd-CONDITIONAL half is verdict-keyed and structurally
  independent of attempt_number, so "against the corrected number" is
  literally true only for F1b. Substantively met.

## H. Disposition reached (still NOT a verdict -- see qa.md)
CONDITIONAL. 6 of 7 criteria MET and independently reproduced; C4 not met as
written; lint red; two proven-behavioural survivors. Attempt 1, no prior
verdicts, so no auto-FAIL trigger. Nothing here can turn a FAIL into a PASS.

COMPLETED: 2026-08-14T07:51:53Z
(the first write of this line carried an INVENTED time, 08:02:31Z, typed
without reading a clock; corrected against `date -u` immediately after.)
