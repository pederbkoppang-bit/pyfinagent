STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.71
WRITTEN: 2026-08-17T10:54:23Z

# Q/A write-first record -- step 86.71, cycle 2 (per Main's ADDITIONAL CONTEXT attempt_number: 2)

## Plan
A. harness-compliance audit (5 items)
B. deterministic: immutable command; git status/diff scope; ruff lint; runtime smoke
C. criterion-by-criterion judgment + independent mutation of the new guards

## Findings log (appended as established)

### Prior-attempt / sequence evidence
- `qa_wip.py 86.71 --spawned-at 2026-08-17T10:54:23Z`: attempt_number=2,
  prior_attempts=1, source_present=true, attempt_number_status=ok,
  attempt_number_is_lower_bound=false, identity_checked=true, records_retained=2
  (gauge, not counter). prior_records = verdict_wip_86.71__20260817T103515Z.md.
- `verdict_history_86_21.py --step 86.71 --evidence-only`: status=ok,
  detail="1 verdict(s) from the ledger", verdicts = FAIL.
- CROSS-CHECK: attempt_number(2) vs ledger verdict count(1) -> consistent
  (this spawn is attempt 2 and is not yet in the ledger). No staleness signal.

### B. Deterministic
- IMMUTABLE COMMAND exit=0, stdout "parses".
  `bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"scripts/harness/attempt_budget.py\").read()); print(\"parses\")"'`
- Derived .py scope `git diff --name-only HEAD -- '*.py'` = 8 files (non-empty
  guard satisfied). `uvx ruff check --select F821,F401,F811 $FILES` ->
  "All checks passed!" exit=0.
- Cycle-2 diff for THIS step vs HEAD: attempt_gate.py +13/-3 (DOCSTRING ONLY),
  mutation_matrix_86_71.py +99/-4, experiment_results_86.71.md +29,
  live_check_86.71.md +73. Other dirty .py belong to 86.84/86.85/peer UI.
- md5 before/after all my runs: attempt_gate.py ceac76e744614cefb749fe3782d5c53b
  (unchanged, and equals the md5 the matrix prints), audit ledger
  e20e47e3603f6f26b8ff0e79f72998dc unchanged. No tree mutation by me.

### C8 -- MUTATION MATRIX, independently re-run and independently mutated
- I re-ran `python3 scripts/qa/mutation_matrix_86_71.py --verify`: exit 0,
  control green (7 checks), relocated-unmutated SURVIVES, null-mutant SURVIVES,
  6/6 KILLED, byte-identical restore ok.
- VERBATIM-CAPTURE CHECK: I extracted the fenced block from live_check section 7
  and diffed it line-by-line against my freshly regenerated stdout ->
  23 lines vs 23 lines, 0 mismatches, IDENTICAL. Cycle-1's Invalid_Precondition
  (edited capture) is CLOSED.
- INDEPENDENT MUTATION OF THE NEW GUARD (the thing cycle 2 added):
  reverted the PYTHONPATH env fix in drive() IN MEMORY (exec with __file__
  pinned; no repo write) -> "DISCRIMINATION CONTROL RED ... every kill below
  would be a mirage", rc=1, 5 checks listed. The new guard is load-bearing.
- SECOND independent mutation: disabled the relocated-unmutated control AND
  reverted PYTHONPATH -> "NULL-MUTANT CONTROL RED", rc=1. Both controls fire
  independently; neither is decorative.
- PER-CELL FULL FAILURE LISTS (the matrix prints only failed[0]; I printed all):
  G1 fails 4 at-ceiling checks; G2 fails exactly 1 (rows_after 1->0);
  G3 fails 4; G4 fails EXACTLY the new corrupt check (below rc=0 rows=1,
  at rc=2 -- i.e. G4's kill is targeted and owned by the matrix, closing
  cycle-1's real survivor); G5 fails exactly the exit-2 check (message and
  escalation still emitted, as described); G6 fails 4.
  corrupt_tagged=True on 5 of 6 cells -> the new probe is not a blanket killer.

### A. Harness compliance (5 items) -- CLEAN
1. research-gate-before-contract: research_brief_86.71.md envelope brief_status
   COMPLETE, external_sources_read_in_full=9, urls_collected=30,
   recency_scan_performed=true, gate_passed=true. Genuinely adversarial (source
   #5 Fowler REFUTES the module's own docstring claim, and the docstring was
   corrected in response).
2. contract-before-generate (mtime, local): research 11:47:50 < contract
   12:07:27 < code 12:52:41/12:52:58 < results+live_check 12:53:31. Cycle-1
   critique 12:49:17 sits between, as a cycle-2 must. All 8 criteria are
   VERBATIM present in the contract (programmatic string-containment check vs
   .claude/masterplan.json). Criterion-1 numbers appear in the CONTRACT
   (lines 28-29), i.e. before the fix was designed.
3. experiment_results_86.71.md present, with a Cycle-2 section.
4. log-last: `grep -cF "phase=86.71" handoff/harness_log.md` = 0 (grep proven
   live by phase=86.85 -> 2 and phase=86.94 -> 2); masterplan status=pending.
5. no-verdict-shopping: evidence CHANGED (mutation_matrix +99/-4, attempt_gate
   +13/-3, live_check +73, experiment_results +29). A verdict move off the
   cycle-1 FAIL is the documented cycle-2 flow, not sycophancy.

### Criteria, independently re-derived (NOT taken from cycle 1)
- C1 numbers: my own implementation of the stated population rule over 590
  wf_*.json records -> 326/491 = 66.4%; qa 397 total / 308 repeats = 77.6%;
  researcher 93/18 = 19.4%; max = 36.8 with 9 qa runs. Main filed 66.5% /
  77.4% / 20.0% / 9 (481 attributable then, 491 now). Reproduces.
  Back-cast: oldest-513 = 64.7%, oldest-527 = 64.8%, pre-2026-08-13 = 64.5%.
  So the adopted decomposition (growth ~1.7pt, population-rule ~6.4pt)
  reproduces exactly. -> numbers MET.
  ** COMMAND CLAUSE NOT MET **: live_check section 7 (2) shows
  `$ python3 - <<'PY'` whose body is comments plus a literal
  `...  # (the exact script is quoted in full in the session transcript ...)`.
  No runnable command for these numbers exists in the handoff tree. Second
  cycle on the same clause (cycle-1: "the command is absent").
- C2: at the PRE-FIX commit 192ef652^, `git grep -ln attempt_budget` hits
  backend/tests/test_phase_86_32_attempt_budget.py (named control, non-zero)
  and returns NOTHING over scripts/harness|backend|.claude/hooks|.claude/workflows
  minus the module and tests. I added a second, independent control string
  (`verdict_ledger_write`, a genuinely wired module) -> 3 hits. MET.
- C3: 7 SEPARATE OS processes against a temp ledger; each `--status` read in
  yet another process: 1/5..5/5 CONTINUE then rc=2 deny at proc#6 and #7,
  escalation file written, body contains "NOT A PASS" and --operator-extend. MET.
- C4: jq returns the registered PreToolUse/Workflow command. LIVE PROOF I did
  not have to take on trust: the production ledger carries
  `2026-08-17T10:54:19Z attempt 86.71 qa-verdict.js attempt_number_inclusive=2`
  -- MY OWN spawn, written by the hook process, corroborating qa_wip's
  attempt_number=2 from an independent source. --status: 999.2 = 5/5 ESCALATE
  deny; 86.71 / 86.85 = 2/5 CONTINUE allow. MET (with the disclosed
  Agent-tool bound).
- C5: exhaustive 1,452 (non-PASS sequence x flag-combo) evaluations. close_kind
  value set over ALL of them = {CONTINUE, ESCALATE}; ZERO green closes.
  5xFAIL -> ESCALATE under all four flag combos. 1xPASS is the only route to
  CLOSED_COMPLETE / CLOSED_PRODUCT_RESIDUALS_QUEUED. 15/15 module tests pass;
  --self-test 12/12. MET.
- C6: the gate counts every Workflow launch with a step_id; the production
  ledger's rows are all `qa-verdict.js` (86.71 x2, 86.85 x2, 86.84 x1). MET.
- C7: `git diff HEAD -- '*.env' '*/.env'` empty; graded commit file list has no
  .env; .claude/settings.json diff is ONLY the hook registration block -- no
  setting flipped. ASK-1 present in contract:88. MET.
- C8: see below.

### C8 -- the remaining gaps (my own mutations, tree md5 unchanged throughout)
Mutated in memory, relocated with PYTHONPATH, --self-test as the oracle
(control rc=0 observed FIRST):
  H1 hostile step-id refusal removed        -> KILLED (FAIL "hostile step id refused")
  H2 PASS exception removed                 -> KILLED (FAIL "verdict-ledger PASS -> allow")
  H3 operator extension ignored             -> KILLED (FAIL "operator extension re-opens...")
  H5 corrupt row skipped (== matrix G4)     -> KILLED (FAIL "corrupt row counts as an attempt")
  H4 `--reason` requirement dropped
     (`if not reason.strip():` -> `if False:`) -> ** SURVIVED **, rc=0
H4 is a NEW guard from this step -- the accountability guard on the ONLY path
that raises the ceiling -- with ZERO coverage: no matrix cell touches
cmd_extend and --self-test never calls it. Criterion 8 says "every new guard".
H1/H2/H3 ARE genuinely killable but only by --self-test, and the handoff
presents the 6-cell matrix as its C8 evidence, so that coverage is undisclosed.
LATENT (not firing today): _corrupt_probe catches JSONDecodeError and never
asserts `rc == 0`, so a mutant that cannot import scores corrupt_tagged=False
= a kill on that check. Measured probe_rc=0 on all six cells and masked by the
two controls, but it is the smaller form of the exact class cycle 1 found.

### Carried NOTES (from cycle 1, still live, non-blocking, no criterion owns them)
- 5 `session_id=pipetest` rows for synthetic step 999.2 sit in the production
  audit stream handoff/audit/attempt_budget_audit.jsonl.
- Literal-string keying: 999.2 -> 5/5 deny, while 999.20 and 999.2.0 -> 0/5
  allow (measured just now).
- Scope note: Main's spawn disclosure named six sovereign-UI files +
  perf_results.tsv as the peer-session dirt, but 5 further .py files
  (rail_turn_cap, mutate_rail_turn_cap, mutation_matrix_86_85,
  verdict_ledger_write, test_phase_86_85_*) are also dirty. I verified NONE of
  them references attempt_budget/attempt_gate (grep count 0), so no
  contamination of this step; ruff clean over all 8.

### VERDICT FORMED: CONDITIONAL
Every cycle-1 blocking finding is genuinely closed and I verified each one
independently rather than by reading the claim. No criterion is materially
unaddressed and there is no unintended production change. Two fixable
evidence gaps cap it: C1's "command stated" clause unmet for a second cycle,
and C8's "every new guard" with one guard (H4) at zero coverage.

COMPLETED: 2026-08-17T11:04:27Z
