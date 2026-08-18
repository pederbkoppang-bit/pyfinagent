STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-15T14:01:37Z
COMPLETED: 2026-08-15T14:13:14Z

# Q/A write-first record -- step 86.85 (EVALUATE)

Role: sole Layer-3 Q/A evaluator. Read `.claude/agents/qa.md` in full at runtime (done, first tool call).

## Plan
- A. harness-compliance audit (5 items)
- B. deterministic: immutable command + git scope + lint + scoped tests
- C. LLM judgment vs 8 immutable criteria + mutation matrix

## Log (append-only)
- 14:01:37Z created record.

## A. Harness compliance (5 items)
- research gate: research_brief_86.85.md present, envelope COMPLETE, gate_passed true, 8 read-in-full, 23 URLs, recency scan true -- VERIFIED by reading contract sec1 (envelope quoted); to re-derive from the brief itself.
- contract-before-generate: mtime check pending.
- experiment_results_86.85.md present (16426 bytes).
- log-last: masterplan 86.85 status = "pending" (not flipped). harness_log check pending.
- no-verdict-shopping: evidence CHANGED. cycle-1 = FAIL (ledger row). commit 5a3b0766 modifies writer +77/-x, matrix +27, results +106. CHANGED -> legitimate fresh respawn.

## B. Deterministic
- IMMUTABLE COMMAND: `parses`, exit 0. REPRODUCED.
- ruff F821,F401,F811 over git-derived scope (scripts/qa/mutation_matrix_86_85.py, scripts/qa/verdict_ledger_write.py): "All checks passed!" exit 0.
- writer --self-test: 13/13 ok, exit 0. REPRODUCED.
- qa_wip.py 86.85 --spawned-at 2026-08-15T14:01:37Z -> attempt_number=2, prior_attempts=1, source_present=true, status ok.
- verdict_history_86_21.py --step 86.85 --evidence-only -> status=ok, verdicts: FAIL (1 row).
- verdict_history_86_21.py --step 86.74 --evidence-only -> status=ok, 8 verdicts (was no_rows_for_step before this step).
- cross-check: attempt_number(2) = ledger count(1) + current spawn -> ledger NOT stale for this step.
- unrelated uncommitted change in tree: backend/api/sovereign_api.py (+1y window) -- NOT in either 86.85 commit; unattributable to this step.

## C. Independent drive of shipped enforceEscalation (brace-extracted from qa-verdict.js, NOT a copy)
CTRL 1 prior C + CONDITIONAL      n=1 auto_fail=false status=ok
DRIVEN 2 prior C + CONDITIONAL    n=2 auto_fail=true  status=ok
CTRL 2 prior C + PASS             n=2 auto_fail=false
CTRL 2 prior C + FAIL             n=2 auto_fail=false
3 prior C + PASS                  n=3 auto_fail=false   <- non-PASS never becomes PASS
[C,C,NO_VERDICT] + CONDITIONAL    n=2 auto_fail=true    <- drop does NOT clear
absent/null sequence              n=null auto_fail=null status=not_supplied  <- unknown != 0
ORDER [PASS,C,C]+C                n=2 auto_fail=true
ORDER rev [C,C,PASS]+C            n=0 auto_fail=false   <- materiality of the ordering bug confirmed
oov token                         n=null status=unparseable
86.74 real priors + CONDITIONAL   n=2 auto_fail=true
ALL AUTHOR CLAIMS IN C4/C6/C7 REPRODUCE EXACTLY.
- scripts/qa/verify_escalation_86_78.mjs: 51 checks, 0 failed, ALL CHECKS PASS.

## D. INDEPENDENT MUTATION MATRIX (Q/A-authored, 8 cells, temp copies, sha256 unchanged)
CONTROL GREEN first (rc=0). Target sha256 identical before/after.
QA-M1 existing_keys -> set()                 KILLED
QA-M2 emit_sequence drops step filter        KILLED
QA-M3 read_rows truncates to last row        KILLED
QA-M4 _dedup_key drops step_id               ***SURVIVED*** (differential CONFIRMED)
QA-M5 verdict stored lowercased              SURVIVED but EQUIVALENT (see below) -> NOT a finding
QA-M6 append_row swallows OSError            ***SURVIVED*** (differential CONFIRMED, MATERIAL)
QA-M7 append_row writes the row twice        KILLED
QA-M8 VALID_VERDICTS loses NO_VERDICT        UNSCORABLE (rc=1 w/o SELF-TEST FAILED; matrix-correct scoring)

### QA-M6 DIFFERENTIAL (first probe did NOT discriminate; re-probed to reach the write)
first probe (ledger path = a directory) -> BOTH exit 1 traceback: read_rows dies first. discarded.
discriminating probe (read-only parent dir 0o500):
  BASELINE exit=4  row_file_created=False  stderr='verdict_ledger_write: failed to append to ...'
  MUTANT   exit=0  row_file_created=False  stdout='{"step_id": "77.7", ... }'  stderr=''
=> a SILENT writer: exit 0, prints the row, nothing on disk. This is exactly the state the
   module docstring forbids ("a silent writer manufactures the LEDGER_EMPTY state the reader
   is built to refuse") and exactly criterion 6's "prove the recorder ran before treating a
   zero as evidence". NO self-test check and NO matrix cell covers this guard.

### QA-M4 DIFFERENTIAL (first probe used cycle keys and never reached the mutated line)
discriminating probe (SAME run_id, DIFFERENT step ids):
  BASELINE step88.1 exit=0 | step99.9 same run_id exit=0 | rows=2
  MUTANT   step88.1 exit=0 | step99.9 same run_id exit=2 | rows=1  (legitimate row REFUSED)
=> real differential; low trigger probability (run ids are uuids) -> lesser finding.

### QA-M5 ruled EQUIVALENT (avoids a plausible-but-wrong finding)
emit_sequence normalises .strip().upper(); the REAL reader verdict_history_86_21.py:169
also does `verdicts.append(v.strip().upper())`. So a lowercased on-disk verdict is
indistinguishable to every consumer in the tree. NOT reported.

## E. CLAIM AUDIT (re-derived, population = every non-blank line of handoff/verdict_ledger.jsonl)
PRE-state at d1c4a79d~1 : 35 rows, 10 step_ids, 86.74=0, recorded_by {main:35}, max date 2026-08-11,
                          run_id present/non-empty/wf_-prefixed = 35/35/35, recorded_at 21/35 -> 14 absent
AT d1c4a79d             : 43 rows  (author's stated 43 REPRODUCES at the stating commit)
CURRENT                 : 44 rows, 12 step_ids, FAIL 6, recorded_at 30/44, 86.85 rows = 1
=> experiment_results sec2 and live_check sec8 state 43 / FAIL 5 / 29-of-43 with NO commit anchor.
   Drift is exactly +1: this step's OWN cycle-1 FAIL row. Self-referential count. The same
   document DOES anchor its C2 remediation figure to d1c4a79d~1, so the anchor was known.
12 rows sharing one microsecond recorded_at: VERIFIED (2026-08-11T08:02:38.670436+00:00 x12)
86.21 positive control at PRE state: [C,C,FAIL,C,C] VERIFIED
10 WIP artifacts for 86.74: VERIFIED (ls | wc -l = 10)
harness_log phase=86.74 rows: 190,191,192,193,195,196 -- cycle 194 absent VERIFIED; '## Cycle 193 ' x2,
'## Cycle 195 ' x2 -> cycle numbers NOT unique VERIFIED
C3 cross-process: 3 separate invocations, write/write/read -> ["CONDITIONAL","CONDITIONAL"] VERIFIED
idempotency on a COPY of the real ledger: replay of (86.74, wf_8c3730a1-32e) exit=2, rows 44->44 VERIFIED

## F. HARNESS-VACUITY PROBE (the matrix itself)
Copied mutation_matrix_86_85.py to tmp (TARGET repointed absolute), added an INERT cell
(one docstring word). Output: `QAX-INERT  SURVIVED`, `8 cells: 7 killed, 1 survived`,
exit=1. The matrix harness is NOT vacuous -- it reports survivors and refuses to exit 0.
Real writer/matrix md5 unchanged; `git status --short scripts/qa/ handoff/verdict_ledger.jsonl`
= empty. ZERO repo writes made by this evaluation.

## G. C7 SWEEP (independent)
4 verdicts x 32 flag masks = 128 combos. 96 wrote and round-tripped the EXACT verdict
through a SEPARATE read process; 32 unkeyed combos refused exit=3; 0 findings.
Case-normalisation only: 'pass'->PASS, 'conditional'->CONDITIONAL, 'PASSED'/'P'/''->exit 3.
No input turns a non-PASS into a PASS. Consumer side: 3 prior C + current PASS -> auto_fail=false.

## H. C2 RESIDUALS
(a) research_brief_86.85.md still carries "33/35" at lines 29 (envelope `summary`), 126 (F4)
    and 182 (C) -- three counts of ledger rows with NO population rule and NO command. A
    correction NOTE was added at :115 ("states three times ... no predicate yields 33") but the
    three occurrences were annotated, not replaced. Annotate-don't-rewrite IS the documented
    convention for a dated gate artifact, and the remediation text does say "marked in place",
    so this is a NOTE-level residual -- but line 29 sits ABOVE the note, inside the envelope.
(b) experiment_results sec2 + live_check sec8 headline block does NOT reproduce today under the
    stated population and quoted command: stated 43 / FAIL 5 / 29-of-43 / 11 step_ids;
    measured now 44 / FAIL 6 / 30-of-44 / 12 step_ids. Reproduces EXACTLY at d1c4a79d.
    Drift = +1 = this step's own cycle-1 FAIL row (self-referential). No `as of <sha>` anchor,
    though the SAME document anchors its C2 remediation figure to d1c4a79d~1.

## I. CONTRACT PLAN ITEM SILENTLY DROPPED
contract_86.85.md sec6.3 promises backend/tests/test_phase_86_85_verdict_ledger_write.py with
5 named tests. It does not exist (`find . -name '*86_85*'` -> only the matrix + artifacts;
`grep -rn verdict_ledger_write backend/tests/ tests/` -> no hits). Neither experiment_results
sec1 ("What was changed") nor sec4 ("What I could NOT verify") discloses the substitution.
Consequence: the writer's ONLY regression coverage is its own --self-test, and NOTHING invokes
it automatically (grep for verdict_ledger_write outside the two scripts + handoff artifacts:
no callers). WARN: scope honesty + test-coverage-delta.

## J. CRITERION ROLL-UP
C1 MET   -- every localisation number re-derived from git at d1c4a79d~1; positive control 86.21
            [C,C,FAIL,C,C]; re-scope test genuinely performed (10 WIP files, cycle 194 absent,
            duplicate '## Cycle 193 '/'## Cycle 195 ' headers). Research brief committed
            2026-08-14T21:41 < contract 2026-08-15T15:44 (git). Contract+code share ONE commit
            so intra-commit order is not gettable from git; localisation is reproducible
            against the pre-build tree, which is the substantive requirement.
C2 PARTIAL -- rule+command present for the headline block, but the numbers no longer reproduce
            (H.b) and three ledger-row counts in the brief still lack rule/command (H.a).
C3 MET   -- reproduced myself across three separate python invocations.
C4 MET   -- shipped enforceEscalation brace-extracted and driven; anti-vacuity control
            (1 prior -> false) present; 86.74 real priors -> n=2 auto_fail=true.
            args.verdict_sequence IS the consumed input (qa-verdict.js:514). Live-launch limit
            disclosed in sec4.5/sec9.4.
C5 MET   -- 86.45 half MEASURED (NO_VERDICT `continue`), 86.79 half correctly out of scope.
C6 MET on the SHIPPED code (drop does not clear: [C,C,NV]+C -> n=2 true; absent -> null not 0).
C7 MET   -- 128-combo sweep, 0 findings.
C8 NOT MET -- procedure is exemplary (control GREEN first, temp copies, sha unchanged, UNSCORABLE
            scoring, harness proven non-vacuous) but "EVERY new guard" fails: the fail-loud I/O
            guard (own exit code 4, own docstring section) and the step_id component of the
            dedup key have NO self-test check and NO matrix cell. QA-M6 SURVIVES the full
            13-check suite and yields a SILENT writer (exit 0, row printed, nothing on disk) --
            the exact fail-open state criterion 6 names.

VERDICT: FAIL. Primary = C8 (demonstrated surviving mutant, material differential, headline
guard). Secondary = C2 residuals. Cycle-1 blockers ARE genuinely fixed and verified; do not redo.
