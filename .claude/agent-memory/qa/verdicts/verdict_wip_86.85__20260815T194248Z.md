STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-15T19:42:48Z

# Q/A write-first record -- step 86.85 (cycle 4)

Criterion supplied by the caller (1 of the masterplan's 8):
  "mutation-test every new guard with the control observed GREEN first and a
   byte-identical restore"  == masterplan criterion #8.

NOTE: the spawn prompt passed `EVIDENCE / FILES TO READ: [object Object]` and
`ADDITIONAL CONTEXT: [object Object]` -- both args serialized as the literal
string "[object Object]", so the evidence pointer list did NOT arrive. Evidence
set derived from git + handoff/current + .claude/masterplan.json.

## Prior-attempt / prior-verdict EVIDENCE (gathered, not applied)

qa_wip.py 86.85 --spawned-at 2026-08-15T19:42:48Z :
  source_present    : true
  attempt_number    : 4   (status ok, INCLUSIVE of me; is_lower_bound true)
  prior_attempts    : 3
  records_retained  : 4   (gauge, NOT used as a counter)
  records_pruned_known : null

verdict_history_86_21.py --step 86.85 --evidence-only :
  status   : ok
  detail   : 3 verdict(s) from the ledger
  verdicts : FAIL -> FAIL -> FAIL

CROSS-CHECK: prior_attempts (3) == ledger verdict count (3). The ledger is NOT
stale for this step. (attempt_number 4 exceeds 3 only because it counts THIS
spawn, which has no verdict yet.)

## A. Harness-compliance audit (5 items) -- CLEAN

1. research-gate-before-contract -- research_brief_86.85.md exists (36,925 B).
   Envelope quoted in the contract: COMPLETE, 8 read in full, 23 URLs, recency
   scan true, gate_passed true. PASS.
2. contract-before-generate -- contract 15:59 local; cycle-4 artifacts
   21:39-21:42. NOTE: the brief's mtime (16:16) is later than the contract's
   (15:59) because the brief was annotated during the cycle-1/2 corrections.
   Both precede every cycle-4 artifact.
3. experiment_results present -- cycle-4 section appended 21:42:17 local
   (19:42:17Z, 31 s before my spawn).
4. log-last -- masterplan 86.85 status = `pending`; harness_log's newest 86.85
   row is Cycle 197 = the cycle-3 FAIL. No cycle-4 row yet. PASS.
5. no-verdict-shopping -- evidence CHANGED since the cycle-3 FAIL:
     M scripts/qa/verdict_ledger_write.py           (+40)
     M scripts/qa/mutation_matrix_86_85.py          (+45)
     ?? scripts/qa/verify_matrix_coverage_86_85.py  (NEW, 15,330 B)
     M handoff/current/experiment_results_86.85.md  (+75)
     M handoff/current/live_check_86.85.md          (+167)
   Documented cycle-2 flow, not a re-spawn on unchanged evidence.
   SEPARATE OBSERVATION for the caller: HEAD carries 64512cdc "phase-86.85:
   cycle-3 FAIL -- 3rd consecutive, ESCALATED to operator, no cycle 4", and a
   4th cycle nevertheless exists. Recorded; not mine to act on.

## B. Deterministic checks -- ALL GREEN

- IMMUTABLE COMMAND -> `parses`, **exit 0**.
- ruff F821,F401,F811 over a DERIVED scope (git diff --name-only HEAD '*.py'
  UNION git ls-files --others '*.py'; non-empty set of 4 files asserted first,
  passed via xargs) -> "All checks passed!", **exit 0**.
- writer `--self-test` -> SELF-TEST PASSED, **23 checks** (re-counted from the
  output; matches the artifact).
- `mutation_matrix_86_85.py` -> **exit 0**. `CONTROL : rc=0 -> GREEN` printed
  FIRST; 14 cells: **14 killed, 0 survived, 0 unscorable**. sha256 before ==
  after = 3e607f1b02a6a4cb71af9c2893dbce022bcae53b09b74c5e919150b28be0a18d.
- `verify_matrix_coverage_86_85.py` -> guards 15, covered 15, uncovered 0, cell
  problems 0; self-control detected the planted guard AND reported it UNCOVERED.
- pytest -k '86_85 or ledger or verdict_ledger' -> **34 passed**, 3498
  deselected. Matches the artifact exactly.
- md5 of all three scripts identical before and after every run I performed.

## C. Author claims independently REPRODUCED

- "12/12 KILLED while main's CLI validation had NO cell": HEAD matrix has
  exactly **12** cells; dropping M13 or M14 turns the coverage gate RED. OK.
- "M14 SURVIVED against the self-test as it stood": reproduced against
  `git show HEAD:scripts/qa/verdict_ledger_write.py` -- the HEAD writer emits
  **20** checks (matching "before: 20 checks") and **M14 SURVIVES** while M13 is
  KILLED. OK.
- drop-one-cell sweep table (C8.5): every row I re-ran reproduces.
- 23 checks / 34 passed / 14 cells 14 killed / 15 guards 15 covered: all
  reproduce.

## D. My own adversarial mutation cells (control GREEN first; zero repo writes)

  QA-A  `if run:` -> `if False:` in _dedup_key                     KILLED
  QA-B  remove emit_sequence's step filter                          KILLED
  QA-C  remove build_row's pre-I/O `_dedup_key(row)` call            KILLED
  QA-D  out.append -> out.insert(0)  (order inversion, NOT [::-1])   KILLED
  QA-E  `if not line.strip(): continue` -> `if False:`              SURVIVED
  QA-F  `except LedgerError:` -> `except Exception:` (existing_keys) SURVIVED

Behavioural differential on both survivors -- NEITHER is a defect:
- QA-E: removing the blank-line skip makes a blank line raise EXIT_IO, i.e. it
  fails CLOSED. A normalization branch, not a refusal. NOTE only.
- QA-F: measured. CONTROL crashes with an unhandled **AttributeError** on a
  ledger line that is valid JSON but not an object; MUTANT refuses correctly
  with EXIT_DUPLICATE (2). The mutant is equivalent-or-better, so the narrow
  `except` is not shown to be load-bearing. RETRACTED as a finding. It does
  expose a small pre-existing robustness nit (non-dict line -> unhandled
  AttributeError rather than LedgerError). NOTE, outside criterion 8.

So: no genuine surviving mutant. No product defect found this cycle.

## E. KNOWN-MEMBER RECALL TEST on the new coverage checker  <-- THE FINDING

Known-member set taken from the checker's OWN docstring, not chosen by me:
"Three consecutive FAILs on step 86.85 were ONE class: a new guard shipped with
no mutation cell (ordering; then fail-loud-I/O + step_id-in-key; then
cycle-fallback)."

Drop each known member's cell, re-run the gate:

  drop M6      (ordering,        cycle-1 QA-M1)  -> rc=0 GREEN  NOT DETECTED
  drop M8      (fail-loud I/O,   cycle-2 QA-M6)  -> rc=1 RED    DETECTED
  drop M9      (step_id-in-key,  cycle-2 QA-M4)  -> rc=0 GREEN  NOT DETECTED
  drop M11+M12 (cycle-fallback,  cycle-3 QA-M2)  -> rc=0 GREEN  NOT DETECTED
  drop M5      (event/write time)                -> rc=0 GREEN  NOT DETECTED
  drop M13 / drop M14 (the author's own tell)    -> rc=1 RED    DETECTED

RECALL = **1 of 4** known members.

Per-cell measured coverage contribution over the enumerated guard set:
  M5, M6, M9, M11, M12 each cover **0** guards.
  Guards covered by MORE THAN ONE cell: **NONE** -- zero redundancy anywhere.
  Enumerated fingerprints mentioning `return out` / `run:` / `cycle:` / `date`:
  **0 / 0 / 0 / 0** -- those guards are not in the enumerated set at all.

## F. The claim that does not reproduce

experiment_results_86.85.md, under "Proof the gate is load-bearing":
  "Five cells (M5, M6, M9, M11, M12) are coverage-redundant -- another cell
   touches the same *guard* because those five target BEHAVIOURS rather than a
   distinct raise/branch."
live_check_86.85.md C8.5:
  "Coverage-redundant means another cell already touches the same *guard*;
   those five cells target BEHAVIOURS (ordering, sequence filtering, dedup-key
   composition, cycle fallback) rather than a distinct `raise`/branch."

The SWEEP reproduces. The CAUSAL CLAUSE does not:
- "another cell touches the same guard" is FALSE -- no guard is covered by more
  than one cell, and those five cells cover zero enumerated guards.
- The true reason is the second half of the same sentence: their targets are
  invisible to the enumeration rule. "Redundant" says the gate is still complete;
  "invisible" says the gate is structurally blind to the four guards that caused
  cycles 1-3. The reassuring reading is the wrong one.
- Concrete residual risk: a maintainer trusting "coverage-redundant" could delete
  M6 -- the ordering cell the cycle-1 Q/A exposed -- and the gate stays GREEN.
  Bounded, because behavioural coverage of ordering does not depend on M6: the
  self-test carries "sequence is oldest->newest" plus the "order fixture is NOT
  palindromic (anti-vacuity)" guard-on-the-guard.

Second, smaller claim defect: C8.5 names the five cells' targets as "(ordering,
sequence filtering, dedup-key composition, cycle fallback)". **No cell targets
emit_sequence's step filter** (grep: none; I had to write QA-B myself), and M5's
actual target -- event-time vs write-time separation -- is omitted. Four
behaviours named for five cells: one absent, one present but unnamed.

Mitigating, and why this is a fixable gap rather than an invalidated result:
- live_check C8.7 already states the blind spot correctly ("A guard expressed
  some other way (a silent `return` with no failure code, a validation inside a
  helper called for effect) is outside it").
- The top-level licence statements are correct and bounded ("a matrix licenses
  exactly 'these 14 mutations were killed'"; "NOT a claim that the guard set
  itself is complete").
- Every executed result reproduces; no product defect was found.

## G. Criterion 8 mapping

- "control observed GREEN first"  -- MET. Printed before any cell; the matrix
  returns early on a red control. Independently reproduced.
- "byte-identical restore"        -- MET by a stronger, disclosed construction:
  no repo write ever occurs, sha256 identical before/after in both scripts, and
  md5 of all three scripts unchanged after every run including my six mutations.
- "mutation-test every new guard" -- MET for the current code (14/14 killed,
  15/15 enumerated guards covered, no genuine survivor among my 6 independent
  cells), but CAPPED: the artifacts' account of why five cells are
  coverage-neutral is refuted by measurement, and the derived-completeness
  mechanism has 1-of-4 recall on the author's own named member set.

## H. Notes (non-blocking)

- live_check heading "## 6. MUTATION MATRIX -- 12/12 killed" is now stale in the
  document's top-level structure; cycle 4 appended rather than annotating it.
- The writer's `_self_test` docstring says "Touches no real file", but the new
  cycle-4 check `cli(["--emit-sequence"])` omits `--ledger` and so READS the real
  handoff/verdict_ledger.jsonl via the module default. Read-only and harmless
  today, but it makes that check's exit code depend on the real ledger's state.
- `mutation_matrix_86_85.py` imports `verify_matrix_coverage_86_85` by bare
  module name; works only because sys.path[0] is the script dir on direct run.
- WORKING TREE: backend/api/sovereign_api.py + 5 frontend/src files are modified
  and uncommitted (mtimes 2026-08-14 13:24-13:29, a day before this step's work)
  -- an unrelated `1y` sovereign red-line window change. NOT produced by 86.85,
  but auto-commit-and-push.sh does `git add -A`, so they would ship under an
  86.85 commit subject.
- C1-C7 were declared "unchanged from cycle 3, not re-litigated" and I did not
  re-derive them. live_check sections 1-9 carry their evidence; section 3's
  driven 3rd-CONDITIONAL is execution-based with anti-vacuity controls.

VERDICT RETURNED: CONDITIONAL (the structured return is the deliverable; this
file is a crash-survival record and is NOT a verdict).

COMPLETED: 2026-08-15T19:58:41Z
