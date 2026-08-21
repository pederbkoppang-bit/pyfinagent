STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.9
WRITTEN: 2026-08-21T09:59:22Z

# Q/A write-first record -- step 90.9, cycle 1 (claimed)

Spawn: Workflow rail, agentType qa. Read qa.md in full at runtime (898 lines, includes
the phase-86.78 consequence-withholding section and the phase-86.79 qa_wip guidance).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command exit code; git status/diff scope; lint; syntax
C. LLM judgment vs 7 immutable criteria; mutation/vacuity attack on the classifier

## Findings (appended as established)

## PRIOR-ATTEMPT EVIDENCE (gathered, not a trigger)
- qa_wip.py 90.9 --spawned-at 2026-08-21T09:59:22Z: source_present=true,
  attempt_number=1, attempt_number_status=ok, prior_attempts=0, prior_records=[],
  records_retained=1 (gauge, my own file).
- verdict_history_86_21.py --step 90.9 --evidence-only: status=no_rows_for_step,
  verdicts=(none). Cross-check prior_attempts(0) vs ledger rows(0): equal -> no
  staleness signal. sequence: no prior verdicts recorded for 90.9.

## HARNESS COMPLIANCE (UTC mtimes)
- research_brief_90.9.md  2026-08-20T20:01:19Z (37862 b)
- contract_90.9.md        2026-08-20T20:21:18Z (10924 b)   [research < contract OK]
- criteria_shape_90_9.py  2026-08-21T08:24:05Z (40702 b)   [contract < code OK]
- mutation_matrix_90_9.py 2026-08-21T08:24:05Z (12775 b)
- live_check_90.9.md      2026-08-21T08:24:05Z (19816 b)
- experiment_results_90.9 2026-08-21T08:25:35Z (16240 b)   [code < results OK]
- harness_log: grep -F "phase=90.9" -> ZERO rows. LOG-LAST satisfied.
- masterplan 90.9 status = "pending". Not flipped. OK.
- HEAD 632a8fc3554b5c73587142dc7d5bf70f14b880a5

## *** BLOCKING FINDING 1: THE IMMUTABLE COMMAND EXITS 1 ***
Ran verbatim:
  bash -c 'python3 scripts/qa/criteria_shape_90_9.py --verify && python3 scripts/qa/mutation_matrix_90_9.py --verify'
EXIT=1  (NON-ZERO)

criteria_shape_90_9.py --verify: "failed: 0" -> exit 0.
mutation_matrix_90_9.py --verify: reports
  BAD  QX   KILLED    expected ERROR
  BAD  QXI  KILLED    expected ERROR
  "2 problem(s): 2 unexpected cell score(s)"  -> exit 1

This is a DIRECT miss of immutable criterion 2's final clause:
  "...and a mutant that fails to run scores ERROR, never a kill"
The shipped harness scores BOTH fails-to-run error controls as KILLED.
The step's own tool says so. This is not my inference; it is the tool's verdict.

md5 at time of run: criteria_shape c2099832a178b7952b21c3b85eb59d30,
mutation_matrix a7a501978673ec1f17c4684872b21b01, masterplan e50b33b994afe3dd1a32a5df2a833cb0

## BLOCKING FINDING 1 -- ROOT CAUSE ESTABLISHED (criterion 2)
The failure is DETERMINISTIC and bifurcates on the INTERPRETER:

RUN A  no venv  -> /usr/bin/python3 = Python 3.9.6   EXIT 0  ERROR 2  (reproduces Main's claim)
RUN B  source .venv/bin/activate -> Python 3.14.4    EXIT 1  ERROR 0  QX/QXI both KILLED

MECHANISM: Python 3.13+ colorizes tracebacks. FORCE_COLOR=3 is injected by the Claude
Code agent runtime (NOT in ~/.zshrc/.zprofile/.zshenv/.bash_profile -- grepped, absent),
and FORCE_COLOR is honoured even when stderr is a PIPE. Measured stderr bytes:
  '\x1b[1;35mNameError\x1b[0m: \x1b[35mname ... is not defined...'
score_error() at mutation_matrix_90_9.py:71-78 tests `if f"{t}:" in observed` -- the
LITERAL "NameError:". With colour the type name and the colon are separated by \x1b[0m,
so the test never matches, err is None, and line 229
  got = "ERROR" if err else ("KILLED" if code != 0 else "SURVIVED")
falls through to KILLED. That is exactly what criterion 2 forbids:
  "a mutant that fails to run scores ERROR, never a kill"
Failure direction is FAIL-DANGEROUS: it INFLATES the kill count (10 -> 12).

SECOND, INDEPENDENT DEFECT IN THE SAME PLACE: the in-run probe that certifies the
discriminator (lines 212-218) is built from HAND-TYPED, UNCOLOURED literals, so it
CANNOT represent the failing case. In RUN B it printed
  "ok: the discriminator reads the TYPE, not the shape"
in the very same run where both cells it certifies misscored. Vacuity shape 5 (fixture
that cannot represent the failure) + shape 11 (mis-attributed kill mechanism).
The module docstring lines 38-44 records the 90.1 cycle-5 lesson "read the type from the
MESSAGE, not the shape of a traceback" -- shape-dependence was replaced by
FORMAT-dependence. The defect relocated one seam, it was not removed.

DISCLOSURE GAP: grep of live_check_90.9.md + experiment_results_90.9.md for
FORCE_COLOR|NO_COLOR|Python 3|interpreter|venv|3\.9|3\.14|environment returns ZERO hits.
Neither artifact states the interpreter its "verbatim" exit-0 was captured under.
CLAUDE.md Critical Rule 1 and qa.md 1 both prescribe `source .venv/bin/activate` first;
project stack is Python 3.14. The green depends on the NON-production 3.9.6 interpreter.

## NOTE: the tree moved DURING evaluate
.claude/masterplan.json md5 e50b33b994afe3dd1a32a5df2a833cb0 (09:59Z) ->
4bc0aad73ca8d8f3eb73a1546bf8ebc9 (later). Current content == HEAD (git diff empty).
Checked: 90.9 node is BYTE-IDENTICAL to HEAD -- criteria NOT amended, status still
"pending". So no criteria-erosion and no premature flip; recorded because it transiently
made the matrix report CONTAINMENT BREACHED on an unrelated run.

## *** BLOCKING FINDING 2: SURVIVING MUTANT ON CRITERION 7 (my own cell, not the author's) ***
QA-MUT-1 / QA-MUT-1c -- run in the author's own sandbox (build_sandbox/drive), NO_COLOR env
so the result is not confounded by finding 1.

MUTATION: move a live read of handoff/verdict_ledger.jsonl into a NEW helper
`_shape_hint()` that `classify()` calls unconditionally on every classification.

  def _shape_hint(criterion) -> int:
      t = Path("handoff/verdict_ledger.jsonl").read_text(encoding="utf-8")
      if not t.strip(): raise SystemExit("MUTANT PROOF: the ledger read did NOT land")
      return len(t)
  def classify(...):
      _shape_hint(criterion)          # <-- live verdict-history read on the label path
      ...

RESULT: exit 0, 0 FAIL lines. *** SURVIVED ***
  [PASS] no classification function references a verdict history, a WIP record, a round
         index or a remaining attempt budget -- neither handed in nor SELF-read
  classification-path I/O calls, in full: load_plan: run(); load_plan: read_text()

DIFFERENTIAL (1c): the helper ABORTS if the read returns empty. "MUTANT PROOF" never
appears in the output, so the read LANDED on every call -- the mutant is live, not inert.
This separates "the scan is blind" from "my mutant did nothing".

ROOT CAUSE: classifier_consequence_refs()/classifier_io_calls() at
criteria_shape_90_9.py:501/533 walk ONLY the bodies of the 8 names in CLASSIFIER_FNS
(:493). The scope is NON-TRANSITIVE -- it does not follow calls. So the guard is defeated
by MOVING the read one function over. Vacuity shape 2 (source-scan defeated by moving the
scanned text). M9 proves the scan RUNS; nothing proves its SCOPE covers the path.

SOLE COVERAGE: criterion 7's other two checks cannot see this. The purity check
(classify(x)==classify(x)) passes -- the helper is deterministic. The signature check
(params == {criterion, variant}) passes -- the read is a SELF-read, and criterion 7 names
"never READS" as the binding verb precisely to cover that half. So the AST scan is the
ONLY coverage for the binding half, and it is defeated. qa.md 4c verdict wiring:
sole-coverage vacuity => BLOCKING.

NOT a defect: criterion 4's write scan is whole-file (write_capable_calls(src) over the
entire source), so it is NOT defeated by the same move. The hole is specific to
criterion 7's scoped scan.

## CRITERION-BY-CRITERION (evidence)
C1 MET. Independent re-derivation at 252090a3 with MY OWN walk (not the script's):
   nodes with id+dict verification+non-empty success_criteria in 86..90 = 156 / 987;
   EXCLUDING 90.9 = 155 / 980 -- exactly the filed pair. Rule is PRINTED (--census
   header, MID variant, inclusion rule + 27 apparatus terms). Filed 403/41.1% does NOT
   reproduce; 438/44.7% printed BESIDE it, filed number NOT edited. Ratio collapses to a
   single 1.65x, inside the filed 1.6-1.9x. Sensitivity table shows the LEVEL is
   rule-dependent (16.6-62.4%) and discloses NARROW=2.21x falls OUTSIDE the filed range.
C3 MET. Sweep 155/155 executed, exit 0 on 99 / exit 2 on 56. Independently drove the REAL
   CLI (--step-file) as a separate PROCESS: 86.47 -> returncode 2, 87.1 -> returncode 0,
   matching the property detector. Sweep uses in-process gate_step (the production
   callee), killed by M3/M4.
C4 MET. masterplan md5 stable across census runs; write_capable_calls() is AST-level and
   whole-file, has a planted-positive control, and catches Path.write_text (the house
   idiom the criterion's two literal patterns miss) -- stricter than the wording. M8 kills
   the vacuous form. Independently confirmed scan(real)=[] .
C5 MET. verdict_ledger.jsonl md5 6e9dc4312b8344968ae8a345c51471e5 unchanged across every
   run I made. 90.9's own node byte-identical to HEAD -- no criterion edited/weakened.
C6 MET. --census prints the proposed bound, the count it would have flagged (56 property /
   44 proxy = the filed 44), NAMES the deferred finding (90.1 cycle-5 QA1/QA1b/QA1c), and
   the disposition is FILED: masterplan 90.12 exists, status "pending", P1, with its own
   verification command. Verified by walking the plan, not grepping. The analysis
   recommends AGAINST the bound as written -- honest negative result.
C2 NOT MET -- finding 1.
C7 NOT MET -- finding 2.

## HARNESS COMPLIANCE -- CLEAN (5/5)
1. research-gate-before-contract: research_brief_90.9.md envelope brief_status COMPLETE,
   external_sources_read_in_full=8 (floor 5), urls_collected=53 (floor 10, and 63 distinct
   http(s) URLs actually present in the file), recency_scan_performed=true with a
   "Recency scan (2024-2026) -- MANDATORY SECTION" at line 273, gate_passed=true (ENFORCED
   re-run of wf_722b01b9-67d, which had been enforced false for double-counting URLs).
2. contract-before-generate: research 20:01:19Z < contract 20:21:18Z < code/live_check
   08:24:05Z < experiment_results 08:25:35Z. All 7 immutable criteria are VERBATIM in
   contract_90.9.md (checked programmatically against masterplan, 7/7).
3. experiment_results_90.9.md present (16240 b).
4. log-last: grep -F "phase=90.9" handoff/harness_log.md -> 0 rows; masterplan status
   "pending". Correct.
5. no-verdict-shopping: cycle 1. qa_wip prior_attempts=0; ledger no_rows_for_step. N/A.

## DETERMINISTIC GATES
- AST parse both subjects: OK.
- ruff --select F821,F401,F811 over the DERIVED commit scope (git show --name-only
  4165e67d -- '*.py', xargs -0, non-empty asserted: 2 files): "All checks passed!" exit 0.
- ruff default (not the gate): 27 style findings (FURB167/PLW1510/F541/ISC004/I001).
  NOTE only -- outside the qa.md gate selection.
- No frontend/** and no backend/** in the commit -> 1b and 1d N/A. No UI claims -> 1c N/A.
- Runtime smoke: both scripts EXECUTED live (--verify, --census, --step-file), not merely
  parsed.
- Unintended production change: NONE. Working tree clean except audit jsonl + my WIP.
  Subject md5s IDENTICAL at start and end of my evaluation (c2099832..., a7a501978...).
- HEAD moved 632a8fc3 -> a3a2d368 mid-evaluation (two masterplan-queue/changelog commits,
  0d59de57 + a3a2d368). Neither touches the 90.9 subject. Findings stand on measured bytes.
- The subjects were committed in 4165e67d "phase-90.9: GENERATE only -- built and
  self-verified, deliberately NOT evaluated" -- honest labelling.

## WHAT IS GENUINELY STRONG (stated so the FAIL is precise, not dismissive)
- 5 of 7 criteria MET with evidence I re-derived independently rather than read.
- The author found and DISCLOSED two of their own surviving mutants (M5/M6) and a real
  regex bug -- `cm.start() < m.start()` compared two matches anchored on the SAME
  quantifier, so `0 < 0` was never true and the corpus-precedence branch had never
  discriminated anything. Fixed to compare NOUN positions; pinned count moved 62 -> 56.
  That is exactly the behaviour the harness wants.
- Criterion 6 returns an HONEST NEGATIVE: the proposed bound is recommended AGAINST,
  because it would have deferred the 90.1 cycle-5 evaluator's own cells.
- Criterion 1 prints 438/44.7% BESIDE the filed 403/41.1% and does not edit the filed
  number, with a 4-variant sensitivity table disclosing that NARROW (2.21x) falls OUTSIDE
  the filed range.

## REMEDIATION (small, mechanical, both)
F1: in mutation_matrix_90_9.py, decolorize before typing the failure --
    obs = re.sub(r"\x1b\[[0-9;]*m", "", out.stdout + "\n" + out.stderr)
    and/or drive() with env={**os.environ, "NO_COLOR":"1", "PYTHON_COLORS":"0"}.
    Then ADD A CELL that feeds score_error a COLORIZED NameError stream, so the probe at
    lines 212-218 can represent the failure it currently cannot. Disclose the interpreter
    in live_check.
F2: make criterion 7's scan reach the whole classification path -- walk the call graph
    from the CLASSIFIER_FNS roots instead of only their own bodies, or assert at RUNTIME
    (an open() shim / audit hook around a classification run) that no path other than the
    plan of record is opened. Then add a cell that moves a consequence read into an
    unlisted helper -- i.e. QA-MUT-1 as a permanent cell.

VERDICT RETURNED: FAIL (criteria 2 and 7 NOT MET).
COMPLETED: 2026-08-21T10:13:35Z
