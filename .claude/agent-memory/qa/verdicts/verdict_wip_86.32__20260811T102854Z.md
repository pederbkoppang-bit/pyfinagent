STATUS: INCOMPLETE -- not a verdict
STEP: 86.32
WRITTEN: 2026-08-11T10:28:54Z

# Q/A cycle 2 for step 86.32 (attempt-budget / escalation)

Prior artifacts on disk at start:
- .claude/agent-memory/qa/verdicts/verdict_wip_86.32__20260811T101142Z.md (11660 bytes,
  mtime 2026-08-11 12:21 local = ~10:21Z) -- an EARLIER spawn of this same cycle that
  appears to have dropped. Treated as EVIDENCE ONLY, never as a verdict; I re-derive
  everything myself.

## Plan
A. harness-compliance audit (5 items)
B. deterministic: immutable command `grep -c "^## Cycle" handoff/harness_log.md`
C. targeted re-verification of the cycle-1 remediation:
   (1) fixture == true 86.28 series?
   (2) guard test_fixture_matches_the_recorded_ledger -- try to defeat it
   (3) M7/M8 kills genuine, not mis-attributed
   (4) did corrections reach EVERY propagation site
   (5) contract original text byte-unchanged under the appended annotation
   (6) anything newly broken

(findings appended below as established)

## B. DETERMINISTIC -- established

- Immutable cmd `bash -c 'grep -c "^## Cycle" handoff/harness_log.md'` -> **1218, exit=0**.
- `git status --short`: only memory/audit/health churn + my own WIP + a researcher memory
  file. NO unintended production change.
- pytest backend/tests/test_phase_86_32_attempt_budget.py -> **15 passed** (file has 15
  `def test_`).
- Author's matrix `python scripts/qa/mutation_matrix_86_32.py` REPRODUCED BY ME:
  control green, **all 8 cells KILLED** (M1..M8), restore md5
  e4ffc1055f964257b237ca2aff6e0677 byte-identical (matches my own pre-run md5).
  M8 also reddens test_86_28_replay... (extra red, not mis-attribution; both named
  tests went red). M7 named-red = test_fixture_matches_the_recorded_ledger, and the
  mechanism IS the symmetric-difference assert (verified by hand: missing
  {(09d,NV),(cdc,F)} / spurious {(09d,F),(cdc,NV)}).

## C1. THE FIXTURE -- independently derived (question 1)

Ledger at `handoff/current/evaluator_critique_86.28.md:11-19` (7 rows):
  1 c1 wf_10c6cbd2-cad CONDITIONAL
  2 c2 wf_d0934c91-70b CONDITIONAL
  - c3 wf_01c83c86-09d **DROPPED**
  3 c3 wf_e262facc-cdc **FAIL**
  4 c4 wf_5a217e41-9b9 CONDITIONAL
  5 c5 wf_344395f1-4ac CONDITIONAL
  - c6 wf_9c55b720-ef3 **DROPPED**
CONFIRMED: the ledger table records only 7 and OMITS wf_e03ec2d0-c07.
`handoff/current/live_check_86.28.md:198-201` = "## 9. CYCLE 7 -- a real survivor,
recovered from a DROPPED Q/A run ... Q/A `wf_e03ec2d0-c07` dropped without a verdict
(174,664 tokens)". Cycle 7 > cycle 6, so position 8 IS correct.
Series = [C, C, NV, F, C, C, NV, NV]. MATCHES the fixture exactly.
Cross-check: masterplan 86.32 `audit_basis` independently states "CONDITIONAL,
CONDITIONAL, DROPPED, FAIL, CONDITIONAL, CONDITIONAL, DROPPED, DROPPED" -- identical.
Ledger file NOT touched by any 86.32 commit (not in 96870e44 stat), so the guard is
not circular-by-edit.

## C2. THE GUARD -- 9 independent defeat attempts, ALL KILLED

Probe run via test-module globals patching (no repo write; qa-write-guard blocked
even a scratchpad write, so heredoc-to-python was used). Control GREEN before and
after.
  Q1 swap rows 1<->2 (same outcome, order-only)      -> KILLED (order assert)
  Q2 pos8 := duplicate of pos7 (id IS in live_check) -> KILLED (distinctness assert)
  Q3 pos8 outcome NO_VERDICT -> CONDITIONAL          -> KILLED
  Q4 pos8 deleted                                     -> KILLED (cycle!=7)
  Q5 spurious row inserted at pos1 (9 rows)           -> KILLED (symmetric difference)
  Q6 parser returns 0 rows (silent empty)             -> KILLED by `len>=7` floor
  Q7 parser returns 6 of 7                            -> KILLED by the same floor
  Q8 parser AND fixture BOTH truncated to 6           -> KILLED (floor is absolute,
     so the circular-drift shape cannot pass)
  Q9 pos8 provenance string falsified                 -> KILLED
Parser verified to return exactly the 7 ledger rows and nothing else (no second
table in the file matches the row regex).
=> The guard is NOT vacuous. It discriminates on membership, order, cardinality,
   provenance and parser health.

## FINDING W1 (Contradiction) -- the wrong ordering SURVIVES in the test file

`backend/tests/test_phase_86_32_attempt_budget.py:275-276`:
    # The legacy counter ends at zero: the CONDITIONAL at attempt 7 wipes the FAIL
    # at attempt 6. This is the defect, asserted rather than described.
Against the fixture the SAME FILE imports and asserts on, attempt 6 = CONDITIONAL
and attempt 7 = NO_VERDICT; the FAIL is at attempt 4 and the CONDITIONAL that wipes
it is at attempt 5. BOTH halves of the sentence are false. Main's remediation claim
named CLAUDE.md F1b + experiment_results criterion 5 + live_check section 1 as the
propagation sites; this fourth site was missed, INSIDE the file the remediation
rewrote (88 lines changed there in 96870e44), 6 lines below newly-added code.
Comment-only: the assertions themselves are correct and my probes confirm the
behaviour. WARN-level, not blocking -- but it is the exact fact the cycle-1 FAIL was
issued for, surviving in the step's own deliverable.
CLAUDE.md:419 and experiment_results:140 are NOT instances: both are explicit
"the earlier revision said X" correction notes, correctly framed.

## FINDING W2 (Invalid_Precondition) -- "## 4. Verbatim command output" is PRE-REMEDIATION

experiment_results_86.32.md:171-194 is labelled "Verbatim command output" and reports
  "13 passed", "RESULT: all 6 cells KILLED", "[restore] md5 638fec28a2bd8c37fb187eb56f0fd3b3".
MEASURED NOW: 15 passed; all 8 cells KILLED; md5 e4ffc1055f964257b237ca2aff6e0677.
PROVENANCE PROVEN: `git show <c>:scripts/harness/attempt_budget.py | md5 -q` gives
  4358683c -> 638fec28...  069908c7 -> 638fec28...  96870e44 -> e4ffc105...  HEAD -> e4ffc105...
so that block is a capture of the PRE-remediation tree pasted under a
post-remediation document. `git diff 069908c7 96870e44 -- experiment_results_86.32.md`
touched exactly ONE line matching (passed|cells KILLED|md5|Verbatim) and it was a
prose line in criterion 5 -- the block was never regenerated.
CONSEQUENCE: the step's central remediation claim ("M7 and M8 pin both halves") has
NO execution evidence anywhere in experiment_results; §4 affirmatively shows a 6-cell
run that predates M7/M8. I reproduced the 8-cell run myself and it is TRUE, but the
artifact contradicts itself and an auditor reading §4 would conclude the remediation
was never executed. qa.md §4b: a "verbatim" capture must be REGENERATED.
live_check_86.32.md is NOT an instance -- I re-ran `python scripts/harness/attempt_budget.py`
and its §1 JSON + escalation block match my run exactly; §3's 1092 = sum(3^k, k=1..6) reproduces.

## Deterministic, continued
- ruff F821,F401,F811 over git-derived scope (3 .py files, non-empty asserted):
  "All checks passed!" exit=0.
- criterion 6: `git diff cf50bde2..HEAD -- .claude/agents/qa.md` EMPTY;
  sha256[:16] 06976b7d4a6072fd at cf50bde2 AND now. Artifact claim reproduces exactly.
- 1d runtime smoke: `import scripts.harness.attempt_budget` OK (5 / 1200000);
  test-module import OK. No consumer of the module outside the test + docs.
- scoped regression: `pytest backend/tests/ -k "harness or budget or phase_86"` ->
  410 passed, 1 skipped, 1 xfailed. Nothing broken.
- remediation STRICTLY STRENGTHENED the suite: 1 vacuous test removed, 3 real ones
  added; -5 asserts (all properties of the constant) / +10 asserts; +2 matrix cells,
  0 removed. The old `len(FIXTURE_86_28)==8` assert is gone but cardinality is now
  guarded via the ledger prefix + last-row checks (proven by my Q4/Q5 probes).

## FINDING W3 (Contradiction, PRE-EXISTING -- not caused by the remediation)

`scripts/harness/attempt_budget.py:50` and `CLAUDE.md:399-400` both state the budget
"leaves **90.9%** of historically-completed steps untouched (27+48+38+28+13 = **154
of 164**)". 154/164 = **93.9%**, not 90.9%. The number contradicts the derivation
printed beside it, under a comment that says "Rationale, MEASURED over 513 runs".
The researcher's own brief prints the correct figure for that exact ratio
(`research_brief_86.32.md:596`: "a cap of 4 covers 154/164 (93.9%)" -- whose *label*
is off by one, since brief:194 gives >4 = 23/164, i.e. <=4 = 141). The histogram
itself reproduces exactly (brief:180-182 == module:49-52; sum 164; <=3 = 113 = the
brief's 68.9%). ATTRIBUTION: present already at 4358683c (GENERATE), so pre-existing
and missed by cycle 1 -- NOT something the remediation broke. p50 419,739 / max
1,832,223 DO reproduce from the brief.

## Criteria (all independently executed, not read)

C1 MET  -- F1b at CLAUDE.md:378, inside "### Failure discipline" (:367), after F1 and
          before F2 (:425); opens "READ THIS TOGETHER WITH F1 ABOVE".
C2 MET  -- Outcome.NO_VERDICT first-class; attempts_used counts all; M2 KILLED; token
          ceiling counts a drop's tokens. Criterion's "~556K": 197,091+184,753+174,664
          = 556,508 -- REPRODUCES.
C3 MET  -- my own sweep, DEEPER than the author's 1,092: 9,840 non-PASS sequences
          (len 1..8) -> CLOSED_PASS = 0 and green closes = 0 across all four flag
          combos. Degenerate budgets (max_attempts 0/-1) fail SAFE to ESCALATE.
          M3 KILLED. Written summary leads with "THIS IS NOT A PASS AND NOT A FAIL".
C4 MET  -- close_kind on the 2026-08-10 fabricated-transcript FAIL returns CONTINUE
          for ALL FOUR (product, evidence) combos; M4 KILLED.
C5 MET  -- fixture == the true 86.28 series (3 independent sources agree: ledger,
          live_check §9, masterplan audit_basis); replay terminates at attempt 5,
          ESCALATE, legacy counter 0; guard survives 0 of my 9 defeat attempts.
C6 MET  -- empty qa.md diff + identical sha256 (see above).
Harness compliance 5/5 CLEAN: research gate (brief_status COMPLETE, gate_passed true,
8 sources >= 5, 17 URLs >= 10, recency section at :461) -> contract cf50bde2 12:01:22
-> code 4358683c 12:10:21; experiment_results present; log-last OK (0 harness_log
entries matching '^## Cycle.*phase=86\.32', masterplan status=pending, retry 0/3);
no verdict-shopping (evidence CHANGED: 96870e44, 618 insertions / 9 files).
3rd-CONDITIONAL: 0 prior CONDITIONALs logged for 86.32 (cycle 1 was FAIL) -> rule N/A.

## Lenses
correctness = PASS | does-it-reproduce = CONDITIONAL (W2) | scope-honesty = PASS-with-a-hole
min = CONDITIONAL.

## VERDICT (returned via StructuredOutput): CONDITIONAL
Not FAIL: every immutable criterion is MET on its literal terms and the cycle-1
blocker is fixed more thoroughly than asked. Not PASS: a block labelled "Verbatim
command output" documents a superseded tree (W2), and a false sentence about the
86.28 ordering survives in the step's own test file (W1) -- on a step whose thesis is
that self-reported evidence must be corroborated.

