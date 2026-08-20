STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.118
WRITTEN: 2026-08-18T09:34:46Z

# Q/A write-first record -- step 86.118 (test-suite triage)

## Attempt / sequence evidence
- `qa_wip.py 86.118 --spawned-at 2026-08-18T09:34:46Z`: source_present=true,
  attempt_number=1, attempt_number_status=ok, prior_attempts=0, prior_records=[].
- `verdict_history_86_21.py --step 86.118 --evidence-only`: status=`no_rows_for_step`,
  verdicts=(none). Cross-check prior_attempts(0) vs ledger rows(0): NOT stale.
- Sequence: first Q/A on this step. Main's disclosure (research gate wf_628cc28c-e10
  PASSED, no prior Q/A) is consistent with both sources.

## A. Harness compliance (5 items)
1. RESEARCH-BEFORE-CONTRACT: research_brief_86.118.md exists (42,052 bytes,
   mtime 08:49:27) < contract_86.118.md (08:53:31) < experiment_results (11:33:35).
   Envelope: brief_status COMPLETE, gate_passed true, external_sources_read_in_full=11
   (>=5), urls_collected=60 (>=10), recency_scan_performed=true (section at :150),
   internal_files_inspected=17. GATE OK.
2. CONTRACT-BEFORE-GENERATE: mtime chain above holds; step commit 1bf26bf8 at 11:33:53.
   OK.
3. EXPERIMENT_RESULTS present (8,149 bytes) + live_check_86.118.md (19,072 bytes). OK.
4. LOG-LAST: `grep -F 86.118 handoff/harness_log.md` returns only two FILING
   references from earlier steps -- no `phase=86.118 result=` row. masterplan
   86.118 status = `pending`. NOT yet logged/flipped. OK.
5. NO-VERDICT-SHOPPING: attempt 1, no prior verdict. N/A. OK.

## B. Deterministic
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python -c "import ast;
  ast.parse(open(\"backend/tests/conftest.py\").read()); print(\"parses\")"'`
  -> stdout `parses`, exit 0. REPRODUCED.
- SCOPE (derived, not typed): `git show --name-only --format="" 1bf26bf8` = 13 files:
  .claude/masterplan.json, 6 backend/tests/*.py, 4 scripts/qa/*.py,
  2 handoff/current/*86.118*. ZERO production backend modules. No unintended
  production change in the step commit.
  Working tree carries 3 uncommitted production files (backend/config/settings.py,
  backend/api/charts.py, backend/agents/claude_code_client.py) + 2 untracked test
  files -- these are the PEER session's 86.120 work, disclosed by Main and excluded
  from the step commit.
- masterplan diff in the commit is PURELY ADDITIVE (only removed line is `-}`);
  added ids 86.120..86.126. No criteria erosion. NOTE: 86.120/121/122 are the peer
  session's filings swept in by `git add -A` (the audit-the-commit-not-the-diff class);
  additive only, no content loss.
- LINT GATE (scope derived from the commit, 10 .py files, non-empty asserted):
  `uvx ruff check --select F821,F401,F811 <10 files>` -> `All checks passed!` exit=0.
- Frontend gate: N/A (no frontend/** in scope).
- Backend runtime smoke: no production backend module changed; test modules imported
  by the pytest runs below.

## C. Claim re-derivation (independent)

### C1. HEADLINE (env leakage) -- REPRODUCED INDEPENDENTLY
  Settings.model_config env_file = <repo>/backend/.env
  LIVE Settings():  paper_risk_judge_reject_binding=True, paper_data_integrity_enabled=True
  DECLARED default: paper_risk_judge_reject_binding=False, paper_data_integrity_enabled=False
  backend/.env:83 PAPER_DATA_INTEGRITY_ENABLED=true ; :84 PAPER_RISK_JUDGE_REJECT_BINDING=true
  -> the live_check's cited line numbers are exact.

### C2. INDEPENDENT MUTATION of the re-aimed assertions (the one Main declared
     "deliberately NOT automated") -- KILLED
  Method: in-process pytest plugin flipping
  `Settings.model_fields[<flag>].default = True` -- NO file on disk touched, so the
  peer session's uncommitted settings.py is untouched.
  MUTANT: 2 failed (`test_60_3_flag_defaults_off`,
          `test_reject_binding_main_path_off_emits_on_blocks`), rc=1, with the
          named assertion in the traceback: `assert True is False ... FieldInfo(...default=True)`
  CONTROL (same fixed selection, unmutated): 2 passed, exit 0.
  ORDERING DISCLOSED: I ran the mutant first, then the control. Both observed.
  CONCLUSION: the `Settings.model_fields[...].default` re-aim is NOT vacuous; it is
  killed by exactly the mutation that matters (a change to the shipped default).

### C3. `||` classifier discrimination -- Main's 3-row claim REPRODUCES, with a caveat
  With the token in the arm NOT followed by `||`:
    BOTH arms missing      -> None            (still GENUINE)  [matches claim]
    one arm EXISTS         -> 'alternative-arm-satisfied'       [matches claim]
    no || at all, missing  -> None            (still GENUINE)  [matches claim]
  CAVEAT (mine, not Main's): with the token in the FIRST arm the pre-existing
  `negation_patterns` rule returns 'absence-asserted' BEFORE the new block is
  reached, so the "BOTH arms missing -> GENUINE" row holds only for second-arm
  tokens. Does not change the verdict -- the pre-existing rule is not this step's.

### C4. BLAST RADIUS of the new rule -- MEASURED over the whole live masterplan
  1,153 verification commands walked; 14 contain `||`.
  Tokens newly classified 'alternative-arm-satisfied': **2**, both the SAME token
  in step 86.31 (`.claude/hooks/lib/qa_write_guard.py`, surfaced by two extractors).
  So the repair excuses exactly the one command it was written for.
  LATENT over-broadness (constructed by me, ZERO live instances):
    `grep -q x CLAUDE.md || test -f MISSING`        -> 'alternative-arm-satisfied'
    `test -f MISSING && echo ok || cat CLAUDE.md`   -> 'alternative-arm-satisfied'
  In both, the missing path IS required on some execution path. NOTE-level: the
  guard tightened correctly for the corpus that exists, and opens a false-negative
  window for command shapes not currently present.

### C5. Effort pin (row 2) -- TRACKING POLICY, not accommodating drift
  `.claude/settings.json:2` = `"max"`. CLAUDE.md:59 independently documents
  "raised xhigh -> max 2026-08-04 by direct operator instruction". The pin is
  anchored to a DOCUMENTED decision, not merely to the current file content, and
  the guard is still an equality assert (not relaxed to key-presence), so it still
  catches a clobber. Correct call.

### C6. Fixture pin did NOT silently disarm the flag-ON tests
  `_make_settings` pins the flag False in `base`, but `base.update(overrides)` and
  every ON test passes `paper_risk_judge_reject_binding=True` explicitly
  (57_1 lines 120, 173, 194, 214). No ON path was turned off by the pin.

### C7. Suite-count reconciliation (contract 3635 vs live_check 3672/3673)
  research_brief:315 and contract:92 cite `19 failed, 3635 passed`; live_check
  cites 3672 and 3673. The peer session's two untracked test files collect
  **38 tests** (measured), so 3635+37=3672 and 3635+38=3673. The delta is fully
  explained by the disclosed concurrent session. Contract number is the RESEARCH
  phase measurement; live_check carries this step's own two runs.

### C8. Git-pin census -- REPRODUCED EXACTLY
  census @ pinned (7739922d) : {'dict': 720, 'str': 126, 'list': 13, 'none': 24}
  -> byte-identical to the asserted tuple. census @ live: dict=1136 (live_check
  said 1132; the masterplan moved by 4 during/after the step, which CONFIRMS the
  pin rationale rather than contradicting it).

### C9. Consumed-evidence rows -- classification VERIFIED, resolver NOT over-broad
  Both artifacts absent from handoff/current/, present at handoff/archive/misc/;
  `git log --diff-filter=D` shows commit fa9aaf8e ("handoff layout backfill --
  archive 315 misc ... out of handoff/current") MOVED them. rglob over the whole
  archive returns EXACTLY 1 match for each name -> unambiguous.

### C10. Deselect (Main's item #3) -- LEGITIMATE, VERIFIED not merely disclosed
  75_17 WITHOUT the deselect: `1 failed, 44 passed` and the single FAILED is
  `test_masterplan_diff_touches_only_the_ten_sibling_insertions` -- the named one.
  WITH it: `44 passed, 1 deselected`. Applied to control and mutant alike, named
  explicitly (not a `-k`), and no cell's NAMED test is the deselected one.

### C11. FULL SUITE -- REPRODUCED BYTE-FOR-BYTE (criterion 6)
  MY run: `8 failed, 3684 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings
           in 518.53s (0:08:38)`
  AUTHOR : `8 failed, 3684 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings
           in 514.34s (0:08:34)`
  All 8 FAILED names match the live_check §5 table 8/8. `1 xpassed` reproduces.

### C12. MUTATION MATRIX -- REPLICATED INDEPENDENTLY
  `python scripts/qa/mutation_86_118.py` (run by me):
  7/7 controls GREEN with the exact collected counts claimed
  (44/18/25/7/13/8/77); `KILLED 13 / 13   SURVIVED 0   UNSCORABLE 0
  EQUIVALENT-BY-DESIGN 1`; 7/7 SHA-256 restores verified; tree clean afterwards.
  guardlib_selftest.py: `cases passed: 45   FAILED: 0`.

### C13. I EXTENDED the matrix on its one gap -- guard HOLDS
  No cell NAMES the 75_19 sibling. I neutered only the
  `alternative-arm-satisfied` return in preflight's BOUND `fp_reason` reference
  (in-memory) ->
  `test_phase_75_19_preflight_calibration::test_live_masterplan_is_currently_clean`
  FAILED (rc=1). So "one line held two tests red" holds on BOTH consumers.
  (My first probe patched the wrong module identity and came back clean -- I
  chased it rather than banking the clean answer.)

### C14. FINDING (mine): M5's kill is ENVIRONMENT-CONTINGENT
  Simulating M5 exactly (base pin removed; an explicit override still wins):
    env AS DEPLOYED (.env=true)  -> unpinned flag reads True  -> KILLED
    env FLAG UN-PROMOTED (false) -> unpinned flag reads False -> SURVIVES
  The matrix reports "13/13 KILLED" unqualified. In a step whose HEADLINE is
  that the suite is not hermetic, its own matrix inheriting that property is a
  scope-honesty gap. NOT a criterion-7 failure (every mechanic of criterion 7 is
  satisfied as run), and the sibling declared-default assertion is killed
  environment-INDEPENDENTLY (C2).

### C15. Bootstrap oracle -- real rejection power, three constructed evasions
  CONTROLS rejected: bare `launchctl bootstrap ...`; `HINT='...'` + `eval "$HINT"`.
  EVADE (offenders=[]):
    `launchctl  bootstrap ...`                    (double space)
    `export HINT='launchctl bootstrap ...'` + `eval "$HINT"`  (the `^VAR=` assign
       regex does not match `export VAR=`, so the var is never recorded)
    `LC=launchctl` + `$LC bootstrap ...`
  NOTE-level: incompleteness on a hygiene guard with a genuine behavioural core
  (M7/M7b/M7c/M7d all kill), strictly better than the oracle it replaced.

### C16. Weakening audit -- DERIVED from the commit, not accepted
  Assert-line accounting across the 6 test files: 4 removed / 5 added.
  Every removal has a replacement:
    `Settings().paper_data_integrity_enabled is False`   -> declared default
    `s_off.paper_risk_judge_reject_binding is False`     -> declared default
    `effortLevel == "xhigh"`                             -> `== "max"`
    `"launchctl bootstrap" not in stripped`  -> `not hits` + 2 oracle fixtures
  ZERO xfail, skip, deleted assertion, widened tolerance, pinned seed.

## E. CRITERION VERDICTS
1 MET       -- exact cmd + 2 runs + counts; honest that a fixed order proves
               nothing about ordering, answered by isolation instead.
2 PARTIAL   -- 19/19 rows carry evidence, BUT row 7
               (`test_phase_62_4_sentinel::infra_path_distinct_exit`) is
               EXPLICITLY left unclassified ("Not yet classified STALE vs
               PRODUCT", live_check §5), and 6 finer labels (ENV LEAKAGE,
               CLASSIFIER FALSE POSITIVE, PROXY ASSERTION, LIFECYCLE META-TEST,
               SUPERSEDED POLICY, census-vs-live-artifact) are never mapped to
               the three buckets the criterion names.
3 MET       -- 86.123/86.124/86.125/86.126 verified present with substantive
               audit_basis + criteria; no PRODUCT-DEFECT test was edited.
4 MET       -- see C16; both re-aims argued AND proven non-vacuous (C2).
5 PARTIAL   -- isolation half PROVEN and reproduced by me (passes alone: 18
               passed; fails in my own full run). MISSING: "identify the shared
               state responsible" for the one ORDERING-ARTIFACT found. The
               contract's own P2 planned it. Also the contract's
               opposite-direction masking anomaly (`test_portfolio_swap` passes
               when the 19 run together, fails alone and in full) is dropped
               from the evidence artifact.
6 MET       -- reproduced byte-for-byte (C11); all 8 named with dispositions,
               which criterion 6 explicitly contemplates.
7 MET       -- replicated (C12), extended (C13), deselect verified (C10),
               EQUIVALENT-BY-DESIGN's declared hand-mutation run by me (C2).

## F. LENSES (worst-of-N)
  correctness      : PASS   (nothing I re-derived was wrong)
  does-it-reproduce: PASS   (suite, matrix, census, headline, selftest all exact)
  scope-honesty    : CONDITIONAL (C14 unqualified 13/13; C2 row-7 and C5
                     shared-state undelivered without being named as gaps)
  min -> CONDITIONAL

COMPLETED: 2026-08-18T09:50:35Z
