# Evaluator critique -- step 86.29

# CYCLE 1 -- RAIL DROP. NO VERDICT.

**Run `wf_d4e2e794-567` (task `wctzb6dqu`), 2026-08-11 06:38-06:49Z. Terminated
with `agent({schema}): subagent completed without calling StructuredOutput
(after in-conversation nudge)` after 197,098 subagent tokens and 42 tool uses.**

**THIS IS NOT A VERDICT.** Per `.claude/rules/research-gate.md` and the CLAUDE.md
harness protocol, an errored/empty return is NO VERDICT, never PASS -- and
equally, never CONDITIONAL. The record below reached an internal assessment of
CONDITIONAL. **That assessment is not adopted, not recorded as this step's
verdict, and does not advance any counter.** Step 86.29 has had ZERO completed
Q/A cycles.

This is the SECOND drop of the day (the first was 86.34's cycle 3 at 185,745
tokens). Both were rescued before the next spawn overwrote the fixed per-step
path -- the durability defect queued as 86.36.

The record is rescued verbatim below because it is **evidence for the re-run**:
it names five reproducible findings, and every one of them was re-measured by
Main and CONFIRMED before being acted on.

```
STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.29
WRITTEN: 2026-08-11T06:38:03Z
COMPLETED: 2026-08-11T06:48:17Z

# Q/A write-first record -- step 86.29, CYCLE 1

Launch: Workflow structured-output rail. Author = pyfinagent-51 (Main).
Contract authored by peer session pyfinagent-06 (c806cad6) BEFORE GENERATE (974297ce).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable cmd `bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n .claude/hooks/archive-handoff.sh'`
C. Re-run the two re-runnable scripts myself
D. Attack the 4 self-nominated weak points + criterion-by-criterion judgment

## Findings log (appended as established)

### A. Harness compliance (5 items) -- ALL CLEAN
- research-gate-before-contract: `handoff/current/research_brief_86.29_rerun.md`
  envelope `brief_status COMPLETE`, `gate_passed: true`, `external_sources_read_in_full: 7`,
  `recency_scan_performed: true`, internal_files 7. The FIRST brief
  (`research_brief_86.29.md`) honestly records gate FAILED (no envelope, stopped
  mid-loop) and was re-run -- that is the rail working, not a skip.
- contract-before-generate: PROVABLE FROM GIT (split authorship).
  c806cad6 (2026-08-10 22:53) touches ONLY `handoff/current/contract_86.29.md`.
  974297ce (2026-08-11 08:37) touches the hook + 2 scripts + results + live_check.
  Contract commit strictly precedes and contains NO code.
- experiment_results present: `handoff/current/experiment_results_86.29.md` (13151 B).
- log-last: `handoff/harness_log.md` -- to be checked. masterplan 86.29 status=pending,
  retry_count=0. NOT yet flipped.
- no-verdict-shopping: CYCLE 1, no prior verdict for 86.29 exists. Nothing to shop.

### B. Deterministic
- IMMUTABLE CMD `bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n ...'` -> **EXIT=0**
- hook sha256[:16] = `6dc68f781edb4fd0` -- MATCHES the digest stated in live_check line 5.
- `git diff --name-only HEAD` = only agent-memory/audit-jsonl/heartbeat churn.
  NO unintended production change. GENERATE commit `974297ce` touched exactly the
  5 files experiment_results section 1 claims. Verified with
  `git diff --name-only 974297ce^ 974297ce`.
- `python scripts/qa/prove_archive_provenance_86_29.py` -> RESULT: PASS (0 problems),
  EXIT=0. Reproduced BY ME. 4/4 mutants KILLED, 3/3 control checks GREEN,
  BEFORE half declares '82.54' from the git-recovered pre-fix hook.
  Isolation: real hook digest unchanged, archive dir list unchanged (819).
- `python scripts/qa/derive_archive_misattribution_86_29.py` -> EXIT=0.
  Reproduced BY ME at tree 33255004: **153 mismatch / 387 agree / 255 unclassified
  / 24 no_contract over 819 dirs**. Recall 2/2, controls 4/4, precision 1.0000,
  0 suspects. EXACTLY the "after" row of experiment_results section 6.

### B2. Live-witness claim (attack point 2) -- MAIN'S READING IS CORRECT
Read `.claude/hooks/archive-handoff.sh:241-244`:
```
for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
    [ -f "$CURRENT_DIR/$f" ] || continue
    [ -f "$target/$f" ] && continue          # derived branch already won
    if rolling_declares_step "$CURRENT_DIR/$f" "$short_sid"; then
```
For 86.31 the derived branch supplied contract/experiment_results/evaluator_critique/
research_brief, so those four hit the `[ -f "$target/$f" ] && continue` early-out;
`handoff/current/research.md` is ABSENT (verified) so it hits the first `continue`.
=> `rolling_declares_step` was NEVER CALLED. `rolling_skipped=0` in
`handoff/archive/phase-86.31/PROVENANCE.md` is consistent with that and with nothing
else. Main's statement "the live run proves the DERIVATION branch ONLY, not the
guard" is EXACT -- neither over- nor under-claimed.
Independently verified the live dir: 5 artifacts + PROVENANCE.md, every head -1
declares 86.31 (`# Contract -- step 86.31` etc.). First archive dir since
2026-08-06 holding its own step's files -- confirmed by me, not taken from the report.

### B3. SECOND live witness appeared DURING this evaluation
`handoff/archive/phase-86.25/` minted 2026-08-11T06:42:07Z by a peer step closure
(archive grew 819 -> 820 between two of my own runs). PROVENANCE: `derived=5
rolling_copied=0 legacy_moved=0 rolling_skipped=0`; every head -1 declares 86.25.
Same shape as 86.31: derivation branch only, guard still never reached.
New evidence Main could not have disclosed. It STRENGTHENS the mechanism claim and
shows the ungraded-infrastructure exposure of section 4b is ONGOING until this closes.

### C. Criterion 2 -- RE-DERIVED BY ME under bash (not read from the artifact)
```
sid=86.29 0/0   sid=86.6 0/0   sid=82.54 0/0   sid=86.31 0/0
sid=86.26 0/0   sid=4.5.9 0/0  sid=25.A 0/0    sid=86.25 0/0
POSITIVE CONTROL (temp dir holding 86.29-contract.md) -> 1
```
NOTE zsh aborts this loop with `no matches found` (nomatch); the hook runs under
bash where an unmatched glob stays literal and `[ -f ]` fails -> 0. Re-ran under
bash, the hook's own shell. Criterion 2 MET, independently.

### D. Lint + scope
- scope DERIVED from the GENERATE commit (working-tree diff is empty; the fix is
  committed at 974297ce): `git diff --name-only 974297ce^ 974297ce -- '*.py'` ->
  2 files, NON-EMPTY asserted. `uvx ruff check --select F821,F401,F811` ->
  "All checks passed!", exit=0.
- qa.md 1b/1c/1d do NOT bind: no `frontend/**`, no `backend/**`, no UI claim.
- `scripts/housekeeping/verify_handoff_layout.py` reports 455 violations -- ALL
  pre-existing "no step-id prefix" complaints about the suffix convention itself
  (it expects a PREFIX). Zero mentions of PROVENANCE.md. Not a regression here.
- No consumer of `handoff/archive/*` enumerates dir contents in a way PROVENANCE.md
  breaks (checked verify_phase_4000_*.sh, quarantine_phantom_archives.py, layout).

### E. MY OWN MUTATION MATRIX (7 cells) -- beyond the author's 4
Control on the shipped hook first, then mutate. Nothing written to the repo:
the hook text is mutated IN MEMORY and written to a tempfile.
```
Y1 KILLED   fall-through `sys.exit(1)` -> `sys.exit(0)`  [MY check only]
Y2 SURVIVED remove `[ -f "$target/$f" ] && continue`     [near-equivalent]
Y3 KILLED   total never zero -> loud branch unreachable
Y4 KILLED   derived branch writes a wrong archive name
Y5 KILLED   invert the declaration comparison
Y6 KILLED   suppress the systemMessage emitter
Y7 SURVIVED variant glob widened to `${base}_*.md`       [author's fixture]
Y7 KILLED   same mutant under a REALISTIC fixture        [MY fixture]
```

### F. FINDINGS (all reproducible)

**F1 [WARN] Fixture cannot represent the criterion-4 failure class.**
`prove.make_scratch` puts ONLY the step-under-test's files in `handoff/current/`;
the real dir holds 410-521 files from ~200 steps. I added three other steps'
artifacts and widened the variant glob to `${base}_*.md` (Y7). Shipped hook:
GREEN, no alien files. Author's `check_right_step`: **SURVIVED**. My realistic
fixture: **KILLED**, 18 alien files copied into `phase-99.1/` incl.
`contract_82.54.md`. That is verbatim the defect criterion 4 names -- "copies
another step's files ... must be a visible failure". qa.md 4c shape #5.

**F2 [WARN] The "unsure -> do not copy" fall-through is covered by NO cell.**
The hook's own comment calls that asymmetry "the whole fix", but every fixture
rolling file DECLARES something, so `sys.exit(1)` (no pattern matched) is never
exercised. My check with non-declaring rolling files: shipped hook GREEN;
mutant Y1 copies all 4 undeclared rolling files. Killed only by my check.

**F3 [WARN] The precision oracle is NOT independent of the classifier.**
`confirm_mismatch` reuses the SAME `_DECLARE` list, differing only in aggregation
(union-of-all vs first-hit). It detects "right pattern, wrong order" and is BLIND
to "grammar does not recognise this header". Concrete class it misses: the
grammar accepts only ASCII `--`. 33 of the 255 "unclassified" dirs DO declare a
step with an EN/EM-DASH (`# Contract — Step 76.9.2`); **7 are genuine mismatches
the census does not count** -- phase-75.5.12 / 76.9.3 / 78.0 / 78.16 / 78.2 /
79.2 all hold 76.9.2's contract, phase-75.1 holds 75.2's. So 153 is a FLOOR
(>=160). C1's letter still holds: those 7 sit in the 49 "genuinely opaque" bucket
that IS explicitly reported as not-clean.

**F4 [WARN] A printed claim does not reproduce.** The census prints, and
live_check D reproduces verbatim: *"no mismatched dir mentions its own step id
anywhere in its contract head."* FALSE -- **47 of 153 do**, e.g.
`handoff/archive/phase-10.5.0/contract.md` head reads
`step: phase-10.5-batch (covers 10.5.0, 10.5.1, ...)`. The tabular line one line
above states the correct narrower property ("appears in no DECLARATION in the
head"); the summary sentence overstates it. The conclusion ("none is the 86.19
truncation shape") survives. phase-10.5.0 also shows the census can over-flag a
legitimate BATCH contract, so 153 has contestable positives as well as >=7 false
negatives.

**F5 [NOTE] Section B is the only evidence block with no `$ command` line**, and
its "456 suffix-convention files" does not reproduce under four rules I tried
(443 / 410 / 521 / 454). NOT load-bearing -- the zeros + positive control, which
are, I reproduced exactly.

### G. Attack points, answered
1. Ungraded live hook: disclosure is COMPLETE and CORRECTLY SIZED (names the
   mechanism, the counterfactual casualty, verifies the outcome itself, and
   states what the witness does NOT prove). It violates NO immutable criterion --
   criterion 3 forbids the DEMONSTRATION touching handoff/archive, and the
   demonstration ran wholly in temp dirs (isolation asserted, re-verified by me);
   the live dirs came from genuine peer closures. Fail-open trap bounds the blast
   radius. **WARN, not blocking.** I agree the disclosure does not exonerate it.
2. Rolling-guard coverage: **Main's reading is EXACT.** Verified in code (B2).
3. Precision oracle: **partially conceded** -- see F3/F4. Not vacuous (the
   SUSPECT/CONFIRMED controls genuinely both fire, verified) but not independent.
4. Anchor guard: **REAL, verified by execution.** Replaced MUTANTS with an absent
   anchor and a no-op replace -> "ANCHOR MISSING -- refusing to score" +
   "MUTATION DID NOT CHANGE THE TEXT -- refusing to score", rc=1 (FAIL).
   BONUS: the BEFORE-half guard is real too -- PRE_FIX_REF=HEAD -> "REFUSED:
   the recovered script already contains the fix", rc=1.

### H. Criterion roll-up
C1 MET (with F3/F4 attached) | C2 MET (re-derived by me) | C3 MET (BEFORE-guard
verified) | C4 MET both branches (with F1/F2 attached) | C5 MET | C6 MET (4/4
killed, reproduced; guard proven real).

### I. 3rd-CONDITIONAL check
`grep -cE "phase=86\.29 result=CONDITIONAL" handoff/harness_log.md` -> **0**.
Only two entries exist, both research-gate: Cycle 1203 GATE-FAILED, Cycle 1205
GATE-PASSED. Counter NOT armed. CONDITIONAL is permitted on this cycle.

### J. VERDICT REACHED: CONDITIONAL
Every criterion MET, code correct under 11 mutations (4 author + 7 mine), both
scripts reproduce, harness compliance clean, no unintended production change.
CONDITIONAL because four reproducible WARN findings remain -- one false sentence
inside an artifact labelled verbatim, a census that misses >=7 members of its own
population behind a one-character grammar gap, and two fixture blind spots, one of
which lets the exact criterion-4 defect class survive the author's suite.
Not FAIL: no criterion unmet, no defect in shipped behaviour.

```

## What Main did with it

**Re-measured every finding rather than trusting any of them.** The two
substantive ones, verified independently:

```
F4: mismatched dirs whose OWN sid appears somewhere in the head: 47 of 153
F3: unclassified dirs with an EN/EM-DASH in a heading: 38 of 255
F3: of those, GENUINE mismatches the ASCII grammar misses: 7
     phase-75.1               actually declares 75.2
     phase-75.5.12            actually declares 76.9.2
     phase-76.9.3             actually declares 76.9.2
     phase-78.0               actually declares 76.9.2
     phase-78.16              actually declares 76.9.2
     phase-78.2               actually declares 76.9.2
     phase-79.2               actually declares 76.9.2
```

All five findings CONFIRMED and remediated -- see `experiment_results_86.29.md`
section 7 and the regenerated `live_check_86.29.md`. The census grammar now
accepts en/em-dash separators, the overstated sentence is corrected and prints
both numbers, and the prove harness gained two behavioural checks and two
mutation cells covering the failure classes its fixture previously could not
express.

**Not treated as a grade.** No criterion is marked MET on the strength of this
record. The fresh Q/A re-derives everything.

**Not verdict-shopping.** There is no prior verdict to shop, and the tree has
changed materially since the drop. Per the CLAUDE.md cycle-2 flow, a fresh spawn
on changed evidence is the documented path.
