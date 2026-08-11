STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.5
WRITTEN: 2026-08-11T12:11:12Z

# Q/A cycle 2 write-first record -- step 86.5

Scope per spawn prompt: verify ONLY the three remediations from cycle 1
(wf_615f6c44-ddc CONDITIONAL) plus anything the remediation broke.
Cycle 1 already independently verified C3, C5, C6 and the C1 arithmetic.

## Plan
- A. Harness-compliance audit (5 items)
- B. Deterministic: immutable verification command exit code; git status/diff scope
- C. Remediation 1 (C1): live_check_86.5.md 18-row file-level table + 17 node-level
      signature rows; GENERATED not typed; is the "literal 26-node-id table not
      derivable" argument sound or an excuse?
- D. Remediation 2 (C4): six DERIVED steps, coupling PROPERTY test, ZERO live-coupled
      claim; the dod4 ref-count discrepancy (68 claimed vs 63 measured)
- E. Remediation 3 (C2): audit_basis populated on all five filed steps
- F. Regression: anything broken

## Findings (appended as established)

### Harness compliance
- research_brief_86.5.md exists (80,672 bytes, 13:38). contract 13:41. Hmm --
  contract mtime 13:41 > brief 13:38 OK. experiment_results 13:54 > contract. OK.
- harness_log: ZERO entries for phase=86.5 -> log-last satisfied; 3rd-CONDITIONAL
  counter = 0 prior CONDITIONALs in the log (cycle 1's CONDITIONAL not yet logged,
  normal mid-cycle). A CONDITIONAL now would be #2, not #3.
- git status clean except hook-managed heartbeat + audit jsonl. No unintended prod change.

### Immutable verification command
`bash -c 'python3 -c "...print(see success_criteria...)"'` -> EXIT 0.
(NB: the masterplan's immutable command is the parse-only python3 -c, NOT the pytest
command quoted in my spawn prompt. Both recorded.)

### Criterion 5 corroborated independently at my own clock
handoff/kill_switch_audit.jsonl NOW: sha256 ab7324eb...455f, 66 lines -- byte-identical
to Main's BEFORE and AFTER pair. Also unchanged across MY mutation experiment below
(sink file never created).

### *** BLOCKING FINDING -- criterion 4 answer is WRONG, and criterion 1 row 1 is mis-classified ***

Main claims: "ZERO of the six are LIVE-COUPLED -- five have no live reach".
The "property" was operationalised as a ref count of `kill_switch|paused|pause`
(the live_check's own `refs` column: five files show 0). That is the SAME PROXY
cycle 1 rejected, relabelled "property". The 36.28 coupling is INDIRECT: per
36.28's own name field, `paper_trader.py` `_kill_switch_refusal_for_buy()` does
`state = self._injected_ks_state or get_state()` -- it falls back to the MODULE
SINGLETON (which replays the real on-disk audit) whenever a test constructs
PaperTrader WITHOUT `kill_switch_state`. A file can therefore be fully coupled
with ZERO textual references to "kill_switch".

MEASURED -- all five "no live reach" files construct PaperTrader uninjected:
  test_64_3_currency_path.py:59            `trader = pt.PaperTrader(s, bq)`
  test_64_4_multi_market_e2e.py:144        `trader = pt.PaperTrader(s, bq)`
  test_phase_70_3_atomic_swap.py:207       `trader = pt.PaperTrader(s, bq)`
  test_price_tolerance_gate.py:63          `return PaperTrader(settings=settings, bq_client=bq)`
  test_phase_70_4_gate_observability.py:68 `trader = pt.PaperTrader(get_settings(), bq)`

MUTATION MATRIX (in-memory only; kill_switch._AUDIT_PATH redirected to a tmp SINK
which was never created; live audit sha256 unchanged before/after):
  CONTROL (live tree, book unpaused): test_64_3_currency_path -> 4 passed, 0 failed
  MUTANT  (singleton forced paused):  test_64_3_currency_path -> **3 FAILED**, 1 passed
     FAILED test_64_3_currency_path_kr_avg_entry_stays_krw
     FAILED test_64_3_currency_path_eu_avg_entry_stays_eur
     FAILED test_64_3_currency_path_us_byte_identical

THREE. Exactly the "3" this file contributed to the 26-failure baseline, and exactly
the "(3 failures)" 36.28's own audit text attributes to the kill-switch coupling.
Those 3 of the 26 are ENVIRONMENT ARTIFACTS (state-dependent on the operator's live
pause state), NOT "already fixed". live_check_86.5.md table A row 1 says
"**already fixed** -- absent from today's run"; 36.28 is still status=pending, so
nothing fixed the coupling. Criterion 1's taxonomy HAS the right bucket
("or shown to be an environment artifact") -- this is a wrong cell, not a gap.

Criterion 4 requires "state which of the 26 are instances of the live-kill-switch-
coupling class". Stated: ZERO. Measured: at least 3. The downstream conclusion
("no duplicate step owed") survives, but by luck again -- the same failure shape
cycle 1 flagged, repeated one level deeper.

### dod4 ref count 68 vs 63 -- RECONCILED (Main said it could not be)
Both are right; they count different things.
  `grep -cE`  (LINES containing a match)      = 63
  `grep -oiE ... | wc -l` (OCCURRENCES)       = 68
Case-insensitivity is NOT the difference: `grep -icE` also returns 63. Main's stated
explanation ("I measure 68 with a case-insensitive pattern") is wrong about its own
cause. Immaterial to the verdict, NOTE-level.

### dod4 tmp-isolation claim -- CONFIRMED
backend/tests/test_dod4_tier1_coverage_investment.py lines 73/86/100/111/131/144:
`monkeypatch.setattr(kill_switch, "_AUDIT_PATH", audit_path)` with
`audit_path = tmp_path / "kill_switch_audit.jsonl"`. Genuine -- but it does NOT
make the file uncoupled: dod4's execute_buy test still constructs PaperTrader
uninjected and goes red under a paused singleton (matrix below).

### MY FIRST PROBE WAS DEFECTIVE -- corrected, recorded (do not repeat)
My first matrix pointed `kill_switch._AUDIT_PATH` at an EMPTY tmp sink. That
starves `baselines_present`, and `_kill_switch_refusal_for_buy` FAILS CLOSED on a
missing baseline -- so the CONTROL arm showed 7 phantom failures. A red control is
an indictment of the probe until proven otherwise. Corrected probe: (a) clean
control = plain `pytest <file>` in its own process, no patching at all; (b) mutant
= `_AUDIT_PATH` pointed at a COPY of the real audit so baselines replay identically,
with `paused` the ONLY variable; (c) one process per file, no cross-file leakage.

### *** THE DECISIVE MATRIX -- criterion 4's answer is INVERTED ***

| file | baseline 2026-08-08 | clean control today | MUTANT (paused only) |
|---|---|---|---|
| test_64_3_currency_path            | 3 | 0 (4 passed)  | **3 failed** |
| test_64_4_multi_market_e2e         | 1 | 0 (6 passed)  | **1 failed** |
| test_dod4_tier1_coverage_investment| 1 | 0 (72 passed) | **1 failed** |
| test_phase_70_3_atomic_swap        | 1 | 0 (11 passed) | **1 failed** |
| test_price_tolerance_gate          | 3 | 0 (6 passed)  | **3 failed** |
| test_phase_70_4_gate_observability | 2 | 0 (7 passed)  | **2 failed** |
| **TOTAL**                          | **11** | **0** | **11** |

Per-file cardinality matches the 2026-08-08 baseline EXACTLY across six independent
files. Live audit sha256 unchanged (ab7324eb...455f, 66 lines) before and after every
run; the tmp sink was never created.

**ALL SIX are live-kill-switch-coupled. ELEVEN of the 26 are that class.**
Stated in live_check §C: "ZERO of the six are LIVE-COUPLED". Inverted.

Consequences:
- live_check table A rows 1, 2, 4, 11, 12, 18 (= 11 failures) say "**already fixed**
  -- absent from today's run". They are ENVIRONMENT ARTIFACTS: the operator's book
  was paused on 2026-08-08 and is unpaused now. Criterion 1 offers exactly that
  bucket; 11 of 26 (42%) sit in the wrong cell. "Already fixed" implies durability --
  these all return the next time the operator pauses.
- experiment_results §2 "H1 ... REFUTED" is itself refuted; its refutation rested on
  the same 0-ref proxy.
- experiment_results §2 causal story ("autonomous_loop.py has 12 commits ... they now
  pass") is contradicted: the tests flip red on a single in-memory flag with the tree
  UNCHANGED. Production commits are not what turned them green. (Main did mark that
  attribution "unestablished" -- correct caution, wrong conclusion kept anyway.)
- 36.28 is still status=pending, so nothing fixed the coupling.
- Downstream conclusion "no duplicate step owed" SURVIVES (36.28 owns the class and no
  duplicates were filed) -- right answer, third consecutive time by luck.

### Criterion 2 -- MET
86.48 audit_basis 1302ch / 86.49 1013 / 86.50 1184 / 86.51 840 / 86.52 1071.
All five: status=pending, harness_required=True, 4-5 success_criteria, parse-only
`ast.parse` verification commands (green-able, verified shape).

### Criterion 6 -- MET
`git status --porcelain backend/tests/` empty. No test edited.

### Criterion 3 -- MET (cycle 1 verified; live_check §B carries 17 measured signatures)
### Criterion 5 -- MET (reproduced independently at my clock, twice)

## VERDICT SENT: FAIL
Criterion 4 answered inverted (ZERO vs measured ALL SIX / 11 of 26); criterion 1's
11 rows mis-disposed as "already fixed" rather than environment artifact. Remedy is
cheap (re-label 11 rows + reverse C4's answer); the five filed steps stand and need
no rework.

COMPLETED: 2026-08-11T12:29:41Z
