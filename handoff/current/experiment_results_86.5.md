# Experiment results -- step 86.5

**Step**: `86.5` (phase-86, P2) | **Phase**: GENERATE | **Date**: 2026-08-11
**Driver**: Main (`pyfinagent-06`) | **Contract**: `32006221` (written BEFORE any code)

**The deliverable is THE STEPS. No test was edited and no production file changed.**

---

## 1. Criterion 5 -- the baseline, and the non-touch PROVEN with a pair

Two full runs, both live-tree:

```
run 1  17 failed, 3417 passed, 12 skipped, 5 xfailed, 1 xpassed in 441.51s
run 2  17 failed, 3417 passed, 12 skipped, 5 xfailed, 1 xpassed in 416.38s
failure SET: byte-identical between runs
```

`handoff/kill_switch_audit.jsonl`, captured around run 2:

```
BEFORE 2026-08-11T11:45:34Z   lines 66   bytes 6618   sha256 ab7324ebf501e3d3886e62a5d8fd2ed4f01f675849702b6553a4df691aab455f
AFTER  2026-08-11T11:52:34Z   lines 66   bytes 6618   sha256 ab7324ebf501e3d3886e62a5d8fd2ed4f01f675849702b6553a4df691aab455f
UNCHANGED: True
```

**86.3's guard held, measured rather than assumed.** My first run (13:14Z) had only
an after-hash; the gate flagged that, so run 2 was bracketed properly.

## 2. Criterion 1 -- ALL 26 accounted for, line by line

The baseline is **26 failed / 3017 passed (2026-08-08)**; today is **17 / 3417**.
That is **not "9 were fixed"**:

| movement | failures | files |
|---|---|---|
| DISAPPEARED entirely | **-14** | 9 |
| count GREW | **+2** | 2 (`75_17` 2->3, `75_sre_ops` 1->2) |
| NEW since baseline | **+3** | 2 (`75_19` x1, `82_48` x2) |
| unchanged | 9 | 7 |

**`26 - 14 + 2 + 3 = 17`.** The suite also gained ~400 tests (3017 -> 3417), so
**26 is STALE, not WRONG** -- the population moved under it.

### The 14 that disappeared -- WHAT I FIRST CONCLUDED WAS WRONG, BOTH TIMES

> **REWRITTEN after the cycle-2 FAIL and the cycle-3 finding.** An earlier revision
> of this section said "TWO HYPOTHESES RAISED AND BOTH REFUTED" and concluded "Only
> `test_phase_23_2_4` is genuinely coupled". **Both hypotheses were in fact CORRECT
> and my refutations were wrong.** The cycle-3 Q/A caught that my previous
> correction reversed H1 by name while leaving H2's "ALSO REFUTED" standing, so the
> old conclusions survived in different words after a literal-string sweep found
> nothing. Corrected here IN PLACE.

**H1 -- the 36.28 kill-switch-coupled cluster. CORRECT.** I raised it, then
"refuted" it with `grep -c 'kill_switch|paused|pause'` (five files returned 0). That
proxy is structurally incapable of seeing the coupling: `paper_trader.py:202` does
`state = self._injected_ks_state or get_state()`, so a test constructing
`PaperTrader` uninjected couples through the module singleton **with zero textual
references**. A flag-flip matrix turns all six RED under `paused`, 11 failures
matching the 2026-08-08 baseline exactly.

**H2 -- environment artifacts. ALSO CORRECT.** I "refuted" it by noting those files
carry 0-1 live references and no skip markers -- the same text-shaped reasoning. The
11 are precisely environment artifacts: green only because the operator's book is
unpaused today, red the moment it pauses. **36.28 owns them and is still
`status: pending`**, so nothing fixed them.

**What I attributed the greening to was also wrong.** I wrote that
`autonomous_loop.py`'s 12 commits and `orchestrator.py`'s 3 explained it, marking
the attribution "unestablished". The flag-flip matrix refutes it **with the tree
unchanged** -- no commit was needed; only the pause state differs.

**The methodological failure, stated once:** all three wrong conclusions came from
asking *what does the source text say* instead of *what changes if the hypothesis is
true*. A grep cannot answer a runtime-reachability question. Flipping the state
answers it in one move.

## 3. Criterion 4 -- ALL SIX ARE COUPLED (11 of the 26). My answer was INVERTED.

**This section previously read "Measured: ONE", and a later revision made it "ZERO".
Both were wrong. The measured answer is ALL SIX.** Corrected after the cycle-2 FAIL;
full evidence in `live_check_86.5.md` §C.

A flag-flip mutation matrix -- control unpatched, mutant with the kill_switch
singleton forced `paused` and the real audit copied -- turns **all six 36.28-named
files RED, 11 failures matching the 2026-08-08 baseline exactly**:

```
64_3=3  64_4=1  dod4=1  70_3=1  price_tolerance=3  70_4=2  = 11
```

**Why every grep-shaped probe was blind:** `paper_trader.py:202` does
`state = self._injected_ks_state or get_state()`, falling back to the module
singleton. A test constructing `PaperTrader` uninjected is coupled with **zero**
textual `kill_switch` references -- and all five do exactly that (verified at
`64_3:59`, `64_4:144`, `70_3:207`, `price_tolerance:63`, `70_4:68`).

**The process failure matters more than the wrong number.** Cycle 1 rejected my
ref-count proxy; my remediation read the same column, called it "the coupling
PROPERTY", and re-derived the same wrong answer with more confidence. **Renaming a
proxy does not make it a property.** And H1 above -- which I raised and refuted with
that grep -- was correct all along.

**No duplicate step is owed for the 11, but NOT because they are uncoupled:** they
are owned by **36.28, still `status: pending`**. Nothing fixed the coupling; the
book is unpaused today, and they return when it pauses.
`test_phase_23_2_4_pause_resume_no_deadlock_live` is separately coupled, is **not**
among the six, and was fixed by 86.3 (`4f10b024`).

## 4. Criterion 3 -- grouping by MEASURED signature

Every group below is justified by the assertion/exception the test actually
produces (captured from run 2), never by filename. Signatures are in
`measurement_86.5_failure_census.md` and the brief's group tables.

## 5. Criterion 2 -- THE FILED STEPS (the deliverable)

**Five steps, by REMEDY rather than by root cause** -- the gate's recommendation
and my agreement: `4+1+9+1+2 = 17`.

| new step | n | remedy shape | THE TRAP IT MUST CARRY |
|---|---|---|---|
| **86.48** | 4 | tests read `Settings()` which resolves `backend/.env`, so they measure an operator override, not the code default they name. Fix: construct Settings with the two fields overridden explicitly. | **DO NOT flip the defaults or `.env`** -- `PAPER_DATA_INTEGRITY_ENABLED` and `PAPER_RISK_JUDGE_REJECT_BINDING` are both `true` and both verified `True`. Green bought that way **disarms two armed money-path flags**. `Settings(_env_file=None)` **does not work** -- it raises `ValidationError` on 4 other required fields (verified). |
| **86.49** | 1 | `test_c6` scans `*.sh` line-by-line, skipping only `#`, and is **heredoc-blind**. Fix: make the scanner heredoc- and quote-aware. | **DO NOT edit `reissue_cc_oauth_token.sh`** -- the hit is a variable printed to the operator inside a `cat <<EOF`, never executed. Editing it **deletes the operator's restart instructions**, which away-ops rail 9 deliberately reserves. |
| **86.50** | 9 | tests frozen against MOVING artifacts (live masterplan, archived handoff files, a superseded config value). Fix: re-baseline or bind to a fixed fixture. | One of these is **NOT** stale: `test_sweep_over_live_masterplan_is_clean` caught a **real** never-existed path (`.claude/hooks/lib/qa_write_guard.py`) in step **86.31**. Fix the reference, do not re-baseline over it. Also includes **a quarantine that was declared and never wired** -- its skip never fires because the 14MB `backend.log` is present. |
| **86.51** | 1 | `test_portfolio_swap`: expected 2 swap SELLs, got 1. **Genuine behavioural regression on the trading path.** | Must **not** be triaged as housekeeping or batched with the above. |
| **86.52** | 2 | `test_phase_82_48`: `assert 'UNKNOWN' == 'BUY'` and a row written that should have been skipped. | **NOT a duplicate of 86.25**, which is the assumption to avoid. 86.25 is `done` and owns exactly this vocabulary defect, with three commits including one titled *"finish the remediation I claimed but did not do"*. This step must establish whether 86.25's fix is **incomplete** or the tests assert unshipped behaviour. Note 86.25's own `:3412` citation is now stale -- the line moved. |

## 6. Criterion 6 -- no test edited

`git status --porcelain backend/tests/` is empty. No test file was modified in this
step; the deliverable is the five filed steps above.

## 7. The headline NEGATIVE, reported

**18 research rounds found no mechanical procedure for deciding test-wrong vs
code-wrong.** That dryness is a finding: the per-case adjudication in the five filed
steps **cannot be automated**, and any future step promising to automate it should
be treated sceptically. It is also why every filed step carries its trap inline --
the next executor will have less context than this triage did.

---

## 8. DISPOSITION -- PARKED, and it cannot close without the operator

**Three graded cycles: CONDITIONAL, FAIL, CONDITIONAL.** The standing goal parks a
step that will not close after two, and there is now a harder reason.

### THE STEP IS STRUCTURALLY UNCLOSEABLE -- OPERATOR ASK #5

`86.5`'s **own immutable verification command exits 1**. Extracted from
`.claude/masterplan.json` and run verbatim via `subprocess(shell=True)`:

```
repr:  '...;ids=[];\nwalk=lambda o:[walk(v) for v in ...'   <- LITERAL backslash-n
returncode = 1
stderr: SyntaxError: unexpected character after line continuation character
```

bash does not expand `\n` inside double quotes, so python receives `;\nwalk`.

**I reported "exit=0" for this command earlier today and that does not reproduce.**
I ran a simplified variant; my spawn prompts carried an elided form; both prior
cycles were handed my transcription rather than the stored string. Only running the
stored bytes catches it.

**The defect predates this triage** -- frozen at `a7911f2e` when the step was
queued. **Criteria are immutable and I will not edit one**, so the step cannot
reach a clean PASS on its own terms.

**ASK #5: authorise an immutability exception to repair 86.5's verification
command** (remove the literal `\n`), or direct that the step close on the
success_criteria alone with the broken command disclosed.

The irony is exact and worth keeping: **86.5's own criterion 2 requires running a
proposed command BEFORE freezing it, because "a criterion that is already red for
unrelated reasons is structurally uncloseable".** The step that teaches the rule was
born breaking it.

### WHAT IS COMPLETE AND STANDS

- **The deliverable**: `86.48`-`86.52` filed, each with `audit_basis`,
  `harness_required: true`, and a verification command **verified exit 0** -- none
  carries the `\n` defect.
- **Criterion 1**: 18-row file-level accounting, all 26 dispositioned, arithmetic
  re-derived member-by-member by the Q/A (`26-14+2+3=17`).
- **Criterion 4**: ALL SIX coupled (11 of 26), owned by `36.28` (still `pending`).
- **Criteria 3, 5, 6**: MET and independently reproduced across cycles.

### WHAT IS OUTSTANDING BESIDES THE ASK

Nothing. Blocker 2 (the superseded narrative) is fixed in §2 above, and the
`dod4` understatement is corrected in `live_check` §C.
