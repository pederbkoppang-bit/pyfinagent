# phase-86.6 -- GENERATE

**Step:** 86.6 -- the channels a conftest guard cannot reach.
**Contract:** `handoff/current/contract_86.6.md`
**Research:** `handoff/current/research_brief_86.6.md` (gate PASSED, `wf_dc58bae7-aef`;
18 sources read in full, 44 URLs, audit-class `coverage.dry` after 10 rounds)

This step was PARKED mid-way overnight and resumed. The parked state is recorded
in `handoff/current/progress_86.6_PARKED.md`; everything below either restates a
measurement from it or is new work done on resumption.

---

## 1. The two channels, and why one guard could never cover both

phase-86.3 closed the HTTP channel: a test that POSTs to the live backend is
refused by a `conftest.py` guard. That guard has two blind spots, and they share
a root -- **a guard lives in one process and on one path, and neither boundary is
where the danger stops.**

| | channel | why 86.3's guard cannot see it |
|---|---|---|
| **A** | a test calls a mutating `kill_switch` method IN-PROCESS while `_AUDIT_PATH` still points at the live file | no network is involved; the bytes go straight to disk |
| **B** | a test SHELLS OUT | a child process loads no conftest, so the guard does not exist there |

## 2. PART A -- prevention, not detection

### Criterion 1 -- the derivation, with its recall VALIDATED

`scripts/qa/derive_live_state_writers_86_6.py` derives the mutating surface FROM
`kill_switch.py` (never a hand-list: **7 methods** reach `_append_audit`) and
AST-scans both test trees for invocations.

The criterion names one test as the probe and rejects any method that reports it
clean. It is flagged:

```
RECALL VALIDATION (criterion 1)
  probe: test_book_safety_69.py::test_peak_reset_dark_by_default
  FLAGGED -- calls reset_peak; redirects=True
```

**Why a static method was required.** A RUNTIME write-detector reports that probe
CLEAN. `reset_peak` returns early while `kill_switch_peak_reset_enabled` is
False, so no bytes ever reach the journal -- the call site sits waiting for the
already-approved KS-PEAK-RESET token. The population must be derived from the
CALL, not from the WRITE.

**Population: 54 tests invoke a mutating kill-switch API; 24 without a redirect
the detector can see.** PRECISION IS UNVALIDATED and I am stating that rather
than burying it: the detector only recognises `_`-prefixed module-scope
fixtures, so a test isolated by a *named* fixture reads as unprotected.
Empirically those files run **164 passed with the live journal byte-identical**,
so none writes today. The risk is LATENT -- which is exactly 86.1's landmine
shape, and the reason this step is a preventer rather than a census.

### Criteria 2, 3, 5 -- the preventer and its cost

Installed at repo-root `conftest.py` import via `sys.addaudithook` (PEP 578).

**The trap it had to avoid, which inverts the repo's own precedent.**
`kill_switch._append_audit` catches `Exception` and swallows it with a
`logger.warning`. Both existing in-repo guards raise `RuntimeError` -- which IS
an `Exception`. A refusal built to the house pattern would be **silently
absorbed: write blocked, refusal swallowed, test GREEN.** So
`LiveStateWriteRefused` derives from `BaseException`, and
`backend/tests/test_phase_86_6_live_state_preventer.py` asserts that property
against the production shape rather than assuming it, with a positive control
proving the swallow really does absorb an `Exception`.

New in this resumption -- **the parked work had no test of its own**:

| criterion | assertion |
|---|---|
| 2 | an in-process append to the live journal raises `LiveStateWriteRefused` AND the file's sha256 is unmoved; the message names the offending test |
| 3 | a tmp-redirected write proceeds normally (the established `ks_tmp_audit` idiom is not broken) |
| 5 | production measured in a **child with `-I` isolated mode**: `conftest` is absent from `sys.modules`, and `KillSwitchState._append_audit` writes successfully |

Criterion 5 is measured rather than argued. "Production never imports conftest"
is a sound argument and the step requires a measurement, so the child asserts
`"conftest" not in sys.modules` and then performs a real append. A positive
control asserts the guard IS active in the pytest process, so "production is
unaffected" cannot be trivially true because nothing is guarded anywhere.

### Criterion 4 -- the live-journal READER is untouched

`test_phase_23_2_4_pause_resume_no_deadlock_live.py` **5 passed**, and
`git diff --stat` on that file is empty, so its 36.21-inherited trigger allowlist
is byte-unchanged. A blanket `_AUDIT_PATH` redirect would have broken it; the
guard blocks WRITES only and reads pass through untouched (asserted).

### Criterion 6 -- the mutation matrix, run entirely on a COPY

The criterion says the mutation must run "against a tmp COPY of the journal,
never the live file", and that second half shapes the whole harness: the obvious
approach -- disable the guard, run the criterion-2 test, watch it fail -- WRITES
TO THE LIVE JOURNAL, because with the guard gone nothing stops the write.
Restoring from a backup afterwards does not satisfy it either; the bytes still
land, and a crash between write and restore leaves them there.

`scripts/qa/mutation_matrix_86_6.py` copies the live journal to tmp, re-executes
the guard source in a child with `_BLOCKED_PATHS` redirected at that copy, and
digests the live file before and after the whole run.

```
id       refused   copy changed   verdict   mutation
CONTROL  True      False          ok        guard intact
M1       False     True           ok        remove the preventer entirely
M2       False     True           ok        downgrade the refusal to a log line
M3       True      False          ok        make the refusal an Exception
M4       True      False          ok        drop 'a' from the write-intent MODE chars
M5       False     True           ok        drop BOTH append legs (mode AND O_APPEND)
  live journal UNCHANGED (byte-identical across the whole matrix)
```

**M4 was predicted to survive and did not, and the guard was right rather than
the matrix.** `_is_write_intent` has TWO independent legs -- the mode string and
the os flags -- and `open(p, "a")` sets `O_APPEND` regardless of the mode chars,
so the flags leg still catches it. I corrected the expectation instead of
weakening the guard, and added **M5** to close the obvious follow-up: a redundant
leg is only PROVEN redundant when removing both opens the hole. It does.

M3 is the interesting cell: it still refuses, and the harness additionally
reports that production's `except Exception` **would** have swallowed it -- the
exact silent-success trap, demonstrated rather than described.

## 3. PART B -- the subprocess channel (entirely new in this resumption)

### Criterion 8 FIRST, because it gates what criterion 7 may do

The criterion forbids changing `smoke_cc_rail_e2e.py`'s `--backend-url` default
unless every production/ops caller is enumerated first. **Census, by grep across
every file type:**

| reference | kind |
|---|---|
| `backend/tests/test_phase_4000_2_cc_rail_smoke.py` | the test module (12 call sites) |
| `scripts/qa/smoke_cc_rail_e2e.py` | the script itself |
| `.claude/masterplan.json`, `handoff/**`, `.claude/agent-memory/**` | records and briefs, not callers |

`~/Library/LaunchAgents/com.pyfinagent.*`, `docs/runbooks/`, `scripts/ops/`:
**no match.** There is **no production or ops caller**, so no runbook can break
and the option is permitted.

### Criterion 7 -- a seam a NEW call site cannot slip past

Two changes to `scripts/qa/smoke_cc_rail_e2e.py`:

1. **`--backend-url` has no default.** It used to default to
   `http://localhost:8000` -- the operator's live backend -- while the script
   MUTATES (`PUT /api/settings/`, `POST /api/analysis/`).
2. **`--allow-live-backend`**, required to target loopback:8000 at all.

Both matter. A required flag closes OMISSION; it does nothing about a new call
site that passes `http://localhost:8000` explicitly, which is the same mistake
with more typing. Guarding only the first would be guarding the instance.

`backend/tests/test_phase_86_6_subprocess_channel.py` **adds the offending call
site itself**, as the criterion demands:

```python
def test_a_new_call_site_that_OMITS_backend_url_is_REJECTED():
    proc = run_smoke("--dry")          # <-- the offending call site
    assert proc.returncode != 0
    assert "--backend-url is REQUIRED" in proc.stderr
```

Verbatim refusals:

```
usage error: --backend-url is REQUIRED and has no default. This script mutates
(PUT /api/settings/, POST /api/analysis/); a default of http://localhost:8000
would aim those at the operator's running backend. ...

refusing to target the LIVE backend at http://localhost:8000 without
--allow-live-backend. ...
```

**Positive controls, so the refusals are not a blanket denial:**
`--allow-live-backend` lifts the refusal (phase-4000.3 needs a real window), and
an ephemeral stub URL (`127.0.0.1:59999`) passes untouched. The guard keys on the
PORT, not the host, precisely because the twelve existing call sites stand up
stubs on `127.0.0.1:0`; keying on the host would have broken every one of them.

**No second definition of "live".** `scripts/qa/live_backend_origin.py` holds the
predicate once and the script imports it. `conftest.py` keeps its own copy --
rewiring an already-graded guard is a risk this step does not need -- and
`test_the_two_live_origin_predicates_AGREE` compares the two across a table so a
drift fails a test instead of silently opening one of the channels. That trade is
stated, not hidden: one authority for the new code, an alarm on the old.

**Regression:** `test_phase_4000_2_cc_rail_smoke.py` **22 passed** -- all twelve
existing call sites unaffected.

## 4. Criterion 9 -- the five channels, explicitly

An isolation claim that names fewer channels than this list is incomplete by
definition. So:

| channel | status | evidence / why not |
|---|---|---|
| **filesystem** | **COVERED for the kill-switch journal + its derived archive dir** | `sys.addaudithook` preventer; mutation matrix M1/M2/M5 |
| **filesystem (rest of `handoff/`)** | **NOT COVERED, deliberately** | blocking the whole tree turns **+7** tests red against a 14-failure baseline -- they write `.autonomous_loop.lock`, `.cycle_heartbeat.json` and a probe under `handoff/logs/`. None is a kill-switch write. A behaviour change to 7 real tests belongs to its own step. **MEASURED AGAIN TONIGHT:** a full-suite run wrote a real acquire/release into the live `handoff/.autonomous_loop.lock` (`released_at 2026-08-09T23:47:31Z`, pid 19697 already dead) -- so this channel is demonstrably still open, not theoretically open. |
| **HTTP** | **COVERED (86.3)** | mutating verbs at loopback:8000 refused in-process |
| **subprocess** | **COVERED for `smoke_cc_rail_e2e.py`** | the seam above. **NOT covered generally**: research measured 72 files that shell out, and each would need its own seam. This step closed the one that MUTATES the live backend. |
| **BigQuery** | **NOT COVERED** | no guard exists. Tests that construct a real `bigquery.Client` reach live datasets. Out of scope here; named so it cannot be mistaken for covered. |
| **module singleton** | **NOT COVERED** | a test mutating `ks._state` in-process is invisible to a filesystem guard. This is 36.28's territory (tests READING live state) and 86.1's surviving mutant M2. |

**Three of six are covered. Two are explicitly out of scope and one is
partially covered.** That is the honest count.

## 5. Files changed

```
scripts/qa/live_backend_origin.py                     (NEW -- one definition of "live")
scripts/qa/smoke_cc_rail_e2e.py                       (no default + --allow-live-backend)
scripts/qa/mutation_matrix_86_6.py                    (NEW -- criterion 6, copy-only)
backend/tests/test_phase_86_6_subprocess_channel.py   (NEW -- criterion 7, 18 tests)
backend/tests/test_phase_86_6_live_state_preventer.py (NEW -- criteria 2/3/5, 8 tests)
conftest.py                                           (the preventer; committed 03b3ea17)
```

## 6. Not claimed

- **The subprocess channel is not closed in general** -- 72 files shell out; one
  seam is closed.
- **BigQuery and module-singleton channels are open.**
- **The rest of the `handoff/` live-state tree is unguarded**, and tonight's run
  proved it by writing the cycle lock.
- The derivation's **precision is unvalidated** (24 flagged; empirically none
  writes today).
