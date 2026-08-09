# Contract -- phase-86.6

**Step:** 86.6 (P1) -- the channels a conftest guard cannot reach are still open:
(A) an in-process test write to `handoff/kill_switch_audit.jsonl` is DETECTED but
not PREVENTED, and (B) a test that SHELLS OUT runs with no guard at all.
**Date:** 2026-08-10. **Cycle:** 198.
**Research gate:** PASSED -- `handoff/current/research_brief_86.6.md`
(run `wf_dc58bae7-aef`: 18 sources read in full >= floor 5, 44 URLs >= floor 10,
recency scan performed, audit-class **coverage.dry = true** after 10 rounds with
2 consecutive dry rounds, 20 internal files inspected; 39,124 chars independently
re-read on disk and all 18 claimed URLs verified present).

> **Gate-run disclosure.** The FIRST attempt (`wf_a2be8f28-a3f`) did the research
> -- 69 tool calls, 223K tokens -- and then **dropped its return without calling
> StructuredOutput**, which is a FAILED gate, never a pass. Write-first is why
> nothing was lost: a 615-line brief was already on disk. The gate was re-run
> with a lean prompt to complete the summary tables and emit the envelope. The
> floors were then met by what the brief actually contains, cross-checked
> independently. This is recorded because a dropped return that leaves a good
> artifact is exactly the situation in which it would be tempting to proceed
> without a gate.

---

## 1. Research-gate summary -- and the finding that would have made my fix vacuous

**MEASURED, not reasoned (pytest 9.0.3 in the project venv):**

```
Failed MRO: ['Failed', 'OutcomeException', 'BaseException', 'object']
bare except Exception -> ESCAPED as Failed
kill_switch shape: logger.warning swallowed: RuntimeError
kill_switch shape: logger.warning swallowed: PermissionError
```

`kill_switch._append_audit` catches `Exception` and logs a warning
(`kill_switch.py:498-499`). **Both existing in-repo guards raise `RuntimeError`**
(`conftest.py:129-141`, `backend/tests/conftest.py:52-58`). `RuntimeError` IS an
`Exception`. **So a filesystem guard built to the established in-repo pattern
would be silently absorbed: the write is blocked, the test goes GREEN, and
nobody learns anything.** `assert` is unsafe for the same reason --
`AssertionError` is an `Exception`.

**Consequence for the design: the refusal MUST derive from `BaseException`.**
This is the single most important thing the research produced, and it is the
opposite of what the repo's own precedent would have led me to write.

**Other load-bearing findings:**

- **`sys.addaudithook` (PEP 578) is the only seam** that covers `open`,
  `subprocess.Popen`, `socket.connect` and `urllib.Request` at once. Blocking is
  supported; measured overhead ~1.05x. **But hooks "cannot be removed or
  replaced" once installed, and PEP 578 explicitly says it "is not sandboxing"**
  -- so it is a guard against accident, not against a determined caller, and the
  artifacts must say so.
- **FOUR module-level constants point at the one live journal**:
  `kill_switch.py:48` (the writer), `paper_trading.py:892` (a duplicate,
  READ-ONLY -- an **isolation illusion**, since redirecting the writer's constant
  leaves this one pointing at the live file), `risk_overrides.py:128`,
  `cron_control.py:60`. Twenty files carry module-level `handoff/` paths.
- **Child processes load NO conftest** (72 files shell out). This is why Part B
  cannot be solved by any conftest-only guard, and why criterion 7 demands a
  seam a FUTURE call site cannot slip past rather than a census of today's.
- **`PYFINAGENT_TEST_NO_BQ` is honoured at only 2 sites**, never in
  `bigquery_client.py`, and is absent from the `tests/` tree.
- The guard **must allow READS** (`test_phase_23_2_4...:253-255` reads the live
  journal and must keep doing so) and must **install at conftest IMPORT**, not
  in an autouse fixture, because collection is what has to be covered.

## 2. Hypothesis

A single `sys.addaudithook` installed at repo-root conftest import, refusing
WRITE-intent operations against a derived live-path set with a
`BaseException`-derived refusal, converts the filesystem channel from detected
to PREVENTED without touching production (conftest is not loaded in production)
and without breaking the test that must keep READING the live journal. The
subprocess channel needs a different seam, because a child process inherits the
environment but not the conftest.

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. "the population of in-process writers is derived by a method that is PROVEN not to produce the known false negative -- run the chosen method against backend/tests/test_book_safety_69.py::test_peak_reset_dark_by_default and show it is FLAGGED; a method that reports that file clean is rejected regardless of what else it finds"
2. "an in-process mutating kill_switch call is PREVENTED (not merely detected) while _AUDIT_PATH points at the live file -- the live journal's sha256 is byte-identical across a run that attempts one, and the attempt fails loudly naming the offending test"
3. "tests that redirect _AUDIT_PATH to tmp still write normally -- asserted against test_phase_36_7_kill_switch_rotation_rearm.py and the 86.3 in-process cycle, both unchanged in status"
4. "test_phase_23_2_4_audit_log_clean_transitions still PASSES and still reads the LIVE journal with its trigger allowlist byte-unchanged -- a blanket _AUDIT_PATH redirect that breaks it is an explicit failure, not a trade-off"
5. "what a refusal does to every PRODUCTION caller of _append_audit is MEASURED and recorded (it currently swallows write errors via logger.warning), with an explicit statement that production is no noisier and no less safe than before"
6. "MUTATION-TEST: removing the preventer must fail the criterion-2 test, and the mutation must be run against a tmp COPY of the journal, never the live file"
7. "PART B: the subprocess channel is closed at a seam that a NEW unguarded call site cannot slip past -- a test FAILS when a run_smoke-style invocation omits --backend-url (or the equivalent for whatever seam is chosen), proven by adding such a call site in the test itself and showing it is rejected; enumerating the current 12 compliant call sites is NOT sufficient because the defect is a future 13th"
8. "PART B: if the fix changes scripts/qa/smoke_cc_rail_e2e.py's --backend-url default, every production/ops caller of that script is enumerated by grep FIRST and shown unbroken (it is an ops script; a newly-required flag can break a runbook) -- and if no caller census is produced, that option is not taken"
9. "the artifacts enumerate filesystem / HTTP / subprocess / BigQuery / module-singleton explicitly and state which are covered and which are not -- an isolation claim that names fewer channels than this list is incomplete by definition"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_book_safety_69.py backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py backend/tests/test_phase_36_12_kill_switch_trading_path_block.py backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py -q --timeout=180`
(baseline measured BEFORE any code: **79 passed, exit 0**)

**live_check (immutable):** "live_check_86.6.md with the line count and sha256 of handoff/kill_switch_audit.jsonl before and after a full backend/tests run, the derived writer population with the test_book_safety_69 false-negative check shown, and the verbatim refusal message"

## 4. Design

**A. Filesystem preventer.** `sys.addaudithook` at repo-root conftest import.
Refuse when the audited event is a write-intent open (or a write-class file
operation) whose resolved path is in the DERIVED live-state set. Reads pass
untouched. The refusal is a `BaseException` subclass so
`kill_switch.py:498-499` cannot absorb it, and its message NAMES the offending
test (via `PYTEST_CURRENT_TEST`) and the path.

**The live-state set is DERIVED, not hand-listed** -- that is criterion 1's
requirement and the reason the previous md5 sweep missed
`handoff/.autonomous_loop.lock` (untracked, so a tracked-file sweep could not
see it). The derivation's RECALL is validated against the named known positive:
if the method reports `test_book_safety_69.py::test_peak_reset_dark_by_default`
clean, the method is rejected regardless of what else it finds.

**B. Subprocess seam.** A conftest guard cannot reach a child. The seam must be
one a NEW call site cannot bypass -- an interlock the child itself checks --
and criterion 7 requires proving it by ADDING an offending call site in the test
and showing it is rejected, not by enumerating the 12 compliant ones.
**If and only if** the chosen seam changes `smoke_cc_rail_e2e.py`'s
`--backend-url` default, criterion 8 requires a grep census of every
production/ops caller FIRST; **if no census is produced, that option is not
taken.**

**C. Production impact (criterion 5) is MEASURED, not assumed.** The guard lives
in conftest, which production never imports -- but that must be shown per
production caller of `_append_audit`, not asserted, and the artifacts must state
plainly that production is no noisier and no less safe.

## 5. Plan

1. **[done]** Research gate -- PASSED, `research_brief_86.6.md`.
2. **[this file]** Contract, BEFORE any code.
3. Derive the live-state set; validate the method's recall against the known
   false negative FIRST (criterion 1), before building anything on it.
4. Build the `BaseException`-derived refusal + the audit hook; prove a blocked
   write is NOT absorbed by `kill_switch.py:498-499` (the vacuity trap).
5. Assert criteria 3 and 4: tmp-redirected tests still write; the live-journal
   READER still passes with its allowlist byte-unchanged.
6. Measure criterion 5 per production caller.
7. Part B seam + the added-offending-call-site proof (criterion 7).
8. Mutation-test against a tmp COPY of the journal, never the live file
   (criterion 6).
9. Enumerate all five channels and state which are covered (criterion 9).
10. Q/A via the Workflow rail; transcribe verbatim; log; flip.

## 6. Traps (measured, from the brief)

- **A `RuntimeError`/`assert` refusal is SWALLOWED.** Use `BaseException`.
  This is the trap the repo's own precedent walks into.
- **Redirecting one constant is an isolation ILLUSION** -- four point at the same
  journal, and one of them (`paper_trading.py:892`) is a read-only duplicate.
- **A conftest guard cannot reach a child process.** 72 files shell out.
- **The guard must not block READS**, or `test_phase_23_2_4...` breaks, which
  criterion 4 calls an explicit failure rather than a trade-off.
- **A zero-write assertion passes trivially against a harness that cannot
  observe a write.** Every such assertion needs a POSITIVE control per subject
  -- the lesson from tonight's 86.17 cycle-1, where a checker sliced away the
  guard it was checking.
- **`sys.addaudithook` is not sandboxing and cannot be uninstalled.** Say so;
  do not oversell it.
- **Never run the mutation harness against the live journal** -- criterion 6
  says a tmp COPY, and tonight's 36.17 pattern (in-memory injection, digests
  asserted unchanged) is the house idiom.

## 7. References

- `handoff/current/research_brief_86.6.md` (18 read in full, 44 URLs).
- PEP 578 -- https://peps.python.org/pep-0578/
- `sys.addaudithook` / audit events -- https://docs.python.org/3/library/audit_events.html
- CWE-390 (detect error without action) -- https://cwe.mitre.org/data/definitions/390.html
- Saltzer & Schroeder -- https://web.mit.edu/Saltzer/www/publications/protection/Basic.html
- Gyori et al., PolDet (test pollution detection) -- https://mir.cs.illinois.edu/marinov/publications/GyoriETAL15PollutionDetection.pdf
- Google SWE Book ch.14 (Larger Testing) -- https://abseil.io/resources/swe-book/html/ch14.html
- Internal: `conftest.py`, `backend/services/kill_switch.py`,
  `backend/services/risk_overrides.py`, `backend/services/cron_control.py`,
  `backend/api/paper_trading.py`, `scripts/qa/smoke_cc_rail_e2e.py`.
