# Experiment results -- phase-82.7

**Step**: SECURITY DEFECT -- credentials written to logs in plaintext via URL query strings.
**Contract**: `handoff/current/contract_82.7.md` | **Research**: `handoff/current/research_brief_82.7.md`

## 0. No credential value appears in any artifact of this step

Keys are named by ENV VAR only. The tests use `SYNTHETIC0000NOTAREALKEY0000000A`.

## 1. The bug was REACHABILITY, not absence

`SecretRedactionFilter` has existed since phase-60.4 and its regex was already correct.
It was attached only inside `backend.main.setup_logging`, whose sole non-test call site
is `backend/main.py:151` -- the FastAPI lifespan. Measured by me:
**`logging.basicConfig(` appears 54 times in 54 files** across `backend/ scripts/
functions/`, each installing a root handler with no filter. So every script/CLI path ran
unprotected.

The research gate CONFIRMED this and corrected one of my readings: the *daily macro cron*
is NOT in the unprotected set -- `register_macro_ingest_cron` is invoked from
`backend/main.py:333`, inside the lifespan, so it inherits the filter. **The
2026-08-03 operator-visible leak came from a manual/CLI backfill**, which is the
unprotected class. The fix therefore must not be scoped to `macro_cron.py`.

## 2. What was built

### 2a. `install_secret_redaction()` -- idempotent, and reachable

New helper in `log_redaction.py`. Attaches the filter to every root handler AND, more
importantly, **to the emitting library loggers by name** (`httpx`, `httpcore`,
`urllib3`, `requests`).

The library-logger attachment is the load-bearing half. Handler-level alone is not enough
for a script: a script may call `basicConfig()` *after* install, creating a fresh
unfiltered handler with nothing to re-attach. A filter on the emitting logger scrubs the
record at the source, before any handler exists or matters. `backend/main.py` now
delegates to the same helper so there is ONE attach path.

**Not** `setup_logging()` from scripts: `.claude/masterplan.json:18101` records it as
single-call-only and destructive (`root.handlers.clear()` + a fresh
`TextIOWrapper` over `sys.stderr.buffer`, observed live as
`ValueError: I/O operation on closed file`).

### 2b. Coverage BY CONSTRUCTION, not by remembering

All 8 leaking modules call `install_secret_redaction()` at import. A leak is only
possible if the leaking module is imported, so the filter is guaranteed present on any
path that can leak. This is what makes the fix survive a new script nobody remembers to
patch -- and it is directly asserted, not assumed:
`test_every_swept_site_installs_redaction_at_import`.

### 2c. Gap G3 -- the exception channel

The phase-60.4 filter rewrote only `record.msg`/`record.args`. An
`httpx.HTTPStatusError` repr carries the full request URL and reaches the handler via
`exc_info`/`exc_text` -- so a call that logged nothing sensitive itself would leak the
moment it FAILED, which is exactly when it gets logged at ERROR. Now pre-rendered,
scrubbed and cached on `record.exc_text`.

**I got this wrong on the first attempt and the failure mode is worth recording.** I
tried `type(exc)(cleaned)` to rebuild the exception with a sanitized message. That
raises for any exception whose `__init__` takes required kwargs -- `HTTPStatusError`
needs `request=` and `response=` -- so it hit the filter's fail-open `except` and
silently left the credential intact. A "fix" that cannot fail loudly
(`feedback_operations_that_cannot_fail_loudly`). Caching `exc_text` needs no
constructor and cannot fail that way.

### 2d. Regression in the EXISTING harness

`scripts/harness/secret_leak_regression.py` already existed, exits non-zero on FAIL and
writes a JSON receipt -- so it was extended, not duplicated. Now **11/11 PASS, exit 0**,
with a case that drives the real `httpx` logger for all three providers.

## 3. Files changed

| File | Change |
|---|---|
| `backend/services/observability/log_redaction.py` | `install_secret_redaction()`; library-logger attachment; G3 exc_text scrub |
| `backend/main.py` | delegate to the shared installer (one attach path) |
| 8 leak-site modules | import-time `install_secret_redaction()` |
| `scripts/harness/secret_leak_regression.py` | +1 case (10 -> 11) |
| `backend/tests/test_phase_82_7_credential_logging.py` | NEW, 32 tests |

## 4. Verbatim verification output

```
$ python -m pytest backend/tests/test_phase_82_7_credential_logging.py -q
................................                                         [100%]
32 passed in 2.79s
```

```
$ python scripts/harness/secret_leak_regression.py
{"wrote": ".../handoff/secret_leak_regression.json", "verdict": "PASS", "tests_passed": 11, "tests_total": 11}
exit=0
```

## 5. Mutation matrix -- attribution measured with `pytest -rf`, not labelled

```
                                                     failed   killed
CONTROL                                                   0
MU1 installer stops filtering library loggers             9   installs_redaction_at_import, install_is_idempotent, +
MU2 fx_rates drops its import-time install                1   every_swept_site_installs_redaction_at_import[fx_rates]
MU3 G3 removed (exception channel unscrubbed)             2   a_raised_http_error_does_not_leak_the_key, exception_type_preserved
MU4 redact_secrets is a no-op                            20   (broad)
MU5 installer is no longer idempotent                     1   install_is_idempotent
RESTORED                                                  0
```

Each mutant is `ast.parse`-validated before running; a mutant that does not compile is
reported SYNTAX-BROKEN and not counted as a kill.

**The mutant that matters is MU1.** The research brief warned that a test calling
`redact_secrets()` directly would re-pin the function that already worked while leaving
the REACHABILITY bug -- the actual defect -- untested. MU1 removes reachability while
leaving the function perfect, and 9 tests die. Separately,
`test_removing_the_filter_from_the_logger_makes_the_key_leak` asserts the suite can
still SEE a leak, so the passing tests are not passing vacuously.

**A test-authoring error of mine, recorded because it nearly produced a false negative.**
That reachability test first stripped the filter from the `httpx` logger only, and
still saw no leak -- so it "passed" while proving nothing. Cause: filters rewrite the
`LogRecord` IN PLACE and all handlers share one record object, so a root handler still
carrying the filter scrubbed the record before the capture handler formatted it. The test
now strips the filter from the library loggers AND every root handler.

## 6. Criterion 2 -- the sweep, and where my own grep failed

The audit-class gate ran 16 query shapes over 4 rounds with 3 dry rounds (K=2 required).
**8 leaking sites, not the 7 I found.** The miss:

`backend/services/fx_rates.py:147` -- a FRED call inside the FX-rate fallback. My grep
looked for a credential adjacent to an f-string URL; this passes the key through a
`params={...}` dict, and `requests` still puts it in the URL. **A single query shape
cannot close an unknown-denominator sweep.** That is the entire argument for marking a
step audit-class, demonstrated on my own work.

Also catalogued: 6 already-safe header-auth sites, including
`backend/news/sources/benzinga.py:66-68` -- a sibling module of leaking site #7, so the
header pattern is already an accepted idiom in the same package.

## 7. Criterion 3 disposition -- all 8 fixed, hardening queued

All 8 are FIXED with respect to logging and each is regression-tested by
`test_every_swept_site_is_covered_by_import`. Moving the credential out of the URL
entirely is queued as **82.32**, because the credential still reaches proxy logs,
middleboxes and APM tools that redaction does not govern.

82.32 carries the brief's research-incomplete warning verbatim: BOTH `fred.stlouisfed.org`
and `finnhub.io/docs` returned 403 to WebFetch and outbound curl is sandbox-blocked, so
both header formats are second-hand. Changing live outbound auth on second-hand
information can silently break ingest, so the step's FIRST task is direct vendor
verification, and it stays blocked rather than guessing. Alpha Vantage supports no header
auth at all and is recorded as bounded residual risk.

## 8. Scope honesty

- **`FRED_API_KEY` rotation is operator-owed and is NOT done by this step.** Shipping
  the code fix does not un-disclose the already-leaked value.
- `docs/archive/pyfinagent-app/tools/alphavantage.py:21,81` has the same leak in
  ARCHIVED DEAD CODE. Not fixed (no behaviour), but it must not be used as a copy source.
- The 4 FRED sites and 2 Alpha Vantage sites still put the key in the URL; only the
  logging exposure is closed.
- No provider auth was changed in this step.

---

## 9. CYCLE 2 -- the FAIL was correct, and the security property really was broken

Cycle 1 (task `wqc2ycjcy`) returned **FAIL**. It was right, and the finding was not a
technicality: **a P0 security step that did not hold its security property.**

### B1 -- I attached to PARENT loggers while the emitters are DESCENDANTS

`_EMITTING_LIBRARY_LOGGERS = ("httpx","httpcore","urllib3","requests")`. But the
loggers that actually print the URL are `urllib3.connectionpool` and
`httpcore.http11`. A logger-level filter does not reach descendant records --
**and `log_redaction.py`'s own docstring says exactly that**, at the top of the file I
was editing. I cited the rule and then broke it.

Reproduced by me before fixing anything:

```
httpx                    leaked=False
urllib3                  leaked=False
urllib3.connectionpool   leaked=True     <-- the emitter
httpcore                 leaked=False
httpcore.http11          leaked=True     <-- the emitter
backend.tools.fred_data  leaked=True
```

The Q/A went further and proved it end-to-end with a **real `requests.get` against a
real local socket**, in the ordering that already exists in the checked-in
`scripts/migrations/extend_historical_data.py:39-41` (import a leak site, *then*
`basicConfig`).

**The fix**: hook `logging.Logger.addHandler`. Handlers see every record that
propagates, from any logger at any depth, so a filtered handler covers descendants that
named-logger filters cannot. The only hole was a handler added AFTER install -- exactly
the script order above -- and the hook closes it at zero per-record cost. The
named-logger leg is kept (free, covers parent-named emitters) but is explicitly
documented as NOT load-bearing.

Verified after the fix, including a logger no enumeration could contain, and in a real
subprocess using the losing order:

```
urllib3.connectionpool  leaked=False      httpcore.http11   leaked=False
backend.tools.fred_data leaked=False      a.brand.new.logger leaked=False
subprocess (import leak site -> basicConfig -> log): leaked=False, marker present=True
```

### B2 -- a guard that drove the wrong transport

`test_every_swept_site_is_covered_by_import` issued an **httpx** request for all 8
sites. Three of them use **requests** (`fx_rates`, `fred_releases`,
`finnhub_earnings`). For those it exercised a transport the module never touches, so it
passed regardless of whether the real channel leaked -- and the real channel *did* leak.
Vacuity shape #5: a fixture that cannot represent the failure.

Replaced with `test_every_swept_site_is_covered_on_its_own_transport`, which drives the
descendant loggers of the module's ACTUAL transport, plus a parametrized
descendant-logger test that includes `a.brand.new.logger.nobody.enumerated`.

### B3 -- a guard that was true by construction

`test_install_survives_a_later_basicConfig` called `basicConfig()` while the `capture`
fixture had already added a root handler. `basicConfig` is a documented **no-op** when
root has handlers -- measured before=1, after=1 -- so the unfiltered handler whose
survival it claimed to prove was never created. Vacuity shape #4.

Rewritten to clear root first (asserting the precondition), so `basicConfig` genuinely
creates a handler, and to assert `basicConfig` actually added one before proceeding.

### B4 -- the 9th shape: exception re-emission

Neither my sweep nor the researcher's looked for it. Modules catch a transport
exception and log it; `str(exc)` reconstructs the credential-bearing URL, so
`logger.warning("failed: %s", exc)` re-emits the key through a `backend.*` logger.
Now covered by the handler hook and pinned by
`test_a_backend_module_logger_re_emitting_a_transport_error_does_not_leak`.

### A surviving mutant I found in my own matrix, and closed

MU8 (drop the `_HOOK_INSTALLED` early return) left the suite fully GREEN, because
`test_install_is_idempotent` counts filters on the httpx logger and cannot see
`logging.Logger.addHandler` being re-wrapped. install() is called by nine modules plus
`main.py`, so an unguarded hook nests ten wrappers deep. Closed by
`test_the_addhandler_hook_wraps_exactly_once`, which asserts function IDENTITY is stable
across repeated installs. MU8 now kills it.

### Mutation matrix v2

```
CONTROL                                              0 failed
MU6 addHandler hook removed (THE cycle-1 blocker)   10 failed
MU7 named-logger leg removed                         1 failed
MU8 hook no longer idempotent                        1 failed   (was 0 -- guard added)
MU3 G3 exception channel removed                     2 failed
MU4 redact_secrets is a no-op                       31 failed
RESTORED                                             0 failed
```

### Full-suite regression -- measured both ways, because this is a global monkeypatch

Hooking `logging.Logger.addHandler` is process-wide, so "my step suite is green" is not
adequate evidence. Ran the whole backend suite with the fix, then again with
`install_secret_redaction()` neutered to a no-op, and diffed the failing sets:

```
WITH 82.7    : 31 failed, 2471 passed, 147.54s
WITHOUT 82.7 : 32 failed        (82.7's own suite excluded)
failures INTRODUCED by 82.7: NONE
```

And one failure is *repaired* by it: `test_phase_75_sre_ops.py::test_c6_redaction_
survives_json_branch` was red before and passes now -- a phase-75 redaction test that
was failing precisely because redaction never actually reached.

### Verification

```
$ python -m pytest backend/tests/test_phase_82_7_credential_logging.py -q
45 passed
$ python scripts/harness/secret_leak_regression.py    -> exit 0, 11/11
```

### What I take from this

The cycle-1 fix was plausible and completely wrong on the one property that mattered.
It passed 32 tests. What made it wrong was not a missing test but **two guards that
could not observe the failure** -- one driving a transport three modules never use, one
depending on a no-op. A suite can be large, green, and blind at the same time. The
lesson is the one already in auto-memory as
`feedback_mutation_test_guards_and_fixtures`, and I re-earned it: mutate the SUBJECT,
and check the guard can see the mutation.

---

## 10. CYCLE 2 verdict -- PASS, and what I changed after it

Cycle 2 (task `w9z4bzhee`) returned **PASS**, all 3 criteria met, zero violated. It
verified the repair rather than inheriting it: a fresh subprocess in the losing order
checked in at `extend_historical_data.py:39-41`, issuing a REAL `requests.get` against a
REAL local socket so `urllib3.connectionpool` emits a genuine record --
`LEAKED_CREDENTIAL=False`, `URL_WAS_LOGGED=True`, `REDACTION_MARKER_PRESENT=True`, so
the result is provably non-vacuous. Its own mutant MU-Q1 killed 11 tests. Its own
control re-ran the 31 full-suite failures with 82.7 neutered and found the symmetric
difference EMPTY -- zero regressions from a process-wide monkeypatch -- while
`test_c6_redaction_survives_json_branch` flips fail->pass.

### Post-PASS edit, disclosed

**I changed one docstring after the PASS.** `test_descendant_loggers_are_covered_not_
just_the_named_parents` was titled "THE CYCLE-1 BLOCKER, pinned". That is FALSE: [WARN-1]
measured it surviving MU-Q1, and I reproduced it -- all 10 parameters pass with the
addHandler hook removed. Fixture ordering is why: `capture` attaches its handler to root
*before* the test body calls install, so the plain handler leg (present since
phase-60.4) scrubs the record and the test cannot tell the cycle-1 and cycle-2 designs
apart.

Comment text only, zero behaviour change -- 45 passed before and after. I corrected it
rather than deferring because a false claim inside a security suite misleads the next
reader, and the behavioural fix (attach `capture` AFTER install) is queued as **82.33**
rather than applied, so the shipped tree stays the graded tree.

The blocker IS pinned -- by `test_install_survives_a_later_basicConfig`,
`test_a_handler_added_after_install_is_filtered` and
`test_the_addhandler_hook_wraps_exactly_once`, which account for the 11 kills. None of
them is this test.

**This is the third time in this phase I have written a claim about which guard catches
what, without measuring it.** The code was right each time; the attribution was wrong.
The habit that fixes it is not "write better docstrings" -- it is: run the mutant, read
the killed test NAMES, and only then write the sentence.

### Also queued from the residuals

- **82.33** -- WARN-1 (the vacuous descendant guard), WARN-2 (8 of 11 kills assert a
  module-state flag rather than absence-of-credential), WARN-3 (two bare
  `except Exception: pass` on a SECURITY filter -- the exact mechanism that hid my first
  G3 attempt; must gain a breadcrumb while STAYING fail-open), NOTE-1 (a pre-existing
  F401 two lines from a hunk this step added).
- **82.32** gains NOTE-3: the regex omits bare `key` and `password`. Measured: those two
  do NOT redact; api_key/apikey/api-key/token/X-Api-Key do. A sweep found ZERO current
  call sites using them, so this is pre-emptive, not a live gap.
- **`retry_count` set to 1** [NOTE-5] -- a genuine cycle-1 FAIL occurred, and the F1
  certified-fallback escalation counts that field.

### Operator action still owed, unchanged by this step

**Rotate `FRED_API_KEY`.** This step closes the logging channel. It does not un-disclose
a value that has already been written to a console.
