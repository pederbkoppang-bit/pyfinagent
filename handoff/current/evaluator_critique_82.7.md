# Evaluator critique -- phase-82.7 (P0 SECURITY)

> **THIS FILE NOW HOLDS TWO VERDICTS.** Sections 0-8 + the first fenced json block are
> **CYCLE 1 (FAIL, 13:25)**, preserved verbatim and NOT overwritten. The **CYCLE 2**
> evaluation by a fresh, independent Q/A begins at the heading "## CYCLE 2" near the
> bottom; the LAST fenced json block in this file is the **verdict of record**.
> Cycle 2 is the documented cycle-2 flow, not verdict-shopping: evidence changed between
> spawns (measured mtimes -- critique 13:25 < test file 13:30 < log_redaction.py 13:38 <
> experiment_results 13:39).

Independent Layer-3 Q/A. Author = Main. This evaluator did not write, edit or
run any production code path; Bash used for non-mutating verification only.

**Credential hygiene:** no credential VALUE appears anywhere in this file.
Keys are named by ENV VAR NAME only. The token `SYNTHETIC0000NOTAREALKEY0000000A`
that appears in the suite is a synthetic placeholder, not a credential.

Status: COMPLETE. **Verdict: FAIL** -- the fenced json block at the bottom of this file
is the verdict of record.

One-line summary: the fix closes the `httpx` channel correctly, but two modules in the
declared "fixed set" still write the credential to a log record at the default INFO level,
and the guard that was supposed to cover them cannot observe their failure.

---

## 0. Deterministic gate results (run by this evaluator)

| Check | Command | Result |
|---|---|---|
| Immutable verification command | `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_7_credential_logging.py -q` | **exit=0**, `32 passed in 3.25s` -- REPRODUCES Main's claim |
| Changed-.py scope (DERIVED, not typed) | `git diff --name-only HEAD -- '*.py'` + `git ls-files --others --exclude-standard -- '*.py'` | 12 files, non-empty guard satisfied |
| ruff F821/F401/F811 over that scope | `uvx ruff check --select F821,F401,F811 <12 files>` | **exit=1**, one F401 |

### ruff finding (verbatim)

```
F401 [*] `backend.econ_calendar.registry.CalendarSource` imported but unused
  --> backend/econ_calendar/sources/finnhub_earnings.py:26:44
Found 1 error.
```

**PRE-EXISTENCE PROVEN, not asserted.** I extracted the HEAD blob
(`git show HEAD:backend/econ_calendar/sources/finnhub_earnings.py`) and linted it
in isolation: same F401, `head_exit=1`. `git diff HEAD` on that file shows the
82.7 hunk adds only the redaction import + call at line ~37; it does not touch
line 26. So this is pre-existing, NOT introduced by 82.7. Recorded as a NOTE, not
a blocker.

**Process note on my own gate:** my first ruff invocation reproduced vacuity
shape #9 from `qa.md` 4c -- the shell here is zsh, which does NOT word-split an
unquoted `$FILES`, so ruff received all 12 paths as ONE filename, printed
`All checks passed!` and exited 0. That false pass is exactly the failure the
qa.md empty-set guard was written for. Re-run with explicit splitting produced
the exit=1 above. Anyone re-running this gate on this machine must split
explicitly.

---

## 1. BLOCKER -- criterion 3 is falsified: two modules in the "fixed set" still emit credential-bearing log records at DEFAULT level

`experiment_results_82.7.md:145` claims: *"All 8 are FIXED with respect to logging and
each is regression-tested by `test_every_swept_site_is_covered_by_import`."*

**That claim does not reproduce.** I produced credential-bearing log records from two of
the eight, at `INFO` (the documented default), after their import-time
`install_secret_redaction()` had run, in the real script shape.

### 1a. Root cause -- the logger leg names PARENTS, but the emitters are DESCENDANTS

`log_redaction.py:29` -- `_EMITTING_LIBRARY_LOGGERS = ("httpx","httpcore","urllib3","requests")`.

`httpx` logs through a logger named exactly `httpx`, so it is covered. The other three
do not: `urllib3` emits through `urllib3.connectionpool`, `httpcore` through
`httpcore.http11` / `httpcore.connection`. Those are **descendant** loggers, and the
module's own docstring (`log_redaction.py:6-8`) states the governing rule: *"logger-level
filters do NOT apply to records emitted by descendant loggers"*. The implementation then
attaches to the parents anyway. The stated mechanism and the implementation disagree.

Measured, with the handler leg deliberately isolated (capture handler attached AFTER
`install_secret_redaction()`, so it carries no filter -- verified `h.filters == []`):

```
  httpx                                    leaked=False -> api_key=***REDACTED***
  urllib3.connectionpool                   leaked=True
  httpcore.http11                          leaked=True
  urllib3.util.retry                       leaked=True
  requests.packages.urllib3.connectionpool leaked=True
```

**3 of the 8 leak sites use `requests`, not `httpx`** -- derived, not assumed:

```
backend/services/fx_rates.py:158                    import requests
backend/econ_calendar/sources/fred_releases.py:20   import requests
backend/econ_calendar/sources/finnhub_earnings.py:24 import requests
```

End-to-end with a REAL `requests.get` against a real local socket, after importing the
real leak-site module `backend.econ_calendar.sources.finnhub_earnings`:

```
http://127.0.0.1:54931 "GET /api/v1/calendar/earnings?from=2026-01-01&token=<SYNTHETIC> HTTP/1.1" 200 None
REAL_LEAK_VIA_REQUESTS: True
```

### 1b. The bigger hole -- `backend.*` module loggers re-emit the credential-bearing exception

Every one of the eight logs the caught exception through its OWN `backend.*` logger, and
an httpx/requests exception's `str()` carries the full request URL, key included:

```
backend/tools/fred_data.py:100                     logger.warning("Failed to fetch FRED series %s: %s", series_id, e)
backend/services/fx_rates.py:172                   logger.warning("fx_rates: FRED fetch %s failed: %s", ccy, e)
backend/econ_calendar/sources/fred_releases.py:77  logger.warning("fred releases fetch failed: %r", exc)
backend/econ_calendar/sources/finnhub_earnings.py:97 logger.warning("finnhub earnings fetch failed: %r", exc)
backend/news/sources/finnhub.py:118, backend/tools/alphavantage.py:75,
backend/tools/social_sentiment.py:146
```

`backend.tools.fred_data` is not in `_EMITTING_LIBRARY_LOGGERS`, so these records are
covered by the **handler leg only** -- i.e. by exactly the pre-82.7 design this step
exists to replace, and which `log_redaction.py:110-113` itself declares insufficient
(*"a script may call basicConfig() AFTER this, creating a fresh unfiltered handler, and
nothing would re-attach"*).

Measured end-to-end, at `INFO`, in that precise shape:

```
LEAK_VIA_BACKEND_MODULE_LOGGER:        True   (backend.tools.fred_data,  httpx exc)
LEAK_VIA_REQUESTS_EXCEPTION_AT_INFO:   True   (backend.services.fx_rates, requests exc)
```

Note the asymmetry in the captured output: the `httpx` INFO line was correctly scrubbed
to `api_key=***REDACTED***` **in the same run** in which the `backend.*` WARNING carried
the key in full. The fix works on the one channel it covers and silently misses the
adjacent one.

### 1c. This is LIVE-REACHABLE from a checked-in script, not a constructed scenario

`scripts/migrations/extend_historical_data.py` has the vulnerable ordering already:

```
39: from backend.backtest.data_ingestion import DataIngestionService   <- import-time install; root has 0 handlers, so 0 filtered
41: logging.basicConfig(level=logging.INFO, ...)                       <- fresh UNFILTERED root handler
```

The import-time install attaches to root handlers *that exist at import time*. Here there
are none, so the only protection installed is the library-logger leg -- which does not
cover `backend.*` loggers or the urllib3/httpcore descendants. This is the same
manual/CLI-backfill class the step identifies as the source of the 2026-08-03
operator-visible disclosure.

### 1d. The guard that should have caught this is vacuous (qa.md 4c shape #5)

`test_every_swept_site_is_covered_by_import` issues an **httpx** request for all eight
parametrizations, including the three modules that use `requests`. For those three the
test exercises a channel the module never uses, so it cannot represent their failure. It
passes for `fx_rates`, `fred_releases` and `finnhub_earnings` regardless of whether their
real channel leaks -- and their real channel does leak.

`test_install_survives_a_later_basicConfig` is vacuous for a second, independent reason
(shape #4, true by construction). `logging.basicConfig()` is a documented no-op when the
root logger already has handlers, and the `capture` fixture adds one before the test body
runs. Measured directly:

```
root handlers before=1  after=1  -> basicConfig created a handler? False
```

So the test never creates the unfiltered handler whose survival it claims to prove, and
the install had already filtered the capture handler via the handler leg.

## 2. Attack A -- criterion 1 mutation (MU-A). Suite IS reachability-sensitive; the criterion-1 test alone is NOT

I built my own mutant rather than replaying Main's: `install_secret_redaction` replaced
in-process with a pre-82.7 reconstruction (root HANDLERS only, no `_EMITTING_LIBRARY_LOGGERS`
leg), patched before collection so the eight modules' import-time calls bind to it.

```
=== MU-A: library-logger leg REMOVED (pre-fix reachability) ===
9 failed, 23 passed
FAILED ...::test_every_swept_site_installs_redaction_at_import[data_ingestion|fred_data|
        fx_rates|fred_releases|alphavantage|social_sentiment|finnhub|finnhub_earnings]
FAILED ...::test_install_is_idempotent
```

**The suite dies -- criterion 1 is not vacuous overall.** Main's MU1 row (9 killed, named
tests) reproduces exactly; the attribution in `experiment_results.md:103` is measured and
honest.

But note precisely WHICH tests survived MU-A: `test_a_real_httpx_request_at_INFO_does_not_log_the_key`
(the headline criterion-1 test) and all 8 `test_every_swept_site_is_covered_by_import`
cases **passed under the pre-fix design**. Cause: the `capture` fixture attaches its
handler to root *before* the test body calls `install_secret_redaction()`, so the handler
leg scrubs the record. The criterion-1 test is killed by removing redaction altogether,
so it is a real guard -- but it cannot distinguish pre-fix from post-fix, which is the
distinction this step is about. WARN-level, not the blocker; recorded because the same
fixture ordering is what hides finding 1d.

## 3. Attack B -- the ninth leak site

**Found: `backend/tools/fred_data.py:100`** (and the same shape at
`backend/services/fx_rates.py:172`, `backend/econ_calendar/sources/fred_releases.py:77`,
`backend/econ_calendar/sources/finnhub_earnings.py:97`, `backend/news/sources/finnhub.py:118`,
`backend/tools/alphavantage.py:75`, `backend/tools/social_sentiment.py:146`).

The query shape neither Main nor the researcher used: instead of sweeping for *a credential
interpolated into a URL*, I swept for **re-emission of the transport exception**, whose
`str()`/`repr()` reconstructs the credential-bearing URL:

```
grep -nE "logger\.(debug|info|warning|error|exception)\(.*(url|URL|resp\.|e\)|exc)" <the 8 modules>
```

This is a leak *inside* the fixed set, so criterion 2's sweep is incomplete on its own
terms: the sweep enumerated URL-construction sites but not credential-bearing-log sites,
and criterion 3 grades the latter.

Adjacent, out of criteria but worth queueing: `backend/tools/fred_data.py:101` puts the
same exception string into the **return value** (`results[series_id] = {"error": str(e)}`),
so the credential travels beyond logging into whatever consumes that dict.

## 4. Attack D -- the fail-open `except` in `filter()`

The G3 `exc_text` approach is sound and I could not defeat it. `logging.Formatter.format`
does honour a pre-set `record.exc_text`, `formatException` needs no constructor, and
`test_a_raised_http_error_does_not_leak_the_key` has a non-vacuity guard
(`assert blob, "nothing was logged"`). Main's disclosure of the earlier
`type(exc)(cleaned)` failure (`experiment_results.md:60-66`) is accurate and is a genuine
`feedback_operations_that_cannot_fail_loudly` instance, honestly recorded.

Residual, WARN-level: the bare `except Exception: pass` at `log_redaction.py:75-76` is
still a silent-failure surface for any FUTURE edit to the filter body -- the exact
mechanism that hid the first G3 attempt. It cannot fail loudly by construction. A
counter/`logging.raiseExceptions`-style breadcrumb would make a future regression visible.
Not blocking on its own.

## 4b. Attack C -- import-time side effects: CLEAN

Main did not run the full backend suite; I did.

```
backend/tests/  ->  31 failed, 2458 passed, 12 skipped, 5 xfailed, 1 xpassed  in 148.89s
```

To decide whether ANY of those are CAUSED by 82.7 rather than pre-existing, I re-ran the
failing set with 82.7 neutered in-process (`install_secret_redaction` -> no-op,
`SecretRedactionFilter.filter` -> passthrough):

```
CONTROL: 82.7 neutered, re-running the 29 failures
29 failed, 1 warning in 24.26s
```

**Identical failure set. None of them is caused by 82.7.** No import cycle, no deadlock,
no corruption of another suite's log capture -- notable because the filter mutates
`record.msg` in place, which is exactly the shape that could have broken `caplog`
assertions elsewhere. The import-time install is safe.

Scope bound, disclosed: I piped the first full run through `tail`, so my captured file held
29 of the 31 `FAILED` lines and the control covers those 29, not all 31. One of the two
uncaptured (`test_64_3_currency_path_kr_avg_entry_stays_krw`) I observed separately failing
on a kill-switch-PAUSED precondition, unrelated to logging. The tree also carries another
session's phase-83 work, which is the likelier origin of the pre-existing 31.

## 4c. Claim audit (qa.md 4b) -- what DID reproduce

Reported symmetrically, not only the misses:

| Claim | Re-derivation | Result |
|---|---|---|
| "`logging.basicConfig(` appears 54 times in 54 files" | AST walk over `backend/ scripts/ functions/` counting `Call` nodes with `func.attr == "basicConfig"`, excluding tests | **REPRODUCES EXACTLY**: 54 calls in 54 files. (A naive `grep` gives 57/56 -- inflated by docstring mentions -- so Main's figure is the more careful one. The researcher's "40 files" is the figure that is wrong.) |
| Immutable command "32 passed" | re-ran | REPRODUCES (32 passed, 3.25s vs Main's 2.79s -- run-to-run, and 32 dots over "32 passed" is internally consistent, not spliced) |
| `secret_leak_regression.py` "11/11 PASS, exit 0" | re-ran | REPRODUCES verbatim, `exit=0` |
| MU1 row "9 killed, `installs_redaction_at_import` + `install_is_idempotent`" | my own independent MU-A | REPRODUCES (9 failed; same tests) |
| 3 immutable criteria copied verbatim into the contract | string containment against `.claude/masterplan.json` | TRUE for all 3 |
| "All 8 are FIXED with respect to logging" | end-to-end per module | **DOES NOT REPRODUCE** -- section 1 |

## 5. Attack E -- the research-gate irregularity: ACCEPTABLE, not a blocker

The researcher's structured return failed, so no envelope was archived under `qa_returns/`.
I judge accepting the on-disk brief LEGITIMATE:

- `.claude/rules/research-gate.md` "Write-first discipline" exists precisely so a brief
  survives a failed end-flush; the envelope is specified as *the final section of the
  brief*, which is present and complete (`gate_passed: true`, 10 read in full, 43 URLs,
  33 snippet-only, `recency_scan_performed: true`, `coverage.audit_class: true`,
  `dry_rounds: 3 >= K_required: 2`, 16 query shapes listed).
- The **delivery mechanism** failed, not the gate. Requiring a re-run would penalise the
  transport, and would discard a gate that demonstrably did its job: the brief caught the
  8th site (`fx_rates.py:147`) that Main's own grep missed, and corrected the masterplan's
  stale `:313` line reference to `:371`.
- `contract_82.7.md:25-31,133-134` discloses the failure, its cause, and the substitution
  explicitly, and Main re-verified the headline claims independently.

I would BLOCK on this only if the brief were absent, partial, or self-reporting
`gate_passed: false`. None applies. Filing the transport defect is worthwhile (it is the
second such drop this session, per the contract).

## 6. Harness compliance audit (5-item)

| # | Item | Result |
|---|---|---|
| 1 | Researcher spawned before contract | YES -- `research_brief_82.7.md` (684 lines), envelope `gate_passed: true`; return-transport failure disclosed. See section 5. |
| 2 | Contract written before GENERATE | YES -- `contract_82.7.md` present, criteria copied verbatim and byte-matching `.claude/masterplan.json` |
| 3 | `experiment_results.md` present with verbatim output | YES -- present; the pytest block reproduces (32 passed; timing differs 2.79s vs my 3.25s, which is expected run-to-run and not a splice -- 32 progress dots over "32 passed" is internally consistent) |
| 4 | LOG-LAST respected | YES -- `grep "phase=82.7" handoff/harness_log.md` returns no cycle header; masterplan `82.7 status = pending`. Correct ordering; nothing flipped ahead of this verdict. |
| 5 | No verdict-shopping | YES -- `retry_count = 0`, zero prior `result=CONDITIONAL` entries for 82.7. This is cycle 1; the 3rd-CONDITIONAL rule does not apply. |

Follow-up step **82.32 exists and is `pending`** in the masterplan, so the deferral half
of criterion 3 is satisfied. The failure is on the "a test asserts the fixed set produces
no credential-bearing log records" half.

## 7. Scope-honesty lens

`experiment_results.md` §8 is unusually good: it discloses the owed `FRED_API_KEY`
rotation, the archived dead-code copy, the fact that the key still rides in the URL, and
that no provider auth changed. Main also volunteered two of its own errors (§5 and §2c).
The single overclaim is §7's "All 8 are FIXED with respect to logging" -- and it is the
one that matters, because it is the sentence that discharges criterion 3.

## 8. What would make this PASS

1. Filter at a level that actually covers the emitters -- e.g. attach to the descendant
   loggers actually used (`urllib3.connectionpool`, `httpcore.http11`, ...), or better,
   stop relying on logger identity: install a `logging.setLogRecordFactory` wrapper, or
   attach the filter to `logging.Logger.handle`/`root` via a module-level
   `logging.Logger.manager` sweep so `backend.*` loggers are covered too.
2. Cover the `backend.*` re-emission channel. Note the trap: adding `"backend"` to
   `_EMITTING_LIBRARY_LOGGERS` would NOT work -- `backend.tools.fred_data` is a descendant
   of `backend`, so the same rule that breaks `urllib3` breaks that too. The options that
   actually work are (a) a `setLogRecordFactory` hook, (b) filtering at the handler on
   every entry point (needs a re-attach that survives later `basicConfig`), or (c) scrubbing
   the exception where it is re-emitted.
3. Make `test_every_swept_site_is_covered_by_import` drive **the library each module
   actually imports** (`requests` for 3 of 8), and add a case that logs the caught
   exception through the module's own `backend.*` logger.
4. Re-shape the `capture` fixture so the handler is attached AFTER install (no handler
   filter), which is what makes these tests able to observe the logger-leg behaviour they
   claim to test.

---

## VERDICT

```json
{
  "ok": false,
  "verdict": "FAIL",
  "reason": "P0 security step does not hold its security property. Criterion 3 ('a test asserts the fixed set produces no credential-bearing log records') is falsified: I reproduced credential-bearing log records at default INFO from two modules IN the fixed set (backend/tools/fred_data.py:100 and backend/services/fx_rates.py:172) after their import-time install ran, in the real script shape -- an ordering already present in the checked-in scripts/migrations/extend_historical_data.py:39-41. Root cause: _EMITTING_LIBRARY_LOGGERS names PARENT loggers (urllib3, httpcore) while the emitters are DESCENDANTS (urllib3.connectionpool, httpcore.http11), contradicting the module's own docstring rule, and backend.* module loggers that re-emit the credential-bearing transport exception are covered only by the pre-82.7 handler leg. The covering guard is vacuous: test_every_swept_site_is_covered_by_import drives httpx for all 8 sites, but 3 of them use requests, so it cannot represent their failure; test_install_survives_a_later_basicConfig is a no-op because basicConfig does nothing when the capture fixture has already added a root handler (measured before=1 after=1). Criterion 1 IS met and its suite is reachability-sensitive (my MU-A killed 9 tests). Criterion 2's sweep is met for URL-construction sites but missed the exception-re-emission shape. Immutable command reproduces (exit=0, 32 passed) -- it is simply not sufficient to detect this. Research-gate irregularity judged ACCEPTABLE. Harness compliance clean (LOG-LAST respected, status still pending, no verdict-shopping, retry_count=0).",
  "violated_criteria": [
    "every hit found by that sweep is either fixed or has its own queued follow-up step; a test asserts the fixed set produces no credential-bearing log records"
  ],
  "violation_details": [
    {
      "violation_type": "Contradiction",
      "action": "install_secret_redaction() attaches SecretRedactionFilter to _EMITTING_LIBRARY_LOGGERS = ('httpx','httpcore','urllib3','requests'), then a real requests.get / a backend.* logger.warning re-emits the credential",
      "state": "Measured at INFO after import-time install: urllib3.connectionpool leaked=True, httpcore.http11 leaked=True, REAL_LEAK_VIA_REQUESTS=True (finnhub_earnings + real socket), LEAK_VIA_BACKEND_MODULE_LOGGER=True (fred_data.py:100), LEAK_VIA_REQUESTS_EXCEPTION_AT_INFO=True (fx_rates.py:172). httpx was correctly redacted in the same run.",
      "constraint": "criterion 3: a test asserts the fixed set produces no credential-bearing log records. log_redaction.py:6-8 states logger-level filters do NOT apply to descendant loggers, yet the implementation attaches to parents and relies on descendants inheriting."
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "test_every_swept_site_is_covered_by_import issues an httpx request for all 8 parametrized leak sites",
      "state": "3 of the 8 use requests, not httpx (fx_rates.py:158, fred_releases.py:20, finnhub_earnings.py:24). For those the test exercises a transport the module never uses, so it passes independently of whether the module's real channel leaks -- and the real channel does leak. qa.md 4c vacuity shape #5.",
      "constraint": "qa.md 4c: a guard that cannot fail when its subject is broken does not count; sole-coverage vacuity on a behavioural criterion is BLOCKING."
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "test_install_survives_a_later_basicConfig calls logging.basicConfig(level=INFO) to create 'a new root handler, unfiltered'",
      "state": "Measured: root handlers before=1, after=1 -- basicConfig is a documented no-op when the root logger already has handlers, and the capture fixture adds one first. The unfiltered handler whose survival the test claims to prove is never created. qa.md 4c vacuity shape #4.",
      "constraint": "qa.md 4c: name the concrete mutation that makes the guard fail; a guard true by construction is a finding, never a pass."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "experiment_results_82.7.md:145 asserts 'All 8 are FIXED with respect to logging and each is regression-tested by test_every_swept_site_is_covered_by_import'",
      "state": "Does not reproduce. Two of the eight emit the credential at default INFO after their import-time install; the cited regression test cannot observe the failure for 3 of the 8.",
      "constraint": "qa.md 4b: every set-membership claim must be re-derivable by the evaluator; prefer FAIL when a claim in a verbatim artifact does not reproduce."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "immutable_verification_command",
    "ruff_F821_F401_F811_on_derived_scope",
    "pre_existence_proof_of_lint_finding",
    "mutation_MU_A_prefix_reachability",
    "logger_leg_isolation_probe",
    "end_to_end_real_requests_leak",
    "end_to_end_backend_logger_exception_leak",
    "basicConfig_noop_vacuity_measurement",
    "ninth_leak_site_sweep",
    "secret_leak_regression_harness",
    "full_backend_test_suite",
    "full_backend_suite_control_run",
    "harness_compliance_audit",
    "research_gate_envelope_review",
    "git_state_measurement"
  ]
}
```

---
---

# CYCLE 2 -- fresh independent Q/A (verdict of record below)

Fresh Q/A instance. I did not author, edit or run any production code path. Bash used for
non-mutating verification only; the ONLY file I wrote is this one, appended (cycle 1 above
is preserved byte-for-byte).

**Credential hygiene:** no credential VALUE appears anywhere in this file. Keys are named
by ENV VAR NAME only. `SYNTHETIC0000NOTAREALKEY0000000A` is a synthetic placeholder, not a
credential. I did not read `backend/.env` and did not grep any log for a key value. **No
real credential value was found in any artifact of this step.**

**Status: COMPLETE. Verdict: PASS** (with 4 WARN / 2 NOTE findings, none criterion-violating).

## C2.0 -- This is cycle 2, and it is the documented cycle-2 flow, NOT verdict-shopping

My spawn prompt described cycle-1 state ("32 passed", a fix with no `addHandler` hook) and
told me to *create* this file. Both were stale. Measured mtimes:

```
13:25:43  handoff/current/evaluator_critique_82.7.md   <- cycle-1 FAIL
13:30:20  backend/tests/test_phase_82_7_credential_logging.py
13:38:21  backend/services/observability/log_redaction.py
13:39:15  handoff/current/experiment_results_82.7.md
```

Evidence changed AFTER the FAIL, in the correct order (fix, then results). That is the
CLAUDE.md canonical cycle-2 flow. The distinguishing test for verdict-shopping -- "did the
files change between spawns?" -- is satisfied. `retry_count=0`, zero prior CONDITIONALs, so
the 3rd-CONDITIONAL rule does not apply. I therefore evaluated the CURRENT tree, not the
prompt's description of it, and appended rather than overwrote so the cycle-1 FAIL survives.

## C2.1 -- Deterministic gates (all run by me)

| Check | Result |
|---|---|
| Immutable command `pytest backend/tests/test_phase_82_7_credential_logging.py -q` | **exit=0, 45 passed** (bare run, exit code read directly, not through a pipe) |
| Changed-`.py` scope, DERIVED (`git diff --name-only HEAD` + `git ls-files --others`) | 12 files, non-empty guard satisfied |
| `uvx ruff check --select F821,F401,F811` over those 12 | **exit=1**, one F401 -- see C2.6 NOTE-1 |
| `scripts/harness/secret_leak_regression.py` | **exit=0** |
| Full backend suite | **31 failed, 2471 passed, 12 skipped, 5 xfailed, 1 xpassed in 144.44s** |
| Full-suite CONTROL (82.7 neutered) | **32 failed** -- see C2.4 |

Note on my own gate: my first bypass sweep used unquoted `--include=*.py`, which zsh
glob-expanded, and grep returned "no matches" for every shape -- a false CLEAN. That is
vacuity shape #9 reproducing on the evaluator. Every sweep below was re-run with
`--include='*.py'` quoted.

## C2.2 -- Attack A: is criterion 1 / the cycle-1 blocker vacuous? NO (my own mutant)

I built my own mutant rather than replaying Main's MU6. `_install_addhandler_hook` neutered
to a no-op in-process **before collection**, so the nine import-time `install()` calls bind
to it. That reconstructs the cycle-1 design exactly: handler leg + named-logger leg, no hook.

```
=== MU-Q1: addHandler hook neutered (== cycle-1 design) ===
11 failed, 34 passed
FAILED ...::test_the_addhandler_hook_wraps_exactly_once
FAILED ...::test_install_survives_a_later_basicConfig
FAILED ...::test_a_handler_added_after_install_is_filtered
FAILED ...::test_every_swept_site_installs_redaction_at_import[x8]
```

**The suite dies on the cycle-1 design. Criterion 1 and the criterion-3 breadth guard are
reachability-sensitive and not vacuous.** Main's MU6 row claims 10; I measure 11. Not a
contradiction -- different mutant shapes (mine also stops `_HOOK_INSTALLED` ever flipping) --
but the row should name the mutant precisely. Recorded as NOTE-2.

**WARN-1 (qa.md 4c shape #11, mis-attributed kill mechanism).**
`test_descendant_loggers_are_covered_not_just_the_named_parents` carries the docstring
*"THE CYCLE-1 BLOCKER, pinned."* **It SURVIVED MU-Q1** -- it is absent from the failure list
above. Cause: the `capture` fixture attaches its handler to root *before* the test body calls
`install_secret_redaction()`, so the plain HANDLER leg -- which existed in cycle 1 and in
phase-60.4 -- scrubs the record. The test cannot distinguish the cycle-1 design from the
cycle-2 design, which is precisely the distinction its docstring claims to pin. The blocker
IS genuinely pinned, by `test_install_survives_a_later_basicConfig` and
`test_a_handler_added_after_install_is_filtered`. So this is a vacuous guard *alongside* two
genuine behavioural ones -- WARN, not blocking, per qa.md 4c verdict wiring.
*Named fix:* attach the capture handler AFTER install (or assert `h.filters == []`) so the
test observes the logger/hook leg it names.

**WARN-2 (proxy assertion).** 8 of my 11 kills come from
`test_every_swept_site_installs_redaction_at_import`, which asserts the module-state flag
`lr._HOOK_INSTALLED` -- not an absence-of-credential behaviour. It is a reachability proxy.
Adequate in combination, but the per-site breadth of criterion 3 leans on a state flag.

## C2.3 -- Attack B: coverage-by-construction, and the ninth site

**Independent sweep, two query shapes neither Main nor the researcher used:**

```
shape A: credential-named var interpolated into a URL f-string, no literal "api_key="
  grep -rnE 'https?://[^"'"'"']*\{[a-zA-Z_.]*(key|token|secret|apikey|passwd|password)[a-zA-Z_.]*\}'
  -> 1 hit: backend/tools/alphavantage.py:57  (already IN the fixed 8)

shape B: credential carried in a params= dict (urlencoded by the lib, still printed in the URL)
  grep -rnE '"(api_key|apikey|token|access_token|auth_token|key|apiKey)"\s*:'
  -> URL-bearing hits: fred_releases.py:66, finnhub_earnings.py:66, finnhub.py:84,
     fx_rates.py:160  (all 4 already IN the fixed 8)
  -> llm_client.py:1184,2168 are SDK constructor kwargs, not URL query strings -- out of
     criterion-2 scope
```

**No ninth URL-query-string leak site found.** Criterion 2's sweep reproduces as complete on
its stated population.

**Hook-bypass hunt** (can a credential-bearing request happen without a filtered handler?):

```
direct handler-list mutation:  scripts/slack_response_agent.py:21, scripts/harness/run_autonomous_loop.py:34
                               -> both are basicConfig(handlers=[...]); CPython basicConfig routes
                                  through root.addHandler(h), so the hook DOES cover them
propagate=False:               only backend/tests/test_phase_60_4_observability.py:247 (test-only)
dictConfig / fileConfig:       none in backend/ scripts/ functions/
```

**NOTE-3 -- a measured latent regex gap, not a live leak.** `_SECRET_PARAM_RE`'s alternation
omits bare `key` and `password`:

```
?key=<32ch>       redacted=False
?password=<32ch>  redacted=False
?api_key=<32ch>   redacted=True     ?apikey= True   ?token= True   ?X-Api-Key= True
```

I then swept production for `[?&](key|apiKey|password|passwd|pwd|sig|signature)=` across
`backend/ scripts/ functions/` -- **zero hits**. So there is no current exposure; this is a
hardening gap worth queueing onto 82.32, not a criterion-3 violation.

## C2.4 -- Attack C: import-time side effects of a PROCESS-WIDE monkeypatch

This is materially riskier than cycle 1: the load-bearing leg now rebinds
`logging.Logger.addHandler` for the whole interpreter, and the filter mutates `record.msg`
in place, which is exactly the shape that could corrupt another suite's log capture.

I compared failure **SETS**, not counts (qa.md 4b: equal cardinality is not equal membership).

```
WITH 82.7   : 31 failed, 2471 passed        (reproduces Main's row exactly)
CONTROL     : install_secret_redaction -> no-op, _install_addhandler_hook -> no-op,
              SecretRedactionFilter.filter -> passthrough; re-ran the exact 31 node-ids
           -> 32 failed
```

The control's 32 = the same 31 (**every one still fails with 82.7 neutered -> symmetric
difference empty -> ZERO failures introduced by 82.7**) PLUS
`test_phase_75_sre_ops.py::test_c6_redaction_survives_json_branch`, which **passes with 82.7
and fails without it**. So Main's claim that 82.7 *repairs* a previously-red phase-75
redaction test reproduces under my own control. No cycle, no deadlock, no caplog corruption.

## C2.5 -- Attacks D and E

**D -- the fail-open `except`.** The `exc_text` approach is sound: `Formatter.format` honours
a pre-set `record.exc_text`, `formatException` needs no constructor, and
`test_exception_type_is_preserved_when_its_message_is_scrubbed` pins that the exception TYPE
survives scrubbing. I could not defeat it. **WARN-3 (carried from cycle 1, unchanged):**
`log_redaction.py:75-76` and the hook's `:159-161` are both bare `except Exception: pass` on
a *security* filter -- by construction they cannot fail loudly, and that is the exact
mechanism that hid the first `type(exc)(cleaned)` attempt. A counter or a
`logging.raiseExceptions`-style breadcrumb would make a future regression visible.

**E -- the research-gate irregularity.** I independently reach cycle 1's conclusion:
ACCEPTABLE. `.claude/rules/research-gate.md` "Write-first discipline" exists precisely so a
brief survives a failed end-flush, and specifies the envelope as the brief's final section --
which is present and complete (`gate_passed: true`, 10 read in full, 43 URLs,
`coverage.dry: true` after 3 dry rounds, `audit_class: true`). The **delivery mechanism**
failed, not the gate; the gate demonstrably did its job (it caught the 8th site
`fx_rates.py` that Main's own grep missed and corrected a stale line reference). Main
disclosed all of it in `contract_82.7.md` and re-verified the headline claims independently.
I would block only if the brief were absent, partial, or self-reporting `gate_passed: false`.
None applies. **NOTE-4:** this is now the second structured-return drop this session; the
transport defect is worth its own queued step.

## C2.6 -- Claim audit (qa.md 4b): everything re-derived by me

| Claim in `experiment_results_82.7.md` §9 | My re-derivation | Result |
|---|---|---|
| `45 passed` | re-ran bare | REPRODUCES (exit=0) |
| `WITH 82.7: 31 failed, 2471 passed` | full suite | **REPRODUCES exactly** |
| `failures INTRODUCED by 82.7: NONE` | control on the 31 node-ids | **REPRODUCES by SET**, not just count |
| `test_c6 ... passes now` | ran it neutered | **REPRODUCES** (fails without, passes with) |
| `secret_leak_regression -> exit 0, 11/11` | re-ran | REPRODUCES, exit=0 |
| MU6 `10 failed` | my own MU-Q1 | direction confirmed; **11**, not 10 -- see NOTE-2 |
| descendant/backend-logger leak now closed | **my own end-to-end, below** | REPRODUCES |
| 3 criteria copied verbatim into the contract | string containment vs `.claude/masterplan.json` | TRUE for all 3 |

**My own end-to-end proof that the cycle-1 blocker is really closed.** Not Main's test -- a
fresh subprocess in the losing order that is checked in at
`scripts/migrations/extend_historical_data.py:39-41` (import a leak site while root has no
handlers, THEN `basicConfig`), issuing a **real `requests.get` against a real local socket**
so `urllib3.connectionpool` emits a genuine record, plus the `backend.tools.fred_data`
exception re-emission channel:

```
URL_WAS_LOGGED(non-vacuous): True
REDACTION_MARKER_PRESENT   : True
LEAKED_CREDENTIAL          : False
```

Both channels cycle 1 proved leaking are now scrubbed, and the run is provably non-vacuous
(the URL WAS logged and a redaction marker IS present).

**NOTE-1 -- the ruff exit=1.** One F401,
`backend/econ_calendar/sources/finnhub_earnings.py:26 CalendarSource imported but unused`.
I proved pre-existence myself rather than inheriting cycle 1's proof: extracted the HEAD blob
with `git show HEAD:...` and linted it in isolation -> same finding, `Found 1 error`. The
82.7 hunk on that file adds only the redaction import + call near line 28 and does not touch
line 26. Gate scope is what the CHANGE defines, so this is a NOTE, not a blocker -- but it
sits two lines from a hunk this step added and is a one-token cleanup.

## C2.7 -- Harness compliance (5-item, run first)

| # | Item | Result |
|---|---|---|
| 1 | Researcher spawned before contract | YES -- `research_brief_82.7.md` 12:54 < `contract_82.7.md` 12:56; envelope `gate_passed: true`; transport failure disclosed (see E) |
| 2 | Contract written before GENERATE | YES -- contract 12:56 < first code edit 12:56:38/12:57:53; all 3 criteria verbatim |
| 3 | `experiment_results` present w/ verbatim output | YES -- and every headline number reproduces (C2.6) |
| 4 | LOG-LAST respected | YES -- `grep -c "phase=82.7" handoff/harness_log.md` = **0**; masterplan `82.7 status = pending`. Nothing flipped ahead of this verdict |
| 5 | No verdict-shopping | YES -- evidence changed between spawns (C2.0); 0 prior CONDITIONALs |

**NOTE-5:** `retry_count` is still `0` despite a genuine cycle-1 FAIL. The F1 discipline uses
that counter for `certified_fallback` escalation, so leaving it at 0 understates consecutive
fails. Bookkeeping only; does not affect this verdict.

Criterion 3's deferral half is satisfied: **82.32 exists and is `pending`** (P2, "move the
credential OUT of the URL where the provider supports a header").

## C2.8 -- Criterion-by-criterion mapping (contract completeness)

| # | Criterion | Covering evidence | Verdict |
|---|---|---|---|
| 1 | FRED fetch w/ httpx at INFO produces no record containing the key | `test_a_real_httpx_request_at_INFO_does_not_log_the_key` (real httpx MockTransport, real httpx logger) + anti-vacuity `test_the_url_is_actually_logged...` + `test_removing_the_filter...makes_the_key_leak`; killed by my MU-Q1 | **MET** |
| 2 | Repo swept for credential-in-URL-query calls, result recorded with file:line for every hit | 8 sites with file:line in `experiment_results_82.7.md`; my independent 2-shape sweep found no ninth | **MET** |
| 3 | Every hit fixed or has a queued follow-up; a test asserts the fixed set produces no credential-bearing log records | All 8 fixed by import-time install; 82.32 queued for the defence-in-depth half; `test_every_swept_site_is_covered_on_its_own_transport` (per-module ACTUAL transport) + `test_a_backend_module_logger_re_emitting_a_transport_error_does_not_leak`; independently confirmed by my own real-socket end-to-end | **MET** |

## C2.9 -- Worst-of-N-lenses (P0 money/security path)

- **correctness lens** -- PASS. The security property holds under my own independent
  reproduction, on both channels cycle 1 proved leaking.
- **does-it-reproduce lens** -- PASS. Every headline number reproduces; the one divergence
  (MU6 10 vs my 11) is a mutant-shape difference, disclosed.
- **scope-honesty lens** -- PASS, and unusually strong. §9 opens with "the FAIL was correct",
  names its own two vacuous guards by shape number, and volunteers a surviving mutant (MU8)
  that Main found in its own matrix and closed. §8 still discloses the owed `FRED_API_KEY`
  rotation, the archived dead-code copy, and that the key still rides in the URL.

`min(lens verdicts) = PASS`.

## C2.10 -- Standing operator action, unchanged

**`FRED_API_KEY` must still be rotated.** This step closes the logging channel; it does not
un-disclose a value that already reached an operator-visible console on 2026-08-03.

---

## VERDICT (CYCLE 2 -- verdict of record)

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria MET, verified independently rather than inherited. The cycle-1 FAIL is genuinely repaired: I reproduced the exact leak scenario myself -- a fresh subprocess in the losing order checked in at scripts/migrations/extend_historical_data.py:39-41, issuing a REAL requests.get against a REAL local socket so urllib3.connectionpool emits a genuine record, plus the backend.tools.fred_data exception re-emission channel -- and measured LEAKED_CREDENTIAL=False with URL_WAS_LOGGED=True and REDACTION_MARKER_PRESENT=True, so the result is provably non-vacuous. My own mutant MU-Q1 (neuter _install_addhandler_hook before collection, reconstructing the cycle-1 design) kills 11 tests, so the suite is sensitive to the cycle-1 blocker. Deterministic gates: immutable command exit=0 / 45 passed; secret_leak_regression exit=0; full backend suite 31 failed / 2471 passed, and a control with 82.7 neutered re-running those exact 31 node-ids leaves all 31 still failing (symmetric difference empty -> ZERO regressions introduced by a process-wide addHandler monkeypatch) while test_phase_75_sre_ops::test_c6_redaction_survives_json_branch flips fail->pass, confirming 82.7 repairs a previously-red redaction test. My independent 9th-site sweep used two query shapes neither Main nor the researcher used and found NO ninth URL-query-string leak site; hook-bypass hunt found no production propagate=False, no dictConfig, and both basicConfig(handlers=[...]) sites route through root.addHandler so the hook covers them. Harness compliance clean: LOG-LAST respected (0 'phase=82.7' entries in harness_log.md, masterplan status still pending), criteria verbatim, researcher before contract, and cycle 2 is the documented changed-evidence flow (measured mtimes: critique 13:25 < tests 13:30 < code 13:38 < results 13:39), not verdict-shopping. Research-gate envelope irregularity judged ACCEPTABLE on independent review. FOUR WARN / FIVE NOTE findings recorded, none criterion-violating: WARN-1 test_descendant_loggers_are_covered_not_just_the_named_parents claims in its docstring to pin THE CYCLE-1 BLOCKER but SURVIVED my MU-Q1 (the capture fixture attaches its handler before install, so the pre-existing handler leg scrubs the record) -- vacuity shape #11 alongside two genuine behavioural guards, WARN not blocking per qa.md 4c; WARN-2 eight of my eleven kills rest on the state flag _HOOK_INSTALLED rather than an absence-of-credential behaviour; WARN-3 two bare 'except Exception: pass' on a security filter cannot fail loudly; NOTE-1 ruff exit=1 on one F401 at finnhub_earnings.py:26, pre-existence proven by me on the HEAD blob and untouched by this diff; NOTE-2 Main's MU6 row says 10 killed where my independent mutant kills 11 (different mutant shape, should be named precisely); NOTE-3 measured latent regex gap -- bare 'key=' and 'password=' are not in _SECRET_PARAM_RE, but a production sweep for those param names returned zero hits so there is no current exposure (queue onto 82.32); NOTE-5 retry_count still 0 after a real FAIL. No real credential VALUE appears in any artifact. OPERATOR ACTION STILL OWED: rotate FRED_API_KEY -- this step closes the logging channel, it does not un-disclose the already-leaked value.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "ruff_F821_F401_F811_on_derived_scope",
    "independent_pre_existence_proof_of_lint_finding",
    "mutation_MU_Q1_own_mutant_cycle1_design_restored",
    "end_to_end_subprocess_real_socket_real_requests_losing_order",
    "backend_module_logger_exception_reemission_probe",
    "independent_ninth_leak_site_sweep_two_novel_shapes",
    "addhandler_hook_bypass_hunt",
    "regex_param_coverage_probe",
    "secret_leak_regression_harness",
    "full_backend_test_suite",
    "full_suite_control_run_symmetric_difference",
    "claim_audit_re_derivation",
    "criteria_verbatim_containment_check",
    "log_last_ordering_check",
    "cycle2_changed_evidence_mtime_check",
    "research_gate_envelope_review",
    "git_state_measurement",
    "worst_of_n_lenses"
  ]
}
```

