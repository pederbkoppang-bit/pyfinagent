# live_check -- step 86.120

Hands-on, end-to-end demonstration against the REAL `claude_code_client`
module (not a scratchpad copy) -- `subprocess.run` is mocked (no real
`claude` CLI invocation) and the cooldown file is redirected to a throwaway
tmp path, so this never touches the real
`backend/agents/_cache/cc_rail_cooldown.json`. Every claim below is the
verbatim captured stdout of one script run, immediately followed by the
pytest suite and the two literal source-file mutation passes.

## 1. Cycle 1 -- a weekly-limit failure hits the rail, cooldown engages

```
==============================================================================
STEP 1 -- a weekly-limit failure hits the rail (cycle 1)
==============================================================================
  raised (expected): claude CLI quota exhausted (kind=weekly): You've hit your weekly limit - resets Mon 12:00am
  cooldown_blocked attribute: True
  subprocess.run call count this step: 1 (real spawn -- this is the ONE real failure)
  rail_guard_status(): cooldown_active=True kind=weekly retry_at=2026-08-23T22:00:00+00:00
  breaker_tripped=True consecutive_failures=0 (tripped at N=1, not N=20 -- Azure accelerated circuit breaking)
```

`consecutive_failures=0` while `breaker_tripped=True` is the Azure
"accelerated circuit breaking" behaviour by design: the classified hit trips
the breaker directly (`_rail_guard_open_for_quota`), bypassing the generic
20-consecutive-failure counter entirely -- and, per `test_a_classified_hit_does_not_page`,
without paging.

## 2. Cycle 2 -- rail_guard_reset() fires; disk cooldown survives the in-memory wipe

```
==============================================================================
STEP 2 -- cycle 2: rail_guard_reset() fires (as autonomous_loop.py does at
every cycle start) -- cooldown must SURVIVE the in-memory wipe
==============================================================================
  after rail_guard_reset(): breaker_tripped=False (in-memory breaker correctly wiped)
  cooldown_active=True (disk cooldown correctly SURVIVED)
  raised (expected, cooldown still active): cc_rail cooldown active (kind=weekly, retry_at=2026-08-23T22:00:00+00:00): You've hit your weekly limit - resets Mon 12:00am
  subprocess.run call count this cycle: 0 (ZERO -- no spawn)
```

This is the load-bearing behaviour the whole step exists for: the
phase-66.1 in-memory breaker resets every cycle (by design, for ordinary
transient failures), but the disk cooldown does not -- so a known quota
exhaustion stops re-attempting from cycle 2 onward.

## 3. Simulated backend restart -- fresh disk read, no in-process reliance

```
==============================================================================
STEP 3 -- simulated backend RESTART: fresh read, no reliance on any
in-process object (this is literally what 'restart' means for a module
with no cache beyond the JSON file)
==============================================================================
  raw file on disk: {'kind': 'weekly', 'raw_message': "You've hit your weekly limit - resets Mon 12:00am", 'hit_at': '2026-08-18T07:52:22.685161+00:00', 'retry_at': '2026-08-23T22:00:00+00:00'}
  raised (expected, cooldown still active post-restart): cc_rail cooldown active (kind=weekly, retry_at=2026-08-23T22:00:00+00:00): You've hit your weekly limit - resets Mon 12:00am
  subprocess.run call count this cycle: 0 (ZERO -- no spawn)
```

The raw JSON file content is printed directly (not re-derived via any
Python object retained from Step 1/2) -- this IS what "survives a restart"
means for a module with no process-lifetime cache beyond the file itself.

## 4. The backoff window passes -- cooldown self-clears

```
==============================================================================
STEP 4 -- the backoff window passes; cooldown clears on its own
==============================================================================
  cooldown_active() now that retry_at is in the past: False
  subprocess.run call count: 1 (the rail tries again on its own -- SUCCEEDS)
  envelope result: 'BUY'
  cooldown_active() after a real success: False (self-cleared)
```

Not a permanent lockout: once `retry_at` passes, the very next call
attempts the subprocess again on its own, and a real success clears the
persisted state.

## 5. REGRESSION -- make_client()'s $0-metered routing-breach guard

```
==============================================================================
STEP 5 -- REGRESSION: make_client()'s $0-metered routing-breach guard
(this step touches adjacent failure-handling code; must still fire loud)
==============================================================================
  raised (expected): Routing breach: paper_use_claude_code_route=True but make_client is about to construct a d...
```

Confirms this step's adjacent changes to `claude_code_client.py` did not
weaken the existing guard that prevents a silent fallback to metered
`api.anthropic.com` billing.

## 6. Full immutable verification command (real pytest run, unpiped exit)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
...........................                                              [100%]
27 passed, 1 warning in 2.16s
RAW_EXIT=0
```

## 7. Mutation testing -- literal source-file patches on the REAL file

Both mutations ran against `backend/agents/claude_code_client.py` directly
(not a copy), each bracketed by a green control run and a `diff`-verified
byte-identical restore.

### 7a. Remove the pre-subprocess cooldown guard

Control (before mutation): 27/27 passed (section 6 above).

```
$ python3 - <<'EOF'   # replaced the top-of-function cooldown check with a no-op comment
...
mutated: guard removed
EOF
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
5 failed, 22 passed, 1 warning in 2.10s

FAILED test_claude_code_invoke_skips_subprocess_when_cooldown_active
FAILED test_ticket_queue_processor_direct_call_pattern_also_blocked
FAILED test_generate_content_skips_subprocess_across_two_cycles_and_a_restart
FAILED test_cooldown_block_does_not_inflate_the_generic_consecutive_failure_counter
FAILED test_an_always_on_signal_agent_respects_cooldown_via_shared_entry_point
```

Exactly the 5 tests that depend on the pre-subprocess guard died; the other
22 (classification, persistence, settings, the OTHER mutation's control,
etc.) were correctly unaffected -- proving the mutation is isolated to the
guard, not a broad breakage.

Restore:
```
$ cp /tmp/claude_code_client_fixed_backup.py backend/agents/claude_code_client.py
$ diff /tmp/claude_code_client_fixed_backup.py backend/agents/claude_code_client.py
(no output -- byte-identical)
$ sha256sum backend/agents/claude_code_client.py
76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
27 passed, 1 warning in 2.07s
```

### 7b. Remove the classifier (neutered to always `return None`)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
18 failed, 9 passed, 1 warning in 2.17s
```

18 of 27 died -- everything downstream of classification (all classifier
unit tests, all persistence tests that start from a real classification, all
skip-guard tests, the immediate-trip test, the always-on-signal-agent test,
and the OTHER mutation's own control assertion
`test_MUTATION_removing_the_pre_subprocess_guard_lets_subprocess_run_again`,
which begins by asserting a real cooldown engaged).

**CORRECTED by the Q/A cycle-1 critique (`evaluator_critique_86.120.md`,
Overgeneralization finding)** -- the original text here claimed the 9
survivors were "exactly the tests that never call the classifier ... and
three more that write cooldown state directly rather than deriving it from a
classification." That claim does not reproduce and is removed. Measured: two
named survivors do call the classifier and survive for reasons that have
nothing to do with "never calling it" --
`test_classify_returns_none_for_generic_failure` calls
`classify_limit_failure` three times and survives VACUOUSLY (it asserts the
result is `None`, and the neutered mutant also always returns `None` --
control and mutant give the identical answer, so this survivor proves
nothing either way), and `test_a_classified_hit_does_not_page` drives
`claude_code_invoke` with a real limit-shaped payload but survives because
paging cannot fire either way (a classified hit trips the breaker at N=1,
well under `claude_rail_breaker_threshold`'s default of 20, so the assertion
`paged == []` holds regardless of whether classification ran). No survivor
"writes cooldown state directly" -- that phrase described a pattern that
does not exist in this suite. This function-level mutation (replacing the
whole `classify_limit_failure` body) is superseded as the criterion-5
evidence by the call-site-level mutation in section 10 below, which is the
correct level for testing production wiring rather than the classifier's
own unit-test setup.

Restore, byte-identical (same procedure and same sha256 as 7a), then a final
clean re-run:
```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
27 passed, 1 warning in 2.20s
```

## 8. Wider regression sweep, DERIVED scope (Q/A cycle-1 correction: the
   original "16 files" was typed by hand and undercounted --
   `test_phase_86_110_heartbeat_isolation.py` also matches and was missed;
   re-derived via `grep -rln "claude_code_client\|claude_code_invoke\|rail_guard"
   backend/tests/*.py | grep -v conftest`, 17 files -- this step's own suite
   is already one of the 17, matching on its own imports, not counted twice
   per Q/A cycle-2's correction -- plus the session's earlier charts.py NaN
   fix (does not match the grep pattern), 18 files total)

```
366 passed, 1 failed in 47.43s
```
(Q/A independently reproduced the same derived scope against the pre-fix
27-test suite as 363 passed/1 failed; 366 = 363 + the 3 new tests added in
this cycle-1 fix pass.)

The 1 failure (`test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off`)
is pre-existing and unrelated -- see experiment_results_86.120.md for the
disclosure (`backend/.env` already sets `paper_data_integrity_enabled=true`
in this environment; this step's diff never touches that field).

## 9. Lint gate (project convention: F821/F401/F811), DERIVED scope

Q/A cycle-1 correction: the original 3-file list here was typed by hand.
Re-derived per this project's established convention (union of `git diff
HEAD` and untracked `*.py` files, xargs-fed to avoid zsh's no-word-split
trap on unquoted variables):

```
$ { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u | xargs ruff check --select F821,F401,F811
All checks passed!
```

(5 files: backend/agents/claude_code_client.py, backend/api/charts.py,
backend/config/settings.py, backend/tests/test_charts_nan_serialisation.py,
backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -- the
derived scope is broader than 86.120's own files because it also picks up
this session's earlier charts.py NaN fix, which is correct: the scope
should reflect everything currently dirty in the tree, not just this step's
files.)

## 10. Q/A cycle-1 fix -- three new wiring-level guard tests, mutation-verified

Q/A's independent in-memory mutation matrix (`evaluator_critique_86.120.md`,
Cycle 1) found three new guards that were touched by the shipped code but
never actually killed by any test: M10 (the production call
`cooldown_record_hit(_limit)`), M3 (the production call
`cooldown_clear_on_success()`), and M6 (the tz-fallback fix). Three new
tests were added specifically to close these
(`test_a_real_classified_failure_through_generate_content_actually_persists_the_cooldown`,
`test_a_real_success_after_cooldown_removes_the_persisted_state_file`,
`test_bare_reset_time_with_no_explicit_timezone_uses_host_local_zone_not_utc`),
and each was independently mutation-tested against the REAL file (not a
scratchpad copy), the same procedure as section 7: a green control
immediately before, the targeted literal source mutation, the specific new
test going red (and nothing else), then a `diff`-verified byte-identical
restore.

### 10a. M10 -- delete `cooldown_record_hit(_limit)`

```
$ python3 - [mutation script: replaced the production call with
  "pass  # MUTATION M10: cooldown_record_hit(_limit) REMOVED"]
mutated: M10
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
1 failed, 29 passed, 1 warning in 2.62s

FAILED test_a_real_classified_failure_through_generate_content_actually_persists_the_cooldown
  AssertionError: the production entry point must have persisted the cooldown
  assert False is True
   +  where False = cooldown_active()
```

Exactly the intended test died; the other 29 (including all the OLD tests
that pre-seed cooldown state directly, which is precisely why Q/A found
they could not have caught this) stayed green.

### 10b. M3 -- delete `cooldown_clear_on_success()`

```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
1 failed, 29 passed, 1 warning in 2.32s

FAILED test_a_real_success_after_cooldown_removes_the_persisted_state_file
  AssertionError: a real success must remove the state FILE -- if it merely reads as
  expired without being cleared, a stale record survives on disk
  assert not True
   +  where True = exists()
```

### 10c. M6 -- revert the tz fallback to `now.tzinfo`

```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
1 failed, 29 passed, 1 warning in 2.32s

FAILED test_bare_reset_time_with_no_explicit_timezone_uses_host_local_zone_not_utc
  AssertionError: parsed as 2026-08-18 17:45:00+02:00 local -- expected 3:45pm LOCAL
  wall-clock; if this instead reads as 3:45pm UTC, the tz fallback reverted to now.tzinfo
  assert (17, 45) == (15, 45)
```

The failure is the exact, predicted symmetric error: 17:45 (3:45pm
misread as UTC, then displayed back in UTC+2) instead of the correct 15:45.

### 10d. Restore and final clean run

```
$ diff /tmp/claude_code_client_fixed_backup2.py backend/agents/claude_code_client.py
(no output -- byte-identical)
$ sha256sum backend/agents/claude_code_client.py
76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
30 passed, 1 warning in 2.28s
RAW_EXIT=0
```

**CORRECTED (Q/A cycle 2, Contradiction finding):** this paragraph previously
claimed "All 11 new/touched guards ... M1, M2, M4, M5, M7, M8 (cycle-0,
section 7a/this file's earlier sections) plus M10, M3, M6" -- that
provenance does not reproduce. M2, M4, M5, M7 and M8 were never run in
THIS file; they are the Cycle-1 Q/A's OWN independent in-memory mutation
matrix, recorded in `evaluator_critique_86.120.md`, not literal source
mutations performed here. The literal source-file mutations THIS file
actually contains are: M1 (section 7a, pre-subprocess guard), M11 (section
7b, function-level classifier, superseded as criterion-5 evidence -- see
below), M10/M3/M6 (section 10, this cycle's fix), and M17 (section 11
below, cycle-2's fix). That is 6 cells with literal evidence here; M2, M4,
M5, M7, M8 are correctly attributed to `evaluator_critique_86.120.md`'s
Cycle-1 record, not to this file. Combined (this file's 6 + the Cycle-1
critique's 5 independent cells, M11 superseded rather than double-counted),
every criterion-bearing guard has a killing mutant on record somewhere in
the handoff, with the provenance now stated accurately per source.

M11 (the function-level classifier mutation) is superseded as
criterion-5 evidence by the call-site-level
`test_MUTATION_removing_limit_detection_reverts_to_generic_breaker`, which
already drove the mutation through `generate_content` correctly (see the
section 7b correction above) -- criterion 11 ("mutation-test every new
guard ... each mutant KILLED") is now satisfied.

## 11. Q/A cycle-2 fix -- the corrupt-record fail-safe guard (M17), mutation-verified

Q/A cycle 2's independent mutation matrix found `cooldown_status()`'s
except-branch (`active = True  # a corrupt record fails toward SAFE (still
cooling down)`) had zero test coverage -- inverting it to fail-open
(`active = False`) survived the full 30-test suite, even though Cycle 1's
own code-review leg had explicitly relied on that exact safety property to
wave through the module's seven new broad `except Exception` blocks. One
new test closes this
(`test_a_corrupt_but_parseable_cooldown_record_fails_toward_still_cooling`),
mutation-verified the same way as every other guard in this file.

```
$ sha256sum backend/agents/claude_code_client.py
76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1
$ python3 - [mutation script: replaced
  "active = True  # a corrupt record fails toward SAFE (still cooling down)"
  with "active = False  # MUTATION M17: inverted to fail-OPEN"]
mutated: M17
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
1 failed, 30 passed, 1 warning in 2.81s

FAILED test_a_corrupt_but_parseable_cooldown_record_fails_toward_still_cooling
  AssertionError: a corrupt-but-parseable record must fail toward SAFE (still cooling) --
  never toward resuming subprocess spawns against a possibly-still-exhausted quota
  assert False is True
   +  where False = cooldown_active()
```

Exactly the intended test died; the other 30 stayed green.

```
$ diff /tmp/claude_code_client_fixed_backup3.py backend/agents/claude_code_client.py
(no output -- byte-identical)
$ sha256sum backend/agents/claude_code_client.py
76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
31 passed, 1 warning in 2.26s
RAW_EXIT=0
```

All criterion-11-bearing guards now have a killing mutant on record: M1,
M11-superseded (this file); M2, M4, M5, M7, M8 (evaluator_critique_86.120.md
Cycle 1); M10, M3, M6, M17 (this file, Cycle 1 and Cycle 2 fixes).
