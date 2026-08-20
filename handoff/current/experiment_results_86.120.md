# Experiment Results -- phase-86.120

## What was built

`backend/agents/claude_code_client.py` gained a disk-persisted, weekly/session/
Opus-quota-aware cooldown for the Claude Code CLI rail, sitting alongside the
pre-existing phase-66.1 per-cycle in-memory breaker (unchanged for ordinary
transient failures). A best-effort classifier (`classify_limit_failure`) reads
the CLI's own documented limit-exhaustion sentences (session / weekly / Opus,
per `code.claude.com/docs/en/errors`) from the FULL stdout / JSON envelope
`result` field -- not the pre-existing truncated `str(exc)[:150]`, which
research measured actually severs the human sentence on the one real captured
failure envelope in this repo. A classified hit is persisted to
`backend/agents/_cache/cc_rail_cooldown.json` (survives `rail_guard_reset()`
and a backend process restart) and checked at the very top of
`claude_code_invoke()` -- before the binary is even resolved -- so it protects
BOTH known callers: `ClaudeCodeClient.generate_content()` and
`backend/services/ticket_queue_processor.py`'s direct call (a second call site
discovered by grepping all consumers before implementing, not assumed). A
classified hit also trips the existing in-cycle breaker immediately (Azure
"accelerated circuit breaking", N=1 rather than N=20) without paging, since a
P1 for an already-classified, already-handled condition would be the exact
alarm-fatigue anti-pattern `scripts/away_ops/auth_state.py`'s own module
docstring warns about.

One real bug was found and fixed DURING implementation, before any test was
written against it: the reset-time parser initially fell back to `now.tzinfo`
(UTC, since the default `now` is `datetime.now(timezone.utc)`) when the CLI
message carried no explicit timezone suffix -- silently misinterpreting a
local wall-clock time as UTC. Fixed to use the host's actual local timezone
via `datetime.now().astimezone().tzinfo`, matching how the CLI actually prints
these times (the doc's `3:45pm` examples carry no explicit zone; the one real
captured example does, `"resets 1am (Europe/Oslo)"`). A second gap was caught
the same way: the regexes originally required `HH:MM` (colon + minutes), but
the one real captured envelope reads `"resets 1am"` -- no minutes at all.
Both were fixed and independently re-verified against the real captured file
before the test suite was written.

## Files changed
- EDIT: `backend/agents/claude_code_client.py` (+322/-2 lines) -- classifier,
  persisted cooldown module (`cooldown_record_hit`/`cooldown_status`/
  `cooldown_clear_on_success`/`cooldown_active`), pre-subprocess gate inside
  `claude_code_invoke`, classification wired into the returncode!=0 branch,
  `_rail_guard_open_for_quota` (N=1 trip, no page), `rail_guard_status()`
  folds in cooldown state, `generate_content()`'s except-block skips the
  generic counter for `cooldown_blocked` errors, success path clears the
  cooldown.
- EDIT: `backend/config/settings.py` (+6 lines) -- new
  `claude_rail_cooldown_default_hours` field (default 6.0h, bounds
  0.5-192h), the safety-net backoff used only when the CLI's own reset-time
  string cannot be confidently parsed.
- NEW/EDIT: `backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py`
  (31 tests total) -- classifier unit tests (real captured envelope +
  labelled-synthesized weekly/Opus + plain-text fallback + the 150-char
  truncation landmine), persistence tests (survives `rail_guard_reset()`,
  survives a simulated restart, expires, self-clears, falls back safely on
  an unparseable reset time, never trusts a past/implausible parsed time),
  THREE wiring-level guard tests added in the Q/A cycle-1 fix pass (drive a
  real classified failure through the production entry point and assert the
  persisted state, not a pre-seeded fixture; assert the cooldown state FILE
  -- not the `cooldown_active()` boolean -- is gone after a real success;
  pin the tz-fallback fix against the host's actual local zone), mock-level
  pre-subprocess skip tests covering BOTH call sites across multiple
  simulated cycles including a simulated restart, a test proving the
  cooldown skip does not inflate the old generic counter, tests proving the
  existing generic breaker is unchanged for non-limit failures, the
  settings field, one always-on signal agent driven through the shared
  `make_client()` entry point, and the `make_client()` $0-metered
  routing-breach REGRESSION test.
- NEW: `backend/agents/_cache/.gitignore` (matches the existing
  `backend/services/_cache/.gitignore` convention: `*` / `!.gitignore`,
  runtime state is never committed).

## Verbatim verification
```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q
..............................                                           [100%]
32 passed, 1 warning in 2.42s
RAW_EXIT=0
```
(27 -> 30 after the Q/A cycle-1 fix pass added three wiring-level guard tests;
30 -> 31 after the Q/A cycle-2 fix pass added one more (the corrupt-record
fail-safe guard, M17); 31 -> 32 after the Q/A cycle-3 fix pass added one more
(the result-field-extraction guard); see "Q/A cycle-1 fix", "Q/A cycle-2 fix"
and "Q/A cycle-3 fix" below. This step PARKED after cycle 3 under this
project's 3rd-consecutive-CONDITIONAL rule -- see
`handoff/current/escalation_86.120_third_conditional.md` -- so this fix is
recorded but was never re-evaluated by a cycle-4 Q/A.)

Wider regression sweep, DERIVED scope (Q/A cycle-1 correction: the original
"16 files" here was typed by hand and undercounted by one --
`test_phase_86_110_heartbeat_isolation.py` also matches and was missed;
re-derived via
`grep -rln "claude_code_client\|claude_code_invoke\|rail_guard" backend/tests/*.py | grep -v conftest`,
17 files (this step's own suite is already one of the 17, matching on its
own imports -- not counted twice per Q/A cycle-2's correction), plus the
charts.py NaN-fix suite from earlier in this session (does not match the
grep pattern), 18 files total):
```
366 passed, 1 failed in 47.43s
```
(Q/A independently reproduced this same derived scope against the pre-fix 27-test
suite as 363 passed/1 failed; 366 = 363 + the 3 new tests added in this cycle.)
The one failure, `test_phase_60_3_data_integrity.py::test_60_3_flag_defaults_off`,
is PRE-EXISTING and unrelated: it asserts `Settings().paper_data_integrity_enabled
is False`, but this environment's `backend/.env` already sets that flag True
(confirmed via `/api/settings/flags`: `"paper_data_integrity_enabled":
{"in_force":true,"env_file":true,...}`) -- a live-environment/test drift on a
flag this step never touches. `git diff --stat backend/config/settings.py`
shows exactly the 6 lines this step added; nothing near that flag.

Lint gate, this project's actual convention (F821/F401/F811 -- confirmed via
the phase-75.11.4 Q/A precedent that the real CI gate scopes to these three
codes, not ruff's full default ruleset):
```
$ ruff check --select F821,F401,F811 <every file this step touched>
All checks passed!
```
(A broad unscoped `ruff check` also flags `UP045`/`BLE001` on lines this step
added, but those same codes ALSO fire on dozens of pre-existing lines in the
same file using the same `Optional[X]` / broad-`except Exception` style
throughout -- confirmed by running the broad check on the untouched parts of
the file. Not a new pattern introduced here.)

## Mutation testing

Two MUTATION cells are named directly in the immutable criteria; both are
implemented as in-suite `monkeypatch`-based tests (`test_MUTATION_*`, the
same idiom other suites in this repo use) AND independently re-verified via
literal source-file mutation (mirroring this session's earlier charts.py fix):

| Mutation | In-suite test | Literal source-patch verification |
|---|---|---|
| Remove the pre-subprocess cooldown guard | `test_MUTATION_removing_the_pre_subprocess_guard_lets_subprocess_run_again` | Removed the guard block from the real file; **exactly the 5 guard-dependent tests died** (`test_claude_code_invoke_skips_subprocess_when_cooldown_active`, `test_ticket_queue_processor_direct_call_pattern_also_blocked`, `test_generate_content_skips_subprocess_across_two_cycles_and_a_restart`, `test_cooldown_block_does_not_inflate_the_generic_consecutive_failure_counter`, `test_an_always_on_signal_agent_respects_cooldown_via_shared_entry_point`); the other survived untouched. sha256 restored byte-identical (`76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1`, verified via `diff`). |
| Remove the classifier (call-site level, criterion 5) | `test_MUTATION_removing_limit_detection_reverts_to_generic_breaker` | Drives the mutation through `generate_content` (production wiring), not the classifier's own unit tests -- this is the correct level for criterion 5 (see the cycle-1 correction below on why the earlier function-level characterization was superseded). |

**Q/A cycle-1 fix (`evaluator_critique_86.120.md`): three guards this table
did not previously cover.** Q/A's own independent mutation matrix found that
deleting `cooldown_record_hit(_limit)` (the sole production call that
persists a classified hit), deleting `cooldown_clear_on_success()`, and
reverting the tz-fallback fix to `now.tzinfo` ALL left the 27-test suite
green -- every existing cooldown test built its fixture state directly via
`cooldown_record_hit(...)` in test setup rather than driving a real failure
through the production entry point, so none of them could have caught a
break in the wiring itself. Three new tests close this, each independently
mutation-verified against the real file (control green, targeted mutant,
byte-identical restore -- full transcripts in `live_check_86.120.md` section
10):

| Guard | New test | Result under its own literal mutation |
|---|---|---|
| `cooldown_record_hit(_limit)` call site | `test_a_real_classified_failure_through_generate_content_actually_persists_the_cooldown` | KILLED -- 1 failed / 29 passed |
| `cooldown_clear_on_success()` call site | `test_a_real_success_after_cooldown_removes_the_persisted_state_file` | KILLED -- 1 failed / 29 passed |
| tz fallback (`datetime.now().astimezone().tzinfo`, not `now.tzinfo`) | `test_bare_reset_time_with_no_explicit_timezone_uses_host_local_zone_not_utc` | KILLED -- 1 failed / 29 passed |

Also corrected: the original function-level classifier mutation (replacing
the whole `classify_limit_failure` body) was characterized as proving
"detection is load-bearing" from its 18/27 kill count. Q/A found 17 of those
18 kills are the classifier's OWN unit tests failing because their test
BODY calls the function directly (tautological -- a unit test for X fails
when X is broken, which proves nothing about production wiring), and two of
the nine claimed survivors were mischaracterized (see the correction in
`live_check_86.120.md` section 7b). The genuine wiring-level evidence for
criterion 5 is `test_MUTATION_removing_limit_detection_reverts_to_generic_breaker`,
which already drove the mutation through `generate_content` correctly and
needed no change.

Every mutation pass ran on the REAL file (not a scratchpad copy) with an
explicit control run (green) immediately before and after each mutation, and
a `diff`-verified byte-identical restore each time. **Provenance correction
(Q/A cycle 2):** an earlier version of this paragraph credited "M1, M2, M4,
M5, M7, M8, M10, M3, M6" to this step's own mutation passes; M2/M4/M5/M7/M8
were never run here -- they are the Cycle-1 Q/A's own independent in-memory
matrix, recorded in `evaluator_critique_86.120.md`. This document's own
literal source-file mutations are M1, M11 (superseded), M10, M3, M6, and
(cycle-2 fix, below) M17.

## Q/A cycle-2 fix -- the corrupt-record fail-safe guard (M17)

Q/A cycle 2's independent mutation matrix found `cooldown_status()`'s
except-branch comment ("a corrupt record fails toward SAFE, still cooling
down") had zero test coverage -- inverting `active = True` to `active =
False` survived the full 30-test suite, even though Cycle 1's own
code-review leg had explicitly relied on that exact claimed property to
wave through the module's seven new broad `except Exception` blocks. One
new test closes this
(`test_a_corrupt_but_parseable_cooldown_record_fails_toward_still_cooling`),
independently mutation-verified against the real file: control 31/31 green
(after the fix), mutant KILLED (1 failed/30 passed, the one intended test),
byte-identical sha256 restore confirmed. Full transcript:
`live_check_86.120.md` section 11.

Two NOTE-level Q/A cycle-2 findings were also fixed in place: the inline
comment on `test_a_real_classified_failure_through_generate_content_actually_persists_the_cooldown`
overclaimed that its second-call assertion isolates the persisted-cooldown
mechanism from the in-cycle breaker -- a differential mutation run (M1
alone vs. M1+M4) showed both trip together, so the comment now correctly
attributes the persisted-state claim to
`test_generate_content_skips_subprocess_across_two_cycles_and_a_restart`
instead; and the "11 guards" provenance/count claim above and in
`live_check_86.120.md` section 10 is corrected as described.

## Q/A cycle-3 fix -- the result-field-extraction guard

Q/A cycle 3's independent 13-cell mutation matrix (a NEW search each time,
not a repeat of prior cycles') found that deleting `classify_limit_failure`'s
JSON-envelope `result`-extraction block (falling through to using the raw
envelope text as the message) survived the full 31-test suite. Two real
consequences, both measured by the Q/A: (a) on the one real captured
envelope, the operator-facing `cooldown_message` (surfaced via
`rail_guard_status()`) degrades from a clean 56-character sentence to a
500-character raw JSON blob; (b) more seriously, a FALSE-POSITIVE risk --
an envelope whose `result` field is a normal, successful analysis but which
merely contains limit-shaped text somewhere else (e.g. a debug echo)
misclassifies as a real quota hit, which would engage a multi-hour cooldown
on a call that actually succeeded. The existing truncation test
(`test_classify_reads_full_message_not_the_150_char_truncation`) could not
catch this -- it only proves the message isn't cut off at 150 characters, not
that it specifically comes from the `result` field.

One new test closes this
(`test_classify_extracts_the_result_field_not_the_raw_envelope`), asserting
both consequences directly and adding the false-positive control Q/A named.
Independently mutation-verified against the real file: control 32/32 green,
the exact mutation applied (the whole extraction block replaced with
`pass`), the one intended test going red (1 failed/31 passed) with the
failure message itself confirming the false-positive risk (`kind='session'`
derived from raw JSON), then diff-verified byte-identical sha256 restore
(unchanged across all four verification passes:
`76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1`).

**This step PARKED after Cycle 3 under this project's 3rd-consecutive-
CONDITIONAL rule** (three straight CONDITIONAL verdicts, each finding a
DIFFERENT, genuinely new mutation-coverage gap while unanimously judging the
shipped code correct) rather than spawning a Cycle 4 that the rule requires
to auto-FAIL regardless of merit. This fix is recorded and mutation-verified
here for completeness, but was never submitted to a Cycle-4 Q/A. See
`handoff/current/escalation_86.120_third_conditional.md` for the full
disposition and the operator decision this needs.

## Success criteria coverage
| Criterion (abbreviated) | Status |
|---|---|
| 1. Recognizes the CLI's specific limit signature, distinct from generic failure | MET -- `classify_limit_failure`; see "reading note" in contract_86.120.md for the compatible-superset reading of run_away_session.sh's message class |
| 2. Cooldown persisted to disk, survives per-cycle reset + process restart | MET -- `cc_rail_cooldown.json`; `test_cooldown_survives_rail_guard_reset`, `test_cooldown_state_persists_across_simulated_process_restart`, and (Q/A cycle-1 fix) `test_a_real_classified_failure_through_generate_content_actually_persists_the_cooldown` closes the wiring-level gap Q/A found |
| 3. Rail skipped BEFORE any subprocess spawn, mock-level assertion, 2+ cycles incl. a restart | MET -- guard sits at the top of `claude_code_invoke`; `test_generate_content_skips_subprocess_across_two_cycles_and_a_restart` |
| 4. Bounded, self-clearing, operator-configurable backoff | MET -- `claude_rail_cooldown_default_hours` Settings field; `test_cooldown_expires_after_retry_at`, and (Q/A cycle-1 fix) `test_a_real_success_after_cooldown_removes_the_persisted_state_file` -- the file-existence check Q/A required, since `cooldown_active()` alone cannot discriminate an expired-but-uncleared record |
| 5. MUTATION: remove detection -> reverts to generic breaker | MET -- see Mutation testing table (call-site-level test; the function-level characterization was corrected) |
| 6. MUTATION: remove the skip guard -> assertion goes red | MET -- see Mutation testing table |
| 7. Existing generic breaker unchanged for non-limit failures | MET -- `test_generic_failure_still_feeds_the_old_breaker_and_never_engages_cooldown`, `test_generic_breaker_still_trips_at_the_configured_threshold` |
| 8. At least one always-on signal agent respects the cooldown | MET -- `test_an_always_on_signal_agent_respects_cooldown_via_shared_entry_point`, driven through the functionally-equivalent call shape `pead_signal.py:300` uses (`make_client(getattr(settings, "pead_signal_model", "claude-haiku-4-5"), None, settings, enable_prompt_caching=False)` -- corrected citation, was miscited as :298 with an inexact "verbatim" claim) |
| 9. REGRESSION: make_client() routing-breach guard intact | MET -- `test_make_client_routing_breach_guard_still_fires` |
| 10. Cooldown observable without grepping logs | MET -- folded into `rail_guard_status()` |
| 11. Mutation discipline: control green first, killed, byte-identical restore | Q/A cycle-1 found 3 guards (M10/M3/M6) with no killing mutant, closed; cycle-2 found a 4th (M17), closed; cycle-3 found a 5th (result-field extraction), closed. NOT re-evaluated by a Cycle-4 Q/A -- step PARKED per the 3rd-consecutive-CONDITIONAL rule; see escalation_86.120_third_conditional.md |

## Scope honesty / follow-up

Not done here, by design (stated in the contract's Out-of-scope section
before implementation started):
- `scripts/away_ops/run_away_session.sh`'s own `LIMIT_HIT` regex (missing
  `weekly`/`Opus`, a real defect on the SEPARATE away-ops harness rail) --
  filing as its own follow-up, not silently folded in or silently ignored.
- Switching the rail to `--output-format stream-json` for the documented
  `system/api_retry` enumerated classifier -- the research brief's own words
  are "a real option, and a real cost"; deferred as a candidate future step.
- Exact reset-time arithmetic remains best-effort. The fallback (a fixed,
  configurable backoff from `hit_at`) is the safety net whenever the parsed
  reset string is ambiguous or the pattern doesn't match at all.

## References
- Contract (pre-commit): `handoff/current/contract_86.120.md`
- Research: `handoff/current/research_brief_86.120.md`
- Real captured limit envelope used in tests:
  `handoff/away_ops/session_pm_20260707T200007Z.json`
