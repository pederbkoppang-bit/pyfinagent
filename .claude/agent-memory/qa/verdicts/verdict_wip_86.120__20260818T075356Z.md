STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.120
WRITTEN: 2026-08-18T07:53:56Z

# Q/A write-first record -- step 86.120 (CC rail limit-aware cooldown)

Spawn started. Reading qa.md (done, in full). Beginning harness-compliance audit.

## Plan
A. Harness compliance (5 items)
B. Deterministic: immutable command, git status/diff scope, lint (F821/F401/F811), runtime smoke
C. Mutation / guard-vacuity independent pass
D. 11 immutable criteria MET/NOT MET

(findings appended below as established)

## A. Harness compliance -- PASS (all 5)
- research_brief_86.120.md 09:31:11 < contract_86.120.md 09:35:22 < settings.py 09:41 < tests 09:49:40 < claude_code_client.py 09:50:40 < experiment_results 09:51:50 < live_check 09:52:57. Order OK.
- masterplan 86.120 status=pending (NOT done). grep -F "86.120" handoff/harness_log.md -> ZERO rows. log-last OK.
- qa_wip: source_present=true, attempt_number=1, prior_attempts=0. verdict_history_86_21 --evidence-only: status=no_rows_for_step, verdicts=(none). NOT a re-spawn -> no verdict-shopping possible. sequence: no prior verdicts recorded (ledger reports no_rows_for_step; nothing writes it automatically).
- masterplan diff = insertion of the 86.120 object ONLY (+26 lines); criteria in my prompt match masterplan verbatim (no erosion, no amendment of any other step).

## B. Deterministic
- IMMUTABLE CMD: `.venv/bin/python -m pytest backend/tests/test_phase_86_120_cc_rail_limit_aware_cooldown.py -q` -> "27 passed, 1 warning in 2.45s", RAW_EXIT=0. REPRODUCED.

## C. INDEPENDENT mutation matrix (in-memory sys.modules injection; tree NEVER written)
Driver: scratchpad/mut86120.py. Reads backend/agents/claude_code_client.py, applies a textual
mutation, execs into a module injected as backend.agents.claude_code_client, runs the step suite.
TREE_SHA_AFTER == 76b47a217489eb5be665db2d6eb354181bde5d2746c515c8da63c6f8dde5dcb1 on EVERY cell
(== the sha Main's evidence claims for the byte-identical restore; verified unchanged after each run).

| cell | mutation | result |
|---|---|---|
| C0 | control (null mutant through the SAME injection harness) | 27 passed, rc=0 -> harness proven non-distorting |
| M1 | delete the pre-subprocess cooldown guard block in claude_code_invoke | 5 failed / 22 passed -> KILLED |
| M2 | delete the classification BLOCK at the call site in claude_code_invoke | 1 failed / 26 passed -> killed by ONE test, and that test asserts the BREAKER TRIP, not cooldown engagement |
| M3 | delete `cooldown_clear_on_success()` wiring inside claude_code_invoke | **27 passed -> SURVIVED** |
| M4 | delete `_rail_guard_open_for_quota(...)` call | 1 failed -> KILLED |
| M5 | revert `if not getattr(exc,'cooldown_blocked',False)` to unconditional record_failure | 1 failed -> KILLED |
| M10 | delete `cooldown_record_hit(_limit)` -- the ONLY production call that persists the cooldown | **27 passed -> SURVIVED (BLOCKING)** |
| M11 | `classify_limit_failure` returns None at function level (== Main's own claimed mutant) | 18 failed / 9 passed -> reproduces Main's "18 of 27, 9 survived" EXACTLY |

### FINDING 1 (BLOCKING) -- M10 survives: criterion 2's production wiring has ZERO coverage
Deleting the single line `cooldown_record_hit(_limit)` from claude_code_invoke leaves the suite 27/27
GREEN. With that line gone the rail classifies the limit, trips the in-cycle breaker, raises -- and
persists NOTHING, so rail_guard_reset() next cycle restores exactly today's behaviour (re-spawn until
the generic breaker). The step's central purpose is defeated and no test notices.
ROOT CAUSE: every cooldown test BUILDS the state itself via
`cooldown_record_hit(classify_limit_failure(...))` in its own setup, then asserts the guard skips.
NO test drives a real limit-shaped failure through claude_code_invoke/generate_content and then
asserts `cooldown_active() is True`. The two end-to-end assertions that do drive a real failure
(test_a_classified_hit_opens_the_in_cycle_breaker_immediately, test_a_classified_hit_does_not_page)
assert the breaker and the pager, never the persisted cooldown.

### FINDING 2 -- M11 vs M2: the claimed "18 of 27 killed" is a MIS-ATTRIBUTED kill mechanism
Main's classifier mutant is FUNCTION-level, so 17 of its 18 kills are tests whose own SETUP calls
classify_limit_failure() to build a fixture -- the test harness breaking, not the production wiring.
The call-site mutation (M2), which is what criterion 5 literally asks for ("remove the limit-signature
detection"), kills only ONE test, via the breaker-trip assertion.

### FINDING 3 -- M3 survives: criterion 4's "a single subsequent success clears the cooldown" wiring untested
test_cooldown_clears_on_success calls the helper DIRECTLY. Deleting the call inside claude_code_invoke
kills nothing. Impact is smaller than FINDING 1 (cooldown_status() derives `active` from retry_at, so a
stale record still reads inactive after the window) but the criterion's wiring is unproven, and
criterion 11 requires every new guard to be mutation-killed.

## C (cont). Additional mutation cells
| cell | mutation | result |
|---|---|---|
| M6 | revert tz fallback `datetime.now().astimezone().tzinfo` -> `now.tzinfo` (the exact bug experiment_results says was found+fixed) | **27 passed -> SURVIVED** |
| M7 | make minutes MANDATORY in _RESET_CLOCK_RE (the 2nd claimed bug) | 1 failed -> KILLED (real-envelope test) |
| M8 | delete the retry_at past/>9d clamp | 1 failed -> KILLED |
M10 and M3 each re-run a 2nd time: both SURVIVED again (27 passed). Tree sha unchanged throughout.

### FINDING 4 -- live_check section 4's "self-cleared" observation does not discriminate
Verbatim: "cooldown_active() now that retry_at is in the past: False" ... "cooldown_active() after
a real success: False (self-cleared)". FALSE before the success and FALSE after -- the observation
is identical whether the clear-on-success wiring exists or not (M3 confirms). "(self-cleared)" is
an INFERENCE the probe cannot support. A discriminating probe asserts the state FILE is gone.

### FINDING 5 -- live_check 7b survivor accounting is wrong for 2 of 9
Claim: "The 9 survivors are exactly the tests that never call the classifier". I enumerated them:
test_classify_returns_none_for_generic_failure (calls the classifier 3x; survives VACUOUSLY -- it
asserts None and the mutant returns None) and test_a_classified_hit_does_not_page (drives
claude_code_invoke with a limit payload; survives because nothing pages either way -- and could not
page anyway at N=1 vs threshold 20) BOTH call the classifier. Also no survivor matches "three more
that write cooldown state directly".

### FINDING 6 -- the wider-sweep number does not reproduce under a DERIVED scope
Claimed "344 passed, 1 failed" over "16 files". DERIVED scope
(`grep -rln "claude_code_client\|claude_code_invoke\|rail_guard" backend/tests/*.py` minus conftest,
+ charts) = 18 files -> **363 passed, 1 failed in 43.27s**. SAME single failure, so the typed scope
hid nothing; direction conservative. The pre-existing-failure disclosure IS correct and verified
(test_60_3_flag_defaults_off, paper_data_integrity_enabled; settings diff = only the 6 new lines).

## B (cont). Remaining deterministic gates
- LINT (derived scope: tracked-modified UNION untracked *.py = 7 files, NUL-delimited xargs):
  "All checks passed!", exit 0. POSITIVE CONTROL on a planted F821/F401 file -> exit 1. Gate is live.
- RUNTIME SMOKE: imports of claude_code_client / settings / charts OK; cooldown_status() ->
  {'cooldown_active': False}; rail_guard_status() keys include cooldown_active; settings default 6.0.
  Live backend /api/health -> 200.
- REAL ENVELOPE CLAIM REPRODUCED: handoff/away_ops/session_pm_20260707T200007Z.json contains
  "hit your session limit \xc2\xb7 resets 1am (Europe/Oslo)" -- bare hour, no minutes. Confirms the
  minutes-optional fix is grounded in real data.
- CONSUMERS: rail_guard_status() readers at autonomous_loop.py:1951/:2651 and llm_client.py:2157 all
  use .get() on named keys -> additive keys are non-breaking. No consumer-contract break.
- SCOPE: no unintended production change inside 86.120's scope. charts.py + its new test = this
  session's earlier disclosed NaN fix; scripts/qa/mutation_86_59.py, rank_stability_86_59.py,
  evaluator_critique_86.59.md, verdict_ledger.jsonl = concurrent PEER-session 86.59/86.118 work.

## D. Criteria
1 MET. 2 CAPPED (wiring unguarded, M10). 3 MET (note: the pre-existing free `claude auth status`
probe at :826 still runs once/cycle -- token-less, out of scope, so "zero INFERENCE spawns" is the
true claim). 4 first half MET / second half UNPROVEN (M3 + FINDING 4). 5 MET. 6 MET. 7 MET. 8 MET
(substance) with prose error (pead_signal.py is :300 and uses getattr(...), not "exact" :298 form).
9 MET (verified the test drives the real llm_client.py:2197->:2211 path). 10 MET as worded.
11 NOT MET -- M3, M6, M10 survive.

VERDICT (for the structured return): CONDITIONAL.

COMPLETED: 2026-08-18T08:07:47Z
