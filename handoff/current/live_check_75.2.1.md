# live_check 75.2.1 -- verbatim evidence (2026-07-20)

## Immutable verification command -- exit 0

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_2_1_push_approval.py -q
........................                                                 [100%]
24 passed in 0.17s
```
(Cycle-1 recorded "22 passed"; cycle 2 added two guards. Refreshed after Q/A
wf_236c0b88-77b flagged the stale figure.)

## Byte-identity proof: NO immutable criterion was amended

```
$ python3 - <<'PY'   # compares .claude/masterplan.json against git show 256867d3:
baseline captured for: ['4.14.4', '4.14.24', '4.17.9']
byte-identity vs 256867d3: VERIFIED for all three
PY
```
Asserted permanently by `test_verification_is_byte_identical_to_baseline`
(parametrized over all three step ids), which diffs the live masterplan against
`git show 256867d3:.claude/masterplan.json`.

## MUTATION EVIDENCE (criterion 6) -- every new guard proven to fail when broken

```
M1  drop the sink-side authorization re-check (leave only the matcher)
    => 2 failed, 20 passed    [CAUGHT] non-operator + unset-operator-id tests

M2  remove the HEAD re-resolve at push time (reopen the TOCTOU)
    => 1 failed, 21 passed    [CAUGHT] test_head_moving_after_approval_refuses_the_push

M3  remove the TTL expiry check
    => 1 failed, 21 passed    [CAUGHT] test_expired_approval_refuses_the_push

M4  register the OPERATOR's own message ts instead of the bot's (self-approval)
    => 6 failed, 16 passed    [CAUGHT] the ts-binding, full-flow and single-use tests

M5  silently AMEND an immutable criterion (4.14.24 verification.command := "true")
    => 1 failed, 21 passed    [CAUGHT] test_verification_is_byte_identical_to_baseline

All five mutations reverted from scratchpad backups; final run: 22 passed.
```

M5 is the load-bearing one for part (a): the annotation approach is only
trustworthy if a future silent amendment is detectable. It is.

## No-regression evidence (the set -> dict change to _pending_push_ts)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_2_slack_control_plane.py \
      backend/tests/test_phase_62_2_operator_tokens.py -q
80 passed in 0.18s
```
Every phase-75.2 guarantee survives: `dict` preserves the `in` and `.clear()`
semantics the existing suite relies on.

## git diff --stat (change surface)

```
 .claude/masterplan.json                            |  59 +++++-   (3 superseded_record annotations + the 75.2.1 install)
 backend/slack_bot/commands.py                      | 218 +++++++--  (request handler, sha/TTL binding, re-validate at push, hoisted regexes)
 backend/tests/test_phase_75_2_slack_control_plane.py |  16 +-     (4 legacy call sites pass head_sha="" explicitly + comment)
?? backend/tests/test_phase_75_2_1_push_approval.py                (new, 24 tests)
```

FOURTH FILE DISCLOSED (cycle 2): `test_phase_75_2_slack_control_plane.py` was changed
and neither change-surface block listed it. Those four call sites opt out of the new sha
re-validation explicitly (their fixture's check_output spy counts every call, including
rev-parse); they assert identity / single-use / to_thread, none of which depends on the
sha binding. 104/104 pass across all three suites.

NOT PART OF THIS STEP: `backend/backtest/experiments/mda_cache.json` is dirty in the
working tree with an mtime predating this step's research. It is regenerated cache, not
75.2.1's surface, and is excluded from this step's commit.

## $0 / control-plane-only confirmation

No LLM call added, removed or repointed. No .env edits. No trading logic touched.
No network added to the push path (deliberately no `git fetch` -- it would widen
the request-to-approval window; the message says the comparison is against the
last-known origin/main rather than implying freshness).

## Operator note

`PUSH` (bare, uppercase, colon-less) in the approval channel is the trigger. It is
colon-less and anchored on purpose: `PUSH REQUEST: main` matches the operator-token
grammar, so an unanchored trigger would make the two handlers ambiguous.
`test_trigger_does_not_collide_with_the_operator_token_grammar` pins that against the
PRODUCTION regex objects.

CYCLE-2 CORRECTION: an earlier revision of this file said the operator-token handler
registers first and would swallow the request. Verified registration order is
  idx 0: handle_push_request | idx 1: handle_operator_token | idx 2: handle_any_message
so PUSH registers FIRST. The conclusion is unchanged (the regexes are disjoint), but
the hazard runs the other way: widening the push pattern would swallow operator TOKENS.

## Cycle-2 mutation evidence (new guards)

```
M6  restore `posted_ts = (resp or {}).get("ts") if isinstance(resp, dict) else None`
    -- the ORIGINAL cycle-1 bug that left the path inert in production:
  $ pytest backend/tests/test_phase_75_2_1_push_approval.py -q
  6 failed, 18 passed                    <-- CAUGHT (cycle-1 suite could not see this)

M7  register the push handler AFTER the catch-all @app.message(''):
  $ pytest backend/tests/test_phase_75_2_1_push_approval.py -q
  1 failed, 23 passed                    <-- CAUGHT (order guard no longer vacuous)

M8  restore the `head_sha: str = ""` default (silently disables TOCTOU re-validation):
  $ pytest backend/tests/test_phase_75_2_1_push_approval.py -q
  1 failed, 23 passed                    <-- CAUGHT by test_register_requires_an_explicit_head_sha
  (Q/A wf_236c0b88-77b ran this independently in cycle 2 and required it be recorded.)

M9  THE VACUITY MUTATION -- regress the _say stub to a plain dict AND restore the
    production isinstance(resp, dict) guard TOGETHER, so the fixture can no longer
    represent the failure:
  BEFORE the cycle-3 fix: suite GREEN while the production push path was inert
    (proven by Q/A wf_236c0b88-77b: 8 passed, exit 0).
  AFTER  the cycle-3 fix:
  $ pytest backend/tests/test_phase_75_2_1_push_approval.py -q
  1 failed, 23 passed                    <-- CAUGHT by test_say_stub_matches_the_production_response_shape

  M9 is the important row. The non-dict fixture is the SINGLE thing that makes M6
  detectable, and until cycle 3 nothing pinned it -- the guard that claimed to
  ("test_say_response_stub_is_not_a_dict") asserted only upstream library facts and
  never referenced the fixture at all. It now inspects the object the fixture
  actually returns.

Both reverted. Full run across all three affected suites:
  $ pytest backend/tests/test_phase_75_2_1_push_approval.py \
           backend/tests/test_phase_75_2_slack_control_plane.py \
           backend/tests/test_phase_62_2_operator_tokens.py -q
  104 passed in 0.24s
```

Verified registration order (dumped from the real register_commands):
```
  idx 0: handle_push_request     pattern='^\s*PUSH\s*$'
  idx 1: handle_operator_token   pattern='^(?:[0-9][0-9.]*\s+)?[A-Z][A-Z0-9 _-]+:\s*.+$|^(?:HALT-DEV|RESUME-DEV)$'
  idx 2: handle_any_message      pattern=''
```
