# Experiment Results — Step 75.2.1: close the two operator-escalated items from 75.2

- **Date:** 2026-07-20 · **Executor:** Opus 4.8 · **Operator directive:** "fix them both after your recommendation"
- **Boundary honored:** control-plane only, $0 metered, **no immutable verification criteria amended** (proven by test + mutation M5).

## What changed

```
 .claude/masterplan.json                              |  59 +++++-  (3 superseded_record annotations + the 75.2.1 step)
 backend/slack_bot/commands.py                        | 218 +++++++--
 backend/tests/test_phase_75_2_slack_control_plane.py |  16 +-      (4 legacy call sites + comment)
?? backend/tests/test_phase_75_2_1_push_approval.py                 (new, 24 tests)
```
(Cycle-1 under-reported this: it listed 189 lines for commands.py and omitted the
fourth file entirely. `backend/backtest/experiments/mda_cache.json` is dirty in the
working tree from before this step and is excluded from the commit.)

## (a) The immutable-criteria collision — recorded, not rewritten

Each of 4.14.4, 4.14.24 and 4.17.9 gains a `superseded_record` sibling. `verification.command` and `verification.success_criteria` are **byte-identical** to commit `256867d3`, verified at write time and pinned permanently by `test_verification_is_byte_identical_to_baseline`. Status stays `done` — per the append-only ADR doctrine the research cited verbatim ("Don't go back and edit accepted records"), the work *was* done; the artifact it inspected is gone.

`superseded_record` is not an invented key: a census over 837 steps found it already in use on step 68.5, and 75.2.1 is its mirror image (there the criteria were unsatisfiable; here they are untouched and only the subject is retired).

Three findings from the research changed what the records say:

1. **The sweep is complete.** All 837 steps across 99 phases were swept across every verification shape (674 dict, 126 str, 13 list, 24 None — a naive `.get("command")` sweep *crashes* on the list-shaped ones). These three are the complete set for the 75.2 cause. `test_no_other_done_step_references_the_deleted_modules` keeps it that way. Step 75.2 itself names the dead modules but **asserts their absence**, so it is correct and excluded.
2. **`f55e6973` deleted seven files, not six.** For 4.17.9 this is decisive: `git log --all --diff-filter=A -- scripts/go_live_drills/self_update_audit_test.py` is **empty** (that script never existed — the command had a name mismatch from day one), while `smoke_test_4_17_9.py` **did** exist and was deleted by 75.2. Its record names **both** causes; naming only one would misattribute the breakage to 75.2. Pinned by `test_4_17_9_record_names_both_causes`.
3. **4.17.9 belongs to a pre-existing 10-member family.** 15 done steps have verification commands referencing absent paths, ten of them phase-29 go_live_drills steps whose plan-side names never matched the on-disk `smoke_test_4_17_N.py` convention. Unrelated to 75.2, **not fixed here**, and disclosed in the record's `scope_disclosure` field rather than quietly folded in.

## (b) The inert push-approval — wired, and a TOCTOU closed on the way

The research surfaced a gap I had not planned for. `_pending_push_ts` was a set of bare strings: it recorded *that* an approval was requested, not *what was shown*. A commit landing between request and reaction would have been pushed **without ever being displayed** — OWASP Transaction Authorization 2.6/2.8, which require the approved data to be re-validated at execution.

So the mapping is now `ts -> (head_sha_shown, expires_at)`:

- **Trigger:** bare, colon-less `PUSH` via `@app.message`, registered before the catch-all. A slash command would need App-Management config, outside a $0/control-plane step. **The colon matters:** `PUSH REQUEST: main` matches the operator-token grammar, so an unanchored trigger would make the two handlers ambiguous. Pinned by a test that uses the **production** regex objects. *(Cycle-2 correction: I originally wrote that the token handler registers first and would swallow the request. It registers second — push is at index 0. The conclusion holds because the regexes are disjoint, but the hazard actually runs the other way: widening the push pattern would swallow operator tokens.)*
- **Authorization twice:** `operator_tokens._authorized(...)` in the matcher *and* re-checked at the sink — a matcher is a capability gate, not an authorization decision. This is the documented countermeasure to hermes-agent #36848 (unset env var skipping the auth block = fail-open) and GHSA-wv26-j37q-2g7p (CWE-863 approval-scope confusion, 2026-05-12).
- **Shows what is being signed (OWASP 1.1):** commit list, count, target ref and HEAD sha, posted into `_APPROVAL_CHANNEL` specifically — the reaction gate only accepts that channel, so a request posted anywhere else is un-approvable by construction.
- **Registers the BOT's ts**, never the operator's. Binding to the operator's own message would be self-approval.
- **Re-validates at execution:** HEAD is re-resolved when the reaction arrives and the push is **refused if it moved**, naming both shas in the reply. Approvals also expire (10 min, OWASP 2.9) — nothing expired before.
- **Staleness stated, not implied:** `origin/main` is a local ref and no `git fetch` is performed (it would add network and widen the request-to-approval window), so the message says the comparison is against the *last-known* `origin/main`.
- **Every 75.2 guarantee preserved:** identity → channel → ts membership → `pop()` *before* the push (a crashed push must not leave a re-approvable ts). All 80 prior tests still pass through the set→dict change.

## Verification

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_2_1_push_approval.py -q
22 passed in 0.22s                      # the step's immutable verification command, exit 0

$ .venv/bin/python -m pytest backend/tests/test_phase_75_2_slack_control_plane.py \
      backend/tests/test_phase_62_2_operator_tokens.py -q
80 passed in 0.18s                      # no regression from set -> dict
```

## Mutation testing (criterion 6) — five guards, five catches

| Mutation | Result |
|---|---|
| M1 drop the sink-side authorization re-check | 2 failed [CAUGHT] |
| M2 remove the HEAD re-resolve (reopen the TOCTOU) | 1 failed [CAUGHT] |
| M3 remove the TTL expiry check | 1 failed [CAUGHT] |
| M4 register the operator's own ts (self-approval) | 6 failed [CAUGHT] |
| M5 **silently amend an immutable criterion** | 1 failed [CAUGHT] |

M5 is what makes part (a) trustworthy: annotating instead of amending is only a real guarantee if a future silent amendment is detectable. Full transcript in `live_check_75.2.1.md`.

This criterion existed because phase-75.3 shipped three guards that could not fail. On a path that authorizes `git push origin main`, a test that passes whether or not the gate works is worse than no test.

## Operator notes

- **Slack bot restart required** for the request path to go live.
- **`slack_operator_user_id` must be set** or every request and every reaction is refused — the intended fail-closed posture.
- The surface is deliberately rare and dense: 2025 HITL literature names *confirmation fatigue* ("users stop reading the payloads and blindly click Approve") as the primary failure mode of approval flows, and an approval nobody reads records consent without its substance.
- **Not addressed here:** the pre-existing 10-member family of done steps whose verification commands name paths that never existed. That is a separate cleanup and would need its own step.

## Cycle-2 addendum (post Q/A wf_facb6070-53a CONDITIONAL)

The Q/A found four issues; the first means **cycle 1's fix did not work in production at all.**

1. **Criterion 3 was NOT MET — the path was still inert.** `AsyncSay` returns
   `AsyncSlackResponse`, which exposes `.get()` but is **not** a dict subclass
   (`issubclass(..., dict) -> False`, verified). My `isinstance(resp, dict)` guard was
   always False, so no ts was ever registered and the operator's checkmark would still
   have been refused — the exact defect 75.2.1 exists to remove. The suite could not see
   it because my `_say` stub returned a **plain dict**: the fixture diverged from
   production in precisely the load-bearing way. Fixed by duck-typing, with the stub
   replaced by a non-dict `_FakeSlackResponse` and a new test pinning that fidelity
   against the real `AsyncSlackResponse`.
2. **`head_sha` had a default** that let a caller silently opt out of the TOCTOU
   re-validation on a git-push authorization path. Now required keyword-only.
3. **A fourth vacuous guard** — the registration-order test asserted only
   `is not None`. Now asserts real indices.
4. **My documentation was inverted** on registration order (push is idx 0, not the token
   handler). Corrected in five places; the collision test now uses the production regex
   objects rather than local copies.

Mutation evidence for the new guards: M6 (restore the `isinstance(dict)` bug) → 6 failed
[CAUGHT]; M7 (register push after the catch-all) → 1 failed [CAUGHT].

Suite 22 → 24; all three affected suites: **104 passed**.

**The pattern worth naming.** Across 75.3 and 75.2.1 the same root cause has now
produced five findings: a guard that cannot fail. Three were tests asserting source
strings; the fourth was a test asserting a tautology; the fifth — this one — was a
*fixture* that could not represent the production type. Mutation testing caught the
first four. It did **not** catch this one, because I mutated the production code while
leaving the stub in place, and the stub was the thing that was wrong. The lesson that
generalizes: mutation-test the guard, and separately ask whether the *fixture* can
represent the failure at all.

## Cycle-3 addendum (post Q/A wf_236c0b88-77b CONDITIONAL)

The Q/A found the **fifth** vacuous guard — precisely where I had defended one.

`test_say_response_stub_is_not_a_dict` asserted only that upstream `AsyncSlackResponse`
is not a dict. Its body never referenced `_FakeSlackResponse` or the fixture, so it
could not fail when the stub regressed. The Q/A proved the consequence empirically
(M9): regress the stub to a plain dict **and** restore the production `isinstance`
bug together, and the suite goes fully **green** while the push path is inert. The
non-dict fixture is the single thing that makes M6 detectable, and nothing pinned it.

I defended that guard explicitly in the cycle-2 spawn prompt ("I argue it pins fixture
fidelity — challenge that"). The challenge was correct and my defense was wrong.

Fixed: `_FakeSlackResponse` is hoisted to module scope and
`test_say_stub_matches_the_production_response_shape` now calls the fixture's own `say`
and asserts on the object it actually returns. M9 now fails (1 failed, 23 passed) where
it previously passed green.

Also corrected: M8 (head_sha default) and M9 rows added to the live_check mutation
table as criterion 6 requires; the stale "22 passed" verification output refreshed to
the real 24; the fourth changed file (`test_phase_75_2_slack_control_plane.py`) and the
true commands.py line count disclosed in both change-surface blocks; and
`mda_cache.json` — dirty from before this step and liable to be swept in by the
`git add -A` auto-commit hook — explicitly excluded from this step's commit.

**Five findings, one root cause, and it kept moving.** A guard that cannot fail: first
as source-string scans (75.3 ×3), then as a tautology (`is not None`), then as a
*fixture* that could not represent the failure, and finally as a guard that *claimed*
to pin the fixture while asserting only a library fact. Mutation testing caught levels
1–2 and 4. It missed level 3 because I mutated production while leaving the broken stub
in place, and it missed level 5 because I never mutated the *test harness itself*. The
generalization: mutate the fixture too, not just the code under test.
