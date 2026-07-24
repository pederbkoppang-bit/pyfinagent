# Contract — Step 75.2.1: close the two operator-escalated items from 75.2

- **Step id:** 75.2.1 (phase-75, P1, executor: opus-4.8/xhigh)
- **Date:** 2026-07-20 · **Operator directive:** "fix them both after your recommendation"
- **Boundary:** control-plane only, $0 metered, **no immutable verification criteria amended**.

## Research-gate summary

Gate PASSED (moderate tier, `handoff/current/research_brief_75.2.1.md`, 389 lines): 8 sources read in full (OWASP Transaction Authorization Cheat Sheet, Slack Bolt/API docs, CPython asyncio, Azure Well-Architected ADR doctrine, GHSA advisory DB, 2025-26 HITL literature), 27 URLs, 3-variant discipline, recency scan with 4 findings, 12 internal files line-anchored.

The brief changed the plan in five material ways, all adopted:

1. **The sweep is complete and the scope is bigger than stated.** All 837 steps across 99 phases were swept, handling every verification shape (674 dict, 126 str, 13 list, 24 None — a naive `.get("command")` sweep *crashes* on the list-shaped ones). The three known collisions **are** the complete set for the 75.2 cause. But 4.17.9 turns out to be one member of a **pre-existing 10-member family**: 15 done steps have verification commands referencing absent paths, ten of them phase-29 go_live_drills steps whose plan names never matched the on-disk `smoke_test_4_17_N.py` convention. That is unrelated to 75.2 and must be disclosed, not quietly folded in.
2. **`f55e6973` deleted SEVEN files, not six** — the six modules plus `scripts/go_live_drills/smoke_test_4_17_9.py`. For 4.17.9 this matters: `git log --all --diff-filter=A -- scripts/go_live_drills/self_update_audit_test.py` is **empty** (that script never existed), while `smoke_test_4_17_9.py` **did** exist (added `1122a021`, deleted `f55e6973`). So 4.17.9 had a name mismatch from day one **and** 75.2 removed its plausible real target. An annotation naming only one of those misattributes the breakage.
3. **A house convention already exists — do not invent a key.** A census over 837 steps found `superseded_by` (6), `superseded_record` (1), `dropped_reason` (5), `deferral_audit` (10), `notes` (191), `audit_basis` (303). The closest prior art is `superseded_record` on step 68.5. 75.2.1 is its mirror image: `verification.*` stays byte-identical and the sibling records collision *facts*, never copies of the criteria.
4. **Status stays `done`.** Per the append-only ADR doctrine the brief cites verbatim — *"Don't go back and edit accepted records"* — the work **was** done; the artifact is gone, not the history.
5. **Part (b) has a real design gap I had not planned for (OWASP 2.6/2.8).** `_pending_push_ts` is a set of bare strings: it records *that* an approval was requested but not *what was shown*. A commit landing between request and reaction is pushed **without ever having been displayed** — textbook TOCTOU. The fix is to bind the approval to the commit set, not just the ts.

## Hypothesis

Recording the collision in a sibling field (never touching `verification.*`) makes a silent landmine auditable without rewriting history; and binding the push approval to `(ts, HEAD sha, expiry)` rather than a bare ts closes the TOCTOU that would otherwise let an unreviewed commit ride an approved reaction.

## Immutable success criteria (verbatim from .claude/masterplan.json step 75.2.1)

1. The verification.command and verification.success_criteria of steps 4.14.4, 4.14.24 and 4.17.9 are BYTE-IDENTICAL to their values at commit 256867d3 (test asserts this against git show); each step gains a sibling annotation field recording the retiring commit, and 4.17.9's annotation states its script was already missing pre-75.2
2. A non-operator request, and a request when slack_operator_user_id is unset, both post NO approval message and register NO ts (fail-closed); test drives the real handler
3. An operator request posts into _APPROVAL_CHANNEL, the posted message includes the pending-commit summary derived from git log origin/main..HEAD, and the posted ts is registered in _pending_push_ts so a subsequent operator reaction on THAT ts performs the push
4. A reaction on a ts that was never registered still performs no push, and an approved ts is single-use (second reaction pushes nothing) -- the 75.2 guarantees are preserved, proven by test not by inspection
5. Every git invocation on the request path runs via asyncio.to_thread (test spies on the dispatch); no subprocess call blocks the Socket-Mode event loop
6. Each new behavioral guard is mutation-tested and the mutation evidence is recorded verbatim in live_check_75.2.1.md -- a guard that cannot fail when its subject is broken does not count

## Plan steps

### (a) Collision record — annotate, never amend

- Add a `superseded_record` sibling to 4.14.4, 4.14.24, 4.17.9 (house convention, mirroring step 68.5). Fields: `retired_by_commit: f55e6973`, `retired_at`, `reason`, `still_runnable: false`, `note`. **`verification.*` is not touched.**
- 4.17.9's record states **both** facts: its command names a script that never existed on disk (name mismatch from day one, `--diff-filter=A` empty), **and** 75.2 separately deleted `smoke_test_4_17_9.py`, its plausible real target.
- Disclose the pre-existing 10-member family in the record + experiment_results so 4.17.9 is not mistaken for a 75.2 casualty.
- Run `scripts/meta/preflight_verify_masterplan.py` before and after (the `$schema` is the bare string `masterplan-v1`, so no validator enforces `additionalProperties`, but the preflight is the project's own gate).

### (b) Push approval — wire it, and close the TOCTOU

- **Trigger:** `@app.message` (a slash command would need App-Management config, out of bounds for a $0/control-plane step). **Colon-less bare `PUSH`**, anchored — `PUSH REQUEST: main` *matches* `_TOKEN_KEYWORD`, so an unanchored trigger would make the two handlers ambiguous. **[CORRECTION, cycle 2: I wrote here and in four other places that the token handler registers first and would swallow the request. It does not — the push handler registers at index 0, BEFORE the token handler and the catch-all. The conclusion stands because the two regexes are disjoint, but the mechanism was backwards, and the real hazard runs the other way: widening the push pattern would make it swallow operator TOKENS. Caught by Q/A wf_facb6070-53a.]** Pinned by a test against the production regex objects.
- **Registration order:** alongside the operator-token handler, **before** the catch-all `@app.message("")` at `:252` (dispatch is first-match-wins).
- **Authorize with the existing predicate:** reuse `operator_tokens._authorized(...)` at the matcher *and* re-check at the sink — the docstring's own rationale ("a matcher is a capability gate, not an authorization decision"), and the documented countermeasure to hermes-agent #36848 (unset env var skipping the auth block = fail-open) and GHSA-wv26-j37q-2g7p (CWE-863 approval-scope confusion, 2026-05-12).
- **Payload off the loop:** `git log origin/main..HEAD --oneline` via `asyncio.to_thread`, reusing the existing shape at `commands.py:72-78`. Empty list → say so, register nothing.
- **Show what is signed (OWASP 1.1):** commit list + count + target ref + resolved HEAD sha, posted into `_APPROVAL_CHANNEL` specifically — the reaction gate only accepts that channel, so a request posted elsewhere is un-approvable by construction.
- **Register the BOT's posted ts** (`say()` returns the `chat.postMessage` response). Registering the operator's own ts would be self-approval.
- **Bind to the commit set + TTL (OWASP 2.6/2.8/2.9):** `_pending_push_ts` becomes `dict[str, tuple[str, float]]` mapping ts → (HEAD sha, expiry). At push time re-resolve HEAD and **refuse if it moved**; reject expired approvals. `dict` keeps `in` and `.clear()` semantics so the 75.2 suite still holds.
- **Staleness honesty:** `origin/main` is a *local* ref. No fetch is performed (it would lengthen the TOCTOU window and add network to a $0 step), so the message says the comparison is against the **last-known** `origin/main` rather than implying freshness.
- **Preserve 75.2 order exactly:** identity → channel → ts membership → `discard()` *before* the push (a crashed push must not leave a re-approvable ts).
- **Confirmation fatigue (recency finding):** 2025 HITL literature names it the primary failure mode — "users stop reading the payloads and blindly click Approve". Keep this surface rare and the message dense.

### Tests

- New `backend/tests/test_phase_75_2_1_push_approval.py`.
- **Test-shape blocker:** the `_App` stub at `test_phase_75_2_slack_control_plane.py:62-77` captures only `event` handlers; `message()` returns `lambda fn: fn` and **discards** the function. The stub must be extended to capture message handlers + their `matchers=` kwargs or the new tests cannot invoke the path.
- Every new behavioral guard **mutation-tested**, evidence verbatim in the live_check (criterion 6). Three times in 75.3 I shipped guards that could not fail; on a `git push` authorization path that is unacceptable.

## References

- `handoff/current/research_brief_75.2.1.md` — 8 sources, full 837-step sweep, key census, OWASP mapping, verified regex collision, test-shape blocker
- `handoff/current/evaluator_critique_75.2.md` — the two escalated items
- 75.2 commits `f55e6973` / `800b3c6b`; 75.3 commit `256867d3` (byte-identity baseline)
