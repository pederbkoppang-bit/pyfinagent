# Experiment Results -- phase-82.59

**Step:** 82.59 (P1) -- two production-wired Slack handler call sites bind
arguments that do not exist.
**Date:** 2026-08-06. **Cycle:** 2 (cycle-1 Q/A returned CONDITIONAL; the blocker was an inert guard of mine -- see §11).
**Contract:** `handoff/current/contract_82.59.md`
**Research brief:** `handoff/current/research_brief_82.59.md` (`gate_passed: true`,
audit-class, dry after 4 rounds / 3 dry, 6 sources read in full, 24 URLs, 14 files)

---

## 1. What changed

| File | Change | Lines |
|------|--------|-------|
| `backend/slack_bot/assistant_lifecycle.py` | both broken registration call sites fixed; `body` declared on listener 1 | +20 / -7 |
| `backend/tests/test_phase_82_59_assistant_lifecycle_binding.py` | new -- 8 tests | 358 (new) |

**No handler bodies were edited** -- §3 shows they were never wrong.

Figures from `git diff --numstat` / `wc -l` / `grep -c '^def test_'`, run as the
last action before writing this file.

## 2. Verbatim output of the immutable verification command

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_59_assistant_lifecycle_binding.py -q
  warnings.warn(
........                                                                 [100%]
8 passed in 0.10s
```

## 3. Which side drifted -- determined from history, not assumed

The step warned me not to assume the call site was wrong. Git blame settles it:

- Handler signatures: `0da5a907` (2026-04-05), **never edited since**.
- Registration block: `cb77f065` (2026-04-06).

One kwarg set -- `(body, client, say, set_status, logger)`, which is **exactly
`handle_user_message`'s correct signature** -- was copy-pasted onto all three
inner calls. So the fix is call-site only, and
`test_the_handler_signatures_were_not_edited_to_fit_the_call_sites` pins that:
it fails if any handler grows `**kwargs`, because loosening the handler would
absorb any wrong call forever and make this defect class permanently invisible.

## 4. Criterion 3 -- Bolt swallows it, verified against the installed package

The operator-visible impact rests entirely on this, so the gate read the
installed source rather than the docs:

- `AsyncioListenerRunner` catches `Exception` at `asyncio_runner.py:70` and `:119`
- routing to `AsyncDefaultListenerErrorHandler`, whose entire body is
  `logger.exception()` (`async_listener_error_handler.py:66-67`)
- and **ack fires BEFORE the listener** in the Socket-Mode branch (`:104-106`),
  so Slack always sees 200.

**So the symptom was a blank assistant panel -- no welcome message, no suggested
prompts -- plus one stderr line.** Never an error anyone would notice, which is
why it survived from April.

`test_bolt_swallows_listener_exceptions_verified_not_assumed` asserts this
against the live package, so if a Bolt upgrade ever changes it, the recorded
rationale fails loudly instead of going quietly stale.

**Bolt could not have caught this.** It never validates listener kwargs at
registration, and at invocation it is *lenient* -- an unknown listener kwarg
becomes `None` plus a warning (`kwargs_injection/async_utils.py:109-111`). The
`TypeError` comes from the plain Python call one frame deeper.

## 5. A SECOND defect the step does not mention

`on_thread_started` never declared `body`, and the inner call hardcoded
`body={}`. So `channel_id` and `thread_ts` were `None` **even if the call had
bound** -- the handler would log `channel=None, thread_ts=None`.

**This is why "completes without raising" is not sufficient**, and criterion 2's
guards assert the handler received the REAL payload: the fixture uses distinct
identifiers (`D82590CHAN`, `1786000000.000100`) and checks they reach the
handler's own log calls. Mutant M_C restores `body={}` and dies.

## 6. Criterion 4 -- all three listeners, and the count holds

Derived structurally (attribute-resolved on the handler class, **not** by bare
name) and re-derived by me at runtime with `inspect.signature().bind()`:

```
MISMATCH handle_thread_started: missing a required argument: 'set_suggested_prompts'
MISMATCH handle_context_changed: got an unexpected keyword argument 'client'
OK       handle_user_message(self, body, client, say, set_status, logger)
```

**Exactly 3 listeners, exactly 2 broken** -- the step's count is correct, and the
gate confirmed it two independent ways (AST walk + reading
`AsyncAssistant._*_listeners` off the live middleware). No fourth site.

**The sweep deliberately does NOT use a bare-name AST resolver.** The gate
measured that a name-based pass over `backend/slack_bot` produced 10 hits, **all
false positives** (`subprocess.run` colliding with local `jobs/*.run` defs). Its
rounds 2-4 found **zero** further instances across 22 files / 176 defs / 45
resolvable kwarg calls / 20 decorated listeners.

## 7. Mutation matrix -- 7 mutants, all killed

```
baseline: rc=0 GREEN

M_A revert BOTH sites to the shipped defect            DIED  (the exact pre-82.59 registration)
M_B thread_started call reverted                       DIED  (criterion 1 + the prompts assertion)
M_C body hardcoded back to {} (the SECOND defect)      DIED  (completing-without-raising is NOT sufficient)
M_D context_changed call reverted                      DIED  (criterion 1 on the second site)
M_E handler grows **kwargs (the tempting wrong fix)    DIED  (loosening would absorb ANY wrong call forever)
M_G bogus kwarg at the THIRD call site  [cycle 2]      DIED  (the Q/A's survivor -- see §11)
M_F listener swallows its own exception                DIED  (a second swallow layer on top of Bolt's)

=== 7 died, 0 survived ===
```

Licenses exactly "these 7 mutants died", not "no survivors". **M_A is the
headline** -- it restores the shipped registration verbatim, pulled from
`git show HEAD:`, and the suite goes red, so criterion 1's "FAILS against the
current code" is demonstrated rather than claimed. **M_G is the one I missed**
(§11); it is run under a `-k user_message` selector so its kill is attributed to
the third listener's own guard, not to criterion 1's bind test.

## 8. Guard traps avoided, each measured by the gate rather than guessed

- **Criterion 1 does not assert on the exception message.** The two instruments
  disagree for the same site: `inspect.bind()` reports *"missing a required
  argument: 'set_suggested_prompts'"* while the live call reports *"got an
  unexpected keyword argument 'client'"*. The guard asserts derived
  **missing/extra sets**.
- **No `assert response.status == 200`** -- vacuous, green against the broken
  code, because ack precedes the listener (§4).
- **Each listener gets its OWN payload.** The gate measured that a shared body
  lets `user_message` complete without raising because every field resolves to
  `None`.
- **Completion alone is not asserted** (§5).

## 9. Regression and lint

```
$ python -m pytest backend/tests/ -q -p no:randomly
31 failed, 2788 passed, 12 skipped, 5 xfailed, 1 xpassed in 343.17s
```

Whole suite, not a `-k` subset -- that shortcut is what hid two failures during
82.51.

**Provenance of the before-side, stated precisely because the Q/A could not
source it from a persisted artifact.** The comparison is against the full-suite
run I made while closing 82.51 earlier in this same session; its FAILED list was
captured to a scratchpad file and diffed line-by-line against this run's:

```
before (post-82.51 run): 31 FAILED lines
after  (this step):      31 FAILED lines
comm -23 (new failures):   (empty)
comm -13 (fixed):          (empty)
```

So "zero new, zero fixed" is a **set** comparison, not a count match. The
`2780 -> 2788` passed figure is arithmetic on those two runs (+8 = the tests
added here); the before-side run itself lives only in this session's scratchpad,
not in a committed artifact, and that is the honest limit of it. The 31 are the
same pre-existing + environment set documented in 82.51 §12.3.

**Lint:** `ruff --select F,E9` over a git-derived, asserted-non-empty 2-file
scope reports 11 errors in `assistant_lifecycle.py`. **All 11 are pre-existing,
compared as SETS and not merely counted** -- `comm` against the HEAD version of
the same file gives an empty diff in both directions:

```
HEAD set: 11 | current set: 11
=== introduced by me === (empty)
=== removed by me ===   (empty)
```

My new test file: `All checks passed!`.

Observed and NOT fixed (pre-existing, outside this step's diff). Composition
**derived** rather than described -- the cycle-1 artifact said "four unused
imports" and its itemization summed to 10 against a stated total of 11:

```
$ ruff check --select F,E9 --output-format=concise backend/slack_bot/assistant_lifecycle.py \
    | grep -oE '\b(F[0-9]+|E[0-9]+)\b' | sort | uniq -c
   5 F401
   2 F541
   4 F841
```

The four `F841` include `channel_id` / `thread_ts` in a later handler, which
may indicate dead logic -- but that is a different function and a different
step.

## 10. What I did NOT do

- **No handler-body edits** (§3) and no `**kwargs` loosening -- pinned by a test.
- **Kept the `thread_context_changed` registration**, which suppresses Bolt's
  built-in `default_thread_context_changed` (`async_assistant.py:255-256`). Our
  handler does real work (tracking the new channel context for the message
  handler), so the registration stays. Recorded because it is a real
  consequence.
- **No Slack message sent.** Every test registers against a stub app with no
  token and no network; `say` / `set_suggested_prompts` are local fakes.
- **Did not confirm the Slack app manifest has Agents & Assistants enabled.**
  The gate flagged this as an open gap: it decides whether the blank panel was
  *active* or *latent*, but not whether the code was wrong. The bot process is
  live under launchd.
- **Did not pin `slack-bolt`.** `requirements.txt:55` carries `>=1.18.0`, a
  floor rather than a pin (installed 1.27.0, published 1.30.0). The fix is
  version-robust; the unpinned floor is queued, not fixed here.

## 11. Cycle 2 -- I wrote the strict guard twice and the loose one once

The cycle-1 Q/A returned **CONDITIONAL** on one blocker, and it is a fair hit.

`test_user_message_listener_still_completes` replaced the subject with
`mock.patch.object(AssistantLifecycleHandler, "handle_user_message",
new=mock.AsyncMock())`. **An AsyncMock accepts any keyword argument**, so no
binding defect at that call site could ever turn the test red. The Q/A proved it
by injecting `bogus_kwarg=1` into the third call site and watching the test stay
green. Its assertions were `await_count == 1` plus a body-identity check --
**literally the "merely invoked" pattern criterion 2 rejects.**

Worse, its docstring said *"Included so a regression in the fix cannot break it
silently."* That claim was measurably false: M_G is exactly such a regression,
at exactly that site, and it did not break.

**The shape of the mistake:** I applied the strict reading of criterion 2 to two
listeners -- driving the real handler and asserting it saw the real payload --
and the loose one to the third, **without disclosing it in §8 where every other
trap is disclosed.** The third site was the one that already worked, so I
guarded it as an afterthought.

**Fixed by driving the REAL handler** and patching one seam deeper, at
`streaming_integration.handle_user_message_with_streaming` (imported
function-locally, so a module-attribute patch lands at call time). The test now
asserts completion rather than invocation: the stream seam was reached with the
real body, `set_status` was called on both sides (set, then cleared), and the
handler's own log line carries the payload's user id. A wrong kwarg is now a
real `TypeError`, and `handle_user_message` re-raises after clearing status, so
it surfaces. Docstring corrected to describe what the test actually does, and
the old shape recorded there so it is not reintroduced.

**M_G is now in the matrix** (§7) under a `-k user_message` selector, so its kill
is attributed to the third listener's own guard rather than to criterion 1's
bind test -- which did kill it, and is why this was WARN and not a coverage hole.

### Two claim defects, also fixed

- **§9 miscounted the pre-existing lint.** I wrote "four unused imports" against
  an actual 5 `F401`, and my itemization summed to 10 against a stated total of
  11. Replaced with a **derived** composition (`uniq -c` over the rule codes).
  The load-bearing claim -- that all 11 are pre-existing -- was already verified
  as a SET against HEAD, and the Q/A reproduced that independently.
- **§9's before-side had no persisted provenance.** The Q/A could not source the
  `2780 passed` / `31 both sides` baseline from any committed artifact. Now
  stated precisely: it is a full-suite run made earlier in this same session
  while closing 82.51, captured to a scratchpad file and diffed as a **set**
  (`comm` empty both directions) -- with the honest limit that the before-side
  run itself is not in a committed artifact.
