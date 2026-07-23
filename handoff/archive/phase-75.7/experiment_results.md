# Experiment Results -- masterplan step 75.7

**Step**: 75.7 -- Audit75 S7, Slack assistant streaming await-correctness + P0 pager integrity
**Cycle**: 3 (cycle-1 + cycle-2 CONDITIONAL on artifact accuracy; all fixed by measurement -- see §7; cycle-3 Q/A wf_568799ec-e34 PASS) | **Date**: 2026-07-23 | **Priority**: P0
**Contract**: `handoff/current/contract.md` | **Research**: `research_brief_75.7.md`
(gate `wf_cfc4bd83-679`, PASSED)

---

## 1. Verbatim verification command output

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_slack_streaming.py -q
12 passed
EXIT=0
```

Full generated transcript (py_compile x4, criterion scans, mutation matrix, regression) in
`handoff/current/live_check_75.7.md`.

## 2. What changed (measured)

### (a) pysvc-01 -- un-awaited streamer coroutines
`streaming_integration.py`: `import asyncio` hoisted to module top; removed
`from concurrent.futures import ThreadPoolExecutor, as_completed`; awaited **2**
`client.chat_stream` + **7** `streamer.append` + **2** `streamer.stop` calls. The
Python-list `.append` calls (`initial_chunks`/`progress_chunks`/`synthesis_parts`) use
distinct variable names, so the await-transform is unambiguous and did not touch them.
Before: `streamer.append(...)` on the coroutine returned by an un-awaited `chat_stream`
raised `AttributeError` into the broad `except` -- every non-DIRECT assistant message died.

### (b) pysvc-02 -- blocking fan-out
Replaced `ThreadPoolExecutor` + `concurrent.futures.as_completed`/`future.result()` (which
blocked the bot's single Socket-Mode loop) with
`tasks = [asyncio.create_task(asyncio.to_thread(_run_agent, at)) ...]` +
`for done in asyncio.as_completed(tasks): agent_type, result, err = await done`.
`_run_agent` returns a **3-tuple** `(agent_type, result, err)` so `await done` never raises
-- preserving both per-agent identity and per-agent error isolation (the gate's flagged
risk that a bare re-raising `await done` would lose the failing agent's identity and its
error card).

### (c) gap1-06 -- two blocking sync calls (NOT three)
`app_home.py`: `_get_live_data()` (3x httpx + 3x subprocess, ~41s worst case) ->
`await asyncio.to_thread(_get_live_data)`; `import asyncio` added (was absent).
`commands.py`: `_read_status()` -> `await asyncio.to_thread(_read_status)`.
**The reaction-handler git push was already `to_thread`-wrapped (phase-75.2.1) and was NOT
touched** -- the step text listed three calls but only two needed the fix (the gate caught
this over-count; re-wrapping the push would double-wrap).

### (d) gap1-02 -- P0 pager exit-code integrity
`settings.py`: new `escalation_phone_e164` field (default `+4794810537`, byte-identical to
the removed literal). `scheduler.py`: the pager now CAPTURES the `CompletedProcess`. On
`returncode != 0` it logs ERROR (not the success line) and posts a Slack fallback
(`P0 iMessage pager FAILED: ...`) so L1 records the L2 miss. **The exception path (imsg
binary missing / timeout) ALSO posts the fallback** -- the gate flagged that the criterion
only tests `returncode=1`, but a `FileNotFoundError` is an equally silent pager failure.
Empty phone -> ERROR + fallback + skip. Before: the return value was discarded and
`"iMessage escalation sent"` logged unconditionally -- a silent kill-switch pager failure
the operator never saw.

## 3. Files
| file | change |
|---|---|
| `backend/slack_bot/streaming_integration.py` | (a) awaits + import; (b) async fan-out |
| `backend/slack_bot/app_home.py` | (c) to_thread + import asyncio |
| `backend/slack_bot/commands.py` | (c) to_thread on `_read_status` |
| `backend/slack_bot/scheduler.py` | (d) pager exit-code + Slack fallback + settings phone |
| `backend/config/settings.py` | (d) `escalation_phone_e164` field |
| `backend/tests/test_phase_75_slack_streaming.py` | NEW -- 12 tests (11 + a cycle-2 fan-out error-isolation test) |

## 4. Mutation matrix -- 7/7 killed, 0 survived
```
M1 revert an awaited streamer.append to sync          -> KILLED (the completion assertion: an un-awaited append coroutine's body never runs, so kinds=['chat_stream','stop'] and `"append" in kinds` fails)
M2 restore ThreadPoolExecutor/as_completed fan-out    -> KILLED (AST assert)
M3a revert _get_live_data to bare sync                -> KILLED
M3b revert _read_status to bare sync                  -> KILLED
M4 pager ignores returncode (log 'sent' unconditional) -> KILLED
M5 restore a phone literal in scheduler.py            -> KILLED
M6 drop the exception-path Slack fallback             -> KILLED
```
Scoped claim: these 7 mutations were killed. Not "the suite has no vacuous guards".

**M1 kill-mechanism CORRECTED (cycle 2).** Cycle 1 credited M1's kill to the `error::RuntimeWarning` + `gc.collect()` leg. Measured: that leg is **inert** -- the un-awaited-coroutine RuntimeWarning fires at coroutine finalization and is swallowed as "Exception ignored while finalizing coroutine", so a warn-only harness does NOT raise. The REAL deterministic kills are: (append-revert) the completion assertion `"append" in kinds` (the coroutine body never records the call); (chat_stream-revert) the AttributeError from calling `.append()` on the un-awaited coroutine object. The RuntimeWarning filter is retained as harmless belt-and-suspenders but is **non-load-bearing** -- do NOT remove the completion assertions on the assumption the filter guards this.

**A comment-token trap I hit and fixed** (same class as 75.5/75.6): criterion 2's first
draft was a substring scan over the function source, which false-failed because my fan-out
COMMENT documents the old `as_completed(futures)`/`future.result()` code. Rewrote it as a
structural AST check over actual Call nodes -- immune to comments, and what "AST assert" in
the criterion actually asks for.

## 5. Regression + honesty
- **Lint** (`ruff --select F821,F401,F811` over the changed **and untracked** `.py` files,
  space-split -- see the cycle-2 correction below): **All checks passed** over the full
  change surface INCLUDING the new test file. 0 introduced in production; the one F401 the
  cycle-1 command missed (an unused `QueryComplexity` import in the test) was removed.
- **Full backend suite** (re-measured cycle 2): 10 failed / **1305 passed** (+12 mine). The 10 are the standing
  live-environment red set; **none of the failing files reference the 75.7 change surface**
  (verified by grep). 0 regressions.
- **Test tooling**: pytest-asyncio is absent; used `asyncio.run()` in sync defs (repo
  precedent). Criterion 1 is anchored on the DETERMINISTIC `AttributeError` (the RuntimeWarning
  fires at GC time and is non-deterministic); the warning filter is module-scoped with a
  forced `gc.collect()`, never a global `error::RuntimeWarning` (would break the suite).
- **Not verified live**: no Slack bot run, no real iMessage sent (the pager is exercised
  offline with `subprocess.run` stubbed). A bot restart is needed for the running process
  to pick these up. No UI surface.

## 7. Cycle-2 record -- the two blockers the Q/A caught (both mine, both reporting)

The cycle-1 Q/A (`wf_8b63c4cd-b25`) returned **CONDITIONAL** and confirmed the CODE is
correct on all 6 immutable criteria (it reproduced M1/M2/M4/M6 kills + the fan-out
isolation itself). Two blockers, both in my artifacts, both fixed with **no production
change**:

**Blocker 1 -- the lint gate false-passed (instance-2 trap, in the module I fixed it in).**
My live_check ran `uvx ruff ... $(git diff --name-only HEAD -- '*.py')` UNQUOTED. This is
zsh, which does NOT word-split an unquoted expansion, so ruff received the newline-joined
blob as ONE path, linted **zero files**, and printed "All checks passed!" exit 0 -- the
exact false-pass qa.md sec1a/sec4b (which I authored last cycle) documents. The scope also
excluded the untracked new test file. Run correctly (`git status --porcelain`-derived,
space-split, includes untracked), ruff exits 1 on a real F401: an unused `QueryComplexity`
import at `test_phase_75_slack_streaming.py:63`. **Fixed**: removed the import; re-ran ruff
over the correct scope -> genuinely clean; the live_check generator now uses the correct
command. That I committed this in the very step-type whose fix I shipped last cycle is the
point -- mechanism helps the Q/A, but the author still has to apply it.

**Blocker 2 -- M1 kill-mechanism misattributed.** Corrected in sec4 above (the completion
assertion, not the inert RuntimeWarning leg). I verified it by measurement: reverting
`await streamer.append` yields `kinds=['chat_stream','stop']` -> the `"append" in kinds`
assertion fails.

**Also fixed** (non-blocking Q/A notes): the phone-literal miscount (sec6 + contract sec5: 4
occurrences across 2 files, not "3 literals"); the 75.7.1 masterplan file paths (they are
under `backend/services/`, not `backend/slack_bot/`); and a NEW test protecting the fan-out
per-agent error-isolation claim (`test_fanout_one_agent_error_does_not_abort_the_others`)
-- 11 -> 12 tests.

**Scope of cycle-2 changes (stated to what is verifiable, not an unprovable absolute):** the
cycle-2 EDITS are to the test file + these handoff docs + the 75.7.1/contract prose. The 5
production files' CONTENT is unchanged in cycle 2 -- their current content meets all 6
immutable criteria (re-verified) and the mutation matrix is 7/7. NOTE their mtimes were
bumped by the mutation harness (which `copy2`-restores each file after each mutant), so mtime
is NOT evidence of a content change; there is no committed cycle-1 baseline to byte-diff
against (75.7 is not yet flipped), so I do not assert byte-identity -- I assert the current
content is correct and criteria-covered.

---

## 6. Out-of-scope -> queue as its own step
Per `feedback_queue_discovered_defects_in_masterplan`: the gate found the literal `+4794810537` at **4 occurrences across 2 files** (`backend/services/sla_monitor.py:20`, `backend/services/queue_notification.py:34/63/164`) that should also
resolve from `settings.escalation_phone_e164`. NOT folded into 75.7 -- queue as **75.7.1**.
