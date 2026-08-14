# Day report -- 2026-08-14

Session: `pyfinagent` main. Freeze 19:30 CEST.

---

## 0. Operator ask -- the OAuth token: CLOSED, and the instruction was WRONG

**The credential is DEAD.** The operator revoked at `claude.ai/settings/claude-code`
and re-authenticated. **That report was not accepted as evidence** --
anthropics/claude-code#43801 documents claude.ai revokes that left tokens
functional -- so the credential was probed directly.

```
negative control  synthetic invalid token    93 -> 401   probe CAN say dead
positive control  operator's live credential 108 -> 400   probe CAN say live
leaked, rejoined across the newline         108 -> 401   DEAD
```

Zero tokens spent (an invalid body means auth is evaluated first: live=400,
dead=401). The token never entered argv -- `handoff/audit/pre_tool_use_audit.jsonl`
is tracked and pushed, so a credential in a shell command would have re-published
the secret this incident is about.

**Two earlier "dead" answers were rejected before this one:**

1. The first 401 **coincided with the negative control**. Fake->401 and
   leaked->401 proves only that the probe emits 401, not that it discriminates.
   The positive control settles it, and was re-run **after** the operator's
   `/login` so the control credential was current.
2. Every early candidate was **<=92 chars** while a live token is **108** -- none
   could have been a whole token. 92 is a token cut in half by the newline it
   wraps across. `13 (stray prefix) + 79 + 29 = 121`, and `121 - 13 = 108`
   exactly. **That rejoin is what an attacker builds by deleting one line break**,
   so it was the candidate that had to be dead. It is.

**THE INSTRUCTION IN THE GOAL FILE AND THE INCIDENT RECORD WAS WRONG.** Both said
rotation is *"`claude /login`, or `claude setup-token`"*. **Neither revokes.**
`setup-token` is **mint-only** and leaves prior tokens valid for their full year;
`/login` refreshes a *different* credential (the Keychain one). There is no CLI
revoke -- anthropics/claude-code#57400 closed **as not planned**, #48373 as
duplicate. The only path is the **claude.ai web UI**. Following the old
instruction would have minted a fresh token and **left the leaked one live**.

Ask **#20**'s plist claim is now **STALE**: 0 of 12 `com.pyfinagent` plists carry
`CLAUDE_CODE_OAUTH_TOKEN` (12 found, 9 with `EnvironmentVariables`), 0 hits across
shell profiles, and away-ops authenticates via `HOME` -> Keychain. So revocation
could not break the scheduled jobs, and **no re-mint was needed**.

Asks **06-2 / 51-4 / #20** CLOSED -- one credential, one closure. Corrected in
`INCIDENT_2026-08-14_credential_exposure.md` §9 and `operator_asks_2026-08-11.md`.

**Still open, now cosmetic:** the dead string remains in git history and on the
fork. History rewriting stays operator-gated under 86.67 criterion 4.

---

## 1. Rail drop rate -- BOTH readings, as required

```
SESSION START   2026-07-13..2026-08-14   570 runs
  EXHAUSTED  44   7.7%
  RETRIED     1          <- the injected probe ONLY

SESSION END (see §5 for the final reading)
  EXHAUSTED  45   7.9%
  RETRIED     3          <- +2, on a WILD drop
```

**THE FIRST GENUINE DATAPOINT -- and it is not the good news it first looks like.**

Run **`wf_2e5ddb63-de9`** (my cycle-1 Q/A for 86.74) was a **real, non-injected
drop**, and the StructuredOutput retry **FIRED on it** -- the error is literally
`completed without calling StructuredOutput (after in-conversation nudge)`.
RETRIED went 1 -> 3.

**But EXHAUSTED also went 44 -> 45.** The retry fired and **did not recover**.
Both agents were nudged; both still failed to emit. So the honest reading is:

> **the retry mechanism reaches wild drops, but firing is not recovering.**

That is a materially different claim from "the retry works", and it is the first
evidence either way. **No rate is being called** -- still far under 20 post-fix
runs, exactly as the goal instructs.

Run id recorded, per the goal: **`wf_2e5ddb63-de9`**.

---

## 2. Step 86.74 (P0, live money) -- NOT CLOSED. Awaiting a verdict.

**Status: code complete, cycle-2 Q/A in flight (`wf_929b36e7-c8a`) at freeze.
The step remains `pending`. No verdict exists, so nothing was flipped.**

### The finding of the day: the falsy-zero fix did NOT fix DELL

After fixing `_extract_position_pct`, the test asserting *"a REJECT still buys"*
**still passed**. Two defects were being conflated:

| defect | what it is |
|---|---|
| **falsy-zero** | a **visible** `0.0` collapses to `None` via `if pct:` |
| **nesting** | with the flag OFF the verdict is **not visible at all** |

With `shape_fix` OFF the full-path judge nests under `risk_assessment["judge"]`,
so the resolver **correctly** returned ABSENT and the 10% default was legitimately
reached. **DELL was a nesting casualty.** Criterion 3 requires the default be
reachable only from a *genuinely absent* verdict, and a nested 0% verdict is
present -- so nested-first resolution is now **unconditional**.

**Had I stopped at the falsy-zero, this would have shipped a P0 "fix" that left
the reported incident live, with a green suite.**

### What is now true

Driven through the real `decide_trades`, DELL's exact input produces **no order in
both flag states**. The 10% default is reachable from **one function** instead of
four drifted seams (`:507` was flag-guarded; `:800`, `:853`, `:878` were
**unguarded under every flag state**). AST count of `or 10.0` sizing idioms: **4 -> 0**.

**Criterion 4's root cause was not where the step assumed.** `tasks/analysis.py`
does pass the three columns -- but that is the API path. The **autonomous loop**'s
`_persist_analysis` called `save_report` **without them at all**, while
`save_report` had accepted them the whole time. Hence 0 of 129 rows.

**The verdict was never lost** -- it sits in the JSON blob, and reading it there
confirms all six 2026-08-13 tickers match the incident's inferred table **6 of 6**.
DELL's `REJECT/0%` is now **directly measured**, retiring the elimination-based
attribution this step's own evidence depended on.

### The cycle-1 Q/A dropped -- and caught four defects in my work first

Write-first preserved both records. All four are fixed; the worst was mine:

1. **My mutation harness could certify itself.** Cells scored `killed = rc != 0`,
   and **pytest exits 5 when `-k` selects nothing** -- a typo'd selector would run
   zero assertions and score KILLED. Measured: bogus selector -> 0 tests, exit 5,
   old rule scores KILLED. Now the selector must be proven live first.
2. **Criterion 3's enumeration was false as written** -- three default-yielding
   families, not one, and an unrecognised state **overrode an explicit 0.0**.
   Unreachable in production (false claim, not live defect), now fails closed and
   is **derived by an exhaustive sweep** instead of asserted.
3. **The assert count `51` came from grep matching a comment** -- the probe
   matched its own documentation, the exact trap I had guarded against for
   `or 10.0`. AST is authoritative: **17 -> 55**.
4. **"Two adjacent failures" was hand-narrowed; the derived set is SEVEN**, two of
   them in the file I cited. All pre-existing (verified), so the substance holds,
   but the set was picked rather than derived.

---

## 3. What I could NOT verify -- stated plainly

1. **No verdict on 86.74.** Cycle 1 dropped; cycle 2 was still running at freeze.
   The step is `pending` and must not be read as done.
2. **The post-fix BQ persisted share** -- needs an autonomous cycle after the
   session-end restart. Proven at the unit seam only.
3. **33 of 34 historical BUYs are UNDETERMINED**, not clean. One confirmed
   inversion (DELL) is **not** an all-clear.
4. **Nothing was driven through the running backend or a browser.** pid 27945
   started 13:30:35 CEST and still holds **pre-fix** code -- committed is not in
   force.
5. **Why NTAP carries `risk_judge_position_pct=4.0` from 2026-07-31** while its
   analysis row persisted no verdict -- untraced.

---

## 4. Pending restart (batched to session end, per standing instruction)

`backend/services/portfolio_manager.py`, `autonomous_loop.py`,
`signal_attribution.py`, `agents/risk_debate.py` are **committed but NOT IN
FORCE**. Running pid **27945**, started **2026-08-14 13:30:35 CEST**, predates
both 86.74 commits.

**The restart was NOT performed**, because 86.74 has no verdict yet and restarting
would put ungraded trading-path code into the live process. That is a deliberate
hold, not an oversight.

## 5. Queued defects (drafted, not yet filed as steps)

Held in the session scratchpad rather than written into the masterplan mid-EVALUATE:

- **D1** two swap-path tests red at HEAD (now known to be **seven** across the tree)
- **D2** the 33 undetermined historical BUYs
- **D3** verify the persisted-verdict fix in BQ after the restart
- **D4** `_extract_position_pct`'s legacy shim still collapses UNPARSEABLE/ABSENT

## 6. Not touched, as instructed

86.81 was **not** re-opened or re-closed. 86.82 / 86.83 remain **parked** -- the
data has not spoken, and one wild-drop datapoint is not a rate. No flag promoted,
no `.env` written, no manual cycle, no metered spend. Paper only. The DELL
position was not liquidated or resized.
