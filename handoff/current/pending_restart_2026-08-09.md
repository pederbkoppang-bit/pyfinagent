# Backend restarts -- 2026-08-09  [RESOLVED]

**Nothing is owed. Both restarts were done in-session and verified against the
RUNNING process.** This file is kept as the record.

## Restart 2 -- 36.17 exit-only stop-loss pass  [DONE 16:56:00Z]

Done promptly after the fix rather than batched to session end, on the
operator's instruction of 2026-08-09. Rationale, measured rather than assumed:

* The batching rule was written after a `bootout`+`bootstrap` race left the
  backend down ~4 minutes. **That was the `bootout` verb, not restarting per
  se.** `kickstart -k` restarts the process AND re-reads `backend/.env`; it does
  not re-read plist `EnvironmentVariables`, which nothing here needed.
* **Restarting while present is safer than restarting and walking away** -- the
  end-of-session pattern is what produced an outage nobody noticed for minutes.
* The only remaining work this session (86.17) touches `.claude/workflows/*.js`
  and `scripts/qa/`, which the backend never imports -- so there was no second
  backend-affecting change to batch with.
* Until this restart the defect was LIVE: a halted cycle would still skip
  stop-loss enforcement.

**Preconditions checked before restarting:**
* `handoff/.autonomous_loop.lock` -> `state: released`, and its pid was DEAD and
  was NOT the backend pid. Lifetime 1.7s -- a pytest run, not a trading cycle
  (see the disclosure below).
* No cycle-step activity in `backend.log`.
* The in-flight Q/A was allowed to land first, so a restart could not fail its
  health probe and manufacture a spurious finding.

**Result:**

```
old backend pid : 84494
new backend pid : 6644          started 2026-08-09 18:56:00 local
health          : ok            addr-in-use hits: 0
```

**IN-FORCE PROOF (the claim that matters).** CORRECTED after the cycle-5 Q/A:
the original version rested on `file mtime < process start`, and **that line no
longer reproduces and now reads the wrong way** -- my own later mutation run
rewrote the file with identical content, bumping mtime to 19:14:49, which is
AFTER the 18:56:00 process start. The `.pyc` was recompiled too, so the
import-time mtime evidence is destroyed and is not re-derivable by anyone.

The conclusion is unchanged and the Q/A confirmed it independently, but it rests
on **content-last-changed**, which is durable, rather than on mtime, which is not:

```
$ git log -1 --format="%h %cd" -- backend/services/autonomous_loop.py
6ca17793 Sun Aug 9 17:54:37 2026 +0200      <- content last CHANGED here

$ ps -p 6644 -o lstart=
søn.  9 aug. 18.56.00 2026                   <- process started 61 min LATER

$ git status --porcelain backend/services/autonomous_loop.py
                                             <- empty: tree == that commit
$ md5 -q backend/services/autonomous_loop.py
58bbf24bde4c5161ac05f26f70fb264e             <- unchanged since cycle 2
```

Content last changed **17:54:37** < process start **18:56:00**, and the tree is
clean at the same md5 the Q/A verified across cycles 2-5. **Therefore the running
process imported this content: the fix IS in force.** mtime is deliberately not
cited -- a same-content rewrite invalidates it while changing nothing that
matters.

State survived the restart:

```
paused: False | armed: True | sod_date: 2026-08-09
sod_nav: 23833.94 | peak_nav: 24666.57 | trailing_dd: 3.3755
paper_cycle_max_seconds (RUNNING): 10800.0
```

## Restart 1 -- PAPER_CYCLE_MAX_SECONDS 7200 -> 10800  [DONE 15:08Z]

`backend/.env` change. `kickstart -k` at 15:08Z, pid 24708 -> 84494. Verified by
reading `10800.0` from `/api/settings/` on the RUNNING backend, not from a fresh
interpreter.

Note `scripts/ops/reissue_cc_oauth_token.sh --verify` **cannot pass any more**:
it raises `KeyError: 'CLAUDE_CODE_OAUTH_TOKEN'` because neither plist carries
that key now. The rail authenticates from the macOS Keychain
(`Claude Code-credentials`). That is a BETTER state -- it is what 62.1.1 /
85.3.3 asked for -- but the script is stale against it.

## DISCLOSURE -- a test run wrote the LIVE cycle-lock file

While checking restart preconditions I found `handoff/.autonomous_loop.lock`
holding `cycle-1786294253`, released 16:50:54Z, **lifetime 1.7 seconds**, pid
5851 (dead, and not the backend). That is a `pytest` run -- mine or the Q/A's --
writing live cycle-lock state, which is exactly the filesystem channel step
**86.6** exists to close.

**Why my isolation checks missed it:** the file is **untracked/gitignored**, and
my before/after md5 checks covered only the three GIT-TRACKED files
(`kill_switch_audit.jsonl`, `cycle_history.jsonl`, `.cycle_heartbeat.json`).
86.6's before/after digest set must span the whole live-state set, not just the
tracked part.

**Practical warning:** while tests run, this lockfile is NOT a reliable "is a
cycle running" signal. Check the pid is alive AND equals the backend pid, and
check the lifetime -- a ~1-2 second lifetime is a test, never a cycle.

## SECOND DISCLOSURE -- I ran mutations against the live file while the backend was ARMED

The cycle-5 Q/A caught what `experiment_results` §12f did not say. That section
bounded the blast radius of the SIGTERM incident against the OLD pid 84494, and
gave no equivalent for what came after: **the cycle-4/5 mutation runs mutated
`backend/services/autonomous_loop.py` nine times while pid 6644 -- an ARMED
backend, `paused: false` -- was running.**

Why no harm occurred, stated as mechanism rather than luck: CPython imports a
module once and serves it from `sys.modules`, so a running process does not
re-read the file. Every mutation was restored within seconds and the file is
byte-identical at `58bbf24b...`.

**Why it was still a bad practice, and the rule for next time:** had the backend
been restarted -- by me, by the launchd watchdog, or by a crash -- during any of
those windows, it would have imported a MUTANT into an armed trading process. The
mutation harness must run against a **copy** of the module, or via the in-memory
`sys.modules` injection the Q/A itself uses (which writes nothing to the repo),
never against the live file while a trading process is armed.
