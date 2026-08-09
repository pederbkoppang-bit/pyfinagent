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

**IN-FORCE PROOF (the claim that matters):**

```
backend pid 6644 started : 2026-08-09 18:56:00
fix commit e98ca260      : 2026-08-09 17:31:45
started 5055s (84 min) AFTER the fix       -> IN FORCE
file md5                 : 58bbf24bde4c5161ac05f26f70fb264e
file mtime               : 2026-08-09 18:40:56   (< process start, so the
                                                  process imported THIS content)
```

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
