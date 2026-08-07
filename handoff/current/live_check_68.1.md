# live_check — masterplan step 68.1 (EXECUTION_BACKEND reaches execution_router)

**Captured 2026-08-07 23:01 CEST** · Cycle 180 · `handoff/current/experiment_results_68.1.md`

Criterion 1 requires "the startup log line printing BOTH the resolved mode AND its source
— env/.env/default — from the real launchd process". Captured **twice**: once with the
variable set, once with it unset, so the artifact proves the *set* value reaches the
router rather than merely that the logging exists.

---

## 1. The two provenance lines, verbatim from `backend.log`

Both emitted by the launchd-managed `com.pyfinagent.backend` process (not a shell run):

```
{"timestamp": "2026-08-07 23:01:14,179", "level": "INFO", "module": "execution_router", "message": "phase-68.1 execution backend: mode=bq_sim source=env (paper-only enforced; default=bq_sim)"}
{"timestamp": "2026-08-07 23:01:53,918", "level": "INFO", "module": "execution_router", "message": "phase-68.1 execution backend: mode=bq_sim source=default (paper-only enforced; default=bq_sim)"}
```

- **23:01:14 — `source=env`**: `EXECUTION_BACKEND` was set in the launchd session and the
  restarted process resolved it *through the router*. This is the criterion's substance:
  a set value demonstrably reaches `execution_router`.
- **23:01:53 — `source=default`**: variable removed, steady state restored. `mode` is
  `bq_sim` in both lines, so **no behaviour changed at any point** (criterion 2's DARK
  guarantee, observed live rather than only asserted in tests).

## 2. launchd process transitions

```
BEFORE            state = running   runs = 6   pid = 89530
after phase 1     state = running   runs = 7   pid = 19899
after phase 2     state = running   runs = 8   pid = 20004
```

Two clean restarts; the agent stayed `running` throughout and `runs` incremented as
expected.

## 3. Health after each restart

```
$ curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/api/paper-trading/portfolio
phase 1: 200
phase 2: 200
```

## 4. No residue left behind

```
$ launchctl getenv EXECUTION_BACKEND
(empty)
```

The variable is gone from the launchd session.

---

## 5. Method — and why it is not the obvious one

**`launchctl setenv` + `kickstart -k`, NOT a plist edit.** Two reasons, both of which
would have produced a wrong or damaging result:

1. **`kickstart -k` does not re-read plist edits.** Had I added `EXECUTION_BACKEND` to
   `com.pyfinagent.backend.plist` and kickstarted, the process would have printed
   `source=default` and I would have read a correct implementation as a broken one. The
   correct verb for a plist change is `bootout`/`bootstrap`, which is blocked by
   `.claude/hooks/pre-tool-use-danger.sh` — so the plist route was both misleading and
   unavailable. (Flagged by the 68.1 research gate.)
2. **A plist entry would have left a permanent footgun.** `env` outranks `dotenv` by
   design, so a value pinned in the plist would silently mask any future
   `backend/.env` setting — replacing one silent-config bug with another. `setenv` is
   session-scoped and reverses with `unsetenv`, which is why phase 2 is not optional.

**Value chosen deliberately:** `bq_sim`, the existing default, so the demonstration could
not alter execution behaviour. `mode=bq_sim` in both lines is the evidence of that.

**Nothing was written to `backend/.env` or to any plist.** Confirmed absent from `.env`
(0 of 85 lines) before and after.

## 6. Timing — sequenced around other live evidence

Deferred twice, on purpose:

- **Until the autonomous cycle finished.** `handoff/.autonomous_loop.lock` held pid 89530
  (the live backend) for a cycle started 20:00 CEST. Restarting would have killed a
  trading cycle mid-flight. The cycle terminated on its own at 22:00:02 CEST
  (`status=timeout`, see step 85.4), releasing the lock.
- **Until after the 23:00 CEST Slack digest.** That digest is criterion-3 evidence for
  step 62.1 and calls this backend; a connection-refused would have raised a P1 page and
  destroyed the other step's only evidence window (it does not recur until Monday). The
  digest was confirmed sent at 23:00:08 before the first restart at 23:01:14.

## 7. Remaining criteria — where their evidence lives

Criteria 2–5 are test-borne, not live-borne, and are recorded in
`experiment_results_68.1.md` §3–§5: 41 tests via the immutable command (exit 0), a 7/7
mutation matrix, the ruff gate, and the LOUD missing-creds ERROR log with its
mutation-guard proving it stays silent when credentials are present.
