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

## 7. The LOUD missing-creds log — verbatim (cycle-2 addition)

The cycle-1 Q/A found this was required by `verification.live_check` and was **not** in
this file — and, more importantly, that the error did not fire at **startup** at all
(criterion 3's actual wording). Both are fixed. Reproduced live, `EXECUTION_BACKEND=alpaca_paper`
with both credentials unset, calling only the startup hook:

```
INFO:backend.services.execution_router:phase-68.1 execution backend: mode=alpaca_paper source=env (paper-only enforced; default=bq_sim)
ERROR:backend.services.execution_router:phase-68.1: EXECUTION_BACKEND=alpaca_paper but Alpaca credentials are MISSING (ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY). Falling back to deterministic MOCK fills (source=mock_alpaca, fixed 30bps slippage) -- these are NOT real Alpaca paper orders. Set the named variables or set EXECUTION_BACKEND=bq_sim to make the intent explicit.
```

One ERROR, at startup, naming **both** variables, and stating plainly that the fills are
not real. Previously this appeared only after the first `submit_order()` — so an operator
who mis-set the mode learned at the first trade of the day instead of at configuration
time. That was the cycle-1 miss.

Pinned in **both** directions, so it can neither disappear nor become boot-spam:

| Mutant | Result |
|---|---|
| E8 — remove the startup check | **killed** (`test_missing_creds_error_fires_at_STARTUP_not_only_at_first_order`) |
| E9 — fire it unconditionally | **killed** (`test_startup_is_silent_when_mode_is_bq_sim`, `test_startup_is_silent_when_alpaca_creds_are_present`) |

## 8. Triple-enforcement test output — per test (cycle-2 addition)

Criterion 4's six tests, run with `-v` rather than reported as an aggregate count:

```
test_paper_base_url_is_pinned_to_the_paper_domain PASSED           [  9%]
test_repo_never_overrides_the_alpaca_base_url PASSED               [ 18%]
test_live_marked_key_prefix_is_refused[PKLIVEABC123] PASSED        [ 27%]
test_live_marked_key_prefix_is_refused[pklive_lower] PASSED        [ 36%]
test_live_marked_key_prefix_is_refused[AKLIVEXYZ] PASSED           [ 45%]
test_paper_trade_false_is_refused PASSED                           [ 54%]
test_ordinary_paper_config_is_allowed[true] PASSED                 [ 63%]
test_ordinary_paper_config_is_allowed[TRUE] PASSED                 [ 72%]
test_ordinary_paper_config_is_allowed[True] PASSED                 [ 81%]
test_ordinary_paper_config_is_allowed[None] PASSED                 [ 90%]
test_every_fill_path_reports_paper_true PASSED                     [100%]
====================== 11 passed, 33 deselected in 0.20s =======================
```

(a) base URL pinned to `paper-api.alpaca.markets` and `url_override` never passed;
(b) live-marked prefixes refused — including a lowercase case, and with
`test_ordinary_paper_config_is_allowed[4 cases]` proving the refusal is not a blanket
denial that would pass vacuously; (c) no mode yields a non-paper fill.

**Honesty note carried from the research gate, not softened:** the (b) prefix filter is
*not* Alpaca's real paper/live discriminator — three official sources read in full
document no format difference between paper and live keys; the environments separate by
**domain**. The filter is implemented because the immutable criterion names it, and it is
labelled belt-and-braces in code, docstring and test. (a) is the load-bearing guard.

## 9. Remaining criteria — where their evidence lives

Criteria 2 and 5 are test-borne, not live-borne, and are recorded in
`experiment_results_68.1.md` §3–§5: **44** tests via the immutable command (exit 0), the
mutation matrix, and the ruff gate.
