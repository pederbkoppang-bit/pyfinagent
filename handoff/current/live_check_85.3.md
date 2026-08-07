# live_check evidence — step 85.3 (away-ops auth alarm unstuck)

Captured 2026-08-07 (cycle 175).

## BEFORE — the stuck state (last pre-fix records)

```
$ cat handoff/away_ops/auth_page_state.json
{"incident_open": true, "opened_at": "2026-07-10T05:30:11.289372+00:00", "detail": "401 in session_am_20260710T053005Z.json", "paged": true}

$ tail -1 handoff/away_ops/health.jsonl   # (pre-fix)
last record: ok=False auth_ok=false detail=401_in_session_am_20260710T053005Z.json ts=2026-08-07T14:05:18Z
```

(The trailing ok:false streak measured by the research gate: **473 consecutive records over 9.85 days** — 2026-07-28T17:09:48Z..2026-08-07T13:35:14Z — with auth the sole failing leg in every one; the 07-10 latch itself is 28 days old. `claude auth status` rc=0 throughout.)

## AFTER — criterion 8: a real invocation, the alarm goes quiet (verbatim)

```
$ bash scripts/away_ops/healthcheck.sh; echo "exit=$?"
exit=0

$ tail -1 handoff/away_ops/health.jsonl
{"ts":"2026-08-07T14:09:24Z","ok":true,"backend":"running:89530","frontend":"running:863","slack_bot":"running:658","api_health":200,"frontend_http":200,"kill_switch_paused":"true","cycle_age_h":18.2,"cycle_fresh_26h":"true","disk_free_gb":127,"adc_ok":true,"gh_ok":true,"auth_ok":"true","auth_detail":"stale_401_ignored_session_am_20260710T053005Z.json","auth_p1":false,"restarts_performed":0,"restart_failed":false,"restart_note":"","p1_raised":false,"log_rotated":false}
```

`ok:true` with `auth_ok:"true"` and the honest detail naming the ignored stale 401. **First healthy record after 473 consecutive false alarms.**

## Latch-clear proof — criterion 5's mechanism, live (verbatim)

```
$ cat handoff/away_ops/auth_page_state.json   # after the same run
{"incident_open": false, "cleared_at": "2026-08-07T14:09:27.735860+00:00", "cleared_by": "healthcheck_healthy"}
```

Written by the seam's `--apply` transition (`auth_state.py::apply_transition`, strict-healthy only) — the close path that existed at healthcheck.sh:154-166 but was unreachable; now reachable, single-writer, and it does NOT fire on probe errors (DD-5 fixed; test-proven).

## Real-incident drill — paging preserved (sanctioned 66.4 drill, verbatim)

```
$ HEALTHCHECK_TEST_AUTH_P1=1 bash scripts/away_ops/healthcheck.sh
AUTH_P1_TEST_DELIVERY=true
drill exit=0

$ cat handoff/away_ops/auth_page_state.json   # drill leaves the latch untouched
{"incident_open": false, "cleared_at": "2026-08-07T14:09:27.735860+00:00", "cleared_by": "healthcheck_healthy"}
```

A real Slack message (labelled "[DRILL 66.4] ... delivery test only") was delivered to the ops channel; no latch write, no auth_p1 in the JSON record — the drill-isolation contract held. The three real-detection legs are unmodified: fresh-401 (inside the window by definition), `auth_status_rc_nonzero` (active re-check, un-aged), and the session-side page.

## Scheduled-tick + session-resumption status at capture time

- `launchctl list` still shows `com.pyfinagent.away-watchdog ... 1` — the last-exit-status of the final PRE-fix scheduled run; the next 30-minute tick (≈14:35Z) runs the fixed script and flips it to 0. The drill run's record (`ts:2026-08-07T14:09:40Z, ok:true, detail:"ok"`) is already the second consecutive healthy record.
- Session resumption: `run_away_session.sh:143` gates the probe block on `incident_open`, which is now false — the next scheduled slot (22:00 local) writes a fresh `session_*.json` instead of `auth-dead-skip`. Beyond today's capture window; the MECHANISM is live and the gate value is proven above. NOTE (85.3.1): the probe predicate itself remains defective — the next REAL 401 would reproduce the 28-day outage until 85.3.1 lands; queued P1.


## Cycle-2 corrections (2026-08-07, after Q/A CONDITIONAL wf_9bdd4eb6-03d — prior text preserved, superseded here)

- **Figures re-derived by the Q/A and adopted**: the consecutive pre-fix ok:false streak is **474 records over 9.87 days** (2026-07-28T17:09:48Z..2026-08-07T14:05:18Z), and the pre-fix totals are **34 ok:true of 510 parseable** records (511 lines incl. the invalid line 37 → DD-2/85.3.2). "Auth was the sole failing leg in all 474" reproduces.
- **DD-5 attribution corrected**: DD-5 is fixed STRUCTURALLY (the clear moved inside the seam, unreachable from a probe error — the crash path never reaches `apply_transition`), and the strict-true guard is defence-in-depth whose 'unknown' arm cannot be reached via the CLI. The earlier "test-proven" phrasing was wrong at the time; it is now ALSO test-proven directly (`test_c5_dd5_apply_transition_guard_driven_directly` drives the function with 'unknown'/'false'/'true'; the Q/A's MQ-A mutant now dies).
- **Drill isolation RESTORED structurally**: the cycle-1 tree ran `--apply` unconditionally before the drill branch, so a drill COULD have written the latch (the earlier "drill leaves the latch untouched" capture was undistinguishing — the latch was already closed). Fixed: healthcheck.sh now withholds `--apply` in drill mode (`APPLY_FLAG` gated on HEALTHCHECK_TEST_AUTH_P1), so the in-file 62.5 comment is true again by construction.
- The re-page-suppression rule (a 401 older than cleared_at never re-opens) now has direct coverage (`test_after_clear_suppresses_repage_for_pre_clear_401`; the Q/A's MQ-C mutant dies).
- Post-fix suite: **13 passed**; matrix M1-M7 re-run whole: **7/7 killed**; lint gate over the derived scope: **"All checks passed!"**.

Naturally-scheduled tick evidence (re-derivable from the live file): every health.jsonl record after 14:09Z shows ok:true; launchctl's away-watchdog last-exit-status flips to 0 on its next scheduled tick of the FIXED script.
