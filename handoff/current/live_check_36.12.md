# live_check — phase-36.12

**Required (masterplan, verbatim):** *Verbatim test output for the reproduce-then-fix pair, plus a
curl of `/api/paper-trading/kill-switch` on a rig you own showing the state after a simulated
post-rotation cycle -- demonstrating it no longer reports ARMED-and-healthy with a forgiven
drawdown.*

## §A. Method, and what was NOT touched

An isolated FastAPI app on **`:8002`** (`rig_36_12.py`, held in the session scratchpad), serving the
REAL `backend.api.paper_trading` router with a stub BQ client. Before the kill-switch state object
is built, `kill_switch._AUDIT_PATH` is repointed at a fresh tmp tree seeded with the exact
2026-07-26 post-rotation shape — pause/resume rows survived, **zero** baseline rows — and
`_audit_archive_dir()` is asserted to follow it. One simulated trading cycle runs in-process, then
the endpoint is curled.

- Operator's `:8000` — **never restarted, never POSTed to**. Confirmed still `200` and still launchd
  pid **`76381`** at teardown. *(Measured with `launchctl print gui/$(id -u)/com.pyfinagent.backend`.
  An earlier reading in this session used `lsof -ti tcp:8000 | head -1`, which returned `20582` — a
  WebKit networking process holding a connection, not the server. The `lsof` figure was wrong and
  the backend was never restarted.)*
- Operator's `:3000` — never driven; `302` at teardown.
- `handoff/kill_switch_audit.jsonl` — md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` before and after both
  rig runs. The rig only ever opens its tmp tree.
- `:8002` listeners at teardown: **0**.

**Disclosed side effect of the rig:** each rig run delivered one real P1 Slack alert. See
`experiment_results_36.12.md` §"OPERATOR: I SENT YOU 17 FALSE P1 SLACK ALERTS TODAY" — measured
count, mechanism, and the fixture that stops it for the test suite. The rig itself is a scratchpad
script and is not re-run.

## §B. Reproduce — the defect, on the pre-fix code

```
check_and_enforce_kill_switch() returned:
{ "triggered": false,
  "breach": { "any_breached": false, "daily_loss_pct": 0.0, "trailing_dd_pct": 0.0,
              "daily_baseline_missing": false, "trailing_baseline_missing": false,
              "armed": true },
  "auto_resume": {"action": "no_op", ...} }

AFTER: sod_nav 18000.0   peak_nav 18000.0        (true historical peak was 24666.57)
rows:  {"event": "peak_update", "nav": 18000.0}
       {"event": "sod_snapshot", "nav": 18000.0, "date": "2026-07-26"}

VERDICT KEYS: triggered=False  any_breached=False  armed=True  blocked='<key absent>'
```

New test file against the unfixed code: **8 failed, 3 passed** — measured against the **11-test**
version of the file as it stood at that moment. The file has since grown to 17 tests (cycle-2
additions), so that pair does not reproduce today; the cycle-1 Q/A independently measured **6 failed,
7 passed** against the 13-test intermediate. All three generations agree the defect reproduces; only
the counts move as tests are added. Caveat added in cycle 2 rather than restating a number that no
longer re-derives.

## §C. Fix — same scenario, post-fix

```
check_and_enforce_kill_switch() returned:
{
  "triggered": false,
  "blocked": true,
  "block_reason": "kill_switch_disarmed_lost_history",
  "pre_armed": false,
  "pre_breach": { "any_breached": false, "daily_baseline_missing": true,
                  "trailing_baseline_missing": true, "armed": false },
  "breach":     { "any_breached": false, "daily_baseline_missing": false,
                  "trailing_baseline_missing": false, "armed": true },
  "auto_resume": {"action": "no_op", ...}
}

rows: {"event": "peak_update", "nav": 18000.0}
      {"event": "sod_snapshot", "nav": 18000.0, "date": "2026-07-26"}
      {"event": "baseline_anchor_on_lost_history", "prior_sod_nav": null,
       "prior_peak_nav": null, "anchored_nav": 18000.0, "blocked_orders": true}
```

`pre_breach.armed: false` is the measurement the pre-fix code destroyed before taking it.

```
$ python -m pytest backend/tests/test_phase_36_12_kill_switch_trading_path_block.py -q
17 passed

$ python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader'      # IMMUTABLE
108 passed, 1 skipped, 2104 deselected
```

*(Both re-measured after the cycle-2 fixes; the cycle-1 figures were `13 passed` / `104 passed,
1 skipped` on the smaller suite.)*

## §D. The required curl — rig `:8002`, AFTER the simulated post-rotation cycle

```
$ curl -s http://localhost:8002/api/paper-trading/kill-switch
{
    "paused": false,
    "pause_reason": null,
    "sod_nav": 18000.0,
    "sod_date": "2026-07-26",
    "peak_nav": 18000.0,
    "baseline_provenance": "lost_history_anchor",
    "current_nav": 18000.0,
    "breach": {
        "daily_loss_breached": false, "daily_loss_pct": 0.0, "daily_loss_limit_pct": 4.0,
        "trailing_dd_breached": false, "trailing_dd_pct": 0.0, "trailing_dd_limit_pct": 10.0,
        "any_breached": false,
        "daily_baseline_missing": false, "trailing_baseline_missing": false,
        "armed": true
    },
    "thresholds": {"daily_loss_limit_pct": 4.0, "trailing_dd_limit_pct": 10.0}
}
```

**Read this honestly.** `armed` is `true` — because the anchor DID happen and the switch really is
armed again. What changed is that the payload is **no longer indistinguishable from a healthy
book**: `baseline_provenance: "lost_history_anchor"` says, on the operator's own status endpoint,
that these baselines were anchored because their history was unrecoverable, so the `0.0%` drawdown
they report starts from a fiction. On the first rig run that key did not exist and the payload was
byte-identical to a clean book — that gap is what the rig caught and what
`baseline_provenance` closes. It survives a restart (replayed from the audit event) and is cleared
only by a deliberate, token-gated `peak_reset`.

The order-blocking half of the demonstration is not visible on this GET at all — it is
`blocked: true` in §C, the `baseline_anchor_on_lost_history` row, the P1 alert, and
`autonomous_loop`'s halt branch. Stated plainly rather than implied.

## §E. Criteria summary

| # | Criterion | Status |
|---|---|---|
| 1 | reproduce the defect first, verbatim | **MET** — §B, recorded before any code changed |
| 2 | no longer silently forgives; state which option | **MET** — declines to place orders; §C |
| 3 | first-ever boot still trades | **MET** — dedicated test + 2 pre-existing tests unchanged |
| 4 | trading-path behaviour on `armed:false` decided explicitly | **MET** — BLOCK; documented in the contract and at the code site |
| 5 | no route to `reset_peak` | **MET** — zero rows + monkeypatched raiser |
| 6 | per-leg independence preserved | **MET** — `evaluate_breach` byte-untouched; 2 tests |
| 7 | mutation-test every new guard | **MET** — 14 mutations, 14 killed on the shipped 17-test suite, incl. the fixture, both discriminator directions, and the two survivors the cycle-1 Q/A found (`M11` probe fail-safe, `M12` neutered halt wiring) |
| 8 | all three operator strings revised in the same change | **MET** — plus a 4th site the step did not list (`kill_switch.py:527-528`); grep guard paired with a behavioural 409 assertion |

## §F. Teardown

```
:8002 listeners: 0
operator :8000 -> 200, launchd pid 76381 (never restarted; verified via launchctl print)
operator :3000 -> 302 (never driven)
handoff/kill_switch_audit.jsonl md5 ce8fb93348bb9a3bbe26f2d91b1bc05e (unchanged)
git diff --stat -- backend/services/kill_switch.py after all mutations: clean
```
