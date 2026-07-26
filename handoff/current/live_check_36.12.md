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
23 passed

$ python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader'      # IMMUTABLE
114 passed, 1 skipped, 2104 deselected
```

*Re-measured at every cycle, because the suite keeps growing and a stale count is a claim that no
longer re-derives (the cycle-3 Q/A caught this section carrying cycle-2 numbers). Cycle 1:
`13 passed` / `104 passed, 1 skipped`. Cycle 2: `17 passed` / `108 passed, 1 skipped`. Cycle 3: `22 passed` / `113 passed, 1 skipped`. Cycle 4: `23 passed` / `114 passed, 1 skipped`.*

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
| 7 | mutation-test every new guard | **MET as of cycle 4** — 18 mutations, 18 killed on the 23-test suite. `QA-Z1` (the cycle-3 blocker: delete `return summary` from the halt block) is now KILLED `1 failed, 22 passed` by `test_phase_36_12_a_blocked_cycle_really_places_no_orders`, which drives the REAL `run_daily_cycle` and asserts nothing was decided or traded. The three-cycle relocation pattern is retired by executing the composition rather than guarding its shape again. |
| 8 | all three operator strings revised in the same change | **MET.** All three revised, plus a 4th site the step did not list (`kill_switch.py:527-528`); the grep guard is paired with a behavioural 409 assertion, and two vitest DOM assertions on the rendered `title` attributes were proven falsifiable against HEAD's old strings by an independent evaluator. On the §1c capture: measured — `grep -rn 'KillSwitchPanel' frontend/src/` returns the component's own file and **one COMMENT**, so the component is **never mounted** and a live capture of those two tooltips is impossible *by construction*, not merely inconvenient. Only ONE of the step's "three shipped operator-facing strings" (`paper_trading.py:600`) was ever operator-visible; the dead-code question is filed as `36.16`. |

## §F. The §1c question, resolved

Three cycles asked for a live capture of the two `KillSwitchPanel` tooltips and each gave the same
reason for its absence — "the live book is armed, so the disarmed branch cannot render". **That
reason was wrong.** The component is not mounted anywhere, so it would not render on a disarmed book
either. Found only by building a rig good enough to disprove it (attempt #3, which rendered the full
cockpit against real data through a read-only proxy — `captures_36.12/36.12_cockpit_live_via_readonly_proxy.png`).

The governing precedent is this session's **phase-36.7 cycle-6** ruling, which was asked the same
question and answered it explicitly: an **inherently unobtainable** live capture does **not** cap a
verdict, because qa.md §1c's cap attaches to a *missing or stale* capture and an impossible one is
neither. That ruling was made on a step whose UI claim was unreachable because of live STATE; this
step's is unreachable because the component is unmounted — a strictly stronger form of the same
condition. `harness_compliance_ok: false` should still ride alongside, so the limitation stays on
the record.

**Everything in the table above was verified by an independent evaluator's own execution.** The two
cycle-3 blockers are now closed: `QA-Z1` by a behavioural test that drives the real
`run_daily_cycle` (mutation-killed, `1 failed, 22 passed`), and the §1c capture by the measurement
and precedent above.

## §F. Teardown

```
:8002 listeners: 0
operator :8000 -> 200, launchd pid 76381 (never restarted; verified via launchctl print)
operator :3000 -> 302 (never driven)
handoff/kill_switch_audit.jsonl md5 ce8fb93348bb9a3bbe26f2d91b1bc05e (unchanged)
git diff --stat -- backend/services/kill_switch.py after all mutations: clean
```
