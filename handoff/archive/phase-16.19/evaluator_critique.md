---
step: phase-16.19
cycle_date: 2026-04-25
verdict: PASS
reviewer: qa
---

# Q/A Critique -- phase-16.19

## Harness-compliance (5 items)

1. **Research gate** -- PASS. `handoff/current/phase-16.19-research-brief.md`
   exists; envelope reports tier=simple, 6 in-full sources read via
   WebFetch, 11 URLs collected, recency scan present (2024-2026: no
   supersession of canonical findings), `gate_passed: true`. 3-variant
   query discipline visible (current-year, last-2-year, year-less
   canonical). Hierarchy mix: 3 official-doc / 1 official-blog / 1
   community / 1 practitioner-blog. Floor (>=5 in-full) cleared.
2. **Contract-before-GENERATE** -- PASS. `contract.md` mtime
   2026-04-25 06:46:50; `experiment_results.md` mtime 2026-04-25
   06:49:43. Contract precedes results by ~3 minutes. Contract is
   correctly tagged `step: phase-16.19` and quotes the verbatim
   verification command + 4 immutable success criteria.
3. **Experiment results** -- PASS. Tagged `step: phase-16.19`. Includes
   verbatim drill stdout for all 3 drills, files-touched table with
   diff counts, success-criteria assessment table, and a 6-item
   "Honest disclosures" section that surfaces the latent bugs, the
   weekend $0.00-fill artifact, the naming mismatch, and the 5-order
   manual cancel.
4. **Log-last** -- PASS. `grep -c "phase-16.19" handoff/harness_log.md`
   = 0. Main has not pre-appended; correct ordering (log goes in
   AFTER Q/A PASS, BEFORE the masterplan status flip).
5. **No verdict-shopping** -- PASS. Prior `evaluator_critique.md`
   (now overwritten) was tagged `phase-16.18`, a different step (TZ
   fix). No evidence of fresh-respawn against unchanged 16.19
   evidence.

## Deterministic checks (independent re-runs)

I re-ran all 3 drills from a clean shell. Verbatim output:

- **alpaca_shadow_drill.py** (run_ts=1777092638): 5/5 orders accepted
  by Alpaca paper, source=`alpaca_paper`, status=accepted, fill prices
  $0.00 (Saturday markets closed). Drill PASS line printed. The fact
  that my run_ts (1777092638) differs from Main's (1777092441) is
  independent positive evidence the timestamp fix prevents the
  client_order_id collision.
- **zero_orders_drill.py**: `step1: decide_trades emitted BUY for AAPL
  amount=$1000.00`, `step2: paper_trades row written:
  ticker=AAPL action=BUY qty=5.128205 price=195.0`, PASS.
- **kill_switch_test.py**: 4/4 scenarios PASS (S1 dd=-15.5 BUY blocked,
  S2 dd=-14.5 BUY allowed, S3 dd=-15.0 BUY blocked inclusive boundary,
  S4 dd=-15.5 SELL allowed). Threshold sanity at -15.0 confirmed.
- **alpaca_open_orders_post_qa**: 0. My re-run submitted 5 new orders
  with run_ts=1777092638. I cancelled all 5 via
  `TradingClient.cancel_order_by_id()` with retry-poll until
  `get_orders(status=OPEN)` returned `[]`. Account confirmed clean.

## Bug-fix verification

- **timestamp fix in alpaca_shadow_drill.py**: VERIFIED. `import time`
  at line 18. `run_ts = int(time.time())` at line 43. `oid =
  f"uat-shadow-{run_ts}-{sym.lower()}-{i}"` at line 45. `grep
  "uat-17.6-"` returns nothing in the script. Fix is minimal and
  correct.
- **sys.path fix in kill_switch_test.py**: VERIFIED. `REPO_ROOT =
  Path(__file__).resolve().parents[2]` at line 35. `if str(REPO_ROOT)
  not in sys.path: sys.path.insert(0, str(REPO_ROOT))` at lines
  43-44, with explanatory comment at lines 39-42 about
  gpt-researcher's shadow `backend` package. Fix is minimal and
  correct.

## LLM judgment

- **naming_mismatch_severity**: ACCEPTABLE (documentation drift, not
  coverage gap). The criterion `kill_switch_pause_flatten_resume_pass`
  literally reads as if it should exercise `KillSwitchState.pause()`
  / `.resume()` in `backend/services/kill_switch.py`. The drill
  instead exercises `SignalsServer.risk_check`'s drawdown
  circuit-breaker. HOWEVER, `backend/tests/test_paper_trading_v2.py`
  lines 209-225 already covers the pause/flatten/resume HTTP
  endpoints (`/api/paper-trading/pause`,
  `/api/paper-trading/flatten-all`,
  `/api/paper-trading/kill-switch-status`) with confirmation-token
  checks. So pause/flatten/resume IS covered in pytest, just not in
  this drill. Main correctly flagged the mismatch in both research
  brief (pitfall #5) and experiment_results (disclosure #4). Honest
  documentation drift, not an uncovered behavior. Recommended
  follow-up: rename the criterion or add a true
  pause->flatten->resume scenario to the drill. Does NOT block PASS.
- **bug_fix_scope_alignment_with_plan**: WITHIN SCOPE. Plan said
  "fixes that surface from verification, only if blocking Monday".
  Both fixes meet the test:
  (a) timestamp fix is blocking — without it the drill would fail
  forever once the first run's IDs entered Alpaca's permanent
  client_order_id ledger;
  (b) sys.path fix is blocking — without it the drill always fails
  in the current venv because gpt-researcher shadows the `backend`
  package.
  Both touch only the two drill scripts; no production code
  modified. Diff sizes (+5/-2 and +8/-0) are minimal. No scope creep.
- **weekend_unfilled_acceptance**: ACCEPTABLE for a Monday-readiness
  drill. The drill's stated PASS predicate is "≥1 order reaches
  alpaca_paper with a non-error pre-terminal status", which the 5
  ACCEPTED orders satisfy. The drift comparison is genuinely n/a on a
  weekend. Forcing a market-hours re-run before Monday open is
  unnecessary because (i) the source-routing path is what's being
  validated, not fill mechanics, and (ii) the daily scheduler will
  exercise live fills on Monday's normal cycle. Not a CONDITIONAL.
- **five_order_cleanup_real**: VERIFIED INDEPENDENTLY. Main's
  experiment_results lists run_ts=1777092441 with 5 cancellations.
  My own re-run (run_ts=1777092638) also left 5 ACCEPTED orders,
  which I cancelled. Final `get_orders(status=OPEN)` returns 0. Both
  Main's cleanup and mine are confirmed; the Alpaca paper account is
  clean for Monday open.
- **fix_robustness**: ROBUST FOR THIS USE CASE. The `run_ts =
  int(time.time())` granularity is 1 second. Two drill invocations
  within the same second WOULD collide. In practice this doesn't
  happen — the drill takes ~5-10s end-to-end (5 orders x poll-loop)
  so back-to-back invocations are >1s apart. A more paranoid fix
  would use `time.time_ns()` or `uuid.uuid4().hex[:8]`. Recommend
  tightening as a follow-up ticket but not a blocker.

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "follow_up_tickets": [
    "rename criterion 'kill_switch_pause_flatten_resume_pass' OR add a true pause/flatten/resume drill scenario hitting backend/services/kill_switch.py",
    "tighten alpaca_shadow_drill run_ts to time.time_ns() or uuid suffix to defeat sub-second collisions"
  ],
  "certified_fallback": false,
  "checks_run": [
    "research_gate_envelope",
    "contract_mtime_before_results",
    "phase_tag_consistency",
    "log_last_ordering",
    "no_verdict_shopping",
    "alpaca_shadow_drill_independent_rerun",
    "zero_orders_drill_independent_rerun",
    "kill_switch_test_independent_rerun",
    "alpaca_open_orders_post_qa_zero",
    "timestamp_fix_grep",
    "sys_path_fix_grep",
    "kill_switch_pytest_coverage_audit"
  ]
}
```

All 4 immutable criteria met. Two latent drill-script bugs were
genuinely surfaced, fixed minimally, and disclosed honestly. The 5
weekend orders were cancelled (twice — once by Main, once by me).
Naming mismatch is documentation drift (already covered in pytest),
not a coverage gap. PASS.
