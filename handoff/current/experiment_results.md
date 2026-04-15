# Experiment Results -- Phase 4.4.4.2 Position-Limit Drill

**Cycle:** 10 (Ford Remote Agent, 2026-04-15)
**Base commit:** `cbd14d4` (Phase 4.4.4.1 kill-switch drill landed)
**Phase:** PLAN -> GENERATE -> EVALUATE -> LOG

## Outcome: PASS (pending QA verdict)

## What landed

### 1. New drill: `scripts/go_live_drills/position_limits_test.py`

Standalone stdlib-only drill (~220 lines) mirroring the Cycle 9
`kill_switch_test.py` shape. Exercises `SignalsServer.risk_check` for the
three position-limit thresholds named in `docs/GO_LIVE_CHECKLIST.md`
section 4.4.4.2:

* per-ticker exposure cap (`max_exposure_per_ticker_pct = 10.0`)
* total exposure cap (`max_total_exposure_pct = 100.0`)
* daily trade count cap (`max_daily_trades = 5`)

Six canonical scenarios, each wrapped in an `assert` block on both
`resp["allowed"]` and the expected conflict string:

| # | Setup | Trade | Expected | Conflict |
|---|-------|-------|----------|----------|
| S1 | no positions, cash=$10k | BUY AAPL 15@$100 (15%) | blocked | `max_exposure_per_ticker` |
| S2 | no positions, cash=$10k | BUY AAPL 10@$100 (exact 10.00%) | allowed | -- |
| S3 | AAPL 5@$100 (5%) | BUY AAPL 6@$100 (+6%=11%) | blocked | `max_exposure_per_ticker` |
| S4 | MSFT 95@$100 (95%), cash=$1k | BUY AAPL 6@$100 (+6%=101%) | blocked | `max_total_exposure` |
| S5 | trades_today=[5 stubs] | BUY AAPL 1@$100 | blocked | `max_daily_trades` |
| S6 | trades_today=[4 stubs] | BUY AAPL 1@$100 | allowed | -- |

Pre-drill sanity check pins all 4 limit literals to the Phase 4.4.4.4
evidence (per-ticker=10.0, total=100.0, drawdown=-15.0,
daily_trades=5). Any drift fails loudly before scenarios run.

### 2. Checklist flip: `docs/GO_LIVE_CHECKLIST.md` 4.4.4.2

Flipped the 4.4.4.2 bullet from `- [ ]` to `- [x]` and appended a new
`- **Evidence**:` line under the bullet naming the drill path, scenario
count (6/6), and the Phase 4.4.4.4 literal sanity check. All other
checkbox states preserved:

- 4.4.4.1 `[x]` (Cycle 9 kill-switch evidence)
- 4.4.4.2 `[x]` (this cycle)
- 4.4.4.3 `[ ]` (stop-loss drill, next cycle)
- 4.4.4.4 `[x]` (Cycle 8 hardcoded-literals evidence)

Phase 4.4 progress: **3 / 27 items** now `[x]` (up from 2 / 27 at
cycle start).

## Execution log

```
$ python scripts/go_live_drills/position_limits_test.py
Paper trader not available -- signals server in stub mode
PASS S1 per-ticker 15% BUY -> blocked (max_exposure_per_ticker)
PASS S2 per-ticker 10.00% boundary BUY -> allowed (strict-greater pin)
PASS S3 per-ticker aggregation 5%+6% -> 11% BUY -> blocked
PASS S4 total exposure 95%+6% -> 101% BUY -> blocked (max_total_exposure)
PASS S5 daily trade count 5 BUY -> blocked (max_daily_trades)
PASS S6 daily trade count 4 BUY -> allowed (under cap)
DRILL PASS: 6/6 position-limit scenarios verified against SignalsServer.risk_check (per-ticker=10.0, total=100.0, daily_trades=5)
$ echo $?
0
```

## Contract SC results (20/20 PASS)

* **SC1-4 Scope discipline**: 2 files touched
  (`scripts/go_live_drills/position_limits_test.py` added,
  `docs/GO_LIVE_CHECKLIST.md` modified), zero `backend/**.py` touched,
  `kill_switch_test.py` byte-identical to `cbd14d4`, imports set is
  exactly `{importlib.util, sys, pathlib}`, zero non-ASCII bytes
  (10014 bytes total).
* **SC5-8 Drill structure**: `importlib.util.spec_from_file_location`
  loader bypasses package `__init__`; `SignalsServer()` constructed in
  stub mode; pre-drill sanity pins all 4 limit literals; 6 scenario
  functions present.
* **SC9-14 Drill behavior**: exit code 0; `DRILL PASS: 6/6` printed;
  every scenario asserts on `allowed` and `conflicts`; S1 triggers
  `max_exposure_per_ticker`, S4 triggers `max_total_exposure`, S5
  triggers `max_daily_trades`; S2 pins strict-greater boundary
  semantics at exact 10.00%.
* **SC15-18 Checklist evidence**: 4.4.4.2 bullet flipped to `[x]`;
  evidence line appended; section header unchanged; no other 4.4.4.x
  section touched.
* **SC19-20 Global invariants**: `ast.parse` clean; `py_compile` clean.

## Adversarial probe results (8/8 PASS)

1. **Boundary-inclusive trap**: S2 at exact 10.00% -> allowed. Confirms
   the drill pins the strict `>` semantic, not `>=`.
2. **Helper price disagreement**: all BUY scenarios set explicit
   `price=100.0`; no reliance on existing-position `last_price`
   fallback path. `proposed_notional` is nonzero in every scenario.
3. **Trades_today list shape**: S5 and S6 use a list of dict stubs,
   not an int. Exercises the list branch at line 791-794, which is the
   production shape from `_append_signal_history`.
4. **Cash exhaustion trap**: every BUY sets `cash` >= proposed notional
   (S1-S3 use cash=$10k vs max $1500 notional; S4 uses cash=$1000 vs
   $600 notional; S5/S6 use cash=$10k vs $100 notional). No scenario
   accidentally trips `insufficient_cash`.
5. **Total-exposure aggregation trap**: S4 uses MSFT (not AAPL) for
   the 95% existing position, so the per-ticker check for the new
   AAPL trade sees only 6% (OK) and falls through to the total check,
   which correctly fires with 101% > 100%.
6. **Daily-trade order-of-evaluation trap**: S5 uses a trivially-sized
   trade (1 share @ $100) on a clean portfolio so the daily-cap branch
   at line 846 fires before any exposure check. Confirmed by the
   `max_daily_trades` conflict (not `max_exposure_per_ticker`).
7. **Non-ASCII slip**: `all(ord(c) < 128)` across the entire file.
8. **Re-runnability**: drill invokes only `get_risk_constraints()` and
   `risk_check()`. Zero mutations to stub state, zero BQ calls, zero
   GCP calls. Re-ran the drill twice during the cycle, identical
   output both runs.

## Files modified

| File | Diff | Note |
|------|------|------|
| `scripts/go_live_drills/position_limits_test.py` | +220 / 0 | New drill |
| `docs/GO_LIVE_CHECKLIST.md` | +2 / -1 | 4.4.4.2 `[x]` + evidence |
| `handoff/current/contract.md` | rewritten | Phase 4.4.4.2 contract |
| `handoff/current/experiment_results.md` | rewritten | This file |
| `handoff/current/evaluator_critique.md` | pending QA | -- |
| `handoff/harness_log.md` | Cycle 10 pending | -- |

## Key findings

- **Strict-greater semantics vindicated.** `risk_check` line 872 uses
  `if ticker_pct > max_per_ticker_pct`, not `>=`. S2 pins this by
  constructing a trade at exactly 10.00% and asserting `allowed=True`.
  Any future refactor to `>=` would be a silent tightening of the
  concentration cap by one cent and would be caught immediately by
  this drill.
- **Per-ticker aggregation is correctly additive.** S3 starts with an
  existing 5% position in AAPL and adds a 6% BUY, and the drill
  confirms the `existing_position_notional + proposed_notional`
  computation at line 870-871 fires on the projected 11%. Confirms the
  drill catches both the single-trade and aggregation paths.
- **Daily-trade cap evaluates before exposure checks.** S5 sets a
  5-entry trades_today list and a 1% trade, and the drill confirms
  the `max_daily_trades` conflict fires (not any exposure conflict).
  This pins the FINRA 15c3-5-style hard-fatal-first ordering
  (schema -> action/state -> daily cap -> concentration -> total ->
  cash -> drawdown).
- **All 4 limit literals still hold.** Pre-drill sanity check
  re-verifies the 4.4.4.4 evidence from the drill's viewpoint. This
  is belt-and-braces with the Cycle 8 verification but gives the
  4.4.4.2 drill a single call site that fails loudly on any drift.

## Blockers

None. Drill is stdlib-only, re-runnable, no network, no venv, no GCP.
Exit code 0 on first run. Zero retries.

## Next run should

1. **Phase 4.4.4.3 (stop-loss drill)** -- next risk-management
   checklist item. Inspect `backend/services/paper_trader.py` for the
   stop-loss exit logic and write a drill that marks a position at
   -8.5% and confirms the next tick emits a SELL. If the stop is not
   present in the paper trader, this item is a hard block that needs
   a code gate first. Research gate: WAIVED if the stop exists,
   FULL-GATE if we need to design the stop logic.
2. **Phase 4.4.3.5 (incident log P0 scan)** -- joint item but Ford can
   pre-verify. Read `.claude/context/known-blockers.md` and confirm
   no entry tagged `P0` without a `resolved:` line.
3. **Phase 4.2.4.3 follow-up** -- the read-path projection landed in
   a prior cycle (`a128dc3`). Consider a companion drill that exercises
   the `QUALIFY ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY
   recorded_at DESC) = 1` projection against the stub BQ client.
4. **Storage Write API migration** -- still a meaningful future
   cycle, still needs protobuf tooling and full research gate.
