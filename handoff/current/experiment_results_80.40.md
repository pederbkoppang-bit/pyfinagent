# Experiment Results — phase-80.40

**Step:** `80.40` (P0) — `/performance` never returned `max_drawdown_pct`.
Date 2026-07-26. Contract: `handoff/current/contract_80.40.md`.

---

## Method

Same Workflow as `36.7` (`wf_b2205517-994`) — parallel research, sequential implementation
(80.40 first), four-lens adversarial verification, synthesis. See `experiment_results_36.7.md`
for the shared methodology note on why this is not self-evaluation.

## What shipped

**`backend/services/perf_metrics.py`** — `compute_max_drawdown_from_snapshots`, the single
source of truth for this figure per the backend-services rule (*"Never compute Sharpe, drawdown,
or alpha outside perf_metrics.py — import from there"*). Delegates arithmetic to the existing
canonical primitive `backend.backtest.analytics.compute_max_drawdown`; no new formula.

- **Sign convention verified, not assumed.** Negative percent (matches `backtest.py`'s existing
  convention, confirmed by reading `analytics.compute_max_drawdown`'s call sites). The frontend's
  `maxDd > -10 ? SAFE : ...` requires this — a positive-magnitude value would satisfy `> -10` at
  every depth and invert the safety verdict silently. Main independently confirmed
  `analytics.py`'s sign convention cannot emit a positive magnitude, and the live measured value
  is `-5.31` (negative, as expected).
- **`None`, never `0.0`, on every degraded path** — 0 rows, 1 row, or a monotonic rise are three
  distinct states that a bare `0.0` fallback would conflate under `80.36`'s presence-not-value
  contract. `-0.0` is normalized to `+0.0` so the payload never carries a negative zero.
- **Non-finite input handling** — a non-finite computed result returns `None` rather than letting
  FastAPI's JSON serializer 500 the whole `/performance` payload (the phase-80.27 NaN class).
- **No extra query, no event-loop cost** — reuses the `snapshots` list the caller already fetched;
  pure numpy over ≤365 floats (~0.04ms measured), no `asyncio.to_thread` needed.

**`backend/api/paper_trading.py`** — wires `max_drawdown_pct` into the `/performance` payload.

**`frontend/src/components/paper-trading/cockpit-helpers.tsx`** — label rename only:
`"Kill switch (-15%)"` → `"Max drawdown (-15%)"` (verified at `:463`; see contract §Criterion 4
for the full reconciliation). No threshold value changed anywhere.

**Tests:** `backend/tests/test_phase_80_40_perf_metrics_drawdown.py` (new, 28 tests after Main's
addition), `frontend/src/components/paper-trading/cockpit-helpers.drawdown.test.tsx` (new).

## Independently reproduced by Main

```
$ curl http://localhost:8001/api/paper-trading/performance   # isolated rig, REAL data
nav: 23838.16   sharpe_ratio: 3.44   max_drawdown_pct: -5.31   pnl_pct: 19.19
```
Numeric, negative, present alongside `sharpe_ratio` — criterion 1 met. Playwright capture on the
same rig shows `/paper-trading/positions` rendering **"Max drawdown (-15%) SAFE"** and
**"Drawdown -5.3% / -15%"** — a real verdict, not `NO DATA`
(`handoff/current/captures_36.7_80.40/36.7_80.40_ARMED_real_verdict.png`).

`git diff --numstat` empty on `settings.py`, `signals_server.py`, `analytics.py`,
`paper_go_live_gate.py`, `drawdown_alarm.py` — no threshold value changed.

## Main's own follow-up fix: R7 (vacuous test, `perf_metrics` half)

`test_negative_zero_is_normalized` used a strictly monotonically-increasing NAV fixture
(`GROWING_ASC`). **Measured directly:** `compute_max_drawdown` on that series returns an exact
`0.0`, never `-0.0` — deleting the `-0.0`-normalization line left the test green regardless.
A genuine `-0.0` requires a tiny real dip: on the real book's NAV scale, a 19-cent-then-16-cent
wobble measures `round(-0.0006711927430634515, 2) == -0.0`. Rewrote the test with that fixture
(kept the monotonic case too, as a distinct code path worth pinning separately). Mutation-killed:
removing the normalization now fails with `assert -1.0 > 0` (i.e., `copysign` detects the sign
flip); restored, passes.

(The `OpsStatusBar` half of `R7` — a container-wide em-dash assertion satisfied by unrelated
sibling error states — is `36.7`'s to record, since it's in `KillSwitchPanel.disarmed.test.tsx`;
see `experiment_results_36.7.md`.)

## Corrections to the implementer's own claims

1. **SELF-CONTRADICTORY:** the implementation summary asserted *"the reality-gap card compares
   two identically-defined numbers on one axis"*, while its own `residual_risks` list correctly
   noted `cockpit-helpers.tsx:295` hardcodes the backtest side as a literal `-12.0%` and
   `optimizer_best.json` has no drawdown key at all. The risk list is right; the summary line is
   wrong and is not repeated here.
2. **FALSE:** *"[PaperVsBacktestCard] behaviour is unchanged by my diff."* True only for the code
   — the *data* flowing through it changed, because 80.40 is what makes `max_drawdown_pct` real
   for the first time. Main confirmed the touched-file diff for `PaperVsBacktestCard`
   (`cockpit-helpers.tsx:266`) is empty — the `?? 0` pattern there is untouched — but with real
   data now present, the rendered text goes `"—%"` → `"-5.3%"` and the `maxDd > -15` colour
   predicate becomes reachable for the first time where it was previously always `0 > -15 = true`.
   That card's own defect (already queued as `80.38`, pre-existing, out of this step's scope) is
   now exercisable with live data rather than dormant. Not a new bug introduced here; a latent one
   made live by supplying real data. Flagged, not fixed, per `80.38`'s existing scope.

## Criterion 3 — the 0-stub mutation, RUN BY MAIN, output verbatim

Cycle 3's Q/A blocked on this and was right to: the matrix previously said only
`KILLED (workflow-reported, reproduced)`, which does not let a reader distinguish "Main ran it" from
"Main relayed the workflow's claim". Criterion 3's second clause — *"record that failing output
verbatim"* — exists precisely to forbid the second-hand form. **Main ran it. This is Main's own
terminal output**, from an in-memory mutant (the function's body replaced with `return 0.0`,
`compile()` + `sys.modules` injection, pattern asserted to match exactly once;
`git diff --stat -- backend/services/perf_metrics.py` empty afterwards):

```
### MUTANT: compute_max_drawdown_from_snapshots body -> `return 0.0`
F                                                                        [100%]
=================================== FAILURES ===================================
________________ test_known_nav_series_pins_the_exact_drawdown _________________

    def test_known_nav_series_pins_the_exact_drawdown():
        """MUST FAIL against a stub returning 0.0 (and against any sign flip)."""
>       assert compute_max_drawdown_from_snapshots(KNOWN_DD_ASC) == KNOWN_DD_PCT
E       AssertionError: assert 0.0 == -30.0
E        +  where 0.0 = compute_max_drawdown_from_snapshots([{'snapshot_date': '2026-01-01', 'total_nav': 100.0}, {'snapshot_date': '2026-01-02', 'total_nav': 150.0}, {'snapshot_date': '2026-01-03', 'total_nav': 105.0}, {'snapshot_date': '2026-01-04', 'total_nav': 160.0}])

backend/tests/test_phase_80_40_perf_metrics_drawdown.py:80: AssertionError
=========================== short test summary info ============================
FAILED backend/tests/test_phase_80_40_perf_metrics_drawdown.py::test_known_nav_series_pins_the_exact_drawdown
1 failed, 27 deselected in 0.11s
```

Whole-file scope under the same mutant: **20 failed, 8 passed** — the stub is caught by the pinning
test, the high-water-mark test, the deeper-trough test, both sign tests, the ordering tests, every
`degraded_paths_return_none_never_zero` case (the stub returns `0.0` where `None` is required — the
exact conflation `80.36`'s presence-not-value contract forbids), and both endpoint tests.

**Controls, each labelled with the selection that produces it** (cycle-4 correction: this
previously read "Control (unmutated, same selection): `3 passed`", which matched neither selection
recorded beside it — the number was real but belonged to a third, unstated selection. Caught by the
cycle-4 Q/A, which reproduced all three itself):

| selection | unmutated | mutated |
|---|---|---|
| `-k test_known_nav_series_pins_the_exact_drawdown` (the named test above) | `1 passed, 27 deselected` | `1 failed, 27 deselected` |
| whole file | `28 passed` | `20 failed, 8 passed` |
| `-k` the three `KNOWN_DD_ASC` pinning tests | `3 passed, 25 deselected` | `3 failed, 25 deselected` |

*(Cycle-5 NOTE, applied after the PASS: the last cell read `3 failed`, a truncation of the summary
line its siblings carry in full. The count was right; the form was inconsistent. Nothing else in
this table is a verbatim capture — the one block that IS labelled verbatim is the failure output
above, which the cycle-5 Q/A diffed line-by-line against its own run and matched exactly.)*

A note on method, since it cost a run: the first attempt exec'd the mutated source **before**
registering the module in `sys.modules`, and `perf_metrics` defines a `@dataclass` — `dataclasses`
resolves `cls.__module__` through `sys.modules` at class-creation time and died with
`AttributeError: 'NoneType' object has no attribute '__dict__'`. Registering the module object first
fixes it. Recorded because a silent version of that failure would look like a surviving mutant.

## Mutation matrix

| # | Mutation | Result |
|---|---|---|
| `compute_max_drawdown_from_snapshots` body → `return 0.0` | criterion 3 + 6 | **KILLED — run by Main, verbatim output above** (`1 failed` on the named test; `20 failed, 8 passed` file-wide) |
| **R7 mutation:** remove the `-0.0` normalization | Main | KILLED — `assert -1.0 > 0` |

## Cycle-4 follow-up (post-Q/A-3 FAIL) — what changed

Cycle 3's verdict was **FAIL**, transcribed verbatim in `handoff/current/evaluator_critique_80.40.md`
(recovered from the evaluator's transcript after an end-flush stall; provenance and why a recovered
FAIL is the conservative thing to act on are documented there). Its one BLOCKING finding and the
three non-blocking ones are all closed, and **no code changed** — `git diff` on this step's backend
files is unchanged from cycle 2; the only source edit is a comment.

1. **BLOCKING — criterion 3 had no firsthand record.** Fixed: Main ran the 0-stub mutation and the
   verbatim output is in the section above, with the file-wide `20 failed, 8 passed` and the
   unmutated control. The matrix row no longer says "workflow-reported".
2. **The ladder comment understated the mismatch.** `cockpit-helpers.tsx` said the all-time-max vs
   current-drawdown mismatch was a reason the row "must not claim to be the kill switch"; in fact
   ladder B's `-15` also gates on CURRENT drawdown, so the mismatch applies to both ladders.
   Comment corrected, and it now states the direction explicitly: an all-time max is never shallower
   than the current drawdown, so this row can warn early but never late.
3. **`live_check_80.40.md` §E contradicted its own RESOLVED block.** Annotated as a PRE-RESTART
   reading (pid `70791` then, launchd pid `76381` now) rather than rewritten.
4. **The two source-scan guards, named by grep and each mapped to its own behavioural counterpart.**
   Cycle-4 correction: the first version of this item named the same guard twice ("…the kill_switch"
   and "the label-string scan" are one test) and sent both counterparts to the same frontend file.
   Re-derived with `grep -n "def test_" … | grep -iE "cockpit|ladder|label"`:

   | source scan | line | its behavioural counterpart |
   |---|---|---|
   | `test_the_two_drawdown_ladders_keep_their_documented_values` | `:366` | `backend/tests/test_phase_75_mcp_truth.py::test_thresholds_are_unchanged:167` — a **backend** test that calls `get_risk_constraints()` on a real server object and asserts the runtime values `-15.0 / -5.0 / -10.0` at `:173-175` |
   | `test_the_cockpit_row_no_longer_claims_to_be_the_kill_switch` | `:387` | `cockpit-helpers.drawdown.test.tsx:80` — `it("no longer labels the row as the kill switch")`: renders `RiskMonitorCard` with `LIVE_PERF` and asserts the **label** (`getByText("Max drawdown (-15%)")`, `queryByText("Kill switch (-15%)")` null). *(Cycle-5 NOTE, applied after the PASS: this said "asserts the verdict"; it asserts the label. The mapping is right — a label scan's counterpart should be a label assertion — only the descriptor was loose. The verdict itself is asserted by the sibling test in the same file.)* |

   Redundant by design: each scan catches a *string* regression (a correct verdict under a wrong
   label, or a comment drifting from the constants) that its behavioural sibling would not notice.
   The label lives here in the handoff, not in the test file — the cycle-3 item was explicitly
   optional, so no test was edited for it.

## Scope honesty

- `80.43` (unvalidated `snapshot_date` sort key, phantom-drawdown class still reachable if every
  row lacks a date) and `80.44` (stale CI test-collection-count pin) queued rather than fixed —
  both are real but neither is this step's immediate safety surface.
- `36.11` (cross-tab threshold inconsistency between this row and `PaperVsBacktestCard`) and
  `36.10` (fail-open resume gates untested) also queued — see those step texts.
- Operator's `:8000` never restarted; `:3000` never driven.

## Cycle-3 follow-up (post-Q/A-2) — what CHANGED in the evidence

Same three changes as `36.7` (shared cycle); stated here for this step's own criteria:

1. **Criterion 1 is now met on the operator's own `:8000`, not on the rig.** After the authorized
   restart (pid `70791` → `76381`), re-measured by Main this cycle:
   ```
   $ curl -s http://localhost:8000/api/paper-trading/performance
   nav: 23838.16   sharpe_ratio: 3.44   max_drawdown_pct: -5.31   pnl_pct: 19.19
   keys: [alpha_pct, benchmark_return_pct, days_active, max_drawdown_pct, nav, pnl_pct,
          round_trip_summary, sharpe_ratio, starting_capital, total_analysis_cost,
          total_buy_trades, total_sell_trades]
   ```
   `max_drawdown_pct` present and numeric alongside `sharpe_ratio` — the exact absence this step
   was filed for, now closed on the live backend. Cycles 1–2 had only the `:8001` rig capture.

2. **Immutable command re-run post-restart, this cycle:**
   ```
   $ python -c "import ast; ast.parse(open('backend/services/perf_metrics.py').read())"   # exit 0
   $ python -m pytest backend/tests/ -q -k 'perf_metrics or drawdown'
   68 passed, 2128 deselected, 1 warning in 5.45s
   ```

3. **PROTOCOL GAP, DISCLOSED — no `evaluator_critique_80.40.md` exists** (measured the same way as
   `36.7`: absent from `handoff/current/`, and never added in git history). `harness_log.md` has
   zero `phase=80.40` entries, so the Q/A's grep-based CONDITIONAL counter would read 0 against a
   true count of 2. Disclosed in the spawn evidence: **a third CONDITIONAL here is an auto-FAIL.**

## Research gate — landed (2 transient 529s, then success)

`handoff/current/research_brief_36.7_80.40.md`: **`gate_passed: true`**, 12 sources read in full
(floor 5), 34 URLs, recency scan performed. Full account in `experiment_results_36.7.md`
(shared brief, both steps' decisions assessed together) — summarized here for this step's two
decisions:

- **`None`-never-`0.0` on every degraded path — HOLDS, strongest finding.** `empyrical`'s own
  degraded path returns `np.nan`, not `0.0` — this step ships the same semantic, with the
  JSON-serialization hazard (`empyrical` doesn't have to solve that) additionally handled.
- **Negative-sign convention — HOLDS.** The code comment's counter-example (R's
  `invert=TRUE` default) is real, verified — but `empyrical`+`pyfolio`+`quantstats` are
  *unanimously* negative, so the split is across the R/Python boundary, not evenly split as the
  comment implied. Docstring corrected below; no code change, since the cockpit-inversion argument
  (a positive value would satisfy `maxDd > -10` at every depth) is decisive on its own regardless
  of the industry-split framing.

Two spawns of the corrective researcher failed before writing anything (transient
`API Error: 529 Overloaded`); the third succeeded. No new defect specific to 80.40 — the one new
defect the research surfaced (`36.12`, order-placing path forgives a missing baseline) is filed
under `36.7`.

