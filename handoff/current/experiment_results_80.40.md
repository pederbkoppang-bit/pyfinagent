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

## Mutation matrix

| # | Mutation | Result |
|---|---|---|
| revert `compute_max_drawdown_from_snapshots` to a `0`-stub | criterion 6 | KILLED (workflow-reported, reproduced) |
| **R7 mutation:** remove the `-0.0` normalization | Main | KILLED — `assert -1.0 > 0` |

## Scope honesty

- `80.43` (unvalidated `snapshot_date` sort key, phantom-drawdown class still reachable if every
  row lacks a date) and `80.44` (stale CI test-collection-count pin) queued rather than fixed —
  both are real but neither is this step's immediate safety surface.
- `36.11` (cross-tab threshold inconsistency between this row and `PaperVsBacktestCard`) and
  `36.10` (fail-open resume gates untested) also queued — see those step texts.
- Operator's `:8000` never restarted; `:3000` never driven.

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

