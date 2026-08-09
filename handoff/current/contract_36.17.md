# Contract -- phase-36.17

**Step:** 36.17 (P1) -- a halted cycle stops enforcing stop-losses, so the book
sits unguarded exactly when it is judged unsafe.
**Date:** 2026-08-09. **Cycle:** 190.
**Research gate:** PASSED -- `handoff/current/research_brief_36.17.md`
(40,384 chars, independently verified by the rail's stage-2 checker: 6 sources
read in full >= floor 5, 29 URLs collected >= floor 10, recency scan performed,
all 6 claimed URLs present in the brief, `gate_passed: true` enforced and
agreed with the self-report). Run `wf_7b26264d-462`.

---

## 1. Research-gate summary

**The premise is confirmed at source, and the step text's line numbers are
stale.** Re-derived 2026-08-09: the Step 5.5 halt `return summary` is at
`backend/services/autonomous_loop.py:1437`; Step 5.6 stop-loss enforcement
begins at `:1439`; `check_stop_losses` is called at `:1471`;
`backfill_missing_stops` at `:1458`; `cycle_halt_reason` at `:141-166`. The
step text's `:1334/:1336` are wrong and must not be copied forward.

**External finding (Q1): "liquidation-only" is a named, standard state,
distinct from a full halt.**

- **MiFID II RTS 6 Art. 12** scopes "kill functionality" to *unexecuted
  orders*; withdrawing from the market is a separate action.
- **ESMA Supervisory Briefing on Algorithmic Trading (2026-02) para 93**
  defines a three-rung ladder -- cancel / withdraw / kill -- as *distinct*
  remedial depths, not one switch.
- **NYSE Pillar Risk Controls v4.7** ships a literal **"Block only -- accept
  cancels"** mode: a blocked entity may still act to *reduce* risk.
- **CME's** all-or-nothing kill switch is the dissenting design, and the brief
  correctly discounts it: it is a *venue gateway* protecting the market, not a
  firm's own risk system, and the firm retains other routes.

`kill_switch.py:14` already states the liquidation-only semantic in prose --
*"Pause = halt new entries; existing positions kept"*. **The code implements
the first half and not the second.**

**External finding (Q2): the failure mode is documented and recent.** MCX
2024-02 / 2025-07 / 2025-10-28: protective orders became worthless while
positions stayed live and prices moved; the loss fell entirely on the holder.
FINRA 15-09's *silence* on post-disable position management is itself evidence
-- the regime presumes a human desk absorbs the residual risk, an assumption an
unattended autonomous loop violates. **The Knight Capital 2012 order could not
be fetched (SEC returned 403) and is deliberately NOT cited.**

**Internal finding 1 -- severity is HIGH, and this is the fact that sets it.**
`check_stop_losses` has **exactly one production caller**,
`autonomous_loop.py:1471`. No API route, no scheduler job, no MCP tool. The
cycle is the *sole* means of stop enforcement, so a halted cycle means zero
enforcement, indefinitely. The `blocked` path is **non-latching**, so it can
recur every cycle without any operator resume ever being required.

**Internal finding 2 -- the SELL path is deliberately ungated, and that is
load-bearing.** `paper_trader.py:278-281` documents that `execute_sell` is
intentionally *not* gated on `paused`, because the kill switch performs its own
flatten **through** `execute_sell`; gating sells would deadlock the pause. BUY
suppression is independently enforced at `execute_buy:282-294`, which fails
closed (`:225-233`). **Consequence:** a stop-loss-only pass inside the halt
branch cannot leak a BUY, because BUYs are blocked at a different choke point.
**But** on the `blocked` path the switch is *not* paused, so that gate returns
None -- blocked-path BUY suppression comes *only* from the `return summary` at
`:1437`, which means **any halt-branch code must still return before Step 6.**

**Internal finding 3 -- `backfill_missing_stops` is a NEW RISK DECISION, not a
protective default, and must be EXCLUDED from the halt pass.** It synthesizes a
price that never existed (`stop = avg_entry_price * (1 - 8%)`). On a book that
has already breached, entry prices are stale and a synthesized stop can land
*above* the current mark -- so backfill-then-check silently converts "this
position has no stop" into "sell this position at market now". On the `blocked`
and `paused` paths that is **a flatten by side effect, on exactly the branches
the design deliberately does not flatten**, and it would violate this step's own
constraint against introducing a flatten on the blocked path. ESMA para 11(5)
supports the distinction: *choosing* a stop level is algorithmic trading;
*enforcing* a previously chosen level is executing a prior decision.

**Q4 verdict: the literature supports option (b).** Option (a) changes the
healthy path (stop-out then `flatten_all` on a breach cycle = two exit paths,
duplicated fee/learn-loop events, a race over the same positions) and runs the
backfill on every path. Option (c) is defensible only with a human watching, and
is **not free**: no operator-facing stop-check tool exists, so choosing it
*creates* work (build the tool + a halted-with-open-positions alert).

**Q5: the Step 5.4 scale-out ordering hazard is a DIFFERENT defect class** --
commission (a discretionary partial close placed on a cycle about to halt)
rather than omission -- and is DARK today (`paper_scale_out_enabled` defaults
False, `settings.py:35`). **The brief recommends NOT bundling it**, because
moving 5.4 below 5.5 changes the documented MTM-freshness ordering at
`:1377-1379`. It will be queued as its own research-gated step.

## 2. Hypothesis

On the `paused` and `blocked` halt paths, `run_daily_cycle` returns at `:1437`
before Step 5.6, so `check_stop_losses` never runs while the book still holds
full exposure. Adding a **stop-loss-only, SELL-only pass inside the halt
branch, before the return, excluding `backfill_missing_stops`**, restores
protective-exit enforcement on exactly those paths, leaves the healthy path
byte-identical, and cannot place a BUY (independently guaranteed at
`execute_buy:282`, plus the retained `return summary`).

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. "A test reproduces the gap FIRST for the PAUSED path and records it verbatim: drive run_daily_cycle with an already-paused state and a position whose current price is below its stop_loss_price, and assert that under CURRENT code check_stop_losses is never called and the position is not closed"
2. "The same reproduction is run for the phase-36.12 `blocked` path, so the fix is proven to cover all three halt reasons rather than only the one that prompted it"
3. "The chosen option (a/b/c) is stated with its reasoning in the artifacts, and if (c) the operator-facing instruction is stated too. No silent fourth option"
4. "If (a) or (b): a halted cycle now enforces stops -- proven by the reproduce test going green -- AND still places NO new BUYs. Assert both; a fix that lets a paused book buy is worse than the defect"
5. "The `triggered` path is unchanged in observable behaviour (it flattens first, so stop enforcement is usually moot) -- assert against a fixture rather than reasoning about it"
6. "Line numbers are RE-DERIVED at fix time (`grep -n 'Step 5.6' backend/services/autonomous_loop.py`), not copied from this step text -- 36.12's commits moved them once already"
7. "MUTATION-TEST every new guard, and mutate BOTH directions of any ordering change (moving the enforcement back must fail the reproduce test)"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or autonomous_loop'`

**live_check (immutable):** "Verbatim test output for the reproduce-then-fix
pair on BOTH the paused and the blocked halt paths, plus the re-derived grep
showing the final ordering of Step 5.5's return and Step 5.6."

## 4. THE OPERATOR DECISION THIS STEP RESERVES

The step text states the fix is *"a RISK-POLICY DECISION THE OPERATOR OWNS, so
research it and present the options rather than picking one silently"*. The
research is complete and points to **(b)**. The three options were presented to
the operator with their trade-offs **before GENERATE began**.

**OPERATOR DECISION, recorded 2026-08-09: option (b)** -- a stop-loss-only pass
inside the halt branch, SELL-only, scoped to `paused` and `blocked`, excluding
`backfill_missing_stops`, with `return summary` still last.

**Reasoning (satisfies criterion 3, "no silent fourth option"):** (b) is the
only option that restores protective-exit enforcement without changing the
healthy path. It matches the named industry state -- NYSE Pillar's "Block only
-- accept cancels" and ESMA para 93's middle rung -- and it delivers the second
half of the semantic `kill_switch.py:14` already promises in prose. (a) was
rejected because it alters the healthy path (double exit + fee/learn-loop
duplication on a breach cycle) and runs the backfill everywhere. (c) was
rejected because it is defensible only with a human watching, is empirically
refuted by the MCX incidents, and is not free -- it would require building an
operator stop-check tool and a halted-with-open-positions alert that do not
exist today.

## 5. Plan

1. **[done]** Research gate -- PASSED, `research_brief_36.17.md`.
2. **[this file]** Contract, written BEFORE any code.
3. **Present options (a)/(b)/(c) to the operator** with the research backing.
4. **Reproduce FIRST** (criteria 1 + 2): a new test file drives
   `run_daily_cycle` end-to-end on the `paused` path and on the `blocked` path
   with a position below its stop, asserting under CURRENT code that
   `check_stop_losses` is never called and the position is not closed. Record
   verbatim output. The canonical drivable example and required mocks are named
   in the brief (§"`run_daily_cycle` IS drivable end-to-end in a test").
5. **Implement the operator's chosen option.** If (b): a SELL-only pass inside
   the halt branch, before `return summary`, scoped to `blocked` and `paused`,
   **excluding `backfill_missing_stops`**, not appending to `summary["steps"]`.
6. **Assert the no-BUY invariant** (criterion 4) and that the `triggered` path
   is observably unchanged against a fixture (criterion 5).
7. **Mutation-test every new guard, both directions** (criterion 7): reverting
   the ordering must turn the reproduce test red.
8. Q/A via the Workflow rail; transcribe the verdict verbatim; append
   `harness_log.md`; flip.

## 6. Traps this step must not fall into (measured, from the brief)

- **Do NOT append to `summary["steps"]` inside the halt branch** -- measured
  collision: it turns `test_phase_36_12_kill_switch...:298` and `:374` RED.
- **Do NOT trust the existing Step 5.6 suite as coverage** -- it never drives
  `run_daily_cycle` (`test_autonomous_loop_step_5_6.py:44-61`); it is
  structurally blind to this defect.
- **Every new test MUST carry the `captured_alerts` fixture** or it pages the
  operator's real Slack (measured at `test_phase_36_12...:70-77`), and must keep
  `cycle_health.get_log` stubbed or it dirties git-tracked handoff files.
- **`check_stop_losses` silently no-ops on a NULL stop or a 0/None price**
  (`if stop and current`, `:804`) -- so a halted book with NULL stops gets
  nothing even under (b). That is an argument for *alerting* on NULL stops
  during a halt, not for backfilling them.
- **Do not clear the halt** -- no `state.resume()`, no mutation of `_paused_at`.
- **Do not disturb** `summary["status"]="halted_kill_switch"` (`:1425`) or
  `summary["halt_reason"]` (`:1426`) -- phase-85.4 depends on both.

## 7. References

- `handoff/current/research_brief_36.17.md` (6 read in full, 29 URLs).
- MiFID II RTS 6 Art. 12 -- https://eur-lex.europa.eu/eli/reg_del/2017/589/oj/eng
- ESMA Supervisory Briefing on Algorithmic Trading in the EU (2026-02), para 93 / 11(5)
- NYSE Pillar Risk Controls v4.7 -- https://www.nyse.com/publicdocs/nyse/NYSE_Pillar_Risk_Controls.pdf
- CME Kill Switch -- https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457088324/Kill+Switch
- FINRA Notice 15-09 -- https://www.finra.org/rules-guidance/notices/15-09
- MCX protective-order failures -- BusinessToday, 2026-06-10
- Internal: `autonomous_loop.py:141-166,1370-1500`, `kill_switch.py:14`,
  `paper_trader.py:225-233,278-294,797-804,926,1468`, `settings.py:35`
