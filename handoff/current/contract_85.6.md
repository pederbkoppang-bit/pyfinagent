# Contract — phase-85.6 (P0 DEADLOCK: the book cannot be un-paused)

**Step id:** 85.6 · **Cycle:** 184 — 2026-08-09
**Order:** RESEARCH done → **this contract** → GENERATE (not started) → Q/A → log → flip.
Predecessor closed this session: **85.4** (commit `8aa3f52e`).

---

## 1. Research gate

`handoff/current/research_brief_85.6.md` — `gate_passed: true`, **7** sources read
in full, **22** URLs, recency scan performed, three-variant query discipline visible.

### The gate refuted TWO more of the step's own premises. Build against the gate.

| Step premise | Measured reality |
|---|---|
| The roll runs at `autonomous_loop.py ~:1271+` | **Wrong file.** There are two `autonomous_loop.py`. `backend/autonomous_loop.py` (620 lines) is the *harness* loop with zero kill-switch references. The trading cycle is `backend/services/autonomous_loop.py`, and the kill-switch call is **`:1375`** (Step 5.5 of 10), behind the analysis phase at `:1148`. |
| The switch latched paused at `2026-08-04T11:43:31Z` | **Wrong date.** Replaying `handoff/kill_switch_audit.jsonl` under `_load_from_audit`'s own semantics: last `resume` = `2026-07-27T06:20:38Z`, first pause after it = **`2026-08-03T09:03:17Z`**. The 08-04 row is real but is one of 12 that day and not a state change. **The outage is a day longer than the step says.** |

Corroborated third finding, recorded so it is not rediscovered: **36 redundant
`trigger:"manual"` pause rows** exist since the last resume, and
`_load_from_audit:270` re-stamps `_paused_at` from every one, so the phase-38.1
2h auto-resume clock resets on each. **Any design leaning on `check_auto_resume`
would look correct and never fire.** This contract does not lean on it.

## 2. The mechanism, re-derived

`update_sod_nav` has exactly **one** production call site — `paper_trader.py:1298`
— inside `check_and_enforce_kill_switch()`, which has exactly **one** production
caller — `backend/services/autonomous_loop.py:1375`, Step 5.5, after the analysis
phase. A cycle that dies in `analyzing` never reaches it.

The literature names this exact bug as a shipped default: resilience4j's
`automaticTransitionFromOpenToHalfOpenEnabled=false` means *"the transition to
HALF_OPEN only happens if a call is made"* — a circuit breaker whose recovery
path is reachable only through the thing it is blocking. Its documented remedy is
an out-of-band transition, not a change to the breaker's thresholds.

## 3. Immutable success criteria — VERBATIM from `.claude/masterplan.json`

> 1. the deadlock is broken at the CODE level, not by hand-writing an audit row: a cycle that dies in `analyzing` must still leave the daily anchor rollable, OR the roll must move to a point every cycle reaches regardless of where it dies -- whichever is chosen, a test drives a cycle that dies mid-analysis and asserts the anchor is fresh afterwards
>
> 2. the 409 refusal message is corrected so it no longer tells the operator 'NO operator action is required ... this refusal clears itself' when that is false; the message must state the ACTUAL unblock condition, and a test asserts the message text matches the real mechanism (it currently claims the roll happens 'at the top of the next cycle' while it runs at paper_trader.py:1298 in the mark/trade region)
>
> 3. the interaction with 85.4 is stated with evidence rather than assumed: either demonstrate that fixing 85.4 alone clears this (a completing cycle reaches the roll and resume then succeeds), or demonstrate that it does not and that this step's own fix is required
>
> 4. resume succeeds end-to-end at least once, proven by a live POST returning 200 and a follow-up kill-switch read showing paused=false and armed=true -- not by inspection
>
> 5. no loosening of the kill switch's protective behaviour: arming a stale leg must not weaken any breach threshold, and a mutation that disables the daily-loss leg still fails a test
>
> 6. no change to order/sizing/risk logic; fresh Q/A PASS

**Verification command (immutable):**
`bash -c 'curl -s --max-time 15 http://127.0.0.1:8000/api/paper-trading/kill-switch | python3 -c "import sys,json;d=json.load(sys.stdin);b=d.get(\"breach\",{});print({k:d.get(k) for k in (\"paused\",\"sod_date\")}, {k:b.get(k) for k in (\"armed\",\"daily_baseline_stale\")})"'`

Unlike 85.4's, **this command can fail**: it hits a live endpoint and its output
changes with real state. It is a genuine gate for criterion 4.

## 4. Chosen design, and the options rejected

**Take criterion 1's SECOND branch: move the roll to a point every cycle reaches.**

A dedicated start-of-day roll runs **early in the cycle**, before screening, from
the stored portfolio NAV (`get_portfolio_state`) — i.e. the previous close's
marked NAV, which is what a *start-of-day* anchor is defined to be. The existing
`:1298` roll stays exactly where it is and becomes a same-day no-op via the
`sod_anchor_needs_reroll(snap, today)` guard that already exists.

### Why the alternatives were rejected — all four from the gate

- **(C) roll on backend startup — REJECTED.** Google SRE lists restart as the
  standard remediation for a deadlocked server, so this fires precisely when it
  is most dangerous, and an MQL5 write-up from six days ago names the failure
  exactly: re-anchoring on an init event means *"From every anchor's point of
  view the limit was never breached."*
- **(D) roll inside `/resume` — REJECTED.** It lets the refusal manufacture its
  own precondition. (The gate notes a *safe* variant — re-deriving the opening
  NAV from `save_daily_snapshot` history, i.e. restoration rather than
  forgiveness. Recorded as the fallback if the chosen design fails Q/A; not built
  here, to keep the diff small.)
- **(B) separate scheduled roller job — VIABLE, NOT CHOSEN.** It adds a second
  writer to a safety journal and a second schedule to keep in sync. The
  in-cycle-early roll gets the same property with no new scheduler surface.
- **(E) operator-token re-anchor — NOT CHOSEN as the primary.** It requires an
  operator, and the whole complaint is that the operator has no action available.

### The safety argument for the NAV source — this is the load-bearing claim

The early roll anchors on the **previous close's** NAV; the old roll anchored on
**today's freshly-marked** NAV. These differ, and the difference must not loosen
anything:

- If the book has **fallen** since the last mark, previous-close NAV is **higher**,
  so measured daily loss is **larger** and the switch fires **earlier** — strictly
  more protective.
- If the book has **risen**, the old behaviour anchored at the higher intraday
  value; anchoring at the previous close means an overnight gain is not handed
  back before the daily-loss clock starts. This is the standard definition of a
  daily loss limit (measured from the prior close), not a relaxation.
- The dangerous direction is anchoring **lower during a drawdown**, which forgives
  the loss. The early roll cannot do that: on a falling book it anchors higher.

**This will be asserted by a test, not left as prose** (see §5, C5).

`paper_trader.py:1290-1308` carries a `DO NOT blanket-reorder this` warning: the
breach decision must stay on the POST-roll state. **This design does not reorder
that.** By the time `check_and_enforce_kill_switch` runs, the anchor is already
today's, `sod_anchor_needs_reroll` returns False, and the breach still evaluates
on a post-roll state. A test pins that invariant.

## 5. Criterion-by-criterion plan

- **C1** — early roll + a test that drives the real `run_daily_cycle` with a fault
  injected mid-analysis (reusing the 85.4 fault-injection harness) and asserts
  `sod_date == today` afterwards. Also asserts the pre-fix arrangement fails the
  same test (mutation).
- **C2** — rewrite the 409 body. It must (a) drop "NO operator action is required"
  and "this refusal clears itself", (b) name the real unblock condition, and
  (c) stay within phase-36.12's ban on telling an operator to wait for an
  automatic re-anchor. Test asserts the banned phrases are ABSENT and the true
  mechanism is named, with the file:line the roll actually lives at.
- **C3** — answer with evidence, and the answer is **NO**: 85.4 does not clear
  this. 85.4's remedies (cycle budget 7200→10800, rail timeout 150→210, merged
  dispatch) are all **operator-gated asks #23/#24/#25 and NOT applied**, so
  Monday's cycle still times out in `analyzing` at the measured 8529s vs 7200s.
  Independently, the 2026-08-05 cycle *did* complete and still traded nothing
  because the switch was paused — so the two gates are separable and both bind.
- **C4** — live proof. The early roll fires within ~1 minute of cycle start, so a
  single manually-triggered cycle rolls the anchor long before it could time out.
  Spend **one** of the two authorized verification cycles; read
  `handoff/.autonomous_loop.lock` (never `last_result`) first. Then `POST /resume`
  and capture the 200 + `paused=false, armed=true`.
- **C5** — mutation tests: disable the daily-loss leg → a test must fail; and an
  explicit non-loosening test per §4.
- **C6** — no order/sizing/risk logic touched; fresh Q/A.

## 6. Hazards and hard prohibitions

- **Do NOT hand-write a `sod_snapshot` row** into `handoff/kill_switch_audit.jsonl`.
  Criterion 1 names this; it is the ask-#21 anti-pattern.
- **Do NOT weaken any breach threshold**, disarm any leg, or widen a limit.
- **A backend restart IS required** before the live leg — the running process
  holds the pre-fix `paper_trader`/`autonomous_loop` in `sys.modules`, so
  committed is not in force (`feedback_committed_is_not_in_force`). Read the
  lockfile first; a restart during a live cycle would kill a real trading run.
- Tests must inject kill-switch state, never read or write the operator's live
  journal — the phase-36.28 class, which 85.4's mutation M9 caught me committing.

## 7. Scope boundary vs 85.5.1

The gate answered 85.5.1's question as a side effect. Recording it here so the
two steps do not merge:

- The RED `test_valid_nav_still_breaches` fixture omits the `sod_date` key
  entirely (`test_book_safety_69.py:79`) while production `_snapshot_locked`
  always emits it → **that is 85.5.1's fixture defect, and it belongs to 85.5.1.**
- But `sod_date=None` with a positive `sod_nav` **is** production-reachable via
  `_load_from_audit:285-295` (a legacy row with no `date` and an unparseable
  `ts`). No live row is currently in that shape.
- **The trailing leg is date-independent and still fires**: in the RED test
  `any_breached` is in fact True; only `daily_loss_breached` is False. Exposure is
  bounded to drawdowns in `[daily_limit, trailing_limit)`.
- It self-heals at `:1297-1298` — **except inside this step's deadlock**, where it
  has now persisted for days. That is 85.6's problem and this contract's fix
  removes it.

## 8. References

- `handoff/current/research_brief_85.6.md` (gate output)
- `backend/services/autonomous_loop.py:1148, :1375`; `backend/services/paper_trader.py:1284-1310`;
  `backend/services/kill_switch.py:515, :444, _load_from_audit:270, :285-295`;
  `backend/api/paper_trading.py:596-630`
- `handoff/current/experiment_results_85.4.md` §2-§3 (why cycles still time out)
- Operator asks #21 (audit-journal provenance), #22 (the deadlock), #23/#24/#25 (85.4 remedies)
