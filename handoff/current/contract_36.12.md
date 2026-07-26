# Contract — phase-36.12

**Step id:** `36.12` (phase-36, priority **P0**, `harness_required: true`)
**Title:** *The order-placing path silently forgives a drawdown instead of surfacing `armed: false`.*
**Written:** 2026-07-26, BEFORE any code was changed for this step (research → contract → generate).

## TIER

| field | value |
|---|---|
| Tier | **T3** |
| Model | Opus 5, effort `max` |
| Rationale | P0 on the pre-trade safety gate of a live paper book. |

## Research gate — PASSED

`handoff/current/research_brief_36.12.md`, tier `moderate`, envelope:
`external_sources_read_in_full: 10` (floor 5), `snippet_only_sources: 13`, `urls_collected: 34`,
`recency_scan_performed: true`, `internal_files_inspected: 14`, **`gate_passed: true`**.
Spawned BEFORE this contract was written; the brief's mtime precedes this file's.

Findings that actually shape the design (each cited to the brief's own labels):

- **F1** — there is *no* named CWE for "a guard that mutates the datum it is about to test";
  CWE-367 TOCTOU explicitly does **not** cover self-modified state. The defensible framing is
  **CWE-424 (alternate path)** + Saltzer–Schroeder fail-safe defaults. This contract does **not**
  claim to be applying an established named pattern.
- **F2** — "block when a control is unevaluable" comes from the software-safety literature
  (arc42/Saltzer–Schroeder; Faramesh §7.1 *"An absent decision ... mapped to DENY by default"*),
  **not** from SEC 15c3-5, which only supplies the prevent-the-entry framing. Do not overstate the
  regulation.
- **F3 (adversarial, and it changed the design)** — NYSE Pillar *documents failing OPEN* on a
  missing price reference for one leg while every other control keeps running. So "everyone fails
  closed" is false. What Pillar never does is let a missing input **manufacture a passing check** —
  which is exactly this defect. Evidence FOR keeping per-leg independence (criterion 6) and
  AGAINST an all-or-nothing short-circuit inside `evaluate_breach`.
- **F4** — Nasdaq and NYSE both require an **out-of-band human act** to re-arm. Direct external
  corroboration of criterion 5 / the owed `KS-PEAK-RESET` token.
- **F5** — the audit-indistinguishability half has a name: **CWE-223** (omission of
  security-relevant information). The literature prescribes **both** remedies, not a choice:
  deny the implicit path **and** emit a distinguishable audited event.
- **F6** — Galera's `grastate.dat` / `safe_to_bootstrap` / `seqno: -1` is the reference
  provenance-marker implementation, but it needs a marker written **in advance**, which this book
  does not have. Hence the derived discriminator below (D1 ∧ D2) rather than a sentinel.
- **F7** — nothing found in the literature on keeping operator text in lockstep with behaviour
  (honest null result). Criterion 8 is justified on first principles: the three strings become
  false statements the moment the behaviour changes.

## Hypothesis (falsifiable)

On a post-rotation cycle where both baselines are unrecoverable, `check_and_enforce_kill_switch`
currently anchors both to today's NAV before measuring, so `armed` is **structurally always True**
on the order-placing path (brief §A: `update_peak` at `:1080` and `update_sod_nav` at `:1089-1090`
both precede `evaluate_breach` at `:1092`, and `:1097` branches only on `any_breached`). If the
armed state is measured **before** the mutations and the cycle refuses to place new orders when a
book with prior history comes up disarmed, then the same scenario stops reporting ARMED-and-healthy
— and the healthy path stays byte-identical.

## Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. `A test reproduces the defect FIRST: with both baselines unrecoverable and a NAV materially below the true historical peak, the current check_and_enforce_kill_switch re-anchors and reports any_breached=False with no armed=false surfaced -- record that failing-intent output verbatim before fixing`
2. `After the fix, that same scenario does NOT silently forgive the drawdown: either the breach is measured against the pre-mutation state, or the trading path declines to place orders while armed is false. State which was chosen and why`
3. `A FIRST-EVER-BOOT (legitimately no prior history) still anchors its baselines and trades normally -- assert this explicitly so the fix does not deadlock a genuinely new book`
4. `The trading path's behaviour when armed is false is decided EXPLICITLY and documented: either it blocks, or `armed` is formally advisory there and the artifacts say so. No silent third option.`
5. `No route to reset_peak is introduced from the trading path (that would bypass the owed KS-PEAK-RESET token, step 79.6) -- assert the fix writes no peak_reset row`
6. `The per-leg independence of evaluate_breach is preserved: losing one baseline must still leave the other leg enforcing`
7. `MUTATION-TEST every new guard; reverting the measure-before-mutate ordering must fail the new test`
8. `All THREE operator-facing strings that currently advise waiting for the next cycle to re-anchor (paper_trading.py:600, KillSwitchPanel.tsx:172, KillSwitchPanel.tsx:221) are revised IN THE SAME CHANGE as the behaviour fix, so they describe the new behaviour rather than recommending the defect. A test or grep asserts none of them still promises an automatic re-anchor.`

**Verification command (immutable):**
```
source .venv/bin/activate && python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader'
```

**live_check (immutable):** *Verbatim test output for the reproduce-then-fix pair, plus a curl of
`/api/paper-trading/kill-switch` on a rig you own showing the state after a simulated post-rotation
cycle -- demonstrating it no longer reports ARMED-and-healthy with a forgiven drawdown.*

## Design — decided, with the rejected alternatives named

### Criterion 2 + 4: **BLOCK new orders. `armed` is load-bearing on the trading path, not advisory.**

The chosen shape, and the two things it deliberately does NOT do:

```
pre  = evaluate_breach(nav, ...)        # (1) NEW: measure ARMED before mutating
first_boot = D1 and D2                  # (2) NEW: provenance, computed before mutating
<existing update_peak / SOD roll>       # (3) UNCHANGED
breach = evaluate_breach(nav, ...)      # (4) UNCHANGED -- still the breach decision
if breach["any_breached"] and not paused:   # (5) UNCHANGED -- a real breach still wins
    flatten + pause; return
<existing check_auto_resume>            # (6) UNCHANGED
if not pre["armed"] and not first_boot: # (7) NEW: unknown != healthy -> refuse this cycle
    record a distinguishable audit event; P1 alert
    return {..., "blocked": True}
return {...}                            # (8) UNCHANGED shape, plus additive keys
```

**REJECTED — blanket "measure before mutate".** Evaluating the *breach* pre-mutation would compute
the daily leg against **yesterday's** SOD, misreading a multi-day move as a same-day loss. Step
`36.9` measured that exact arithmetic on this book at `daily_loss_pct = 4.0` — i.e. it would fire
`flatten_all` on the first cycle after a restart. The SOD daily roll is a *legitimate*
pre-measurement mutation (a daily-loss limit is by definition measured from today's open). Only the
`None → anchor` case is the defect. **An executor must not "fix" the roll.**

**REJECTED — pausing instead of blocking.** `state.pause()` latches and requires an operator
resume, and `paper_trading.py:593` 409s that resume while `armed` is false. Pausing here creates a
circular wedge: resume needs armed baselines, and the anchor that produces them lives on the path
that just refused to run (`36.9` finding 3's wedge, reached by a new route). The block is
**non-latching and per-cycle**; existing positions are untouched, matching the module's own
documented semantics (`kill_switch.py:13` — *"Pause = halt new entries; existing positions kept"*).

**REJECTED — flattening on absence.** `evaluate_breach`'s own docstring already argues this
(`kill_switch.py:448-452`): flattening a healthy book because a housekeeping sweep moved a file is
a *new destructive behaviour*. F3 corroborates (Pillar degrades a leg; it does not halt).

**Ordering is load-bearing.** The breach branch stays ahead of the disarmed branch, so with one
baseline present and one missing a real breach on the surviving leg still flattens. The existing
`test_dod4_tier1_coverage_investment.py:968` fixture is exactly that state and catches the wrong
order.

### Criterion 3: the first-ever-boot discriminator

`armed is False` is also true for a genuinely new book, so the block needs provenance or it
deadlocks one. Galera's marker (F6) must be written in advance and this book has none, so:

- **D1 — audit-stream provenance:** has any `sod_snapshot` / `peak_update` / `peak_reset` row
  **ever** existed across all sources? Implemented as a new read-only helper in `kill_switch.py`
  (keeps audit-stream knowledge in one module; `36.8` reworks that surface and can rework this
  with it).
- **D2 — book provenance:** `total_nav != starting_capital` ⇒ the book has traded. Free; both
  fields are already in the `portfolio` dict at `:1077`.

`first_ever_boot = (not D1) and D2-says-untraded`. Ambiguity therefore resolves to
**lost-history → block**, the conservative direction.

**Residual, stated not hidden:** a book sitting coincidentally at exactly its starting capital
*and* with a wiped audit trail is classified as new and trades one cycle unprotected. Accepted:
the alternative (block a genuinely new book forever) is worse, and the live book is nowhere near
that state (`nav ≈ 23838` vs `starting_capital 20000`).

### Criterion 5: no route to `reset_peak`, proven two ways

1. Read the isolated audit file after the new path runs; assert **zero** `peak_reset` rows.
2. `monkeypatch` `reset_peak` to raise for the duration of the test. **Assertion 1 alone is a
   guard that cannot fail** — `reset_peak` is DARK today and returns `None` silently, so a stray
   call would write nothing and assertion 1 would pass anyway (`feedback_mutation_test_guards_and_fixtures`).

### Criterion 5 (cont.) / CWE-223: the distinguishable audit event

The accidental anchor currently writes a bare `peak_update` row with no `old_peak`, no `trigger` —
forensically identical to a legitimate ratchet (brief §E). The fix emits an **additional,
informational** row (`baseline_anchor_on_lost_history`) carrying the pre-anchor state. It is a
**new event name that `_load_from_audit`'s elif-chain ignores**, so replay semantics are
byte-unchanged and `peak_reset` remains the only assignment-semantics event. `36.8` (which also
wants a re-anchor event) can consume this one rather than inventing a second — flagged in that
step's text.

### Criterion 8: the strings — and a FOURTH site the step does not list

Three listed (line numbers re-verified today, all still accurate): `paper_trading.py:600`,
`KillSwitchPanel.tsx:172`, `KillSwitchPanel.tsx:221`. Plus **`kill_switch.py:527-528`**, whose
docstring enumerates the operator-visible surfaces and becomes incomplete once a trading-path block
exists. `OpsStatusBar.tsx:318-325` makes no re-anchor promise — verified, left alone.

Guard: a test asserts the three old promise phrases appear **zero** times across those files
(including in comments — comment-carried tokens have tripped this project's scans three times),
**paired with** a behavioural assertion that the 409 body names the new block, because a bare
source-scan is the weak guard shape.

### Criterion 6 + the `36.11` coordination

`evaluate_breach`'s internals stay **byte-untouched** — per-leg independence is a
preserve-this-property criterion, not a build-this one. The new read of `armed` uses **direct
indexing (`breach["armed"]`)**, not `.get("armed", True)`: the call is same-process so the key is
always present, and a third fail-open default would be one more thing for `36.11` to litigate.

## Files to change

| Path | Change |
|---|---|
| `backend/services/kill_switch.py` | + `baseline_history_exists()` (read-only, D1) and a `record_lost_history_anchor()` audit writer. `evaluate_breach` untouched. Docstring at `:527-528` extended with the new surface. |
| `backend/services/paper_trader.py` | `check_and_enforce_kill_switch`: pre-mutation measurement, provenance, the disarmed block, additive return keys. |
| `backend/services/autonomous_loop.py` | `:1287` halt branch also honours `blocked`. |
| `backend/api/paper_trading.py` | `:600` 409 body rewritten to the new behaviour. |
| `frontend/src/components/KillSwitchPanel.tsx` | `:172` + `:221` titles rewritten. |
| `backend/tests/test_phase_36_12_*.py` | New. **Must port the autouse live-file write-protect fixture** — otherwise that brace exists only in the 36.7 module (brief §G). |

## Anti-patterns guarded

1. **Guard-that-cannot-fail** — every new guard gets a named mutation, including the two-sided
   discriminator mutation (force `first_ever_boot` always-True ⇒ the lost-history test must go red;
   always-False ⇒ the first-boot test must go red) and the fixture itself.
2. **Source-scan-as-behavioural-test** — criterion 8's grep is paired with a behavioural assertion.
3. **Fixing the roll** — the SOD daily re-anchor is legitimate; touching it ships a false-flatten
   regression. Called out above and repeated in the code comment.
4. **Routing around the operator token** — no `reset_peak` call from the trading path; asserted two ways.
5. **Silent absorption of an out-of-scope defect** — see below.

## Out of scope (and why)

- **Recovering the true historical peak.** This step converts a *silent* forgiveness into a *loud,
  audited, order-blocking* one. Restoring the real high-water mark from archives is `36.8`;
  deliberately re-anchoring it downward is the operator's call behind `KS-PEAK-RESET` (`79.6`).
- **`36.9`'s three findings** (stale `sod_date`, `nav_invalid` reporting armed, `sod_nav=0.0`
  wedge). `36.9` also wants to widen the `sod_nav is None` test at `:1089`; this step restructures
  the surrounding function, so **`36.9` must rebase onto this change** — recorded here and in the
  results file.
- **`execute_buy` has no kill-switch gate at all** — found while auditing (brief §"Scope-adjacent
  defect"). `backend/agents/mcp_servers/signals_server.py:444` calls `paper_trader.execute_buy`
  directly, and `is_paused()` is consulted in exactly two places repo-wide
  (`paper_trader.py:1097`, `autonomous_loop.py:1287`) — Main re-verified both by grep. A block
  implemented in the loop is therefore incomplete for that path. **Filed as its own research-gated
  masterplan step, not absorbed here** (`feedback_queue_discovered_defects_in_masterplan`).

## Risk after this step passes

- The block is **one cycle**: the same cycle anchors (loudly) and re-arms, so cycle N+1 trades
  again against a baseline that no longer reflects the lost history. That is a deliberate
  trade-off against creating a new permanent-lockout class (the `36.8` failure mode).
- The MCP `execute_buy` path stays ungated until its own step lands.
- `36.7` is still `pending` (its code is committed, cycle-4 Q/A in flight). This step is built
  entirely on 36.7's `armed` flag; if that commit were ever reverted, this step evaporates.
